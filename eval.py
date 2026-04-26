"""
Evaluate from-scratch models on Chinese held-out data and Chinese benchmarks.

Usage:
    python eval.py --model_path output/scratch-baseline-d512-L12-20k/final --mode baseline
    python eval.py --model_path output/scratch-block-d512-L12-20k/final --mode block
    python eval.py --model_path output/scratch-full-d512-L12-20k/final --mode full
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from eval_helpers import (
    build_mcq_prompt,
    extract_ceval_choices,
    extract_cmmlu_choices,
    get_mcq_answer,
    get_mcq_question,
    get_sample_text,
    normalize_answer_label,
    should_stop_ppl,
)
from modeling_attnres import Qwen3AttnResForCausalLM
from transformers import AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from datasets import load_dataset


CEVAL_SUBJECTS = [
    "accountant",
    "advanced_mathematics",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "chinese_language_and_literature",
    "civil_servant",
    "clinical_medicine",
    "college_chemistry",
    "college_economics",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "ideological_and_moral_cultivation",
    "law",
    "legal_professional",
    "logic",
    "mao_zedong_thought",
    "marxism",
    "metrology_engineer",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics",
    "modern_chinese_history",
    "operating_system",
    "physician",
    "plant_protection",
    "probability_and_statistics",
    "professional_tour_guide",
    "sports_science",
    "tax_accountant",
    "teacher_qualification",
    "urban_and_rural_planner",
    "veterinary_medicine",
]

CMMLU_SUBJECTS = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", required=True, choices=["baseline", "block", "full"])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_samples", type=int, default=200,
                   help="Number of text chunks for Chinese perplexity")
    p.add_argument("--ppl_dataset", default="opencsg/Fineweb-Edu-Chinese-V2.2")
    p.add_argument("--ppl_dataset_source", default="modelscope",
                   choices=["modelscope", "huggingface"])
    p.add_argument("--ppl_dataset_name", default="default")
    p.add_argument("--ppl_split", default="train")
    p.add_argument("--ppl_skip_samples", type=int, default=10000,
                   help="Skip the first N documents before computing held-out perplexity")
    p.add_argument("--mcq_max_subjects", type=int, default=8,
                   help="Max number of subjects to evaluate for C-Eval / CMMLU")
    p.add_argument("--mcq_max_samples", type=int, default=40,
                   help="Max evaluation samples per subject")
    p.add_argument("--fewshot", type=int, default=5,
                   help="Number of few-shot examples per subject")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_model(model_path, mode, device):
    if mode == "baseline":
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    else:
        model = Qwen3AttnResForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    model.eval()
    return model


def load_tokenizer(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


def load_training_dataset(dataset_source, dataset_name, config_name, split):
    if dataset_source == "modelscope":
        from modelscope.msdatasets import MsDataset

        load_kwargs = dict(split=split, use_streaming=True, trust_remote_code=True)
        if config_name and config_name.lower() != "none":
            load_kwargs["subset_name"] = config_name

        try:
            return MsDataset.load(dataset_name, **load_kwargs)
        except Exception:
            if load_kwargs.get("subset_name") == "default":
                load_kwargs.pop("subset_name")
                return MsDataset.load(dataset_name, **load_kwargs)
            raise

    hf_name = None if config_name in (None, "", "none") else config_name
    if hf_name == "default":
        hf_name = None
    return load_dataset(dataset_name, name=hf_name, split=split,
                        streaming=True, trust_remote_code=True)


def load_modelscope_subset_dataset(dataset_name, subset_name, split):
    from modelscope.msdatasets import MsDataset

    return MsDataset.load(
        dataset_name,
        subset_name=subset_name,
        split=split,
        trust_remote_code=True,
    )


def eval_chinese_perplexity(model, tokenizer, seq_len, num_samples, device,
                            dataset_name, dataset_config, dataset_source,
                            split, skip_samples):
    ds = load_training_dataset(dataset_source, dataset_name, dataset_config, split)

    if skip_samples > 0 and hasattr(ds, "skip"):
        ds = ds.skip(skip_samples)

    buffer = []
    nlls = []
    total_tokens = 0

    for sample in tqdm(ds, desc="Chinese PPL"):
        if should_stop_ppl(total_tokens, num_samples, seq_len):
            break

        text = get_sample_text(sample)
        if not text:
            continue

        # Bound per-document tokenization so a single very long sample does not
        # trigger huge tokenizer outputs or max-length warnings.
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len * max(2, num_samples),
        )
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        buffer.extend(ids)

        while len(buffer) >= seq_len and not should_stop_ppl(total_tokens, num_samples, seq_len):
            chunk_ids = buffer[:seq_len]
            buffer = buffer[seq_len:]
            chunk = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids=chunk)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="sum")
            nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(nll.item())
            total_tokens += shift_labels.numel()

            if total_tokens >= num_samples * seq_len:
                break

    avg_nll = sum(nlls) / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, total_tokens


def score_choice_labels(model, tokenizer, prompt, labels, device):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    scores = {}

    for label in labels:
        label_ids = tokenizer(label, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        full_ids = torch.cat([prompt_ids, label_ids], dim=1)

        with torch.no_grad():
            logits = model(input_ids=full_ids).logits

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = full_ids[:, 1:]
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        label_len = label_ids.size(1)
        scores[label] = target_log_probs[0, -label_len:].sum().item()

    return scores


def gather_fewshot_examples(ds, num_examples, choice_extractor):
    examples = []
    for sample in ds:
        examples.append({
            "question": str(get_mcq_question(sample)).strip(),
            "choices": choice_extractor(sample),
            "answer": normalize_answer_label(get_mcq_answer(sample)),
        })
        if len(examples) >= num_examples:
            break
    return examples


def eval_multichoice_subject(model, tokenizer, dataset_name, subject, eval_split,
                             fewshot_split, max_samples, num_fewshot,
                             choice_extractor, device):
    eval_ds = load_modelscope_subset_dataset(dataset_name, subject, eval_split)
    fewshot_ds = load_modelscope_subset_dataset(dataset_name, subject, fewshot_split)
    fewshot_examples = gather_fewshot_examples(fewshot_ds, num_fewshot, choice_extractor)

    correct = 0
    total = 0

    for sample in eval_ds:
        question = str(get_mcq_question(sample)).strip()
        choices = choice_extractor(sample)
        answer = normalize_answer_label(get_mcq_answer(sample))
        prompt = build_mcq_prompt(question, choices, fewshot_examples=fewshot_examples)
        scores = score_choice_labels(model, tokenizer, prompt, ("A", "B", "C", "D"), device)
        prediction = max(scores, key=scores.get)

        correct += int(prediction == answer)
        total += 1
        if total >= max_samples:
            break

    return correct, total


def eval_ceval(model, tokenizer, device, max_subjects=8, max_samples=40, fewshot=5):
    subjects = CEVAL_SUBJECTS[:max_subjects]
    results = []

    for subject in tqdm(subjects, desc="C-Eval"):
        correct, total = eval_multichoice_subject(
            model=model,
            tokenizer=tokenizer,
            dataset_name="modelscope/ceval-exam",
            subject=subject,
            eval_split="val",
            fewshot_split="dev",
            max_samples=max_samples,
            num_fewshot=fewshot,
            choice_extractor=extract_ceval_choices,
            device=device,
        )
        acc = correct / total if total else 0.0
        results.append((subject, acc, correct, total))

    total_correct = sum(item[2] for item in results)
    total_count = sum(item[3] for item in results)
    avg_acc = total_correct / total_count if total_count else 0.0
    return avg_acc, total_correct, total_count, results


def eval_cmmlu(model, tokenizer, device, max_subjects=8, max_samples=40, fewshot=5):
    subjects = CMMLU_SUBJECTS[:max_subjects]
    results = []

    for subject in tqdm(subjects, desc="CMMLU"):
        correct, total = eval_multichoice_subject(
            model=model,
            tokenizer=tokenizer,
            dataset_name="modelscope/cmmlu",
            subject=subject,
            eval_split="test",
            fewshot_split="dev",
            max_samples=max_samples,
            num_fewshot=fewshot,
            choice_extractor=extract_cmmlu_choices,
            device=device,
        )
        acc = correct / total if total else 0.0
        results.append((subject, acc, correct, total))

    total_correct = sum(item[2] for item in results)
    total_count = sum(item[3] for item in results)
    avg_acc = total_correct / total_count if total_count else 0.0
    return avg_acc, total_correct, total_count, results


def print_subject_results(title, results):
    print(title)
    for subject, acc, correct, total in results:
        print(f"  {subject:30s} acc={acc:.4f} ({correct}/{total})")
    print()


def main():
    args = parse_args()

    print(f"Loading {args.mode} model from {args.model_path}...")
    model = load_model(args.model_path, args.mode, args.device)
    tokenizer = load_tokenizer(args.model_path)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | mode={args.mode}")
    print()

    print("=" * 50)
    print("Chinese Held-out Perplexity")
    print("=" * 50)
    nll, ppl, n_tokens = eval_chinese_perplexity(
        model=model,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        device=args.device,
        dataset_name=args.ppl_dataset,
        dataset_config=args.ppl_dataset_name,
        dataset_source=args.ppl_dataset_source,
        split=args.ppl_split,
        skip_samples=args.ppl_skip_samples,
    )
    print(f"  Loss: {nll:.4f} | PPL: {ppl:.2f} | Tokens: {n_tokens}")
    print()

    print("=" * 50)
    print("C-Eval")
    print("=" * 50)
    ceval_acc, ceval_correct, ceval_total, ceval_results = eval_ceval(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_subjects=args.mcq_max_subjects,
        max_samples=args.mcq_max_samples,
        fewshot=args.fewshot,
    )
    print(f"  Accuracy: {ceval_acc:.4f} ({ceval_correct}/{ceval_total})")
    print_subject_results("  Per-subject:", ceval_results)

    print("=" * 50)
    print("CMMLU")
    print("=" * 50)
    cmmlu_acc, cmmlu_correct, cmmlu_total, cmmlu_results = eval_cmmlu(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_subjects=args.mcq_max_subjects,
        max_samples=args.mcq_max_samples,
        fewshot=args.fewshot,
    )
    print(f"  Accuracy: {cmmlu_acc:.4f} ({cmmlu_correct}/{cmmlu_total})")
    print_subject_results("  Per-subject:", cmmlu_results)

    print("=" * 50)
    print(f"SUMMARY ({args.mode})")
    print("=" * 50)
    print(f"  Chinese Held-out PPL: {ppl:.2f}")
    print(f"  C-Eval Acc:          {ceval_acc:.4f}")
    print(f"  CMMLU Acc:           {cmmlu_acc:.4f}")


if __name__ == "__main__":
    main()
