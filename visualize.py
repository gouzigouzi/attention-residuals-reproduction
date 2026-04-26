"""
Offline visualization script for AttnRes models.

Example:
    python visualize.py \
        --model_path output/scratch-block-d512-L12-20k/final \
        --mode block \
        --num_texts 3 \
        --out_dir ./output/visualizations
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from app import (
    extract_weights,
    load_tokenizer,
    plot_layer_deps,
    plot_token_weights,
)
from eval import load_training_dataset
from eval_helpers import get_sample_text
from modeling_attnres import Qwen3AttnResForCausalLM
from visualize_helpers import make_sample_dirname, select_visualization_texts

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mode", required=True, choices=["block", "full"])
    parser.add_argument("--text", default="广义相对论认为，引力来源于时空的弯曲。", help="Optional single text to visualize directly")
    parser.add_argument("--dataset", default="opencsg/Fineweb-Edu-Chinese-V2.2")
    parser.add_argument("--dataset_name", default="default")
    parser.add_argument("--dataset_source", default="modelscope", choices=["modelscope", "huggingface"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_texts", type=int, default=1)
    parser.add_argument("--min_chars", type=int, default=32)
    parser.add_argument("--max_chars", type=int, default=400)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="./output/visualizations")
    return parser.parse_args()


def load_model(model_path, mode, device):
    model = Qwen3AttnResForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    model_mode = getattr(model.config, "attnres_mode", None)
    if model_mode is not None and model_mode != mode:
        raise ValueError(
            f"Mode mismatch: checkpoint was trained with attnres_mode={model_mode!r}, "
            f"but visualize.py was called with --mode {mode!r}"
        )
    return model


def collect_text_samples(ds, num_texts, min_chars, max_chars):
    texts = []
    for sample in ds:
        text = get_sample_text(sample).strip()
        if len(text) < min_chars:
            continue
        text = text[:max_chars].strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= num_texts:
            break
    return texts


def save_sample_visualizations(model, tokenizer, text, sample_dir, num_layers, max_tokens):
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )["input_ids"].to(next(model.parameters()).device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    captured = extract_weights(model, input_ids)

    os.makedirs(sample_dir, exist_ok=True)

    heatmap_fig = plot_layer_deps(captured, num_layers, title="AttnRes Weights", mode=model.config.attnres_mode)
    heatmap_path = os.path.join(sample_dir, "layer_dependencies.png")
    heatmap_fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")

    token_paths = []
    for layer_idx in range(num_layers):
        for sublayer in ("attn", "mlp"):
            fig = plot_token_weights(captured, tokens, layer_idx, sublayer, num_layers)
            out_path = os.path.join(sample_dir, f"layer_{layer_idx:02d}_{sublayer}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            token_paths.append(out_path)

    metadata = {
        "text": text,
        "num_input_tokens": int(input_ids.shape[1]),
        "heatmap_path": heatmap_path,
        "token_weight_paths": token_paths,
    }
    metadata_path = os.path.join(sample_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.mode} model from {args.model_path}...")
    model = load_model(args.model_path, args.mode, args.device)
    tokenizer = load_tokenizer(args.model_path)
    num_layers = model.config.num_hidden_layers

    sampled_texts = []
    if not (args.text or "").strip():
        print(f"Sampling {args.num_texts} texts from {args.dataset} ({args.dataset_source})...")
        ds = load_training_dataset(args.dataset_source, args.dataset, args.dataset_name, args.split)
        sampled_texts = collect_text_samples(ds, args.num_texts, args.min_chars, args.max_chars)
        if len(sampled_texts) < args.num_texts:
            print(f"Collected {len(sampled_texts)} texts, fewer than requested {args.num_texts}.")

    texts = select_visualization_texts(args.text, sampled_texts)
    if args.text:
        print("Using manually provided text from --text.")

    run_dir = os.path.join(
        args.out_dir,
        args.mode,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    manifest = []
    for idx, text in enumerate(texts):
        sample_dir = os.path.join(run_dir, make_sample_dirname(idx, text))
        print(f"Generating visualizations for sample {idx}: {sample_dir}")
        metadata_path = save_sample_visualizations(
            model=model,
            tokenizer=tokenizer,
            text=text,
            sample_dir=sample_dir,
            num_layers=num_layers,
            max_tokens=args.max_tokens,
        )
        manifest.append({"sample_dir": sample_dir, "metadata_path": metadata_path})

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved outputs to {run_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
