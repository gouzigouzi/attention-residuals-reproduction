import re


def get_sample_text(sample):
    return sample.get("text") or sample.get("content") or ""


def get_mcq_question(sample):
    return sample.get("question") or sample.get("Question") or ""


def get_mcq_answer(sample):
    answer = sample.get("answer")
    if answer is None:
        answer = sample.get("Answer")
    return answer


def normalize_answer_label(answer):
    if answer is None:
        raise ValueError("Answer cannot be None")

    text = str(answer).strip().upper()
    match = re.search(r"[ABCD]", text)
    if match is None:
        raise ValueError(f"Could not parse answer label from {answer!r}")
    return match.group(0)


def extract_ceval_choices(sample):
    return {label: str(sample[label]).strip() for label in ("A", "B", "C", "D")}


def extract_cmmlu_choices(sample):
    raw_choices = sample.get("choices")
    if raw_choices is None:
        return {label: str(sample[label]).strip() for label in ("A", "B", "C", "D")}

    choices = {}
    for idx, choice in enumerate(raw_choices):
        label = "ABCD"[idx]
        text = str(choice).strip()
        text = re.sub(r"^\(?[ABCD]\)?[\.\s、:：]*", "", text, flags=re.IGNORECASE)
        choices[label] = text.strip()
    return choices


def build_mcq_prompt(question, choices, fewshot_examples=None):
    sections = []
    for example in fewshot_examples or []:
        sections.append(
            "题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：{answer}".format(
                question=example["question"],
                A=example["choices"]["A"],
                B=example["choices"]["B"],
                C=example["choices"]["C"],
                D=example["choices"]["D"],
                answer=example["answer"],
            )
        )

    sections.append(
        "题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：".format(
            question=question,
            A=choices["A"],
            B=choices["B"],
            C=choices["C"],
            D=choices["D"],
        )
    )
    return "\n\n".join(sections)


def should_stop_ppl(total_tokens, num_samples, seq_len):
    return total_tokens >= num_samples * seq_len
