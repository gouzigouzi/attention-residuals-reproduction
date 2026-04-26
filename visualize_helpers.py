import re


def slugify_text(text, max_len=48):
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", text)
    text = text.strip("._-")
    if not text:
        text = "sample"
    return text[:max_len]


def make_sample_dirname(index, text):
    return f"sample_{index:03d}_{slugify_text(text)}"


def select_visualization_texts(manual_text, sampled_texts):
    text = (manual_text or "").strip()
    if text:
        return [text]
    return list(sampled_texts)


def source_axis_label(mode):
    return "Source Block" if mode == "block" else "Source State"


def source_tick_labels(mode, count):
    prefix = "B" if mode == "block" else "S"
    return [f"{prefix}{idx}" for idx in range(count)]


def format_heatmap_annotation(value):
    return f"{value:.2f}"


def select_annotation_color(value, vmax):
    threshold = vmax * 0.5
    return "black" if value >= threshold else "white"
