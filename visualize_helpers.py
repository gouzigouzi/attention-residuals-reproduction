import re

import matplotlib
import numpy as np
import torch
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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


def load_tokenizer(model_path):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


def compute_softmax_weights(blocks, partial_block, proj, norm, recency_bias):
    v = torch.stack(blocks + [partial_block], dim=0)
    k = norm(v)
    query = proj.weight.view(-1)
    logits = torch.einsum("d, n b t d -> n b t", query, k)
    logits[-1] = logits[-1] + recency_bias
    weights = logits.softmax(dim=0)
    return weights.float()


def extract_weights(model, input_ids):
    """Run a forward pass and capture per-token AttnRes softmax weights."""
    import modeling_attnres as attnres_mod

    captured = {"attn": {}, "mlp": {}}
    original_forwards = {}

    for layer_idx, layer in enumerate(model.model.layers):
        original_forwards[layer_idx] = layer.forward

        def make_patched_forward(lyr, idx):
            def patched_forward(blocks, partial_block, **kwargs):
                w_attn = compute_softmax_weights(
                    blocks, partial_block, lyr.attn_res_proj, lyr.attn_res_norm, lyr.attn_res_bias
                )
                captured["attn"][idx] = {
                    "weights_mean": w_attn.mean(dim=(1, 2)).detach().cpu().numpy(),
                    "weights_all": w_attn.detach().cpu().numpy(),
                }

                h_attn = attnres_mod.block_attn_res(
                    blocks, partial_block, lyr.attn_res_proj, lyr.attn_res_norm, lyr.attn_res_bias
                )
                h = lyr._apply_gate(partial_block, h_attn, "attn")
                attn_out, _ = lyr.self_attn(
                    hidden_states=lyr.input_layernorm(h),
                    attention_mask=kwargs.get("attention_mask"),
                    position_ids=kwargs.get("position_ids"),
                    past_key_values=kwargs.get("past_key_values"),
                    use_cache=kwargs.get("use_cache", False),
                    cache_position=kwargs.get("cache_position"),
                    position_embeddings=kwargs.get("position_embeddings"),
                )
                post_attn_partial = partial_block + attn_out

                if lyr.attnres_mode == "full":
                    blocks_mlp = blocks + [post_attn_partial]
                else:
                    blocks_mlp = blocks

                w_mlp = compute_softmax_weights(
                    blocks_mlp, post_attn_partial, lyr.mlp_res_proj, lyr.mlp_res_norm, lyr.mlp_res_bias
                )
                captured["mlp"][idx] = {
                    "weights_mean": w_mlp.mean(dim=(1, 2)).detach().cpu().numpy(),
                    "weights_all": w_mlp.detach().cpu().numpy(),
                }

                h_mlp = attnres_mod.block_attn_res(
                    blocks_mlp, post_attn_partial, lyr.mlp_res_proj, lyr.mlp_res_norm, lyr.mlp_res_bias
                )
                h = lyr._apply_gate(post_attn_partial, h_mlp, "mlp")
                mlp_out = lyr.mlp(lyr.post_attention_layernorm(h))
                final_partial = post_attn_partial + mlp_out

                if lyr.attnres_mode == "full" or lyr.is_block_boundary:
                    new_blocks = blocks_mlp + [final_partial]
                else:
                    new_blocks = blocks_mlp
                return new_blocks, final_partial

            return patched_forward

        layer.forward = make_patched_forward(layer, layer_idx)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for layer_idx, layer in enumerate(model.model.layers):
            layer.forward = original_forwards[layer_idx]

    return captured


def plot_layer_deps(captured, num_layers, title="", mode="block"):
    """Plot the Kimi-style layer dependency heatmap."""
    num_sublayers = num_layers * 2
    max_sources = 0
    for sublayer in ("attn", "mlp"):
        for data in captured[sublayer].values():
            max_sources = max(max_sources, len(data["weights_mean"]))

    matrix = np.full((num_sublayers, max_sources), np.nan)
    for layer_idx in range(num_layers):
        if layer_idx in captured["attn"]:
            weights = captured["attn"][layer_idx]["weights_mean"]
            matrix[layer_idx * 2, :len(weights)] = weights
        if layer_idx in captured["mlp"]:
            weights = captured["mlp"][layer_idx]["weights_mean"]
            matrix[layer_idx * 2 + 1, :len(weights)] = weights

    fig_w = max(10, max_sources * 0.4 + 3) if max_sources > 20 else 10
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#1a1a2e")
    masked = np.ma.masked_invalid(matrix)
    vmin = 0.0
    vmax = 0.7
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    y_labels = []
    y_colors = []
    for layer_idx in range(num_layers):
        y_labels.extend([f"Attn {layer_idx}", f"MLP {layer_idx}"])
        y_colors.extend(["#4CAF50", "#FF9800"])

    ax.set_yticks(range(num_sublayers))
    ax.set_yticklabels(y_labels, fontsize=6)
    for idx, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(y_colors[idx])
        tick.set_fontweight("bold")

    ax.set_xticks(range(max_sources))
    ax.set_xticklabels(source_tick_labels(mode, max_sources), fontsize=8)

    for row_idx in range(num_sublayers):
        for col_idx in range(max_sources):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                format_heatmap_annotation(value),
                ha="center",
                va="center",
                fontsize=6,
                color=select_annotation_color(value, vmax),
            )

    ax.set_xlabel(source_axis_label(mode), fontsize=10)
    ax.set_ylabel("Sublayer", fontsize=10)
    ax.set_title(title or "Attention Residuals: Layer Dependencies", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Weight")
    plt.tight_layout()
    return fig


def plot_token_weights(captured, tokens, layer_idx, sublayer, num_layers):
    """Plot per-token AttnRes weights for a specific sublayer."""
    data = captured[sublayer].get(layer_idx)
    if data is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    weights = data["weights_all"][:, 0, :]
    n_sources, n_tokens = weights.shape
    n_tokens = min(n_tokens, len(tokens))

    fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.5), max(4, n_sources * 0.4)))
    cmap = plt.cm.viridis.copy()
    im = ax.imshow(weights[:, :n_tokens], aspect="auto", cmap=cmap, vmin=0, vmax=0.7)
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tokens[:n_tokens], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Source")
    ax.set_title(f"Layer {layer_idx} {sublayer.upper()} - Per-Token Weights", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig
