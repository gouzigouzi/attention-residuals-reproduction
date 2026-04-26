"""
Train Qwen3 / Qwen3-AttnRes from scratch on FineWeb-Edu.

Usage:
    # Baseline (no AttnRes)
    torchrun --nproc_per_node=8 train.py --mode baseline

    # Block AttnRes
    torchrun --nproc_per_node=8 train.py --mode block

    # Full AttnRes
    torchrun --nproc_per_node=8 train.py --mode full
"""

import argparse
import contextlib
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Local import

from modeling_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM
from train_helpers import batch_token_chunks, reorder_stream_for_training, split_token_buffer_into_chunks
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="baseline", choices=["baseline", "block", "full"],
                   help="baseline (standard Qwen3), block (Block AttnRes), full (Full AttnRes)")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of AttnRes blocks (for block mode)")
    p.add_argument("--gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"],
                   help="Gate type for mixing AttnRes output with residual stream")
    p.add_argument("--dataset", default="opencsg/Fineweb-Edu-Chinese-V2.2")
    p.add_argument("--dataset_name", default="default")
    p.add_argument("--dataset_source", default="modelscope",
                   choices=["modelscope", "huggingface"],
                   help="Where to load the training dataset from")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokenizer_source", default="modelscope",
                   choices=["modelscope", "huggingface", "local"],
                   help="Where to load the tokenizer from")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=1, help="per-GPU")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--use_wandb", action="store_true",
                   help="Enable Weights & Biases logging. Disabled by default.")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def load_training_dataset(dataset_source, dataset_name, config_name, seed, rank):
    if dataset_source == "modelscope":
        from modelscope.msdatasets import MsDataset

        load_kwargs = dict(split="train", use_streaming=True)
        if config_name and config_name.lower() != "none":
            load_kwargs["subset_name"] = config_name

        try:
            return MsDataset.load(dataset_name, **load_kwargs)
        except Exception:
            if load_kwargs.get("subset_name") == "default":
                load_kwargs.pop("subset_name")
                return MsDataset.load(dataset_name, **load_kwargs)
            raise

    from datasets import load_dataset
    return load_dataset(dataset_name, name=config_name, split="train",
                        streaming=True, trust_remote_code=True)


def shard_stream_for_rank(ds, rank, world_size, seed):
    return reorder_stream_for_training(ds, rank, world_size, seed)


def token_stream(dataset_source, dataset_name, config_name, tokenizer, seq_len, rank, world_size, seed):
    ds = load_training_dataset(dataset_source, dataset_name, config_name, seed, rank)
    ds = shard_stream_for_rank(ds, rank, world_size, seed)
    buf = []
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        chunks, buf = split_token_buffer_into_chunks(buf, seq_len)
        for chunk in chunks:
            yield torch.tensor(chunk, dtype=torch.long)


def load_tokenizer(tokenizer_name, tokenizer_source):
    if tokenizer_source == "modelscope":
        from modelscope import snapshot_download

        tokenizer_dir = snapshot_download(
            tokenizer_name,
            allow_file_pattern=[
                "config.json",
                "tokenizer*",
                "vocab.*",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
        return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    if tokenizer_source == "local":
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def build_model(args, device):
    """Build model from scratch based on mode."""
    common = dict(
        vocab_size=151936,  # Qwen3 tokenizer vocab
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=args.hidden_size // args.num_heads,
    )

    if args.mode == "baseline":
        config = Qwen3Config(**common)
        model = Qwen3ForCausalLM(config)
    else:
        config = Qwen3AttnResConfig(
            attnres_num_blocks=args.num_blocks,
            attnres_recency_bias_init=0.0,  # zero init — paper default
            attnres_mode=args.mode,
            attnres_gate_type=args.gate_type,
            **common,
        )
        model = Qwen3AttnResForCausalLM(config)

    model = model.to(dtype=torch.bfloat16, device=device)
    return model


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = f"{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"

    # ── distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    # ── W&B ──
    use_wandb = False
    if is_main and args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.run_name, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    # ── model ──
    if is_main:
        print(f"Building {args.mode} model...")

    model = build_model(args, device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | mode={args.mode} | d={args.hidden_size} L={args.num_layers}")
        if args.mode != "baseline":
            n_attnres = sum(p.numel() for n, p in model.named_parameters() if "res_" in n)
            print(f"AttnRes params: {n_attnres/1e3:.1f}K")

    # find_unused_parameters needed when some params aren't in the forward graph
    find_unused = args.gate_type != "bias"
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)

    # ── optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
    )

    # ── data ──
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_source)
    stream = token_stream(args.dataset_source, args.dataset, args.dataset_name, tokenizer,
                          args.seq_len, rank, world_size, args.seed)

    # ── training ──
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0

    for batch_chunks in batch_token_chunks(stream, args.batch_size):
        if global_step >= args.steps:
            break

        batch = torch.stack(batch_chunks).to(device)
        input_ids = batch
        labels = batch

        sync_context = (
            model.no_sync()
            if accum_step < args.grad_accum - 1
            else contextlib.nullcontext()
        )

        with sync_context:
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss / args.grad_accum
            loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.seq_len * input_ids.size(0)

        if accum_step < args.grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {global_step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | grad_norm {grad_norm:.3f} | "
                      f"{tok_sec/1e3:.1f}k tok/s")

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_sec,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        if is_main and global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"step-{global_step}")
            model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint → {ckpt_dir}")

    if is_main:
        final_dir = os.path.join(args.out_dir, "final")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
