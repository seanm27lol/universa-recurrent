"""Train one language model on WikiText-103 and evaluate its perplexity (TI-TNN real benchmark).

Every variant gets the same data order, optimizer, schedule, token budget and seed; only the
feed-forward block differs (see ti_tnn_models.py). Perplexity is exp(mean token cross-entropy) over the
whole validation and test splits, in non-overlapping windows of the training context; it is
comparable across these runs, not with published numbers that use other tokenizers or sliding
windows.

    .venv-phase2/bin/python research/ti_tnn/bench/lm.py --ffn ti_tnn --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script


def variant_name(args) -> str:
    return models.variant_name(args.ffn, args.bridge_cost)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("runs/titnn_data/wikitext103"))
    p.add_argument("--ffn", choices=models.FFN_KINDS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--d-edge", type=int, default=128)
    p.add_argument("--balance-coef", type=float, default=0.01)
    p.add_argument("--bridge-cost", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=32, help="sequences per step")
    p.add_argument("--tokens", type=float, default=2e8, help="training token budget")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=0, help="steps between quick validation checks (0: 10 per run)")
    p.add_argument("--quick-eval-windows", type=int, default=64)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_bench/lm"))
    return p.parse_args(argv)


def load_split(data: Path, split: str) -> np.ndarray:
    return np.memmap(data / f"{split}.bin", dtype=np.uint16, mode="r")


def sample_batch(tokens: np.ndarray, batch: int, context: int, gen: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(tokens) - context - 1, (batch,), generator=gen).tolist()
    chunk = np.stack([tokens[s:s + context + 1] for s in starts]).astype(np.int64)
    t = torch.from_numpy(chunk).to(device, non_blocking=True)
    return t[:, :-1], t[:, 1:]


def lr_at(step: int, steps: int, args) -> float:
    warm = max(1, int(args.warmup_frac * steps))
    if step < warm:
        return args.lr * (step + 1) / warm
    progress = (step - warm) / max(1, steps - warm)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, tokens: np.ndarray, context: int, batch: int, device, max_windows: int | None = None) -> dict:
    """Mean cross-entropy over non-overlapping windows, with inference-time routing statistics."""
    model.eval()
    n = (len(tokens) - 1) // context
    if max_windows is not None:
        n = min(n, max_windows)
    total_loss, total_tokens = 0.0, 0
    runs, fires, counted = [], [], 0
    usage = None
    for start in range(0, n, batch):
        idx = list(range(start, min(n, start + batch)))
        chunk = np.stack([tokens[i * context:i * context + context + 1] for i in idx]).astype(np.int64)
        t = torch.from_numpy(chunk).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, stats = model(t[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += t[:, 1:].numel()
        if stats and "top1" in stats[0]:
            runs.append(float(np.mean([float(s["expert_runs"]) for s in stats])))
            if "fire_rate" in stats[0]:
                fires.append([float(s["fire_rate"]) for s in stats])
            counts = torch.stack([torch.bincount(s["top1"], minlength=model.cfg.n_experts) for s in stats]).float().cpu()
            usage = counts if usage is None else usage + counts
            counted += 1
    model.train()
    loss = total_loss / total_tokens
    out = {"loss": loss, "perplexity": math.exp(loss), "tokens": total_tokens, "windows": n}
    if counted:
        share = (usage / usage.sum(1, keepdim=True)).tolist()
        out.update({"expert_runs_per_token": float(np.mean(runs)), "expert_share_by_layer": share,
                    "max_expert_share": float(max(max(r) for r in share))})
        if fires:
            out["fire_rate_by_layer"] = np.mean(np.array(fires), axis=0).tolist()
            out["fire_rate"] = float(np.mean(fires))
    return out


def train(args) -> dict:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    meta = json.loads((args.data / "meta.json").read_text())
    train_tokens = load_split(args.data, "train")
    cfg = models.ModelConfig(vocab_size=meta["vocab_size"], context=args.context, d_model=args.d_model,
                             n_layers=args.layers, n_heads=args.heads, ffn=args.ffn, n_experts=args.experts,
                             d_edge=args.d_edge, balance_coef=args.balance_coef, bridge_cost=args.bridge_cost)
    model = models.LanguageModel(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    steps = int(args.tokens // (args.batch * args.context))
    eval_every = args.eval_every or max(1, steps // 10)
    gen = torch.Generator().manual_seed(1_000 + args.seed)  # the same data order for every variant
    val_tokens = load_split(args.data, "validation")
    curve, log = [], []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, args)
        x, y = sample_batch(train_tokens, args.batch, args.context, gen, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, stats = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            aux = sum((s["aux_loss"] for s in stats if "aux_loss" in s), torch.zeros((), device=device))
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == steps - 1:
            fire = [float(s["fire_rate"]) for s in stats if "fire_rate" in s]
            entry = {"step": step, "loss": loss.item(), "aux": aux.item(), "lr": opt.param_groups[0]["lr"],
                     "elapsed": time.time() - t0, "fire_rate": float(np.mean(fire)) if fire else None}
            log.append(entry)
            print(f"  [{args.ffn} seed {args.seed}] step {step}/{steps} loss {entry['loss']:.4f} aux {entry['aux']:.4f} "
                  f"fire {entry['fire_rate']} {entry['elapsed']:.0f}s", flush=True)
        if (step + 1) % eval_every == 0 and step + 1 < steps:
            q = evaluate(model, val_tokens, args.context, args.batch, device, args.quick_eval_windows)
            curve.append({"step": step + 1, "tokens": (step + 1) * args.batch * args.context, "val_loss": q["loss"]})
    train_seconds = time.time() - t0
    final = {split: evaluate(model, load_split(args.data, split), args.context, args.batch, device)
             for split in ("validation", "test")}
    params = models.parameter_counts(model)
    runs_inf = final["validation"].get("expert_runs_per_token", 1.0)
    runs_train = 2.0 if args.ffn in ("moe_top2", "ti_tnn") else 1.0
    return {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_config": vars(cfg), "data_meta": meta, "steps": steps,
        "tokens_trained": steps * args.batch * args.context,
        "parameters": params,
        "compute": {"ffn_macs_per_token_per_layer_inference": models.ffn_macs_per_token(cfg, runs_inf),
                    "ffn_macs_per_token_per_layer_training": models.ffn_macs_per_token(cfg, runs_train),
                    "dense_ffn_macs_per_token_per_layer": 8 * cfg.d_model ** 2},
        "timing": {"train_seconds": train_seconds,
                   "train_tokens_per_second": steps * args.batch * args.context / train_seconds,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None},
        "validation": final["validation"], "test": final["test"], "val_curve": curve, "train_log": log,
    }


def main(argv=None):
    args = parse_args(argv)
    env = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    if torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0), capability=".".join(map(str, torch.cuda.get_device_capability(0))))
    torch.backends.cuda.matmul.allow_tf32 = True
    result = train(args)
    result["environment"] = env
    out_dir = args.out / f"{variant_name(args)}-seed{args.seed}-{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    v, t = result["validation"], result["test"]
    print(f"{args.ffn} seed {args.seed}: validation perplexity {v['perplexity']:.2f}, test {t['perplexity']:.2f}, "
          f"{result['timing']['train_tokens_per_second']:.0f} tokens/s -> {out_dir}")


if __name__ == "__main__":
    main()
