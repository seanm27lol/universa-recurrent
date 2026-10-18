"""Variable-hop lookups: where adaptive compute should win first (TI-TNN adaptive-compute benchmark, part A).

Each sequence first states P facts as pairs "a f(a)", where f is a random P-cycle on P of the A symbols
(so no hop is ever a fixed point). A stream follows. Most of its tokens are easy: a count, in which C_i
is followed by C_(i+1 mod M), so each easy target follows from the current token alone. Q hard queries
are mixed in, each "H_k x", answered at x by f^k(x): one hop for half the queries, two or three for
the rest.

A k-hop answer needs about k + 1 layers (one to bind each fact, one per hop), while an easy token needs
about one. The facts are needed as keys only at the layers where hops read them. So a model that can
spend compute per token should need far less than a fixed-depth model of equal accuracy. The task is
built to favor adaptive compute, as a necessary condition: if TI-TNN cannot win here, it will not win
on WikiText-103 (lm.py). The facts are given in context, not stored in the weights.

Every variant, seed and step count sees the same kind of fresh random sequences; evaluation uses one
fixed set of sequences for every run.

    .venv-phase2/bin/python research/ti_tnn/bench/hops.py --ffn ti_tnn_adaptive --target-compute 0.3 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

EVAL_SEED = 20261016
KIND_NAMES = ("bos", "fact", "hop_token", "easy", "query")
REFERENCE_SIZE = (128, 8, 4)


@dataclass
class Task:
    symbols: int = 32   # A
    facts: int = 8      # P: facts per sequence (a P-cycle)
    max_hops: int = 3   # K
    counts: int = 16    # M: easy-token alphabet
    queries: int = 12   # Q: hard queries per sequence
    length: int = 128   # T
    one_hop_share: int = 50  # percent of queries with one hop; the rest are uniform over 2..K

    @property
    def vocab(self) -> int:
        return 1 + self.symbols + self.max_hops + self.counts

    def __post_init__(self):
        if self.length - 1 - 2 * self.facts < 2 * self.queries:
            raise ValueError("sequence too short for the facts and queries")
        if self.facts > self.symbols or self.max_hops >= self.facts:
            raise ValueError("need facts <= symbols and max_hops < facts")
        if not 0 <= self.one_hop_share <= 100:
            raise ValueError("one_hop_share is a percentage")


def make_batch(n: int, task: Task, gen: torch.Generator, max_hops: int | None = None) -> dict:
    """n sequences: tokens [n, T], targets (-100 where unscored), kind (KIND_NAMES index), hops (k at queries)."""
    A, P, M, Q, T = task.symbols, task.facts, task.counts, task.queries, task.length
    SYM, HOP, CNT = 1, 1 + A, 1 + A + task.max_hops  # token meanings come from the full task, always
    K = task.max_hops if max_hops is None else max_hops  # the most hops sampled now (a curriculum lowers it)
    cycle = torch.rand(n, A, generator=gen).argsort(-1)[:, :P] + SYM  # f(cycle[j]) = cycle[j + 1 mod P]
    order = torch.rand(n, P, generator=gen).argsort(-1)
    facts = torch.stack([cycle.gather(1, order), cycle.gather(1, (order + 1) % P)], -1).reshape(n, 2 * P)

    stream = T - 1 - 2 * P
    slots = stream - 2 * Q + Q  # easy tokens take one position, queries two
    is_q = torch.zeros(n, slots, dtype=torch.bool)
    is_q.scatter_(1, torch.rand(n, slots, generator=gen).argsort(-1)[:, :Q], True)
    width = 1 + is_q.long()
    start = width.cumsum(1) - width
    rank = (~is_q).long().cumsum(1) - 1  # the easy token's index in the count
    c0 = torch.randint(M, (n, 1), generator=gen)
    j = torch.randint(P, (n, slots), generator=gen)
    one = torch.rand(n, slots, generator=gen) < task.one_hop_share / 100
    k = torch.where(one, 1, torch.randint(2, K + 1, (n, slots), generator=gen)) if K > 1 else torch.ones(n, slots, dtype=torch.long)

    tokens = torch.empty(n, stream, dtype=torch.long)
    targets = torch.full((n, stream), -100, dtype=torch.long)
    kind = torch.full((n, stream), KIND_NAMES.index("easy"), dtype=torch.long)
    hops = torch.zeros(n, stream, dtype=torch.long)
    tokens.scatter_(1, start, torch.where(is_q, HOP + k - 1, CNT + (c0 + rank) % M))
    targets.scatter_(1, start, torch.where(is_q, torch.full_like(k, -100), CNT + (c0 + rank + 1) % M))
    kind.scatter_(1, start, torch.where(is_q, KIND_NAMES.index("hop_token"), KIND_NAMES.index("easy")))
    rows = torch.arange(n)[:, None].expand(n, slots)[is_q]
    pos = (start + 1)[is_q]
    tokens[rows, pos] = cycle.gather(1, j)[is_q]
    targets[rows, pos] = cycle.gather(1, (j + k) % P)[is_q]
    kind[rows, pos] = KIND_NAMES.index("query")
    hops[rows, pos] = k[is_q]

    head = lambda fill, w: torch.full((n, w), fill, dtype=torch.long)
    return {"tokens": torch.cat([head(0, 1), facts, tokens], 1),
            "targets": torch.cat([head(-100, 1 + 2 * P), targets], 1),
            "kind": torch.cat([head(0, 1), head(1, 2 * P), kind], 1),
            "hops": torch.cat([head(0, 1 + 2 * P), hops], 1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ffn", choices=models.FFN_KINDS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--d-edge", type=int, default=32)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--small-mult", type=int, default=1)
    p.add_argument("--balance-coef", type=float, default=0.01)
    p.add_argument("--bridge-cost", type=float, default=0.01)
    p.add_argument("--target-compute", type=float, default=1.0)
    p.add_argument("--dual-lr", type=float, default=0.01)
    p.add_argument("--expert-init", choices=models.EXPERT_INITS, default="legacy")
    p.add_argument("--cap-warmup-frac", type=float, default=0.0,
                   help="adaptive variants: anneal the cap from full compute to the target over this fraction of steps")
    p.add_argument("--router-full-bias", type=float, default=0.0,
                   help="adaptive variants: initial compute-router bias toward the full option")
    p.add_argument("--skip-keeps-kv", action="store_true",
                   help="adaptive variants: a skipped token still writes its key and value")
    p.add_argument("--shared-expert-frac", type=float, default=0.0,
                   help="mixtures: share of the hidden width given to an always-on shared MLP")
    p.add_argument("--hop-curriculum-frac", type=float, default=0.0,
                   help="train on at most 1 hop, then 2, ... up to max_hops, in equal stages over this fraction of steps")
    p.add_argument("--balanced-loss", action="store_true",
                   help="weight easy tokens and hard queries equally in the loss (default: per scored position)")
    for field, default in asdict(Task()).items():
        p.add_argument(f"--{field.replace('_', '-')}", dest=f"task_{field}", type=int, default=default,
                       help=f"task: {field}")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-sequences", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_efficiency/hops"))
    return p.parse_args(argv)


def variant_name(args) -> str:
    name = models.variant_name(args.ffn, args.bridge_cost, args.target_compute,
                               models.size_tag(args.d_model, args.layers, args.ffn_mult, REFERENCE_SIZE))
    if args.expert_init != "legacy":
        name += f"_init{args.expert_init}"
    if args.cap_warmup_frac > 0:
        name += f"_warm{round(100 * args.cap_warmup_frac)}"
    if args.router_full_bias:
        name += f"_bias{args.router_full_bias:g}"
    if args.skip_keeps_kv:
        name += "_kv"
    if args.balanced_loss:
        name += "_bal"
    if args.shared_expert_frac > 0:
        name += f"_sh{round(100 * args.shared_expert_frac)}"
    if args.ffn not in ("dense", "mod") and args.experts != 8:
        name += f"_e{args.experts}"
    if args.hop_curriculum_frac > 0:
        name += f"_cur{round(100 * args.hop_curriculum_frac)}"
    if args.steps != 6000:
        name += f"_s{args.steps}"
    return name


def lr_at(step: int, steps: int, args) -> float:
    warm = max(1, int(args.warmup_frac * steps))
    if step < warm:
        return args.lr * (step + 1) / warm
    progress = (step - warm) / max(1, steps - warm)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def to(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def relative_compute(model, stats, shape, device) -> torch.Tensor:
    """[B, T]: each token's block compute averaged over layers, as a fraction of a full block."""
    if not stats or "choice" not in stats[0]:
        return torch.ones(shape, device=device)
    costs = torch.tensor(models.option_costs(model.cfg), device=device)
    return torch.stack([costs[s["choice"]] for s in stats]).mean(0)


@torch.no_grad()
def evaluate(model, task: Task, n: int, batch: int, device) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    K = task.max_hops
    hit = torch.zeros(K + 1)
    seen = torch.zeros(K + 1)
    easy_hit = easy_seen = 0.0
    kind_compute = torch.zeros(len(KIND_NAMES))
    kind_count = torch.zeros(len(KIND_NAMES))
    hop_compute = torch.zeros(K + 1)
    macs = tokens = loss_sum = scored = 0.0
    for start in range(0, n, batch):
        b = to(make_batch(min(batch, n - start), task, gen), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, stats = model(b["tokens"])
        logits = logits.float()
        mask = b["targets"] >= 0
        loss_sum += F.cross_entropy(logits[mask], b["targets"][mask], reduction="sum").item()
        scored += float(mask.sum())
        correct = (logits.argmax(-1) == b["targets"]).float()
        query = b["kind"] == KIND_NAMES.index("query")
        easy = b["kind"] == KIND_NAMES.index("easy")
        rel = relative_compute(model, stats, b["tokens"].shape, device)
        for h in range(1, K + 1):
            sel = query & (b["hops"] == h)
            hit[h] += float(correct[sel].sum())
            seen[h] += float(sel.sum())
            hop_compute[h] += float(rel[sel].sum())
        easy_hit += float(correct[easy].sum())
        easy_seen += float(easy.sum())
        for i in range(len(KIND_NAMES)):
            sel = b["kind"] == i
            kind_compute[i] += float(rel[sel].sum())
            kind_count[i] += float(sel.sum())
        N = b["tokens"].numel()
        macs += models.macs_per_token(model.cfg, stats, b["tokens"].shape)["non_embedding"] * N
        tokens += N
    model.train()
    by_hop = {str(h): float(hit[h] / seen[h]) for h in range(1, K + 1)}
    return {"hard_accuracy": float(np.mean(list(by_hop.values()))), "accuracy_by_hops": by_hop,
            "easy_accuracy": easy_hit / easy_seen, "loss": loss_sum / scored, "sequences": n,
            "non_embedding_macs_per_token": macs / tokens,
            "relative_compute": float(kind_compute.sum() / kind_count.sum()),
            "relative_compute_by_kind": {name: float(kind_compute[i] / kind_count[i]) for i, name in enumerate(KIND_NAMES)},
            "relative_compute_by_hops": {str(h): float(hop_compute[h] / seen[h]) for h in range(1, K + 1)}}


def task_of(args) -> Task:
    return Task(**{f: getattr(args, f"task_{f}") for f in asdict(Task())})


def curriculum_max_hops(task: Task, step: int, steps: int, frac: float) -> int:
    """The most hops sampled at a step: s in stage s of max_hops equal stages spanning the first `frac` of
    training, then all of them. Only sampling changes; token meanings stay those of the full task.
    (Before 2026-10-18 the curriculum shrank the task itself, which also shifted the counting tokens'
    IDs between stages; the eight development runs that used it are affected.)"""
    if frac <= 0:
        return task.max_hops
    stage = 1 + int(step / max(1.0, frac * steps / task.max_hops))
    return min(task.max_hops, stage)


def task_loss(logits: torch.Tensor, b: dict, balanced: bool) -> torch.Tensor:
    """Mean cross-entropy over scored positions; or, balanced, the mean over easy tokens and the mean over
    hard queries, weighted equally (the hard queries are otherwise about 12% of the scored positions)."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    if not balanced:  # exactly the frozen benchmark's loss
        return F.cross_entropy(flat, b["targets"].reshape(-1), ignore_index=-100)
    ce = F.cross_entropy(flat, b["targets"].reshape(-1), ignore_index=-100, reduction="none")
    kind = b["kind"].reshape(-1)
    easy, query = kind == KIND_NAMES.index("easy"), kind == KIND_NAMES.index("query")
    return 0.5 * ce[easy].mean() + 0.5 * ce[query].mean()


def train(args, task: Task | None = None) -> dict:
    task = task or task_of(args)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg = models.ModelConfig(vocab_size=task.vocab, context=task.length, d_model=args.d_model, n_layers=args.layers,
                             n_heads=args.heads, ffn=args.ffn, n_experts=args.experts, d_edge=args.d_edge,
                             balance_coef=args.balance_coef, bridge_cost=args.bridge_cost, ffn_mult=args.ffn_mult,
                             small_mult=args.small_mult, target_compute=args.target_compute, dual_lr=args.dual_lr,
                             expert_init=args.expert_init, router_full_bias=args.router_full_bias,
                             skip_keeps_kv=args.skip_keeps_kv, shared_expert_frac=args.shared_expert_frac)
    model = models.LanguageModel(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    gen = torch.Generator().manual_seed(1_000 + args.seed)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args)
        b = to(make_batch(args.batch, task, gen, curriculum_max_hops(task, step, args.steps, args.hop_curriculum_frac)), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, stats = model(b["tokens"])
            loss = task_loss(logits, b, args.balanced_loss)
            aux = sum((s["aux_loss"] for s in stats if "aux_loss" in s), torch.zeros((), device=device))
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.dual_update(stats, models.cap_schedule(args.target_compute, step, args.steps, args.cap_warmup_frac))
        if step % args.log_every == 0 or step == args.steps - 1:
            realized = [float(s["realized_cost"]) for s in stats if "realized_cost" in s]
            entry = {"step": step, "loss": loss.item(), "aux": aux.item(), "elapsed": time.time() - t0,
                     "realized_compute": float(np.mean(realized)) if realized else None,
                     "cost_price": float(model.cost_price) if hasattr(model, "cost_price") else None}
            log.append(entry)
            print(f"  [{variant_name(args)} seed {args.seed}] step {step}/{args.steps} loss {entry['loss']:.4f} "
                  f"compute {entry['realized_compute']} price {entry['cost_price']} {entry['elapsed']:.0f}s", flush=True)
    train_seconds = time.time() - t0
    result = evaluate(model, task, args.eval_sequences, 256, device)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "task": asdict(task), "model_config": vars(cfg), "parameters": models.parameter_counts(model),
            "source": __import__("lm").source_version(),
            "same_size_dense_non_embedding_macs_per_token": models.full_block_macs_per_token(cfg, task.length),
            "train_seconds": train_seconds, "eval": result, "train_log": log}


def main(argv=None):
    args = parse_args(argv)
    env = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    result = train(args)
    result["environment"] = env
    out_dir = args.out / f"{variant_name(args)}-seed{args.seed}-{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    e = result["eval"]
    print(f"{variant_name(args)} seed {args.seed}: hard accuracy {e['hard_accuracy']:.3f} "
          f"(by hops {e['accuracy_by_hops']}), easy {e['easy_accuracy']:.3f}, "
          f"compute {e['relative_compute']:.3f}, {e['non_embedding_macs_per_token']:.0f} MACs/token -> {out_dir}")


if __name__ == "__main__":
    main()
