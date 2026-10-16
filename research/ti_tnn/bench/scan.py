"""Train one model on a SCAN split and score exact-match accuracy (TI-TNN real benchmark).

SCAN (Lake and Baroni 2018, https://github.com/brendenlake/SCAN) maps commands to actions:
"jump twice" -> "I_JUMP I_JUMP". Splits:

    simple      random 80/20 split
    length      train on short action sequences (up to 22 actions), test on longer ones (24 to 48)
    addprim_jump / addprim_turn_left
                the primitive appears in training only on its own; the test asks for it in
                combinations ("jump twice", "turn left after walk")

The model is the decoder of ti_tnn_models.py, reading "command <sep> actions <eos>"; only the
action tokens are trained. A test example counts as correct only if greedy decoding reproduces
the whole action sequence and stops.

    .venv-phase2/bin/python research/ti_tnn/bench/scan.py --split addprim_jump --ffn ti_tnn --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

BASE = "https://raw.githubusercontent.com/brendenlake/SCAN/master/"
SPLITS = {
    "simple": ("simple_split/tasks_train_simple.txt", "simple_split/tasks_test_simple.txt"),
    "length": ("length_split/tasks_train_length.txt", "length_split/tasks_test_length.txt"),
    "addprim_jump": ("add_prim_split/tasks_train_addprim_jump.txt", "add_prim_split/tasks_test_addprim_jump.txt"),
    "addprim_turn_left": ("add_prim_split/tasks_train_addprim_turn_left.txt",
                          "add_prim_split/tasks_test_addprim_turn_left.txt"),
}
SPECIAL = ["<pad>", "<sep>", "<eos>"]
PAD, SEP, EOS = 0, 1, 2


def download(split: str, root: Path) -> dict:
    """Fetch a split's two files once; return their lines and sha256."""
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for part, rel in zip(("train", "test"), SPLITS[split]):
        path = root / rel.replace("/", "__")
        if not path.exists():
            with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                path.write_bytes(r.read())
        data = path.read_bytes()
        out[part] = {"lines": [ln for ln in data.decode().splitlines() if ln.strip()],
                     "sha256": hashlib.sha256(data).hexdigest(), "file": rel}
    return out


def parse(line: str) -> tuple[list[str], list[str]]:
    command, actions = line.split(" OUT: ")
    return command.removeprefix("IN: ").split(), actions.split()


def build_vocab(*line_sets) -> dict:
    words = sorted({w for lines in line_sets for ln in lines for part in parse(ln) for w in part})
    return {w: i for i, w in enumerate(SPECIAL + words)}


def encode(lines: list[str], vocab: dict) -> list[tuple[list[int], list[int]]]:
    return [([vocab[w] for w in c], [vocab[w] for w in a]) for c, a in map(parse, lines)]


def make_batch(pairs, device):
    seqs = [c + [SEP] + a + [EOS] for c, a in pairs]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    target = torch.full((len(seqs), T), -100, dtype=torch.long)
    for i, ((c, a), s) in enumerate(zip(pairs, seqs)):
        ids[i, :len(s)] = torch.tensor(s)
        start = len(c)  # the position of <sep> predicts the first action
        target[i, start:start + len(a) + 1] = torch.tensor(a + [EOS])
    return ids.to(device), target.to(device)


@torch.no_grad()
def greedy_accuracy(model, pairs, device, batch: int, max_new: int) -> dict:
    """Greedy decoding without padding: prompts of equal length are decoded together."""
    model.eval()
    correct, runs, fires = 0, [], []
    by_length: dict[int, list] = {}
    for pair in pairs:
        by_length.setdefault(len(pair[0]), []).append(pair)
    for group in by_length.values():
        for start in range(0, len(group), batch):
            chunk = group[start:start + batch]
            ids = torch.tensor([c + [SEP] for c, _ in chunk], device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            out = [[] for _ in chunk]
            for _ in range(max_new):
                logits, stats = model(ids[:, -model.cfg.context:])
                nxt = logits[:, -1].argmax(-1)
                if stats and "expert_runs" in stats[0]:
                    runs.append(sum(float(s["expert_runs"]) for s in stats) / len(stats))
                    if "fire_rate" in stats[0]:
                        fires.append(sum(float(s["fire_rate"]) for s in stats) / len(stats))
                for i, token in enumerate(nxt.tolist()):
                    if not done[i]:
                        if token == EOS:
                            done[i] = True
                        else:
                            out[i].append(token)
                if done.all():
                    break
                ids = torch.cat([ids, nxt[:, None]], dim=1)
            correct += sum(int(bool(done[i]) and out[i] == chunk[i][1]) for i in range(len(chunk)))
    model.train()
    res = {"exact_match": correct / len(pairs), "n": len(pairs)}
    if runs:
        res["expert_runs_per_token"] = sum(runs) / len(runs)
    if fires:
        res["fire_rate"] = sum(fires) / len(fires)
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--ffn", choices=models.FFN_KINDS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=Path("runs/titnn_data/scan"))
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--context", type=int, default=72)
    p.add_argument("--d-edge", type=int, default=64)
    p.add_argument("--balance-coef", type=float, default=0.01)
    p.add_argument("--bridge-cost", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_bench/scan"))
    return p.parse_args(argv)


def train(args) -> dict:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    raw = download(args.split, args.data)
    vocab = build_vocab(raw["train"]["lines"], raw["test"]["lines"])
    train_pairs, test_pairs = encode(raw["train"]["lines"], vocab), encode(raw["test"]["lines"], vocab)
    longest = max(len(c) + len(a) + 2 for c, a in train_pairs + test_pairs)
    if longest > args.context:
        raise ValueError(f"context {args.context} is shorter than the longest example ({longest})")
    torch.manual_seed(args.seed)
    cfg = models.ModelConfig(vocab_size=len(vocab), context=args.context, d_model=args.d_model, n_layers=args.layers,
                             n_heads=args.heads, ffn=args.ffn, n_experts=args.experts, d_edge=args.d_edge,
                             balance_coef=args.balance_coef, bridge_cost=args.bridge_cost)
    model = models.LanguageModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(2_000 + args.seed)  # the same example order for every variant
    warm = max(1, int(args.warmup_frac * args.steps))
    log, t0 = [], time.time()
    for step in range(args.steps):
        lr = args.lr * min((step + 1) / warm, 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, args.steps - warm))))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(len(train_pairs), (args.batch,), generator=gen).tolist()
        ids, target = make_batch([train_pairs[i] for i in idx], device)
        logits, stats = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
        aux = sum((s["aux_loss"] for s in stats if "aux_loss" in s), torch.zeros((), device=device))
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": loss.item(), "elapsed": time.time() - t0})
            print(f"  [scan {args.split} {args.ffn} seed {args.seed}] step {step} loss {loss.item():.4f}", flush=True)
    train_seconds = time.time() - t0
    max_new = max(len(a) for _, a in train_pairs + test_pairs) + 2
    train_subset = [train_pairs[i] for i in torch.randperm(len(train_pairs), generator=torch.Generator().manual_seed(7))[:2000].tolist()]
    return {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_config": vars(cfg), "data": {p: {"file": raw[p]["file"], "sha256": raw[p]["sha256"],
                                                "examples": len(raw[p]["lines"])} for p in raw},
        "vocab_size": len(vocab), "parameters": models.parameter_counts(model),
        "train_seconds": train_seconds, "train_log": log,
        "test": greedy_accuracy(model, test_pairs, device, args.eval_batch, max_new),
        "train_sample": greedy_accuracy(model, train_subset, device, args.eval_batch, max_new),
    }


def main(argv=None):
    args = parse_args(argv)
    env = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    if torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0))
    result = train(args)
    result["environment"] = env
    out_dir = args.out / f"{args.split}-{models.variant_name(args.ffn, args.bridge_cost)}-seed{args.seed}-{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"scan {args.split} {args.ffn} seed {args.seed}: test exact match {result['test']['exact_match']:.4f} "
          f"(train sample {result['train_sample']['exact_match']:.4f}) -> {out_dir}")


if __name__ == "__main__":
    main()
