"""Teacher forcing for C4 and T4: identical targets, loss terms, recipe and example order; immutable checkpoints.

**Example.** The smoke run trains T4 and c4_params (the parameter-matched shared controller) on the same 128 training-pool questions (seed 100, at most 100
updates). Update u uses examples [32u, 32u + 32) of a fixed stream: the concatenation of seeded per-epoch
permutations of the 128. Each update sums the loss terms of its 32 examples over all their gold steps, in micro-batches
of 8 with gradient accumulation, divides by the number of steps, clips the gradient norm to 1.0 and takes one AdamW
step (lr 3e-4, weight decay 0.01). Both arms see the same examples in the same order, and their trunks start from the
same initialization (the trunk is built first, under the same seed).

**The training rule.** At every gold step t the trunk reads the gold prefix a_<t (teacher forcing) and the grammar
state before a_t. The loss terms are:
- cross-entropy of the family classifier against the gold family (STOP included), masked to the legal families;
- cross-entropy of the operator, from the gold family's head (T4) or from the shared MLP masked to the gold family
  (C4), masked to the legal operators;
- for pointer operators, cross-entropy of the pointer (table, column, or span start and span end given the gold
  start), masked to the legal targets.

The weights are 1, 1 and 1. They are written into the run configuration before fitting, with the sampling rule.

**Checkpoints.** A checkpoint holds the model, the optimizer, the update counter, the Python, NumPy and torch (CPU and
CUDA) RNG states, and the configuration. It is serialized once, named by the sha256 of its bytes
(`ckpt_u000100_<sha16>.pt`) and never overwritten. `load_checkpoint` re-hashes the file before use. Resuming from
update k reproduces the uninterrupted run exactly, because the sampling stream is a pure function of (seed, update).
`build_model` seeds Python, NumPy and torch, so the saved RNG states, and with them the bytes of an uninterrupted
run's checkpoints, agree across processes. A resumed run's checkpoint holds equal weights, optimizer state and RNG
states, but pickle can lay the same objects out differently, so the verifier compares `model_digest` and
`optimizer_digest` (tensors in a fixed order), not file hashes. On resume,
`train_log.jsonl` keeps the lines before the resume point; later lines move to `train_log.superseded.jsonl`, so an
abandoned stretch of training stays visible.

**Limits.** This is supervised imitation of gold traversals only: no reinforcement, no execution feedback, no repair,
no beam. The smoke run's 100 updates on 128 questions test the machinery; they say nothing about the comparison.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import actions as A
from . import contracts as C
from . import controller as K

TRAIN_VERSION = "train-v1"
SAMPLING_RULE = ("a stream of example indices: the concatenation of torch.randperm(n) permutations, epoch e drawn with "
                 "a torch.Generator seeded by int(sha256(f'{seed}:epoch:{e}')[:15], 16); update u takes stream "
                 "positions [u * effective_batch, (u + 1) * effective_batch), split into consecutive micro-batches")
NORMALIZATION = ("sum over the effective batch of every loss term at every supervised step (STOP included), divided "
                 "by the effective batch's number of supervised steps")


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    seed: int = C.SMOKE_TRAIN["seed"]
    updates: int = C.SMOKE_TRAIN["max_updates"]
    effective_batch: int = 32
    micro_batch: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01      # AdamW, applied to every parameter
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {"family": 1.0, "op": 1.0, "pointer": 1.0})
    checkpoint_every: int = 0       # 0: the final update only
    device: str = "cpu"
    sampling: str = SAMPLING_RULE
    normalization: str = NORMALIZATION
    version: str = TRAIN_VERSION

    def as_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def _epoch_seed(seed: int, epoch: int) -> int:
    return int(hashlib.sha256(f"{seed}:epoch:{epoch}".encode()).hexdigest()[:15], 16)


def batch_indices(n: int, update: int, cfg: TrainConfig) -> list:
    """The example indices of one update (a pure function of n, seed, update and the batch size)."""
    out = []
    start = update * cfg.effective_batch
    for pos in range(start, start + cfg.effective_batch):
        epoch, k = divmod(pos, n)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(_epoch_seed(cfg.seed, epoch)))
        out.append(int(perm[k]))
    return out


def identity_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_config(run_dir: Path, payload: dict) -> str:
    """config.json, written before fitting; an existing one must be identical (a run's configuration never changes)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"{path} exists with a different configuration")
    else:
        path.write_text(text)
    return identity_of(payload)


# --- checkpoints -------------------------------------------------------------------------------------------------------
def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def tensor_digest(state: dict) -> str:
    """sha256 over a state dict's tensors in sorted-key order (name, dtype, shape, bytes): equal weights give an equal
    digest whatever the serializer does. The verifier compares models with this, not with file hashes."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous().reshape(-1)
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def optimizer_digest(state: dict) -> str:
    """sha256 over an optimizer state dict: param groups as canonical JSON, then each parameter's state tensors in
    (parameter index, key) order."""
    h = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for index in sorted(state["state"]):
        for key in sorted(state["state"][index]):
            value = state["state"][index][key]
            h.update(json.dumps([index, key]).encode())
            h.update(tensor_digest({"v": value}).encode() if torch.is_tensor(value) else json.dumps(value).encode())
    return h.hexdigest()


def save_checkpoint(run_dir: Path, model, optimizer, update: int, config: dict) -> dict:
    """Serialize once, name by content hash, never overwrite. Returns {"path", "sha256", "update", "bytes",
    "model_digest", "optimizer_digest"}. A resumed run's file can differ in bytes from the uninterrupted run's (pickle
    shares equal objects differently) while every tensor is equal; the digests are what a verifier compares."""
    buffer = io.BytesIO()
    torch.save({"version": TRAIN_VERSION, "update": update, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": _rng_state(), "config": config,
                "controller": model.cfg.as_dict()}, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = Path(run_dir) / "checkpoints" / f"ckpt_u{update:06d}_{digest[:16]}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"{path} exists with different content")
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return {"path": str(path), "sha256": digest, "update": update, "bytes": len(data),
            "model_digest": tensor_digest(model.state_dict()), "optimizer_digest": optimizer_digest(optimizer.state_dict())}


def load_checkpoint(path) -> dict:
    """Read a checkpoint after checking that its bytes still match the hash in its name."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not path.name.split("_")[-1].startswith(digest[:16]):
        raise ValueError(f"{path.name}: content hash {digest[:16]} does not match the file name")
    ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    ckpt["sha256"] = digest
    return ckpt


def restore(ckpt: dict, model, optimizer=None, rng: bool = True) -> int:
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if rng:
        _set_rng_state(ckpt["rng"])
    return int(ckpt["update"])


def model_from_checkpoint(ckpt: dict, device: str = "cpu") -> K.Controller:
    cfg = K.ControllerConfig(**ckpt["controller"])
    model = K.Controller(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device)


# --- fitting -----------------------------------------------------------------------------------------------------------
def build_model(arm: str, backbone_width: int, seed: int) -> K.Controller:
    """The arm's controller, initialized under `seed` (the trunk first, so equal seeds give equal trunks). Python's and
    NumPy's generators are seeded too, so the RNG state a checkpoint saves is the same in every process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return K.Controller(K.matched_config(arm, backbone_width))


def make_optimizer(model, cfg: TrainConfig):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps,
                             weight_decay=cfg.weight_decay)


class Prefetcher:
    """Reads the next update's file-backed token states on a background thread while the current update computes.
    In-memory examples need nothing. File reads release the GIL, so the read overlaps the GPU work."""

    def __init__(self, examples: list, cfg: TrainConfig, start: int):
        import concurrent.futures
        self.examples, self.cfg = examples, cfg
        self.lazy = any(e.tokens is None for e in examples)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if self.lazy else None
        self.pending = {}
        if self.lazy and start < cfg.updates:
            self._submit(start)

    def _load(self, update: int) -> dict:
        idx = batch_indices(len(self.examples), update, self.cfg)
        return {self.examples[i].task_id: self.examples[i].token_states() for i in idx
                if self.examples[i].tokens is None}

    def _submit(self, update: int) -> None:
        self.pending[update] = self.pool.submit(self._load, update)

    def get(self, update: int) -> dict:
        """The states for `update` (blocking until read), after scheduling update + 1."""
        if not self.lazy:
            return {}
        if update + 1 < self.cfg.updates:
            self._submit(update + 1)
        future = self.pending.pop(update, None) or self.pool.submit(self._load, update)
        return future.result()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


PROFILER_STEP = None  # set by `cli train-timing --profile`; called once per update


def micro_batches(examples: list, idx: list, micro_batch: int) -> list:
    """The update's examples in micro-batches of similar context length, so little is padded. The update's example set,
    and so its summed loss and gradient, are those of `idx`; only the grouping (and with it the floating-point
    summation order) changes."""
    order = sorted(idx, key=lambda i: (examples[i].n_tokens, i))
    return [order[m0:m0 + micro_batch] for m0 in range(0, len(order), micro_batch)]


def fit(model: K.Controller, examples: list, cfg: TrainConfig, run_dir, *, resume_from=None, extra_config=None,
        log=print) -> dict:
    """Train for cfg.updates updates (from 0, or after the update stored in `resume_from`)."""
    run_dir = Path(run_dir)
    if model.cfg.arm != cfg.arm:
        raise ValueError(f"model arm {model.cfg.arm} != config arm {cfg.arm}")
    if cfg.effective_batch % cfg.micro_batch:
        raise ValueError("the effective batch must be a multiple of the micro-batch")
    if any(e.steps is None for e in examples):
        raise ValueError("every training example needs teacher-forcing steps")
    config = {"train": cfg.as_dict(), "controller": model.cfg.as_dict(),
              "parameters": K.parameter_report(model), "examples": [e.task_id for e in examples],
              "grammar": A.vocabulary_summary(), **(extra_config or {})}
    config_identity = write_config(run_dir, config)
    device = torch.device(cfg.device)
    model.to(device).train()
    optimizer = make_optimizer(model, cfg)
    start = 0
    if resume_from is not None:
        start = restore(load_checkpoint(resume_from), model, optimizer) + 1
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():  # lines past the resume point (or a whole earlier attempt) move aside, never vanish
        lines = log_path.read_text().splitlines()
        keep = [ln for ln in lines if ln.strip() and json.loads(ln)["update"] < start]
        moved = [ln for ln in lines if ln.strip() and json.loads(ln)["update"] >= start]
        if moved:
            with open(run_dir / "train_log.superseded.jsonl", "a") as fh:
                fh.write("".join(ln + "\n" for ln in moved))
            log_path.write_text("".join(ln + "\n" for ln in keep))
    checkpoints = []
    n = len(examples)
    prefetch = Prefetcher(examples, cfg, start)
    try:
        return _fit_loop(model, examples, cfg, run_dir, optimizer, start, n, prefetch, log_path, checkpoints, config,
                         config_identity, device, log)
    finally:
        prefetch.close()


def _fit_loop(model, examples, cfg, run_dir, optimizer, start, n, prefetch, log_path, checkpoints, config,
              config_identity, device, log) -> dict:
    for update in range(start, cfg.updates):
        t0 = time.perf_counter()
        idx = batch_indices(n, update, cfg)
        loaded = prefetch.get(update)
        t_wait = time.perf_counter() - t0
        steps_total = sum(len(examples[i].steps) for i in idx)
        optimizer.zero_grad(set_to_none=True)
        sums = {"total": 0.0, "family": 0.0, "op": 0.0, "pointer": 0.0}
        counts = {"steps": 0, "op_steps": 0, "pointer_steps": 0}
        if hasattr(model.semantic, "calls"):
            model.semantic.calls.clear()
        for group in micro_batches(examples, idx, cfg.micro_batch):
            batch = K.collate([examples[i] for i in group], device=device, token_states=loaded)
            out = model.losses(batch, cfg.loss_weights)
            (out["total"] / steps_total).backward()
            for key in sums:
                sums[key] += float(out[key].detach())
            for key in counts:
                counts[key] += out[key]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        record = {"update": update, "loss": sums["total"] / steps_total,
                  "family": sums["family"] / steps_total, "op": sums["op"] / steps_total,
                  "pointer": sums["pointer"] / steps_total, **counts, "examples": [examples[i].task_id for i in idx],
                  "grad_norm": grad_norm, "lr": cfg.lr, "wall_seconds": time.perf_counter() - t0,
                  "prefetch_wait_seconds": t_wait,
                  "head_calls": list(getattr(model.semantic, "calls", []))}
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if log and (update % 10 == 0 or update == cfg.updates - 1):
            log(f"{cfg.arm} update {update}: loss {record['loss']:.4f} (family {record['family']:.4f}, op "
                f"{record['op']:.4f}, pointer {record['pointer']:.4f}), {record['wall_seconds']:.2f} s")
        if PROFILER_STEP is not None:
            PROFILER_STEP()
        last = update == cfg.updates - 1
        if last or (cfg.checkpoint_every and (update + 1) % cfg.checkpoint_every == 0):
            checkpoints.append(save_checkpoint(run_dir, model, optimizer, update, config))
    return {"config_identity": config_identity, "run_dir": str(run_dir), "checkpoints": checkpoints,
            "log": str(log_path)}


# --- data: training examples from task ids -----------------------------------------------------------------------------
def gold_steps(gold_sql: str, encoding) -> tuple:
    """(steps, None) for a supervisable gold query, or (None, reason)."""
    from . import sql_to_ir
    plan, error = sql_to_ir.try_convert(gold_sql, encoding.schema.index)
    if error is not None:
        return None, f"unsupported: {error.reason}"
    try:
        return A.trace(plan, encoding.schema, encoding.spans), None
    except A.NotEncodable as e:
        return None, f"not encodable: {e.reason}"


def prepare_examples(task_ids, backbone, *, cache_root=None, lazy: bool = False, log=print) -> tuple:
    """Encode training tasks (frozen backbone, cached) and attach their gold teacher-forcing steps.
    Returns (prepared, skipped {task_id: reason}, costs {task_id: cost}). With `lazy`, each example keeps only the
    path of its cache file and training reads the states per update (the prefetcher overlaps the reads)."""
    from . import data, encode, retrieval
    from . import schema as schema_mod
    prepared, skipped, costs = [], {}, {}
    for n, tid in enumerate(task_ids):
        task = data.task(tid)
        if task.split != "train":
            raise ValueError(f"{tid} is a {task.split} question; training reads the training pool only")
        try:
            enc = encode.encode_request(backbone, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                        cache_root=cache_root)
        except encode.ContextTooLong as e:
            skipped[tid] = f"context too long: {e}"
            continue
        costs[tid] = enc.cost
        steps, reason = gold_steps(data.gold(tid).sql, enc)
        if steps is None:
            skipped[tid] = reason
            continue
        tokens_file = str(encode.cache_path(enc.cost["cache_key"], cache_root)) if lazy else None
        prepared.append(K.prepare(enc, steps, tokens_file=tokens_file))
        if log and (n + 1) % 32 == 0:
            log(f"prepared {n + 1}/{len(task_ids)} ({len(skipped)} skipped)")
    return prepared, skipped, costs


def smoke_train_ids(tokenizer, n: int = C.SMOKE_TRAIN["size"], log=None) -> tuple:
    """contracts.SMOKE_TRAIN: the training-pool candidates in smoke order (pins/smoke_train_candidates_v1.json),
    keeping those whose gold SQL compiles to the action grammar (conversion, then a traversal whose literals ground
    in this tokenizer's question/evidence tokens), first n. CPU only: tokenization, no backbone forward."""
    from . import data, encode, retrieval
    from . import schema as schema_mod
    pin = json.loads((C.PINS / "smoke_train_candidates_v1.json").read_text())
    kept, skipped = [], {}
    for tid in pin["task_ids"]:
        if len(kept) >= n:
            break
        task = data.task(tid)
        ctx, serialized, _ids, offsets = encode.fit_context(task, schema_mod.load_schema(task.db_id),
                                                            retrieval.retrieve(task), tokenizer,
                                                            C.Limits().max_context_tokens)

        class _Enc:  # the parts of an Encoding that gold_steps reads
            schema = A.SchemaItems(serialized.items)
            spans = encode.span_index(ctx, offsets)

        steps, reason = gold_steps(data.gold(tid).sql, _Enc)
        if steps is None:
            skipped[tid] = reason
        else:
            kept.append(tid)
    if len(kept) < n:
        raise RuntimeError(f"only {len(kept)} of the {len(pin['task_ids'])} candidates are supervisable")
    return kept, skipped


def train_arm(arm: str, task_ids, backbone, run_dir, *, updates: int = C.SMOKE_TRAIN["max_updates"],
              seed: int = C.SMOKE_TRAIN["seed"], device: "str | None" = None, micro_batch: int = 8,
              cache_root=None, prepared=None, resume_from=None, lazy: bool = False, log=print) -> dict:
    """The CLI entry point: encode (or reuse `prepared`), build the arm under `seed`, fit, checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    skipped, costs = {}, {}
    if prepared is None:
        prepared, skipped, costs = prepare_examples(task_ids, backbone, cache_root=cache_root, lazy=lazy, log=log)
    model = build_model(arm, backbone.hidden_size if backbone is not None else prepared[0].items.shape[1], seed)
    cfg = TrainConfig(arm=arm, seed=seed, updates=updates, micro_batch=micro_batch, device=device)
    extra = {"backbone": backbone.identity() if backbone is not None else None, "skipped": skipped,
             "requested_task_ids": list(task_ids)}
    result = fit(model, prepared, cfg, run_dir, resume_from=resume_from, extra_config=extra, log=log)
    result.update(skipped=skipped, encode_costs=costs, examples=len(prepared))
    return result


def grammar_coverage(tokenizer, task_ids=None, log=None) -> dict:
    """Per split: how many converted gold queries the action grammar encodes, why the rest do not, and how their
    literals ground (constants, NUM spans, TEXT spans by case form and wildcard). Literals are located in the request's
    real context, tokenized by `tokenizer` (CPU only). Returns {"summary": ..., "encodable": {task_id: n_actions}}."""
    from collections import Counter

    from . import data, encode, retrieval, sql_to_ir
    from . import schema as schema_mod
    from . import ir
    dev = set(data.dev_databases())
    rows = data.rows() if task_ids is None else [data.row(t) for t in task_ids]
    summary = {s: {"converted": 0, "encodable": 0, "decode_mismatch": 0, "not_encodable": Counter(),
                   "literals": Counter(), "actions": 0, "max_actions": 0, "families": Counter()}
               for s in ("train", "dev")}
    encodable = {}
    for n, r in enumerate(rows):
        s = summary["dev" if r.db_id in dev else "train"]
        task = data.task(r.task_id)
        full = schema_mod.load_schema(r.db_id)
        ctx, serialized, _ids, offsets = encode.fit_context(task, full, retrieval.retrieve(task), tokenizer,
                                                            C.Limits().max_context_tokens)
        schema_items = A.SchemaItems(serialized.items)
        plan, error = sql_to_ir.try_convert(r.sql, schema_items.index)
        if error is not None:
            continue
        s["converted"] += 1
        spans = encode.span_index(ctx, offsets)
        probe = A._Encoder(A.Machine(schema_items, spans))
        for node in ir.walk(plan):
            if isinstance(node, ir.Literal):
                try:
                    op, _ = probe.literal(node)
                    s["literals"]["CONST" if op.startswith("CONST") else op] += 1
                except A.NotEncodable:
                    s["literals"]["not groundable"] += 1
        try:
            steps = A.trace(plan, schema_items, spans)
        except A.NotEncodable as e:
            s["not_encodable"][e.reason] += 1
            continue
        s["encodable"] += 1
        if A.decode([st.action for st in steps[:-1]], schema_items, spans) != plan:
            s["decode_mismatch"] += 1
        s["actions"] += len(steps)
        s["max_actions"] = max(s["max_actions"], len(steps))
        s["families"].update(A.FAMILIES[st.family] if st.family < 4 else "STOP" for st in steps)
        encodable[r.task_id] = len(steps)
        if log and (n + 1) % 500 == 0:
            log(f"grammar coverage: {n + 1}/{len(rows)}")
    for s in summary.values():
        s["mean_actions"] = round(s.pop("actions") / max(1, s["encodable"]), 2)
        for key in ("not_encodable", "literals", "families"):
            s[key] = dict(sorted(s[key].items()))
    return {"summary": summary, "encodable": encodable}
