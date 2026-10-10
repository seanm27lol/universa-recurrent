"""Phase Eight: when does the answer stop needing the program? Attention knockout, exploratory.

protocols/phase_eight_knockout_brief.md is the frozen design. Example: for

    x = 7 / y = 16 / y = y + 1 / y = y - 3 / What is x?

the model answers 7. From layer k onward, attention from later positions to
the program tokens is blocked (Geva et al. 2023). If the answer survives, it
no longer needed direct access to the program after layer k. Scoring is by
teacher forcing: the correct answer's digit tokens are appended, and an answer
counts as correct when every digit is the argmax at its position.
It runs the target model only, on reused calibration and pilot prompts.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .metrics import answer_tokens
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .target import pad_to_bucket
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_eight_knockout_brief.md"
FORMAT = "open_weight_lingua.answer_knockout.v1"
VARIANTS = ("answer", "after_program")
GRID_STEPS = 8
RELEASE_TOLERANCE = 0.05
KIND = {"assign": "literal", "add": "arithmetic", "subtract": "arithmetic", "copy": "copy"}


# ---------------------------------------------------------------- spans and sequences


def program_span(tokenizer, ids: list[int], body: str) -> tuple[int, int]:
    """First and last token of the program text inside the prompt (inclusive)."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    at = text.find(body)
    if at < 0 or text.find(body, at + 1) >= 0:
        raise ValueError("program text not found exactly once in the decoded prompt")
    first = next(p for p in range(len(ids)) if len(tokenizer.decode(ids[: p + 1], skip_special_tokens=False)) > at)
    last = next(p for p in range(len(ids)) if body in tokenizer.decode(ids[: p + 1], skip_special_tokens=False))
    if tokenizer.decode(ids[first : last + 1], skip_special_tokens=False).strip() != body:
        raise ValueError("located span does not decode to the program")
    return first, last


def layer_grid(layers: int) -> list[int]:
    return [j * layers // GRID_STEPS for j in range(GRID_STEPS)]


def teacher_forced(row: dict, tokenizer) -> dict:
    """Prompt plus answer digits (no EOS), the predicting positions and their targets."""
    digits = answer_tokens(tokenizer, row["answer"])[:-1]
    ids = list(row["input_ids"]) + digits
    start = len(row["input_ids"]) - 1
    return {"ids": ids, "positions": list(range(start, start + len(digits))), "targets": digits}


# ---------------------------------------------------------------- the knockout


@contextmanager
def knockout(model, from_layer: int, queries: list[int], keys: list[int]):
    """Block attention from `queries` to `keys` in every layer >= from_layer, all heads.

    Implemented as a forward pre-hook on each attention module that adds the
    dtype minimum to the additive mask at those (query, key) cells. Requires
    eager attention's float mask; anything else fails loudly.
    """
    handles = []
    query_index = torch.as_tensor(queries, dtype=torch.long)
    key_index = torch.as_tensor(keys, dtype=torch.long)

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or not torch.is_floating_point(mask) or mask.dim() != 4:
            raise RuntimeError("knockout needs eager attention's 4-D additive float mask")
        mask = mask.clone()
        if len(queries) and len(keys):
            q = query_index.to(mask.device)[:, None]
            k = key_index.to(mask.device)[None, :]
            mask[:, :, q, k] = torch.finfo(mask.dtype).min
        kwargs["attention_mask"] = mask
        return args, kwargs

    try:
        for index, layer in enumerate(decoder_layers(model)):
            if index >= from_layer:
                handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def logits_at(model, ids: list[int], positions: list[int], device: str) -> torch.Tensor:
    tensor = torch.tensor([ids], device=device)
    mask = torch.ones_like(tensor)
    tensor, mask = pad_to_bucket(tensor, mask)
    return model(input_ids=tensor, attention_mask=mask, use_cache=False).logits[0, positions].float().cpu()


def score(logits: torch.Tensor, targets: list[int]) -> dict:
    logprobs = logits.log_softmax(-1)
    target = torch.as_tensor(targets)
    return {"correct": bool((logits.argmax(-1) == target).all()),
            "logprob": float(logprobs.gather(1, target[:, None]).sum())}


def evaluate_row(model, row: dict, sequence: dict, span: tuple[int, int], layers: int, device: str) -> dict:
    """Baseline, the no-op control, and every (variant, k) for one prompt."""
    ids, positions, targets = sequence["ids"], sequence["positions"], sequence["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        noop = logits_at(model, ids, positions, device)
    if not torch.equal(noop, baseline):
        raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    keys = list(range(span[0], span[1] + 1))
    queries = {"answer": positions, "after_program": list(range(span[1] + 1, len(ids)))}
    out = {"baseline": score(baseline, targets)}
    for variant in VARIANTS:
        for k in layer_grid(layers):
            with knockout(model, k, queries[variant], keys):
                out[f"{variant}@{k}"] = score(logits_at(model, ids, positions, device), targets)
    return out


# ---------------------------------------------------------------- execution


def prompt_rows(manifest: dict, split: str) -> list[dict]:
    out = []
    for row in manifest["inputs"]:
        statements = program(row["prompt"])
        variable = row["variable"]
        if str(interpret(statements)[variable]) != row["answer"]:
            raise ValueError(f"{row['id']}: answer disagrees with the interpreter")
        last_update = next(s for s in reversed(statements) if s.variable == variable)
        out.append({"id": row["id"], "group_id": row["group_id"], "split": split, "variable": variable,
                    "prompt": row["prompt"], "input_ids": row["input_ids"], "answer": row["answer"],
                    "asked_last_update_kind": KIND[last_update.operation],
                    "last_line_assigns_asked": statements[-1].variable == variable})
    return out


def summarize(rows: list[dict], results: list[dict], layers: int) -> dict:
    conditions = ["baseline"] + [f"{v}@{k}" for v in VARIANTS for k in layer_grid(layers)]

    def table(keep):
        chosen = [r for row, r in zip(rows, results) if keep(row)]
        return {c: {"n": len(chosen), "accuracy": float(np.mean([r[c]["correct"] for r in chosen])),
                    "mean_logprob": float(np.mean([r[c]["logprob"] for r in chosen]))} for c in conditions} if chosen else {}

    by_split = {split: table(lambda r, s=split: r["split"] == s) for split in ("calibration", "pilot")}
    release = {}
    for variant in VARIANTS:
        entry = {}
        for split in ("calibration", "pilot"):
            cells = by_split[split]
            ok = [cells[f"{variant}@{k}"]["accuracy"] >= cells["baseline"]["accuracy"] - RELEASE_TOLERANCE for k in layer_grid(layers)]
            qualifying = [k for i, k in enumerate(layer_grid(layers)) if all(ok[i:])]
            entry[split] = qualifying[0] if qualifying else None
        release[variant] = entry
    strata = {}
    for name, levels in (("asked_last_update_kind", ("literal", "arithmetic", "copy")), ("last_line_assigns_asked", (True, False))):
        strata[name] = {str(level): table(lambda r, n=name, v=level: r[n] == v) for level in levels}
    return {"layers": layers, "grid": layer_grid(layers), "by_split": by_split,
            "release_layer": release, "release_rule": "smallest grid k with this and every larger k within 0.05 of baseline accuracy",
            "strata_all_splits": strata}


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    for manifest in (calibration, pilot):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    rows = prompt_rows(calibration, "calibration") + prompt_rows(pilot, "pilot")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    spans = [program_span(tokenizer, r["input_ids"], program_text(r["prompt"])) for r in rows]
    sequences = [teacher_forced(r, tokenizer) for r in rows]
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows), "variants": VARIANTS, "release_tolerance": RELEASE_TOLERANCE,
        "rows": [{"id": r["id"], "split": r["split"], "span": s, "positions": q["positions"], "targets": q["targets"]}
                 for r, s, q in zip(rows, spans, sequences)],
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    layers = len(decoder_layers(model))
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, (row, sequence, span) in enumerate(zip(rows, sequences, spans)):
            result = evaluate_row(model, row, sequence, span, layers, args.device)
            results.append(result)
            stream.write(json.dumps({"id": row["id"], **result}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    summary = summarize(rows, results, layers)
    write_json(run_dir.path / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p8-knockout-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error, trace = "FAILED", None, None
    try:
        run(args, run_dir, timings)
        status = "COMPLETE"
    except BaseException as failure:  # recorded in completion.json, then re-raised
        error, trace = f"{type(failure).__name__}: {failure}", traceback.format_exc()
        raise
    finally:
        results = run_dir.path / "results.jsonl"
        write_json(run_dir.path / "completion.json", {
            "status": status, "error": error, "traceback": trace,
            "rows_written": sum(1 for _ in results.open()) if results.exists() else 0,
            "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
            "setup_seconds": float(os.environ["OWL_SETUP_SECONDS"]) if "OWL_SETUP_SECONDS" in os.environ else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        })
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
