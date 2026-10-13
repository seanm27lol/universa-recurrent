"""Phase Eighteen: at which layers do the program tokens read the question? Exploratory.

protocols/phase_eighteen_marking_layers_brief.md is the frozen design. Example:

    What is x? Reply with only the integer. / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (y ends at 14)

Phases Sixteen and Seventeen found that, with the question first, the program
tokens mark the asked variable by attending to the question; blocking that
attention makes the late read take the other variable's value. Here the same
block (every program token -> every question token) is applied only in some
layers:

    from_k    layers k, k+1, ..., L-1       (the mark must be read before k)
    until_k   layers 0, 1, ..., k-1         (the mark must be read at k or later)

for k on Phase Eight's grid. Question-first prompts on programs ending on the
other variable, where the mark decides the answer; teacher-forced scoring of
the answer and of the other variable's final value.
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

from .answer_knockout import RELEASE_TOLERANCE, layer_grid, logits_at, score
from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .question_route_knockout import route_row
from .runner import release_models, source_identity

PROTOCOL = PROJECT / "protocols" / "phase_eighteen_marking_layers_brief.md"
FORMAT = "open_weight_lingua.marking_layers_knockout.v1"


@contextmanager
def block_layers(model, layers: set[int], queries: list[int], keys: list[int]):
    """Block attention from `queries` to `keys` in the given layers only (all heads); cf. answer_knockout.knockout."""
    handles = []
    query_index = torch.as_tensor(queries, dtype=torch.long)
    key_index = torch.as_tensor(keys, dtype=torch.long)

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or not torch.is_floating_point(mask) or mask.dim() != 4:
            raise RuntimeError("the block needs eager attention's 4-D additive float mask")
        mask = mask.clone()
        if len(queries) and len(keys):
            mask[:, :, query_index.to(mask.device)[:, None], key_index.to(mask.device)[None, :]] = torch.finfo(mask.dtype).min
        kwargs["attention_mask"] = mask
        return args, kwargs

    try:
        for index, layer in enumerate(decoder_layers(model)):
            if index in layers:
                handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def conditions(layers: int) -> dict[str, set[int]]:
    grid = layer_grid(layers)
    out = {"baseline": set(), "program_blind": set(range(layers))}
    out.update({f"from_{k}": set(range(k, layers)) for k in grid[1:]})
    out.update({f"until_{k}": set(range(k)) for k in grid[1:]})
    return out


def eligible(row: dict) -> bool:
    """Programs ending on the other variable, answer independent of it, other value distinct."""
    return not row["last_line_assigns_asked"] and not row["depends_on_other"] and "other" in row["values"]


def build_rows(calibration: dict, pilot: dict, tokenizer) -> list[dict]:
    rows = [route_row(item, split, tokenizer) for split, manifest in (("calibration", calibration), ("pilot", pilot))
            for item in manifest["inputs"]]
    return [r for r in rows if eligible(r)]


def evaluate_row(model, row: dict, layers: int, device: str) -> dict:
    answer = row["sequences"]["answer"]
    baseline = logits_at(model, answer["ids"], answer["positions"], device)
    with block_layers(model, set(), answer["positions"], []):
        if not torch.equal(logits_at(model, answer["ids"], answer["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    program_tokens = list(range(row["span"][0], row["span"][1] + 1))
    question = list(range(row["question"][0], row["question"][1] + 1))
    out = {}
    for name, chosen in conditions(layers).items():
        scored = {}
        for value, q in row["sequences"].items():
            with block_layers(model, chosen, program_tokens, question):
                scored[value] = score(logits_at(model, q["ids"], q["positions"], device), q["targets"])
        out[name] = scored
    return out


def summarize(rows: list[dict], results: list[dict], layers: int) -> dict:
    names = list(conditions(layers))

    def table(keep):
        chosen = [r for row, r in zip(rows, results) if keep(row)]
        return {"n": len(chosen), **{c: {"accuracy": float(np.mean([r[c]["answer"]["correct"] for r in chosen])),
                                         "says_other": float(np.mean([r[c]["other"]["correct"] for r in chosen]))} for c in names}}

    by_split = {s: table(lambda r, s=s: r["split"] == s) for s in ("calibration", "pilot")}
    grid = layer_grid(layers)[1:]
    windows = {}
    for split, cells in by_split.items():
        base = cells["baseline"]["says_other"]
        # Last layer that matters: the smallest k such that blocking from k and every later grid layer stays within tolerance.
        ok_from = [cells[f"from_{k}"]["says_other"] <= base + RELEASE_TOLERANCE for k in grid]
        from_release = next((k for i, k in enumerate(grid) if all(ok_from[i:])), None)
        # First layer that matters: the largest k such that blocking until k and every earlier grid layer stays within tolerance.
        ok_until = [cells[f"until_{k}"]["says_other"] <= base + RELEASE_TOLERANCE for k in grid]
        until_onset = next((k for i, k in reversed(list(enumerate(grid))) if all(ok_until[: i + 1])), None)
        windows[split] = {"from_release": from_release, "until_onset": until_onset}
    return {"layers": layers, "grid": layer_grid(layers), "all": table(lambda r: True), "by_split": by_split,
            "window": windows, "rule": "says_other within +0.05 of baseline (Phase Eight's tolerance)"}


# ---------------------------------------------------------------- execution


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    for manifest in (calibration, pilot):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    rows = build_rows(calibration, pilot, tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(args.protocol), "protocol_sha256": sha256_file(args.protocol),
        "source_file_sha256": source_identity(), "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows), "stimuli_sha256": sha256_file(stimuli),
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    layers = len(decoder_layers(model))
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            result = evaluate_row(model, row, layers, args.device)
            results.append(result)
            stream.write(json.dumps({"id": row["id"], **result}) + "\n")
            stream.flush()
            if index % 64 == 0:
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
    parser.add_argument("--protocol", type=Path, default=PROTOCOL,
                        help="the frozen brief this run executes under (default: this module's own brief)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p18-layers-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
            "results_sha256": sha256_file(results) if results.exists() else None,
            "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
            "setup_seconds": float(os.environ["OWL_SETUP_SECONDS"]) if "OWL_SETUP_SECONDS" in os.environ else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        })
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
