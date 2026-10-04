"""Phase Four survey: can a probe read the running state at each statement boundary?

protocols/phase_four_brief.md is the frozen design. Example: in

    x = 9 / y = 8 / y = 3 / y = y + 3 / x = x + 1

after the fourth line x is 9 (carried from line 1) and y is 6 (just computed by
arithmetic). At the token completing each line from the second on, one probe
per variable and layer tries to read x's and y's values as of that line.
Accuracy is reported by category: updated by a literal, by arithmetic or by a
copy, or carried. It runs the target model only.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np
import torch

from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .state_survey import ALPHA_GRID, PERMUTATION_SEED, TRAIN_GROUPS, VALUES, stored
from .target import pad_to_bucket
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_four_brief.md"
FORMAT = "open_weight_lingua.state_trace_survey.v1"
CATEGORIES = ("literal", "arithmetic", "copy", "carried")
R_THRESHOLD, R_COMPUTED = 0.80, 0.50


# ---------------------------------------------------------------- programs, positions, labels


def program_rows(manifest: dict, split_name: str) -> list[dict]:
    """One row per (group, side): both queries share every program token."""
    seen, out = set(), []
    for row in manifest["inputs"]:
        key = (row["group_id"], row["side"])
        if key in seen:
            continue
        seen.add(key)
        index = int(row["group_id"].rsplit("-", 1)[1])
        part = ("train" if index < TRAIN_GROUPS else "select") if split_name == "calibration" else "test"
        out.append({**row, "program_key": f"{row['group_id']}-{row['side']}", "part": part})
    return out


def boundary_positions(tokenizer, ids: list[int], prompt: str) -> list[int]:
    """Token completing statement i, for i = 2..n (1-based)."""
    lines = program_text(prompt).split("\n")
    out, start = [], 0
    for i in range(2, len(lines) + 1):
        target = "\n".join(lines[:i])
        for position in range(start, len(ids)):
            if target in tokenizer.decode(ids[: position + 1], skip_special_tokens=False):
                out.append(position)
                start = position + 1
                break
        else:
            raise ValueError(f"statement {i} not found in the decoded prompt")
    return out


def trace_labels(prompt: str) -> list[dict]:
    """Per boundary i >= 2: x and y as of line i, each variable's category, computed flag, copy guess."""
    statements = program(prompt)
    lines = program_text(prompt).split("\n")
    out = []
    for i in range(2, len(statements) + 1):
        state = interpret(statements[:i])
        written = {int(n) for n in re.findall(r"\d+", "\n".join(lines[:i]))}
        last = statements[i - 1]
        entry = {"line": i}
        for variable in ("x", "y"):
            if last.variable == variable:
                category = {"assign": "literal", "add": "arithmetic", "subtract": "arithmetic", "copy": "copy"}[last.operation]
            else:
                category = "carried"
            assigned = [s.operand for s in statements[:i] if s.variable == variable and s.operation == "assign"]
            entry[variable] = {
                "value": state[variable],
                "category": category,
                "computed": state[variable] not in written,
                "copy_guess": assigned[-1],
            }
        out.append(entry)
    return out


# ---------------------------------------------------------------- probes (GPU float64, same method as Stage 0)


def fit_predict(train, labels_train, evaluate, alphas, device) -> list[torch.Tensor]:
    """Dual ridge one-hot probe; argmax predictions on `evaluate` for each alpha."""
    x = torch.as_tensor(train, dtype=torch.float64, device=device)
    e = torch.as_tensor(evaluate, dtype=torch.float64, device=device)
    mean = x.mean(dim=0, keepdim=True)
    x, e = x - mean, e - mean
    gram = x @ x.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    scale = float(torch.diagonal(gram).mean())
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    targets = torch.nn.functional.one_hot(torch.as_tensor(labels_train, device=device), VALUES).to(torch.float64)
    projected = eigenvectors.T @ targets
    kernel = e @ x.T
    out = []
    for alpha in alphas:
        coefficients = eigenvectors @ (projected / (eigenvalues + alpha * scale)[:, None])
        out.append((kernel @ coefficients).argmax(dim=1).cpu().numpy())
    return out


def category_accuracy(predicted, truth, categories, computed, wanted: str, only_computed=False):
    mask = categories == wanted
    if only_computed:
        mask &= computed
    return float(np.mean(predicted[mask] == truth[mask])) if mask.any() else None


def survey(features: np.ndarray, records: list[dict], device: str) -> dict:
    """Per layer and variable: alpha chosen on select (pooled accuracy); per-category accuracies."""
    part = np.array([r["part"] for r in records])
    train, select, test = part == "train", part == "select", part == "test"
    evaluate = select | test
    rng = np.random.default_rng(PERMUTATION_SEED)
    grid = {}
    for layer in range(features.shape[1]):
        block = features[:, layer]
        entry = {}
        for variable in ("x", "y"):
            values = np.array([r[variable]["value"] for r in records])
            categories = np.array([r[variable]["category"] for r in records])
            computed = np.array([r[variable]["computed"] for r in records])
            predictions = fit_predict(block[train], values[train], block[evaluate], ALPHA_GRID, device)
            n_select = int(select.sum())
            scores = [float(np.mean(p[:n_select] == values[select])) for p in predictions]
            best = max(range(len(ALPHA_GRID)), key=lambda i: (scores[i], i))
            chosen = predictions[best]
            result = {"alpha": ALPHA_GRID[best], "select_overall": scores[best]}
            for split_name, mask, predicted in (("select", select, chosen[:n_select]), ("test", test, chosen[n_select:])):
                result[split_name] = {
                    "overall": float(np.mean(predicted == values[mask])),
                    **{c: category_accuracy(predicted, values[mask], categories[mask], computed[mask], c) for c in CATEGORIES},
                    **{f"{c}_computed": category_accuracy(predicted, values[mask], categories[mask], computed[mask], c, True) for c in CATEGORIES},
                }
            permuted = fit_predict(block[train], rng.permutation(values[train]), block[test], (ALPHA_GRID[best],), device)[0]
            result["test_permuted_overall"] = float(np.mean(permuted == values[test]))
            entry[variable] = result
        grid[layer] = entry
        if layer % 8 == 0:
            print(f"  probes: layer {layer}", flush=True)
    return grid


def _layer_for(grid: dict, category: str) -> int:
    """argmax over layers of min_v select accuracy for a category; ties to the earlier layer."""
    def key(layer):
        scores = [grid[layer][v]["select"][category] for v in ("x", "y")]
        return (min(s if s is not None else -1.0 for s in scores), -layer)
    return max(grid, key=key)


def readings(grid: dict) -> dict:
    carried_layer, arithmetic_layer = _layer_for(grid, "carried"), _layer_for(grid, "arithmetic")
    carried = {v: grid[carried_layer][v]["test"] for v in ("x", "y")}
    arithmetic = {v: grid[arithmetic_layer][v]["test"] for v in ("x", "y")}

    def at_least(value, threshold):
        return value is not None and value >= threshold

    r1 = all(at_least(carried[v]["carried"], R_THRESHOLD) and at_least(carried[v]["carried_computed"], R_COMPUTED) for v in ("x", "y"))
    r2 = all(at_least(arithmetic[v]["arithmetic"], R_THRESHOLD) for v in ("x", "y"))
    return {
        "R1": {"layer": carried_layer, "test": {v: {k: carried[v][k] for k in ("carried", "carried_computed", "overall")} for v in ("x", "y")}, "supported": r1},
        "R2": {"layer": arithmetic_layer, "test": {v: {k: arithmetic[v][k] for k in ("arithmetic", "arithmetic_computed", "literal", "overall")} for v in ("x", "y")}, "supported": r2},
        "gate_G4": {"passed": r1 and r2},
    }


def baselines(records: list[dict]) -> dict:
    test = [r for r in records if r["part"] == "test"]
    out = {}
    for variable in ("x", "y"):
        out[variable] = {}
        for category in CATEGORIES:
            rows = [r for r in test if r[variable]["category"] == category]
            out[variable][category] = {
                "rows": len(rows),
                "copy_last_literal": float(np.mean([r[variable]["copy_guess"] == r[variable]["value"] for r in rows])) if rows else None,
                "computed_fraction": float(np.mean([r[variable]["computed"] for r in rows])) if rows else None,
            }
    return out


# ---------------------------------------------------------------- execution


@torch.inference_mode()
def capture(model, programs: list[dict], device: str, layers: int) -> np.ndarray:
    blocks = []
    for index, row in enumerate(programs):
        ids = torch.tensor([row["input_ids"]], device=device)
        mask = torch.tensor([row["attention_mask"]], device=device)
        ids, mask = pad_to_bucket(ids, mask)
        hidden = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True).hidden_states
        if len(hidden) != layers + 1:
            raise RuntimeError("unexpected hidden-state count")
        stacked = torch.stack([h[0] for h in hidden[1:]])  # [L, T, d]
        for position in row["boundaries"]:
            blocks.append(stored(stacked[:, position]))
        if index % 128 == 0:
            print(f"  capture {index}/{len(programs)}", flush=True)
    return np.stack(blocks)


def run(args, run_dir: RunDirectory, timings: dict):
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    for manifest in (calibration, pilot):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    programs = program_rows(calibration, "calibration") + program_rows(pilot, "pilot")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"])
    records = []
    for row in programs:
        row["boundaries"] = boundary_positions(tokenizer, row["input_ids"], row["prompt"])
        for position, label in zip(row["boundaries"], trace_labels(row["prompt"])):
            records.append({"program_key": row["program_key"], "part": row["part"], "position": position, **label})
    family = args.lock.stem.replace("model-lock-", "") if args.lock.stem != "model-lock" else "qwen2.5-7b"
    write_json(
        run_dir.path / "manifest.json",
        {
            "format": FORMAT,
            "family": family,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_head": _git_head(),
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_file_sha256": source_identity(),
            "model_lock_sha256": sha256_file(args.lock),
            "inputs": {
                "calibration_run": args.calibration_run.name,
                "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                "pilot_run": args.pilot_run.name,
                "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json"),
            },
            "programs": len(programs),
            "boundaries": len(records),
            "alpha_grid": ALPHA_GRID,
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
            "records": records,
        },
    )
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    features = capture(model, programs, args.device, config.num_hidden_layers)
    del model
    release_models()
    timings["capture"] = time.perf_counter() - start
    start = time.perf_counter()
    grid = survey(features, records, args.device)
    timings["probes"] = time.perf_counter() - start
    summary = {
        "layers": int(features.shape[1]),
        "rows": {p: sum(r["part"] == p for r in records) for p in ("train", "select", "test")},
        "baselines": baselines(records),
        **readings(grid),
    }
    results = [{"layer": layer, **entry} for layer, entry in sorted(grid.items())]
    return results, summary


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
    family = args.lock.stem.replace("model-lock-", "") if args.lock.stem != "model-lock" else "qwen2.5-7b"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p4-trace-{family}-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error = "FAILED", None
    try:
        results, summary = run(args, run_dir, timings)
        write_json(run_dir.path / "results.json", results)
        write_json(run_dir.path / "summary.json", summary)
        status = "COMPLETE"
    except Exception as failure:  # recorded in completion.json, then re-raised
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        write_json(
            run_dir.path / "completion.json",
            {
                "status": status,
                "error": error,
                "timings_seconds": timings,
                "wall_seconds": time.perf_counter() - started,
                "setup_seconds": float(os.environ.get("OWL_SETUP_SECONDS", "nan")),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            },
        )
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
