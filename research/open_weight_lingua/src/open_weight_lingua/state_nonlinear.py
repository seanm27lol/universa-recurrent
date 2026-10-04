"""Phase Five: can a small nonlinear probe read the program state where linear ones could not?

protocols/phase_five_brief.md is the frozen design. Example: for

    x = 9 / y = 8 / y = 3 / y = y + 3 / x = x + 1 / What is x?

no linear probe read x = 10 and y = 6 together at any surveyed site (Phase
Three), nor the 6 right after `y = y + 3` (Phase Four). Here a one-hidden-layer
MLP gets the same chance under the same selection rules, with a permuted-label
control at every selected site.

Part A reuses Phase Three's saved activations; Part B re-captures Phase Four's
statement boundaries at the positions its manifest recorded. Both input runs
are read-only. It runs the target model only, for Part B's capture.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid
import zlib

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import torch

from .artifacts import RunDirectory, sha256_file, write_json
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .state_survey import PERMUTATION_SEED, POSITIONS, VALUES, gate_g0, select_site
from .state_trace_survey import CATEGORIES, baselines, boundary_positions, capture, program_rows, readings

PROTOCOL = PROJECT / "protocols" / "phase_five_brief.md"
FORMAT = "open_weight_lingua.state_nonlinear.v1"
SEED = 501100
WEIGHT_DECAYS = (0.01, 0.1)
EPOCHS, EVAL_EVERY, HIDDEN, DROPOUT, LEARNING_RATE = 300, 10, 256, 0.1, 1e-3
CONTROL_MAX = 0.20


# ---------------------------------------------------------------- the probe


def fit_seed(*identity) -> int:
    """Fixed per fit: 501100 plus a CRC of the fit's identity (part, site, variable)."""
    return SEED + zlib.crc32("|".join(map(str, identity)).encode()) % 1_000_000


def mlp_probe(train_x, train_y, select_x, select_y, evaluate_x, *, device: str, seed: int, choice=None) -> dict:
    """Full-batch MLP probe; returns argmax predictions on select and evaluate rows.

    Without `choice`, (weight decay, epoch) is chosen on select accuracy, checked
    every EVAL_EVERY epochs; ties go to the larger decay, then the earlier epoch.
    With `choice=(decay, epoch)` those are fixed (the permuted-label control).
    """
    x = torch.as_tensor(np.asarray(train_x), dtype=torch.float32, device=device)
    mean, std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    s = (torch.as_tensor(np.asarray(select_x), dtype=torch.float32, device=device) - mean) / std
    e = (torch.as_tensor(np.asarray(evaluate_x), dtype=torch.float32, device=device) - mean) / std
    y = torch.as_tensor(np.asarray(train_y), dtype=torch.long, device=device)
    select_truth = np.asarray(select_y)
    decays, last = (WEIGHT_DECAYS, EPOCHS) if choice is None else ((choice[0],), choice[1])
    best = None
    for decay in decays:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.GELU(), torch.nn.Dropout(DROPOUT), torch.nn.Linear(HIDDEN, VALUES)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=decay)
        for epoch in range(1, last + 1):
            model.train()
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
            if epoch % EVAL_EVERY or (choice is not None and epoch != last):
                continue
            model.eval()
            with torch.no_grad():
                select_pred = model(s).argmax(dim=1).cpu().numpy()
                evaluate_pred = model(e).argmax(dim=1).cpu().numpy()
            accuracy = float(np.mean(select_pred == select_truth))
            key = (accuracy, decay, -epoch)
            if best is None or key > best[0]:
                best = (key, {"weight_decay": decay, "epoch": epoch, "select_accuracy": accuracy,
                              "select_pred": select_pred, "evaluate_pred": evaluate_pred})
    return best[1]


def _accuracy(predicted, truth, mask=None):
    if mask is not None:
        if not mask.any():
            return None
        predicted, truth = predicted[mask], truth[mask]
    return float(np.mean(predicted == truth))


# ---------------------------------------------------------------- Part A: the final state (Phase Three's question)


def part_a(rows: list[dict], load, device: str) -> tuple[list, dict]:
    """`rows` are Phase Three's manifest rows; `load(position)` returns its [N, L, d] features.

    Entries keep Phase Three's shape (`entry[v]["onehot"]`) so its frozen
    site rule and gate G0 apply unchanged.
    """
    part = np.array([r["part"] for r in rows])
    train, select, test = part == "train", part == "select", part == "test"
    asked = np.array([r["id"].rsplit("-", 1)[1] for r in rows])
    values = {v: np.array([r["labels"][v]["value"] for r in rows]) for v in ("x", "y")}
    computed = {v: np.array([r["labels"][v]["computed"] for r in rows]) for v in ("x", "y")}
    grid = {}
    for position in POSITIONS:
        features = load(position)
        for layer in range(features.shape[1]):
            block = features[:, layer]
            entry = {}
            for v in ("x", "y"):
                fit = mlp_probe(block[train], values[v][train], block[select], values[v][select], block[test],
                                device=device, seed=fit_seed("A", position, layer, v))
                predicted, truth = fit["evaluate_pred"], values[v][test]
                entry[v] = {"onehot": {
                    "weight_decay": fit["weight_decay"], "epoch": fit["epoch"],
                    "select": fit["select_accuracy"],
                    "select_computed": _accuracy(fit["select_pred"], values[v][select], computed[v][select]),
                    "test": _accuracy(predicted, truth),
                    "test_computed": _accuracy(predicted, truth, computed[v][test]),
                    "test_asked": _accuracy(predicted, truth, asked[test] == v),
                    "test_not_asked": _accuracy(predicted, truth, asked[test] != v),
                }}
            grid[(position, layer)] = entry
        print(f"  part A: {position} done", flush=True)
    position, layer = select_site(grid)
    chosen = grid[(position, layer)]
    block = load(position)[:, layer]
    rng = np.random.default_rng(PERMUTATION_SEED)
    controls = {}
    for v in ("x", "y"):
        probe = chosen[v]["onehot"]
        fit = mlp_probe(block[train], rng.permutation(values[v][train]), block[select], values[v][select], block[test],
                        device=device, seed=fit_seed("A-control", position, layer, v),
                        choice=(probe["weight_decay"], probe["epoch"]))
        controls[v] = _accuracy(fit["evaluate_pred"], values[v][test])
    g0 = gate_g0(chosen)
    summary = {
        "rows": {p: int(np.sum(part == p)) for p in ("train", "select", "test")},
        "selected_site": {"position": position, "layer": layer, "probes": chosen, "control_test": controls},
        "gate_G5a": {**g0, "control_max": max(controls.values()),
                     "passed": g0["passed"] and max(controls.values()) <= CONTROL_MAX},
    }
    results = [{"position": p, "layer": i, **entry}
               for (p, i), entry in sorted(grid.items(), key=lambda kv: (POSITIONS.index(kv[0][0]), kv[0][1]))]
    return results, summary


# ---------------------------------------------------------------- Part B: the running state (Phase Four's question)


def part_b(features: np.ndarray, records: list[dict], device: str) -> tuple[list, dict]:
    """One MLP per layer and variable, pooled over boundaries; Phase Four's readings and layer rule."""
    part = np.array([r["part"] for r in records])
    train, select, test = part == "train", part == "select", part == "test"
    values = {v: np.array([r[v]["value"] for r in records]) for v in ("x", "y")}
    categories = {v: np.array([r[v]["category"] for r in records]) for v in ("x", "y")}
    computed = {v: np.array([r[v]["computed"] for r in records]) for v in ("x", "y")}
    grid = {}
    for layer in range(features.shape[1]):
        block = features[:, layer]
        entry = {}
        for v in ("x", "y"):
            fit = mlp_probe(block[train], values[v][train], block[select], values[v][select], block[test],
                            device=device, seed=fit_seed("B", layer, v))
            result = {"weight_decay": fit["weight_decay"], "epoch": fit["epoch"], "select_overall": fit["select_accuracy"]}
            for name, mask, predicted in (("select", select, fit["select_pred"]), ("test", test, fit["evaluate_pred"])):
                truth, cats, comp = values[v][mask], categories[v][mask], computed[v][mask]
                result[name] = {
                    "overall": _accuracy(predicted, truth),
                    **{c: _accuracy(predicted, truth, cats == c) for c in CATEGORIES},
                    **{f"{c}_computed": _accuracy(predicted, truth, (cats == c) & comp) for c in CATEGORIES},
                }
            entry[v] = result
        grid[layer] = entry
        if layer % 8 == 0:
            print(f"  part B: layer {layer}", flush=True)
    summary = readings(grid)
    del summary["gate_G4"]
    rng = np.random.default_rng(PERMUTATION_SEED)
    for name in ("R1", "R2"):
        layer = summary[name]["layer"]
        block = features[:, layer]
        controls = {}
        for v in ("x", "y"):
            probe = grid[layer][v]
            fit = mlp_probe(block[train], rng.permutation(values[v][train]), block[select], values[v][select], block[test],
                            device=device, seed=fit_seed("B-control", name, layer, v),
                            choice=(probe["weight_decay"], probe["epoch"]))
            controls[v] = _accuracy(fit["evaluate_pred"], values[v][test])
        summary[name]["probe_supported"] = summary[name].pop("supported")
        summary[name]["control_test"] = controls
        summary[name]["supported"] = summary[name]["probe_supported"] and max(controls.values()) <= CONTROL_MAX
    summary["gate_G5b"] = {"passed": summary["R1"]["supported"] and summary["R2"]["supported"]}
    summary["rows"] = {p: int(np.sum(part == p)) for p in ("train", "select", "test")}
    summary["baselines"] = baselines(records)
    return [{"layer": layer, **entry} for layer, entry in sorted(grid.items())], summary


# ---------------------------------------------------------------- execution


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def _programs_and_records(args, trace_manifest: dict, tokenizer) -> tuple[list, list]:
    """Phase Four's programs with its recorded boundaries; records reordered to capture order."""
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    programs = program_rows(calibration, "calibration") + program_rows(pilot, "pilot")
    by_program = {}
    for record in trace_manifest["records"]:
        by_program.setdefault(record["program_key"], []).append(record)
    records = []
    for row in programs:
        mine = by_program.pop(row["program_key"])
        row["boundaries"] = [r["position"] for r in mine]
        if row["boundaries"] != boundary_positions(tokenizer, row["input_ids"], row["prompt"]):
            raise ValueError(f"recorded boundaries differ for {row['program_key']}")
        if any(r["part"] != row["part"] for r in mine):
            raise ValueError(f"recorded split differs for {row['program_key']}")
        records.extend(mine)
    if by_program:
        raise ValueError("Phase Four records name programs absent from the input runs")
    return programs, records


def run(args, run_dir: RunDirectory, timings: dict):
    survey_manifest = json.loads((args.survey_run / "manifest.json").read_text())
    trace_manifest = json.loads((args.trace_run / "manifest.json").read_text())
    family = _family(args.lock)
    for manifest in (survey_manifest, trace_manifest):
        if manifest["model_lock_sha256"] != sha256_file(args.lock) or manifest["family"] != family:
            raise ValueError("lock or family differs from the Phase Three / Phase Four runs")
        if (manifest["inputs"]["calibration_run"], manifest["inputs"]["pilot_run"]) != (args.calibration_run.name, args.pilot_run.name):
            raise ValueError("calibration or pilot run differs from the Phase Three / Phase Four inputs")
    features_path = args.survey_run / "raw" / "features.safetensors"
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"])
    programs, records = _programs_and_records(args, trace_manifest, tokenizer)
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
                "phase_three_run": args.survey_run.name,
                "phase_three_manifest_sha256": sha256_file(args.survey_run / "manifest.json"),
                "phase_three_features_sha256": sha256_file(features_path),
                "phase_four_run": args.trace_run.name,
                "phase_four_manifest_sha256": sha256_file(args.trace_run / "manifest.json"),
                "calibration_run": args.calibration_run.name,
                "pilot_run": args.pilot_run.name,
            },
            "probe": {"hidden": HIDDEN, "dropout": DROPOUT, "learning_rate": LEARNING_RATE, "epochs": EPOCHS,
                      "eval_every": EVAL_EVERY, "weight_decays": WEIGHT_DECAYS, "seed": SEED,
                      "permutation_seed": PERMUTATION_SEED, "control_max": CONTROL_MAX},
            "boundaries": len(records),
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
        },
    )
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    features = capture(model, programs, args.device, config.num_hidden_layers)
    del model
    release_models()
    save_file({"boundaries": features}, str(run_dir.path / "raw" / "boundary_features.safetensors"))
    timings["capture"] = time.perf_counter() - start
    start = time.perf_counter()
    with safe_open(str(features_path), framework="numpy") as handle:
        results_a, summary_a = part_a(survey_manifest["rows"], handle.get_tensor, args.device)
    timings["part_a"] = time.perf_counter() - start
    start = time.perf_counter()
    results_b, summary_b = part_b(features, records, args.device)
    timings["part_b"] = time.perf_counter() - start
    summary = {
        "layers": int(features.shape[1]),
        "part_a": summary_a,
        "part_b": summary_b,
        "gate_G5": {"passed": summary_a["gate_G5a"]["passed"] or summary_b["gate_G5b"]["passed"]},
    }
    return {"part_a": results_a, "part_b": results_b}, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--survey-run", type=Path, required=True, help="Phase Three Stage 0 run (read-only)")
    parser.add_argument("--trace-run", type=Path, required=True, help="Phase Four run (read-only)")
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p5-nonlinear-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
