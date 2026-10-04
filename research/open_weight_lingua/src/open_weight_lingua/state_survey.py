"""Phase Three Stage 0: where does the model hold both variables' values?

protocols/phase_three_brief.md is the frozen design. Example: for

    x = 9 / y = 8 / y = 3 / y = y + 3 / x = x + 1 / What is x?

x is 10 and y is 6 when the question comes. For each of five token positions
and every transformer block, this survey fits a linear probe that reads x's
value and one that reads y's value from the block output, and reports how
often they are right on held-out programs. It runs the target model only.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from safetensors.numpy import save_file
import torch

from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import literals, program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .target import pad_to_bucket
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_three_brief.md"
FORMAT = "open_weight_lingua.state_survey.v1"
POSITIONS = ("program_end", "query_variable", "question_mark", "user_end", "answer")
VALUES = 20
TRAIN_GROUPS = 192  # calibration groups 0..191 train, 192..255 select; the pilot split is held out
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # times the mean diagonal of the training Gram matrix
PERMUTATION_SEED = 301100
G0_MIN_ACCURACY, G0_MIN_COMPUTED = 0.80, 0.50


# ---------------------------------------------------------------- positions and labels


def _first_containing(tokenizer, ids: list[int], text: str) -> int:
    for position in range(len(ids)):
        if text in tokenizer.decode(ids[: position + 1], skip_special_tokens=False):
            return position
    raise ValueError(f"text not found in decoded prompt: {text[-30:]!r}")


def positions(tokenizer, row: dict) -> dict:
    """The five frozen positions of one prompt, on its own tokens."""
    ids, prompt = row["input_ids"], row["prompt"]
    body = program_text(prompt)
    variable = row["variable"]
    found = {
        "program_end": _first_containing(tokenizer, ids, body),
        "query_variable": _first_containing(tokenizer, ids, f"{body}\nWhat is {variable}"),
        "question_mark": _first_containing(tokenizer, ids, f"{body}\nWhat is {variable}?"),
        "user_end": _first_containing(tokenizer, ids, prompt),
        "answer": int(row["position"]),
    }
    order = [found[name] for name in POSITIONS]
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError(f"positions out of order for {row['id']}: {found}")
    return found


def labels(row: dict) -> dict:
    """Final values, the computed flags and the last-literal copy baseline."""
    statements = program(row["prompt"])
    state = interpret(statements)
    written = literals(row["prompt"])
    out = {}
    for variable in ("x", "y"):
        assigned = [s.operand for s in statements if s.variable == variable and s.operation == "assign"]
        out[variable] = {
            "value": state[variable],
            "computed": state[variable] not in written,
            "copy_guess": assigned[-1],
        }
    return out


# ---------------------------------------------------------------- probes


def _fit_predict(gram_train, gram_eval, targets, alphas):
    """Dual ridge: predictions on eval rows for each alpha, via one eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(gram_train)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # a Gram matrix is PSD; clip roundoff
    projected = eigenvectors.T @ targets
    scale = float(np.mean(np.diag(gram_train)))
    if not np.isfinite(scale) or scale <= 0.0:  # constant features: no signal, but no NaNs either
        scale = 1.0
    out = []
    for alpha in alphas:
        coefficients = eigenvectors @ (projected / (eigenvalues + alpha * scale)[:, None])
        out.append(gram_eval @ coefficients)
    return out


def probe_accuracies(train, select, test, y_train, y_select, y_test, computed_select, computed_test, permuted=None):
    """Primary (one-hot) and scalar probes; alpha chosen on the select split.

    Ties in select accuracy go to the larger alpha. Returns select and test
    accuracies, the computed-subset test accuracy and the chosen alphas.
    """
    mean = train.mean(axis=0, keepdims=True)
    train, select, test = train - mean, select - mean, test - mean
    gram = train @ train.T
    k_select, k_test = select @ train.T, test @ train.T
    result = {}
    for name, labels_train in (("onehot", y_train), ("permuted", permuted)):
        if labels_train is None:
            continue
        onehot = np.eye(VALUES)[labels_train]
        stacked = np.vstack([k_select, k_test])
        predictions = _fit_predict(gram, stacked, onehot, ALPHA_GRID)
        scores = [float(np.mean(p[: len(select)].argmax(1) == y_select)) for p in predictions]
        best = max(range(len(ALPHA_GRID)), key=lambda i: (scores[i], i))
        chosen = predictions[best][len(select):].argmax(1)
        chosen_select = predictions[best][: len(select)].argmax(1)
        result[name] = {
            "alpha": ALPHA_GRID[best],
            "select": scores[best],
            "select_computed": float(np.mean(chosen_select[computed_select] == y_select[computed_select])) if computed_select.any() else None,
            "test": float(np.mean(chosen == y_test)),
            "test_computed": float(np.mean(chosen[computed_test] == y_test[computed_test])) if computed_test.any() else None,
        }
    scalar_targets = (y_train - y_train.mean())[:, None].astype(np.float64)
    stacked = np.vstack([k_select, k_test])
    predictions = _fit_predict(gram, stacked, scalar_targets, ALPHA_GRID)
    rounded = [np.clip(np.rint(p[:, 0] + y_train.mean()), 0, VALUES - 1).astype(int) for p in predictions]
    scores = [float(np.mean(r[: len(select)] == y_select)) for r in rounded]
    best = max(range(len(ALPHA_GRID)), key=lambda i: (scores[i], i))
    result["scalar"] = {
        "alpha": ALPHA_GRID[best],
        "select": scores[best],
        "test": float(np.mean(rounded[best][len(select):] == y_test)),
    }
    return result


def select_site(grid: dict) -> tuple[str, int]:
    """argmax of min(acc_x, acc_y) on select; ties: select-split computed min, earlier layer, earlier position."""
    def key(item):
        (position, layer), entry = item
        select_min = min(entry["x"]["onehot"]["select"], entry["y"]["onehot"]["select"])
        computed = [entry[v]["onehot"]["select_computed"] for v in ("x", "y")]
        computed_min = min(c for c in computed if c is not None) if any(c is not None for c in computed) else 0.0
        return (select_min, computed_min, -layer, -POSITIONS.index(position))
    (position, layer), _ = max(grid.items(), key=key)
    return position, layer


def gate_g0(entry: dict) -> dict:
    accuracy = min(entry["x"]["onehot"]["test"], entry["y"]["onehot"]["test"])
    computed = [entry[v]["onehot"]["test_computed"] for v in ("x", "y")]
    computed_min = min(c for c in computed if c is not None)
    return {
        "held_out_min_accuracy": accuracy,
        "held_out_min_computed_accuracy": computed_min,
        "passed": accuracy >= G0_MIN_ACCURACY and computed_min >= G0_MIN_COMPUTED,
    }


# ---------------------------------------------------------------- execution


def _rows(manifest_path: Path, split_name: str):
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for row in manifest["inputs"]:
        index = int(row["group_id"].rsplit("-", 1)[1])
        if split_name == "calibration":
            part = "train" if index < TRAIN_GROUPS else "select"
        else:
            part = "test"
        rows.append({**row, "part": part})
    return manifest, rows


@torch.inference_mode()
def capture(model, rows: list[dict], where: dict, layers: int, device: str) -> dict:
    """Block outputs (hidden_states[1..L]) at every position, as float16 [N, L, d]."""
    features = {name: [] for name in POSITIONS}
    for index, row in enumerate(rows):
        ids = torch.tensor([row["input_ids"]], device=device)
        mask = torch.tensor([row["attention_mask"]], device=device)
        ids, mask = pad_to_bucket(ids, mask)
        hidden = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True).hidden_states
        if len(hidden) != layers + 1:
            raise RuntimeError("unexpected hidden-state count")
        stacked = torch.stack([h[0] for h in hidden[1:]])  # [L, T, d]
        for name in POSITIONS:
            features[name].append(stacked[:, where[row["id"]][name]].float().cpu().numpy().astype(np.float16))
        if index % 128 == 0:
            print(f"  capture {index}/{len(rows)}", flush=True)
    return {name: np.stack(values) for name, values in features.items()}


def survey(features: dict, rows: list[dict], label: dict) -> dict:
    parts = np.array([r["part"] for r in rows])
    rng = np.random.default_rng(PERMUTATION_SEED)
    grid = {}
    for name in POSITIONS:
        data = features[name].astype(np.float32)
        for layer in range(data.shape[1]):
            block = data[:, layer]
            entry = {}
            for variable in ("x", "y"):
                values = np.array([label[r["id"]][variable]["value"] for r in rows])
                computed = np.array([label[r["id"]][variable]["computed"] for r in rows])
                train, select, test = (parts == "train"), (parts == "select"), (parts == "test")
                entry[variable] = probe_accuracies(
                    block[train].astype(np.float64), block[select].astype(np.float64), block[test].astype(np.float64),
                    values[train], values[select], values[test], computed[select], computed[test],
                    permuted=rng.permutation(values[train]),
                )
            grid[(name, layer)] = entry
        print(f"  probes done: {name}", flush=True)
    return grid


def baselines(rows: list[dict], label: dict) -> dict:
    test = [r for r in rows if r["part"] == "test"]
    out = {}
    for variable in ("x", "y"):
        values = [label[r["id"]][variable]["value"] for r in test]
        out[variable] = {
            "copy_last_literal": float(np.mean([label[r["id"]][variable]["copy_guess"] == v for r, v in zip(test, values)])),
            "majority_value": float(max(np.bincount(values, minlength=VALUES)) / len(values)),
            "computed_fraction": float(np.mean([label[r["id"]][variable]["computed"] for r in test])),
        }
    return out


def run(args, run_dir: RunDirectory, timings: dict):
    calibration_manifest, calibration_rows = _rows(args.calibration_run / "manifest.json", "calibration")
    pilot_manifest, pilot_rows = _rows(args.pilot_run / "manifest.json", "pilot")
    for manifest in (calibration_manifest, pilot_manifest):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    rows = calibration_rows + pilot_rows
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"])
    where = {r["id"]: positions(tokenizer, r) for r in rows}
    label = {r["id"]: labels(r) for r in rows}
    write_json(
        run_dir.path / "manifest.json",
        {
            "format": FORMAT,
            "family": args.lock.stem.replace("model-lock-", "") if args.lock.stem != "model-lock" else "qwen2.5-7b",
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
            "positions": POSITIONS,
            "alpha_grid": ALPHA_GRID,
            "train_groups": TRAIN_GROUPS,
            "permutation_seed": PERMUTATION_SEED,
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
            "rows": [{"id": r["id"], "part": r["part"], "positions": where[r["id"]], "labels": label[r["id"]]} for r in rows],
        },
    )
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    features = capture(model, rows, where, config.num_hidden_layers, args.device)
    del model
    release_models()
    timings["capture"] = time.perf_counter() - start
    save_file({name: array for name, array in features.items()}, str(run_dir.path / "raw" / "features.safetensors"))
    start = time.perf_counter()
    grid = survey(features, rows, label)
    timings["probes"] = time.perf_counter() - start
    position, layer = select_site(grid)
    summary = {
        "layers": int(features[POSITIONS[0]].shape[1]),
        "rows": {part: sum(r["part"] == part for r in rows) for part in ("train", "select", "test")},
        "baselines": baselines(rows, label),
        "selected_site": {"position": position, "layer": layer, "probes": grid[(position, layer)]},
        "gate_G0": gate_g0(grid[(position, layer)]),
        "best_per_position": {
            name: max(
                ((layer_index, grid[(name, layer_index)]) for layer_index in range(int(features[name].shape[1]))),
                key=lambda item: min(item[1]["x"]["onehot"]["select"], item[1]["y"]["onehot"]["select"]),
            )[0]
            for name in POSITIONS
        },
    }
    results = [{"position": p, "layer": layer_index, **entry} for (p, layer_index), entry in sorted(grid.items(), key=lambda kv: (POSITIONS.index(kv[0][0]), kv[0][1]))]
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p3-stage0-{family}-{stamp}-{uuid.uuid4().hex[:8]}")
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
