"""CPU audit of saved Phase Six features; no model forward or reserved split.

Three instruments share one training Gram eigendecomposition: the frozen
centered one-hot ridge, affine one-hot ridge, and affine scalar ridge.
The reused pilot is exploratory; this audit cannot reopen Phase Six gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
METHODS = ("frozen_onehot", "affine_onehot", "affine_scalar")
CATEGORIES = ("literal", "arithmetic", "copy", "carried")
VARIABLES = ("x", "y")
PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ridge_predictions(train, evaluate, targets, alphas=ALPHAS):
    """One eigendecomposition, all named targets and all three readouts.

    The affine categorical prediction adds the training one-hot mean. Its
    slope equals the frozen slope because centered X is orthogonal to the
    constant vector. Scalar targets are explicitly centered, with their
    training mean added before round-to-even and clipping to 0..19.
    """
    x = np.asarray(train, dtype=np.float64)
    e = np.asarray(evaluate, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    x, e = x - mean, e - mean
    gram = x @ x.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.)
    scale = float(np.diag(gram).mean())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.
    matrices, priors, means = [], {}, {}
    for name, values in targets.items():
        values = np.asarray(values, dtype=np.int64)
        onehot = np.eye(20)[values]
        priors[name], means[name] = onehot.mean(0), float(values.mean())
        matrices.append(np.column_stack((onehot, values - means[name])))
    projected = vectors.T @ np.column_stack(matrices)
    evaluation_basis = (e @ x.T) @ vectors
    output = {name: {method: [] for method in METHODS} for name in targets}
    for alpha in alphas:
        scores = (evaluation_basis / (eigenvalues + alpha * scale)) @ projected
        for index, name in enumerate(targets):
            block = scores[:, index * 21:(index + 1) * 21]
            output[name]["frozen_onehot"].append(block[:, :20].argmax(1))
            output[name]["affine_onehot"].append((block[:, :20] + priors[name]).argmax(1))
            output[name]["affine_scalar"].append(np.clip(np.rint(block[:, 20] + means[name]), 0, 19).astype(np.int64))
    return {name: {method: np.stack(predictions) for method, predictions in methods.items()}
            for name, methods in output.items()}


def alpha_index(predictions, truth):
    scores = np.mean(predictions == np.asarray(truth)[None, :], axis=1)
    return max(range(len(scores)), key=lambda index: (scores[index], index))


def accuracy(predicted, truth, mask=None):
    if mask is not None:
        predicted, truth = predicted[mask], truth[mask]
    return float(np.mean(predicted == truth)) if len(truth) else None


def categorized(predicted, truth, categories, computed):
    result = {"overall": accuracy(predicted, truth)}
    for category in CATEGORIES:
        result[category] = accuracy(predicted, truth, categories == category)
        result[category + "_computed"] = accuracy(predicted, truth, (categories == category) & computed)
    return result


def reading_summary(grid):
    result = {}
    for name, category in (("R1", "carried"), ("R2", "arithmetic")):
        layer = max(grid, key=lambda index: (min(grid[index]["asked"][v]["select"][category]
                                                for v in VARIABLES), -index))
        values = {v: grid[layer]["asked"][v]["test"] for v in VARIABLES}
        controls = {v: grid[layer]["asked"][v]["permuted_" + name] for v in VARIABLES}
        meets = all(values[v][category] is not None and values[v][category] >= .80 for v in VARIABLES)
        if name == "R1":
            meets = meets and all(values[v]["carried_computed"] is not None and values[v]["carried_computed"] >= .50 for v in VARIABLES)
        result[name] = {"layer": layer, "test": values, "control_test": controls,
                        "meets_probe_thresholds": meets, "controls_at_most_020": max(controls.values()) <= .20}
    result["both_readings_meet_thresholds_and_controls"] = all(result[name]["meets_probe_thresholds"] and
                                                              result[name]["controls_at_most_020"] for name in ("R1", "R2"))
    return result


def best_final(entries):
    result = {}
    for position in ("program_end", "answer"):
        for condition in ("asked", "not_asked"):
            best = max((row for row in entries if row["position"] == position),
                       key=lambda row: (min(row[condition + "_" + v]["select"] for v in VARIABLES), -row["layer"]))
            result[position + "_" + condition] = {"layer": best["layer"], **{v: best[condition + "_" + v] for v in VARIABLES}}
    return result


def positive_controls():
    fixtures = {
        "scalar_state": (np.tile(np.arange(20), 4)[:, None], np.tile(np.arange(20), 4)),
        "imbalanced_affine_separable": (np.array([0.] * 45 + [.4] * 35 + [.6] * 20)[:, None], np.array([0] * 80 + [1] * 20)),
    }
    result = {}
    for name, (train, labels) in fixtures.items():
        # Reversed independent rows have the same stipulated noiseless support.
        predictions = ridge_predictions(train, train[::-1], {"value": labels})["value"]
        result[name] = {method: [accuracy(row, labels[::-1]) for row in predictions[method]] for method in METHODS}
    return result


def metadata(manifest):
    records, rows = manifest["records"], manifest["rows"]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids")
    group_parts = {}
    for record in records:
        group_parts.setdefault(record["group_id"], set()).add(record["part"])
    if any(len(parts) != 1 for parts in group_parts.values()):
        raise ValueError("a group crosses a split")
    expected = {"train": 192, "select": 64, "test": 128}
    if {part: sum(parts == {part} for parts in group_parts.values()) for part in expected} != expected:
        raise ValueError("not the frozen Phase Six groups")
    last = {record["id"]: index for index, record in enumerate(records)}
    row_last = np.array([last[row["id"]] for row in rows])
    def arrays(items):
        return {"part": np.array([item["part"] for item in items]), "asked": np.array([item["asked"] for item in items]),
                "values": {v: np.array([item[v]["value"] for item in items]) for v in VARIABLES},
                "categories": {v: np.array([item[v]["category"] for item in items]) for v in VARIABLES},
                "computed": {v: np.array([item[v]["computed"] for item in items]) for v in VARIABLES}}
    return arrays(records), arrays([records[index] for index in row_last]), row_last


def comparison(original, replay):
    """Compare every original metric/key against its CPU counterpart."""
    numeric_differences, alpha_changes = [], []
    def walk(left, right, path):
        if isinstance(left, dict):
            for key, value in left.items():
                walk(value, right[key], path + "/" + str(key))
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            delta = float(right) - float(left)
            numeric_differences.append((path, delta))
            if path.endswith("/alpha") and delta:
                alpha_changes.append({"path": path, "original": left, "replay": right})
        elif left != right:
            raise ValueError(f"replay schema/value mismatch: {path}: {left!r} vs {right!r}")
    for old, new in zip(original, replay, strict=True):
        walk(old, new, str(old.get("position", "boundary")) + "/" + str(old["layer"]))
    differences = [{"path": path, "delta": delta} for path, delta in numeric_differences if delta]
    return {"metrics_compared": len(numeric_differences), "changed_metrics": len(differences),
            "max_absolute_difference": max((abs(delta) for _, delta in numeric_differences), default=0.),
            "alpha_changes": alpha_changes, "differences": differences}


def audit_run(run_path, output):
    started = time.perf_counter()
    output.mkdir()
    manifest = json.loads((run_path / "manifest.json").read_text())
    original = json.loads((run_path / "results.json").read_text())
    old_summary = json.loads((run_path / "summary.json").read_text())
    boundary_meta, final_meta, last = metadata(manifest)
    grids = {method: {} for method in METHODS}
    finals = {method: [] for method in METHODS}
    predictions = {}
    rng = np.random.default_rng(301100)
    permutations = {}
    for reading in ("R1", "R2"):
        for variable in VARIABLES:
            train = (boundary_meta["part"] == "train") & (boundary_meta["asked"] == variable)
            permutations[(reading, variable)] = rng.permutation(boundary_meta["values"][variable][train])

    def retain(key, array):
        predictions.setdefault(key, []).append(np.asarray(array, dtype=np.int8))

    def fit_block(block, meta, layer, position=None):
        is_boundary = position is None
        layer_entries = {method: ({"asked": {}, "not_asked": {}} if is_boundary else {"layer": layer, "position": position}) for method in METHODS}
        for prompt_asked in VARIABLES:
            subset = meta["asked"] == prompt_asked
            train, select, test = (subset & (meta["part"] == part) for part in ("train", "select", "test"))
            targets = {v: meta["values"][v][train] for v in VARIABLES}
            if is_boundary:
                targets.update({reading: permutations[(reading, prompt_asked)] for reading in ("R1", "R2")})
            n_select = int(select.sum())
            predicted = ridge_predictions(block[train], np.concatenate((block[select], block[test])), targets)
            for v in VARIABLES:
                condition = "asked" if v == prompt_asked else "not_asked"
                for method in METHODS:
                    all_predictions = predicted[v][method]
                    index = alpha_index(all_predictions[:, :n_select], meta["values"][v][select])
                    p_select, p_test = all_predictions[index, :n_select], all_predictions[index, n_select:]
                    if is_boundary:
                        entry = {"alpha": ALPHAS[index],
                                 "select": categorized(p_select, meta["values"][v][select], meta["categories"][v][select], meta["computed"][v][select]),
                                 "test": categorized(p_test, meta["values"][v][test], meta["categories"][v][test], meta["computed"][v][test])}
                        if condition == "asked":
                            for reading in ("R1", "R2"):
                                entry["permuted_" + reading] = accuracy(predicted[reading][method][index, n_select:], meta["values"][v][test])
                        layer_entries[method][condition][v] = entry
                    else:
                        layer_entries[method][condition + "_" + v] = {"alpha": ALPHAS[index],
                            "select": accuracy(p_select, meta["values"][v][select]),
                            "test": accuracy(p_test, meta["values"][v][test]),
                            "test_computed": accuracy(p_test, meta["values"][v][test], meta["computed"][v][test])}
                    key = f'{position or "boundary"}__{method}__{condition}__{v}'
                    retain(key + "__select", p_select)
                    retain(key + "__test", p_test)
        return layer_entries

    with safe_open(str(run_path / "raw/features.safetensors"), framework="np") as tensors:
        layers = tensors.get_slice("boundaries").get_shape()[1]
        for layer in range(layers):
            block = tensors.get_slice("boundaries")[:, layer, :]
            entries = fit_block(block, boundary_meta, layer)
            for method in METHODS:
                grids[method][layer] = entries[method]
            entries = fit_block(block[last], final_meta, layer, "program_end")
            for method in METHODS:
                finals[method].append(entries[method])
            del block
            entries = fit_block(tensors.get_slice("answer")[:, layer, :], final_meta, layer, "answer")
            for method in METHODS:
                finals[method].append(entries[method])
            if layer == 0 or (layer + 1) % 4 == 0 or layer + 1 == layers:
                elapsed = time.perf_counter() - started
                print(json.dumps({"run": run_path.name, "layer": layer + 1, "layers": layers, "seconds": elapsed,
                                  "estimated_remaining_seconds": elapsed / (layer + 1) * (layers - layer - 1)}), flush=True)

    # Original final rows are grouped by position, then layer.
    for method in METHODS:
        finals[method].sort(key=lambda row: (("program_end", "answer").index(row["position"]), row["layer"]))
    replay_boundaries = [{"layer": layer, **entry} for layer, entry in grids["frozen_onehot"].items()]
    replay = {"boundaries": comparison(original["boundaries"], replay_boundaries),
              "final_state": comparison(original["final_state"], finals["frozen_onehot"])}
    readings = {method: reading_summary(grid) for method, grid in grids.items()}
    final_summary = {method: best_final(entries) for method, entries in finals.items()}
    selected_replay = {name: {"original_layer": old_summary[name]["layer"],
                             "cpu_layer": readings["frozen_onehot"][name]["layer"],
                             "original_controls": old_summary[name]["control_test"],
                             "cpu_controls": readings["frozen_onehot"][name]["control_test"]} for name in ("R1", "R2")}
    selected_replay["final_state"] = {key: {"original_layer": value["layer"], "cpu_layer": final_summary["frozen_onehot"][key]["layer"]}
                                     for key, value in old_summary["final_state"].items()}
    save(output / "results.json", {"boundaries": grids, "final_state": finals})
    np.savez_compressed(output / "predictions.npz", **{key: np.stack(value) for key, value in predictions.items()})
    save(output / "metadata.json", {"source_run": str(run_path), "rows": manifest["rows"], "records": manifest["records"],
                                    "prediction_axis_order": "layer, rows in source manifest order restricted to split and asked condition"})
    summary = {"family": manifest["family"], "source_run": str(run_path), "layers": layers,
               "readings": readings, "final_state": final_summary, "frozen_replay": replay, "selected_replay": selected_replay,
               "original_gate_U6": old_summary["gate_U6"], "phase_six_gates_reopened": False,
               "interpretation": "Exploratory audit on a reused pilot; thresholds describe readout performance, not causal use or a new confirmatory gate.",
               "seconds": time.perf_counter() - started}
    save(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), PROJECT / "tests/test_state_probe_audit.py", PROJECT / "protocols/phase_six_brief.md"]
    paths += [Path(__file__).with_name(name) for name in ("state_question_first.py", "state_trace_survey.py", "state_survey.py")]
    paths += [run / name for run in args.runs for name in ("manifest.json", "results.json", "summary.json", "raw/features.safetensors")]
    inputs = {str(path.resolve()): sha(path) for path in paths}
    protocol = {"format": "open_weight_lingua.saved_readout_audit.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "methods": METHODS, "alpha_grid": ALPHAS, "selection": "alpha: select accuracy, ties larger alpha; layers: select min(x,y), ties earlier",
                "scope": "Boundary asked/not-asked; R1 carried and computed, R2 arithmetic; program-end and answer finals",
                "controls": "Fixed noiseless scalar and imbalanced affine fixtures; selected reading permutation labels seed301100",
                "data": "Same saved train/select/pilot group masks. Reused pilot is exploratory. No new validation, model forward, GPU or deduplication.",
                "inference": "Descriptive accuracies and frozen operational thresholds only; no new significance or causality claims",
                "cpu_threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "files_sha256": inputs}
    save(args.output / "protocol.lock.json", protocol)
    controls = positive_controls()
    save(args.output / "positive_controls.json", controls)
    summaries = [audit_run(run, args.output / json.loads((run / "manifest.json").read_text())["family"]) for run in args.runs]
    after = {path: sha(path) for path in inputs}
    if after != inputs:
        raise RuntimeError("an audited source or input changed during execution")
    save(args.output / "summary.json", {"positive_controls": controls, "runs": summaries, "input_hashes_verified_after": True,
                                      "seconds": time.perf_counter() - started, "new_model_forwards": 0, "gpu_used": False})
    print(json.dumps({"complete": str(args.output), "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
