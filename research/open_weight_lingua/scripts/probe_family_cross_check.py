"""Post-hoc comparison of probe families on saved activations (Phase Six review).

Usage:
    python scripts/probe_family_cross_check.py --family gemma3-12b --runs RUNS_DIR --out OUT.json

Exploratory; it can never change a frozen gate. Example: after `y = y + 3`
Phase Four's probe read y's new value about 20% of the time. That probe is a
ridge regression onto one-hot labels with centred inputs and no intercept, so
a weak code could be missed for reasons of the probe rather than the model.
This script refits the same readouts with three probe families:

- ridge_onehot: the existing probe (state_trace_survey.fit_predict);
- affine_logistic: multinomial logistic regression with an intercept, on
  standardized inputs, L2-penalized (fit exactly in the span of the training
  rows, where the L2 optimum lies);
- scalar_ridge: ridge regression with an intercept onto the integer value,
  rounded and clipped to 0..19.

Each family's setting (penalty, then layer) is chosen on the calibration
split: fit on train groups, select on select groups. The pilot split is
reported as held out, but it was reused across Phases Three to Six, so those
numbers are exploratory, not confirmatory. Validation splits are never read.

Datasets, all read-only:
- after_boundaries: Phase Five's re-capture at Phase Four's statement
  boundaries (question after the program);
- first_boundaries: Phase Six's boundaries (question first), asked and not
  asked;
- after_final / first_final: final values at the program's last token and the
  answer position (Phase Three; Phase Six), asked and not asked.

Positive controls, where the value is explicitly present:
- literal updates (the boundary token is the value itself);
- the asked final value at the answer position;
- a planted code: the true running value added to real activations as a
  one-hot or a scalar direction, at graded strengths.

It also splits held-out boundary accuracy by whether the boundary's text
prefix occurs among the training boundaries.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
import torch

from open_weight_lingua.description_census import program_text
from open_weight_lingua.state_survey import ALPHA_GRID, TRAIN_GROUPS, VALUES
from open_weight_lingua.state_trace_survey import CATEGORIES, category_accuracy, fit_predict
from open_weight_lingua.stats import group_bootstrap

M2_RUNS = "/home/seanjazm27/projects/universa-recurrent-m2/research/open_weight_lingua/runs"
INPUTS = {
    "gemma3-12b": {
        "phase_three": "p3-stage0-gemma3-12b-20261004T062811Z-745ea17c",
        "phase_four": "p4-trace-gemma3-12b-20261004T161553Z-517f9189",
        "phase_five": "p5-nonlinear-gemma3-12b-20261004T211741Z-421ba7af",
        "phase_six": "p6-question-first-gemma3-12b-20261004T231457Z-96041d06",
        "calibration": "calibration-20260923T042912Z-d0e9499f",
        "pilot": "pilot-20260923T043613Z-f7e71d7b",
    },
    "qwen2.5-7b": {
        "phase_three": "p3-stage0-qwen2.5-7b-20261004T074740Z-840871a1",
        "phase_four": "p4-trace-qwen2.5-7b-20261004T164315Z-27667f5a",
        "phase_five": "p5-nonlinear-qwen2.5-7b-20261004T214157Z-354536e6",
        "phase_six": "p6-question-first-qwen2.5-7b-20261004T235340Z-4f429ce9",
        "calibration": f"{M2_RUNS}/calibration-20260921T235017Z-fd111b21",
        "pilot": f"{M2_RUNS}/pilot-20260921T235825Z-6164d210",
    },
}
LOGISTIC_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # L2 weight on standardized inputs; mean cross-entropy loss
PLANT_STRENGTHS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)  # planted norm / mean centred-row norm
PLANT_SEED = 611100
BOOTSTRAP_SEED = 601100
FAMILY_NAMES = ("ridge_onehot", "affine_logistic", "scalar_ridge")


# ---------------------------------------------------------------- probe families (each returns predictions per setting)


def ridge_onehot(train_x, train_y, eval_x):
    return [np.asarray(p) for p in fit_predict(train_x, train_y, eval_x, ALPHA_GRID, "cpu")]


def scalar_ridge(train_x, train_y, eval_x):
    x = torch.as_tensor(train_x, dtype=torch.float64)
    e = torch.as_tensor(eval_x, dtype=torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    x, e = x - mean, e - mean
    gram = x @ x.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    scale = float(torch.diagonal(gram).mean())
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    target = torch.as_tensor(np.asarray(train_y), dtype=torch.float64)
    offset = float(target.mean())
    projected = eigenvectors.T @ (target - offset)
    kernel = e @ x.T
    out = []
    for alpha in ALPHA_GRID:
        coefficients = eigenvectors @ (projected / (eigenvalues + alpha * scale))
        out.append(np.clip(np.rint((kernel @ coefficients + offset).numpy()), 0, VALUES - 1).astype(int))
    return out


def affine_logistic(train_x, train_y, eval_x):
    """L2 multinomial logistic regression with an unpenalized intercept.

    The L2 optimum lies in the span of the standardized training rows, so it
    is fit exactly in that span's orthonormal coordinates F = U S^½ (n × r),
    with eval rows projected by P = Zᵀ U S^-½. Warm-started from strong to
    weak penalty.
    """
    x = torch.as_tensor(train_x, dtype=torch.float64)
    e = torch.as_tensor(eval_x, dtype=torch.float64)
    mean, std = x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True).clamp_min(1e-6)
    z, ze = (x - mean) / std, (e - mean) / std
    eigenvalues, eigenvectors = torch.linalg.eigh(z @ z.T)
    keep = eigenvalues > eigenvalues.max() * 1e-10
    eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[:, keep]
    features = eigenvectors * eigenvalues.sqrt()
    eval_features = ze @ (z.T @ (eigenvectors / eigenvalues.sqrt()))
    labels = torch.as_tensor(np.asarray(train_y), dtype=torch.long)
    weights = torch.zeros(features.shape[1], VALUES, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(VALUES, dtype=torch.float64, requires_grad=True)
    by_penalty = {}
    for penalty in sorted(LOGISTIC_GRID, reverse=True):
        optimizer = torch.optim.LBFGS([weights, bias], lr=1.0, max_iter=500, tolerance_grad=1e-7,
                                      tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(features @ weights + bias, labels) + 0.5 * penalty * (weights * weights).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            by_penalty[penalty] = (eval_features @ weights + bias).argmax(dim=1).numpy()
    return [by_penalty[penalty] for penalty in LOGISTIC_GRID]


PROBES = {"ridge_onehot": (ridge_onehot, ALPHA_GRID), "affine_logistic": (affine_logistic, LOGISTIC_GRID),
          "scalar_ridge": (scalar_ridge, ALPHA_GRID)}


def fit(family, block, labels, part, rows):
    """Fit on train rows, choose the setting on select accuracy (ties: stronger regularization), predict select and test."""
    train, select, test = rows & (part == "train"), rows & (part == "select"), rows & (part == "test")
    function, grid = PROBES[family]
    predictions = function(block[train], labels[train], np.concatenate([block[select], block[test]]))
    n_select = int(select.sum())
    scores = [float(np.mean(p[:n_select] == labels[select])) for p in predictions]
    best = max(range(len(grid)), key=lambda i: (scores[i], grid[i] if family == "affine_logistic" else i))
    return grid[best], predictions[best][:n_select], predictions[best][n_select:]


def _categories(predicted, truth, categories, computed):
    return {
        "overall": float(np.mean(predicted == truth)),
        **{c: category_accuracy(predicted, truth, categories, computed, c) for c in CATEGORIES},
        **{f"{c}_computed": category_accuracy(predicted, truth, categories, computed, c, True) for c in CATEGORIES},
    }


# ---------------------------------------------------------------- data


def _program_rows(manifest, split):
    seen, out = set(), []
    for row in manifest["inputs"]:
        key = f"{row['group_id']}-{row['side']}"
        if key in seen:
            continue
        seen.add(key)
        index = int(row["group_id"].rsplit("-", 1)[1])
        part = ("train" if index < TRAIN_GROUPS else "select") if split == "calibration" else "test"
        out.append({"program_key": key, "group_id": row["group_id"], "side": row["side"], "part": part,
                    "lines": program_text(row["prompt"]).split("\n")})
    return out


def after_boundaries(paths):
    """Phase Five features with Phase Four records in Phase Five's capture order (program order)."""
    calibration = json.loads((paths["calibration"] / "manifest.json").read_text())
    pilot = json.loads((paths["pilot"] / "manifest.json").read_text())
    programs = _program_rows(calibration, "calibration") + _program_rows(pilot, "pilot")
    by_program = {}
    for record in json.loads((paths["phase_four"] / "manifest.json").read_text())["records"]:
        by_program.setdefault(record["program_key"], []).append(record)
    records = []
    for program in programs:
        for record in by_program.pop(program["program_key"]):
            if record["part"] != program["part"]:
                raise ValueError("split mismatch")
            records.append({**record, "group_id": program["group_id"], "side": program["side"], "asked": None,
                            "prefix": "\n".join(program["lines"][: record["line"]])})
    if by_program:
        raise ValueError("Phase Four records without programs")
    with safe_open(str(paths["phase_five"] / "raw" / "boundary_features.safetensors"), framework="numpy") as handle:
        features = handle.get_tensor("boundaries")
    if len(features) != len(records):
        raise ValueError("features and records disagree")
    return features, records


def first_boundaries(paths):
    manifest = json.loads((paths["phase_six"] / "manifest.json").read_text())
    prompts = {row["id"]: row["prompt"] for row in manifest["rows"]}
    records = []
    for record in manifest["records"]:
        question, body = prompts[record["id"]].split("\n", 1)
        records.append({**record, "prefix": question + "\n" + "\n".join(body.split("\n")[: record["line"]])})
    with safe_open(str(paths["phase_six"] / "raw" / "features.safetensors"), framework="numpy") as handle:
        features, answers = handle.get_tensor("boundaries"), handle.get_tensor("answer")
    return features, answers, records, manifest["rows"]


# ---------------------------------------------------------------- analyses


def boundary_survey(features, records, conditions, log):
    """grid[family][condition][layer][v] and held-out predictions, Phase Four's per-layer probe for each family."""
    part = np.array([r["part"] for r in records])
    asked = np.array([r["asked"] or "" for r in records])
    values = {v: np.array([r[v]["value"] for r in records]) for v in ("x", "y")}
    categories = {v: np.array([r[v]["category"] for r in records]) for v in ("x", "y")}
    computed = {v: np.array([r[v]["computed"] for r in records]) for v in ("x", "y")}
    grid = {f: {c: {} for c in conditions} for f in FAMILY_NAMES}
    held_out = {}
    for layer in range(features.shape[1]):
        block = features[:, layer]
        for condition in conditions:
            for v in ("x", "y"):
                rows = {"all": np.ones(len(records), bool), "asked": asked == v, "not_asked": (asked != v) & (asked != "")}[condition]
                select, test = rows & (part == "select"), rows & (part == "test")
                for family in FAMILY_NAMES:
                    setting, select_pred, test_pred = fit(family, block, values[v], part, rows)
                    grid[family][condition].setdefault(layer, {})[v] = {
                        "setting": setting,
                        "select": _categories(select_pred, values[v][select], categories[v][select], computed[v][select]),
                        "test": _categories(test_pred, values[v][test], categories[v][test], computed[v][test]),
                    }
                    held_out[(family, condition, layer, v)] = test_pred
        log(f"    layer {layer} done")
    return grid, held_out


def layer_for(layers: dict, category: str) -> int:
    """Phase Four's rule: argmax over layers of min over x, y of select accuracy; ties to the earlier layer."""
    def key(layer):
        scores = [layers[layer][v]["select"][category] for v in ("x", "y")]
        return (min(s if s is not None else -1.0 for s in scores), -layer)
    return max(layers, key=key)


def readings(grid):
    out = {}
    for family, by_condition in grid.items():
        for condition, layers in by_condition.items():
            entry = {}
            for name, category, keys in (("carried", "carried", ("carried", "carried_computed")),
                                         ("arithmetic", "arithmetic", ("arithmetic", "arithmetic_computed")),
                                         ("literal_control", "literal", ("literal",))):
                layer = layer_for(layers, category)
                entry[name] = {"layer": layer, **{f"{split}_{k}": [layers[layer][v][split][k] for v in ("x", "y")]
                                                  for split in ("select", "test") for k in keys},
                               "settings": [layers[layer][v]["setting"] for v in ("x", "y")]}
            out[f"{family}/{condition}"] = entry
    return out


def best_any_layer(grid):
    out = {}
    for family, by_condition in grid.items():
        for condition, layers in by_condition.items():
            out[f"{family}/{condition}"] = {
                c: max(min((layers[layer][v]["test"][c] if layers[layer][v]["test"][c] is not None else -1.0) for v in ("x", "y"))
                       for layer in layers)
                for c in ("arithmetic", "carried", "carried_computed", "literal")
            }
    return out


def prefix_novelty(records, held_out, read):
    """Held-out accuracy at each family's chosen layers, split by whether the text prefix occurs among training boundaries."""
    seen = {r["prefix"] for r in records if r["part"] == "train"}
    out = {}
    for key, entry in read.items():
        family, condition = key.split("/")
        result = {}
        for name, category in (("arithmetic", "arithmetic"), ("carried", "carried")):
            layer = entry[name]["layer"]
            hits = {"seen": [], "novel": []}
            for v in ("x", "y"):
                subset = [r for r in records if r["part"] == "test" and
                          {"all": True, "asked": r["asked"] == v, "not_asked": r["asked"] not in (v, None)}[condition]]
                for record, guess in zip(subset, held_out[(family, condition, layer, v)], strict=True):
                    if record[v]["category"] == category:
                        hits["seen" if record["prefix"] in seen else "novel"].append(int(guess == record[v]["value"]))
            result[name] = {k: {"n": len(h), "accuracy": float(np.mean(h)) if h else None} for k, h in hits.items()}
        out[key] = result
    test = [r for r in records if r["part"] == "test"]
    out["held_out_boundaries_with_seen_prefix"] = {"seen": sum(r["prefix"] in seen for r in test), "total": len(test)}
    return out


def question_effect(records, held_out, family, layer, category):
    """Asked minus not-asked correctness on held-out boundaries, paired by group, side, line and variable."""
    test = [r for r in records if r["part"] == "test"]
    correct = {}
    for condition in ("asked", "not_asked"):
        for v in ("x", "y"):
            subset = [r for r in test if (r["asked"] == v) == (condition == "asked")]
            for record, guess in zip(subset, held_out[(family, condition, layer, v)], strict=True):
                correct[(condition, v, record["group_id"], record["side"], record["line"])] = int(guess == record[v]["value"])
    per_group = {}
    for record in test:
        v = record["asked"]
        if record[v]["category"] != category:
            continue
        key = (v, record["group_id"], record["side"], record["line"])
        total, count = per_group.get(record["group_id"], (0, 0))
        per_group[record["group_id"]] = (total + correct[("asked", *key)] - correct[("not_asked", *key)], count + 1)

    def pooled(groups):
        return sum(t for t, _ in groups) / sum(n for _, n in groups)

    interval = group_bootstrap(list(per_group.values()), pooled, seed=BOOTSTRAP_SEED)
    return {"layer": layer, "difference": interval.point, "ci95": [interval.lower, interval.upper], "groups": len(per_group)}


def final_values(features_by_position, rows, finals, computed, log):
    """Per position, family, condition and layer; summary at the layer maximizing min-over-variables select accuracy."""
    part = np.array([r["part"] for r in rows])
    asked = np.array([r["asked"] for r in rows])
    out = {}
    for position, features in features_by_position.items():
        for family in FAMILY_NAMES:
            for condition in ("asked", "not_asked"):
                layers = {}
                for layer in range(features.shape[1]):
                    layers[layer] = {}
                    for v in ("x", "y"):
                        subset = (asked == v) if condition == "asked" else (asked != v)
                        setting, select_pred, test_pred = fit(family, features[:, layer], finals[v], part, subset)
                        select, test = subset & (part == "select"), subset & (part == "test")
                        mask = computed[v][test]
                        layers[layer][v] = {"setting": setting, "select": float(np.mean(select_pred == finals[v][select])),
                                            "test": float(np.mean(test_pred == finals[v][test])),
                                            "test_computed": float(np.mean(test_pred[mask] == finals[v][test][mask])) if mask.any() else None}
                best = max(layers, key=lambda layer: (min(layers[layer][v]["select"] for v in ("x", "y")), -layer))
                out[f"{position}/{family}/{condition}"] = {"layer": best, **{v: layers[best][v] for v in ("x", "y")}}
            log(f"    final {position} {family} done")
    return out


def planted(features, records, layer, log):
    """Add the true running value as an explicit code at graded strengths; each family refit with its own selection."""
    part = np.array([r["part"] for r in records])
    rng = np.random.default_rng(PLANT_SEED)
    block = features[:, layer].astype(np.float64)
    centred = block - block[part == "train"].mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centred[part == "train"], axis=1).mean())
    basis, _ = np.linalg.qr(rng.normal(size=(block.shape[1], VALUES + 1)))
    onehot_code, scalar_code = basis[:, :VALUES].T, basis[:, VALUES]
    out = {}
    for v in ("x", "y"):
        values = np.array([r[v]["value"] for r in records])
        arithmetic = np.array([r[v]["category"] == "arithmetic" for r in records])
        test = part == "test"
        for code_name in ("onehot", "scalar"):
            signal = onehot_code[values] if code_name == "onehot" else np.outer((values - 9.5) / 5.77, scalar_code)
            signal = signal / np.linalg.norm(signal, axis=1, keepdims=True).mean()
            for strength in PLANT_STRENGTHS:
                planted_block = block + strength * scale * signal
                for family in FAMILY_NAMES:
                    _, _, test_pred = fit(family, planted_block, values, part, np.ones(len(records), bool))
                    out.setdefault(f"{code_name}/{family}/{v}", {})[str(strength)] = {
                        "test_overall": float(np.mean(test_pred == values[test])),
                        "test_arithmetic": float(np.mean(test_pred[arithmetic[test]] == values[test][arithmetic[test]])),
                    }
            log(f"    planted {v} {code_name} done")
    return {"layer": layer, "mean_centred_norm": scale, "results": out}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=sorted(INPUTS), required=True)
    parser.add_argument("--runs", type=Path, required=True, help="directory holding the input run directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()

    def log(message):
        print(f"[{time.perf_counter() - started:7.0f}s] {message}", flush=True)

    paths = {k: (Path(v) if v.startswith("/") else args.runs / v) for k, v in INPUTS[args.family].items()}
    result = {"family": args.family, "inputs": {k: p.name for k, p in paths.items()}, "logistic_grid": LOGISTIC_GRID,
              "alpha_grid": ALPHA_GRID, "plant_strengths": PLANT_STRENGTHS,
              "note": "Settings chosen on calibration select; pilot numbers are exploratory (reused split)."}

    log("after_boundaries")
    features, records = after_boundaries(paths)
    grid, held_out = boundary_survey(features, records, ("all",), log)
    read = readings(grid)
    result["after_boundaries"] = {"readings": read, "best_any_layer": best_any_layer(grid),
                                  "prefix_novelty": prefix_novelty(records, held_out, read)}
    plant_layer = features.shape[1] // 2
    result["planted"] = planted(features, records, plant_layer, log)
    del features

    log("first_boundaries")
    features, answers, records, rows = first_boundaries(paths)
    grid, held_out = boundary_survey(features, records, ("asked", "not_asked"), log)
    read = readings(grid)
    result["first_boundaries"] = {
        "readings": read, "best_any_layer": best_any_layer(grid), "prefix_novelty": prefix_novelty(records, held_out, read),
        "question_effect": {family: {category: question_effect(records, held_out, family, read[f"{family}/asked"][name]["layer"], category)
                                     for name, category in (("arithmetic", "arithmetic"), ("carried", "carried"))}
                            for family in FAMILY_NAMES},
    }

    log("first_final")
    last = {}
    for index, record in enumerate(records):
        last[record["id"]] = index
    final_rows = [{"id": row["id"], "part": row["part"], "asked": row["id"].rsplit("-", 1)[1]} for row in rows]
    finals = {v: np.array([records[last[r["id"]]][v]["value"] for r in final_rows]) for v in ("x", "y")}
    computed = {v: np.array([records[last[r["id"]]][v]["computed"] for r in final_rows]) for v in ("x", "y")}
    end = features[[last[r["id"]] for r in final_rows]]
    del features
    result["first_final"] = final_values({"program_end": end, "answer": answers}, final_rows, finals, computed, log)
    del end, answers

    log("after_final")
    manifest = json.loads((paths["phase_three"] / "manifest.json").read_text())
    final_rows = [{"id": r["id"], "part": r["part"], "asked": r["id"].rsplit("-", 1)[1]} for r in manifest["rows"]]
    finals = {v: np.array([r["labels"][v]["value"] for r in manifest["rows"]]) for v in ("x", "y")}
    computed = {v: np.array([r["labels"][v]["computed"] for r in manifest["rows"]]) for v in ("x", "y")}
    with safe_open(str(paths["phase_three"] / "raw" / "features.safetensors"), framework="numpy") as handle:
        by_position = {p: handle.get_tensor(p) for p in ("program_end", "answer")}
    result["after_final"] = final_values(by_position, final_rows, finals, computed, log)
    result["wall_seconds"] = time.perf_counter() - started
    args.out.write_text(json.dumps(result, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
