"""Phase Six: does asking the question first make the model track the asked variable?

protocols/phase_six_brief.md is the frozen design. Example:

    What is y at the end of this program? Reply with only the integer.
    x = 9 / y = 8 / y = 3 / y = y + 3 / x = x + 1

Phases Three to Five asked the question after the program, so while reading
the model could not know which variable mattered, and no probe found a
running state. Here the same programs are asked question-first. If the model
now tracks the asked variable, the 6 after `y = y + 3` should be readable at
that line's last token. The variable not asked, on the same programs, is the
control. It runs the target model only.
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
from .description_census import program_text
from .edit_diagnostics import PROJECT, _git_head
from .metrics import answer_text_matches
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .state_survey import ALPHA_GRID, PERMUTATION_SEED, TRAIN_GROUPS, stored
from .state_trace_survey import CATEGORIES, baselines, category_accuracy, fit_predict, readings, trace_labels
from .stats import group_bootstrap
from .target import TARGET_BUCKET, Site, Target, pad_to_bucket

PROTOCOL = PROJECT / "protocols" / "phase_six_brief.md"
FORMAT = "open_weight_lingua.state_question_first.v1"
QUESTION_TEMPLATE = "What is {variable} at the end of this program? Reply with only the integer.\n{program}"
CONVENTIONS = {"gemma3-12b": "rstrip", "qwen2.5-7b": "raw"}  # each family's frozen answer convention
U6_MIN, CONTROL_MAX = 0.80, 0.20
BOOTSTRAP_SEED = 601100
MAX_NEW_TOKENS = 8  # Target.greedy's default, as in the pilots
CONDITIONS = ("asked", "not_asked")


# ---------------------------------------------------------------- prompts, positions, labels


def prompt_rows(manifest: dict, split_name: str) -> list[dict]:
    """One question-first row per Phase Two prompt; the program and answer are unchanged."""
    out = []
    for row in manifest["inputs"]:
        index = int(row["group_id"].rsplit("-", 1)[1])
        part = ("train" if index < TRAIN_GROUPS else "select") if split_name == "calibration" else "test"
        out.append({
            "id": row["id"], "group_id": row["group_id"], "side": row["side"], "variable": row["variable"],
            "part": part, "answer": row["answer"],
            "original_prompt": row["prompt"], "original_input_ids": row["input_ids"],
            "prompt": QUESTION_TEMPLATE.format(variable=row["variable"], program=program_text(row["prompt"])),
        })
    return out


def question_first_boundaries(tokenizer, ids: list[int], prompt: str) -> list[int]:
    """Token completing statement i, for i = 2..n, searched after the question line."""
    question, body = prompt.split("\n", 1)
    lines = body.split("\n")
    out, start = [], 0
    for i in range(2, len(lines) + 1):
        target = question + "\n" + "\n".join(lines[:i])
        for position in range(start, len(ids)):
            if target in tokenizer.decode(ids[: position + 1], skip_special_tokens=False):
                out.append(position)
                start = position + 1
                break
        else:
            raise ValueError(f"statement {i} not found in the decoded prompt")
    return out


def tokenize(tokenizer, rows: list[dict]) -> None:
    for row in rows:
        ids = list(tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True))
        if len(ids) + MAX_NEW_TOKENS > TARGET_BUCKET:
            raise ValueError(f"{row['id']}: question-first prompt does not fit the pinned bucket")
        row.update(input_ids=ids, attention_mask=[1] * len(ids), answer_position=len(ids) - 1,
                   boundaries=question_first_boundaries(tokenizer, ids, row["prompt"]))


def boundary_records(rows: list[dict]) -> list[dict]:
    """One record per (prompt, statement boundary); labels are Phase Four's, from the program alone."""
    records = []
    for row in rows:
        labels = trace_labels(row["original_prompt"])
        if len(labels) != len(row["boundaries"]):
            raise ValueError(f"{row['id']}: boundaries and labels disagree")
        for position, label in zip(row["boundaries"], labels):
            records.append({"id": row["id"], "group_id": row["group_id"], "side": row["side"],
                            "asked": row["variable"], "part": row["part"], "position": position, **label})
    return records


# ---------------------------------------------------------------- Stage 0: behaviour


def behaviour(target: Target, rows: list[dict], convention: str) -> list[dict]:
    """Greedy answers in both formats; scored under the family's frozen convention."""
    out = []
    for index, row in enumerate(rows):
        entry = {"id": row["id"], "part": row["part"], "variable": row["variable"]}
        for name, key in (("question_first", "input_ids"), ("question_after", "original_input_ids")):
            ids = row[key]
            ids_t, mask_t = target.tensors([ids], [[1] * len(ids)])
            generation = target.greedy(ids_t, mask_t, Site(layer=0, position=len(ids) - 1), max_tokens=MAX_NEW_TOKENS)
            entry[name] = {"text": generation["text"], "terminated": generation["terminated"],
                           "correct": answer_text_matches(generation["text"], row["answer"], convention)}
        out.append(entry)
        if index % 256 == 0:
            print(f"  behaviour {index}/{len(rows)}", flush=True)
    return out


def behaviour_summary(results: list[dict]) -> dict:
    def accuracy(name, keep):
        chosen = [r[name]["correct"] for r in results if keep(r)]
        return float(np.mean(chosen)) if chosen else None

    out = {}
    for name in ("question_first", "question_after"):
        out[name] = {
            "all": accuracy(name, lambda r: True),
            "pilot_split": accuracy(name, lambda r: r["part"] == "test"),
            "pilot_split_correct": sum(r[name]["correct"] for r in results if r["part"] == "test"),
            **{f"asks_{v}": accuracy(name, lambda r, v=v: r["variable"] == v) for v in ("x", "y")},
        }
    return out


# ---------------------------------------------------------------- Stage 1: probes


def _fit(block, values, part, rows, device, labels_train=None):
    """Phase Four's probe on one subset; alpha on select; returns (alpha, select and test predictions)."""
    train, select, test = rows & (part == "train"), rows & (part == "select"), rows & (part == "test")
    evaluate = np.concatenate([block[select], block[test]])
    targets = values[train] if labels_train is None else labels_train
    predictions = fit_predict(block[train], targets, evaluate, ALPHA_GRID, device)
    n_select = int(select.sum())
    scores = [float(np.mean(p[:n_select] == values[select])) for p in predictions]
    best = max(range(len(ALPHA_GRID)), key=lambda i: (scores[i], i))
    return ALPHA_GRID[best], predictions[best][:n_select], predictions[best][n_select:]


def _categories(predicted, truth, categories, computed) -> dict:
    return {
        "overall": float(np.mean(predicted == truth)),
        **{c: category_accuracy(predicted, truth, categories, computed, c) for c in CATEGORIES},
        **{f"{c}_computed": category_accuracy(predicted, truth, categories, computed, c, True) for c in CATEGORIES},
    }


def survey(features: np.ndarray, records: list[dict], device: str):
    """Per layer, condition and variable: Phase Four's probe; held-out predictions kept for E6."""
    part = np.array([r["part"] for r in records])
    asked = np.array([r["asked"] for r in records])
    values = {v: np.array([r[v]["value"] for r in records]) for v in ("x", "y")}
    categories = {v: np.array([r[v]["category"] for r in records]) for v in ("x", "y")}
    computed = {v: np.array([r[v]["computed"] for r in records]) for v in ("x", "y")}
    grid, held_out = {}, {}
    for layer in range(features.shape[1]):
        block = features[:, layer]
        entry = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            for v in ("x", "y"):
                rows = (asked == v) if condition == "asked" else (asked != v)
                alpha, select_pred, test_pred = _fit(block, values[v], part, rows, device)
                select, test = rows & (part == "select"), rows & (part == "test")
                entry[condition][v] = {
                    "alpha": alpha,
                    "select": _categories(select_pred, values[v][select], categories[v][select], computed[v][select]),
                    "test": _categories(test_pred, values[v][test], categories[v][test], computed[v][test]),
                }
                held_out[(layer, condition, v)] = test_pred
        grid[layer] = entry
        if layer % 8 == 0:
            print(f"  probes: layer {layer}", flush=True)
    return grid, held_out


def gate_readings(features: np.ndarray, records: list[dict], grid: dict, device: str) -> dict:
    """Phase Four's readings on the asked condition, each with a permuted-label control at its layer."""
    summary = readings({layer: entry["asked"] for layer, entry in grid.items()})
    del summary["gate_G4"]
    part = np.array([r["part"] for r in records])
    asked = np.array([r["asked"] for r in records])
    rng = np.random.default_rng(PERMUTATION_SEED)
    for name in ("R1", "R2"):
        layer = summary[name]["layer"]
        controls = {}
        for v in ("x", "y"):
            values = np.array([r[v]["value"] for r in records])
            rows = asked == v
            alpha = grid[layer]["asked"][v]["alpha"]
            train, test = rows & (part == "train"), rows & (part == "test")
            (permuted,) = fit_predict(features[train, layer], rng.permutation(values[train]), features[test, layer], (alpha,), device)
            controls[v] = float(np.mean(permuted == values[test]))
        summary[name]["probe_supported"] = summary[name].pop("supported")
        summary[name]["control_test"] = controls
        summary[name]["supported"] = summary[name]["probe_supported"] and max(controls.values()) <= CONTROL_MAX
    return summary


def question_effect(records: list[dict], held_out: dict, layer: int, category: str) -> dict:
    """Asked minus not-asked correctness on held-out boundaries of one category, pooled over x and y.

    Pairs share the group, side, line and variable; only the question differs.
    Whole pilot groups are resampled, so both sides of a group move together.
    """
    test = [r for r in records if r["part"] == "test"]
    correct = {}
    for condition in CONDITIONS:
        for v in ("x", "y"):
            subset = [r for r in test if (r["asked"] == v) == (condition == "asked")]
            predicted = held_out[(layer, condition, v)]
            for record, guess in zip(subset, predicted, strict=True):
                correct[(condition, v, record["group_id"], record["side"], record["line"])] = int(guess == record[v]["value"])
    per_group, asked_hits, other_hits = {}, [], []
    for record in test:
        v = record["asked"]
        if record[v]["category"] != category:
            continue
        key = (v, record["group_id"], record["side"], record["line"])
        a, b = correct[("asked", *key)], correct[("not_asked", *key)]
        asked_hits.append(a)
        other_hits.append(b)
        total, count = per_group.get(record["group_id"], (0, 0))
        per_group[record["group_id"]] = (total + a - b, count + 1)
    if not per_group:
        return {"category": category, "layer": layer, "pairs": 0}

    def pooled(groups):
        return sum(total for total, _ in groups) / sum(count for _, count in groups)

    interval = group_bootstrap(list(per_group.values()), pooled, seed=BOOTSTRAP_SEED)
    return {"category": category, "layer": layer, "pairs": len(asked_hits), "groups": len(per_group),
            "asked_accuracy": float(np.mean(asked_hits)), "not_asked_accuracy": float(np.mean(other_hits)),
            "difference": interval.point, "ci95": [interval.lower, interval.upper],
            "resamples": interval.resamples, "seed": interval.seed}


def _subset_accuracy(predicted, truth, mask):
    return float(np.mean(predicted[mask] == truth[mask])) if mask.any() else None


def final_state(boundary_features: np.ndarray, answer_features: np.ndarray, rows: list[dict], records: list[dict], device: str) -> list[dict]:
    """Descriptive: final values at the program's last token and at the answer position, asked and not asked."""
    last = {}
    for index, record in enumerate(records):
        last[record["id"]] = index  # records are in prompt order, lines ascending
    end = boundary_features[[last[row["id"]] for row in rows]]
    part = np.array([row["part"] for row in rows])
    asked = np.array([row["variable"] for row in rows])
    finals = {v: np.array([records[last[row["id"]]][v]["value"] for row in rows]) for v in ("x", "y")}
    computed = {v: np.array([records[last[row["id"]]][v]["computed"] for row in rows]) for v in ("x", "y")}
    out = []
    for position, features in (("program_end", end), ("answer", answer_features)):
        for layer in range(features.shape[1]):
            entry = {"position": position, "layer": layer}
            for condition in CONDITIONS:
                for v in ("x", "y"):
                    subset = (asked == v) if condition == "asked" else (asked != v)
                    alpha, select_pred, test_pred = _fit(features[:, layer], finals[v], part, subset, device)
                    select, test = subset & (part == "select"), subset & (part == "test")
                    entry[f"{condition}_{v}"] = {
                        "alpha": alpha,
                        "select": float(np.mean(select_pred == finals[v][select])),
                        "test": float(np.mean(test_pred == finals[v][test])),
                        "test_computed": _subset_accuracy(test_pred, finals[v][test], computed[v][test]),
                    }
            out.append(entry)
    return out


def best_final(entries: list[dict]) -> dict:
    """Per position and condition: the layer maximizing min over x, y of select accuracy (ties: earlier)."""
    out = {}
    for position in ("program_end", "answer"):
        for condition in CONDITIONS:
            rows = [e for e in entries if e["position"] == position]
            best = max(rows, key=lambda e: (min(e[f"{condition}_{v}"]["select"] for v in ("x", "y")), -e["layer"]))
            out[f"{position}_{condition}"] = {"layer": best["layer"], **{v: best[f"{condition}_{v}"] for v in ("x", "y")}}
    return out


# ---------------------------------------------------------------- execution


@torch.inference_mode()
def capture(model, rows: list[dict], device: str, layers: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries, answers = [], []
    for index, row in enumerate(rows):
        ids = torch.tensor([row["input_ids"]], device=device)
        mask = torch.tensor([row["attention_mask"]], device=device)
        ids, mask = pad_to_bucket(ids, mask)
        hidden = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True).hidden_states
        if len(hidden) != layers + 1:
            raise RuntimeError("unexpected hidden-state count")
        stacked = torch.stack([h[0] for h in hidden[1:]])  # [L, T, d]
        for position in row["boundaries"]:
            boundaries.append(stored(stacked[:, position]))
        answers.append(stored(stacked[:, row["answer_position"]]))
        if index % 256 == 0:
            print(f"  capture {index}/{len(rows)}", flush=True)
    return np.stack(boundaries), np.stack(answers)


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict):
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    for manifest in (calibration, pilot):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    family = _family(args.lock)
    convention = CONVENTIONS[family]
    rows = prompt_rows(calibration, "calibration") + prompt_rows(pilot, "pilot")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"])
    tokenize(tokenizer, rows)
    records = boundary_records(rows)
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
            "question_template": QUESTION_TEMPLATE,
            "answer_convention": convention,
            "alpha_grid": ALPHA_GRID,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "prompts": len(rows),
            "boundaries": len(records),
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
            "rows": [{k: row[k] for k in ("id", "part", "prompt", "input_ids", "answer_position", "boundaries")} for row in rows],
            "records": records,
        },
    )
    model = load_model(paths["target"], "target", args.device)
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    start = time.perf_counter()
    generations = behaviour(Target(model, tokenizer), rows, convention)
    timings["behaviour"] = time.perf_counter() - start
    start = time.perf_counter()
    boundary_features, answer_features = capture(model, rows, args.device, config.num_hidden_layers)
    del model
    release_models()
    save_file({"boundaries": boundary_features, "answer": answer_features}, str(run_dir.path / "raw" / "features.safetensors"))
    timings["capture"] = time.perf_counter() - start
    start = time.perf_counter()
    grid, held_out = survey(boundary_features, records, args.device)
    gated = gate_readings(boundary_features, records, grid, args.device)
    effects = {category: question_effect(records, held_out, gated["R2"]["layer"], category) for category in CATEGORIES}
    finals = final_state(boundary_features, answer_features, rows, records, args.device)
    timings["probes"] = time.perf_counter() - start
    behaviour_scores = behaviour_summary(generations)
    u6 = behaviour_scores["question_first"]["all"] >= U6_MIN
    e6 = effects["arithmetic"]
    summary = {
        "layers": int(boundary_features.shape[1]),
        "prompts": {p: sum(r["part"] == p for r in rows) for p in ("train", "select", "test")},
        "boundaries": {p: sum(r["part"] == p for r in records) for p in ("train", "select", "test")},
        "behaviour": behaviour_scores,
        "gate_U6": {"question_first_accuracy": behaviour_scores["question_first"]["all"], "passed": u6},
        **gated,
        "gate_G6": {"passed": u6 and gated["R1"]["supported"] and gated["R2"]["supported"], "void": not u6},
        "effect_E6": {**e6, "holds": u6 and e6.get("pairs", 0) > 0 and e6["ci95"][0] > 0, "void": not u6},
        "question_effects": effects,
        "final_state": best_final(finals),
        "baselines": baselines([r for r in records if r["asked"] == "x"]),
    }
    results = {"generations": generations, "boundaries": [{"layer": layer, **entry} for layer, entry in sorted(grid.items())],
               "final_state": finals}
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p6-question-first-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
