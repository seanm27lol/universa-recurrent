"""Phases Fourteen and Fifteen confirmatory: the late read with the question first, on a fifth fresh split.

protocols/phase_fifteen_confirmatory_fresh.md is the frozen design. Example:

    What is x? ... / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (y ends at 14)

The exploratory phases found that with the question before the program:
- the post-program positions cannot answer from the program's last line
  alone (no end-of-program summary);
- Gemma's read ends later than with the question after, Qwen's at the same
  layer;
- hiding the other variable's lines from the post-program positions removes
  most question-first errors on programs that end on that variable.

This module generates new groups (seed 210001, excluding Phase Two and the
four earlier fresh splits), runs every prompt in both orders once, and tests
eight pre-registered claims.

    run       one family on the GPU: generate, record hashes, knock out
    evaluate  CPU: paired measures, Bonferroni bootstrap, verdicts
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from .answer_knockout import knockout, logits_at, score
from .artifacts import RunDirectory, sha256_file, write_json
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .question_order_knockout import order_row
from .runner import release_models, source_identity
from .scale_confirm import FRESH_INVENTORIES as EARLIER_INVENTORIES
from .scale_confirm import fresh_groups as fourth_fresh_groups
from .selection_confirm import fresh_groups as third_fresh_groups
from .span_knockout_confirm import fresh_groups as first_fresh_groups
from .splits import SPLIT_ORDER, build_plan
from .supersession_confirm import fresh_groups as second_fresh_groups
from .tasks import canonical, generate_groups
from .variable_choice_knockout import choice_row

PROTOCOL = PROJECT / "protocols" / "phase_fifteen_confirmatory_fresh.md"
FORMAT = "open_weight_lingua.order_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
SPLIT = "fresh_2026_10e"
GROUPS = 512
FRESH_INVENTORIES = {**EARLIER_INVENTORIES,
                     "fresh_2026_10d": "e2c9ab251db8cd495fd9cafa60d0f289483d99ae11fd8524b4b3269990bbede7"}
LATE_LAYER = {"gemma3-12b": 30, "qwen2.5-7b": 24}  # each family's question-after release layer (Phase Eight)
FIRST_CONDITIONS = ("baseline", "block_other", "only_last_line", "floor", "late")
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 807100, "qwen2.5-7b": 807101}
COVERAGE = 1 - 0.05 / 8
HYPOTHESES = (  # (id, family, measure, stratum, direction, threshold)
    ("G1", "gemma3-12b", "residual_gap", "main", "below", 0.05),
    ("G2", "gemma3-12b", "recovery", "main", "above", 0.05),
    ("G3", "gemma3-12b", "summary", "main", "below", 0.10),
    ("G4", "gemma3-12b", "late_loss", "all", "above", 0.05),
    ("Q1", "qwen2.5-7b", "residual_gap", "main", "below", 0.05),
    ("Q2", "qwen2.5-7b", "recovery", "main", "above", 0.05),
    ("Q3", "qwen2.5-7b", "summary", "main", "below", 0.10),
    ("Q4", "qwen2.5-7b", "late_loss", "all", "below", 0.05),
)


def fresh_groups(plan, count: int = GROUPS):
    """New groups excluding Phase Two and the four earlier fresh splits (each verified against its inventory)."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    excluded = {canonical(p) for name in SPLIT_ORDER for g in plan.groups[name] for p in (g.source, g.counterfactual)}
    for name, generate in (("fresh_2026_10", first_fresh_groups), ("fresh_2026_10b", second_fresh_groups),
                           ("fresh_2026_10c", third_fresh_groups), ("fresh_2026_10d", fourth_fresh_groups)):
        groups, stats = generate(plan)
        if stats["inventory_sha256"] != FRESH_INVENTORIES[name]:
            raise ValueError(f"{name} does not regenerate to its recorded inventory")
        excluded |= {canonical(p) for g in groups for p in (g.source, g.counterfactual)}
    groups, stats = generate_groups(SPLIT, count, excluded_programs=excluded)
    if {canonical(p) for g in groups for p in (g.source, g.counterfactual)} & excluded:
        raise ValueError("a fresh program repeats an earlier program")
    return groups, stats


def _ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))


def fresh_rows(groups, tokenizer) -> list[dict]:
    """One row per prompt, carrying both question orders."""
    rows = []
    for group in groups:
        for base in group.variants():
            item = {**base, "input_ids": _ids(tokenizer, base["prompt"])}
            after = order_row(item, SPLIT, "after", tokenizer)
            first = order_row(item, SPLIT, "first", tokenizer)
            choice = choice_row(item, SPLIT, "first", tokenizer)
            if choice["span"] != first["span"] or choice["input_ids"] != first["input_ids"]:
                raise ValueError(f"{base['id']}: the two row builders disagree")
            rows.append({
                "id": base["id"], "group_id": base["group_id"], "variable": base["variable"], "answer": base["answer"],
                "last_line_assigns_asked": first["last_line_assigns_asked"], "depends_on_other": choice["depends_on_other"],
                "asked_last_update_kind": first["asked_last_update_kind"],
                "after": {k: after[k] for k in ("prompt", "input_ids", "span", "sequence")},
                "first": {**{k: first[k] for k in ("prompt", "input_ids", "span", "sequence", "before_last_line")},
                          "block_other": choice["keys"]["block_other"],
                          "other_sequence": choice["sequences"].get("other")},
            })
    return rows


def main_stratum(row: dict) -> bool:
    """Programs ending on the other variable, whose answer does not depend on a copy of it."""
    return not row["last_line_assigns_asked"] and not row["depends_on_other"]


# ---------------------------------------------------------------- run (GPU)


def _noop_checked(model, row_id: str, sequence: dict, device: str) -> torch.Tensor:
    baseline = logits_at(model, sequence["ids"], sequence["positions"], device)
    with knockout(model, 0, sequence["positions"], []):
        if not torch.equal(logits_at(model, sequence["ids"], sequence["positions"], device), baseline):
            raise RuntimeError(f"{row_id}: the no-op hook changed the logits")
    return baseline


def evaluate_row(model, row: dict, late_layer: int, device: str) -> dict:
    after, first = row["after"], row["first"]
    out = {"after": {"baseline": score(_noop_checked(model, row["id"], after["sequence"], device), after["sequence"]["targets"])}}
    seq = first["sequence"]
    program = list(range(first["span"][0], first["span"][1] + 1))
    blocks = {"block_other": (0, first["block_other"]), "only_last_line": (0, first["before_last_line"]),
              "floor": (0, program), "late": (late_layer, program)}
    scored = {"baseline": score(_noop_checked(model, row["id"], seq, device), seq["targets"])}
    for name, (layer, keys) in blocks.items():
        with knockout(model, layer, list(range(first["span"][1] + 1, len(seq["ids"]))), keys):
            scored[name] = score(logits_at(model, seq["ids"], seq["positions"], device), seq["targets"])
    other = first["other_sequence"]
    if other is not None:  # descriptive: how often the other variable's final value is said
        for name in ("baseline", "block_other"):
            keys = first["block_other"] if name == "block_other" else []
            with knockout(model, 0, list(range(first["span"][1] + 1, len(other["ids"]))), keys):
                scored[f"{name}_says_other"] = score(logits_at(model, other["ids"], other["positions"], device), other["targets"])
    out["first"] = scored
    return out


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    late_layer = LATE_LAYER[family]
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    groups, stats = fresh_groups(build_plan())
    rows = fresh_rows(groups, tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(), "model_lock_sha256": sha256_file(args.lock),
        "plan_hash": PLAN_HASH, "split": SPLIT, "generation": stats, "excluded_fresh_inventories": FRESH_INVENTORIES,
        "stimuli_sha256": sha256_file(stimuli), "late_layer": late_layer,
        "prompts": len(rows), "main_stratum_prompts": sum(main_stratum(r) for r in rows),
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], **evaluate_row(model, row, late_layer, args.device)}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    return {"prompts": len(rows)}


# ---------------------------------------------------------------- evaluate (CPU)


def measures(result: dict) -> dict:
    after, first = result["after"], result["first"]
    correct = {name: int(first[name]["correct"]) for name in FIRST_CONDITIONS}
    return {"recovery": correct["block_other"] - correct["baseline"],
            "residual_gap": int(after["baseline"]["correct"]) - correct["block_other"],
            "summary": correct["only_last_line"] - correct["floor"],
            "late_loss": correct["baseline"] - correct["late"]}


def load_family(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    results = [json.loads(line) for line in (run / "results.jsonl").open()]
    stimuli = json.loads((run / "stimuli.json").read_text())
    if completion["status"] != "COMPLETE" or len(results) != manifest["prompts"]:
        raise ValueError(f"{run}: incomplete runs never enter the confirmatory analysis")
    if manifest["plan_hash"] != PLAN_HASH or manifest["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError(f"{run}: plan or protocol differs from the frozen ones")
    if manifest["stimuli_sha256"] != sha256_file(run / "stimuli.json") or [s["id"] for s in stimuli] != [r["id"] for r in results]:
        raise ValueError(f"{run}: stimuli and results disagree")
    return manifest, stimuli, results


def bootstrap(stimuli, results, name: str, seed: int) -> dict:
    groups = sorted({s["group_id"] for s in stimuli})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for s, r in zip(stimuli, results):
        sums[index[s["group_id"]]] += measures(r)[name]
        counts[index[s["group_id"]]] += 1
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean": float(sums.sum() / counts.sum()), "prompts": int(counts.sum()), "groups": len(groups),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


def describe(stimuli, results) -> dict:
    def rate(chosen, order, name):
        return float(np.mean([r[order][name]["correct"] for r in chosen])) if chosen else None

    out = {}
    for stratum, keep in (("all", lambda s: True), ("main", main_stratum)):
        chosen = [r for s, r in zip(stimuli, results) if keep(s)]
        said = [r for r in chosen if "baseline_says_other" in r["first"]]
        out[stratum] = {"n": len(chosen), "after_baseline": rate(chosen, "after", "baseline"),
                        **{f"first_{c}": rate(chosen, "first", c) for c in FIRST_CONDITIONS},
                        "first_says_other": {c: rate(said, "first", f"{c}_says_other") for c in ("baseline", "block_other")}}
    return out


def evaluate(runs: list[Path]) -> dict:
    families = {}
    for run in runs:
        manifest, stimuli, results = load_family(run)
        families[manifest["family"]] = (run, manifest, stimuli, results)
    inventories = {m["generation"]["inventory_sha256"] for _, m, _, _ in families.values()}
    if len(inventories) > 1:
        raise ValueError("the families were given different fresh programs")
    verdicts = []
    for test_id, family, name, stratum, direction, threshold in HYPOTHESES:
        if family not in families:
            verdicts.append({"id": test_id, "family": family, "measure": name, "status": "NOT_RUN"})
            continue
        _, _, stimuli, results = families[family]
        chosen = [(s, r) for s, r in zip(stimuli, results) if stratum == "all" or main_stratum(s)]
        estimate = bootstrap([s for s, _ in chosen], [r for _, r in chosen], name, BOOTSTRAP_SEEDS[family])
        low, high = estimate["interval"]
        supported = low > threshold if direction == "above" else high < threshold
        verdicts.append({"id": test_id, "family": family, "measure": name, "stratum": stratum, "direction": direction,
                         "threshold": threshold, **estimate, "coverage": COVERAGE,
                         "status": "SUPPORTED" if supported else "NOT_SUPPORTED"})
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "fresh_inventory_sha256": next(iter(inventories)), "hypotheses": verdicts,
            "families": {f: {"run": str(run), "prompts": len(results), "describe": describe(stimuli, results)}
                         for f, (run, _, stimuli, results) in families.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--lock", type=Path, required=True)
    go.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    go.add_argument("--device", default="cuda")
    go.add_argument("--run-dir", type=Path)
    check = sub.add_parser("evaluate")
    check.add_argument("--runs", type=Path, nargs="+", required=True)
    check.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        result = evaluate(args.runs)
        write_json(args.output, result)
        for h in result["hypotheses"]:
            print(json.dumps({k: h.get(k) for k in ("id", "measure", "mean", "interval", "status")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p15-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
