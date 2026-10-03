"""Frozen consistency-dose diagnostic for the AV-description edit (D6).

protocols/edit_consistency_dose.md is the frozen design; this module implements
it and nothing else. Example: a Gemma-3-12B description that says "Result:
10", "x = 10", ... and, in its "Final token" slot, "10" three times mentions 10
eleven times. Rewriting all eleven to 11 moved the patched answer to 11 in D3;
rewriting only the slot's three did not. D6 rewrites the first quarter, half
and three-quarters of the mentions, and separately every mention except the
slot's, to see whether agreement or location drives the flip.

The D3 and calibration runs are read, never written. The AV is not loaded.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from safetensors.torch import load_file
import torch

from . import answer_slot
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import integer_answer
from .edit_diagnostics import CONVENTION, PROJECT, RESAMPLES, _family, _git_head
from .edit_replication import _behavior
from .geometry import direction_metrics
from .nla_adapter import Reconstructor
from .preflight import (
    compatibility,
    inspect_metadata,
    load_model,
    model_paths,
    read_lock,
    verify_models,
)
from .runner import release_models, source_identity
from .stats import paired_logprob_difference
from .target import Site, Target

PROTOCOL = PROJECT / "protocols" / "edit_consistency_dose.md"
FORMAT = "open_weight_lingua.edit_consistency_dose.v1"
SEED = 209100
DOSES = {"K25": 0.25, "K50": 0.50, "K75": 0.75}
CONDITIONS = ("K100", "K25", "K50", "K75", "NS")
ESTIMATED_TARGET_SECONDS = 3600
THROUGHPUT_FACTOR = 3.0


def mention_spans(text: str, value: int) -> list[tuple[int, int]]:
    """Standalone occurrences of value, in text order (edit_everywhere's pattern)."""
    pattern = re.compile(answer_slot._STANDALONE.format(value=value))
    return [m.span() for m in pattern.finditer(text)]


def rewrite_spans(text: str, spans, new_value: int) -> str:
    out = text
    for start, stop in sorted(spans, reverse=True):
        out = out[:start] + str(new_value) + out[stop:]
    return out


def dose_texts(text: str, a: int, c: int) -> dict:
    """K25/K50/K75 (first ceil(f*n) mentions) and NS (all but the slot candidates)."""
    spans = mention_spans(text, a)
    slot = set(answer_slot.parse(text).spans)
    texts = {
        name: rewrite_spans(text, spans[: math.ceil(fraction * len(spans))], c)
        for name, fraction in DOSES.items()
    }
    texts["NS"] = rewrite_spans(text, [s for s in spans if s not in slot], c)
    return {"texts": texts, "mentions": len(spans), "outside_slot": len([s for s in spans if s not in slot])}


def d6_plan(d3_plan: list[dict]) -> list[dict]:
    plan = []
    for entry in d3_plan:
        if entry["population"] != "primary":
            continue
        a, c = entry["values"]["a"], entry["values"]["c"]
        doses = dose_texts(entry["conditions"]["E0"]["text"], a, c)
        conditions = {"K100": {"text": entry["conditions"]["E1g"]["text"]}}
        conditions.update({name: {"text": text} for name, text in doses["texts"].items()})
        plan.append(
            {
                "id": entry["id"],
                "values": entry["values"],
                "mentions": doses["mentions"],
                "outside_slot": doses["outside_slot"],
                "conditions": conditions,
            }
        )
    return plan


def _hit(condition: dict, value: int) -> float:
    ok = condition.get("status") == "ok"
    return float(ok and integer_answer(condition.get("generation"), CONVENTION) == value)


def summarize_d6(records: list[dict], gates: dict, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Frozen D6 statistics, paired over receivers; base rates and L against D3's E0."""
    hits = {n: [_hit(r["conditions"].get(n, {}), r["values"]["c"]) for r in records] for n in CONDITIONS}
    base = [_hit(r["d3_E0"], r["values"]["c"]) for r in records]
    slot_only = [_hit(r["d3_E1"], r["values"]["c"]) for r in records]
    summary = {
        "receivers": len(records),
        "gates": gates,
        "reference": {"E0_hit_rate": sum(base) / len(records), "E1_slot_only_hit_rate": sum(slot_only) / len(records)},
        "conditions": {},
    }
    for name in CONDITIONS:
        kept = sum(
            integer_answer(r["conditions"].get(name, {}).get("generation"), CONVENTION) == r["values"]["a"]
            for r in records
        )
        shift_edit, shift_base = [], []
        for r in records:
            scores = r["conditions"].get(name, {}).get("scores", {})
            c, a = str(r["values"]["c"]), str(r["values"]["a"])
            if c in scores and a in scores:
                shift_edit.append(scores[c] - scores[a])
                shift_base.append(r["d3_E0"]["scores"][c] - r["d3_E0"]["scores"][a])
        moved = paired_logprob_difference(hits[name], base, resamples=resamples, seed=seed)
        entry = {
            "hit_rate": sum(hits[name]) / len(records),
            "hit_vs_base": {"point": moved.point, "lower": moved.lower},
            "retained_answer_rate": kept / len(records),
            "mean_cosine_to_original": sum(r["conditions"][name]["cosine"] for r in records) / len(records),
        }
        if shift_edit:
            shift = paired_logprob_difference(shift_edit, shift_base, resamples=resamples, seed=seed)
            entry["logprob_shift"] = {"point": shift.point, "lower": shift.lower, "receivers": len(shift_edit)}
        summary["conditions"][name] = entry
    trend = paired_logprob_difference(hits["K75"], hits["K25"], resamples=resamples, seed=seed)
    summary["paired"] = {"K75_minus_K25": {"point": trend.point, "lower": trend.lower}}
    ns, full = summary["conditions"]["NS"], summary["conditions"]["K100"]
    summary["readings"] = {
        "R6a_more_agreement_more_flips": trend.lower > 0,
        "R6b_slot_not_needed": ns["hit_vs_base"]["lower"] > 0 and ns["hit_rate"] >= 0.5 * full["hit_rate"],
        "interpretable": bool(gates.get("harness_reproduced")),
    }
    return summary


def run_d6(args, run: RunDirectory, timings: dict):
    d3_manifest = json.loads((args.d3_run / "manifest.json").read_text())
    d3_plan = json.loads((args.d3_run / "plan.json").read_text())
    d3_results = {r["id"]: r for r in json.loads((args.d3_run / "results.json").read_text())}
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    if d3_manifest["calibration"]["run"] != args.calibration_run.name:
        raise ValueError("calibration run differs from the one D3 used")
    inputs = {r["id"]: r for r in calibration["inputs"]}
    lock_sha = sha256_file(args.lock)
    if lock_sha != d3_manifest["model_lock_sha256"]:
        raise ValueError("lock differs from the D3 run's lock")
    plan = d6_plan(d3_plan)
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models(lock, paths)
    timings["hash_verification"] = time.perf_counter() - start
    tokenizers, _, ar_meta, _ = inspect_metadata(paths, lock)
    write_json(
        run.path / "manifest.json",
        {
            "format": FORMAT,
            "family": _family(args.lock),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_head": _git_head(),
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_file_sha256": source_identity(),
            "answer_slot_rule_sha256": answer_slot.RULE_SHA256,
            "doses": DOSES,
            "model_lock_sha256": lock_sha,
            "d3": {
                "run": args.d3_run.name,
                "manifest_sha256": sha256_file(args.d3_run / "manifest.json"),
                "results_sha256": sha256_file(args.d3_run / "results.json"),
                "plan_sha256": sha256_file(args.d3_run / "plan.json"),
            },
            "calibration": {"run": args.calibration_run.name, "plan_hash": calibration["plan_hash"]},
            "statistics": {"resamples": RESAMPLES, "seed": SEED, "convention": CONVENTION},
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
            "plan": plan,
        },
    )
    originals = {}
    for entry in plan:
        saved = load_file(str(args.calibration_run / "raw" / f"{entry['id']}-original.safetensors"))
        layer, row, position = (int(v) for v in saved["site_int64"])
        originals[entry["id"]] = (saved["original"], float(saved["retained_norm_float32"]), Site(layer, position, row))

    start = time.perf_counter()
    ar = Reconstructor(
        load_model(paths["ar"], "ar", args.device), tokenizers["ar"], ar_meta, paths["ar"] / "value_head.safetensors"
    )
    directions = {}
    for entry in plan:
        for name, condition in entry["conditions"].items():
            vector = ar.reconstruct(condition["text"]).float().cpu()
            directions[(entry["id"], name)] = vector
            condition["cosine"] = direction_metrics(originals[entry["id"]][0], vector)["cosine"]
    del ar
    release_models()
    timings["ar_reconstruct"] = time.perf_counter() - start

    start = time.perf_counter()
    target = Target(load_model(paths["target"], "target", args.device), tokenizers["target"])
    records, replayed = [], 0
    for done, entry in enumerate(plan, start=1):
        rid = entry["id"]
        original, norm, site = originals[rid]
        ids, mask = target.tensors([inputs[rid]["input_ids"]], [inputs[rid]["attention_mask"]])
        p0 = load_file(str(args.d3_run / "raw" / f"{rid}-p0.safetensors"))["p0_next_token_logits"]
        answers = sorted({str(v) for v in entry["values"].values() if v is not None})
        record = {
            "id": rid,
            "values": entry["values"],
            "mentions": entry["mentions"],
            "outside_slot": entry["outside_slot"],
            "d3_E0": d3_results[rid]["conditions"]["E0"],
            "d3_E1": d3_results[rid]["conditions"]["E1"],
            "conditions": {},
        }
        for name, condition in entry["conditions"].items():
            try:
                result = _behavior(target, ids, mask, site, original, norm, directions[(rid, name)], answers, p0)
            except (ValueError, RuntimeError) as failure:
                result = {"status": "failed", "error": str(failure)}
            record["conditions"][name] = {**result, "cosine": condition["cosine"]}
        kept = d3_results[rid]["conditions"]["E1g"]
        full = record["conditions"]["K100"]
        record["gate_K100_replay"] = full.get("generation") == kept["generation"] and full.get("scores") == kept["scores"]
        replayed += record["gate_K100_replay"]
        records.append(record)
        print(f"  D6: {rid} n={entry['mentions']} replay={record['gate_K100_replay']}", flush=True)
        if done >= 4:
            projected = (time.perf_counter() - start) / done * len(plan)
            if projected > THROUGHPUT_FACTOR * ESTIMATED_TARGET_SECONDS:
                raise RuntimeError(f"throughput stop: target stage projects {projected:.0f} s")
    del target
    release_models()
    timings["target_behavior"] = time.perf_counter() - start
    gates = {"K100_replay_bitwise": replayed, "receivers": len(records), "harness_reproduced": replayed == len(records)}
    return records, summarize_d6(records, gates)


def audit(run_dir: Path) -> dict:
    """Recount every condition's hit rate from saved generations, without summarize_d6."""
    results = json.loads((run_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    problems = []
    for name in CONDITIONS:
        hits = 0
        for r in results:
            generation = r["conditions"].get(name, {}).get("generation")
            text = generation["text"].rstrip() if generation and generation["terminated"] else None
            hits += text == str(r["values"]["c"])
        if abs(hits / len(results) - summary["conditions"][name]["hit_rate"]) > 1e-12:
            problems.append(f"{name} hit rate does not recount")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--d3-run", type=Path)
    parser.add_argument("--calibration-run", type=Path)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit", type=Path, help="recount a completed D6 run and exit")
    args = parser.parse_args(argv)
    if args.audit is None and None in (args.lock, args.d3_run, args.calibration_run):
        parser.error("--lock, --d3-run and --calibration-run are required unless --audit is given")
    if args.lock is not None and _family(args.lock) != "gemma3-12b":
        parser.error("D6 is frozen for Gemma-3-12B only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.audit is not None:
        print(json.dumps(audit(args.audit), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RunDirectory(args.run_dir or PROJECT / "runs" / f"editdose-d6-gemma3-12b-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error = "FAILED", None
    try:
        records, summary = run_d6(args, run, timings)
        write_json(run.path / "results.json", records)
        write_json(run.path / "summary.json", summary)
        status = "COMPLETE"
    except Exception as failure:  # recorded in completion.json, then re-raised
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        write_json(
            run.path / "completion.json",
            {
                "status": status,
                "error": error,
                "timings_seconds": timings,
                "wall_seconds": time.perf_counter() - started,
                "setup_seconds": float(os.environ.get("OWL_SETUP_SECONDS", "nan")),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            },
        )
        print(f"{status}: {run.path}", flush=True)


if __name__ == "__main__":
    main()
