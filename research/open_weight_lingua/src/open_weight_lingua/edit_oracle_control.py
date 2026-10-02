"""Frozen oracle-text control for the consistent answer-slot edit (D4).

protocols/edit_oracle_control.md is the frozen design; this module implements
it and nothing else. Example: D3 rewrote every "10" in the AV's description
of `x = 9; ...; x = x + 1` to "11", and the patched Gemma-3-12B answered 11.
D4 patches the same receiver with hand-written texts that also say 11 — "The
current value of x is 11.", a three-paragraph template, a bare answer slot,
and "A short note that mentions the number 11." — and asks whether the AV's
edited description moves the answer more often than any of them.

The D3 and calibration runs are read, never written. The AV is not loaded.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

from safetensors.torch import load_file
import torch

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
from .steering import TEMPLATES
from .target import Site, Target

PROTOCOL = PROJECT / "protocols" / "edit_oracle_control.md"
FORMAT = "open_weight_lingua.edit_oracle_control.v1"
SEED = 207100
ORACLES = {
    "O_terse": TEMPLATES["terse"],
    "O_structured": TEMPLATES["structured"],
    "O_slot": 'Final token "\n" ends the question, immediately expecting a numeric answer like "{value}".',
    "O_number": "A short note that mentions the number {value}.",
}
ESTIMATED_TARGET_SECONDS = 3600
THROUGHPUT_FACTOR = 3.0


def oracle_texts(variable: str, value: int) -> dict:
    return {name: template.format(variable=variable, value=value) for name, template in ORACLES.items()}


def d4_plan(d3_plan: list[dict], inputs: dict) -> list[dict]:
    """D3's primary receivers: the replayed E1g text plus every oracle asserting c."""
    plan = []
    for entry in d3_plan:
        if entry["population"] != "primary":
            continue
        c = entry["values"]["c"]
        conditions = {"E1g": {"text": entry["conditions"]["E1g"]["text"]}}
        for name, text in oracle_texts(inputs[entry["id"]]["variable"], c).items():
            conditions[name] = {"text": text}
        plan.append({"id": entry["id"], "values": entry["values"], "conditions": conditions})
    return plan


def _hit(condition: dict, value: int) -> float:
    ok = condition.get("status") == "ok"
    return float(ok and integer_answer(condition.get("generation"), CONVENTION) == value)


def summarize_d4(records: list[dict], gates: dict, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Frozen D4 statistics, paired over receivers; base rate and scores from D3's E0."""
    names = ["E1g", *ORACLES]
    hits = {n: [_hit(r["conditions"].get(n, {}), r["values"]["c"]) for r in records] for n in names}
    base = [_hit(r["d3_E0"], r["values"]["c"]) for r in records]
    summary = {"receivers": len(records), "gates": gates, "conditions": {}, "paired": {}}
    for name in names:
        keep = [
            float(integer_answer(r["conditions"].get(name, {}).get("generation"), CONVENTION) == r["values"]["a"])
            for r in records
        ]
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
            "retained_answer_rate": sum(keep) / len(records),
            "off_target_rate": 1.0 - sum(keep) / len(records) - sum(hits[name]) / len(records),
            "mean_cosine_to_original": sum(r["conditions"][name]["cosine"] for r in records) / len(records),
        }
        if shift_edit:
            shift = paired_logprob_difference(shift_edit, shift_base, resamples=resamples, seed=seed)
            entry["logprob_shift"] = {"point": shift.point, "lower": shift.lower, "receivers": len(shift_edit)}
        summary["conditions"][name] = entry
    for name in ORACLES:
        forward = paired_logprob_difference(hits["E1g"], hits[name], resamples=resamples, seed=seed)
        reverse = paired_logprob_difference(hits[name], hits["E1g"], resamples=resamples, seed=seed)
        summary["paired"][name] = {
            "E1g_minus_oracle": {"point": forward.point, "lower": forward.lower},
            "oracle_minus_E1g": {"point": reverse.point, "lower": reverse.lower},
        }
    conditions = summary["conditions"]

    def r4c(name: str) -> bool:
        entry = conditions[name]
        return entry["hit_vs_base"]["lower"] > 0 and entry.get("logprob_shift", {}).get("lower", 0.0) > 0

    summary["readings"] = {
        "R4a_av_description_adds_beyond_number": all(
            summary["paired"][name]["E1g_minus_oracle"]["lower"] > 0 for name in ORACLES
        ),
        "R4b_hand_written_text_moves_answer_as_often": any(
            conditions[name]["hit_rate"] >= conditions["E1g"]["hit_rate"] and conditions[name]["hit_vs_base"]["lower"] > 0
            for name in ORACLES
        ),
        "R4c_oracle_moves_answer": {name: r4c(name) for name in ORACLES},
        "interpretable": bool(gates.get("harness_reproduced")),
    }
    return summary


def run_d4(args, run: RunDirectory, timings: dict):
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
    plan = d4_plan(d3_plan, inputs)
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
            "oracle_templates": ORACLES,
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
        record = {"id": rid, "values": entry["values"], "d3_E0": d3_results[rid]["conditions"]["E0"], "conditions": {}}
        for name, condition in entry["conditions"].items():
            try:
                result = _behavior(target, ids, mask, site, original, norm, directions[(rid, name)], answers, p0)
            except (ValueError, RuntimeError) as failure:
                result = {"status": "failed", "error": str(failure)}
            record["conditions"][name] = {**result, "cosine": condition["cosine"]}
        kept = d3_results[rid]["conditions"]["E1g"]
        record["gate_E1g_replay"] = (
            record["conditions"]["E1g"].get("generation") == kept["generation"]
            and record["conditions"]["E1g"].get("scores") == kept["scores"]
        )
        replayed += record["gate_E1g_replay"]
        records.append(record)
        print(f"  D4: {rid} replay={record['gate_E1g_replay']}", flush=True)
        if done >= 4:
            projected = (time.perf_counter() - start) / done * len(plan)
            if projected > THROUGHPUT_FACTOR * ESTIMATED_TARGET_SECONDS:
                raise RuntimeError(f"throughput stop: target stage projects {projected:.0f} s")
    del target
    release_models()
    timings["target_behavior"] = time.perf_counter() - start
    gates = {"E1g_replay_bitwise": replayed, "receivers": len(records), "harness_reproduced": replayed == len(records)}
    return records, summarize_d4(records, gates)


def audit(run_dir: Path) -> dict:
    """Recount every condition's hit rate from saved generations, without summarize_d4."""
    results = json.loads((run_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    problems = []
    for name in ["E1g", *ORACLES]:
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
    parser.add_argument("--audit", type=Path, help="recount a completed D4 run and exit")
    args = parser.parse_args(argv)
    if args.audit is None and None in (args.lock, args.d3_run, args.calibration_run):
        parser.error("--lock, --d3-run and --calibration-run are required unless --audit is given")
    if args.lock is not None and _family(args.lock) != "gemma3-12b":
        parser.error("D4 is frozen for Gemma-3-12B only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.audit is not None:
        print(json.dumps(audit(args.audit), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RunDirectory(args.run_dir or PROJECT / "runs" / f"editoracle-d4-gemma3-12b-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error = "FAILED", None
    try:
        records, summary = run_d4(args, run, timings)
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
