"""Frozen receiver-specificity control for the consistent edit (D5).

protocols/edit_foreign_control.md is the frozen design; this module implements
it and nothing else. Example: D3 rewrote every "10" in the AV's description of
receiver `x = 9; ...; x = x + 1` to "11" and the patched Gemma-3-12B answered
11. D5 instead takes the AV's description of a *different* program that also
asks for x, rewrites its own answer everywhere to 11, and patches that into the
same receiver. If it works as well, the effect belongs to AV-register text that
states a number, not to this activation.

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

PROTOCOL = PROJECT / "protocols" / "edit_foreign_control.md"
FORMAT = "open_weight_lingua.edit_foreign_control.v1"
SEED = 208100
CONDITIONS = ("F0", "F1", "F2")
ESTIMATED_TARGET_SECONDS = 3600
THROUGHPUT_FACTOR = 3.0


def donors(primary_ids: list[str], inputs: dict) -> dict:
    """Next primary receiver, cyclically, that queries the same variable."""
    out = {}
    for index, rid in enumerate(primary_ids):
        variable = inputs[rid]["variable"]
        for step in range(1, len(primary_ids)):
            candidate = primary_ids[(index + step) % len(primary_ids)]
            if inputs[candidate]["variable"] == variable:
                out[rid] = candidate
                break
        else:
            raise ValueError(f"no donor with the same variable for {rid}")
    return out


def d5_plan(d3_plan: list[dict], inputs: dict) -> list[dict]:
    """F0 (own edited), F1 (donor edited to this c), F2 (donor unedited)."""
    primary = {entry["id"]: entry for entry in d3_plan if entry["population"] == "primary"}
    order = list(primary)
    donor_of = donors(order, inputs)
    plan = []
    for rid in order:
        entry, donor = primary[rid], primary[donor_of[rid]]
        donor_text = donor["conditions"]["E0"]["text"]
        if answer_slot.parse(donor_text).lead != donor["values"]["a"]:
            raise ValueError(f"donor {donor['id']} is not a primary description")
        c = entry["values"]["c"]
        plan.append(
            {
                "id": rid,
                "donor_id": donor["id"],
                "values": {**entry["values"], "donor_a": donor["values"]["a"]},
                "donor_states_c_already": donor["values"]["a"] == c,
                "conditions": {
                    "F0": {"text": entry["conditions"]["E1g"]["text"]},
                    "F1": {"text": answer_slot.edit_everywhere(donor_text, c)},
                    "F2": {"text": donor_text},
                },
            }
        )
    return plan


def _answer(condition: dict) -> int | None:
    if condition.get("status") != "ok":
        return None
    return integer_answer(condition.get("generation"), CONVENTION)


def summarize_d5(records: list[dict], gates: dict, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Frozen D5 statistics, paired over receivers; L is measured against D3's E0."""
    hit_c = {n: [float(_answer(r["conditions"].get(n, {})) == r["values"]["c"]) for r in records] for n in CONDITIONS}
    summary = {
        "receivers": len(records),
        "donor_states_c_already": sum(r["donor_states_c_already"] for r in records),
        "gates": gates,
        "conditions": {},
    }
    for name in CONDITIONS:
        answers = [_answer(r["conditions"].get(name, {})) for r in records]
        keep = sum(a == r["values"]["a"] for a, r in zip(answers, records))
        donor = sum(a == r["values"]["donor_a"] for a, r in zip(answers, records))
        shift_edit, shift_base = [], []
        for r in records:
            scores = r["conditions"].get(name, {}).get("scores", {})
            c, a = str(r["values"]["c"]), str(r["values"]["a"])
            if c in scores and a in scores:
                shift_edit.append(scores[c] - scores[a])
                shift_base.append(r["d3_E0"]["scores"][c] - r["d3_E0"]["scores"][a])
        entry = {
            "hit_rate_c": sum(hit_c[name]) / len(records),
            "retained_answer_rate": keep / len(records),
            "donor_answer_rate": donor / len(records),
            "mean_cosine_to_original": sum(r["conditions"][name]["cosine"] for r in records) / len(records),
        }
        if shift_edit:
            shift = paired_logprob_difference(shift_edit, shift_base, resamples=resamples, seed=seed)
            entry["logprob_shift_c"] = {"point": shift.point, "lower": shift.lower, "receivers": len(shift_edit)}
        summary["conditions"][name] = entry
    own_minus_foreign = paired_logprob_difference(hit_c["F0"], hit_c["F1"], resamples=resamples, seed=seed)
    edit_over_unedited = paired_logprob_difference(hit_c["F1"], hit_c["F2"], resamples=resamples, seed=seed)
    adoptable = [r for r in records if r["values"]["donor_a"] not in (r["values"]["a"], r["values"]["c"])]
    summary["paired"] = {
        "F0_minus_F1": {"point": own_minus_foreign.point, "lower": own_minus_foreign.lower},
        "F1_minus_F2": {"point": edit_over_unedited.point, "lower": edit_over_unedited.lower},
    }
    summary["donor_adoption_F2"] = {
        "receivers": len(adoptable),
        "rate": (
            sum(_answer(r["conditions"].get("F2", {})) == r["values"]["donor_a"] for r in adoptable) / len(adoptable)
            if adoptable
            else None
        ),
    }
    f0, f1 = summary["conditions"]["F0"]["hit_rate_c"], summary["conditions"]["F1"]["hit_rate_c"]
    summary["readings"] = {
        "R5a_own_description_matters": own_minus_foreign.lower > 0,
        "R5b_foreign_description_works_as_well": f1 >= f0 and edit_over_unedited.lower > 0,
        "interpretable": bool(gates.get("harness_reproduced")),
    }
    return summary


def run_d5(args, run: RunDirectory, timings: dict):
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
    plan = d5_plan(d3_plan, inputs)
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
        d3_answers = {str(v) for k, v in entry["values"].items() if k != "donor_a" and v is not None}
        answers = sorted(d3_answers | {str(entry["values"]["donor_a"])})
        record = {
            "id": rid,
            "donor_id": entry["donor_id"],
            "values": entry["values"],
            "donor_states_c_already": entry["donor_states_c_already"],
            "d3_E0": d3_results[rid]["conditions"]["E0"],
            "conditions": {},
        }
        for name, condition in entry["conditions"].items():
            try:
                result = _behavior(target, ids, mask, site, original, norm, directions[(rid, name)], answers, p0)
            except (ValueError, RuntimeError) as failure:
                result = {"status": "failed", "error": str(failure)}
            record["conditions"][name] = {**result, "cosine": condition["cosine"]}
        kept = d3_results[rid]["conditions"]["E1g"]
        replay = record["conditions"]["F0"]
        record["gate_F0_replay"] = replay.get("generation") == kept["generation"] and {
            k: v for k, v in replay.get("scores", {}).items() if k in d3_answers
        } == kept["scores"]
        replayed += record["gate_F0_replay"]
        records.append(record)
        print(f"  D5: {rid} donor={entry['donor_id']} replay={record['gate_F0_replay']}", flush=True)
        if done >= 4:
            projected = (time.perf_counter() - start) / done * len(plan)
            if projected > THROUGHPUT_FACTOR * ESTIMATED_TARGET_SECONDS:
                raise RuntimeError(f"throughput stop: target stage projects {projected:.0f} s")
    del target
    release_models()
    timings["target_behavior"] = time.perf_counter() - start
    gates = {"F0_replay_bitwise": replayed, "receivers": len(records), "harness_reproduced": replayed == len(records)}
    return records, summarize_d5(records, gates)


def audit(run_dir: Path) -> dict:
    """Recount the hit rates for c from saved generations, without summarize_d5."""
    results = json.loads((run_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    problems = []
    for name in CONDITIONS:
        hits = 0
        for r in results:
            generation = r["conditions"].get(name, {}).get("generation")
            text = generation["text"].rstrip() if generation and generation["terminated"] else None
            hits += text == str(r["values"]["c"])
        if abs(hits / len(results) - summary["conditions"][name]["hit_rate_c"]) > 1e-12:
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
    parser.add_argument("--audit", type=Path, help="recount a completed D5 run and exit")
    args = parser.parse_args(argv)
    if args.audit is None and None in (args.lock, args.d3_run, args.calibration_run):
        parser.error("--lock, --d3-run and --calibration-run are required unless --audit is given")
    if args.lock is not None and _family(args.lock) != "gemma3-12b":
        parser.error("D5 is frozen for Gemma-3-12B only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.audit is not None:
        print(json.dumps(audit(args.audit), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RunDirectory(args.run_dir or PROJECT / "runs" / f"editforeign-d5-gemma3-12b-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error = "FAILED", None
    try:
        records, summary = run_d5(args, run, timings)
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
