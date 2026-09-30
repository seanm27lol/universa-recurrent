"""Frozen out-of-sample replication of the consistent answer-slot edit (D3).

protocols/edit_replication.md is the frozen design; this module implements it
and nothing else. Example: on the Gemma-3-12B pilot split (D1), rewriting only
the quoted "Final token" candidates of `y = 3`'s description to 2 left the
model answering 3, because the description also says "Result: 3" and "y = 3";
rewriting every mention of 3 flipped the answer to 2 in 25 of 64 receivers.
D3 asks whether that consistent edit moves the answer on 256 fresh groups
(the calibration split, used before only for the P4 fit), and whether writing
any other number everywhere moves the answer to that number.

Fresh AV descriptions are generated here, because the calibration stage never
ran the AV. The calibration and pilot runs are read, never written.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from safetensors.torch import load_file
import torch

from . import answer_slot
from .artifacts import RunDirectory, save_numeric, sha256_file, write_json
from .description_census import integer_answer
from .edit_diagnostics import (
    CONVENTION,
    PROJECT,
    RESAMPLES,
    SEED,
    _family,
    _git_head,
    receivers,
    requested_values,
)
from .geometry import direction_metrics, restore_norm
from .metrics import next_token_kl
from .nla_adapter import Reconstructor, Verbalizer
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

PROTOCOL = PROJECT / "protocols" / "edit_replication.md"
FORMAT = "open_weight_lingua.edit_replication.v1"
EDITS = (("E1", "c"), ("E1g", "c"), ("E2g", "o"), ("E3g", "d"))
REPLAY_RECEIVERS = 4
ESTIMATED_SECONDS = 12000
THROUGHPUT_FACTOR = 3.0
COMPARATOR_HIT_RATE = 0.30


def d3_plan(inputs: dict, descriptions: dict) -> list[dict]:
    """Populations, requested values and the exact text of every D3 condition."""
    plan = []
    for rid in receivers(inputs):
        text = descriptions.get(rid)
        slot = answer_slot.parse(text) if text is not None else answer_slot.Slot("absent", (), ())
        values = requested_values(inputs, rid)
        if slot.status != "eligible":
            population = "excluded"
        else:
            population = "primary" if slot.lead == values["a"] else "secondary"
        conditions = {}
        if population != "excluded":
            conditions["E0"] = {"value": None, "text": text}
            conditions["E1"] = {"value": values["c"], "text": answer_slot.edit_slot(text, values["c"])}
            conditions["E1g"] = {"value": values["c"], "text": answer_slot.edit_everywhere(text, values["c"])}
            if values["o"] is not None:
                conditions["E2g"] = {"value": values["o"], "text": answer_slot.edit_everywhere(text, values["o"])}
            conditions["E3g"] = {"value": values["d"], "text": answer_slot.edit_everywhere(text, values["d"])}
            for condition in conditions.values():
                condition["text_sha256"] = hashlib.sha256(condition["text"].encode()).hexdigest()
        plan.append(
            {
                "id": rid,
                "slot": {"status": slot.status, "candidates": list(slot.candidates)},
                "population": population,
                "values": values,
                "conditions": conditions,
            }
        )
    return plan


def _answer(record: dict) -> int | None:
    return integer_answer(record.get("generation"), CONVENTION) if record.get("status") == "ok" else None


def summarize_d3(records: list[dict], gates: dict, *, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Frozen D3 statistics over the primary population; ITT for failed conditions."""
    primary = [r for r in records if r["population"] == "primary"]
    summary = {
        "populations": {
            name: sum(r["population"] == name for r in records) for name in ("primary", "secondary", "excluded")
        },
        "descriptions_ok": sum(r.get("description", {}).get("status") == "ok" for r in records),
        "gates": gates,
        "edits": {},
    }
    for condition, key in EDITS:
        rows = [r for r in primary if r["values"][key] is not None]
        if not rows:
            summary["edits"][condition] = {"receivers": 0}
            continue
        hit = [float(_answer(r["conditions"].get(condition, {})) == r["values"][key]) for r in rows]
        base = [float(_answer(r["conditions"]["E0"]) == r["values"][key]) for r in rows]
        keep = [float(_answer(r["conditions"].get(condition, {})) == r["values"]["a"]) for r in rows]
        shift_edit, shift_base, excluded = [], [], 0
        for r in rows:
            edit_scores = r["conditions"].get(condition, {}).get("scores", {})
            base_scores = r["conditions"]["E0"].get("scores", {})
            wanted, own = str(r["values"][key]), str(r["values"]["a"])
            if all(k in s for s in (edit_scores, base_scores) for k in (wanted, own)):
                shift_edit.append(edit_scores[wanted] - edit_scores[own])
                shift_base.append(base_scores[wanted] - base_scores[own])
            else:
                excluded += 1
        delta = paired_logprob_difference(hit, base, resamples=resamples, seed=seed)
        entry = {
            "receivers": len(rows),
            "value": key,
            "hit_rate": sum(hit) / len(rows),
            "base_rate": sum(base) / len(rows),
            "delta": {"point": delta.point, "lower": delta.lower},
            "retained_answer_rate": sum(keep) / len(rows),
            "logprob_shift_excluded": excluded,
        }
        if shift_edit:
            shift = paired_logprob_difference(shift_edit, shift_base, resamples=resamples, seed=seed)
            entry["logprob_shift"] = {"point": shift.point, "lower": shift.lower, "receivers": len(shift_edit)}
        summary["edits"][condition] = entry

    def moved(condition: str) -> bool:
        entry = summary["edits"].get(condition, {})
        return (
            entry.get("receivers", 0) > 0
            and entry["delta"]["lower"] > 0
            and entry.get("logprob_shift", {}).get("lower", 0.0) > 0
        )

    summary["readings"] = {
        "R3a_consistent_edit_moves_behavior": moved("E1g"),
        "R3b_number_general": moved("E1g") and moved("E2g") and moved("E3g"),
        "comparator_E1g_hit_rate_at_least_0.30": summary["edits"].get("E1g", {}).get("hit_rate", 0.0)
        >= COMPARATOR_HIT_RATE,
        "interpretable": bool(gates.get("harness_reproduced")),
    }
    return summary


def _behavior(target, ids, mask, site, original, norm, direction, answers, p0_logits) -> dict:
    """One P2-style patched condition: greedy, answer scores, next-token KL."""
    replacement = restore_norm(direction, norm, dtype=original.dtype)
    logits = target.forward(ids, mask, site, replacement)
    next_logits = logits[0, site.position].float().cpu()
    del logits
    generation = target.greedy(ids, mask, site, replacement)
    scores = {
        answer: target.score_answer(ids, mask, site, answer, original, replacement)["log_probability"]
        for answer in answers
    }
    return {
        "status": "ok",
        "generation": generation,
        "answer": integer_answer(generation, CONVENTION),
        "scores": scores,
        "next_token_kl": next_token_kl(p0_logits, next_logits),
    }


def _replay_plan(d1_run: Path) -> list[dict]:
    """The first primary pilot receivers of the completed D1 run, with their saved records."""
    manifest = json.loads((d1_run / "manifest.json").read_text())
    results = {r["id"]: r for r in json.loads((d1_run / "results.json").read_text())}
    chosen = [p for p in manifest["plan"] if p["population"] == "primary"][:REPLAY_RECEIVERS]
    return [{"plan": p, "record": results[p["id"]]} for p in chosen]


def run_d3(args, run: RunDirectory, timings: dict):
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    inputs = {r["id"]: r for r in calibration["inputs"]}
    pilot_inputs = {r["id"]: r for r in json.loads((args.pilot_run / "manifest.json").read_text())["inputs"]}
    lock_sha = sha256_file(args.lock)
    if lock_sha != calibration["model_lock_sha256"]:
        raise ValueError("lock differs from the calibration run's lock")
    replay = _replay_plan(args.d1_run)
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models(lock, paths)
    timings["hash_verification"] = time.perf_counter() - start
    tokenizers, av_meta, ar_meta, _ = inspect_metadata(paths, lock)
    ids_list = receivers(inputs)
    write_json(
        run.path / "manifest.json",
        {
            "format": FORMAT,
            "family": _family(args.lock),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_head": _git_head(),
            "protocol_sha256": sha256_file(PROTOCOL),
            "source_file_sha256": source_identity(),
            "answer_slot_rule": {"definition": answer_slot.rule_definition(), "sha256": answer_slot.RULE_SHA256},
            "model_lock_sha256": lock_sha,
            "calibration": {
                "run": args.calibration_run.name,
                "manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                "plan_hash": calibration["plan_hash"],
            },
            "replay": {
                "d1_run": args.d1_run.name,
                "d1_manifest_sha256": sha256_file(args.d1_run / "manifest.json"),
                "pilot_run": args.pilot_run.name,
                "receivers": [item["plan"]["id"] for item in replay],
            },
            "receivers": ids_list,
            "statistics": {"resamples": RESAMPLES, "seed": SEED, "convention": CONVENTION},
            "verified_artifact_bytes": verified,
            "software": compatibility(args.device),
        },
    )

    # Stage 1: target. Re-capture every receiver (gate), and its unpatched P0.
    start = time.perf_counter()
    target = Target(load_model(paths["target"], "target", args.device), tokenizers["target"])
    records, originals, p0 = {}, {}, {}
    for rid in ids_list:
        saved = load_file(str(args.calibration_run / "raw" / f"{rid}-original.safetensors"))
        layer, row, position = (int(v) for v in saved["site_int64"])
        site = Site(layer, position, row)
        ids, mask = target.tensors([inputs[rid]["input_ids"]], [inputs[rid]["attention_mask"]])
        captured = target.capture(ids, mask, site)
        logits = target.forward(ids, mask)
        p0[rid] = logits[0, position].float().cpu()
        del logits
        originals[rid] = (saved["original"], float(saved["retained_norm_float32"]), site)
        records[rid] = {
            "id": rid,
            "values": requested_values(inputs, rid),
            "gates": {"capture_equals_calibration_original": bool(torch.equal(captured.vector, saved["original"]))},
            "p0_generation": target.greedy(ids, mask, site, None),
        }
        save_numeric(run.path / "raw" / f"{rid}-p0.safetensors", {"p0_next_token_logits": p0[rid]})
    del target
    release_models()
    timings["target_capture_and_p0"] = time.perf_counter() - start

    # Stage 2: AV descriptions of the saved (gate-checked) originals.
    start = time.perf_counter()
    av = Verbalizer(load_model(paths["av"], "av", args.device), tokenizers["av"], av_meta)
    descriptions = {}
    for index, rid in enumerate(ids_list):
        result = av.verbalize(originals[rid][0])
        records[rid]["description"] = asdict(result)
        if result.status == "ok":
            descriptions[rid] = result.description
        print(f"  AV: {rid} {result.status}", flush=True)
        if index + 1 >= 8:
            projected = (time.perf_counter() - start) / (index + 1) * len(ids_list)
            if projected > THROUGHPUT_FACTOR * ESTIMATED_SECONDS:
                raise RuntimeError(f"throughput stop: AV stage projects {projected:.0f} s")
    del av
    release_models()
    timings["av_generate"] = time.perf_counter() - start
    plan = d3_plan(inputs, descriptions)
    write_json(run.path / "plan.json", plan)

    # Stage 3: AR over every condition text, plus the replay texts.
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
    for item in replay:
        for name in ("E0", "E1g"):
            directions[("replay", item["plan"]["id"], name)] = ar.reconstruct(
                item["plan"]["conditions"][name]["text"]
            ).float().cpu()
    del ar
    release_models()
    timings["ar_reconstruct"] = time.perf_counter() - start

    # Stage 4: target behavior for eligible receivers, then the D1 replay.
    start = time.perf_counter()
    target = Target(load_model(paths["target"], "target", args.device), tokenizers["target"])
    out = []
    for entry in plan:
        rid = entry["id"]
        record = {**records[rid], "slot": entry["slot"], "population": entry["population"], "conditions": {}}
        if entry["population"] != "excluded":
            original, norm, site = originals[rid]
            ids, mask = target.tensors([inputs[rid]["input_ids"]], [inputs[rid]["attention_mask"]])
            answers = sorted({str(v) for v in entry["values"].values() if v is not None})
            for name, condition in entry["conditions"].items():
                try:
                    result = _behavior(target, ids, mask, site, original, norm, directions[(rid, name)], answers, p0[rid])
                except (ValueError, RuntimeError) as failure:
                    result = {"status": "failed", "error": str(failure)}
                record["conditions"][name] = {
                    **result,
                    "value": condition["value"],
                    "text_sha256": condition["text_sha256"],
                    "cosine": condition["cosine"],
                }
            print(f"  D3: {rid} {entry['population']}", flush=True)
        out.append(record)
    replay_ok = 0
    for item in replay:
        rid, saved_record = item["plan"]["id"], item["record"]
        saved = load_file(str(args.pilot_run / "raw" / f"{rid}-original.safetensors"))
        layer, row, position = (int(v) for v in saved["site_int64"])
        site = Site(layer, position, row)
        ids, mask = target.tensors([pilot_inputs[rid]["input_ids"]], [pilot_inputs[rid]["attention_mask"]])
        behavior = load_file(str(args.pilot_run / "raw" / f"{rid}-behavior.safetensors"))
        answers = sorted({str(v) for v in item["plan"]["values"].values() if v is not None})
        same = True
        for name in ("E0", "E1g"):
            result = _behavior(
                target, ids, mask, site, saved["original"], float(saved["retained_norm_float32"]),
                directions[("replay", rid, name)], answers, behavior["P0"],
            )
            kept = saved_record["conditions"][name]
            same = same and result["generation"] == kept["generation"] and result["scores"] == kept["scores"]
        replay_ok += same
    del target
    release_models()
    timings["target_behavior"] = time.perf_counter() - start

    capture_ok = sum(r["gates"]["capture_equals_calibration_original"] for r in out)
    gates = {
        "capture_equals_calibration_original": capture_ok,
        "receivers": len(out),
        "d1_replay_bitwise": replay_ok,
        "d1_replayed": len(replay),
        "harness_reproduced": capture_ok == len(out) and replay_ok == len(replay),
    }
    return out, summarize_d3(out, gates)


def audit(run_dir: Path) -> dict:
    """Recount the E1g and E1 hit rates from saved results, without summarize_d3."""
    results = json.loads((run_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    problems = []
    for condition, key in EDITS:
        rows = [r for r in results if r["population"] == "primary" and r["values"][key] is not None]
        if not rows:
            continue
        hits = 0
        for r in rows:
            generation = r["conditions"].get(condition, {}).get("generation")
            text = generation["text"].rstrip() if generation and generation["terminated"] else None
            hits += text == str(r["values"][key])
        if abs(hits / len(rows) - summary["edits"][condition]["hit_rate"]) > 1e-12:
            problems.append(f"{condition} hit rate does not recount")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--calibration-run", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--d1-run", type=Path)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit", type=Path, help="recount a completed D3 run and exit")
    args = parser.parse_args(argv)
    required = (args.lock, args.calibration_run, args.pilot_run, args.d1_run)
    if args.audit is None and any(value is None for value in required):
        parser.error("--lock, --calibration-run, --pilot-run and --d1-run are required unless --audit is given")
    if args.lock is not None and _family(args.lock) != "gemma3-12b":
        parser.error("D3 is frozen for Gemma-3-12B only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.audit is not None:
        print(json.dumps(audit(args.audit), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RunDirectory(args.run_dir or PROJECT / "runs" / f"editrep-d3-gemma3-12b-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error = "FAILED", None
    try:
        records, summary = run_d3(args, run, timings)
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
