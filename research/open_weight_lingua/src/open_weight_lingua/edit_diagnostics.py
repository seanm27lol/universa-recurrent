"""Frozen diagnostics of the answer-slot edit channel and an end-of-program site.

protocols/edit_channel_diagnostics.md is the frozen design; this module
implements it and nothing else. Example: a Gemma-3-12B receiver asks for x,
whose answer is 10, and its saved description ends ``expecting "10"``. D1
rewrites that slot to the counterfactual 11 (and to controls), reconstructs
each text with the AR, patches it in exactly as the pilot's P2, and records
whether the greedy answer becomes the written number. D2 captures the same
prompt at the token that completes the program instead of the answer
position, asks the AV to describe it, and counts variable-state statements.

The pilot run is read, never written. Harness gates compare every fresh
recomputation of a pilot quantity with the saved one bitwise.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from safetensors.torch import load_file
import torch

from . import answer_slot
from .artifacts import RunDirectory, save_numeric, sha256_file, write_json
from .description_census import (
    bound_values,
    integer_answer,
    literals,
    names,
    numbers,
    program,
    program_text,
)
from .geometry import direction_metrics, restore_norm
from .metrics import next_token_kl
from .nla_adapter import Reconstructor, Verbalizer
from .runner import release_models, source_identity
from .preflight import (
    compatibility,
    inspect_metadata,
    load_model,
    model_paths,
    read_lock,
    verify_models,
)
from .stats import paired_logprob_difference
from .target import Site, Target
from .tasks import interpret
from .text_edits import RULE_SHA256 as FROZEN_RULE_SHA256
from .text_edits import parse as frozen_parse

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT / "protocols" / "edit_channel_diagnostics.md"
FORMAT = "open_weight_lingua.edit_diagnostics.v1"
SEED = 206100
RESAMPLES = 3000
CONVENTION = "rstrip"
COMPARATOR_HIT_RATE = 0.30
STATE_FLOOR_COUNT, STATE_FLOOR_RATE = 32, 0.25
D2_AV_REPLAY = 8
THROUGHPUT_FACTOR = 3.0
ESTIMATED_SECONDS = {"d1": {"gemma3-12b": 1500, "gemma3-27b": 2700}, "d2": {"gemma3-12b": 4800}}
EDITS = (("E1", "c"), ("E1g", "c"), ("E2", "o"), ("E3", "d"))


# ---------------------------------------------------------------- planning


def receivers(inputs: dict) -> list[str]:
    """The rows the frozen edit rule parses: side A, querying the affected variable."""
    return [rid for rid, row in inputs.items() if row["side"] == "A" and row["affected"]]


def counterpart(rid: str) -> str:
    return rid.replace("-A-", "-B-") if "-A-" in rid else rid.replace("-B-", "-A-")


def requested_values(inputs: dict, rid: str) -> dict:
    a = int(inputs[rid]["answer"])
    c = int(inputs[counterpart(rid)]["answer"])
    o = 2 * a - c
    return {"a": a, "c": c, "o": o if 0 <= o <= 19 else None, "d": (a + 10) % 20}


def d1_plan(inputs: dict, descriptions: dict) -> list[dict]:
    """Populations, requested values and the exact texts of every condition."""
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
                conditions["E2"] = {"value": values["o"], "text": answer_slot.edit_slot(text, values["o"])}
            conditions["E3"] = {"value": values["d"], "text": answer_slot.edit_slot(text, values["d"])}
            for condition in conditions.values():
                condition["text_sha256"] = hashlib.sha256(condition["text"].encode()).hexdigest()
        plan.append(
            {
                "id": rid,
                "group_id": inputs[rid]["group_id"],
                "slot": {"status": slot.status, "candidates": list(slot.candidates)},
                "population": population,
                "values": values,
                "conditions": conditions,
            }
        )
    return plan


def program_end_position(tokenizer, input_ids: list[int], prompt: str) -> int:
    """First position whose decoded prefix contains the whole program text."""
    text = program_text(prompt)
    for position in range(len(input_ids)):
        if text in tokenizer.decode(input_ids[: position + 1], skip_special_tokens=False):
            return position
    raise ValueError("program text not found in the decoded prompt")


# ---------------------------------------------------------------- statistics


def _answer(record: dict) -> int | None:
    return integer_answer(record.get("generation"), CONVENTION) if record.get("status") == "ok" else None


def summarize_d1(records: list[dict], *, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Frozen D1 statistics over the primary population; ITT for failed conditions."""
    executed = [r for r in records if r["population"] != "excluded"]
    gates = {
        name: sum(bool(r["gates"].get(name)) for r in executed)
        for name in ("G0_logits", "G0_greedy", "G1_direction", "G1_replacement", "G1_greedy")
    }
    reproduced = bool(executed) and all(count == len(executed) for count in gates.values())
    primary = [r for r in executed if r["population"] == "primary"]
    summary = {
        "populations": {
            name: sum(r["population"] == name for r in records)
            for name in ("primary", "secondary", "excluded")
        },
        "gates": {"executed": len(executed), "passed": gates, "harness_reproduced": reproduced},
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
        off = [
            float(_answer(r["conditions"].get(condition, {})) not in (r["values"]["a"], r["values"][key]))
            for r in rows
        ]
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
            "off_target_rate": sum(off) / len(rows),
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

    e1 = summary["edits"].get("E1", {})
    summary["readings"] = {
        "slot_edit_moves_behavior": moved("E1"),
        "number_general": moved("E1") and moved("E2") and moved("E3"),
        "comparator_hit_rate_at_least_0.30": e1.get("hit_rate", 0.0) >= COMPARATOR_HIT_RATE,
        "interpretable": reproduced,
    }
    secondary = [r for r in executed if r["population"] == "secondary"]
    summary["secondary"] = {
        condition: {
            "receivers": len([r for r in secondary if r["values"][key] is not None]),
            "hits": sum(
                _answer(r["conditions"].get(condition, {})) == r["values"][key]
                for r in secondary
                if r["values"][key] is not None
            ),
        }
        for condition, key in EDITS
    }
    return summary


def census(description: str | None, prompt: str, affected: str) -> dict:
    """One description's variable-state census (text only)."""
    text = description or ""
    state = interpret(program(prompt))
    written = literals(prompt)
    frozen = frozen_parse(text)
    out = {"available": description is not None}
    for variable in ("x", "y"):
        bound = bound_values(text, variable)
        out[variable] = {
            "frozen_status": frozen[variable].status,
            "named": names(text, variable),
            "final": state[variable],
            "computed": state[variable] not in written,
            "bound_true": state[variable] in bound,
            "bound_any": bool(bound),
            "final_anywhere": state[variable] in numbers(text),
        }
    out["affected"] = affected
    out["slot"] = answer_slot.parse(text).status
    return out


def summarize_d2(end_census: list[dict], pilot_census: list[dict]) -> dict:
    def table(rows: list[dict]) -> dict:
        affected = [r[r["affected"]] for r in rows]
        computed = [c for c in affected if c["computed"]]
        return {
            "descriptions": len(rows),
            "available": sum(r["available"] for r in rows),
            "frozen_eligible_affected": sum(c["frozen_status"] == "eligible" for c in affected),
            "frozen_any_hit_either_variable": sum(
                any(r[v]["frozen_status"] != "absent" for v in ("x", "y")) for r in rows
            ),
            "names_both_variables": sum(r["x"]["named"] and r["y"]["named"] for r in rows),
            "affected_bound_any": sum(c["bound_any"] for c in affected),
            "affected_bound_true": sum(c["bound_true"] for c in affected),
            "affected_computed": len(computed),
            "affected_computed_bound_true": sum(c["bound_true"] for c in computed),
            "affected_computed_anywhere": sum(c["final_anywhere"] for c in computed),
            "both_finals_anywhere": sum(r["x"]["final_anywhere"] and r["y"]["final_anywhere"] for r in rows),
            "slot_status": {s: sum(r["slot"] == s for r in rows) for s in answer_slot.STATUSES},
        }

    end, pilot = table(end_census), table(pilot_census)
    rate = end["affected_computed_bound_true"] / end["affected_computed"] if end["affected_computed"] else 0.0
    return {
        "end_of_program": end,
        "pilot_site": pilot,
        "readings": {
            "R2a_bound_true_at_least_32": end["affected_bound_true"] >= STATE_FLOOR_COUNT,
            "R2b_computed_bound_true_rate": rate,
            "R2b_rate_at_least_0.25": rate >= STATE_FLOOR_RATE,
            "states_variable_state": end["affected_bound_true"] >= STATE_FLOOR_COUNT and rate >= STATE_FLOOR_RATE,
        },
    }


# ---------------------------------------------------------------- execution


def _family(lock_path: Path) -> str:
    return lock_path.stem.replace("model-lock-", "")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_pilot(pilot: Path):
    manifest = json.loads((pilot / "manifest.json").read_text())
    rows = {r["id"]: r for r in json.loads((pilot / "results.json").read_text())}
    inputs = {r["id"]: r for r in manifest["inputs"]}
    descriptions = {
        rid: row["description"]["description"]
        for rid, row in rows.items()
        if row.get("description", {}).get("status") == "ok"
    }
    return manifest, rows, inputs, descriptions


def _manifest(args, part, pilot_manifest, plan, verified, report):
    return {
        "format": FORMAT,
        "part": part,
        "family": _family(args.lock),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(),
        "answer_slot_rule": {"definition": answer_slot.rule_definition(), "sha256": answer_slot.RULE_SHA256},
        "frozen_edit_rule_sha256": FROZEN_RULE_SHA256,
        "model_lock_sha256": sha256_file(args.lock),
        "pilot": {
            "run": args.pilot_run.name,
            "manifest_sha256": sha256_file(args.pilot_run / "manifest.json"),
            "results_sha256": sha256_file(args.pilot_run / "results.json"),
            "plan_hash": pilot_manifest["plan_hash"],
            "answer_convention": pilot_manifest.get("answer_convention"),
        },
        "statistics": {"resamples": RESAMPLES, "seed": SEED, "convention": CONVENTION},
        "verified_artifact_bytes": verified,
        "software": report,
        "plan": plan,
    }


def _check_throughput(part, family, started, done, total):
    if done < 4:
        return
    projected = (time.perf_counter() - started) / done * total
    if projected > THROUGHPUT_FACTOR * ESTIMATED_SECONDS[part][family]:
        raise RuntimeError(
            f"throughput stop: projected {projected:.0f} s exceeds {THROUGHPUT_FACTOR}x "
            f"the frozen estimate {ESTIMATED_SECONDS[part][family]} s"
        )


def _saved(pilot: Path, rid: str, kind: str) -> dict:
    return load_file(str(pilot / "raw" / f"{rid}-{kind}.safetensors"))


def run_d1(args, run: RunDirectory, timings: dict) -> tuple[dict, list, dict]:
    pilot_manifest, rows, inputs, descriptions = _load_pilot(args.pilot_run)
    plan = d1_plan(inputs, descriptions)
    lock = read_lock(args.lock)
    if sha256_file(args.lock) != pilot_manifest["model_lock_sha256"]:
        raise ValueError("lock differs from the pilot run's lock")
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models(lock, paths)
    timings["hash_verification"] = time.perf_counter() - start
    tokenizers, _, ar_meta, _ = inspect_metadata(paths, lock)
    manifest = _manifest(args, "d1", pilot_manifest, plan, verified, compatibility(args.device))
    write_json(run.path / "manifest.json", manifest)
    executed = [entry for entry in plan if entry["population"] != "excluded"]

    start = time.perf_counter()
    ar = Reconstructor(
        load_model(paths["ar"], "ar", args.device), tokenizers["ar"], ar_meta, paths["ar"] / "value_head.safetensors"
    )
    directions = {}
    for entry in executed:
        original = _saved(args.pilot_run, entry["id"], "original")["original"]
        for name, condition in entry["conditions"].items():
            vector = ar.reconstruct(condition["text"]).float().cpu()
            directions[(entry["id"], name)] = vector
            condition["cosine"] = direction_metrics(original, vector)["cosine"]
            save_numeric(run.path / "raw" / f"{entry['id']}-{name}-direction.safetensors", {"ar_direction": vector})
        print(f"  AR: {entry['id']}", flush=True)
    del ar
    release_models()
    timings["ar_load_and_reconstruct"] = time.perf_counter() - start

    start = time.perf_counter()
    target = Target(load_model(paths["target"], "target", args.device), tokenizers["target"])
    records, done = [], 0
    for entry in plan:
        record = {k: entry[k] for k in ("id", "group_id", "slot", "population", "values")}
        record["gates"], record["conditions"] = {}, {}
        if entry["population"] == "excluded":
            records.append(record)
            continue
        rid = entry["id"]
        saved = _saved(args.pilot_run, rid, "original")
        behavior = _saved(args.pilot_run, rid, "behavior")
        saved_direction = _saved(args.pilot_run, rid, "direction")["ar_direction"]
        original, norm = saved["original"], float(saved["retained_norm_float32"])
        layer, row, position = (int(v) for v in saved["site_int64"])
        site = Site(layer, position, row)
        ids, mask = target.tensors([inputs[rid]["input_ids"]], [inputs[rid]["attention_mask"]])
        if not torch.equal(ids.cpu(), saved["input_ids_int64"]):
            raise RuntimeError(f"{rid}: manifest input ids differ from the saved ids")
        logits = target.forward(ids, mask)
        p0 = logits[0, position].float().cpu()
        del logits
        record["gates"]["G0_logits"] = bool(torch.equal(p0, behavior["P0"]))
        record["gates"]["G0_greedy"] = target.greedy(ids, mask, site, None) == rows[rid]["conditions"]["P0"]["generation"]
        record["gates"]["G1_direction"] = bool(torch.equal(directions[(rid, "E0")], saved_direction))
        answers = sorted({str(v) for v in entry["values"].values() if v is not None})
        for name, condition in entry["conditions"].items():
            tick = time.perf_counter()
            try:
                replacement = restore_norm(directions[(rid, name)], norm, dtype=original.dtype)
                if name == "E0":
                    record["gates"]["G1_replacement"] = bool(torch.equal(replacement, behavior["P2_replacement"]))
                logits = target.forward(ids, mask, site, replacement)
                next_logits = logits[0, position].float().cpu()
                del logits
                generation = target.greedy(ids, mask, site, replacement)
                if name == "E0":
                    record["gates"]["G1_greedy"] = generation == rows[rid]["conditions"]["P2"]["generation"]
                scores = {
                    answer: target.score_answer(ids, mask, site, answer, original, replacement)["log_probability"]
                    for answer in answers
                }
            except (ValueError, RuntimeError) as failure:
                record["conditions"][name] = {"status": "failed", "error": str(failure), "value": condition["value"]}
                continue
            record["conditions"][name] = {
                "status": "ok",
                "value": condition["value"],
                "text_sha256": condition["text_sha256"],
                "cosine": condition["cosine"],
                "generation": generation,
                "answer": integer_answer(generation, CONVENTION),
                "scores": scores,
                "next_token_kl": next_token_kl(behavior["P0"], next_logits),
                "seconds": time.perf_counter() - tick,
            }
        records.append(record)
        done += 1
        print(f"  D1: {rid} {record['gates']}", flush=True)
        _check_throughput("d1", _family(args.lock), start, done, len(executed))
    timings["target_behavior"] = time.perf_counter() - start
    return manifest, records, summarize_d1(records)


def run_d2(args, run: RunDirectory, timings: dict) -> tuple[dict, list, dict]:
    pilot_manifest, rows, inputs, descriptions = _load_pilot(args.pilot_run)
    lock = read_lock(args.lock)
    if sha256_file(args.lock) != pilot_manifest["model_lock_sha256"]:
        raise ValueError("lock differs from the pilot run's lock")
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models(lock, paths)
    timings["hash_verification"] = time.perf_counter() - start
    tokenizers, av_meta, _, _ = inspect_metadata(paths, lock)
    ids_list = receivers(inputs)
    plan = []
    for rid in ids_list:
        other = rid.rsplit("-", 1)[0] + "-" + ("y" if rid.endswith("-x") else "x")
        position = program_end_position(tokenizers["target"], inputs[rid]["input_ids"], inputs[rid]["prompt"])
        plan.append(
            {
                "id": rid,
                "other_id": other,
                "end_position": position,
                "end_token": tokenizers["target"].decode([inputs[rid]["input_ids"][position]]),
            }
        )
    manifest = _manifest(args, "d2", pilot_manifest, plan, verified, compatibility(args.device))
    write_json(run.path / "manifest.json", manifest)

    start = time.perf_counter()
    target = Target(load_model(paths["target"], "target", args.device), tokenizers["target"])
    captures, records = {}, []
    for entry in plan:
        rid = entry["id"]
        saved = _saved(args.pilot_run, rid, "original")
        layer, row, position = (int(v) for v in saved["site_int64"])
        ids, mask = target.tensors([inputs[rid]["input_ids"]], [inputs[rid]["attention_mask"]])
        pilot_site = target.capture(ids, mask, Site(layer, position, row))
        end = target.capture(ids, mask, Site(layer, entry["end_position"], 0))
        other_ids, other_mask = target.tensors(
            [inputs[entry["other_id"]]["input_ids"]], [inputs[entry["other_id"]]["attention_mask"]]
        )
        other_end = target.capture(other_ids, other_mask, Site(layer, entry["end_position"], 0))
        captures[rid] = end.vector
        save_numeric(run.path / "raw" / f"{rid}-end.safetensors", {"end": end.vector})
        records.append(
            {
                "id": rid,
                "end_position": entry["end_position"],
                "end_token": entry["end_token"],
                "gates": {"G2_pilot_site_capture": bool(torch.equal(pilot_site.vector, saved["original"]))},
                "end_capture_equals_other_query": bool(torch.equal(end.vector, other_end.vector)),
                "end_norm": end.norm,
            }
        )
    del target
    release_models()
    timings["target_capture"] = time.perf_counter() - start

    start = time.perf_counter()
    av = Verbalizer(load_model(paths["av"], "av", args.device), tokenizers["av"], av_meta)
    for index, record in enumerate(records):
        rid = record["id"]
        if index < D2_AV_REPLAY:
            replay = av.verbalize(_saved(args.pilot_run, rid, "original")["original"])
            record["gates"]["G3_av_replay"] = replay.token_ids == rows[rid]["description"]["token_ids"]
        description = av.verbalize(captures[rid])
        record["description"] = asdict(description)
        print(f"  AV-end: {rid} {description.status}", flush=True)
        _check_throughput("d2", _family(args.lock), start, index + 1, len(records))
    timings["av_generate"] = time.perf_counter() - start

    end_census, pilot_census = [], []
    for record in records:
        rid = record["id"]
        affected = inputs[rid]["variable"]
        text = record["description"]["description"] if record["description"]["status"] == "ok" else None
        record["census"] = census(text, inputs[rid]["prompt"], affected)
        end_census.append(record["census"])
        pilot_census.append(census(descriptions.get(rid), inputs[rid]["prompt"], affected))
    summary = summarize_d2(end_census, pilot_census)
    summary["gates"] = {
        "G2_pilot_site_capture": sum(r["gates"]["G2_pilot_site_capture"] for r in records),
        "G3_av_replay": sum(bool(r["gates"].get("G3_av_replay")) for r in records),
        "G3_replayed": min(D2_AV_REPLAY, len(records)),
        "end_capture_equals_other_query": sum(r["end_capture_equals_other_query"] for r in records),
        "rows": len(records),
    }
    summary["gates"]["harness_reproduced"] = (
        summary["gates"]["G2_pilot_site_capture"] == len(records)
        and summary["gates"]["G3_av_replay"] == summary["gates"]["G3_replayed"]
    )
    return manifest, records, summary


# ---------------------------------------------------------------- audit


def audit(run_dir: Path) -> dict:
    """Recount D1 hits/gates or D2 coverage from saved results, without the summarizers."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    problems = []
    if manifest["part"] == "d1":
        primary = [r for r in results if r["population"] == "primary"]
        for condition, key in EDITS:
            rows = [r for r in primary if r["values"][key] is not None]
            if not rows:
                continue
            hits = 0
            for r in rows:
                generation = r["conditions"].get(condition, {}).get("generation")
                text = generation["text"].rstrip() if generation and generation["terminated"] else None
                hits += text == str(r["values"][key])
            if abs(hits / len(rows) - summary["edits"][condition]["hit_rate"]) > 1e-12:
                problems.append(f"{condition} hit rate does not recount")
    else:
        count = sum(
            frozen_parse(r["description"]["description"])[r["census"]["affected"]].status == "eligible"
            for r in results
            if r["description"]["status"] == "ok"
        )
        if count != summary["end_of_program"]["frozen_eligible_affected"]:
            problems.append("frozen-rule coverage does not recount")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "part": manifest["part"]}


# ---------------------------------------------------------------- entry point


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("d1", "d2"))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit", type=Path, help="recount a completed diagnostic run and exit")
    args = parser.parse_args(argv)
    if args.audit is None and (args.part is None or args.lock is None or args.pilot_run is None):
        parser.error("--part, --lock and --pilot-run are required unless --audit is given")
    if args.part == "d2" and args.lock is not None and _family(args.lock) != "gemma3-12b":
        parser.error("D2 is frozen for Gemma-3-12B only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.audit is not None:
        print(json.dumps(audit(args.audit), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or PROJECT / "runs" / f"editdiag-{args.part}-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}"
    run = RunDirectory(run_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error, summary = "FAILED", None, None
    try:
        runner = run_d1 if args.part == "d1" else run_d2
        _, records, summary = runner(args, run, timings)
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
                "part": args.part,
                "timings_seconds": timings,
                "wall_seconds": time.perf_counter() - started,
                "setup_seconds": float(os.environ.get("OWL_SETUP_SECONDS", "nan")),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            },
        )
        print(f"{status}: {run.path}", flush=True)


if __name__ == "__main__":
    main()
