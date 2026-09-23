"""Smoke, numerical calibration, and one pilot; locked validation is absent."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import time
import uuid
import torch

from .artifacts import RunDirectory, save_numeric, sha256_file, write_json
from .audit import (
    FROZEN_THRESHOLDS,
    MEDIAN_RECORD_FORMAT,
    MEDIAN_RECORD_NAME,
    summarize,
)
from .controls import FIT_SIDECAR_NAME, fit_pca, load_fit, reconstruct, save_fit
from .geometry import direction_metrics, restore_norm
from .metrics import answer_tokens, next_token_kl
from .nla_adapter import Verbalizer, Reconstructor
from .preflight import (
    compatibility,
    fetch_models,
    inspect_metadata,
    load_model,
    model_paths,
    read_lock,
    verify_models,
)
from .splits import DEFAULT_COUNTS, build_plan, tokenize_split, validate_plan
from . import vllm_backend
from .target import SUFFIX_DRIFT_BOUND, TARGET_BUCKET, Site, Target
from .tasks import (
    PROMPT_TEMPLATE,
    SPLIT_SEEDS,
    VARIABLES,
    generate_groups,
    interpret,
    tokenize_groups,
)
from .text_edits import (
    RULE_SHA256,
    RULE_VERSION,
    agreement as description_agreement,
    edit as apply_edit,
    parse as parse_description,
    rule_definition,
)

PROJECT = Path(__file__).resolve().parents[2]

STAGE_SEEDS = {"smoke": 201100, "calibration": 202100, "pilot": 203100}
MODEL_STAGES = (
    "target_load_and_identity",
    "av_load_and_generate",
    "ar_load_and_reconstruct",
    "target_reload_and_behavior",
)


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def stage(timings, name, device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    print(f"Stage: {name}", flush=True)
    try:
        yield
    finally:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start


def count_forwards(model, counts, key):
    counts[key] = 0

    def count(_module, _inputs):
        counts[key] += 1

    return model.model.register_forward_pre_hook(count)


def source_identity():
    files = [
        *PROJECT.glob("src/**/*.py"),
        *PROJECT.glob("scripts/*"),
        *PROJECT.glob("protocols/*.md"),
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
    ]
    return {
        str(path.relative_to(PROJECT)): sha256_file(path)
        for path in sorted(files)
        if path.is_file()
    }


def build_manifest(
    lock,
    lock_path,
    inputs,
    statistics_record,
    metadata,
    report,
    *,
    stage="smoke",
    plan=None,
    group_counts_source="brief default table",
    tokenization=None,
    fit=None,
    fit_sidecar_sha256=None,
    calibration_median=None,
    av_backend="eager",
    ar_backend="eager",
):
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    manifest = {
        "schema_version": 1,
        "stage": "engineering_smoke",
        "groups": 8,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "source_file_sha256": source_identity(),
        "model_lock": lock,
        "model_lock_sha256": sha256_file(lock_path),
        "inputs": inputs,
        "generator": statistics_record,
        "split_seed": SPLIT_SEEDS["smoke"],
        "seed": 201100,
        "target_template": PROMPT_TEMPLATE,
        "metadata": metadata,
        "software": report,
        "site": f"model.layers.{metadata['layer']} output; last non-padding assistant-prefix token",
        "backend": "local-transformers-eager-no-cache",
        "av_backend": av_backend,
        "ar_backend": ar_backend,
        "dtype": "bfloat16",
        "av_decoding": {
            "greedy": True,
            "max_new_tokens": 200,
            "descriptions_per_activation": 1,
        },
        "target_decoding": {"greedy": True, "max_new_tokens": 8},
        "identity_atol": 1e-5,
        "identity_rtol": 1e-5,
        "suffix_causality_gate": "same-length dummy-suffix bitwise equality",
        "suffix_drift_bound": SUFFIX_DRIFT_BOUND,
        "suffix_drift_bound_justification": "frozen before the pilot from 2026-09-21 GB10 smoke data (brief §4): measured maximum cross-length relative drift 3.09e-2 from BF16 kernel reduction reassociation; bound 1e-1 adds safety factor ~3 and is never relaxed after inspecting validation results",
        "target_bucket": TARGET_BUCKET,
        "target_padding_policy": "all target forwards right-padded to fixed length 128 for kernel-shape pinning on sm_121; masked pads contribute exactly zero (bitwise-proven); pinned regime established 2026-09-21 pre-pilot after the first pilot attempt's identity-gate failure; supersedes the unpadded smoke/calibration runs",
        "greedy_identity_gate": "exact greedy equality required; divergence accepted only with measured within-bound prefix drift at the divergence step (same frozen SUFFIX_DRIFT_BOUND and 2026-09-21 justification); repaired pre-pilot after the first pilot attempt failed this gate on variant pilot-0000-A-x with no conditions executed; no pilot/validation outcomes inspected",
        "norm_policy": "receiver original float32 norm retained separately (4 bytes)",
        "fixed_norm_diagnostic": "median of all 32 smoke extraction norms, frozen before reconstruction; not a scientific calibration norm",
        "shuffled_description": "next group cyclically, same variant ordinal; input-independent derangement",
        "baseline_fit_identity": None,
        "P4": "NOT IMPLEMENTED: Milestone 2 calibration-fitted PCA",
        "edits": "NOT IMPLEMENTED: Milestone 2",
        "answer_tokenization": "separate canonical integer IDs plus EOS, append without retokenizing prefix",
        "metrics": [
            "exact_answer",
            "answer_agreement",
            "KL(P0||condition)",
            "full_answer_log_probability",
            "cosine",
            "unit_direction_squared_l2",
            "norm_error",
        ],
        "limits": "No scientific success threshold assessed; single-vector intervention retains original context.",
    }
    if "vllm" in (av_backend, ar_backend):
        manifest["vllm_backend"] = {
            "worker": report.get("vllm_worker"),
            "cross_backend_equivalence": (
                "NOT CLAIMED: vllm outputs are a distinct measurement backend; "
                "equivalence is measured by scripts/check_vllm_equivalence.sh, "
                "and a gate FAIL forbids mixing vllm outputs into eager-regime evidence"
            ),
        }
    if stage == "smoke":
        return manifest
    manifest.update(
        stage={"calibration": "numerical_calibration", "pilot": "pilot"}[stage],
        groups=plan.counts[stage],
        plan_hash=plan.plan_hash,
        split_counts=dict(plan.counts),
        group_counts_source=group_counts_source,
        split_seed=SPLIT_SEEDS[stage],
        seed=STAGE_SEEDS[stage],
        generator={
            "plan_stats": plan.stats[stage],
            "tokenization": tokenization,
        },
        frozen_thresholds=FROZEN_THRESHOLDS,
    )
    if stage == "calibration":
        manifest.update(
            fixed_norm_diagnostic="this stage fits the scientific calibration median over all successful extraction norms; frozen in calibration_median_norm.json",
            shuffled_description="NOT USED this stage: calibration extracts activations only",
            P4="this stage fits the P4 baseline: PCA on pooled unlabeled calibration unit directions; identity recorded in completion.json and baseline_fit.json",
            edits="NOT RUN this stage: edit coverage is measured in the pilot",
            limits="Numerical calibration only; no reconstruction, patching, behavior, or edit coverage is measured in this stage.",
        )
        return manifest
    manifest.update(
        fixed_norm_diagnostic=f"frozen calibration median norm from the calibration stage (baseline fit {fit.identity}); replaces the Milestone 1 smoke-local median",
        norm_policy="receiver original float32 norm retained separately (4 bytes); the calibration_median_norm diagnostic applies the frozen calibration median recorded in this manifest",
        P4="PCA fitted on pooled unlabeled calibration unit directions; rank = min(fitted_rank, floor(text_bytes/2)) with float16 coefficients (no-more-than-budget); receiver norm restored at patch time",
        edits="frozen rule (version/sha256 recorded): edit the affected variable's stated current value to the group's counterfactual value; same-delta wrong-variable control when eligible; absent/ambiguous descriptions get a status, never a fabricated replacement",
        edit_rule={
            "version": RULE_VERSION,
            "sha256": RULE_SHA256,
            "definition": rule_definition(),
        },
        baseline_fit_identity=fit.identity,
        baseline_fit_sidecar_sha256=fit_sidecar_sha256,
        calibration_median_norm=calibration_median,
        metrics=[
            *manifest["metrics"],
            "paired_accuracy_loss_points",
            "paired_logprob_difference",
            "editing_effect",
            "pilot_block_estimates",
        ],
        limits="One bounded pilot; pooled estimates only; keep/stop derives only from the frozen thresholds; validation is never auto-started.",
    )
    return manifest


def check_target_bucket_fit(inputs, *, bucket=TARGET_BUCKET, generation_ceiling=8):
    """Loud pre-inference guarantee for kernel-shape pinning.

    Every prompt plus its worst-case suffix (the greedy generation ceiling or
    the answer suffix, whichever is longer) must fit the pinned bucket. The
    bucket length B never changes silently; an over-bucket input stops the run
    before any model forward.
    """
    for row in inputs:
        prompt = len(row["input_ids"])
        suffix = len(row.get("answer_token_ids_including_eos") or ())
        worst = max(generation_ceiling, suffix)
        if prompt + worst > bucket:
            raise ValueError(
                f"{row['id']}: prompt length {prompt} plus worst-case suffix "
                f"{worst} exceeds the pinned target bucket {bucket}"
            )


def _av_stage(run, rows, activations, descriptions, *, backend, paths, tokenizers, av_meta, device, timings, calls):
    """One AV pass over every row; eager in-process or the vLLM subprocess."""
    with stage(timings, "av_load_and_generate", device):
        if backend == "vllm":
            records, completion = vllm_backend.verbalize_batch(
                av_path=paths["av"],
                av_meta=av_meta,
                tokenizer=tokenizers["av"],
                vectors={
                    row["id"]: activations[row["id"]].vector for row in rows
                },
            )
            write_json(run.path / "av_vllm_worker.json", completion)
            calls["av"] = completion["model_forward_calls"]
            for row in rows:
                row["description"] = records[row["id"]]
                row["av_seconds"] = completion["rows"][row["id"]]["seconds"]
                if records[row["id"]]["status"] == "ok":
                    descriptions[row["id"]] = records[row["id"]]["description"]
                print(
                    f"  AV: {row['id']} {records[row['id']]['status']}", flush=True
                )
            return
        av = Verbalizer(
            load_model(paths["av"], "av", device), tokenizers["av"], av_meta
        )
        handle = count_forwards(av.model, calls, "av")
        for row in rows:
            start = time.perf_counter()
            try:
                result = av.verbalize(activations[row["id"]].vector)
                row["description"] = asdict(result)
                if result.status == "ok":
                    descriptions[row["id"]] = result.description
            except (ValueError, RuntimeError) as error:
                row["description"] = {"status": "failed", "error": str(error)}
            row["av_seconds"] = time.perf_counter() - start
            print(f"  AV: {row['id']} {row['description']['status']}", flush=True)
        handle.remove()
        del av
        release_models()


def _ar_reconstruction_row(row, vector, activations, run):
    directions_entry = {
        "status": "ok",
        **direction_metrics(activations[row["id"]].vector, vector),
    }
    save_numeric(
        run.path / "raw" / f"{row['id']}-direction.safetensors",
        {"ar_direction": vector},
    )
    return directions_entry


def _ar_stage(run, rows, activations, descriptions, directions, *, backend, paths, tokenizers, ar_meta, device, timings, calls, edit_plans=None, edited_directions=None, wrong_directions=None):
    """One AR pass over descriptions (plus pilot edit texts); eager or vLLM."""
    with stage(timings, "ar_load_and_reconstruct", device):
        if backend == "vllm":
            items = [
                (row["id"], descriptions[row["id"]])
                for row in rows
                if row["id"] in descriptions
            ]
            edit_items = []
            for group_id, plan in (edit_plans or {}).items():
                if plan.get("edited_description") is not None:
                    edit_items.append((f"{group_id}#edited", plan["edited_description"]))
                wrong_text = plan["wrong_variable"].get("edited_description")
                if wrong_text is not None:
                    edit_items.append((f"{group_id}#wrong_variable", wrong_text))
            if items or edit_items:
                vectors, completion = vllm_backend.reconstruct_batch(
                    ar_path=paths["ar"],
                    ar_meta=ar_meta,
                    tokenizer=tokenizers["ar"],
                    head_path=paths["ar"] / "value_head.safetensors",
                    items=[*items, *edit_items],
                )
                write_json(run.path / "ar_vllm_worker.json", completion)
                calls["ar"] = completion["model_forward_calls"]
                row_stats = completion["rows"]
            else:
                vectors, row_stats = {}, {}
            edit_lookup = dict(edit_items)
            for row in rows:
                if row["id"] not in descriptions:
                    row["reconstruction"] = {
                        "status": "skipped",
                        "reason": "AV description unavailable",
                    }
                    continue
                stats = row_stats.get(row["id"], {})
                if row["id"] in vectors:
                    directions[row["id"]] = vectors[row["id"]]
                    row["reconstruction"] = _ar_reconstruction_row(
                        row, vectors[row["id"]], activations, run
                    )
                else:
                    row["reconstruction"] = {
                        "status": "failed",
                        "error": stats.get("error", "worker returned no direction"),
                    }
                row["ar_seconds"] = stats.get("seconds")
            for group_id, plan in (edit_plans or {}).items():
                receiver_id = plan["receiver_id"]
                for label, store in (
                    ("edited", edited_directions),
                    ("wrong_variable", wrong_directions),
                ):
                    key = f"{group_id}#{label}"
                    if key not in edit_lookup:
                        continue
                    stats = row_stats.get(key, {})
                    if key in vectors:
                        store[group_id] = vectors[key]
                        plan[f"{label}_reconstruction"] = {
                            "status": "ok",
                            **direction_metrics(
                                activations[receiver_id].vector, vectors[key]
                            ),
                        }
                        save_numeric(
                            run.path
                            / "raw"
                            / f"{receiver_id}-{label}-direction.safetensors",
                            {"ar_direction": vectors[key]},
                        )
                    else:
                        plan[f"{label}_reconstruction"] = {
                            "status": "failed",
                            "error": stats.get("error", "worker returned no direction"),
                        }
                    plan[f"{label}_reconstruction"]["seconds"] = stats.get("seconds")
            return
        ar = Reconstructor(
            load_model(paths["ar"], "ar", device),
            tokenizers["ar"],
            ar_meta,
            paths["ar"] / "value_head.safetensors",
        )
        handle = count_forwards(ar.model, calls, "ar")
        for row in rows:
            if row["id"] not in descriptions:
                row["reconstruction"] = {
                    "status": "skipped",
                    "reason": "AV description unavailable",
                }
                continue
            start = time.perf_counter()
            try:
                vector = ar.reconstruct(descriptions[row["id"]])
                directions[row["id"]] = vector
                row["reconstruction"] = _ar_reconstruction_row(
                    row, vector, activations, run
                )
            except (ValueError, RuntimeError) as error:
                row["reconstruction"] = {"status": "failed", "error": str(error)}
            row["ar_seconds"] = time.perf_counter() - start
        for group_id, edit_plan in (edit_plans or {}).items():
            receiver_id = edit_plan["receiver_id"]
            attempts = []
            if edit_plan.get("edited_description") is not None:
                attempts.append(
                    ("edited", edit_plan["edited_description"], edited_directions)
                )
            wrong_text = edit_plan["wrong_variable"].get("edited_description")
            if wrong_text is not None:
                attempts.append(
                    ("wrong_variable", wrong_text, wrong_directions)
                )
            for label, text, store in attempts:
                start = time.perf_counter()
                try:
                    vector = ar.reconstruct(text)
                    store[group_id] = vector
                    edit_plan[f"{label}_reconstruction"] = {
                        "status": "ok",
                        **direction_metrics(
                            activations[receiver_id].vector, vector
                        ),
                    }
                    save_numeric(
                        run.path
                        / "raw"
                        / f"{receiver_id}-{label}-direction.safetensors",
                        {"ar_direction": vector},
                    )
                except (ValueError, RuntimeError) as error:
                    edit_plan[f"{label}_reconstruction"] = {
                        "status": "failed",
                        "error": str(error),
                    }
                edit_plan[f"{label}_reconstruction"]["seconds"] = (
                    time.perf_counter() - start
                )
        handle.remove()
        del ar
        release_models()


def execute_smoke(
    run, paths, tokenizers, av_meta, ar_meta, inputs, device, timings, calls, backends=None
):
    backends = backends or {"av": "eager", "ar": "eager"}
    rows = [
        {"id": row["id"], "group_id": row["group_id"], "conditions": {}}
        for row in inputs
    ]
    activations, descriptions, directions = {}, {}, {}
    try:
        with stage(timings, "target_load_and_identity", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = count_forwards(target.model, calls, "target_identity")
            for input_row, row in zip(inputs, rows):
                row["attempted"] = True
                ids, mask = target.tensors(
                    [input_row["input_ids"]], [input_row["attention_mask"]]
                )
                record, gates, generation = target.identity_gate(
                    ids, mask, Site(av_meta.layer, input_row["position"])
                )
                activations[row["id"]] = record
                row.update(
                    identity=gates,
                    baseline_generation=generation,
                    native_dtype=str(record.vector.dtype),
                    original_norm=record.norm,
                    hook_path=record.hook_path,
                    hidden_state_index=record.hidden_state_index,
                )
                save_numeric(
                    run.path / "raw" / f"{row['id']}-original.safetensors",
                    {
                        "original": record.vector,
                        "retained_norm_float32": torch.tensor(
                            record.norm, dtype=torch.float32
                        ),
                        "site_int64": torch.tensor(
                            [record.site.layer, record.site.row, record.site.position],
                            dtype=torch.int64,
                        ),
                        "input_ids_int64": ids.cpu(),
                        "attention_mask_int64": mask.cpu(),
                    },
                )
                print(f"  Identity: {row['id']}", flush=True)
            handle.remove()
            del target
            release_models()
        median_norm = statistics.median(record.norm for record in activations.values())
        _av_stage(
            run,
            rows,
            activations,
            descriptions,
            backend=backends["av"],
            paths=paths,
            tokenizers=tokenizers,
            av_meta=av_meta,
            device=device,
            timings=timings,
            calls=calls,
        )
        _ar_stage(
            run,
            rows,
            activations,
            descriptions,
            directions,
            backend=backends["ar"],
            paths=paths,
            tokenizers=tokenizers,
            ar_meta=ar_meta,
            device=device,
            timings=timings,
            calls=calls,
        )
        with stage(timings, "target_reload_and_behavior", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = count_forwards(target.model, calls, "target_behavior")
            random = torch.Generator(device="cpu").manual_seed(201100)
            for index, (input_row, row) in enumerate(zip(inputs, rows)):
                record = activations[row["id"]]
                ids, mask = target.tensors(
                    [input_row["input_ids"]], [input_row["attention_mask"]]
                )
                # Within-group A/B counterpart at the same query, independent of its true answer.
                donor_index = index + 2 if input_row["side"] == "A" else index - 2
                donor = inputs[donor_index]
                shuffled_id = inputs[(index + 4) % len(inputs)]["id"]

                def patch(vector, norm=record.norm):
                    return restore_norm(vector, norm, dtype=record.vector.dtype)

                replacements = {
                    "P0": None,
                    "P1": record.vector,
                    "P2": patch(directions[row["id"]])
                    if row["id"] in directions
                    else None,
                    "P3": patch(directions[shuffled_id])
                    if shuffled_id in directions
                    else None,
                    "P5": patch(torch.randn(record.vector.shape, generator=random)),
                    "donor": patch(activations[donor["id"]].vector),
                    "smoke_median_norm": patch(directions[row["id"]], median_norm)
                    if row["id"] in directions
                    else None,
                }
                evidence = {}
                for condition, replacement in replacements.items():
                    if replacement is None and condition != "P0":
                        row["conditions"][condition] = {
                            "status": "failed",
                            "reason": "required reconstruction unavailable",
                        }
                        continue
                    start = time.perf_counter()
                    logits = target.forward(
                        ids,
                        mask,
                        record.site if replacement is not None else None,
                        replacement,
                    )
                    next_logits = logits[0, record.site.position].float().cpu()
                    del logits
                    evidence[condition] = next_logits
                    if condition == "P1" and not torch.allclose(
                        evidence["P0"], next_logits, atol=1e-5, rtol=1e-5
                    ):
                        raise RuntimeError(
                            "raw restoration changed logits after target reload"
                        )
                    generation = target.greedy(ids, mask, record.site, replacement)
                    scores = {
                        answer: target.score_answer(
                            ids, mask, record.site, answer, record.vector, replacement
                        )
                        for answer in sorted({input_row["answer"], donor["answer"]})
                    }
                    row["conditions"][condition] = {
                        "status": "ok",
                        "generation": generation,
                        "next_token_kl": next_token_kl(evidence["P0"], next_logits),
                        "answer_scores": scores,
                        "norm_error": float(
                            torch.linalg.vector_norm(replacement.float())
                        )
                        - record.norm
                        if replacement is not None
                        else 0.0,
                        "seconds": time.perf_counter() - start,
                    }
                    if replacement is not None:
                        evidence[f"{condition}_replacement"] = replacement
                if row["conditions"]["P0"]["generation"] != row["baseline_generation"]:
                    raise RuntimeError(
                        "target reload changed unmodified greedy generation"
                    )
                # P0 (unpatched) versus P1 (pinned original) greedy trajectories
                # are the same pinned-shape computation; "exact" is the
                # expectation and drift_diverged is retained backstop evidence.
                row["p0_p1_greedy_gate"] = target.greedy_identity(
                    ids,
                    mask,
                    record.site,
                    record.vector,
                    row["conditions"]["P0"]["generation"],
                    row["conditions"]["P1"]["generation"],
                )
                row["controls"] = {
                    "donor_id": donor["id"],
                    "shuffled_description_id": shuffled_id,
                    "smoke_median_norm": median_norm,
                }
                row["payload_bytes"] = {
                    "description_utf8": len(descriptions[row["id"]].encode("utf-8"))
                    if row["id"] in descriptions
                    else None,
                    "retained_norm_float32": 4,
                    "site_three_int64": 24,
                    "native_selected_vector": record.vector.numel()
                    * record.vector.element_size(),
                    "retained_input_ids_int64": ids.numel() * 8,
                    "retained_attention_mask_int64": mask.numel() * 8,
                    "target_bucket_padding": "retained ids/mask keep logical length; every target forward was right-padded to the manifest target_bucket for kernel-shape pinning",
                    "shared_metadata_and_weights": "see manifest/model_lock and inventory; not included in text bytes",
                }
                save_numeric(
                    run.path / "raw" / f"{row['id']}-behavior.safetensors", evidence
                )
                print(f"  Behavior: {row['id']}", flush=True)
            handle.remove()
            del target
            release_models()
    finally:
        # Even a failed identity gate or backend leaves partial evidence/accounting.
        write_json(run.path / "results.json", rows)
    return rows


def _group_counts(raw):
    """Dev/test override for the split table; recorded in the manifest."""
    if raw is None:
        return None
    try:
        counts = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --group-counts-json: {exc}") from exc
    if not isinstance(counts, dict) or any(
        type(value) is not int or value < 1 for value in counts.values()
    ):
        raise ValueError("--group-counts-json must map split names to positive counts")
    return counts


def load_calibration_inputs(path):
    """Load the pinned PCA fit and its frozen calibration median record.

    Accepts the calibration run directory, a directory holding the fit, or a
    direct path to baseline_fit.safetensors / baseline_fit.json.
    """
    directory = Path(path)
    if directory.is_file() or directory.suffix:
        directory = directory.parent
    fit = load_fit(directory)
    sidecar_path = directory / MEDIAN_RECORD_NAME
    try:
        record = json.loads(sidecar_path.read_text())
    except OSError as exc:
        raise ValueError(
            f"missing calibration median record next to the fit: {exc}"
        ) from exc
    if record.get("format") != MEDIAN_RECORD_FORMAT:
        raise ValueError("unsupported calibration median record")
    median = record.get("median_norm")
    if (
        isinstance(median, bool)
        or not isinstance(median, (int, float))
        or not math.isfinite(median)
        or median <= 0
    ):
        raise ValueError("invalid calibration median norm")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("plan_hash"))):
        raise ValueError("calibration median record lacks a plan hash")
    return fit, record


def _edit_plan(group, description):
    """One group's frozen-rule edit plan; never fabricates a statement.

    The receiver is the source-side row querying the affected variable. The
    wrong-variable control applies the same signed delta the affected edit
    applied to the text, negated when the 0..19 range requires it;
    out-of-range controls are reported, not forced.
    """
    receiver_id = f"{group.group_id}-A-{group.affected}"
    other = next(variable for variable in VARIABLES if variable != group.affected)
    plan = {"receiver_id": receiver_id, "wrong_variable": {"variable": other}}
    if description is None:
        return {
            **plan,
            "status": "description_unavailable",
            "affected_status": None,
        }
    parsed = parse_description(description)
    status = parsed[group.affected].status
    plan.update(
        status=status,
        affected_status=status,
        parse={
            variable: {
                "status": parsed[variable].status,
                "value": parsed[variable].value,
                "detail": parsed[variable].detail,
            }
            for variable in VARIABLES
        },
        agreement=description_agreement(description, group.source),
    )
    if status != "eligible":
        return plan
    counterfactual_value = interpret(group.counterfactual)[group.affected]
    edited = apply_edit(description, group.affected, counterfactual_value)
    plan.update(
        requested_value=counterfactual_value,
        original_value=edited.original_value,
        value_span=list(edited.value_span),
        edited_description=edited.description,
    )
    plan["wrong_variable"]["status"] = parsed[other].status
    if parsed[other].status != "eligible":
        return plan
    delta = counterfactual_value - edited.original_value
    candidate = parsed[other].value + delta
    if not 0 <= candidate <= 19:
        candidate = parsed[other].value - delta
    if not 0 <= candidate <= 19:
        plan["wrong_variable"]["status"] = "out_of_range"
        return plan
    wrong = apply_edit(description, other, candidate)
    plan["wrong_variable"].update(
        requested_value=candidate,
        original_value=wrong.original_value,
        value_span=list(wrong.value_span),
        edited_description=wrong.description,
    )
    return plan


def execute_calibration(
    run, paths, tokenizers, av_meta, plan, inputs, device, timings, calls
):
    """Target-only extraction over the calibration split, then the frozen fit.

    No reconstruction, patching, or behavior is measured. The PCA fit and the
    frozen median norm are written at the run root so the pilot can load them
    with --calibration-fit.
    """
    rows = [
        {"id": row["id"], "group_id": row["group_id"], "conditions": {}}
        for row in inputs
    ]
    activations = {}
    fit, median_record = None, None
    try:
        with stage(timings, "target_load_and_extract", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = count_forwards(target.model, calls, "target_calibration")
            for input_row, row in zip(inputs, rows):
                row["attempted"] = True
                start = time.perf_counter()
                try:
                    ids, mask = target.tensors(
                        [input_row["input_ids"]], [input_row["attention_mask"]]
                    )
                    record = target.capture(
                        ids, mask, Site(av_meta.layer, input_row["position"])
                    )
                except (ValueError, RuntimeError) as error:
                    row["extraction"] = {"status": "failed", "error": str(error)}
                    print(f"  Extract: {row['id']} failed", flush=True)
                    continue
                activations[row["id"]] = record
                row.update(
                    extraction={"status": "ok"},
                    capture_seconds=time.perf_counter() - start,
                    native_dtype=str(record.vector.dtype),
                    original_norm=record.norm,
                    hook_path=record.hook_path,
                    hidden_state_index=record.hidden_state_index,
                )
                save_numeric(
                    run.path / "raw" / f"{row['id']}-original.safetensors",
                    {
                        "original": record.vector,
                        "retained_norm_float32": torch.tensor(
                            record.norm, dtype=torch.float32
                        ),
                        "site_int64": torch.tensor(
                            [record.site.layer, record.site.row, record.site.position],
                            dtype=torch.int64,
                        ),
                        "input_ids_int64": ids.cpu(),
                        "attention_mask_int64": mask.cpu(),
                    },
                )
            handle.remove()
            del target
            release_models()
        if not activations:
            raise RuntimeError(
                "no calibration activations captured; cannot fit the baseline"
            )
        with stage(timings, "fit_baseline", "cpu"):
            fit = fit_pca(
                [
                    activations[row["id"]].vector
                    for row in rows
                    if row["id"] in activations
                ]
            )
            median = statistics.median(
                record.norm for record in activations.values()
            )
            save_fit(fit, run.path)
            median_record = {
                "format": MEDIAN_RECORD_FORMAT,
                "median_norm": median,
                "extractions": len(activations),
                "planned_prompt_variants": len(inputs),
                "plan_hash": plan.plan_hash,
                "counts": dict(plan.counts),
                "policy": "median of all successful calibration extraction norms, "
                "frozen before any pilot reconstruction; never selected from "
                "validation outcomes",
            }
            write_json(run.path / MEDIAN_RECORD_NAME, median_record)
            print(
                f"  Baseline fit: rank {fit.fitted_rank}, "
                f"median norm {median:.6g}, {len(activations)} extractions",
                flush=True,
            )
    finally:
        # Partial extraction evidence and accounting survive a failed stage.
        write_json(run.path / "results.json", rows)
    return rows, fit, median_record


def execute_pilot(
    run,
    paths,
    tokenizers,
    av_meta,
    ar_meta,
    inputs,
    groups,
    fit,
    calibration_median,
    device,
    timings,
    calls,
    backends=None,
):
    """One pilot over all conditions; same site/hook path and gates as smoke."""
    backends = backends or {"av": "eager", "ar": "eager"}
    rows = [
        {"id": row["id"], "group_id": row["group_id"], "conditions": {}}
        for row in inputs
    ]
    activations, descriptions, directions = {}, {}, {}
    edited_directions, wrong_directions = {}, {}
    try:
        with stage(timings, "target_load_and_identity", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = count_forwards(target.model, calls, "target_identity")
            for input_row, row in zip(inputs, rows):
                row["attempted"] = True
                ids, mask = target.tensors(
                    [input_row["input_ids"]], [input_row["attention_mask"]]
                )
                start = time.perf_counter()
                record, gates, generation = target.identity_gate(
                    ids, mask, Site(av_meta.layer, input_row["position"])
                )
                row["identity_seconds"] = time.perf_counter() - start
                activations[row["id"]] = record
                row.update(
                    identity=gates,
                    baseline_generation=generation,
                    native_dtype=str(record.vector.dtype),
                    original_norm=record.norm,
                    hook_path=record.hook_path,
                    hidden_state_index=record.hidden_state_index,
                )
                save_numeric(
                    run.path / "raw" / f"{row['id']}-original.safetensors",
                    {
                        "original": record.vector,
                        "retained_norm_float32": torch.tensor(
                            record.norm, dtype=torch.float32
                        ),
                        "site_int64": torch.tensor(
                            [record.site.layer, record.site.row, record.site.position],
                            dtype=torch.int64,
                        ),
                        "input_ids_int64": ids.cpu(),
                        "attention_mask_int64": mask.cpu(),
                    },
                )
                print(f"  Identity: {row['id']}", flush=True)
            handle.remove()
            del target
            release_models()
        _av_stage(
            run,
            rows,
            activations,
            descriptions,
            backend=backends["av"],
            paths=paths,
            tokenizers=tokenizers,
            av_meta=av_meta,
            device=device,
            timings=timings,
            calls=calls,
        )
        edit_plans = {
            group.group_id: _edit_plan(
                group,
                descriptions.get(f"{group.group_id}-A-{group.affected}"),
            )
            for group in groups.values()
        }
        _ar_stage(
            run,
            rows,
            activations,
            descriptions,
            directions,
            backend=backends["ar"],
            paths=paths,
            tokenizers=tokenizers,
            ar_meta=ar_meta,
            device=device,
            timings=timings,
            calls=calls,
            edit_plans=edit_plans,
            edited_directions=edited_directions,
            wrong_directions=wrong_directions,
        )
        with stage(timings, "target_reload_and_behavior", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = count_forwards(target.model, calls, "target_behavior")
            random = torch.Generator(device="cpu").manual_seed(STAGE_SEEDS["pilot"])
            for index, (input_row, row) in enumerate(zip(inputs, rows)):
                record = activations[row["id"]]
                ids, mask = target.tensors(
                    [input_row["input_ids"]], [input_row["attention_mask"]]
                )
                # Within-group A/B counterpart at the same query, independent of its true answer.
                donor_index = index + 2 if input_row["side"] == "A" else index - 2
                donor = inputs[donor_index]
                shuffled_id = inputs[(index + 4) % len(inputs)]["id"]

                def patch(vector, norm=record.norm):
                    return restore_norm(vector, norm, dtype=record.vector.dtype)

                p4_direction, p4_accounting = None, None
                if row["id"] in descriptions:
                    try:
                        p4_direction, p4_accounting = reconstruct(
                            fit,
                            record.vector,
                            text_bytes=len(
                                descriptions[row["id"]].encode("utf-8")
                            ),
                        )
                    except ValueError:
                        p4_direction = None
                replacements = {
                    "P0": None,
                    "P1": record.vector,
                    "P2": patch(directions[row["id"]])
                    if row["id"] in directions
                    else None,
                    "P3": patch(directions[shuffled_id])
                    if shuffled_id in directions
                    else None,
                    "P4": patch(p4_direction)
                    if p4_direction is not None
                    else None,
                    "P5": patch(torch.randn(record.vector.shape, generator=random)),
                    "donor": patch(activations[donor["id"]].vector),
                    "calibration_median_norm": patch(
                        directions[row["id"]], calibration_median
                    )
                    if row["id"] in directions
                    else None,
                }
                edit_plan = edit_plans[row["group_id"]]
                if input_row["side"] == "A" and edit_plan["status"] == "eligible":
                    replacements["edited"] = (
                        patch(edited_directions[row["group_id"]])
                        if row["group_id"] in edited_directions
                        else None
                    )
                    if (
                        edit_plan["wrong_variable"].get("edited_description")
                        is not None
                    ):
                        replacements["wrong_variable_edit"] = (
                            patch(wrong_directions[row["group_id"]])
                            if row["group_id"] in wrong_directions
                            else None
                        )
                evidence = {}
                for condition, replacement in replacements.items():
                    if replacement is None and condition != "P0":
                        row["conditions"][condition] = {
                            "status": "failed",
                            "reason": "required reconstruction unavailable",
                        }
                        continue
                    start = time.perf_counter()
                    logits = target.forward(
                        ids,
                        mask,
                        record.site if replacement is not None else None,
                        replacement,
                    )
                    next_logits = logits[0, record.site.position].float().cpu()
                    del logits
                    evidence[condition] = next_logits
                    if condition == "P1" and not torch.allclose(
                        evidence["P0"], next_logits, atol=1e-5, rtol=1e-5
                    ):
                        raise RuntimeError(
                            "raw restoration changed logits after target reload"
                        )
                    generation = target.greedy(ids, mask, record.site, replacement)
                    scores = {
                        answer: target.score_answer(
                            ids, mask, record.site, answer, record.vector, replacement
                        )
                        for answer in sorted({input_row["answer"], donor["answer"]})
                    }
                    row["conditions"][condition] = {
                        "status": "ok",
                        "generation": generation,
                        "next_token_kl": next_token_kl(evidence["P0"], next_logits),
                        "answer_scores": scores,
                        "norm_error": float(
                            torch.linalg.vector_norm(replacement.float())
                        )
                        - record.norm
                        if replacement is not None
                        else 0.0,
                        "seconds": time.perf_counter() - start,
                    }
                    if condition == "P4" and p4_accounting is not None:
                        row["conditions"]["P4"]["accounting"] = p4_accounting
                    if replacement is not None:
                        evidence[f"{condition}_replacement"] = replacement
                if row["conditions"]["P0"]["generation"] != row["baseline_generation"]:
                    raise RuntimeError(
                        "target reload changed unmodified greedy generation"
                    )
                # P0 (unpatched) versus P1 (pinned original) greedy trajectories
                # are the same pinned-shape computation; "exact" is the
                # expectation and drift_diverged is retained backstop evidence.
                row["p0_p1_greedy_gate"] = target.greedy_identity(
                    ids,
                    mask,
                    record.site,
                    record.vector,
                    row["conditions"]["P0"]["generation"],
                    row["conditions"]["P1"]["generation"],
                )
                row["controls"] = {
                    "donor_id": donor["id"],
                    "shuffled_description_id": shuffled_id,
                    "calibration_median_norm": calibration_median,
                }
                if input_row["side"] == "A" and input_row["affected"]:
                    row["edit"] = edit_plan
                elif input_row["side"] == "A":
                    row["edit"] = {
                        "status": edit_plan["status"],
                        "receiver_id": edit_plan["receiver_id"],
                    }
                row["payload_bytes"] = {
                    "description_utf8": len(descriptions[row["id"]].encode("utf-8"))
                    if row["id"] in descriptions
                    else None,
                    "p4_coefficients_float16": p4_accounting["coefficient_bytes"]
                    if p4_accounting is not None
                    else None,
                    "edited_description_utf8": len(
                        edit_plan["edited_description"].encode("utf-8")
                    )
                    if input_row["side"] == "A"
                    and edit_plan.get("edited_description") is not None
                    else None,
                    "retained_norm_float32": 4,
                    "site_three_int64": 24,
                    "native_selected_vector": record.vector.numel()
                    * record.vector.element_size(),
                    "retained_input_ids_int64": ids.numel() * 8,
                    "retained_attention_mask_int64": mask.numel() * 8,
                    "target_bucket_padding": "retained ids/mask keep logical length; every target forward was right-padded to the manifest target_bucket for kernel-shape pinning",
                    "shared_metadata_and_weights": "see manifest/model_lock and inventory; not included in text bytes",
                }
                save_numeric(
                    run.path / "raw" / f"{row['id']}-behavior.safetensors", evidence
                )
                print(f"  Behavior: {row['id']}", flush=True)
            handle.remove()
            del target
            release_models()
    finally:
        # Even a failed identity gate or backend leaves partial evidence/accounting.
        write_json(run.path / "results.json", rows)
    return rows


def pilot_cost_projection(timings, calls, rows, pilot_groups):
    """Linear per-group scaling to 512 validation groups plus calibration.

    An estimate reported for a scope decision; it never starts validation.
    """
    measured = {name: timings.get(name, 0.0) for name in MODEL_STAGES}
    total = sum(measured.values())
    per_group = total / pilot_groups if pilot_groups else None
    identity_seconds = [
        row["identity_seconds"] for row in rows if row.get("identity_seconds")
    ]
    mean_identity = (
        statistics.fmean(identity_seconds) if identity_seconds else None
    )
    forward_calls = sum(calls.values())
    projected_validation = per_group * 512 if per_group is not None else None
    projected_calibration = (
        mean_identity * 4 * DEFAULT_COUNTS["calibration"]
        if mean_identity is not None
        else None
    )
    projected_total = (
        projected_validation + projected_calibration
        if projected_validation is not None and projected_calibration is not None
        else None
    )
    return {
        "measured_pilot_groups": pilot_groups,
        "measured_stage_seconds": measured,
        "model_forward_calls": dict(calls),
        "per_group_seconds": per_group,
        "mean_identity_seconds_per_variant": mean_identity,
        "projected_validation_seconds_512_groups": projected_validation,
        "projected_calibration_seconds_256_groups": projected_calibration,
        "projected_validation_forward_calls": (
            forward_calls / pilot_groups * 512 if pilot_groups else None
        ),
        "projected_total_seconds": projected_total,
        "budget_seconds": FROZEN_THRESHOLDS["stage_budget_seconds"],
        "within_budget": (
            projected_total <= FROZEN_THRESHOLDS["stage_budget_seconds"]
            if projected_total is not None
            else None
        ),
        "basis": "linear scaling of measured pilot per-group stage times to 512 "
        "validation groups, plus 256 calibration groups estimated from "
        "per-variant identity/capture times; an estimate, not a duration promise",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "calibration", "pilot"),
        default="smoke",
        help="smoke is the Milestone 1 engineering gate; locked validation is not implemented",
    )
    parser.add_argument(
        "--lock", type=Path, default=PROJECT / "configs/model-lock.json"
    )
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--av-path", type=Path)
    parser.add_argument("--ar-path", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--calibration-fit",
        type=Path,
        help="baseline fit directory from a completed calibration run (required for --stage pilot)",
    )
    parser.add_argument(
        "--group-counts-json",
        help=argparse.SUPPRESS,  # dev/test split-table override; recorded in the manifest
    )
    parser.add_argument(
        "--fetch-models",
        action="store_true",
        help="explicitly download only the locked model artifacts",
    )
    parser.add_argument(
        "--av-backend",
        choices=("eager", "vllm"),
        default="eager",
        help="AV description backend; vllm is an opt-in distinct measurement backend "
        "(requires the .venv-vllm worker; no vllm/eager equivalence is claimed)",
    )
    parser.add_argument(
        "--ar-backend",
        choices=("eager", "vllm"),
        default="eager",
        help="AR reconstruction backend; same opt-in vllm constraints as --av-backend",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "pilot" and args.calibration_fit is None:
        parser.error(
            "--stage pilot requires --calibration-fit from a completed calibration run"
        )
    if args.stage != "pilot" and args.calibration_fit is not None:
        parser.error("--calibration-fit is only valid with --stage pilot")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir is None:
        args.run_dir = (
            PROJECT
            / "runs"
            / (
                datetime.now(timezone.utc).strftime(f"{args.stage}-%Y%m%dT%H%M%SZ-")
                + uuid.uuid4().hex[:8]
            )
        )
    run = RunDirectory(args.run_dir)
    started, timings, calls = time.perf_counter(), {}, {}
    backends = {"av": args.av_backend, "ar": args.ar_backend}
    status, error, report, manifest = "FAILED", None, {}, None
    plan, fit, median_record, rows = None, None, None, []
    try:
        with stage(timings, "compatibility", "cpu"):
            report = compatibility(args.device)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(STAGE_SEEDS[args.stage])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["tf32"] = False
            report["deterministic_algorithms"] = (
                torch.are_deterministic_algorithms_enabled()
            )
            if "vllm" in backends.values():
                if not args.device.startswith("cuda"):
                    raise ValueError("the vllm backend requires the cuda device")
                report["vllm_worker"] = vllm_backend.probe_worker()
        lock = read_lock(args.lock)
        paths = model_paths(
            lock,
            args.cache,
            {role: getattr(args, f"{role}_path") for role in ("target", "av", "ar")},
        )
        with stage(timings, "fetch_and_hash_verification", "cpu"):
            if args.fetch_models:
                report["fetch"] = fetch_models(lock, paths)
            else:
                report["fetch"] = {"new_artifact_bytes": 0, "requested": False}
            report["verified_artifact_bytes"] = verify_models(lock, paths)
        with stage(timings, "metadata_and_task_preprocessing", "cpu"):
            tokenizers, av_meta, ar_meta, metadata = inspect_metadata(paths)
            if args.stage == "smoke":
                groups, generation_stats = generate_groups("smoke", 8)
                inputs = tokenize_groups(groups, tokenizers["target"])
                for row in inputs:
                    row["answer_token_ids_including_eos"] = answer_tokens(
                        tokenizers["target"], row["answer"]
                    )
                manifest = build_manifest(
                    lock, args.lock, inputs, generation_stats, metadata, report,
                    av_backend=backends["av"], ar_backend=backends["ar"],
                )
            else:
                plan = build_plan(_group_counts(args.group_counts_json))
                validate_plan(plan)
                if args.stage == "pilot":
                    fit, median_record = load_calibration_inputs(
                        args.calibration_fit
                    )
                    if median_record["plan_hash"] != plan.plan_hash:
                        raise ValueError(
                            "calibration artifacts come from a different split plan"
                        )
                tokenized = tokenize_split(plan, args.stage, tokenizers["target"])
                inputs = tokenized["rows"]
                if args.stage == "pilot":
                    for row in inputs:
                        row["answer_token_ids_including_eos"] = answer_tokens(
                            tokenizers["target"], row["answer"]
                        )
                manifest = build_manifest(
                    lock,
                    args.lock,
                    inputs,
                    None,
                    metadata,
                    report,
                    stage=args.stage,
                    plan=plan,
                    group_counts_source="dev override --group-counts-json"
                    if args.group_counts_json
                    else "brief default table",
                    tokenization={
                        key: tokenized[key]
                        for key in (
                            "accepted",
                            "rejected",
                            "rejections",
                            "prior_excluded_prompts",
                        )
                    },
                    fit=fit,
                    fit_sidecar_sha256=sha256_file(
                        Path(args.calibration_fit) / FIT_SIDECAR_NAME
                    )
                    if fit is not None
                    else None,
                    calibration_median=median_record["median_norm"]
                    if median_record
                    else None,
                    av_backend=backends["av"],
                    ar_backend=backends["ar"],
                )
            write_json(
                run.path / "manifest.json", manifest
            )  # before the first model forward
            # Loud pre-inference shape guarantee; B never changes silently.
            check_target_bucket_fit(inputs)
        if args.stage == "smoke":
            rows = execute_smoke(
                run,
                paths,
                tokenizers,
                av_meta,
                ar_meta,
                inputs,
                args.device,
                timings,
                calls,
                backends,
            )
        elif args.stage == "calibration":
            rows, fit, median_record = execute_calibration(
                run,
                paths,
                tokenizers,
                av_meta,
                plan,
                inputs,
                args.device,
                timings,
                calls,
            )
        else:
            rows = execute_pilot(
                run,
                paths,
                tokenizers,
                av_meta,
                ar_meta,
                inputs,
                {group.group_id: group for group in plan.groups["pilot"]},
                fit,
                median_record["median_norm"],
                args.device,
                timings,
                calls,
                backends,
            )
        summary = summarize(manifest, rows)
        status = (
            "COMPLETE"
            if not summary["failed_groups"] and not summary["skipped_groups"]
            else "COMPLETE_WITH_FAILURES"
        )
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}".replace(str(Path.home()), "<home>")
        print(error, flush=True)
    if manifest is not None:
        rows = (
            json.loads((run.path / "results.json").read_text())
            if (run.path / "results.json").exists()
            else []
        )
        write_json(run.path / "summary.json", summarize(manifest, rows))
    report["timings_seconds"] = timings
    report["model_forward_calls"] = calls
    report["process_max_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    report["setup_and_cpu_tests_seconds"] = (
        float(os.environ["OWL_SETUP_SECONDS"])
        if "OWL_SETUP_SECONDS" in os.environ
        else None
    )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(run.path / "compatibility.json", report)
    summary_record = (
        json.loads((run.path / "summary.json").read_text())
        if (run.path / "summary.json").exists()
        else None
    )
    if args.stage == "smoke":
        result = {
            "status": status,
            "error": error,
            "stage": "engineering_smoke",
            "planned_groups": 8,
            "real_model_checks": "NOT RUN" if manifest is None else status,
            "scientific_pilot": "NOT RUN",
            "locked_validation": "NOT RUN",
        }
    elif args.stage == "calibration":
        result = {
            "status": status,
            "error": error,
            "stage": "numerical_calibration",
            "planned_groups": manifest["groups"]
            if manifest is not None
            else DEFAULT_COUNTS["calibration"],
            "real_model_checks": "NOT RUN" if manifest is None else status,
            "scientific_pilot": "NOT RUN",
            "locked_validation": "NOT RUN",
            "plan_hash": manifest["plan_hash"] if manifest is not None else None,
            "baseline_fit_identity": fit.identity if fit is not None else None,
            "calibration_median_norm": median_record["median_norm"]
            if median_record
            else None,
        }
    else:
        result = {
            "status": status,
            "error": error,
            "stage": "pilot",
            "planned_groups": manifest["groups"]
            if manifest is not None
            else DEFAULT_COUNTS["pilot"],
            "real_model_checks": "NOT RUN" if manifest is None else status,
            "scientific_pilot": "NOT RUN" if manifest is None else status,
            "locked_validation": "NOT RUN",
            "plan_hash": manifest["plan_hash"] if manifest is not None else None,
            "baseline_fit_identity": manifest["baseline_fit_identity"]
            if manifest is not None
            else None,
            "decision": summary_record["decision"] if summary_record else None,
            "statistics": summary_record["statistics"] if summary_record else None,
            "edit_eligibility": summary_record["edit_eligibility"]
            if summary_record
            else None,
            "projected_validation_cost": pilot_cost_projection(
                timings,
                calls,
                rows,
                summary_record["groups"] if summary_record else 0,
            ),
        }
    write_json(run.path / "completion.json", result)
    with (run.path / "report.md").open("x") as stream:
        if args.stage == "smoke":
            stream.write(
                f"# Phase Two engineering smoke\n\nStatus: **{status}**. Eight groups, 32 prompt variants planned.\n\n"
            )
            if error:
                stream.write(f"Failure: {error}\n\n")
            if manifest is not None:
                summary = json.loads((run.path / "summary.json").read_text())
                stream.write(
                    f"Groups: {len(summary['successful_groups'])} successful, {len(summary['failed_groups'])} failed, {len(summary['skipped_groups'])} skipped.\n\n"
                )
                stream.write(
                    "| Condition | Exact answers / all variants | Valid KL mean |\n|---|---:|---:|\n"
                )
                for condition, measured in summary["metrics"].items():
                    stream.write(
                        f"| {condition} | {measured['exact_answers']} / 32 | {measured['mean_valid_next_token_kl']} |\n"
                    )
            stream.write(
                "\nNo scientific pilot or validation was run. P4/PCA and text-edit coverage are Milestone 2 work. The median-norm smoke diagnostic is fitted to smoke inputs; scientific calibration is not done.\n\n"
            )
            stream.write(
                "The AR reconstructs direction; original norm is a separately retained four-byte channel. The original prompt and all other vectors remain available. This is not whole-state compression, recurrent-depth testing, semantic proof, or a speedup.\n\n"
            )
            stream.write(
                "See summary.json for all denominators, results.json for every variant, compatibility.json for setup/stage costs and calls, and inventory.json for retained versus packaged evidence. Raw vectors/full-vocabulary logits stay local; the ZIP cannot independently replay KL.\n"
            )
        elif args.stage == "calibration":
            stream.write(
                f"# Phase Two numerical calibration\n\nStatus: **{status}**. {result['planned_groups']} groups, {4 * result['planned_groups']} prompt variants planned.\n\n"
            )
            if error:
                stream.write(f"Failure: {error}\n\n")
            if summary_record is not None:
                measured = summary_record["calibration"]
                stream.write(
                    f"Extractions: {measured['extractions']} successful, {measured['failed_extractions']} failed, {measured['unattempted_variants']} unattempted.\n\n"
                )
                stream.write(
                    f"Frozen calibration median norm: {measured['median_norm']}.\n\n"
                )
                stream.write(
                    f"Baseline fit identity: `{result['baseline_fit_identity']}` (rank and shared bytes in baseline_fit.json).\n\n"
                )
            stream.write(
                "This stage loads only the target, extracts calibration activations, fits the unlabeled PCA baseline, and freezes the median norm. No reconstruction, patching, behavior, or edit coverage is measured here; the pilot consumes baseline_fit.safetensors and calibration_median_norm.json.\n\n"
            )
            stream.write(
                "See summary.json for all denominators, results.json for every variant, compatibility.json for setup/stage costs and forward calls, and inventory.json for retained versus packaged evidence.\n"
            )
        else:
            stream.write(
                f"# Phase Two pilot\n\nStatus: **{status}**. {result['planned_groups']} groups, {4 * result['planned_groups']} prompt variants planned.\n\n"
            )
            if error:
                stream.write(f"Failure: {error}\n\n")
            if summary_record is not None:
                summary = summary_record
                stream.write(
                    f"Groups: {len(summary['successful_groups'])} successful, {len(summary['failed_groups'])} failed, {len(summary['skipped_groups'])} skipped; {summary['uneditable_groups']} not edit-eligible.\n\n"
                )
                decision = summary["decision"]
                criteria = decision["criteria"]
                stream.write(
                    "## Pilot decision\n\nDerived only from the frozen thresholds quoted in manifest.json (phase_two_brief.md Sections 7-9 design choices).\n\n"
                )
                stream.write("| Criterion | Value | Rule | Met |\n|---|---:|---|:--:|\n")
                accuracy = criteria["unmodified_accuracy"]
                accuracy_value = accuracy["value"]
                stream.write(
                    f"| Unmodified exact-answer accuracy | {f'{accuracy_value:.4f}' if accuracy_value is not None else 'n/a'} | >= {accuracy['floor']} | {accuracy['met']} |\n"
                )
                sensitivity = criteria["intervention_sensitivity"]
                stream.write(
                    f"| Donor/perturbation sensitivity | {'present' if sensitivity['present'] else 'absent'} | required | {sensitivity['met']} |\n"
                )
                loss = criteria["p2_accuracy_loss"]
                upper = loss["upper_points"]
                stream.write(
                    f"| P2 accuracy loss, one-sided 95% upper | {f'{upper:.3f} pp' if upper is not None else 'not computable'} | <= {loss['limit_points']} pp | {loss['met']} |\n"
                )
                contrast = criteria["p2_minus_p3_logprob"]
                lower = contrast["lower"]
                stream.write(
                    f"| P2 - P3 correct-answer log prob, one-sided 95% lower | {f'{lower:.4f}' if lower is not None else 'not computable'} | > 0 | {contrast['met']} |\n"
                )
                eligible = criteria["edit_eligible_groups"]
                stream.write(
                    f"| Edit-eligible groups | {eligible['value']} | >= {eligible['floor']} | {eligible['met']} |\n\n"
                )
                stream.write(
                    f"Recommendation: **{decision['recommendation']}**. Unmet criteria: {', '.join(decision['unmet_criteria']) or 'none'}. Validation is never auto-started.\n\n"
                )
                projection = result["projected_validation_cost"]
                projected = projection["projected_total_seconds"]
                stream.write(
                    f"Projected validation cost: {f'{projected:.0f}' if projected is not None else 'n/a'} s against the {projection['budget_seconds']} s budget (within budget: {projection['within_budget']}). Linear extrapolation of measured pilot stage times to 512 validation groups plus 256 calibration groups; an estimate for a scope decision, not a duration promise.\n\n"
                )
                coverage = summary["edit_eligibility"]
                stream.write(
                    f"Edit coverage: {coverage['eligible_groups']} of {coverage['groups']} groups eligible ({coverage['eligible_fraction']:.3f}); exclusion reasons {coverage['exclusion_reasons']}.\n\n"
                )
                editing = summary["statistics"]["editing_effect"]
                if "effect" in editing:
                    effect = editing["effect"]
                    stream.write(
                        f"Editing effect on counterfactual preference: {effect['point']:.4f} [{effect['lower']:.4f}, {effect['upper']:.4f}] over {editing['records_used']} records.\n\n"
                    )
                else:
                    stream.write(
                        f"Editing effect: untested ({editing['reason']}).\n\n"
                    )
            stream.write(
                "P4 compares language with a generic PCA direction under a no-more-than-budget rule; the calibration_median_norm diagnostic uses the frozen calibration median, not a smoke-local median. Text edits are conditional on the frozen parser; absent or ambiguous descriptions are reported, never replaced. This is not whole-state compression, semantic proof, or a speedup.\n\n"
            )
            stream.write(
                "See summary.json for all denominators and the statistics battery, results.json for every variant, compatibility.json for setup/stage costs and calls, and inventory.json for retained versus packaged evidence. Raw vectors/full-vocabulary logits stay local; the ZIP cannot independently replay KL.\n"
            )
    # Bundle time and final report-writing are included in the wall-clock record,
    # saved outside the ZIP to avoid a self-referential inventory/hash cycle.
    run.reports_zip()
    write_json(
        run.path / "wall_clock.json",
        {
            "runner_through_bundle_seconds": time.perf_counter() - started,
            "setup_seconds": report["setup_and_cpu_tests_seconds"],
            "not_in_reports_zip": True,
        },
    )
    print(f"{status}: {run.path}", flush=True)
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
