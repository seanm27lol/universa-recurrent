"""Reconstructed-difference steering assay on the reused pilot split.

Post-Phase-Two measurement, frozen in protocols/steering_assay_brief.md on
2026-09-22 before any steering run: does a *difference* between two language
descriptions produce targeted behavioral change when applied as the NLA
paper's norm-scaled steering patch h' = h + alpha * ||h|| * delta/||delta||
(Fraser-Taliente, Kantamneni, Ong et al. 2026, "Planning in Poetry")? This is
a steering measurement, not a preservation claim and not semantic read-out.
Arm 1 reuses the pinned pilot run's saved AR direction vectors (read-only via
--pilot-run; the AV and AR models are not loaded for it); arm 2 pushes frozen
oracle templates through the live AR. Oracle text is an intervention
instrument written by us, never claimed as read-out semantics. The pilot
split is reused deliberately; the closed pilot's recorded outcomes stand, and
nothing here amends them.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import uuid

import torch

from .artifacts import RunDirectory, load_numeric, save_numeric, sha256_file, write_json
from .geometry import unit_direction
from .metrics import answer_tokens, next_token_kl
from .nla_adapter import Reconstructor
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
from .target import SUFFIX_DRIFT_BOUND, TARGET_BUCKET, Site, Target
from .tasks import PROMPT_TEMPLATE, SPLIT_SEEDS, VARIABLES

PROJECT = Path(__file__).resolve().parents[2]

STEERING_STAGE = "steering"
STEERING_SEED = 205100
"""Fixed torch seed for the steering stage; distinct from every Phase Two seed."""

ALPHA_GRID = (-1.0, 0.5, 1.0, 2.0)
"""Frozen steering strengths; alpha = 0 is implied by the P0 reference row."""

EVALUATION_ALPHA = 1.0
"""The frozen primary strength at which all gating criteria are evaluated."""

TEMPLATES = {
    "terse": "The current value of {variable} is {value}.",
    "structured": (
        "Structured math format with code block showing a short sequence of "
        "variable assignments and arithmetic on the variables x and y, ending "
        "with a question asking for the value of {variable}.\n\n"
        "Tracing the assignments in order, the value of {variable} is {value} "
        'when the question is asked, so the expected reply is the single '
        'integer "{value}".\n\n'
        'Final token of the prompt closes the question "What is {variable}?", '
        'strongly expecting "{value}" to complete the answer.'
    ),
}
"""Frozen oracle intervention texts (protocol §3); instruments, not read-outs.

The structured style mimics the three-paragraph "Structured ..." register the
pilot's saved AV descriptions all used (reports/pilot_description_analysis.md:
a structural opening, a speculative answer paragraph, a "Final token" close),
while asserting the counterfactual value the AV never asserted.
"""

ORACLE_STYLES = tuple(TEMPLATES)

STEERING_CONDITIONS = (
    "av_diff",
    "av_diff_control",
    "oracle_terse",
    "oracle_structured",
    "wrongvar_terse",
    "wrongvar_structured",
)
"""Per-row steered conditions: two arms plus their matched controls."""

STEERING_ARMS = {
    "av_difference": {
        "condition": "av_diff",
        "control": "av_diff_control",
        "source": "pinned pilot run's saved AR directions of the group's own AV "
        "descriptions (affected-query A/B rows); no AV or AR model call",
        "control_source": "same construction on the same-value (unaffected) "
        "query rows, whose answers do not change between A and B",
    },
    "oracle_terse": {
        "condition": "oracle_terse",
        "control": "wrongvar_terse",
        "source": "live AR on the frozen terse oracle template asserting the "
        "affected variable's A and B values",
        "control_source": "same template asserting the unaffected variable's "
        "value and the value shifted by the affected variable's signed delta",
    },
    "oracle_structured": {
        "condition": "oracle_structured",
        "control": "wrongvar_structured",
        "source": "live AR on the frozen structured oracle template asserting "
        "the affected variable's A and B values",
        "control_source": "same template asserting the unaffected variable's "
        "value and the value shifted by the affected variable's signed delta",
    },
}

STEERING_FROZEN = {
    "alpha_grid": list(ALPHA_GRID),
    "alpha_zero": "implied by the in-run P0 reference row; never separately executed",
    "evaluation_alpha": EVALUATION_ALPHA,
    "flip_to_B_threshold": 0.30,
    "y_integrity_threshold": 0.90,
    "control_moved_threshold": 0.10,
    "bootstrap_resamples": 3000,
    "bootstrap_seed": 205100,
    "success_rule": "an arm shows targeted steering iff C1 flip_to_B rate >= "
    "0.30 and C2 y_intact rate >= 0.90 and C3 the arm's own control-delta "
    "moved_x rate < 0.10, all over all groups (intention-to-test) at the "
    "evaluation alpha; C4 (sign-flip flip rate at alpha=-1 <= alpha=1) is "
    "supporting evidence only, never a gate",
    "declared_in": "protocols/steering_assay_brief.md section 5, frozen "
    "2026-09-22 before any steering outcome was read",
    "status": "design choices, not paper claims; never weakened after reading outcomes",
}

GENERATION_CEILING = 8
"""Same target greedy ceiling as the pinned pilot."""


def template_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_alpha(alpha: float) -> str:
    return "%.6g" % alpha


def condition_key(condition: str, alpha: float) -> str:
    return f"{condition}@{format_alpha(alpha)}"


EXPECTED_ROW_CONDITIONS = ("P0", "P1") + tuple(
    condition_key(condition, alpha)
    for condition in STEERING_CONDITIONS
    for alpha in ALPHA_GRID
)
"""Gate conditions plus every steered condition on the frozen grid: 26 total."""


def render_template(style: str, variable: str, value: int) -> str:
    """Render one frozen oracle template; the only substitutions ever made."""
    if style not in TEMPLATES:
        raise ValueError("unknown oracle template style")
    if variable not in VARIABLES:
        raise ValueError("unknown variable")
    if type(value) is not int or not 0 <= value <= 19:
        raise ValueError("value must be a canonical integer in 0..19")
    return TEMPLATES[style].format(variable=variable, value=value)


def template_records() -> dict:
    """Verbatim templates plus content hashes, for the locked manifest."""
    return {
        style: {"text": text, "sha256": template_sha256(text)}
        for style, text in TEMPLATES.items()
    }


def description_delta(steered: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """delta = AR(d_steered) - AR(d_orig), the paper's edit direction."""
    if steered.shape != original.shape or steered.ndim != 1:
        raise ValueError("delta operands must be same-width one-dimensional vectors")
    delta = steered.float() - original.float()
    if not torch.isfinite(delta).all():
        raise ValueError("nonfinite description difference")
    return delta


def compose_patch(
    vector: torch.Tensor,
    retained_norm: float,
    delta: torch.Tensor,
    alpha: float,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """h' = h + alpha * n * u(delta); float32 math, one cast to native dtype.

    Additive steering: the patched norm is not restored to n. A degenerate
    (near-zero or nonfinite) delta fails closed via unit_direction.
    """
    if not math.isfinite(retained_norm) or retained_norm <= 1e-12:
        raise ValueError("retained norm must be finite and positive")
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    base = vector.float()
    direction = unit_direction(delta)
    if direction.shape != base.shape:
        raise ValueError("delta width does not match the receiver vector")
    result = (base + float(alpha) * retained_norm * direction).to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError("steering patch overflows the target dtype")
    return result


def wrong_variable_candidate(other_value: int, delta_value: int) -> int | None:
    """The unaffected variable's value shifted by the affected delta.

    Mirrors the pilot's frozen wrong-variable convention: apply the same
    signed delta, negate it when the direct candidate leaves 0..19, and report
    (never force) a control that cannot stay in range.
    """
    for candidate in (other_value + delta_value, other_value - delta_value):
        if candidate != other_value and 0 <= candidate <= 19:
            return candidate
    return None


@dataclass(frozen=True)
class PilotArtifacts:
    """Read-only view of the pinned pilot run; nothing is copied or amended."""

    path_sha256: str
    manifest_sha256: str
    plan_hash: str
    completion_status: str
    directions: dict  # row id -> saved AR direction (float32), or unavailable
    originals: dict  # row id -> saved native original vector, or None
    p0_generations: dict  # row id -> saved P0 generation record, or None
    direction_row_status: dict  # row id -> "ok" or the pilot's failure status


def load_pilot_run(
    path,
    *,
    plan_hash: str,
    width: int,
    direction_rows,
    original_rows,
    p0_rows,
) -> PilotArtifacts:
    """Load the pinned pilot run's saved evidence; fail loudly on a mismatch.

    The pilot run is accessed read-only at run time. Its stage, plan hash and
    completion status must match this assay's rebuilt plan exactly: the assay
    consumes the pinned pilot split, never an arbitrary directory.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"pilot run directory not found: {path}")
    try:
        pilot_manifest = json.loads((path / "manifest.json").read_text())
        completion = json.loads((path / "completion.json").read_text())
        results = json.loads((path / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable pilot run evidence: {exc}") from exc
    if pilot_manifest.get("stage") != "pilot":
        raise ValueError("the --pilot-run directory is not a completed pilot run")
    if pilot_manifest.get("plan_hash") != plan_hash:
        raise ValueError(
            "pilot run plan hash mismatch: the steering assay consumes exactly "
            "the pinned pilot split"
        )
    if completion.get("status") != "COMPLETE":
        raise ValueError(
            f"pilot run completion status is {completion.get('status')!r}; "
            "the steering assay requires a COMPLETE pilot"
        )
    pilot_rows = {row["id"]: row for row in results}
    directions, originals, p0_generations, row_status = {}, {}, {}, {}
    for row_id in direction_rows:
        record = pilot_rows.get(row_id, {})
        status = record.get("reconstruction", {}).get("status")
        if status != "ok":
            row_status[row_id] = status or "missing_from_pilot_results"
            continue
        direction_path = path / "raw" / f"{row_id}-direction.safetensors"
        if not direction_path.is_file():
            raise ValueError(
                f"pilot results record {row_id} as reconstructed but the saved "
                "direction file is absent; the pinned pilot input is corrupt"
            )
        values = load_numeric(direction_path)
        if set(values) != {"ar_direction"}:
            raise ValueError(f"unexpected tensors in {direction_path.name}")
        direction = values["ar_direction"]
        if direction.shape != (width,) or direction.dtype != torch.float32:
            raise ValueError(f"saved pilot direction {row_id} has wrong shape/dtype")
        unit_direction(direction)  # fail closed on a degenerate saved direction
        directions[row_id] = direction
        row_status[row_id] = "ok"
    for row_id in original_rows:
        original_path = path / "raw" / f"{row_id}-original.safetensors"
        originals[row_id] = None
        if original_path.is_file():
            values = load_numeric(original_path)
            original = values.get("original")
            if original is not None and original.shape == (width,):
                originals[row_id] = original
    for row_id in p0_rows:
        generation = (
            pilot_rows.get(row_id, {}).get("conditions", {}).get("P0", {})
        ).get("generation")
        p0_generations[row_id] = generation
    return PilotArtifacts(
        path_sha256=hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest(),
        manifest_sha256=sha256_file(path / "manifest.json"),
        plan_hash=pilot_manifest["plan_hash"],
        completion_status=completion["status"],
        directions=directions,
        originals=originals,
        p0_generations=p0_generations,
        direction_row_status=row_status,
    )


def steering_inputs(tokenized_rows, tokenizer) -> list:
    """Keep the side-A receiver rows; attach counterfactual answers/controls.

    Per group: the affected-query row (patched and asked about the affected
    variable) and the integrity row (same patch, asked about the unaffected
    variable). The counterfactual answer is the group's B-side answer to the
    same question; for the unaffected variable it equals the A answer by
    construction.
    """
    by_group = {}
    for row in tokenized_rows:
        by_group.setdefault(row["group_id"], []).append(row)
    executed = []
    for group_id, group_rows in by_group.items():
        side_a = [row for row in group_rows if row["side"] == "A"]
        affected_row = next(row for row in side_a if row["affected"])
        other_row = next(row for row in side_a if not row["affected"])
        b_affected = next(
            row
            for row in group_rows
            if row["side"] == "B" and row["variable"] == affected_row["variable"]
        )
        b_other = next(
            row
            for row in group_rows
            if row["side"] == "B" and row["variable"] == other_row["variable"]
        )
        delta_value = int(b_affected["answer"]) - int(affected_row["answer"])
        candidate = wrong_variable_candidate(int(other_row["answer"]), delta_value)
        for row, b_row, is_affected in (
            (affected_row, b_affected, True),
            (other_row, b_other, False),
        ):
            out = dict(row)
            out["counterfactual_answer"] = b_row["answer"]
            out["role"] = "affected_query" if is_affected else "integrity_query"
            out["wrong_variable_candidate"] = None if is_affected else candidate
            scored = (
                {out["answer"], out["counterfactual_answer"]}
                if is_affected
                else {out["answer"]}
            )
            if not is_affected and candidate is not None:
                scored.add(str(candidate))
            out["scored_answers"] = sorted(scored)
            out["answer_token_ids_including_eos"] = answer_tokens(
                tokenizer, out["answer"]
            )
            out["scored_answer_token_ids"] = {
                answer: answer_tokens(tokenizer, answer) for answer in scored
            }
            executed.append(out)
    return executed


def check_steering_bucket_fit(inputs, *, bucket=TARGET_BUCKET) -> None:
    """Loud pre-inference shape guarantee; mirrors runner.check_target_bucket_fit.

    Every prompt plus its worst-case suffix (the greedy ceiling or the longest
    scored answer suffix, including wrong-variable candidates) must fit the
    pinned bucket; the bucket never changes silently.
    """
    for row in inputs:
        prompt = len(row["input_ids"])
        suffix = max(len(ids) for ids in row["scored_answer_token_ids"].values())
        worst = max(GENERATION_CEILING, suffix)
        if prompt + worst > bucket:
            raise ValueError(
                f"{row['id']}: prompt length {prompt} plus worst-case suffix "
                f"{worst} exceeds the pinned target bucket {bucket}"
            )


def _release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _stage(timings, name, device):
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


def _count_forwards(model, counts, key):
    counts[key] = 0

    def count(_module, _inputs):
        counts[key] += 1

    return model.model.register_forward_pre_hook(count)


def _source_identity():
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


def _group_map(inputs):
    """Per group: the affected-query and integrity input rows."""
    groups = {}
    for row in inputs:
        slot = "affected" if row["affected"] else "other"
        groups.setdefault(row["group_id"], {})[slot] = row
    for group_id, pair in groups.items():
        if set(pair) != {"affected", "other"}:
            raise ValueError(f"{group_id}: expected exactly the two receiver rows")
    return groups


def _cosine_to_saved_difference(delta, reference):
    if reference is None:
        return None
    try:
        return float(unit_direction(delta) @ unit_direction(reference))
    except ValueError:
        return None


def execute_steering(
    run,
    paths,
    tokenizers,
    av_meta,
    ar_meta,
    inputs,
    pilot: PilotArtifacts,
    device,
    timings,
    calls,
):
    """One steering run over both arms and all controls on the α grid.

    Same site/hook path and gates as the pilot: identity gates and a captured
    original per receiver row, then the behavior stage with P0/P1 references
    and every steered condition at every frozen alpha. Failures are recorded
    per condition and count against the arm (intention-to-test).
    """
    rows = [
        {"id": row["id"], "group_id": row["group_id"], "conditions": {}}
        for row in inputs
    ]
    groups = _group_map(inputs)
    activations = {}
    deltas, group_evidence = {}, {}
    try:
        with _stage(timings, "target_load_and_identity", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = _count_forwards(target.model, calls, "target_identity")
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
                saved_original = pilot.originals.get(row["id"])
                row.update(
                    identity=gates,
                    baseline_generation=generation,
                    native_dtype=str(record.vector.dtype),
                    original_norm=record.norm,
                    hook_path=record.hook_path,
                    hidden_state_index=record.hidden_state_index,
                )
                row["cross_run"] = {
                    "original_vector_bitwise_equal_pilot": (
                        bool(torch.equal(record.vector, saved_original))
                        if saved_original is not None
                        else None
                    )
                }
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
            _release_models()
        with _stage(timings, "ar_load_and_reconstruct", device):
            ar = Reconstructor(
                load_model(paths["ar"], "ar", device),
                tokenizers["ar"],
                ar_meta,
                paths["ar"] / "value_head.safetensors",
            )
            handle = _count_forwards(ar.model, calls, "ar")
            for group_id, pair in groups.items():
                affected, other = pair["affected"]["variable"], pair["other"]["variable"]
                answers = {
                    "A": int(pair["affected"]["answer"]),
                    "B": int(pair["affected"]["counterfactual_answer"]),
                }
                delta_value = answers["B"] - answers["A"]
                other_value = int(pair["other"]["answer"])
                candidate = pair["other"]["wrong_variable_candidate"]
                reference = None
                saved_a = pilot.originals.get(f"{group_id}-A-{affected}")
                saved_b = pilot.originals.get(f"{group_id}-B-{affected}")
                if saved_a is not None and saved_b is not None:
                    reference = saved_b.float() - saved_a.float()
                evidence = {
                    "affected": affected,
                    "other": other,
                    "affected_answers": {
                        "A": pair["affected"]["answer"],
                        "B": pair["affected"]["counterfactual_answer"],
                    },
                    "delta_value": delta_value,
                    "wrong_variable_candidate": {
                        "value": candidate,
                        "status": "ok" if candidate is not None else "out_of_range",
                    },
                    "cosine_reference": "unit direction of the pinned pilot's "
                    "saved (B-affected minus A-affected) original vectors; "
                    "descriptive geometry only, never a gate",
                    "deltas": {},
                }

                def record_delta(condition, steered, original, **detail):
                    try:
                        delta = description_delta(steered, original)
                        unit_direction(delta)  # fail closed on a degenerate delta
                    except (ValueError, RuntimeError) as error:
                        evidence["deltas"][condition] = {
                            "status": "failed",
                            "reason": str(error),
                        }
                        return
                    deltas[(group_id, condition)] = delta
                    evidence["deltas"][condition] = {
                        "status": "ok",
                        "delta_norm": float(torch.linalg.vector_norm(delta)),
                        "cosine_with_saved_counterfactual_difference": (
                            _cosine_to_saved_difference(delta, reference)
                        ),
                        **detail,
                    }
                    save_numeric(
                        run.path
                        / "raw"
                        / f"{group_id}-delta-{condition}.safetensors",
                        {"delta": delta},
                    )

                for condition, variables in (
                    ("av_diff", (affected, affected)),
                    ("av_diff_control", (other, other)),
                ):
                    id_a = f"{group_id}-A-{variables[0]}"
                    id_b = f"{group_id}-B-{variables[1]}"
                    dir_a = pilot.directions.get(id_a)
                    dir_b = pilot.directions.get(id_b)
                    if dir_a is None or dir_b is None:
                        missing = [
                            f"{name} ({pilot.direction_row_status.get(name)})"
                            for name, value in ((id_a, dir_a), (id_b, dir_b))
                            if value is None
                        ]
                        evidence["deltas"][condition] = {
                            "status": "failed",
                            "reason": "pilot direction unavailable: "
                            + ", ".join(missing),
                        }
                        continue
                    record_delta(
                        condition,
                        dir_b,
                        dir_a,
                        source="pilot_saved_ar_directions",
                        source_rows=[id_a, id_b],
                    )
                for style in ORACLE_STYLES:
                    for condition, variable, value_a, value_b in (
                        (
                            f"oracle_{style}",
                            affected,
                            answers["A"],
                            answers["B"],
                        ),
                        (f"wrongvar_{style}", other, other_value, candidate),
                    ):
                        if value_b is None:
                            evidence["deltas"][condition] = {
                                "status": "failed",
                                "reason": "wrong-variable candidate out of range",
                            }
                            continue
                        text_a = render_template(style, variable, value_a)
                        text_b = render_template(style, variable, value_b)
                        start = time.perf_counter()
                        try:
                            recon_a = ar.reconstruct(text_a)
                            recon_b = ar.reconstruct(text_b)
                        except (ValueError, RuntimeError) as error:
                            evidence["deltas"][condition] = {
                                "status": "failed",
                                "reason": str(error),
                                "style": style,
                                "text_a": text_a,
                                "text_b": text_b,
                                "text_a_sha256": template_sha256(text_a),
                                "text_b_sha256": template_sha256(text_b),
                            }
                            continue
                        record_delta(
                            condition,
                            recon_b,
                            recon_a,
                            source="live_ar_oracle_template",
                            style=style,
                            text_a=text_a,
                            text_b=text_b,
                            text_a_sha256=template_sha256(text_a),
                            text_b_sha256=template_sha256(text_b),
                            ar_seconds=time.perf_counter() - start,
                        )
                group_evidence[group_id] = evidence
                print(f"  AR: {group_id}", flush=True)
            handle.remove()
            del ar
            _release_models()
        with _stage(timings, "target_reload_and_behavior", device):
            target = Target(
                load_model(paths["target"], "target", device), tokenizers["target"]
            )
            handle = _count_forwards(target.model, calls, "target_behavior")
            for input_row, row in zip(inputs, rows):
                record = activations[row["id"]]
                ids, mask = target.tensors(
                    [input_row["input_ids"]], [input_row["attention_mask"]]
                )
                group_id = row["group_id"]
                evidence = {}
                is_affected_row = input_row["affected"]
                base_scores = (
                    sorted({input_row["answer"], input_row["counterfactual_answer"]})
                    if is_affected_row
                    else [input_row["answer"]]
                )

                def run_condition(target, key, replacement, score_answers):
                    start = time.perf_counter()
                    logits = target.forward(
                        ids,
                        mask,
                        record.site if replacement is not None else None,
                        replacement,
                    )
                    next_logits = logits[0, record.site.position].float().cpu()
                    del logits
                    evidence[key] = next_logits
                    if key == "P1" and not torch.allclose(
                        evidence["P0"], next_logits, atol=1e-5, rtol=1e-5
                    ):
                        raise RuntimeError(
                            "raw restoration changed logits after target reload"
                        )
                    generation = target.greedy(ids, mask, record.site, replacement)
                    scores = {
                        answer: target.score_answer(
                            ids,
                            mask,
                            record.site,
                            answer,
                            record.vector,
                            replacement,
                        )
                        for answer in score_answers
                    }
                    measured = {
                        "status": "ok",
                        "generation": generation,
                        "next_token_kl": next_token_kl(
                            evidence["P0"], next_logits
                        ),
                        "answer_scores": scores,
                        "seconds": time.perf_counter() - start,
                    }
                    if replacement is not None:
                        measured["patch_norm"] = float(
                            torch.linalg.vector_norm(replacement.float())
                        )
                        measured["norm_ratio"] = measured["patch_norm"] / record.norm
                        evidence[f"{key}_replacement"] = replacement
                    return measured

                row["conditions"]["P0"] = run_condition(target, "P0", None, base_scores)
                if row["conditions"]["P0"]["generation"] != row["baseline_generation"]:
                    raise RuntimeError(
                        "target reload changed unmodified greedy generation"
                    )
                row["conditions"]["P1"] = run_condition(
                    target, "P1", record.vector, base_scores
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
                for condition in STEERING_CONDITIONS:
                    delta = deltas.get((group_id, condition))
                    detail = group_evidence[group_id]["deltas"][condition]
                    for alpha in ALPHA_GRID:
                        key = condition_key(condition, alpha)
                        if delta is None:
                            row["conditions"][key] = {
                                "status": "failed",
                                "reason": detail.get(
                                    "reason", "steering delta unavailable"
                                ),
                                "steering_condition": condition,
                                "alpha": alpha,
                            }
                            continue
                        score_answers = base_scores
                        candidate = input_row["wrong_variable_candidate"]
                        if (
                            not is_affected_row
                            and condition.startswith("wrongvar")
                            and candidate is not None
                        ):
                            score_answers = sorted({*base_scores, str(candidate)})
                        replacement = compose_patch(
                            record.vector,
                            record.norm,
                            delta,
                            alpha,
                            dtype=record.vector.dtype,
                        )
                        measured = run_condition(target, key, replacement, score_answers)
                        measured["steering_condition"] = condition
                        measured["alpha"] = alpha
                        row["conditions"][key] = measured
                pilot_p0 = pilot.p0_generations.get(row["id"])
                row["cross_run"]["pilot_p0_generation"] = pilot_p0
                row["cross_run"]["p0_generation_matches_pilot"] = (
                    row["conditions"]["P0"]["generation"] == pilot_p0
                    if pilot_p0 is not None
                    else None
                )
                if is_affected_row:
                    row["steering"] = group_evidence[group_id]
                else:
                    row["steering"] = {
                        "role": "integrity_query",
                        "group_evidence_row": groups[group_id]["affected"]["id"],
                    }
                row["payload_bytes"] = {
                    "native_selected_vector": record.vector.numel()
                    * record.vector.element_size(),
                    "retained_norm_float32": 4,
                    "site_three_int64": 24,
                    "retained_input_ids_int64": ids.numel() * 8,
                    "retained_attention_mask_int64": mask.numel() * 8,
                    "steering_delta_float32_each": next(
                        iter(deltas.values())
                    ).numel()
                    * 4
                    if deltas
                    else None,
                    "steering_deltas_per_group": len(STEERING_CONDITIONS),
                    "oracle_texts_utf8": "per group in the affected row's steering evidence",
                    "target_bucket_padding": "retained ids/mask keep logical length; every target forward was right-padded to the manifest target_bucket for kernel-shape pinning",
                    "shared_metadata_and_weights": "see manifest/model_lock and inventory; not included in text bytes",
                }
                save_numeric(
                    run.path / "raw" / f"{row['id']}-behavior.safetensors", evidence
                )
                print(f"  Behavior: {row['id']}", flush=True)
            handle.remove()
            del target
            _release_models()
    finally:
        # Even a failed identity gate or backend leaves partial evidence/accounting.
        write_json(run.path / "results.json", rows)
    return rows


def build_steering_manifest(
    lock,
    lock_path,
    inputs,
    metadata,
    report,
    *,
    plan,
    pilot: PilotArtifacts,
    tokenization,
    group_counts_source,
):
    """Locked manifest, written before the first model forward."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    return {
        "schema_version": 1,
        "stage": STEERING_STAGE,
        "groups": plan.counts["pilot"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "source_file_sha256": _source_identity(),
        "model_lock": lock,
        "model_lock_sha256": sha256_file(lock_path),
        "inputs": inputs,
        "plan_hash": plan.plan_hash,
        "split_counts": dict(plan.counts),
        "group_counts_source": group_counts_source,
        "split_seed": SPLIT_SEEDS["pilot"],
        "seed": STEERING_SEED,
        "reused_split": {
            "split": "pilot",
            "note": "the same 128 pilot groups, re-measured: this is a new "
            "measurement on a reused split, not a new split and not a pilot "
            "rerun; the closed pilot's recorded outcomes stand unchanged",
        },
        "pilot_run": {
            "path_sha256": pilot.path_sha256,
            "manifest_sha256": pilot.manifest_sha256,
            "plan_hash": pilot.plan_hash,
            "completion_status": pilot.completion_status,
            "access": "read-only at run time via --pilot-run; never copied into the repository",
            "arm1_inputs": "raw/{row}-direction.safetensors saved AR directions",
        },
        "target_template": PROMPT_TEMPLATE,
        "metadata": metadata,
        "software": report,
        "site": f"model.layers.{metadata['layer']} output; last non-padding assistant-prefix token",
        "backend": "local-transformers-eager-no-cache",
        "dtype": "bfloat16",
        "av_decoding": "NOT RUN: the AV is never loaded in this assay; arm 1 "
        "consumes the pinned pilot run's saved AR direction vectors "
        "(read-only) and arm 2 uses frozen oracle templates",
        "ar_decoding": "released AR with its trained value head; arm-2 oracle "
        "and control texts only; the AR contributes directions, never norms",
        "target_decoding": {"greedy": True, "max_new_tokens": GENERATION_CEILING},
        "identity_atol": 1e-5,
        "identity_rtol": 1e-5,
        "suffix_causality_gate": "same-length dummy-suffix bitwise equality",
        "suffix_drift_bound": SUFFIX_DRIFT_BOUND,
        "suffix_drift_bound_justification": "frozen before the pilot from 2026-09-21 GB10 smoke data (brief §4): measured maximum cross-length relative drift 3.09e-2 from BF16 kernel reduction reassociation; bound 1e-1 adds safety factor ~3 and is never relaxed after inspecting validation results",
        "target_bucket": TARGET_BUCKET,
        "target_padding_policy": "all target forwards right-padded to fixed length 128 for kernel-shape pinning on sm_121; masked pads contribute exactly zero (bitwise-proven); pinned regime established 2026-09-21 pre-pilot after the first pilot attempt's identity-gate failure; supersedes the unpadded smoke/calibration runs",
        "greedy_identity_gate": "exact greedy equality required; divergence accepted only with measured within-bound prefix drift at the divergence step (same frozen SUFFIX_DRIFT_BOUND and 2026-09-21 justification); repaired pre-pilot after the first pilot attempt failed this gate on variant pilot-0000-A-x with no conditions executed; no pilot/validation outcomes inspected",
        "steering_recipe": "NLA reconstructed-difference steering "
        "(Fraser-Taliente, Kantamneni, Ong et al. 2026, Planning in Poetry): "
        "delta = AR(d_steered) - AR(d_orig); patch h' = h + alpha * ||h|| * "
        "delta/||delta|| at the receiver site; additive patch, the patched "
        "norm is not restored; https://transformer-circuits.pub/2026/nla/ . "
        "The paper reports ~50% success with sometimes-incoherent completions "
        "on its own target; no paper rate is imported here",
        "scale_policy": "alpha times the receiver row's own retained float32 "
        "norm (4 bytes); alpha grid frozen; alpha=0 implied by the P0 "
        "reference row and never separately executed",
        "arms": STEERING_ARMS,
        "oracle_templates": template_records(),
        "controls": {
            "wrong_variable_oracle": "wrongvar_terse / wrongvar_structured: the "
            "same template style asserting the unaffected variable's value "
            "versus that value shifted by the affected delta; the x-answer "
            "must not move (criterion C3)",
            "same_value_av_difference": "av_diff_control: saved AR directions "
            "of the A/B rows whose queried answer does not change; the "
            "x-answer must not move (criterion C3 for arm 1)",
            "alpha_zero": "no-op, implied by the P0 reference row",
            "sign_flip": "alpha = -1 applies every delta in the opposite direction",
        },
        "alpha_grid": list(ALPHA_GRID),
        "evaluation_alpha": EVALUATION_ALPHA,
        "steering_frozen": STEERING_FROZEN,
        "answer_tokenization": "separate canonical integer IDs plus EOS, append without retokenizing prefix",
        "metrics": [
            "exact_answer",
            "flip_to_B",
            "retained_A",
            "moved_x",
            "y_intact",
            "L_counterfactual_minus_original_logprob",
            "KL(P0||condition)",
            "full_answer_log_probability",
            "delta_norm",
            "cosine_with_saved_counterfactual_difference",
        ],
        "generator": {"tokenization": tokenization},
        "limits": "Steering rates are behavioral measurements on one "
        "checkpoint, one task family, and one site; they are not semantic "
        "proofs. Oracle text is an intervention instrument, never read-out "
        "semantics. The reused pilot split's recorded outcomes stand; this "
        "assay's outcomes are reported separately. Not whole-state "
        "compression, not a new steering method, not a speedup.",
    }


def _group_counts(raw):
    """Dev/test override for the split table; mirrors runner._group_counts."""
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=PROJECT / "configs/model-lock.json"
    )
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--av-path", type=Path)
    parser.add_argument("--ar-path", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--pilot-run",
        type=Path,
        required=True,
        help="completed pinned pilot run directory, accessed read-only at run "
        "time for arm-1 saved AR directions and cross-run P0/original checks",
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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    from .audit import summarize  # late import: audit.py imports this module

    args = parse_args(argv)
    if args.run_dir is None:
        args.run_dir = (
            PROJECT
            / "runs"
            / (
                datetime.now(timezone.utc).strftime(f"{STEERING_STAGE}-%Y%m%dT%H%M%SZ-")
                + uuid.uuid4().hex[:8]
            )
        )
    run = RunDirectory(args.run_dir)
    started, timings, calls = time.perf_counter(), {}, {}
    status, error, report, manifest = "FAILED", None, {}, None
    plan, pilot, rows = None, None, []
    try:
        with _stage(timings, "compatibility", "cpu"):
            report = compatibility(args.device)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(STEERING_SEED)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["tf32"] = False
            report["deterministic_algorithms"] = (
                torch.are_deterministic_algorithms_enabled()
            )
        lock = read_lock(args.lock)
        paths = model_paths(
            lock,
            args.cache,
            {role: getattr(args, f"{role}_path") for role in ("target", "av", "ar")},
        )
        with _stage(timings, "fetch_and_hash_verification", "cpu"):
            if args.fetch_models:
                report["fetch"] = fetch_models(lock, paths)
            else:
                report["fetch"] = {"new_artifact_bytes": 0, "requested": False}
            report["verified_artifact_bytes"] = verify_models(lock, paths)
        with _stage(timings, "metadata_and_task_preprocessing", "cpu"):
            tokenizers, av_meta, ar_meta, metadata = inspect_metadata(paths)
            plan = build_plan(_group_counts(args.group_counts_json))
            validate_plan(plan)
            tokenized = tokenize_split(plan, "pilot", tokenizers["target"])
            inputs = steering_inputs(tokenized["rows"], tokenizers["target"])
            groups = _group_map(inputs)
            direction_rows, original_rows = [], []
            for group_id, pair in groups.items():
                affected, other = pair["affected"]["variable"], pair["other"]["variable"]
                direction_rows += [
                    f"{group_id}-A-{affected}",
                    f"{group_id}-B-{affected}",
                    f"{group_id}-A-{other}",
                    f"{group_id}-B-{other}",
                ]
                original_rows += [
                    f"{group_id}-A-{affected}",
                    f"{group_id}-A-{other}",
                    f"{group_id}-B-{affected}",
                ]
            pilot = load_pilot_run(
                args.pilot_run,
                plan_hash=plan.plan_hash,
                width=av_meta.width,
                direction_rows=direction_rows,
                original_rows=original_rows,
                p0_rows=[row["id"] for row in inputs],
            )
            manifest = build_steering_manifest(
                lock,
                args.lock,
                inputs,
                metadata,
                report,
                plan=plan,
                pilot=pilot,
                tokenization={
                    key: tokenized[key]
                    for key in (
                        "accepted",
                        "rejected",
                        "rejections",
                        "prior_excluded_prompts",
                    )
                },
                group_counts_source="dev override --group-counts-json"
                if args.group_counts_json
                else "brief default table (reused pilot split)",
            )
            write_json(
                run.path / "manifest.json", manifest
            )  # before the first model forward
            # Loud pre-inference shape guarantee; B never changes silently.
            check_steering_bucket_fit(inputs)
        rows = execute_steering(
            run,
            paths,
            tokenizers,
            av_meta,
            ar_meta,
            inputs,
            pilot,
            args.device,
            timings,
            calls,
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
    result = {
        "status": status,
        "error": error,
        "stage": STEERING_STAGE,
        "planned_groups": manifest["groups"]
        if manifest is not None
        else DEFAULT_COUNTS["pilot"],
        "real_model_checks": "NOT RUN" if manifest is None else status,
        "plan_hash": manifest["plan_hash"] if manifest is not None else None,
        "pilot_run": manifest["pilot_run"] if manifest is not None else None,
        "decision": summary_record["decision"] if summary_record else None,
    }
    write_json(run.path / "completion.json", result)
    with (run.path / "report.md").open("x") as stream:
        stream.write(
            f"# Post-Phase-Two steering assay\n\nStatus: **{status}**. "
            f"{result['planned_groups']} reused pilot groups, two receiver rows each "
            f"(affected query + integrity query), frozen alpha grid "
            f"{list(ALPHA_GRID)}.\n\n"
        )
        if error:
            stream.write(f"Failure: {error}\n\n")
        if summary_record is not None:
            stream.write(
                f"Groups: {len(summary_record['successful_groups'])} successful, "
                f"{len(summary_record['failed_groups'])} failed, "
                f"{len(summary_record['skipped_groups'])} skipped.\n\n"
            )
            decision = summary_record["decision"]
            stream.write(
                "## Steering decision\n\nDerived only from the frozen criteria "
                "quoted in manifest.json (protocols/steering_assay_brief.md "
                "section 5 design choices, frozen pre-run).\n\n"
            )
            for arm, record in decision["arms"].items():
                criteria = record["criteria"]

                def cell(value):
                    return f"{value:.4f}" if value is not None else "n/a"

                stream.write(f"### {arm}\n\n| Criterion | Value | Rule | Met |\n|---|---:|---|:--:|\n")
                c1 = criteria["c1_flip_to_B"]
                stream.write(
                    f"| C1 flip_to_B at alpha=1 | {cell(c1['value'])} | >= {c1['threshold']} | {c1['met']} |\n"
                )
                c2 = criteria["c2_y_integrity"]
                stream.write(
                    f"| C2 y_intact at alpha=1 | {cell(c2['value'])} | >= {c2['threshold']} | {c2['met']} |\n"
                )
                c3 = criteria["c3_control_specificity"]
                stream.write(
                    f"| C3 control moved_x at alpha=1 ({c3['control']}) | {cell(c3['value'])} | < {c3['threshold']} | {c3['met']} |\n"
                )
                c4 = criteria["c4_direction_check"]
                stream.write(
                    f"| C4 direction check (flip at alpha=-1 <= alpha=1, non-gating) | {cell(c4['value_negative'])} <= {cell(c4['value_evaluation'])} | supporting | {c4['met']} |\n\n"
                )
                stream.write(f"Arm success (C1 and C2 and C3): **{record['success']}**.\n\n")
            stream.write(
                f"Successful arms: {decision['successful_arms'] or 'none'}.\n\n"
            )
            stream.write("## Per-condition summary at every alpha\n\n")
            stream.write(
                "| Condition | alpha | flip_to_B | y_intact | moved_x | mean L (valid) | mean KL x | mean KL y |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for key, measured in summary_record["per_condition"].items():
                preference = measured["L"]
                mean_l = (
                    f"{preference['mean']:.4f}"
                    if preference.get("mean") is not None
                    else "n/a"
                )
                kl_x = measured["mean_valid_next_token_kl_x"]
                kl_y = measured["mean_valid_next_token_kl_y"]
                stream.write(
                    f"| {measured['condition']} | {measured['alpha']} | "
                    f"{measured['flip_to_B']['rate']:.4f} | "
                    f"{measured['y_intact']['rate']:.4f} | "
                    f"{measured['moved_x']['rate']:.4f} | {mean_l} | "
                    f"{f'{kl_x:.4f}' if kl_x is not None else 'n/a'} | "
                    f"{f'{kl_y:.4f}' if kl_y is not None else 'n/a'} |\n"
                )
            cross = summary_record["cross_run"]
            stream.write(
                f"\nCross-run determinism versus the pinned pilot's saved records "
                f"(descriptive, never a gate): P0 generations matching "
                f"{cross['pilot_p0_generation']}, bitwise-equal fresh originals "
                f"{cross['original_vector_bitwise']}.\n"
            )
        stream.write(
            "\nSteering rates are behavioral measurements on one checkpoint, one "
            "task family, and one site; they are not semantic proofs. Oracle text "
            "is an intervention instrument written by the authors, never read-out "
            "semantics. Arm 1 reuses the pinned pilot run's saved AR directions "
            "read-only; the closed pilot's recorded outcomes stand unchanged. "
            "This is not whole-state compression, not a new steering method, and "
            "not a speedup.\n\n"
        )
        stream.write(
            "See summary.json for all denominators, intervals and the decision, "
            "results.json for every receiver row, compatibility.json for "
            "setup/stage costs and forward calls, and inventory.json for retained "
            "versus packaged evidence. Raw vectors/full-vocabulary logits stay "
            "local; the ZIP cannot independently replay KL.\n"
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
