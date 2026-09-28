"""Measured vLLM/eager equivalence gate for the opt-in AV/AR backend.

Replays a completed eager run's saved evidence through the vLLM worker and
compares against the saved eager outputs:

- AV: the run's raw/*-original.safetensors go back through the vLLM verbalizer;
  the bar is 100% token-identical greedy continuations per row.
- AR: the run's saved descriptions go through the vLLM reconstructor; the bar
  is min per-row direction cosine >= 0.9999, with the norm-relative error
  distribution declared.

Any shortfall means vllm is a distinct measurement backend: it stays
non-default and its outputs may not be mixed into eager-regime evidence. The
reference run directory is read-only; all replay evidence lands in --output.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from . import vllm_backend
from .artifacts import load_numeric, sha256_file, write_json
from .nla_adapter import ar_prompt, av_prompt
from .preflight import inspect_metadata, model_paths, read_lock, verify_models

AV_TOKEN_IDENTITY_BAR = 1.0
AR_MIN_COSINE_BAR = 0.9999
PROJECT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unit(vector):
    value = vector.float()
    norm = float(torch.linalg.vector_norm(value))
    _require(math.isfinite(norm) and norm > 1e-12, "degenerate reference vector")
    return value / norm


def load_reference(run_dir):
    """Saved eager evidence; the reference run stays untouched."""
    run_dir = Path(run_dir)
    _require(run_dir.is_dir(), f"reference run missing: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = json.loads((run_dir / "results.json").read_text())
    _require(manifest.get("av_backend", "eager") == "eager", "reference must be eager")
    _require(manifest.get("ar_backend", "eager") == "eager", "reference must be eager")
    originals, descriptions, directions = {}, {}, {}
    for row in rows:
        row_id = row["id"]
        original_path = run_dir / "raw" / f"{row_id}-original.safetensors"
        if original_path.is_file():
            originals[row_id] = load_numeric(original_path)["original"]
        record = row.get("description") or {}
        if record.get("status") is not None:
            descriptions[row_id] = record
        direction_path = run_dir / "raw" / f"{row_id}-direction.safetensors"
        if direction_path.is_file():
            directions[row_id] = load_numeric(direction_path)["ar_direction"]
    return manifest, rows, originals, descriptions, directions


def compare_av(eager_records, vllm_records):
    """Per-row greedy token identity; first divergence locates the flip."""
    rows = []
    identical = 0
    for row_id in sorted(eager_records):
        eager = eager_records[row_id]
        record = vllm_records.get(row_id) or {}
        eager_ids = eager.get("token_ids") or []
        vllm_ids = record.get("token_ids") or []
        same = vllm_ids == eager_ids and record.get("status") == eager.get("status")
        identical += int(same)
        shared = min(len(eager_ids), len(vllm_ids))
        first = next(
            (i for i in range(shared) if eager_ids[i] != vllm_ids[i]),
            shared if len(eager_ids) != len(vllm_ids) else None,
        )
        rows.append(
            {
                "id": row_id,
                "identical": same,
                "eager_status": eager.get("status"),
                "vllm_status": record.get("status"),
                "eager_tokens": len(eager_ids),
                "vllm_tokens": len(vllm_ids),
                "first_divergence": first,
                "prefix_token_agreement": (
                    (first if first is not None else len(eager_ids)) / len(eager_ids)
                    if eager_ids
                    else None
                ),
            }
        )
    total = len(rows)
    return {
        "rows": total,
        "identical_rows": identical,
        "token_identity_rate": identical / total if total else None,
        "bar": AV_TOKEN_IDENTITY_BAR,
        "bar_met": total > 0 and identical == total,
        "per_row": rows,
    }


def compare_ar(eager_directions, vllm_vectors):
    """Per-row cosine and norm-relative error between directions."""
    rows = []
    for row_id in sorted(eager_directions):
        if row_id not in vllm_vectors:
            rows.append({"id": row_id, "status": "missing_vllm_direction"})
            continue
        eager, measured = eager_directions[row_id], vllm_vectors[row_id]
        cosine = float(_unit(eager) @ _unit(measured))
        norm_eager = float(torch.linalg.vector_norm(eager.float()))
        norm_measured = float(torch.linalg.vector_norm(measured.float()))
        relative = (
            abs(norm_measured - norm_eager) / norm_eager if norm_eager > 0 else None
        )
        rows.append(
            {
                "id": row_id,
                "status": "ok",
                "cosine": cosine,
                "norm_relative_error": relative,
                "eager_norm": norm_eager,
                "vllm_norm": norm_measured,
            }
        )
    cosines = [row["cosine"] for row in rows if row.get("status") == "ok"]
    errors = [row["norm_relative_error"] for row in rows if row.get("status") == "ok"]
    return {
        "rows": len(rows),
        "compared": len(cosines),
        "cosine": {
            "min": min(cosines) if cosines else None,
            "median": statistics.median(cosines) if cosines else None,
            "max": max(cosines) if cosines else None,
        },
        "norm_relative_error": {
            "min": min(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
        "bar": f"min cosine >= {AR_MIN_COSINE_BAR}",
        "bar_met": bool(cosines)
        and min(cosines) >= AR_MIN_COSINE_BAR
        and len(cosines) == len(rows),
        "per_row": rows,
    }


def run_gate(run_dir, lock, cache, output, *, engine=None):
    run_dir = Path(run_dir)
    manifest, _rows, originals, descriptions, directions = load_reference(run_dir)
    _require(originals, "reference run has no saved original activations")
    lock_data = read_lock(lock)
    paths = model_paths(lock_data, cache)
    verified = verify_models(lock_data, paths)
    tokenizers, av_meta, ar_meta, _metadata = inspect_metadata(paths)
    prompt_ids, position = av_prompt(tokenizers["av"], av_meta)
    _require(
        prompt_ids == manifest["metadata"]["av_prompt_ids"]
        and position == manifest["metadata"]["av_injection_position"],
        "reference manifest AV prompt disagrees with the verified models",
    )
    _require(
        ar_prompt(tokenizers["ar"], ar_meta, "tokenizer preflight")
        == manifest["metadata"]["ar_probe_ids"],
        "reference manifest AR probe disagrees with the verified models",
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    av_dir, ar_dir = output / "av_worker", output / "ar_worker"

    av_records, av_completion = vllm_backend.verbalize_batch(
        av_path=paths["av"],
        av_meta=av_meta,
        tokenizer=tokenizers["av"],
        vectors=originals,
        engine=engine,
        work_dir=av_dir,
    )
    eager_av = {row_id: descriptions[row_id] for row_id in originals}
    av_report = compare_av(eager_av, av_records)

    items = [
        (row_id, descriptions[row_id]["description"])
        for row_id in sorted(descriptions)
        if descriptions[row_id].get("status") == "ok"
        and row_id in directions
        and descriptions[row_id].get("description")
    ]
    _require(items, "reference run has no saved ok descriptions with directions")
    ar_vectors, ar_completion = vllm_backend.reconstruct_batch(
        ar_path=paths["ar"],
        ar_meta=ar_meta,
        tokenizer=tokenizers["ar"],
        head_path=paths["ar"] / "value_head.safetensors",
        items=items,
        engine=engine,
        work_dir=ar_dir,
    )
    ar_report = compare_ar(
        {row_id: directions[row_id] for row_id, _ in items}, ar_vectors
    )

    verdict = "PASS" if av_report["bar_met"] and ar_report["bar_met"] else "FAIL"
    report = {
        "gate": "vllm_equivalence",
        "verdict": verdict,
        "reference_run": str(run_dir),
        "reference_run_readonly": True,
        "model_lock_sha256": sha256_file(lock),
        "verified_artifact_bytes": verified,
        "bars": {
            "av_token_identity_rate": AV_TOKEN_IDENTITY_BAR,
            "ar_min_cosine": AR_MIN_COSINE_BAR,
            "ar_norm_relative_error": "declared (distribution reported)",
        },
        "av": av_report,
        "ar": ar_report,
        "worker_evidence": {
            "av_completion": "av_worker/out/completion.json",
            "ar_completion": "ar_worker/out/completion.json",
            "av_forward_calls": av_completion["model_forward_calls"],
            "ar_forward_calls": ar_completion["model_forward_calls"],
        },
        "consequence": (
            "PASS: vllm reproduced the eager bar on this bundle; backend remains "
            "opt-in. FAIL or partial: vllm is a distinct measurement backend, "
            "stays non-default, and its outputs may not be mixed into "
            "eager-regime evidence."
        ),
    }
    write_json(output / "gate.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock", type=Path, default=PROJECT / "configs/model-lock.json"
    )
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="fraction of TOTAL unified memory per worker engine; account for "
        "other residents (default: worker default 0.25)",
    )
    args = parser.parse_args(argv)
    engine = (
        {"gpu_memory_utilization": args.gpu_memory_utilization}
        if args.gpu_memory_utilization is not None
        else None
    )
    report = run_gate(args.run_dir, args.lock, args.cache, args.output, engine=engine)
    print(
        json.dumps(
            {"verdict": report["verdict"], "gate_json": str(args.output / "gate.json")}
        )
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
