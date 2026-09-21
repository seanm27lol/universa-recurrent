"""Eight-group engineering smoke only; scientific stages are deliberately absent."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
import uuid
import torch

from .artifacts import RunDirectory, save_numeric, sha256_file, write_json
from .audit import summarize
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
from .target import Site, Target
from .tasks import PROMPT_TEMPLATE, SPLIT_SEEDS, generate_groups, tokenize_groups

PROJECT = Path(__file__).resolve().parents[2]


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


def build_manifest(lock, lock_path, inputs, statistics_record, metadata, report):
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    return {
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
        "dtype": "bfloat16",
        "av_decoding": {
            "greedy": True,
            "max_new_tokens": 200,
            "descriptions_per_activation": 1,
        },
        "target_decoding": {"greedy": True, "max_new_tokens": 8},
        "identity_atol": 1e-5,
        "identity_rtol": 1e-5,
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


def execute_smoke(
    run, paths, tokenizers, av_meta, ar_meta, inputs, device, timings, calls
):
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
        with stage(timings, "av_load_and_generate", device):
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
        with stage(timings, "ar_load_and_reconstruct", device):
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
                    row["reconstruction"] = {
                        "status": "ok",
                        **direction_metrics(activations[row["id"]].vector, vector),
                    }
                    save_numeric(
                        run.path / "raw" / f"{row['id']}-direction.safetensors",
                        {"ar_direction": vector},
                    )
                except (ValueError, RuntimeError) as error:
                    row["reconstruction"] = {"status": "failed", "error": str(error)}
                row["ar_seconds"] = time.perf_counter() - start
            handle.remove()
            del ar
            release_models()
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
                if (
                    row["conditions"]["P0"]["generation"]
                    != row["conditions"]["P1"]["generation"]
                ):
                    raise RuntimeError(
                        "raw replacement changed greedy generation after reload"
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
        "--fetch-models",
        action="store_true",
        help="explicitly download only the locked model artifacts",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.run_dir is None:
        args.run_dir = (
            PROJECT
            / "runs"
            / (
                datetime.now(timezone.utc).strftime("smoke-%Y%m%dT%H%M%SZ-")
                + uuid.uuid4().hex[:8]
            )
        )
    run = RunDirectory(args.run_dir)
    started, timings, calls = time.perf_counter(), {}, {}
    status, error, report, manifest = "FAILED", None, {}, None
    try:
        with stage(timings, "compatibility", "cpu"):
            report = compatibility(args.device)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(201100)
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
        with stage(timings, "fetch_and_hash_verification", "cpu"):
            if args.fetch_models:
                report["fetch"] = fetch_models(lock, paths)
            else:
                report["fetch"] = {"new_artifact_bytes": 0, "requested": False}
            report["verified_artifact_bytes"] = verify_models(lock, paths)
        with stage(timings, "metadata_and_task_preprocessing", "cpu"):
            tokenizers, av_meta, ar_meta, metadata = inspect_metadata(paths)
            groups, generation_stats = generate_groups("smoke", 8)
            inputs = tokenize_groups(groups, tokenizers["target"])
            for row in inputs:
                row["answer_token_ids_including_eos"] = answer_tokens(
                    tokenizers["target"], row["answer"]
                )
            manifest = build_manifest(
                lock, args.lock, inputs, generation_stats, metadata, report
            )
            write_json(
                run.path / "manifest.json", manifest
            )  # before the first model forward
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
    result = {
        "status": status,
        "error": error,
        "stage": "engineering_smoke",
        "planned_groups": 8,
        "real_model_checks": "NOT RUN" if manifest is None else status,
        "scientific_pilot": "NOT RUN",
        "locked_validation": "NOT RUN",
    }
    write_json(run.path / "completion.json", result)
    with (run.path / "report.md").open("x") as stream:
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
