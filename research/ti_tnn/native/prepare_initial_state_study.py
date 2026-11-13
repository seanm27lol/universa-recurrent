"""CPU-only data, evaluation and resource preparation. There is no execute option."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from . import data as D
from . import initial_state_study as study
from .prepare_program_study import _audit_packed_split

MODES = ("dense", "shared", "untyped", "typed")
EPOCHS = 16
BATCH_SIZE = 8


def pack(examples, tokenizer, directory):
    """Reuse the established intact packing and audit, including all auxiliary targets."""
    documents = [D.tokenize_structured(example, tokenizer) for example in examples]
    runs = [D.TokenRun(i, document, len(document.input_ids)) for i, document in enumerate(documents)]
    arrays, stats = D.pack_runs(runs, context_length=256, packing="intact",
                               structured_overflow="error", position_policy="document")
    consumed, checks = _audit_packed_split("prepared", examples, documents, arrays, stats, tokenizer,
                                          context_length=256, position_policy="document")
    # Frozen GPT2 has one value token per register for this 0..31 domain.
    if checks["answer_tokens_supervised"] != 16 * len(examples):
        raise ValueError("answer tokenization changed; do not silently change matched exposure")
    if any(len(tokenizer.encode(e.prompt).ids) + 48 > 256 for e in examples):
        raise ValueError("full prompt plus generation bound exceeds context")
    for name in D.ARRAY_NAMES:
        D._write_array(directory, name, arrays[name])
    D._atomic_jsonl(directory / "documents.jsonl", consumed)
    return {**checks, "padded_slots": int(arrays["input_ids"].size),
            "padding_tokens": int(arrays["key_padding_mask"].sum()),
            "numeric_value_tokens": 4 * len(examples),
            "formatting_eos_tokens": 13 * len(examples),
            "route_targets": int(arrays["route_mask"].sum()),
            "state_targets": int(arrays["state_mask"].sum())}, arrays


def resource_projection(endpoint, evaluation_count, bundle, preprocessing_seconds):
    """Conservative measured cost model; estimates are not execution authorization."""
    bundle = Path(bundle)
    status = json.loads((bundle / "run_status.json").read_text())
    if status.get("complete") is not True or status.get("failure") is not None:
        raise ValueError("resource reference must be a completed study")
    full = [status["children"][f"78-{mode}-full_sequence"] for mode in ("dense", "untyped", "typed")]
    throughput = min(c["additional_exposure_tokens"] / c["train_seconds"] for c in full)
    overhead = max(c["measured_stage_seconds"]["training_invocation_wall"] - c["train_seconds"]
                   - c["checkpoint_seconds"] for c in full)
    eval_rate = max(c["measured_stage_seconds"]["official_evaluation"] / 830 for c in full)
    checkpoint_rate = max(c["checkpoint_seconds"] / 23 for c in full)
    diagnostics = max(c["measured_stage_seconds"]["answer_diagnostics"] for c in full)
    updates = endpoint["updates"]
    checkpoints = (updates - 1) // 500 + 2  # initial, periodic, final
    base = {
        "training_with_data_routing_forward_backward_optimizer": endpoint["full_label_tokens"] / throughput,
        "loading_logging_periodic_dev_and_unclassified_invocation_overhead":
            overhead * max(1.0, updates / 10176),
        "checkpoints": checkpoints * checkpoint_rate,
        "generation_CE_state_readout": len(study.POPULATIONS) * evaluation_count * eval_rate,
        "answer_component_diagnostics": diagnostics,
        "independent_scoring_and_dependency_checks": 5.0,
        "bounded_CPU_FP32_reconstruction": 10.0,
    }
    # No new shared-arm measurement exists; do not invent one. Use the slowest
    # observed arm plus 25% for all shared-arm components, then the global margin.
    arms = {mode: {key: value * (1.25 if mode == "shared" else 1.0) for key, value in base.items()}
            for mode in MODES}
    for values in arms.values():
        values["total"] = sum(values.values())
    common = {"measured_CPU_preparation": preprocessing_seconds, "launch_and_final_export_allowance": 60.0}
    total = 2 * sum(v["total"] for v in arms.values()) + sum(common.values())
    watchdog = total / 3600 * 1.5
    allocation = math.ceil((watchdog + 0.05) * 4) / 4
    checkpoint_bytes = max(c["checkpoint"]["bytes"] for c in full)
    return {
        "status": "proposal_not_authorized", "conditions": 2, "models": list(MODES),
        "per_model_per_condition_seconds": arms, "common_seconds": common,
        "projected_hours": total / 3600, "watchdog_hours": watchdog,
        "shutdown_allowance_hours": 0.05, "allocation_proposal_hours": allocation,
        "checkpoint_count_per_run": checkpoints,
        "checkpoint_storage_bytes_all_runs": 8 * checkpoints * checkpoint_bytes,
        "additional_data_logs_export_storage_allowance_bytes": 2 * 1024**3,
        "historical_peak_reserved_GiB": 9.09765625,
        "planning_memory_GiB": 12, "memory_limit_is_not_a_measurement": True,
        "measurement_sha256": {"run_status.json": D.sha256_file(bundle / "run_status.json")},
        "basis": {"slowest_full_sequence_tokens_per_second": throughput,
                  "max_invocation_overhead_seconds": overhead,
                  "max_official_evaluation_seconds_per_saved_output": eval_rate,
                  "max_checkpoint_seconds_per_file": checkpoint_rate,
                  "reference_saved_outputs_per_child": 830,
                  "shared_proxy_multiplier": 1.25,
                  "reference": "completed #79 full_sequence children only"},
        "limits": "shared proxy and new population timing are unmeasured; no claim of compute matching; "
                  "all CPU evaluation time remains charged if a GPU reservation is held; "
                  "recovery/failures consume the same allocation, never a refreshed cap",
    }


def prepare(output, tokenizer_path, history_paths, reference_bundle, *, seed=2026111201,
            composed_patterns=192, states_per_pattern=8, evaluation_count=128):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    tokenizer = D.load_gpt2_tokenizer(tokenizer_path)
    history = study.read_history(history_paths)
    dataset = study.generate(history, seed=seed, composed_patterns=composed_patterns,
                             states_per_pattern=states_per_pattern, evaluation_count=evaluation_count)
    invariants = study.audit(dataset, history)
    output.mkdir(parents=True)
    packing = {}
    paired_arrays = []
    for condition in study.CONDITIONS:
        root = output / condition
        examples = dataset["training"][condition]
        stats, arrays = pack(examples, tokenizer, root / "train")
        packing[condition] = {"train": stats}
        paired_arrays.append(arrays)
        D._atomic_jsonl(root / "training.jsonl", [e.to_record() for e in examples])
        for population in study.POPULATIONS:
            examples = dataset["evaluations"][condition][population]
            D._atomic_jsonl(root / "evaluation" / f"{population}.jsonl", [e.to_record() for e in examples])
            D._atomic_jsonl(root / "prompts" / f"{population}.jsonl", [e.inference_record() for e in examples])
            # dev/ makes each condition usable by the existing native trainer.
            directory = root / "dev" if population == "dev" else root / "evaluation_arrays" / population
            packing[condition][population], _ = pack(examples, tokenizer, directory)
        lookup = {e.prompt: e.target_text for e in dataset["training"][condition]}
        baselines = {}
        for name, examples in dataset["evaluations"][condition].items():
            baselines[name] = {
                "total": len(examples),
                "exact_lookup_correct": sum(lookup.get(e.prompt) == e.target_text for e in examples),
                "supplied_program_executor_correct": len(examples),
                "no_op_correct": sum(e.initial == e.execution.final for e in examples),
                "no_op_register_correct": sum(a == b for e in examples for a, b in zip(e.initial, e.execution.final)),
                "register_denominator": 4 * len(examples),
            }
        D._atomic_json(root / "baselines.json", {
            "scope": "construction references, not neural performance; targets independently executed in audit",
            "populations": baselines})
        D._atomic_json(root / "manifest.json", {"format": "native-initial-state-condition-v1",
                                               "condition": condition, "position_policy": "document",
                                               "splits": packing[condition]})
    # Values and state labels differ; structural layout and auxiliary route masks
    # do not. Equality is checked on the actual arrays, not inferred from counts.
    for name in ("positions", "doc_ids", "key_padding_mask", "source_ids", "route_targets",
                 "route_mask", "state_mask"):
        if not np.array_equal(paired_arrays[0][name], paired_arrays[1][name]):
            raise ValueError(f"paired {name} differs")
    if not np.array_equal(paired_arrays[0]["labels"] == D.IGNORE_INDEX,
                          paired_arrays[1]["labels"] == D.IGNORE_INDEX):
        raise ValueError("paired full-loss mask differs")
    train = packing["multi_state"]["train"]
    if train != packing["repeat_one"]["train"]:
        raise ValueError("paired token accounting differs")
    endpoint = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "drop_last": False,
                "updates": EPOCHS * math.ceil(train["rows"] / BATCH_SIZE),
                "full_label_tokens": EPOCHS * train["loss_tokens"],
                "input_tokens": EPOCHS * train["tokens"],
                "padded_slots": EPOCHS * train["padded_slots"],
                "answer_eos_tokens": EPOCHS * (train["answer_tokens_supervised"] + train["eos_tokens_supervised"]),
                "numeric_value_tokens": EPOCHS * train["numeric_value_tokens"],
                "formatting_eos_tokens": EPOCHS * train["formatting_eos_tokens"],
                "cursor": EPOCHS * train["rows"]}
    projection = resource_projection(endpoint, evaluation_count, reference_bundle, time.perf_counter() - started)
    root = Path(__file__).parent
    protocol = {
        "format": "native-initial-state-preparation-v1", "status": "prepared_not_authorized_or_launched",
        "training_launched": False, "optimizer_updates_performed": 0, "reservation_created": False,
        "data_seed": seed, "model_seed": 2026111202, "sampler_seed": 2026111203,
        "conditions": list(study.CONDITIONS), "modes": list(MODES),
        "initialization": "fresh random matched seed; no #78/#79 parent reuse",
        "objective": "full_sequence", "lambda_balance": 0.01, "lambda_route": 0.1, "lambda_state": 0.1,
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
                      "betas": [0.9, 0.999], "eps": 1e-8, "grad_clip": 1.0},
        "scheduler": {"name": "existing_cosine", "warmup_fraction": 0.02, "final_lr_fraction": 0.1},
        "checkpoint_every": 500, "dev_every": 500, "dev_batches": 0,
        "selection": "fixed final endpoint only, no intermediate or dev-based selection",
        "generation": {"greedy": True, "max_new_tokens": 48, "use_cache": True,
                       "prompt_policy": "error", "target_length_used_for_decoding": False},
        "endpoint_per_run": endpoint, "packing": packing, "invariants": invariants,
        "tokenizer_sha256": D.sha256_file(tokenizer_path),
        "source_sha256": {name: D.sha256_file(root / name) for name in (
            "initial_state_study.py", "prepare_initial_state_study.py", "data.py", "structured.py",
            "prepare_program_study.py", "independent/rescore_generations.py",
            "independent/dependency_diagnostic.py", "independent/score_initial_state_study.py",
            "train.py", "sampler.py", "model.py", "losses.py", "generation.py", "checkpoint.py",
            "configs/engineering.json")},
        "historical_exclusion_sha256": history["sha256"],
        "historical_exclusion_scope": "explicit supplied native structured inventory; missing files fail closed; "
                                      "not a claim about unlisted or unknown datasets",
        "resource_projection": projection,
    }
    D._atomic_json(output / "pattern_plan.json", {k: dataset[k] for k in
                                                 ("patterns", "slots", "distinct_target_counts")})
    protocol["artifact_sha256"] = {str(p.relative_to(output)): D.sha256_file(p)
                                   for p in sorted(output.rglob("*")) if p.is_file()}
    D._atomic_json(output / "protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--history-inventory", required=True, help="private JSON with explicit files list")
    parser.add_argument("--reference-bundle", required=True)
    args = parser.parse_args()
    files = json.loads(Path(args.history_inventory).read_text())["files"]
    protocol = prepare(args.output, args.tokenizer, files, args.reference_bundle)
    print(json.dumps({"status": protocol["status"], "endpoint": protocol["endpoint_per_run"],
                      "invariants": protocol["invariants"], "resource_projection": protocol["resource_projection"]}))


if __name__ == "__main__":
    main()
