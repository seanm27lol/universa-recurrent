"""Freeze the existing mixed preparation and execution binding on CPU only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import sys
import subprocess

from . import data as D, ledger, run_initial_state_study as I, run_mixed_learning as R


def verify_sampler_cost(resource, preparation, source_root):
    """Bind CPU setup and a conservative live-sampler proxy to the same arrays/code.

    Historical throughput already includes the smaller sampler. Retaining that
    cost and adding the full new planning walk deliberately overcounts it rather
    than assuming occupancy adjustment covers the larger row permutation.
    """
    basis = resource["cpu_sampler_measurement"]
    D.verify_file(basis["path"], basis["sha256"])
    measured = I.read_json(basis["path"])
    if measured["preparation_sha256"] != D.sha256_file(Path(preparation) / "manifest.json"):
        raise RuntimeError("CPU sampler measurement uses another preparation")
    if measured["endpoint"] != {"updates": R.ENDPOINT["updates"], "loss_tokens": R.ENDPOINT["full_label_tokens"]}:
        raise RuntimeError("CPU sampler measurement did not plan the full endpoint")
    endpoint = I.read_json(Path(preparation) / "manifest.json")["endpoint"]
    if (measured["identities"]["train"] != endpoint["data_identity"]
            or measured["identities"]["order"] != endpoint["order_identity"]):
        raise RuntimeError("CPU sampler measurement data/order identity mismatch")
    required = {"research/ti_tnn/native/train.py", "research/ti_tnn/native/sampler.py"}
    if set(basis["source_files"]) != required:
        raise RuntimeError("CPU sampler measurement source binding incomplete")
    I.verify_hashes(source_root, basis["source_files"])
    commit = measured["source_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise RuntimeError("invalid CPU measurement source commit")
    for relative, digest in basis["source_files"].items():
        original = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=source_root)
        if hashlib.sha256(original).hexdigest() != digest:
            raise RuntimeError("CPU measurement source differs from the bound sampler")
    setup = measured["total_seconds"]
    live = measured["timings"]["clone_and_endpoint_planning_seconds"]
    stages = measured["timings"]
    if (set(stages) != {"dataset_load_and_checks_seconds", "train_sampler_seconds", "dev_sampler_seconds",
                       "clone_and_endpoint_planning_seconds"}
            or not all(math.isfinite(n) and n >= 0 for n in stages.values())
            or setup + 1e-6 < math.fsum(stages.values())):
        raise ValueError("CPU setup total must include all measured stages")
    if not all(math.isfinite(n) and n > 0 for n in (setup, live)) or setup < live:
        raise ValueError("invalid CPU sampler timings")
    for arm in resource["arms"].values():
        if arm["cpu_training_setup_seconds"] != setup or arm["cpu_live_sampler_allowance_seconds"] != live:
            raise RuntimeError("full CPU sampler cost omitted or changed")
        expected = sum(arm[k] for k in ("train_seconds", "complete_evaluation_seconds", "reconstruction_seconds")) + setup + live
        if not math.isclose(arm["total_seconds"], expected, rel_tol=0, abs_tol=1e-8):
            raise RuntimeError("complete resource components do not sum to total")
    return basis


def freeze(preparation, resource_proposal, output):
    preparation, resource_proposal, output = map(lambda p: Path(p).resolve(), (preparation, resource_proposal, output))
    if output.exists():
        raise FileExistsError("preserve the existing frozen binding; use a versioned new directory")
    root = Path(__file__).resolve().parents[3]
    commit = I.source_commit(root, clean=True)
    manifest = I.read_json(preparation / "manifest.json")
    resource = I.read_json(resource_proposal)
    if (resource["preparation_sha256"] != D.sha256_file(preparation / "manifest.json")
            or resource["actual_fixed_endpoint"] != manifest["endpoint"]):
        raise RuntimeError("resource/preparation endpoint binding mismatch")
    I.verify_hashes(preparation, manifest["artifacts"])
    cpu_cost_basis = verify_sampler_cost(resource, preparation, root)
    endpoint = manifest["endpoint"]
    observed = {"requested_loss_tokens": endpoint["requested_loss_tokens"],
        "full_label_tokens": endpoint["actual_loss_tokens"], "updates": endpoint["updates"],
        "cursor": endpoint["cursor"], "batch_size": endpoint["batch_size"]}
    if observed != R.ENDPOINT:
        raise RuntimeError("fixed endpoint changed")
    # v1 is retained as historical provenance; its 35% planning fields are superseded
    # by the shared ledger's margin. Copy only the disjoint raw cost components.
    raw_fields = ("train_seconds", "complete_evaluation_seconds", "complete_evaluation_components_seconds",
        "reconstruction_seconds", "checkpoints", "complete_train_dev_passes", "seconds_per_complete_dev_pass",
        "original_profile_loss_tokens_per_second", "transferred_loss_tokens_per_second",
        "cpu_training_setup_seconds", "cpu_live_sampler_allowance_seconds")
    p = {"format": R.FORMAT, "status": "frozen_cpu_preparation_no_execution_authorization",
        "modes": list(R.MODES), "objective": "full_sequence", "model_seed": 20261108, "sampler_seed": 20261108,
        "initialization": {"kind": "fresh_random", "parent_checkpoint": None, "seed": 20261108},
        "endpoint_per_run": observed, "ordered_exposure": copy.deepcopy(endpoint),
        "lambda_balance": .01, "lambda_route": .1, "lambda_state": .1,
        "optimizer": {"learning_rate": .0003, "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1.},
        "scheduler": {"warmup_fraction": .02, "final_lr_fraction": .1},
        "checkpoint_every": 500, "dev_every": 1000, "dev_batches": 16,
        "selection": "final endpoint only; every failed/missing arm remains explicitly incomplete",
        "preparation_sha256": D.sha256_file(preparation / "manifest.json"),
        "artifact_sha256": manifest["artifacts"], "tokenizer_sha256": D.TOKENIZER_SPEC["sha256"],
        "capability_loss_tokens": {s: v["loss_tokens"] for s, v in manifest["capability"].items()},
        "dataset_contract": {"historical_manifest_sha256": manifest["historical_manifest_sha256"],
            "register_protocol_sha256": manifest["register_protocol_sha256"],
            "packing": "intact", "position_policy": "document", "ordinary_context": "row-local chunks",
            "whole_program_underfill": {s: v["allocation"]["structured"]["underfill"] for s, v in manifest["splits"].items()},
            "source_loss_tokens": endpoint["source_loss_tokens"],
            "actual_source_loss_proportions": {s: n / endpoint["actual_loss_tokens"] for s, n in endpoint["source_loss_tokens"].items()},
            "exclusions": "pinned license labels, exact-content dedup before unchanged historical 1% dev split; no new confirmation claim",
            "auxiliary_schedule": manifest["auxiliary_schedule"]},
        "evaluation_contract": {"language_code": {"documents_each": 128, "source_selection": "content-SHA order, held-out documents",
            "context": "up to 256 document-beginning tokens, one row per document",
            "metric": "sum main next-token CE divided by nonmasked shifted source-token count; auxiliary penalties excluded",
            "mask": "padding and first token excluded; no cross-document attention",
            "generations_each": 64, "prompt_tokens": 128, "max_new_tokens": 32,
            "code_metric": "32-reference-token exact match and positional token agreement; no code execution or semantic correctness claim"},
            "synthetic": {"populations": list(R.POPULATIONS), "examples_each": 128, "max_new_tokens": 48,
                "role": "historical secondary monitor; #80 train names are not mixed training fit; no perfection gate"},
            "state": "all four passes/slots, separate readout and answer CE; not causal use",
            "causal": "not run; no shaping or WSP", "numerical": "bounded CPU FP32, retain strict-allclose failures"},
        "resource_projection": {"per_model_per_condition_seconds": {m: {"total": resource["arms"][m]["total_seconds"],
            "components": {k: resource["arms"][m][k] for k in raw_fields}} for m in R.MODES},
            "common_seconds": {"loading_warmup_and_reserved_identity_preflight_allowance": 540.},
            "shutdown_allowance_hours": .05, "checkpoint_storage_bytes_all_runs": resource["checkpoint_bytes"],
            "additional_data_logs_export_storage_allowance_bytes": resource["additional_storage_allowance_bytes"],
            "required_free_bytes": 64 * 1024**3, "planning_memory_gib": 12,
            "source_proposal_sha256": D.sha256_file(resource_proposal),
            "cpu_sampler_measurement": cpu_cost_basis,
            "historical_profile_protocol_sha256": resource["source_measurement_protocol_sha256"],
            "limits": "historical timing transfer, not fresh GPU profiles; CPU preparation outside reservation reported separately; "
                      "full CPU setup plus cloned endpoint-planning duration as conservative live-sampler proxy; "
                      "old smaller sampler costs deliberately retained too; no fresh GPU throughput measurement; "
                      "all in-reservation evaluation/recovery costs count; no added profiling optimizer steps; shared 50% watchdog margin retained"}}
    native = Path(__file__).parent
    files = [*native.glob("*.py"), *native.joinpath("independent").glob("*.py"), native / "configs/engineering.json",
             root / "requirements-native-lock.txt", native.parent / "scaling/ledger.py"]
    p["execution"] = {"study_id": R.STUDY_ID, "source_commit": commit,
        "source_files": {str(f.relative_to(root)): D.sha256_file(f) for f in sorted(files)},
        "python": sys.executable, "environment_prefix": sys.prefix, "installed_versions": I.locked_environment(root),
        "preparation": str(preparation), "output": str(ledger.shared_runs().resolve().parent / "ti_tnn_native_mixed_learning_v1"),
        "allocation_hours": R.ALLOCATION, "native_cap_hours": ledger.ADDITIONAL_HOURS,
        "training_configs": R.training_configs(p, preparation)}
    p["resource_projection"]["admission_at_freeze"] = R.projection(p, {}, 0., ledger.usage()["native_remaining_hours"])
    output.mkdir(parents=True, exist_ok=False)
    I.write_json(output / "protocol.json", p)
    return p


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "resource-proposal", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    p = freeze(args.preparation, args.resource_proposal, args.output)
    print(json.dumps({"protocol_sha256": D.sha256_file(Path(args.output) / "protocol.json"),
        "source_commit": p["execution"]["source_commit"], "admission": p["resource_projection"]["admission_at_freeze"]}, indent=2))


if __name__ == "__main__":
    main()
