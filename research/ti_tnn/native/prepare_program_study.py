"""Build packed arrays for the varied-program construction study, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import data as D
from . import program_study
from . import structured

BASE_COMMIT = "e876e284f7ba2f7726573518c70c35be607c3da7"
SPLIT_ROLES = {
    "train": "4096 varied length-2..6 programs with all initial values in [0,24)",
    "dev": "128 new disjoint program/initial pairs from the train distribution",
    "initial_values": "128 train instruction sequences with every scalar initial value in [24,32)",
    "unseen_composition": "128 length-2..6 programs containing multiply immediately followed by add",
    "longer_length": "128 length-8..12 programs without the reserved composition",
}


def _source_identity(seed: int) -> dict:
    source_paths = {
        "generator": Path(program_study.__file__),
        "builder": Path(__file__),
        "structured_schema": Path(structured.__file__),
        "packing": Path(D.__file__),
        "independent_executor": Path(D.__file__).parent / "independent" / "rescore_generations.py",
    }
    return {
        "kind": "new synthetic executable register programs",
        "generator": "native.program_study.generate_dataset",
        "seed": seed,
        "base_commit": BASE_COMMIT,
        "branch": "ti-tnn-native-varied-programs",
        "population_policy": "generated from the seed; no completed 32-example or old evaluation population is read",
        "sources": {
            name: {"path": str(path.resolve()), "sha256": D.sha256_file(path)}
            for name, path in source_paths.items()
        },
        "external_corpora_used": False,
        "qwen_or_old_evaluation_populations_used": False,
    }


def _answer_columns(example: structured.StructuredExample, tokenizer) -> tuple[list[int], int]:
    plain_ids, offsets = structured.encode_with_offsets(tokenizer, example.full_text)
    answer_start = len(example.prompt)
    answer_indices = [index for index, (start, end) in enumerate(offsets)
                      if end > start and start < len(example.full_text) and end > answer_start]
    if not answer_indices:
        raise D.DataError(f"{example.example_id}: no answer tokens were identified")
    return answer_indices, len(plain_ids)


def _audit_packed_split(split: str, examples, tokenized, arrays: dict[str, np.ndarray], stats: dict,
                        tokenizer, *, context_length: int, position_policy: str) -> tuple[list[dict], dict]:
    if stats["structured_examples_intact"] != len(examples) or stats["long_examples_excluded"]:
        raise D.DataError(f"{split}: packing dropped or excluded an example")
    if stats["tokens"] != sum(len(document.input_ids) for document in tokenized):
        raise D.DataError(f"{split}: packed token conservation failed")
    if np.any(arrays["labels"][arrays["key_padding_mask"]] != D.IGNORE_INDEX):
        raise D.DataError(f"{split}: padding has a supervised label")
    consumed = []
    answer_tokens = 0
    eos_tokens = 0
    for doc_id, (example, document) in enumerate(zip(examples, tokenized)):
        length = len(document.input_ids)
        plain_ids, _ = structured.encode_with_offsets(tokenizer, example.full_text)
        if document.input_ids.tolist() != [*plain_ids, D.EOS_TOKEN_ID]:
            raise D.DataError(f"{example.example_id}: packed tokenization differs from full prompt+target+EOS")
        locations = np.argwhere(arrays["doc_ids"] == doc_id)
        if len(locations) != length or len(set(locations[:, 0].tolist())) != 1:
            raise D.DataError(f"{example.example_id}: document was dropped or split across rows")
        row = int(locations[0, 0])
        columns = locations[:, 1].tolist()
        if columns != list(range(columns[0], columns[0] + length)):
            raise D.DataError(f"{example.example_id}: document tokens are not contiguous")
        start = columns[0]
        selection = slice(start, start + length)
        if arrays["key_padding_mask"][row, selection].any():
            raise D.DataError(f"{example.example_id}: live tokens are marked as padding")
        if not np.array_equal(arrays["input_ids"][row, selection], document.input_ids):
            raise D.DataError(f"{example.example_id}: packed input tokens differ from tokenization")
        if not np.all(arrays["source_ids"][row, selection] == D.SOURCE_IDS["structured"]):
            raise D.DataError(f"{example.example_id}: incorrect structured source id")
        expected_labels = document.input_ids.copy()
        expected_labels[0] = D.IGNORE_INDEX
        if not np.array_equal(arrays["labels"][row, selection], expected_labels):
            raise D.DataError(f"{example.example_id}: labels are not exact causal input IDs with only document start ignored")
        if position_policy == "document":
            expected_positions = np.arange(length, dtype=np.int32)
        else:
            expected_positions = np.arange(start, start + length, dtype=np.int32)
        if not np.array_equal(arrays["positions"][row, selection], expected_positions):
            raise D.DataError(f"{example.example_id}: positions violate {position_policy} policy")
        for name in ("route_targets", "route_mask", "state_targets", "state_mask"):
            if not np.array_equal(arrays[name][row, selection], getattr(document, name)):
                raise D.DataError(f"{example.example_id}: {name} was not packed intact")
        answer_indices, eos_index = _answer_columns(example, tokenizer)
        answer_indices = [*answer_indices, eos_index]
        supervised = arrays["labels"][row, start + np.asarray(answer_indices, dtype=np.int32)]
        if np.any(supervised == D.IGNORE_INDEX) or not np.array_equal(
                supervised, document.input_ids[np.asarray(answer_indices, dtype=np.int32)]):
            raise D.DataError(f"{example.example_id}: an answer or EOS token is not supervised")
        if arrays["labels"][row, start + eos_index] != D.EOS_TOKEN_ID:
            raise D.DataError(f"{example.example_id}: EOS label is missing")
        answer_tokens += len(answer_indices) - 1
        eos_tokens += 1
        consumed.append({
            "doc_id": doc_id,
            "example_id": example.example_id,
            "content_sha256": example.content_sha256,
            "program_length": len(example.program),
            "final_operation": example.program[-1].op,
            "initial": list(example.initial),
            "tokens_total": length,
            "tokens_consumed": length,
            "tokens_unconsumed": 0,
            "partial": False,
            "row": row,
            "start_column": start,
            "end_column_exclusive": start + length,
            "answer_token_columns": [start + index for index in answer_indices[:-1]],
            "eos_token_column": start + eos_index,
            "answer_and_eos_supervised": True,
        })
    expected_loss_tokens = sum(len(document.input_ids) - 1 for document in tokenized)
    if stats["loss_tokens"] != expected_loss_tokens:
        raise D.DataError(f"{split}: loss-token count does not include every noninitial document token")
    if np.any(arrays["positions"][arrays["key_padding_mask"]] != 0):
        raise D.DataError(f"{split}: padding positions are nonzero")
    summary = {
        "examples": len(examples),
        "rows": stats["sequences"],
        "tokens": stats["tokens"],
        "loss_tokens": stats["loss_tokens"],
        "expected_loss_tokens_from_intact_documents": expected_loss_tokens,
        "documents_dropped": 0,
        "documents_split": 0,
        "answer_tokens_supervised": answer_tokens,
        "eos_tokens_supervised": eos_tokens,
        "padding_label_leakage_tokens": 0,
        "noncausal_or_mismatched_labels": 0,
        "position_errors": 0,
        "maximum_document_tokens": max(len(document.input_ids) for document in tokenized),
        "context_length": context_length,
        "evidence_boundary": "static packing and supervision checks only; no learning result",
    }
    return consumed, summary


def _artifact_hashes(output: Path) -> dict:
    return {
        str(path.relative_to(output)): D.sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "audit.json"}
    }


def prepare(output, tokenizer_path, *, train: int = program_study.DEFAULT_COUNTS["train"],
            dev: int = program_study.DEFAULT_COUNTS["dev"],
            initial_values: int = program_study.DEFAULT_COUNTS["initial_values"],
            unseen_composition: int = program_study.DEFAULT_COUNTS["unseen_composition"],
            longer_length: int = program_study.DEFAULT_COUNTS["longer_length"],
            seed: int = program_study.DEFAULT_SEED, context_length: int = D.CONTEXT_LENGTH,
            position_policy: str = "document", train_evaluation_count: int = 128,
            intervention_count: int = 32) -> dict:
    """Generate, independently validate, tokenize, and intact-pack every split."""
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError(f"refusing to overwrite nonempty program-study output: {output}")
    if context_length != D.CONTEXT_LENGTH:
        raise ValueError(f"this study requires context length {D.CONTEXT_LENGTH}")
    if position_policy not in ("row", "document"):
        raise ValueError("position_policy must be 'row' or 'document'")
    tokenizer_path = Path(tokenizer_path).resolve()
    tokenizer = D.load_gpt2_tokenizer(tokenizer_path)
    dataset = program_study.generate_dataset(
        train=train, dev=dev, initial_values=initial_values,
        unseen_composition=unseen_composition, longer_length=longer_length, seed=seed)
    counts = {split: len(dataset[split]) for split in program_study.SPLIT_NAMES}
    invariants = program_study.assert_program_study_invariants(dataset, expected_counts=counts)
    target_validation = program_study.validate_targets(dataset)
    selections = program_study.select_subsets(
        dataset, seed=seed, train_evaluation_count=train_evaluation_count,
        intervention_count=intervention_count)
    baselines = program_study.baseline_report(dataset)
    tokenized = {split: [] for split in program_study.SPLIT_NAMES}
    for split in program_study.SPLIT_NAMES:
        for example in dataset[split]:
            document = D.tokenize_structured(example, tokenizer)
            if len(document.input_ids) > context_length:
                raise D.DataError(
                    f"{example.example_id}: {len(document.input_ids)} tokens exceed context {context_length}")
            tokenized[split].append(document)

    output.mkdir(parents=True, exist_ok=True)
    structured_files = program_study.save_structured_jsonl(dataset, output / "structured")
    baselines_path = output / "structured" / "baselines.json"
    D._atomic_json(baselines_path, baselines)
    baseline_entry = {
        "path": str(baselines_path),
        "sha256": D.sha256_file(baselines_path),
        "report": baselines,
    }
    selections_path = output / "selections.json"
    D._atomic_json(selections_path, selections)
    selection_entry = {
        "path": str(selections_path),
        "sha256": D.sha256_file(selections_path),
        **selections,
    }
    source_identity = _source_identity(seed)
    tokenizer_identity = {
        **D.TOKENIZER_SPEC,
        "path": str(tokenizer_path),
        "sha256": D.sha256_file(tokenizer_path),
        "hash_verified_by": "native.data.load_gpt2_tokenizer",
    }
    manifest = {
        "format": "native-varied-program-study-data-v1",
        "evidence_boundary": "data construction and static supervision checks only; no training or learning evaluation",
        "seed": seed,
        "context_length": context_length,
        "packing": "intact",
        "position_policy": position_policy,
        "structured_overflow": "error",
        "ignore_index": D.IGNORE_INDEX,
        "eos_token_id": D.EOS_TOKEN_ID,
        "model_input_arrays": list(D.MODEL_INPUT_ARRAYS),
        "target_arrays": list(D.TARGET_ARRAYS),
        "label_rule": "labels[t]=input_ids[t]; only padding and each intact document's first token are ignored",
        "split_roles": SPLIT_ROLES,
        "counts": counts,
        "source_identity": source_identity,
        "tokenizer": tokenizer_identity,
        "structured_files": structured_files,
        "baselines": baseline_entry,
        "selections": selection_entry,
        "construction_invariants": invariants,
        "target_validation": target_validation,
        "splits": {},
        "audit_file": str(output / "audit.json"),
    }
    for split in program_study.SPLIT_NAMES:
        runs = [D.TokenRun(doc_id, document, len(document.input_ids))
                for doc_id, document in enumerate(tokenized[split])]
        arrays, stats = D.pack_runs(
            runs, context_length=context_length, packing="intact",
            structured_overflow="error", position_policy=position_policy)
        consumed, packing_audit = _audit_packed_split(
            split, dataset[split], tokenized[split], arrays, stats, tokenizer,
            context_length=context_length, position_policy=position_policy)
        array_files = {
            name: D._write_array(output / split, name, arrays[name]) for name in D.ARRAY_NAMES}
        consumed_path = output / f"{split}_consumed_documents.jsonl"
        D._atomic_jsonl(consumed_path, consumed)
        manifest["splits"][split] = {
            **stats,
            "rows": stats["sequences"],
            "examples": len(dataset[split]),
            "arrays": array_files,
            "consumed_documents": len(consumed),
            "consumed_document_file": str(consumed_path),
            "consumed_document_sha256": D.sha256_file(consumed_path),
            "structured_jsonl_sha256": structured_files[split]["sha256"],
            "construction_checks": packing_audit,
        }
    train_rows = manifest["splits"]["train"]["rows"]
    manifest["train_epoch"] = {
        "batch_size": 8,
        "drop_last": False,
        "rows": train_rows,
        "loss_tokens_per_epoch": manifest["splits"]["train"]["loss_tokens"],
        "updates_per_epoch": math.ceil(train_rows / 8),
        "rule": "one pass over every packed train row, including the final partial batch; sum actual supervised labels",
    }
    manifest["artifact_sha256"] = _artifact_hashes(output)
    D._atomic_json(output / "manifest.json", manifest)
    audit = {
        "format": "native-varied-program-study-audit-v1",
        "evidence_boundary": manifest["evidence_boundary"],
        "manifest_path": str(output / "manifest.json"),
        "manifest_sha256": D.sha256_file(output / "manifest.json"),
        "split_roles": SPLIT_ROLES,
        "source_identity": source_identity,
        "tokenizer": tokenizer_identity,
        "structured_files": structured_files,
        "baselines": baseline_entry,
        "selections": selection_entry,
        "construction_invariants": invariants,
        "target_validation": target_validation,
        "splits": manifest["splits"],
        "train_epoch": manifest["train_epoch"],
        "artifact_sha256": _artifact_hashes(output),
    }
    D._atomic_json(output / "audit.json", audit)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m native.prepare_program_study", description=__doc__.split("\n\n")[0])
    result.add_argument("--output", required=True, help="new or empty data output directory")
    result.add_argument("--tokenizer", required=True, help="path to the hash-pinned GPT2 tokenizer.json")
    result.add_argument("--train", type=int, default=program_study.DEFAULT_COUNTS["train"])
    result.add_argument("--dev", type=int, default=program_study.DEFAULT_COUNTS["dev"])
    result.add_argument("--initial-values", type=int, default=program_study.DEFAULT_COUNTS["initial_values"])
    result.add_argument("--unseen-composition", type=int, default=program_study.DEFAULT_COUNTS["unseen_composition"])
    result.add_argument("--longer-length", type=int, default=program_study.DEFAULT_COUNTS["longer_length"])
    result.add_argument("--seed", type=int, default=program_study.DEFAULT_SEED)
    result.add_argument("--position-policy", choices=("row", "document"), default="document",
                        help="document resets positions at intact example boundaries; row preserves absolute row positions")
    result.add_argument("--train-evaluation-count", type=int, default=128)
    result.add_argument("--intervention-count", type=int, default=32)
    return result


def build(args: argparse.Namespace) -> dict:
    return prepare(
        args.output, args.tokenizer, train=args.train, dev=args.dev,
        initial_values=args.initial_values, unseen_composition=args.unseen_composition,
        longer_length=args.longer_length, seed=args.seed, position_policy=args.position_policy,
        train_evaluation_count=args.train_evaluation_count, intervention_count=args.intervention_count)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    manifest = build(args)
    summary = {
        split: {
            "examples": values["examples"],
            "rows": values["rows"],
            "tokens": values["tokens"],
            "loss_tokens": values["loss_tokens"],
        }
        for split, values in manifest["splits"].items()
    }
    summary["train_epoch"] = manifest["train_epoch"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
