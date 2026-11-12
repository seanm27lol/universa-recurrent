"""Build and inspect answer-only loss companions for packed program examples.

A companion marks the 16 target-text tokens and final EOS in every intact document.
Its numeric-value and formatting/EOS partitions distinguish register results from the
surrounding syntax. The original packed arrays stay immutable: model inputs retain the
complete prompt, and only an external loss-label selection consumes these masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .data import (
    ARRAY_NAMES,
    ARRAY_DTYPES,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    MODEL_INPUT_ARRAYS,
    SOURCE_IDS,
    TORCH_LONG_ARRAYS,
    PackedDataset,
    load_gpt2_tokenizer,
    sha256_file,
)
from .structured import encode_with_offsets

FORMAT = "native-program-answer-mask-companions-v1"
PROGRAM_SPLITS = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
ANSWER_MASK = "answer_mask"
ANSWER_VALUE_MASK = "answer_value_mask"
FORMATTING_EOS_MASK = "formatting_eos_mask"
MASK_ARRAY_NAMES = (ANSWER_MASK, ANSWER_VALUE_MASK, FORMATTING_EOS_MASK)
MASK_DTYPES = {name: np.dtype("bool") for name in MASK_ARRAY_NAMES}
EXPECTED_PER_DOCUMENT = {
    ANSWER_MASK: 17,
    ANSWER_VALUE_MASK: 4,
    FORMATTING_EOS_MASK: 13,
}
FULL_TRAIN_EPOCH = {
    "full_label_exposure_tokens": 348165,
    "answer_eos_tokens": 69632,
    "answer_value_tokens": 16384,
    "formatting_eos_tokens": 53248,
}
_TARGET_PATTERN = re.compile(
    r"r0=(0|[1-9][0-9]*) r1=(0|[1-9][0-9]*) "
    r"r2=(0|[1-9][0-9]*) r3=(0|[1-9][0-9]*)"
)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank JSONL line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            records.append(record)
    return records


def _file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def masks_to_labels(labels, answer_mask):
    """Return loss labels outside ``answer_mask`` as IGNORE_INDEX, without changing inputs."""
    if labels.shape != answer_mask.shape:
        raise ValueError(f"label shape {labels.shape} differs from answer-mask shape {answer_mask.shape}")
    try:
        import torch

        if torch.is_tensor(labels):
            if not torch.is_tensor(answer_mask):
                raise TypeError("tensor labels require a tensor answer mask")
            return labels.masked_fill(~answer_mask.bool(), IGNORE_INDEX)
    except ImportError:
        pass
    return np.where(np.asarray(answer_mask, dtype=bool), labels, IGNORE_INDEX).astype(labels.dtype, copy=False)


def _numeric_value_spans(record: Mapping) -> tuple[tuple[int, int], ...]:
    target = record.get("target_text")
    prompt = record.get("prompt")
    if not isinstance(target, str) or not isinstance(prompt, str):
        raise ValueError(f"{record.get('example_id')}: prompt and target_text must be strings")
    match = _TARGET_PATTERN.fullmatch(target)
    if match is None or any(int(value) >= 32 for value in match.groups()):
        raise ValueError(f"{record.get('example_id')}: target is not four canonical modulo-32 register assignments")
    return tuple((len(prompt) + match.start(index), len(prompt) + match.end(index)) for index in range(1, 5))


def _validate_document_location(arrays: Mapping[str, np.ndarray], item: Mapping, token_count: int) -> tuple[int, int]:
    doc_id = item.get("doc_id")
    locations = np.argwhere(arrays["doc_ids"] == doc_id)
    if len(locations) != token_count or len(set(locations[:, 0].tolist())) != 1:
        raise ValueError(f"{item.get('example_id')}: consumed document is not intact with {token_count} tokens")
    row = int(locations[0, 0])
    columns = locations[:, 1].tolist()
    start = columns[0]
    if columns != list(range(start, start + token_count)):
        raise ValueError(f"{item.get('example_id')}: document columns are not contiguous")
    if item.get("row") != row or item.get("start_column") != start or item.get("end_column_exclusive") != start + token_count:
        raise ValueError(f"{item.get('example_id')}: consumed row/columns differ from packed doc_ids")
    if arrays["key_padding_mask"][row, start:start + token_count].any():
        raise ValueError(f"{item.get('example_id')}: live document tokens overlap padding")
    if not np.all(arrays["source_ids"][row, start:start + token_count] == SOURCE_IDS["structured"]):
        raise ValueError(f"{item.get('example_id')}: document is not marked as structured")
    return row, start


def build_split_answer_masks(
    dataset: PackedDataset,
    consumed: Sequence[Mapping],
    source_records: Sequence[Mapping],
    tokenizer,
    *,
    split: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Validate and construct one split's three disjoint boolean mask arrays in memory."""
    arrays = dataset.arrays
    shape = arrays["input_ids"].shape
    masks = {name: np.zeros(shape, dtype=np.bool_) for name in MASK_ARRAY_NAMES}
    by_id = {record.get("example_id"): record for record in source_records}
    if len(by_id) != len(source_records):
        raise ValueError(f"{split}: duplicate structured example IDs")
    seen_docs = set()
    documents = []
    for item in consumed:
        example_id = item.get("example_id")
        if example_id not in by_id:
            raise ValueError(f"{split}: consumed document {example_id!r} has no source record")
        record = by_id[example_id]
        doc_id = item.get("doc_id")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) or doc_id in seen_docs:
            raise ValueError(f"{split}: duplicate or invalid consumed doc_id {doc_id!r}")
        seen_docs.add(doc_id)
        full_text = record["prompt"] + record["target_text"]
        content_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        if item.get("content_sha256") != content_sha256:
            raise ValueError(f"{example_id}: consumed content hash differs from source text")
        plain_ids, offsets = encode_with_offsets(tokenizer, full_text)
        document_ids = [*map(int, plain_ids), EOS_TOKEN_ID]
        if item.get("tokens_total") != len(document_ids) or item.get("tokens_consumed") != len(document_ids):
            raise ValueError(f"{example_id}: consumed token counts differ from full prompt+target+EOS")
        row, start = _validate_document_location(arrays, item, len(document_ids))
        if not np.array_equal(arrays["input_ids"][row, start:start + len(document_ids)], document_ids):
            raise ValueError(f"{example_id}: packed input IDs differ from the pinned tokenization")
        answer_start = len(record["prompt"])
        answer_indices = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and left < len(full_text) and right > answer_start
        ]
        expected_answer_columns = [start + index for index in answer_indices]
        if list(item.get("answer_token_columns", [])) != expected_answer_columns:
            raise ValueError(f"{example_id}: consumed answer columns differ from tokenizer offsets")
        if item.get("eos_token_column") != start + len(plain_ids):
            raise ValueError(f"{example_id}: consumed EOS column differs from the full tokenization")
        answer_columns = [*expected_answer_columns, start + len(plain_ids)]
        if len(answer_columns) != EXPECTED_PER_DOCUMENT[ANSWER_MASK]:
            raise ValueError(
                f"{example_id}: found {len(answer_columns)} answer+EOS tokens, expected 17"
            )
        value_indices = set()
        for left, right in _numeric_value_spans(record):
            overlapping = [
                index for index in answer_indices
                if offsets[index][0] >= left and offsets[index][1] <= right
            ]
            if len(overlapping) != 1 or offsets[overlapping[0]] != (left, right):
                raise ValueError(f"{example_id}: numeric value span {(left, right)} is not exactly one token")
            value_indices.add(overlapping[0])
        if len(value_indices) != EXPECTED_PER_DOCUMENT[ANSWER_VALUE_MASK]:
            raise ValueError(f"{example_id}: numeric register values do not identify exactly four tokens")
        value_columns = {start + index for index in value_indices}
        answer_set = set(answer_columns)
        formatting_columns = answer_set - value_columns
        if len(formatting_columns) != EXPECTED_PER_DOCUMENT[FORMATTING_EOS_MASK]:
            raise ValueError(f"{example_id}: formatting/EOS complement does not contain 13 tokens")
        masks[ANSWER_MASK][row, answer_columns] = True
        masks[ANSWER_VALUE_MASK][row, sorted(value_columns)] = True
        masks[FORMATTING_EOS_MASK][row, sorted(formatting_columns)] = True
        expected_labels = np.asarray(document_ids, dtype=np.int32)
        expected_labels[0] = IGNORE_INDEX
        if not np.array_equal(arrays["labels"][row, start:start + len(document_ids)], expected_labels):
            raise ValueError(f"{example_id}: packed labels differ at document start or within the document")
        documents.append({
            "doc_id": doc_id,
            "example_id": example_id,
            "row": row,
            "start_column": start,
            "end_column_exclusive": start + len(document_ids),
            "answer_eos_columns": answer_columns,
            "answer_value_columns": sorted(value_columns),
            "formatting_eos_columns": sorted(formatting_columns),
            "first_answer_column": answer_columns[0],
            "prompt_columns_unmasked": list(range(start, answer_columns[0])),
        })
    live_docs = set(map(int, arrays["doc_ids"][~arrays["key_padding_mask"]].tolist()))
    if seen_docs != live_docs or len(consumed) != len(source_records):
        raise ValueError(f"{split}: consumed documents, source records, and packed live documents differ")
    union = masks[ANSWER_VALUE_MASK] | masks[FORMATTING_EOS_MASK]
    if not np.array_equal(union, masks[ANSWER_MASK]) or np.any(
        masks[ANSWER_VALUE_MASK] & masks[FORMATTING_EOS_MASK]
    ):
        raise AssertionError(f"{split}: answer mask partitions are not disjoint and exhaustive")
    if np.any(masks[ANSWER_MASK] & arrays["key_padding_mask"]):
        raise AssertionError(f"{split}: an answer mask overlaps padding")
    if np.any(arrays["labels"][masks[ANSWER_MASK]] == IGNORE_INDEX):
        raise AssertionError(f"{split}: an answer position lacks an LM label")
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    for name, count in counts.items():
        expected = EXPECTED_PER_DOCUMENT[name] * len(consumed)
        if count != expected:
            raise AssertionError(f"{split}: {name} has {count} tokens, expected {expected}")
    return masks, {
        "documents": len(consumed),
        "rows": len(dataset),
        "full_label_exposure_tokens": int(np.count_nonzero(arrays["labels"] != IGNORE_INDEX)),
        "answer_eos_tokens": counts[ANSWER_MASK],
        "answer_value_tokens": counts[ANSWER_VALUE_MASK],
        "formatting_eos_tokens": counts[FORMATTING_EOS_MASK],
        "mask_union_valid": True,
        "all_answer_positions_have_lm_labels": True,
        "document_positions": documents,
    }


def _write_new_array(directory: Path, name: str, array: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npy"
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "dtype": str(array.dtype), "shape": list(array.shape)}


def build_answer_companions(data_root, output_directory, *, tokenizer=None, tokenizer_path=None) -> dict:
    """Write all five companion splits into a new or empty directory; never train a model."""
    root = Path(data_root).resolve()
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to write companions into nonempty output: {output}")
    manifest_path = root / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("position_policy") != "document":
        raise ValueError("answer companions require intact documents with document-local positions")
    tokenizer_path = Path(tokenizer_path or source_manifest["tokenizer"]["path"]).resolve()
    tokenizer = load_gpt2_tokenizer(tokenizer_path) if tokenizer is None else tokenizer
    source_hashes = {manifest_path: sha256_file(manifest_path), tokenizer_path: sha256_file(tokenizer_path)}
    built = {}
    split_sources = {}
    for split in PROGRAM_SPLITS:
        split_manifest = source_manifest["splits"][split]
        consumed_path = root / f"{split}_consumed_documents.jsonl"
        records_path = root / "structured" / f"{split}.jsonl"
        for path in (consumed_path, records_path):
            source_hashes[path] = sha256_file(path)
        if split_manifest.get("consumed_document_sha256") != source_hashes[consumed_path]:
            raise ValueError(f"{split}: consumed-document hash differs from the source manifest")
        if split_manifest.get("structured_jsonl_sha256") != source_hashes[records_path]:
            raise ValueError(f"{split}: structured-record hash differs from the source manifest")
        dataset = PackedDataset(root, split)
        original_arrays = {}
        for name in ARRAY_NAMES:
            path = root / split / f"{name}.npy"
            digest = sha256_file(path)
            source_hashes[path] = digest
            expected = split_manifest["arrays"][name]
            if digest != expected["sha256"]:
                raise ValueError(f"{split}/{name}: array hash differs from the source manifest")
            if list(dataset.arrays[name].shape) != list(expected["shape"]) or dataset.arrays[name].dtype != ARRAY_DTYPES[name]:
                raise ValueError(f"{split}/{name}: array shape or dtype differs from the source manifest")
            original_arrays[name] = {
                "path": str(path.resolve()), "sha256": digest,
                "shape": list(dataset.arrays[name].shape), "dtype": str(dataset.arrays[name].dtype),
            }
        consumed = _read_jsonl(consumed_path)
        records = _read_jsonl(records_path)
        masks, audit = build_split_answer_masks(dataset, consumed, records, tokenizer, split=split)
        if split == "train" and len(consumed) == 4096:
            actual = {key: audit[key] for key in FULL_TRAIN_EPOCH}
            if actual != FULL_TRAIN_EPOCH:
                raise ValueError(f"full train epoch exposure differs from the required contract: {actual}")
        built[split] = (masks, audit)
        split_sources[split] = {
            "consumed_documents": _file_identity(consumed_path),
            "structured_records": _file_identity(records_path),
            "original_arrays": original_arrays,
        }
    for path, digest in source_hashes.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"source changed during companion construction: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to write companions into nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "format": FORMAT,
        "status": "companion_data_only_no_model_or_training",
        "source_data_root": str(root),
        "source_manifest": _file_identity(manifest_path),
        "tokenizer": {
            **dict(source_manifest["tokenizer"]),
            "path": str(tokenizer_path),
            "sha256": source_hashes[tokenizer_path],
        },
        "model_input_arrays_unchanged": list(MODEL_INPUT_ARRAYS),
        "mask_arrays": list(MASK_ARRAY_NAMES),
        "mask_dtypes": {name: "bool" for name in MASK_ARRAY_NAMES},
        "mask_use": "external loss-label selection only; never model input, route target, or state target",
        "prompt_visibility": "all prompt input_ids, positions, doc_ids, and key_padding_mask values remain unchanged and visible",
        "first_answer_token_included": True,
        "expected_per_document": EXPECTED_PER_DOCUMENT,
        "full_train_epoch_contract": FULL_TRAIN_EPOCH,
        "splits": {},
    }
    for split in PROGRAM_SPLITS:
        masks, audit = built[split]
        array_files = {name: _write_new_array(output / split, name, masks[name]) for name in MASK_ARRAY_NAMES}
        full_contract = dict(FULL_TRAIN_EPOCH) if split == "train" else None
        actual_contract = {key: audit[key] for key in FULL_TRAIN_EPOCH} if split == "train" else None
        result["splits"][split] = {
            **split_sources[split],
            **audit,
            "arrays": array_files,
            "full_train_epoch_expected": full_contract,
            "full_train_epoch_actual": actual_contract,
            "full_train_epoch_contract_applies": split == "train" and audit["documents"] == 4096,
        }
    encoded = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with (output / "manifest.json").open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return result


class AnswerMaskedDataset(PackedDataset):
    """Packed arrays plus external masks; ``batch(indices)`` preserves the original API options."""

    def __init__(self, path, split: str | None = None, mask_directory=None):
        super().__init__(path, split)
        if mask_directory is None:
            directory = Path(path).resolve() / "answer_masks"
        else:
            directory = Path(mask_directory).resolve()
        if not (directory / f"{ANSWER_MASK}.npy").is_file() and split is not None:
            directory = directory / split
        self.mask_directory = directory
        self.masks = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in MASK_ARRAY_NAMES
        }
        for name, mask in self.masks.items():
            if mask.shape != self.arrays["input_ids"].shape or mask.dtype != MASK_DTYPES[name]:
                raise ValueError(f"{directory}/{name}.npy has inconsistent shape or dtype")
        if not np.array_equal(self.masks[ANSWER_VALUE_MASK] | self.masks[FORMATTING_EOS_MASK], self.masks[ANSWER_MASK]):
            raise ValueError(f"{directory}: answer mask union is inconsistent")
        if np.any(self.masks[ANSWER_VALUE_MASK] & self.masks[FORMATTING_EOS_MASK]):
            raise ValueError(f"{directory}: numeric and formatting/EOS masks overlap")
        if np.any(self.arrays["labels"][self.masks[ANSWER_MASK]] == IGNORE_INDEX):
            raise ValueError(f"{directory}: an answer position has no LM label")

    def __getitem__(self, index):
        return {**super().__getitem__(index), **{name: mask[index] for name, mask in self.masks.items()}}

    def batch(self, indices, *, as_torch: bool = True, device=None, model_inputs_only: bool = False):
        result = super().batch(indices, as_torch=as_torch, device=device, model_inputs_only=model_inputs_only)
        if model_inputs_only:
            return result
        for name, values in self.masks.items():
            selected = np.ascontiguousarray(values[indices])
            if as_torch:
                import torch

                result[name] = torch.from_numpy(selected).to(device) if device is not None else torch.from_numpy(selected)
            else:
                result[name] = selected
        return result

    @property
    def answer_loss_tokens(self) -> int:
        return int(self.masks[ANSWER_MASK].sum())


def _dataset_arrays(dataset) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    if isinstance(dataset, Mapping):
        return dataset, dataset
    masks = getattr(dataset, "masks", None)
    if masks is None:
        raise ValueError("coverage requires an AnswerMaskedDataset or a mapping containing mask arrays")
    return dataset.arrays, masks


def _position_stats(values: Sequence[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
    }


def answer_position_coverage(dataset) -> dict:
    """Measure answer labels and absent auxiliary targets at the same packed positions."""
    arrays, masks = _dataset_arrays(dataset)
    doc_ids = arrays["doc_ids"]
    live = ~arrays["key_padding_mask"]
    ordered_docs = list(dict.fromkeys(map(int, doc_ids[live].tolist())))
    records = []
    for doc_id in ordered_docs:
        locations = np.argwhere(doc_ids == doc_id)
        if len(set(locations[:, 0].tolist())) != 1:
            raise ValueError(f"doc_id {doc_id}: coverage requires an intact document in one row")
        row = int(locations[0, 0])
        columns = locations[:, 1]
        answer_columns = columns[masks[ANSWER_MASK][row, columns]]
        if len(answer_columns) != 17:
            raise ValueError(f"doc_id {doc_id}: expected 17 answer+EOS positions, found {len(answer_columns)}")
        value_count = int(masks[ANSWER_VALUE_MASK][row, answer_columns].sum())
        format_count = int(masks[FORMATTING_EOS_MASK][row, answer_columns].sum())
        label_count = int(np.count_nonzero(arrays["labels"][row, answer_columns] != IGNORE_INDEX))
        route = arrays["route_mask"][row, answer_columns]
        state = arrays["state_mask"][row, answer_columns]
        route_anchor_columns = columns[np.any(arrays["route_mask"][row, columns], axis=-1)]
        state_anchor_columns = columns[np.any(arrays["state_mask"][row, columns], axis=(-2, -1))]
        if not len(route_anchor_columns) or not len(state_anchor_columns):
            raise ValueError(f"doc_id {doc_id}: no instruction anchor in route/state masks")
        last_route_anchor = int(route_anchor_columns[-1])
        last_state_anchor = int(state_anchor_columns[-1])
        if last_route_anchor != last_state_anchor:
            raise ValueError(f"doc_id {doc_id}: route and state final anchors differ")
        first_answer = int(answer_columns[0])
        first_answer_position = int(arrays["positions"][row, first_answer])
        anchor_position = int(arrays["positions"][row, last_route_anchor])
        distance = first_answer_position - anchor_position
        if distance <= 0:
            raise ValueError(f"doc_id {doc_id}: final instruction anchor is not upstream of the answer")
        records.append({
            "doc_id": doc_id,
            "answer_eos_positions": 17,
            "answer_eos_positions_with_lm_labels": label_count,
            "answer_value_positions": value_count,
            "formatting_eos_positions": format_count,
            "route_mask_true_at_answer_positions": int(route.sum()),
            "route_positions_with_any_target_at_answer": int(np.any(route, axis=-1).sum()),
            "state_mask_true_at_answer_positions": int(state.sum()),
            "state_positions_with_any_target_at_answer": int(np.any(state, axis=(-2, -1)).sum()),
            "first_answer_position": first_answer_position,
            "last_instruction_anchor": anchor_position,
            "downstream_distance": distance,
            "intervening_tokens_between_anchor_and_answer": distance - 1,
        })
    result = {
        "documents": len(records),
        "full_lm_label_tokens": int(np.count_nonzero(arrays["labels"] != IGNORE_INDEX)),
        "answer_eos_positions": sum(item["answer_eos_positions"] for item in records),
        "answer_eos_positions_with_lm_labels": sum(item["answer_eos_positions_with_lm_labels"] for item in records),
        "answer_value_positions": sum(item["answer_value_positions"] for item in records),
        "formatting_eos_positions": sum(item["formatting_eos_positions"] for item in records),
        "route_mask_true_at_answer_positions": sum(item["route_mask_true_at_answer_positions"] for item in records),
        "route_positions_with_any_target_at_answer": sum(item["route_positions_with_any_target_at_answer"] for item in records),
        "state_mask_true_at_answer_positions": sum(item["state_mask_true_at_answer_positions"] for item in records),
        "state_positions_with_any_target_at_answer": sum(item["state_positions_with_any_target_at_answer"] for item in records),
        "full_lm_supervision_includes_every_answer_position": all(
            item["answer_eos_positions_with_lm_labels"] == 17 for item in records
        ),
        "route_state_targets_exclude_every_answer_position": all(
            item["route_mask_true_at_answer_positions"] == item["state_mask_true_at_answer_positions"] == 0
            for item in records
        ),
    }
    for name in (
        "first_answer_position", "last_instruction_anchor", "downstream_distance",
        "intervening_tokens_between_anchor_and_answer",
    ):
        result[name] = _position_stats([item[name] for item in records])
    return result


def compare_answer_position_coverage(training_dataset, longer_dataset) -> dict:
    """Compare training with longer programs without introducing another evaluation population."""
    training = answer_position_coverage(training_dataset)
    longer = answer_position_coverage(longer_dataset)
    deltas = {}
    for name in (
        "first_answer_position", "last_instruction_anchor", "downstream_distance",
        "intervening_tokens_between_anchor_and_answer",
    ):
        deltas[name] = longer[name]["mean"] - training[name]["mean"]
    return {
        "training": training,
        "longer_program": longer,
        "longer_minus_training_mean": deltas,
        "supervision_boundary": (
            "full_sequence LM labels include all 17 answer+EOS positions per document; "
            "route/state masks have zero targets at those positions"
        ),
        "ordered_exposure_not_masked_token_matching": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.answer_analysis", description=__doc__.split("\n\n")[0])
    result.add_argument("--data", required=True, help="existing five-split program-study data root")
    result.add_argument("--output", required=True, help="new or empty companion mask directory")
    result.add_argument("--tokenizer", help="hash-pinned tokenizer.json; defaults to the source manifest path")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = build_answer_companions(args.data, args.output, tokenizer_path=args.tokenizer)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "format": result["format"],
        "splits": {
            split: {
                "documents": values["documents"],
                "answer_eos_tokens": values["answer_eos_tokens"],
                "answer_value_tokens": values["answer_value_tokens"],
                "formatting_eos_tokens": values["formatting_eos_tokens"],
            }
            for split, values in result["splits"].items()
        },
        "model_or_training_launched": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
