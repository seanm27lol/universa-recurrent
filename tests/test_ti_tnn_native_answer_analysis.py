"""Self-contained alignment and immutability checks for answer-mask companions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import answer_analysis as A  # noqa: E402
from native import data as D  # noqa: E402
from native.structured import Instruction, execute, format_prompt  # noqa: E402


class RegexTokenizer:
    pattern = re.compile(r" r|\d+|.", re.DOTALL)

    def encode(self, text, add_special_tokens=False):
        tokens = list(self.pattern.finditer(text))
        return SimpleNamespace(
            ids=[zlib.crc32(match.group().encode("utf-8")) % 40000 + 1 for match in tokens],
            offsets=[match.span() for match in tokens],
        )


def make_record(split):
    initial = (1, 2, 3, 4)
    instructions = (Instruction("set", 0, value=5), Instruction("add", 1, src=0))
    prompt = format_prompt(instructions, initial)
    target = execute(instructions, initial).target_text
    identity = hashlib.sha256((prompt + target).encode("utf-8")).hexdigest()
    return {
        "example_id": f"structured-{split}-{identity[:20]}",
        "split": split,
        "prompt": prompt,
        "target_text": target,
        "program": [str(instruction) for instruction in instructions],
        "initial": list(initial),
        "modulus": 32,
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def make_data_root(tmp_path):
    root = tmp_path / "data"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(b"self-contained test tokenizer")
    tokenizer = RegexTokenizer()
    manifest = {
        "format": "native-varied-program-study-data-v1",
        "position_policy": "document",
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": D.sha256_file(tokenizer_path),
        },
        "splits": {},
    }
    for split in A.PROGRAM_SPLITS:
        record = make_record(split)
        records_path = root / "structured" / f"{split}.jsonl"
        write_jsonl(records_path, [record])
        full_text = record["prompt"] + record["target_text"]
        plain_ids, offsets = D.encode_with_offsets(tokenizer, full_text) if hasattr(D, "encode_with_offsets") else (None, None)
        if plain_ids is None:
            from native.structured import encode_with_offsets
            plain_ids, offsets = encode_with_offsets(tokenizer, full_text)
        ids = np.asarray([*plain_ids, D.EOS_TOKEN_ID], dtype=np.int32)
        length = len(ids)
        route_targets = np.zeros((length, 4), dtype=np.int8)
        route_mask = np.zeros((length, 4), dtype=np.bool_)
        state_targets = np.zeros((length, 4, 4), dtype=np.int16)
        state_mask = np.zeros((length, 4, 4), dtype=np.bool_)
        line = record["program"][-1]
        start = record["prompt"].index(line)
        end = start + len(line)
        anchor = [index for index, (left, right) in enumerate(offsets) if left < end and right > start][-1]
        route_mask[anchor] = True
        state_mask[anchor] = True
        document = D.TokenizedDocument(
            D.Document("structured", record["example_id"], full_text),
            ids,
            route_targets,
            route_mask,
            state_targets,
            state_mask,
        )
        arrays, _ = D.pack_runs(
            [D.TokenRun(0, document, length)],
            context_length=256,
            packing="intact",
            position_policy="document",
        )
        array_entries = {
            name: D._write_array(root / split, name, arrays[name]) for name in D.ARRAY_NAMES
        }
        answer_start = len(record["prompt"])
        answer_indices = [
            index for index, (left, right) in enumerate(offsets)
            if right > left and left < len(full_text) and right > answer_start
        ]
        assert len(answer_indices) == 16
        consumed = [{
            "doc_id": 0,
            "example_id": record["example_id"],
            "content_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
            "program_length": 2,
            "final_operation": "add",
            "initial": record["initial"],
            "tokens_total": length,
            "tokens_consumed": length,
            "tokens_unconsumed": 0,
            "partial": False,
            "row": 0,
            "start_column": 0,
            "end_column_exclusive": length,
            "answer_token_columns": answer_indices,
            "eos_token_column": length - 1,
            "answer_and_eos_supervised": True,
        }]
        consumed_path = root / f"{split}_consumed_documents.jsonl"
        write_jsonl(consumed_path, consumed)
        manifest["splits"][split] = {
            "arrays": array_entries,
            "consumed_document_sha256": D.sha256_file(consumed_path),
            "structured_jsonl_sha256": D.sha256_file(records_path),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root, tokenizer_path


def all_hashes(root):
    return {str(path): D.sha256_file(path) for path in root.rglob("*") if path.is_file()}


def test_companion_alignment_prompt_visibility_and_immutable_sources(tmp_path):
    root, tokenizer_path = make_data_root(tmp_path)
    before = all_hashes(root)
    before[D.sha256_file(tokenizer_path)] = D.sha256_file(tokenizer_path)
    output = tmp_path / "new-masks"
    manifest = A.build_answer_companions(root, output, tokenizer=RegexTokenizer(), tokenizer_path=tokenizer_path)

    assert manifest["format"] == A.FORMAT
    assert manifest["first_answer_token_included"]
    assert set(manifest["model_input_arrays_unchanged"]) == set(D.MODEL_INPUT_ARRAYS)
    assert not set(A.MASK_ARRAY_NAMES) & set(D.MODEL_INPUT_ARRAYS)
    for split in A.PROGRAM_SPLITS:
        dataset = A.AnswerMaskedDataset(root, split, output)
        plain = D.PackedDataset(root, split)
        masks = dataset.masks
        for name in A.MASK_ARRAY_NAMES:
            assert masks[name].dtype == np.bool_
            assert masks[name].shape == plain.arrays["input_ids"].shape
        assert int(masks[A.ANSWER_MASK].sum()) == 17
        assert int(masks[A.ANSWER_VALUE_MASK].sum()) == 4
        assert int(masks[A.FORMATTING_EOS_MASK].sum()) == 13
        assert np.array_equal(masks[A.ANSWER_VALUE_MASK] | masks[A.FORMATTING_EOS_MASK], masks[A.ANSWER_MASK])
        assert not np.any(masks[A.ANSWER_VALUE_MASK] & masks[A.FORMATTING_EOS_MASK])
        position = manifest["splits"][split]["document_positions"][0]
        first_answer = position["first_answer_column"]
        assert masks[A.ANSWER_MASK][0, first_answer]
        assert not masks[A.ANSWER_MASK][0, first_answer - 1]
        assert position["answer_value_columns"] == [first_answer + 3, first_answer + 7, first_answer + 11, first_answer + 15]
        assert position["formatting_eos_columns"][-1] == position["end_column_exclusive"] - 1
        assert np.all(plain.arrays["labels"][masks[A.ANSWER_MASK]] != D.IGNORE_INDEX)
        batch = dataset.batch([0], as_torch=False)
        original = plain.batch([0], as_torch=False)
        for name in D.ARRAY_NAMES:
            assert np.array_equal(batch[name], original[name])
        model_batch = dataset.batch([0], as_torch=False, model_inputs_only=True)
        assert set(model_batch) == set(D.MODEL_INPUT_ARRAYS)
        for name in D.MODEL_INPUT_ARRAYS:
            assert np.array_equal(model_batch[name], original[name])
        loss_labels = A.masks_to_labels(batch["labels"], batch[A.ANSWER_MASK])
        assert np.count_nonzero(loss_labels != D.IGNORE_INDEX) == 17
        assert loss_labels[0, first_answer] == batch["labels"][0, first_answer]
        assert np.all(loss_labels[0, :first_answer] == D.IGNORE_INDEX)
        assert np.array_equal(batch["labels"], original["labels"])
    assert all_hashes(root) == {key: value for key, value in before.items() if Path(key).is_file()}
    assert D.sha256_file(tokenizer_path) == before[D.sha256_file(tokenizer_path)]


def test_companion_refuses_nonempty_output(tmp_path):
    root, tokenizer_path = make_data_root(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty output"):
        A.build_answer_companions(root, output, tokenizer=RegexTokenizer(), tokenizer_path=tokenizer_path)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not replace"
    assert not (output / "manifest.json").exists()
