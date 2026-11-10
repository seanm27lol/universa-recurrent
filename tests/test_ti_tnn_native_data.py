"""Offline tests for pinned source readers, dedup/splits and shared native arrays."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import data  # noqa: E402
from native.structured import generate_dataset  # noqa: E402


def make_tokenizer(documents, structured):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    texts = [document.text for document in documents]
    texts += [example.full_text for examples in structured.values() for example in examples]
    tokenizer.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    return tokenizer


def ordinary_documents(per_source=200):
    return [
        *[data.Document("wikipedia", f"wiki-{i}", f"wikipedia article{i} body") for i in range(per_source)],
        *[data.Document("code", f"code-{i}", f"code sample{i} body\nprint({i})") for i in range(per_source)],
    ]


def manual_tokenized(source, identity, ids):
    length = len(ids)
    document = data.Document(source, identity, "x")
    return data.TokenizedDocument(
        document, np.asarray(ids, dtype=np.int32),
        np.zeros((length, 4), dtype=np.int8), np.zeros((length, 4), dtype=np.bool_),
        np.zeros((length, 4, 4), dtype=np.int16), np.zeros((length, 4, 4), dtype=np.bool_))


def test_exact_content_dedup_precedes_independently_derived_one_percent_split():
    documents = [data.Document("wikipedia", f"w{i}", f"unique text {i}") for i in range(200)]
    documents.insert(7, data.Document("code", "duplicate", documents[3].text))
    unique, audit = data.deduplicate_documents(documents)
    assert len(unique) == 200 and audit["duplicate_documents"] == 1
    assert audit["duplicates"][0]["kept"] == "w3"
    train, dev, split = data.deterministic_dev_split(unique)
    expected_order = sorted(unique, key=lambda d: hashlib.sha256(
        f"{data.SPLIT_SALT}:{d.content_sha256}".encode()).hexdigest())
    assert len(dev) == 2 and len(train) == 198
    expected_dev = {d.content_sha256 for d in expected_order[:2]}
    assert [d.content_sha256 for d in dev] == [
        d.content_sha256 for d in unique if d.content_sha256 in expected_dev]
    assert split["dev_content_sha256"] == [d.content_sha256 for d in expected_order[:2]]
    assert not {d.content_sha256 for d in train} & {d.content_sha256 for d in dev}
    with pytest.raises(ValueError, match="before splitting"):
        data.deterministic_dev_split(documents)


def test_pack_masks_sequence_and_document_starts_padding_and_conserves_every_token():
    first = manual_tokenized("wikipedia", "first", [11, 12])
    second = manual_tokenized("code", "second", [21, 22, 23])
    arrays, stats = data.pack_runs([data.TokenRun(7, first, 2), data.TokenRun(9, second, 3)], context_length=4)
    assert arrays["input_ids"].tolist() == [[11, 12, 21, 22], [23, 50256, 50256, 50256]]
    assert arrays["source_ids"].tolist() == [[0, 0, 1, 1], [1, -1, -1, -1]]
    assert arrays["source_ids"].dtype == np.int8
    assert arrays["doc_ids"].tolist() == [[7, 7, 9, 9], [9, -1, -1, -1]]
    assert arrays["key_padding_mask"].tolist() == [[False] * 4, [False, True, True, True]]
    assert arrays["labels"].tolist() == [[-100, 12, -100, 22], [-100, -100, -100, -100]]
    assert arrays["positions"].tolist() == [[0, 1, 2, 3], [0, 0, 0, 0]]
    real = ~arrays["key_padding_mask"]
    assert arrays["input_ids"][real].tolist() == [11, 12, 21, 22, 23]
    assert stats["tokens"] == 5 and stats["padding_tokens"] == 3
    assert stats["loss_tokens"] == np.count_nonzero(arrays["labels"] != -100) == 2
    assert stats["source_tokens"] == {"wikipedia": 2, "code": 3, "structured": 0}
    assert stats["source_loss_tokens"] == {"wikipedia": 1, "code": 1, "structured": 0}
    shifted_labels = arrays["labels"][:, 1:]
    shifted_sources = arrays["source_ids"][:, 1:]
    for source, source_id in data.SOURCE_IDS.items():
        attributed = (shifted_sources == source_id) & (shifted_labels != -100)
        assert np.count_nonzero(attributed) == stats["source_loss_tokens"][source]
    assert not arrays["route_mask"].any() and not arrays["state_mask"].any()


def test_mixture_actual_counts_audits_and_memory_mapped_shared_batches(tmp_path):
    documents = ordinary_documents()
    structured = generate_dataset(train=5, dev=3, unseen_composition=2, longer_length=2, seed=23)
    tokenizer = make_tokenizer(documents, structured)
    manifest = data.build_dataset(documents, structured["train"], structured["dev"], tokenizer, tmp_path,
                                  context_length=32, train_tokens=200, code_generation_count=3)
    train = manifest["splits"]["train"]
    assert train["source_tokens"] == {"wikipedia": 90, "code": 90, "structured": 20}
    assert train["token_fractions"] == {"wikipedia": 0.45, "code": 0.45, "structured": 0.10}
    assert "source_ids" in data.TARGET_ARRAYS and "source_ids" in data.ARRAY_NAMES
    assert "source_ids" not in data.MODEL_INPUT_ARRAYS
    assert manifest["source_id_map"] == data.SOURCE_IDS
    datasets = data.load_dataset(tmp_path)
    for split, dataset in datasets.items():
        stats = manifest["splits"][split]
        assert dataset.actual_loss_tokens == stats["loss_tokens"]
        assert stats["loss_tokens"] == sum(stats["source_loss_tokens"].values())
        assert stats["tokens"] == sum(stats["source_tokens"].values())
        assert stats["tokens"] + stats["padding_tokens"] == len(dataset) * 32
        assert set(dataset.model_inputs(0)) == set(data.MODEL_INPUT_ARRAYS)
        source_ids = dataset.arrays["source_ids"]
        assert source_ids.dtype == np.int8 and source_ids.shape == dataset.arrays["input_ids"].shape
        assert np.array_equal(source_ids == -1, dataset.arrays["key_padding_mask"])
        assert stats["arrays"]["source_ids"]["dtype"] == "int8"
        assert stats["arrays"]["source_ids"]["sha256"] == data.sha256_file(tmp_path / split / "source_ids.npy")
        for source, source_id in data.SOURCE_IDS.items():
            source_mask = source_ids == source_id
            assert np.count_nonzero(source_mask) == stats["source_tokens"][source]
            assert np.count_nonzero(source_mask & (dataset.arrays["labels"] != -100)) == (
                stats["source_loss_tokens"][source])
        batch = dataset.batch([0])
        assert batch["input_ids"].dtype == torch.int64
        assert batch["source_ids"].dtype == torch.int8
        assert batch["route_targets"].shape == (1, 32, 4)
        assert batch["state_targets"].shape == (1, 32, 4, 4)
        assert batch["key_padding_mask"].dtype == torch.bool
        assert set(dataset.model_batch([0], as_torch=False)) == set(data.MODEL_INPUT_ARRAYS)
        assert dataset.arrays["input_ids"].flags.writeable is False or dataset.arrays["input_ids"].base is not None
    audit = [json.loads(line) for line in (tmp_path / "documents.jsonl").read_text().splitlines()]
    assert all(record["tokens_total"] == record["tokens_consumed"] + record["tokens_unconsumed"]
               for record in audit)
    for split in ("train", "dev"):
        for source in data.SOURCE_ORDER:
            consumed = sum(record["tokens_consumed"] for record in audit
                           if record["split"] == split and record["source"] == source)
            assert consumed == manifest["splits"][split]["allocated_tokens"][source]
    structured_ids = {record["doc_id"] for record in audit if record["split"] == "train"
                      and record["source"] == "structured" and record["doc_id"] is not None}
    train_arrays = datasets["train"].arrays
    labelled_doc_ids = set(np.unique(train_arrays["doc_ids"][train_arrays["route_mask"].any(axis=2)]))
    assert labelled_doc_ids <= structured_ids
    unique, _ = data.deduplicate_documents(documents)
    ordinary_train, ordinary_dev, _ = data.deterministic_dev_split(unique)
    train_hashes = {document.content_sha256 for document in ordinary_train}
    dev_code = {document.identity: document for document in ordinary_dev if document.source == "code"}
    records = [json.loads(line) for line in (tmp_path / "code_generation.jsonl").read_text().splitlines()]
    assert dev_code and len(records) == manifest["code_generation"]["records"] == min(3, len(dev_code))
    assert manifest["code_generation"]["count_limit"] == 3
    assert manifest["code_generation"]["sha256"] == data.sha256_file(tmp_path / "code_generation.jsonl")
    for record in records:
        assert set(record) == {"example_id", "content_sha256", "prompt", "target_text"}
        assert record["example_id"] in dev_code and record["content_sha256"] not in train_hashes
        document = dev_code[record["example_id"]]
        assert document.text.startswith(record["prompt"] + record["target_text"])
        assert record["prompt"].strip() and record["target_text"].strip()
        assert len(record["target_text"]) <= data.CODE_GENERATION_DEFAULT_TARGET_CHARS


def test_repeated_build_is_byte_identical_and_explicit_overbudget_fails(tmp_path):
    documents = ordinary_documents(200)
    structured = generate_dataset(train=3, dev=2, unseen_composition=1, longer_length=1)
    tokenizer = make_tokenizer(documents, structured)
    first = data.build_dataset(documents, structured["train"], structured["dev"], tokenizer, tmp_path / "a",
                               context_length=16, train_tokens=100)
    data.build_dataset(documents, structured["train"], structured["dev"], tokenizer, tmp_path / "b",
                       context_length=16, train_tokens=100)
    for split in ("train", "dev"):
        for name in data.ARRAY_NAMES:
            assert (tmp_path / "a" / split / f"{name}.npy").read_bytes() == (
                tmp_path / "b" / split / f"{name}.npy").read_bytes()
        assert first["splits"][split]["loss_tokens"] > 0
    with pytest.raises(ValueError, match="requested token budget"):
        data.build_dataset(documents, structured["train"], structured["dev"], tokenizer, tmp_path / "c",
                           context_length=16, train_tokens=10**9)


def test_finite_allocation_never_cycles_and_uses_largest_remainder():
    available = {"wikipedia": 100, "code": 80, "structured": 100}
    assert data.allocate_token_budget(20, available) == {"wikipedia": 9, "code": 9, "structured": 2}
    automatic = data.allocate_token_budget(None, available)
    assert sum(automatic.values()) == 177
    assert automatic == {"wikipedia": 80, "code": 79, "structured": 18}
    assert all(automatic[source] <= available[source] for source in data.SOURCE_ORDER)
    with pytest.raises(ValueError, match="requested token budget"):
        data.allocate_token_budget(1000, available)


def test_code_generation_export_is_deterministic_bounded_and_train_disjoint(tmp_path):
    train = [data.Document("code", "train-code", "train = True\nprint(train)\n")]
    dev = [
        data.Document("code", "dev-zero", "x = 0\nprint(x)\n"),
        data.Document("code", "dev-one", "def one():\n    return 1\n"),
        data.Document("code", "dev-long", "x = 2\n" + "y" * 40 + "\n"),
        data.Document("code", "dev-single", "print(3)"),
        data.Document("wikipedia", "dev-wiki", "title\nbody\n"),
    ]
    first = data.export_code_generation(train, dev, tmp_path / "first.jsonl", count=2, max_target_chars=20)
    second = data.export_code_generation(train, dev, tmp_path / "second.jsonl", count=2, max_target_chars=20)
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    records = [json.loads(line) for line in (tmp_path / "first.jsonl").read_text().splitlines()]
    eligible = sorted(dev[:2], key=lambda document: hashlib.sha256(
        f"{data.SPLIT_SALT}:{document.content_sha256}".encode()).hexdigest())
    assert [record["example_id"] for record in records] == [document.identity for document in eligible]
    assert first["records"] == 2
    assert first["excluded_documents"] == first["examined_documents"] - 2
    all_records = data.export_code_generation(train, dev, tmp_path / "all.jsonl", count=4, max_target_chars=20)
    assert all_records["records"] == 2 and all_records["examined_documents"] == 4
    assert all_records["excluded_documents"] == 2
    assert first["train_content_overlap"] == 0 and first["auxiliary_labels"] is False
    assert second["sha256"] == first["sha256"]
    for record, document in zip(records, eligible):
        assert set(record) == {"example_id", "content_sha256", "prompt", "target_text"}
        assert record["content_sha256"] == document.content_sha256
        assert document.text.startswith(record["prompt"] + record["target_text"])
        assert 0 < len(record["target_text"]) <= 20
    capped = data.export_code_generation(train, dev, tmp_path / "capped.jsonl", count=1, max_target_chars=20)
    assert capped["records"] == 1
    with pytest.raises(data.DataError, match="overlaps packed train"):
        data.export_code_generation(train, [train[0]], tmp_path / "overlap.jsonl")
    with pytest.raises(ValueError, match="count"):
        data.export_code_generation(train, dev, tmp_path / "bad.jsonl",
                                    count=data.CODE_GENERATION_MAX_COUNT + 1)
    with pytest.raises(ValueError, match="target bound"):
        data.export_code_generation(train, dev, tmp_path / "bad.jsonl", max_target_chars=0)


def test_bounded_hash_verified_download_uses_complete_temporary_files(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"exact bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "nested" / "download.bin"
    assert data.download_verified(source.as_uri(), destination, digest, max_bytes=100) == destination
    assert destination.read_bytes() == b"exact bytes"
    assert not (tmp_path / "nested" / "download.bin.part").exists()
    with pytest.raises(data.HashMismatch):
        data.download_verified(source.as_uri(), tmp_path / "bad.bin", "0" * 64, max_bytes=100)
    assert not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.part").exists()
    with pytest.raises(data.DownloadError):
        data.download_verified(source.as_uri(), tmp_path / "small.bin", digest, max_bytes=3)
    assert not (tmp_path / "small.bin").exists()


def test_gzip_jsonl_license_filter_and_audit(tmp_path):
    path = tmp_path / "code.json.gz"
    records = [
        {"content": "apache content", "license": "apache-2.0"},
        {"content": "mit content", "license": "MIT"},
        {"content": "copyleft content", "license": "gpl-3.0"},
        {"content": "", "license": "isc"},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    documents, inventory = data.read_codeparrot_jsonl(path)
    assert [document.text for document in documents] == ["apache content", "mit content"]
    assert inventory["rows"] == 4 and inventory["retained"] == 2 and inventory["empty_excluded"] == 1
    assert inventory["rejected_by_license"] == {"gpl-3.0": 1}
    assert inventory["retained_by_license"] == {"apache-2.0": 1, "mit": 1}
    assert "not file-level legal clearance" in inventory["license_notice"]


def test_parquet_reader_preserves_row_identities_and_counts_empty_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "wiki.parquet"
    pq.write_table(pa.table({"text": ["alpha", "", "beta"]}), path, row_group_size=1)
    documents, inventory = data.read_wikipedia_parquet(path, batch_size=1)
    assert [document.text for document in documents] == ["alpha", "beta"]
    assert [document.identity for document in documents] == [f"{path.name}:row-000000000",
                                                               f"{path.name}:row-000000002"]
    assert inventory["rows"] == 3 and inventory["retained"] == 2 and inventory["empty_excluded"] == 1
    assert inventory["complete_file"] is True


def test_source_and_tokenizer_pins_are_complete_and_explicit():
    assert data.CONTEXT_LENGTH == 256 and data.VOCAB_SIZE == 50257 and data.EOS_TOKEN_ID == 50256
    assert data.TOKENIZER_SPEC["revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert len(data.SOURCE_SPECS["wikipedia"]["files"]) == 2
    assert data.SOURCE_SPECS["code"]["files"][0]["sha256"] == (
        "3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94")
    assert set(data.ALLOWED_CODE_LICENSES) == {"apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "mit"}
    assert "not file-level legal clearance" in data.LICENSE_DISCLAIMER


def test_prepare_data_cli_records_document_positions_and_preserves_legacy(tmp_path, monkeypatch):
    from native import prepare_data

    documents = ordinary_documents()
    structured = generate_dataset(train=5, dev=3, unseen_composition=2, longer_length=2, seed=23)
    tokenizer = make_tokenizer(documents, structured)
    monkeypatch.setattr(data, "download_pinned", lambda cache: ({}, tmp_path / "tokenizer.json"))
    monkeypatch.setattr(data, "ingest_sources", lambda paths: (documents, {"sources": {}}))
    monkeypatch.setattr(data, "load_gpt2_tokenizer", lambda path: tokenizer)
    common = [
        "--structured-train", "5", "--structured-dev", "3", "--structured-unseen", "2",
        "--structured-longer", "2", "--seed", "23", "--code-generation-count", "0",
        "--packing", "intact", "--structured-overflow", "exclude",
    ]
    assert prepare_data.main([
        "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "document"),
        "--position-policy", "document", *common,
    ]) == 0
    assert prepare_data.main([
        "--cache", str(tmp_path / "cache"), "--output", str(tmp_path / "legacy"), *common,
    ]) == 0
    assert prepare_data.parser().parse_args([]).position_policy == "row"
    document_manifest = json.loads((tmp_path / "document" / "manifest.json").read_text())
    legacy_manifest = json.loads((tmp_path / "legacy" / "manifest.json").read_text())
    assert document_manifest["position_policy"] == "document"
    assert legacy_manifest["position_policy"] == "row"
    saw_internal_document_start = False
    for split in ("train", "dev"):
        assert document_manifest["splits"][split]["position_policy"] == "document"
        assert legacy_manifest["splits"][split]["position_policy"] == "row"
        document_positions = np.load(tmp_path / "document" / split / "positions.npy")
        legacy_positions = np.load(tmp_path / "legacy" / split / "positions.npy")
        doc_ids = np.load(tmp_path / "document" / split / "doc_ids.npy")
        padding = np.load(tmp_path / "document" / split / "key_padding_mask.npy")
        starts = np.concatenate((np.ones((len(doc_ids), 1), dtype=bool), doc_ids[:, 1:] != doc_ids[:, :-1]), axis=1)
        starts &= ~padding
        assert np.all(document_positions[starts] == 0)
        saw_internal_document_start |= any(np.count_nonzero(starts[row]) > 1 for row in range(len(starts)))
        for row in range(len(doc_ids)):
            live = ~padding[row]
            assert legacy_positions[row, live].tolist() == list(range(int(live.sum())))
        for path in sorted((tmp_path / "document" / split).glob("*.npy")):
            if path.name != "positions.npy":
                assert np.array_equal(np.load(path), np.load(tmp_path / "legacy" / split / path.name)), path.name
    assert saw_internal_document_start
