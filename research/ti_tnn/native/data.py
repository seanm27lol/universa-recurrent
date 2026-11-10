"""Auditable context-256 packing for all native recurrent-model arms.

For example, if a Wikipedia article follows a code file in a packed row, the article's
first label is ``-100``: the code token must not predict it.  ``doc_ids`` differ and
``key_padding_mask`` marks only right padding, so the model has enough information to
block both cross-document and padded attention.  Labels use the usual causal-LM
convention: logits at ``t-1`` are scored against ``labels[t]`` externally.

The builder hashes complete pinned downloads, deduplicates exact UTF-8 document content
before a deterministic one-percent dev split, and writes a document audit.  A finite
45/45/10 Wikipedia/code/structured allocation never cycles a shard.  If a budget ends
inside a document, its consumed and unconsumed token counts are explicit; unconsumed
suffixes are not silently treated as training data.  Every source document appears in
at most one contiguous run.  NPY files support memory mapping and their SHA256 hashes
make byte-identical arm consumption checkable.

Only the four arrays in ``MODEL_INPUT_ARRAYS`` are model inputs.  Labels and all
execution-derived arrays are external targets, never extra forward-pass arguments.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .structured import (N_EXPERTS, N_PASSES, N_SLOTS, N_STATE_CLASSES, StructuredExample, align_example,
                         encode_with_offsets)

CONTEXT_LENGTH = 256
VOCAB_SIZE = 50257
EOS_TOKEN_ID = 50256
IGNORE_INDEX = -100
DEV_FRACTION = 0.01
SPLIT_SALT = "ti-tnn-native:document-dev:v1"
MIXTURE_PARTS = {"wikipedia": 9, "code": 9, "structured": 2}
MIXTURE = {name: parts / 20 for name, parts in MIXTURE_PARTS.items()}
SOURCE_ORDER = tuple(MIXTURE_PARTS)
ALLOWED_CODE_LICENSES = frozenset({"apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "mit"})
LICENSE_DISCLAIMER = (
    "License-label filtering is a provenance filter, not file-level legal clearance; "
    "retained files can still require individual license and notice review."
)

TOKENIZER_SPEC = {
    "repo": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "file": "tokenizer.json",
    "sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "vocab_size": VOCAB_SIZE,
    "eos_token_id": EOS_TOKEN_ID,
    "url": "https://huggingface.co/openai-community/gpt2/resolve/"
           "607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
    "max_bytes": 8 * 1024 * 1024,
}
SOURCE_SPECS = {
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "files": (
            {"file": "20231101.en/train-00000-of-00041.parquet",
             "sha256": "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677"},
            {"file": "20231101.en/train-00001-of-00041.parquet",
             "sha256": "6cd57a5825312ca966313d4a1b614690b5c4fb53cee9b124bc0e999977eb36cf"},
        ),
        "max_bytes_per_file": 2 * 1024 ** 3,
        "format": "parquet",
    },
    "code": {
        "dataset": "codeparrot/codeparrot-clean",
        "revision": "35a59fb025bc0a102f7d96eac09d145b896d487b",
        "files": ({"file": "file-000000000001.json.gz",
                   "sha256": "3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94"},),
        "max_bytes_per_file": 4 * 1024 ** 3,
        "format": "gzip_jsonl",
        "allowed_licenses": sorted(ALLOWED_CODE_LICENSES),
        "license_notice": LICENSE_DISCLAIMER,
    },
}
for _source in SOURCE_SPECS.values():
    for _file in _source["files"]:
        _file["url"] = (f"https://huggingface.co/datasets/{_source['dataset']}/resolve/"
                        f"{_source['revision']}/{_file['file']}")

SOURCE_IDS = {"wikipedia": 0, "code": 1, "structured": 2}
PADDING_SOURCE_ID = -1
CODE_GENERATION_DEFAULT_COUNT = 64
CODE_GENERATION_MAX_COUNT = 1024
CODE_GENERATION_DEFAULT_TARGET_CHARS = 256
CODE_GENERATION_MAX_TARGET_CHARS = 1024
MODEL_INPUT_ARRAYS = ("input_ids", "positions", "doc_ids", "key_padding_mask")
TARGET_ARRAYS = ("labels", "source_ids", "route_targets", "route_mask", "state_targets", "state_mask")
ARRAY_NAMES = ("input_ids", "labels", "source_ids", "positions", "doc_ids", "key_padding_mask",
               "route_targets", "route_mask", "state_targets", "state_mask")
ARRAY_DTYPES = {
    "input_ids": np.dtype("int32"), "labels": np.dtype("int32"), "source_ids": np.dtype("int8"),
    "positions": np.dtype("int32"), "doc_ids": np.dtype("int32"), "key_padding_mask": np.dtype("bool"),
    "route_targets": np.dtype("int8"), "route_mask": np.dtype("bool"),
    "state_targets": np.dtype("int16"), "state_mask": np.dtype("bool"),
}
TORCH_LONG_ARRAYS = frozenset({"input_ids", "labels", "positions", "doc_ids", "route_targets", "state_targets"})


class DataError(RuntimeError):
    pass


class HashMismatch(DataError):
    pass


class DownloadError(DataError):
    pass


@dataclass(frozen=True)
class Document:
    """One source row with a stable file/row identity and exact-text content hash."""

    source: str
    identity: str
    text: str
    content_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("document text must be str")
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_sha256 and self.content_sha256 != digest:
            raise ValueError(f"{self.identity}: incorrect content hash")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", digest)


@dataclass
class TokenizedDocument:
    document: Document
    input_ids: np.ndarray
    route_targets: np.ndarray
    route_mask: np.ndarray
    state_targets: np.ndarray
    state_mask: np.ndarray

    def __post_init__(self) -> None:
        self.input_ids = np.asarray(self.input_ids, dtype=np.int32)
        length = len(self.input_ids)
        if length == 0:
            raise ValueError(f"{self.document.identity}: tokenizer produced no tokens")
        expected = {
            "route_targets": (length, N_PASSES), "route_mask": (length, N_PASSES),
            "state_targets": (length, N_PASSES, N_SLOTS), "state_mask": (length, N_PASSES, N_SLOTS),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=ARRAY_DTYPES[name])
            if value.shape != shape:
                raise ValueError(f"{self.document.identity}: {name} has shape {value.shape}, expected {shape}")
            setattr(self, name, value)


@dataclass
class TokenRun:
    doc_id: int
    tokenized: TokenizedDocument
    length: int

    @property
    def source(self) -> str:
        return self.tokenized.document.source


def sha256_file(path: str | os.PathLike, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | os.PathLike, expected_sha256: str, *, max_bytes: int | None = None) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise HashMismatch(f"{path}: {path.stat().st_size} bytes exceeds {max_bytes}-byte bound")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise HashMismatch(f"{path}: sha256 {actual}, expected {expected_sha256}")
    return path


def download_verified(url: str, destination: str | os.PathLike, expected_sha256: str, *, max_bytes: int,
                      timeout: float = 60.0) -> Path:
    """Download one complete object to a temporary file, enforcing size and SHA256 before replacement."""
    if max_bytes <= 0 or timeout <= 0:
        raise ValueError("download size and timeout bounds must be positive")
    destination = Path(destination)
    if destination.exists():
        return verify_file(destination, expected_sha256, max_bytes=max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ti-tnn-native-data/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise DownloadError(f"{url}: declared size {declared} exceeds {max_bytes}-byte bound")
            with open(temporary, "wb") as sink:
                while True:
                    chunk = response.read(min(1024 * 1024, max_bytes + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadError(f"{url}: exceeded {max_bytes}-byte bound")
                    digest.update(chunk)
                    sink.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise HashMismatch(f"{url}: sha256 {digest.hexdigest()}, expected {expected_sha256}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def source_url(source: str, file: Mapping) -> str:
    return file.get("url") or SOURCE_SPECS[source]["files"][0]["url"]


def download_pinned(cache_directory: str | os.PathLike) -> tuple[dict[str, list[Path]], Path]:
    """Fetch exactly the two complete Wikipedia shards, one code shard and tokenizer pin."""
    cache = Path(cache_directory)
    paths = {}
    for source, spec in SOURCE_SPECS.items():
        paths[source] = []
        for file in spec["files"]:
            destination = cache / "datasets" / source / Path(file["file"]).name
            paths[source].append(download_verified(file["url"], destination, file["sha256"],
                                                   max_bytes=spec["max_bytes_per_file"]))
    tokenizer = download_verified(TOKENIZER_SPEC["url"], cache / "tokenizer" / "tokenizer.json",
                                  TOKENIZER_SPEC["sha256"], max_bytes=TOKENIZER_SPEC["max_bytes"])
    return paths, tokenizer


def load_gpt2_tokenizer(path: str | os.PathLike):
    """Load only the hash-pinned GPT2 tokenizer JSON and check its vocabulary/EOS contract."""
    from tokenizers import Tokenizer

    verify_file(path, TOKENIZER_SPEC["sha256"], max_bytes=TOKENIZER_SPEC["max_bytes"])
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise DataError(f"tokenizer vocabulary is {tokenizer.get_vocab_size()}, expected {VOCAB_SIZE}")
    if tokenizer.token_to_id("<|endoftext|>") != EOS_TOKEN_ID:
        raise DataError("pinned tokenizer has an unexpected EOS id")
    return tokenizer


def read_wikipedia_parquet(path: str | os.PathLike, *, batch_size: int = 1024) -> tuple[list[Document], dict]:
    """Read every Parquet row in stable row-group order; empty rows are counted explicitly."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet input requires the pinned pyarrow in requirements-native.txt") from error
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path = Path(path)
    documents = []
    rows = 0
    empty = 0
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise DataError(f"{path}: no text column")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        for record in batch.to_pylist():
            text = record["text"]
            if not isinstance(text, str):
                raise DataError(f"{path}: row {rows} has non-string text")
            identity = f"{path.name}:row-{rows:09d}"
            if text:
                documents.append(Document("wikipedia", identity, text))
            else:
                empty += 1
            rows += 1
    return documents, {"rows": rows, "retained": len(documents), "empty_excluded": empty,
                       "row_groups": parquet.num_row_groups, "complete_file": True}


def read_codeparrot_jsonl(path: str | os.PathLike) -> tuple[list[Document], dict]:
    """Read all gzip JSONL lines and retain only the five explicit license labels."""
    path = Path(path)
    documents = []
    rejected = Counter()
    retained_licenses = Counter()
    rows = 0
    empty = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                raise DataError(f"{path}: blank JSONL line {line_number} would make row identities ambiguous")
            record = json.loads(line)
            text = record.get("content")
            license_label = record.get("license")
            if not isinstance(text, str) or not isinstance(license_label, str):
                raise DataError(f"{path}: line {line_number} needs string content and license")
            label = license_label.strip().lower()
            if label not in ALLOWED_CODE_LICENSES:
                rejected[label] += 1
            elif text:
                documents.append(Document("code", f"{path.name}:line-{line_number:09d}", text))
                retained_licenses[label] += 1
            else:
                empty += 1
            rows += 1
    return documents, {
        "rows": rows, "retained": len(documents), "empty_excluded": empty,
        "rejected_by_license": dict(sorted(rejected.items())),
        "retained_by_license": dict(sorted(retained_licenses.items())),
        "allowed_licenses": sorted(ALLOWED_CODE_LICENSES), "license_notice": LICENSE_DISCLAIMER,
        "complete_file": True,
    }


def ingest_sources(paths: Mapping[str, Sequence[str | os.PathLike]]) -> tuple[list[Document], dict]:
    """Hash and completely read the pinned source files, returning documents plus an inventory."""
    documents = []
    inventory = {"sources": {}, "forbidden_sources": ["The Stack gated data", "WikiText evaluation populations",
                                                       "Qwen targets", "BIRD"]}
    for source in SOURCE_ORDER[:2]:
        spec = SOURCE_SPECS[source]
        expected_files = spec["files"]
        source_paths = [Path(path) for path in paths.get(source, ())]
        if len(source_paths) != len(expected_files):
            raise DataError(f"{source}: expected {len(expected_files)} complete files, got {len(source_paths)}")
        entries = []
        for file, path in zip(expected_files, source_paths):
            verify_file(path, file["sha256"], max_bytes=spec["max_bytes_per_file"])
            if spec["format"] == "parquet":
                read, file_inventory = read_wikipedia_parquet(path)
            else:
                read, file_inventory = read_codeparrot_jsonl(path)
            documents.extend(read)
            entries.append({"file": file["file"], "local_name": path.name, "sha256": file["sha256"],
                            "bytes": path.stat().st_size, **file_inventory})
        inventory["sources"][source] = {"dataset": spec["dataset"], "revision": spec["revision"],
                                        "files": entries, "documents_retained": sum(e["retained"] for e in entries)}
    return documents, inventory


def deduplicate_documents(documents: Iterable[Document]) -> tuple[list[Document], dict]:
    """Keep first source-order occurrence of each exact-content SHA256, before any split."""
    unique = []
    duplicates = []
    seen = {}
    total = 0
    for document in documents:
        total += 1
        if document.content_sha256 in seen:
            duplicates.append({"dropped": document.identity, "kept": seen[document.content_sha256],
                               "content_sha256": document.content_sha256})
        else:
            seen[document.content_sha256] = document.identity
            unique.append(document)
    audit = {"rule": "first occurrence in source/file/row order; SHA256 of exact UTF-8 text; before dev split",
             "input_documents": total, "unique_documents": len(unique), "duplicate_documents": len(duplicates),
             "duplicates": duplicates}
    return unique, audit


def deterministic_dev_split(documents: Sequence[Document], *, fraction: float = DEV_FRACTION,
                            salt: str = SPLIT_SALT) -> tuple[list[Document], list[Document], dict]:
    """Assign exactly round(fraction*n) unique documents to dev using keyed content hashes."""
    if not 0 <= fraction <= 1:
        raise ValueError("dev fraction must be in [0, 1]")
    hashes = [document.content_sha256 for document in documents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("deduplicate exact document content before splitting")
    ordered = sorted(documents, key=lambda d: (
        hashlib.sha256(f"{salt}:{d.content_sha256}".encode("utf-8")).hexdigest(), d.content_sha256))
    dev_count = math.floor(len(documents) * fraction + 0.5)
    dev_hashes = {document.content_sha256 for document in ordered[:dev_count]}
    train = [document for document in documents if document.content_sha256 not in dev_hashes]
    dev = [document for document in documents if document.content_sha256 in dev_hashes]
    audit = {"rule": "sort by sha256('<salt>:<exact-content-sha256>'), take first floor(n*fraction+0.5)",
             "salt": salt, "fraction": fraction, "documents": len(documents),
             "train_documents": len(train), "dev_documents": len(dev),
             "dev_content_sha256": [document.content_sha256 for document in ordered[:dev_count]]}
    return train, dev, audit


def tokenize_ordinary(document: Document, tokenizer, *, eos_token_id: int = EOS_TOKEN_ID) -> TokenizedDocument:
    ids, _ = encode_with_offsets(tokenizer, document.text)
    ids.append(int(eos_token_id))
    length = len(ids)
    zeros = {
        "route_targets": np.zeros((length, N_PASSES), dtype=np.int8),
        "route_mask": np.zeros((length, N_PASSES), dtype=np.bool_),
        "state_targets": np.zeros((length, N_PASSES, N_SLOTS), dtype=np.int16),
        "state_mask": np.zeros((length, N_PASSES, N_SLOTS), dtype=np.bool_),
    }
    return TokenizedDocument(document, np.asarray(ids, dtype=np.int32), **zeros)


def tokenize_structured(example: StructuredExample, tokenizer, *, eos_token_id: int = EOS_TOKEN_ID) -> TokenizedDocument:
    aligned = align_example(example, tokenizer, eos_token_id=eos_token_id)
    document = Document("structured", example.example_id, example.full_text, example.content_sha256)
    return TokenizedDocument(document, **{name: aligned[name] for name in (
        "input_ids", "route_targets", "route_mask", "state_targets", "state_mask")})


def allocate_token_budget(total_tokens: int | None, available: Mapping[str, int]) -> dict[str, int]:
    """Largest-remainder 9:9:2 allocation, bounded by every finite source length."""
    if set(available) != set(SOURCE_ORDER):
        raise ValueError(f"available must have exactly {SOURCE_ORDER}")
    if any(not isinstance(available[name], int) or available[name] < 0 for name in SOURCE_ORDER):
        raise ValueError("available token counts must be nonnegative integers")
    requested = total_tokens is not None
    if total_tokens is None:
        total_tokens = min(available[name] * 20 // MIXTURE_PARTS[name] for name in SOURCE_ORDER)
    if not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens <= 0:
        raise ValueError("token budget must be a positive integer (and every mixture source must be nonempty)")
    while total_tokens > 0:
        raw = {name: total_tokens * MIXTURE_PARTS[name] for name in SOURCE_ORDER}
        allocation = {name: raw[name] // 20 for name in SOURCE_ORDER}
        remainder = total_tokens - sum(allocation.values())
        by_remainder = sorted(SOURCE_ORDER, key=lambda name: (-(raw[name] % 20), SOURCE_ORDER.index(name)))
        for name in by_remainder[:remainder]:
            allocation[name] += 1
        if all(allocation[name] <= available[name] for name in SOURCE_ORDER):
            return allocation
        if requested:
            raise ValueError(f"requested token budget does not fit available tokens {dict(available)}; "
                             f"attempted allocation {allocation}; no source is cycled or truncated to force it")
        total_tokens = min(total_tokens - 1,
                           *(available[name] * 20 // MIXTURE_PARTS[name] for name in SOURCE_ORDER))
    raise ValueError(f"no positive 45/45/10 allocation fits available tokens {dict(available)}")


def _runs_for_split(tokenized: Mapping[str, Sequence[TokenizedDocument]], allocation: Mapping[str, int]):
    runs = {name: [] for name in SOURCE_ORDER}
    consumed_records = []
    all_records = []
    next_doc_id = 0
    for source in SOURCE_ORDER:
        remaining = allocation[source]
        for tokenized_document in tokenized.get(source, ()):
            total = len(tokenized_document.input_ids)
            take = min(remaining, total)
            base = {"identity": tokenized_document.document.identity, "source": source,
                    "content_sha256": tokenized_document.document.content_sha256,
                    "tokens_total": total, "tokens_consumed": take, "tokens_unconsumed": total - take}
            if take:
                doc_id = next_doc_id
                next_doc_id += 1
                runs[source].append(TokenRun(doc_id, tokenized_document, take))
                consumed_records.append({"doc_id": doc_id, "partial": take < total, **base})
                all_records.append({"doc_id": doc_id, **base})
            else:
                all_records.append({"doc_id": None, **base})
            remaining -= take
            if remaining == 0:
                # Later documents are still audited with zero consumption by this loop.
                continue
        if remaining != 0:
            raise AssertionError(f"{source}: allocation exceeded available tokens")
    return runs, consumed_records, all_records


def interleave_runs(runs: Mapping[str, Sequence[TokenRun]], allocation: Mapping[str, int]) -> list[TokenRun]:
    """Weighted progress order over whole contiguous document runs; every run is emitted once."""
    positions = {name: 0 for name in SOURCE_ORDER}
    emitted = {name: 0 for name in SOURCE_ORDER}
    result = []
    while True:
        eligible = [name for name in SOURCE_ORDER if positions[name] < len(runs.get(name, ()))]
        if not eligible:
            break
        source = min(eligible, key=lambda name: (
            emitted[name] / allocation[name] if allocation[name] else math.inf, SOURCE_ORDER.index(name)))
        run = runs[source][positions[source]]
        positions[source] += 1
        emitted[source] += run.length
        result.append(run)
    if emitted != dict(allocation):
        raise AssertionError(f"interleaving emitted {emitted}, expected {dict(allocation)}")
    return result


def _blank_arrays(token_count: int, context_length: int) -> dict[str, np.ndarray]:
    rows = (token_count + context_length - 1) // context_length
    arrays = {
        "input_ids": np.full((rows, context_length), EOS_TOKEN_ID, dtype=np.int32),
        "labels": np.full((rows, context_length), IGNORE_INDEX, dtype=np.int32),
        "source_ids": np.full((rows, context_length), PADDING_SOURCE_ID, dtype=np.int8),
        "positions": np.zeros((rows, context_length), dtype=np.int32),
        "doc_ids": np.full((rows, context_length), -1, dtype=np.int32),
        "key_padding_mask": np.ones((rows, context_length), dtype=np.bool_),
        "route_targets": np.zeros((rows, context_length, N_PASSES), dtype=np.int8),
        "route_mask": np.zeros((rows, context_length, N_PASSES), dtype=np.bool_),
        "state_targets": np.zeros((rows, context_length, N_PASSES, N_SLOTS), dtype=np.int16),
        "state_mask": np.zeros((rows, context_length, N_PASSES, N_SLOTS), dtype=np.bool_),
    }
    arrays["positions"][:] = np.arange(context_length, dtype=np.int32)
    return arrays


def pack_runs(runs: Sequence[TokenRun], *, context_length: int = CONTEXT_LENGTH,
              packing: str = "legacy", structured_overflow: str = "error",
              position_policy: str = "row") -> tuple[dict[str, np.ndarray], dict]:
    """Pack runs, optionally keeping examples intact and resetting positions at document boundaries."""
    if context_length <= 0:
        raise ValueError("context length must be positive")
    if packing not in ("legacy", "intact"):
        raise ValueError(f"unknown packing version {packing!r}")
    if structured_overflow not in ("error", "exclude"):
        raise ValueError("structured_overflow must be 'error' or 'exclude'")
    if position_policy not in ("row", "document"):
        raise ValueError(f"position_policy must be 'row' or 'document', not {position_policy!r}")
    if packing == "legacy" and (structured_overflow != "error" or position_policy != "row"):
        raise ValueError("legacy packing requires error overflow and row positions")
    included = []
    excluded = []
    for run in runs:
        if packing == "intact" and run.source == "structured" and run.length > context_length:
            if structured_overflow == "error":
                raise DataError(
                    f"{run.tokenized.document.identity}: structured example has {run.length} tokens, "
                    f"longer than context {context_length}; use structured_overflow='exclude' to audit exclusion"
                )
            excluded.append({"identity": run.tokenized.document.identity, "tokens": run.length})
        else:
            included.append(run)
    token_count = sum(run.length for run in included)
    if token_count <= 0:
        raise ValueError("cannot pack an empty token stream")
    storage_tokens = token_count
    if packing == "intact":
        storage_cursor = 0
        for run in included:
            if run.source == "structured":
                column = storage_cursor % context_length
                if column and column + run.length > context_length:
                    storage_cursor += context_length - column
            storage_cursor += run.length
        storage_tokens = storage_cursor
    arrays = _blank_arrays(storage_tokens, context_length)
    cursor = 0
    structured_rows_padded = 0
    for run in included:
        if packing == "intact" and run.source == "structured":
            column = cursor % context_length
            if column and column + run.length > context_length:
                cursor += context_length - column
                structured_rows_padded += 1
        tokenized = run.tokenized
        run_start = cursor
        remaining = run.length
        offset = 0
        while remaining:
            row, column = divmod(cursor, context_length)
            count = min(remaining, context_length - column)
            selection = slice(column, column + count)
            source_slice = slice(offset, offset + count)
            arrays["input_ids"][row, selection] = tokenized.input_ids[source_slice]
            arrays["source_ids"][row, selection] = SOURCE_IDS[run.source]
            arrays["doc_ids"][row, selection] = run.doc_id
            arrays["key_padding_mask"][row, selection] = False
            for name in ("route_targets", "route_mask", "state_targets", "state_mask"):
                arrays[name][row, selection] = getattr(tokenized, name)[source_slice]
            cursor += count
            offset += count
            remaining -= count
        if cursor - run_start != run.length:
            raise AssertionError("run token conservation failed")
    arrays["labels"][:] = arrays["input_ids"]
    arrays["labels"][arrays["key_padding_mask"]] = IGNORE_INDEX
    arrays["labels"][:, 0] = IGNORE_INDEX
    different_document = arrays["doc_ids"][:, 1:] != arrays["doc_ids"][:, :-1]
    arrays["labels"][:, 1:] = np.where(different_document, IGNORE_INDEX, arrays["labels"][:, 1:])
    if position_policy == "document":
        starts = np.concatenate((np.ones((len(arrays["doc_ids"]), 1), dtype=bool), different_document), axis=1)
        start_columns = np.maximum.accumulate(np.where(starts, np.arange(context_length), 0), axis=1)
        arrays["positions"][:] = np.arange(context_length)[None, :] - start_columns
    arrays["positions"][arrays["key_padding_mask"]] = 0
    if not np.array_equal(arrays["source_ids"] == PADDING_SOURCE_ID, arrays["key_padding_mask"]):
        raise AssertionError("source ids and padding mask disagree")
    loss_mask = arrays["labels"] != IGNORE_INDEX
    source_tokens = {}
    source_loss_tokens = {}
    for source, source_id in SOURCE_IDS.items():
        source_mask = arrays["source_ids"] == source_id
        source_tokens[source] = int(np.count_nonzero(source_mask))
        source_loss_tokens[source] = int(np.count_nonzero(source_mask & loss_mask))
    actual_loss = int(np.count_nonzero(loss_mask))
    if actual_loss != sum(source_loss_tokens.values()) or token_count != sum(source_tokens.values()):
        raise AssertionError("source-id accounting differs from labels or input tokens")
    stats = {
        "sequences": len(arrays["input_ids"]), "tokens": token_count,
        "padding_tokens": int(np.count_nonzero(arrays["key_padding_mask"])),
        "loss_tokens": actual_loss,
        "source_tokens": source_tokens,
        "source_loss_tokens": source_loss_tokens,
        "token_fractions": {name: source_tokens[name] / token_count for name in SOURCE_ORDER},
        "packing_version": packing,
        "position_policy": position_policy,
        "structured_overflow": structured_overflow,
        "structured_rows_padded": structured_rows_padded,
        "structured_examples_intact": sum(run.source == "structured" for run in included),
        "long_examples_excluded": len(excluded),
        "long_excluded_tokens": sum(item["tokens"] for item in excluded),
        "long_examples": excluded,
    }
    return arrays, stats


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_array(directory: Path, name: str, array: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if array.dtype != ARRAY_DTYPES[name]:
        raise TypeError(f"{name}: dtype {array.dtype}, expected {ARRAY_DTYPES[name]}")
    path = directory / f"{name}.npy"
    temporary = directory / f"{name}.npy.tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "dtype": str(array.dtype), "shape": list(array.shape)}


def _tokenizer_identity(tokenizer) -> dict:
    identity = {"name": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)}
    if hasattr(tokenizer, "get_vocab_size"):
        identity["vocab_size"] = int(tokenizer.get_vocab_size())
    return identity


def _code_prompt_target(document: Document, max_target_chars: int) -> tuple[str, str] | None:
    """Split before the final nonempty line, preserving the exact prompt prefix and indentation."""
    offset = 0
    selected = None
    for raw_line in document.text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if line.strip():
            selected = offset, line
        offset += len(raw_line)
    if selected is None:
        return None
    line_start, line = selected
    prompt = document.text[:line_start]
    target = line.rstrip()
    if not prompt.strip() or not target.strip() or len(target) > max_target_chars:
        return None
    if not document.text.startswith(prompt + target):
        raise AssertionError(f"{document.identity}: code prompt/target is not an exact document prefix")
    return prompt, target


def export_code_generation(train_documents: Sequence[Document], dev_documents: Sequence[Document],
                           output_path: str | os.PathLike, *, count: int = CODE_GENERATION_DEFAULT_COUNT,
                           max_target_chars: int = CODE_GENERATION_DEFAULT_TARGET_CHARS,
                           split_salt: str = SPLIT_SALT) -> dict:
    """Export bounded final-line generation targets drawn exclusively from held-out dev code documents."""
    if (not isinstance(count, int) or isinstance(count, bool)
            or not 0 <= count <= CODE_GENERATION_MAX_COUNT):
        raise ValueError(f"code generation count must be in [0, {CODE_GENERATION_MAX_COUNT}]")
    if (not isinstance(max_target_chars, int) or isinstance(max_target_chars, bool)
            or not 1 <= max_target_chars <= CODE_GENERATION_MAX_TARGET_CHARS):
        raise ValueError(f"code generation target bound must be in [1, {CODE_GENERATION_MAX_TARGET_CHARS}]")
    train_hashes = {document.content_sha256 for document in train_documents}
    dev_code = sorted((document for document in dev_documents if document.source == "code"), key=lambda document: (
        hashlib.sha256(f"{split_salt}:{document.content_sha256}".encode("utf-8")).hexdigest(),
        document.content_sha256, document.identity))
    records = []
    examined = 0
    excluded = 0
    for document in dev_code:
        if len(records) == count:
            break
        examined += 1
        if document.content_sha256 in train_hashes:
            raise DataError(f"{document.identity}: code generation document overlaps packed train content")
        prompt_target = _code_prompt_target(document, max_target_chars)
        if prompt_target is None:
            excluded += 1
            continue
        prompt, target = prompt_target
        records.append({"example_id": document.identity, "content_sha256": document.content_sha256,
                        "prompt": prompt, "target_text": target})
    path = Path(output_path)
    _atomic_jsonl(path, records)
    return {
        "path": str(path), "sha256": sha256_file(path), "records": len(records),
        "count_limit": count, "max_target_chars": max_target_chars,
        "selection_rule": "held-out dev code only, ordered by the keyed content hash used for the document split",
        "prompt_target_rule": "exact prefix before the final nonempty line; target retains indentation and excludes trailing whitespace/newlines",
        "dev_code_documents": len(dev_code), "examined_documents": examined,
        "excluded_documents": excluded, "train_content_overlap": 0,
        "auxiliary_labels": False,
    }


def build_dataset(documents: Iterable[Document], structured_train: Sequence[StructuredExample],
                  structured_dev: Sequence[StructuredExample], tokenizer, output_directory: str | os.PathLike, *,
                  context_length: int = CONTEXT_LENGTH, train_tokens: int | None = None, dev_tokens: int | None = None,
                  split_salt: str = SPLIT_SALT, source_inventory: Mapping | None = None,
                  tokenizer_spec: Mapping | None = None, eos_token_id: int = EOS_TOKEN_ID,
                  code_generation_count: int = CODE_GENERATION_DEFAULT_COUNT,
                  code_generation_target_chars: int = CODE_GENERATION_DEFAULT_TARGET_CHARS,
                  packing: str = "legacy", structured_overflow: str = "error",
                  position_policy: str = "row") -> dict:
    """Build and write shared train/dev arrays plus enough metadata to audit every document."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    ordinary, deduplication = deduplicate_documents(documents)
    ordinary_train, ordinary_dev, split_audit = deterministic_dev_split(ordinary, salt=split_salt)
    split_audit["fraction"] = DEV_FRACTION
    structured_hashes = {example.content_sha256 for example in (*structured_train, *structured_dev)}
    if structured_hashes & {document.content_sha256 for document in ordinary}:
        raise DataError("structured and ordinary content overlap")
    tokenized = {"train": {}, "dev": {}}
    for split, ordinary_documents, examples in (("train", ordinary_train, structured_train),
                                                 ("dev", ordinary_dev, structured_dev)):
        for source in SOURCE_ORDER[:2]:
            tokenized[split][source] = [tokenize_ordinary(document, tokenizer, eos_token_id=eos_token_id)
                                        for document in ordinary_documents if document.source == source]
        tokenized[split]["structured"] = [tokenize_structured(example, tokenizer, eos_token_id=eos_token_id)
                                          for example in examples]
    if packing == "intact" and structured_overflow == "exclude":
        for split in ("train", "dev"):
            tokenized[split]["structured"] = [
                document for document in tokenized[split]["structured"]
                if len(document.input_ids) <= context_length
            ]
    manifest = {
        "version": "ti-tnn-native-data-v1", "context_length": context_length,
        "packing_version": packing, "structured_overflow": structured_overflow,
        "position_policy": position_policy,
        "vocab_size": VOCAB_SIZE, "eos_token_id": eos_token_id, "ignore_index": IGNORE_INDEX,
        "passes": N_PASSES, "experts": N_EXPERTS, "slots": N_SLOTS, "n_state_classes": N_STATE_CLASSES,
        "tokenizer": dict(tokenizer_spec or _tokenizer_identity(tokenizer)),
        "mixture": MIXTURE, "mixture_rule": "finite largest-remainder 9:9:2 token allocation; no shard cycling",
        "label_rule": "labels[t]=input_ids[t]; -100 at padding, each sequence start, and every document transition",
        "source_id_map": SOURCE_IDS,
        "padding_rule": "right padding; input EOS, doc_id/source_id -1, key_padding_mask true, label -100, position 0",
        "auxiliary_boundary": "labels/source/route/state arrays are external targets and are never model forward inputs",
        "model_input_arrays": list(MODEL_INPUT_ARRAYS), "target_arrays": list(TARGET_ARRAYS),
        "source_inventory": dict(source_inventory or {}),
        "splits": {},
    }
    document_audit = []
    for split, budget in (("train", train_tokens), ("dev", dev_tokens)):
        available = {source: sum(len(document.input_ids) for document in tokenized[split][source])
                     for source in SOURCE_ORDER}
        allocation = allocate_token_budget(budget, available)
        runs_by_source, consumed, all_documents = _runs_for_split(tokenized[split], allocation)
        runs = interleave_runs(runs_by_source, allocation)
        arrays, stats = pack_runs(
            runs, context_length=context_length, packing=packing, structured_overflow=structured_overflow,
            position_policy=position_policy,
        )
        split_directory = output / split
        array_files = {name: _write_array(split_directory, name, arrays[name]) for name in ARRAY_NAMES}
        consumed_path = output / f"{split}_consumed_documents.jsonl"
        _atomic_jsonl(consumed_path, consumed)
        for record in all_documents:
            document_audit.append({"split": split, **record})
        manifest["splits"][split] = {
            **stats, "available_tokens": available, "allocated_tokens": allocation,
            "unconsumed_tokens": {source: available[source] - allocation[source] for source in SOURCE_ORDER},
            "consumed_documents": len(consumed),
            "consumed_document_file": str(consumed_path), "consumed_document_sha256": sha256_file(consumed_path),
            "arrays": array_files,
        }
    manifest["code_generation"] = export_code_generation(
        ordinary_train, ordinary_dev, output / "code_generation.jsonl", count=code_generation_count,
        max_target_chars=code_generation_target_chars, split_salt=split_salt)
    _atomic_json(output / "deduplication.json", deduplication)
    _atomic_json(output / "document_split.json", split_audit)
    _atomic_jsonl(output / "documents.jsonl", document_audit)
    manifest["deduplication"] = {"path": str(output / "deduplication.json"),
                                  "sha256": sha256_file(output / "deduplication.json"),
                                  **{key: value for key, value in deduplication.items() if key != "duplicates"}}
    manifest["document_split"] = {"path": str(output / "document_split.json"),
                                   "sha256": sha256_file(output / "document_split.json"),
                                   **{key: value for key, value in split_audit.items() if key != "dev_content_sha256"}}
    manifest["documents"] = {"path": str(output / "documents.jsonl"),
                              "sha256": sha256_file(output / "documents.jsonl")}
    _atomic_json(output / "manifest.json", manifest)
    return manifest


class PackedDataset:
    """Memory-mapped shared arrays; ``batch`` returns targets for external losses only."""

    def __init__(self, path: str | os.PathLike, split: str | None = None):
        self.directory = Path(path) / split if split is not None else Path(path)
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in ARRAY_NAMES}
        rows = len(self.arrays["input_ids"])
        for name, array in self.arrays.items():
            if len(array) != rows or array.dtype != ARRAY_DTYPES[name]:
                raise DataError(f"{self.directory}/{name}.npy has inconsistent rows or dtype")

    def __len__(self) -> int:
        return len(self.arrays["input_ids"])

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        return {name: array[index] for name, array in self.arrays.items()}

    def model_inputs(self, index) -> dict[str, np.ndarray]:
        return {name: self.arrays[name][index] for name in MODEL_INPUT_ARRAYS}

    def batch(self, indices, *, as_torch: bool = True, device=None, model_inputs_only: bool = False):
        names = MODEL_INPUT_ARRAYS if model_inputs_only else ARRAY_NAMES
        result = {}
        for name in names:
            value = np.ascontiguousarray(self.arrays[name][indices])
            if not as_torch:
                result[name] = value
                continue
            import torch

            tensor = torch.from_numpy(value)
            if name in TORCH_LONG_ARRAYS:
                tensor = tensor.long()
            result[name] = tensor.to(device) if device is not None else tensor
        return result

    def model_batch(self, indices, *, as_torch: bool = True, device=None):
        return self.batch(indices, as_torch=as_torch, device=device, model_inputs_only=True)

    @property
    def actual_loss_tokens(self) -> int:
        return int(np.count_nonzero(self.arrays["labels"] != IGNORE_INDEX))


def load_dataset(output_directory: str | os.PathLike) -> dict[str, PackedDataset]:
    return {split: PackedDataset(output_directory, split) for split in ("train", "dev")}
