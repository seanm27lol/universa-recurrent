"""Bounded mixed-corpus preparation; no training, reservation or network access.

An instruction's answer needs its complete program, even at a token quota.
Ordinary language may use context-sized chunks; structured examples may not.
This adapter reuses the existing tokenizer, split, packer, sampler and targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import data as D, structured as S

FORMAT = "native-mixed-learning-preparation-v1"
SEED = 20261108
TRAIN_INPUT_TOKENS = 44_000_000
DEV_INPUT_TOKENS = 1_000_000
REQUESTED_LOSS_TOKENS = 40_000_000
PROMPT_TOKENS = 128
CONTINUATION_TOKENS = 32
CAPABILITY_DOCUMENTS = 128
GENERATION_DOCUMENTS = 64
REGISTER_PROTOCOL = "bd108e03f51aefbc5bf8ce1cc0086c874d450b79ecf5668a23b275d1f37f50df"
HISTORICAL_MANIFEST = "93ca481cdb586d0dbd274da731f4e196c15d9cd97da86902077580f43301f90a"


def read_examples(path):
    """Reconstruct current validated targets from unchanged reference programs."""
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        example = S.make_example(row["split"], tuple(S.Instruction.parse(x) for x in row["program"]), row["initial"])
        if example.prompt != row["prompt"] or example.target_text != row["target_text"]:
            raise ValueError("historical program/reference mismatch")
        yield example


def bounded_runs(documents, examples, tokenizer, budgets, *, context_length=256):
    """Tokenize only finite prefixes; retain whole structured examples at quotas.

    Ordinary final documents may contribute a prefix. Every such cut is recorded.
    An over-context structured example is excluded explicitly; a fitting example
    that crosses the quota ends that source with a documented underfill.
    """
    runs = {source: [] for source in D.SOURCE_ORDER}
    audit = {source: {"requested": int(budgets[source]), "consumed": 0, "excluded_long": [],
                      "quota_next_tokens": None, "documents": []} for source in D.SOURCE_ORDER}
    next_id = 0
    for source in D.SOURCE_ORDER:
        candidates = examples if source == "structured" else (d for d in documents if d.source == source)
        remaining = int(budgets[source])
        if remaining <= 0:
            raise ValueError("all three source budgets must be positive")
        for item in candidates:
            tokenized = D.tokenize_structured(item, tokenizer) if source == "structured" else D.tokenize_ordinary(item, tokenizer)
            length = len(tokenized.input_ids)
            if source == "structured" and length > context_length:
                audit[source]["excluded_long"].append({"id": tokenized.document.identity, "tokens": length})
                continue
            if source == "structured" and length > remaining:
                audit[source]["quota_next_tokens"] = length
                break
            take = min(length, remaining)
            runs[source].append(D.TokenRun(next_id, tokenized, take))
            audit[source]["documents"].append({"doc_id": next_id, "identity": tokenized.document.identity,
                "content_sha256": tokenized.document.content_sha256, "tokens_total": length,
                "tokens_consumed": take, "partial": take != length})
            next_id += 1
            remaining -= take
            if remaining == 0:
                break
        if remaining and (source != "structured" or audit[source]["quota_next_tokens"] is None):
            raise ValueError(f"insufficient finite {source} inventory; refusing a reduced comparison")
        audit[source]["consumed"] = budgets[source] - remaining
        audit[source]["underfill"] = remaining
    actual = {source: a["consumed"] for source, a in audit.items()}
    return D.interleave_runs(runs, actual), audit


def write_packed(runs, directory):
    arrays, stats = D.pack_runs(runs, packing="intact", structured_overflow="error", position_policy="document")
    stats["arrays"] = {name: D._write_array(Path(directory), name, arrays[name]) for name in D.ARRAY_NAMES}
    return stats


def capability_records(documents, tokenizer, *, count=CAPABILITY_DOCUMENTS, generation_count=GENERATION_DOCUMENTS):
    """Choose held-out documents by hash, never by model performance.

    CE covers up to 256 tokens from the actual document beginning, one row per
    document. Continuations use exactly 128 prompt tokens and 32 reference tokens
    from that same beginning. No left truncation or hidden target-length cue.
    These are development documents, not a new confirmation benchmark.
    """
    result = {}
    for source in ("wikipedia", "code"):
        chosen, continuations, rows = [], [], []
        for doc in sorted((d for d in documents if d.source == source), key=lambda d: (d.content_sha256, d.identity)):
            tokenized = D.tokenize_ordinary(doc, tokenizer)
            raw = tokenized.input_ids[:-1].tolist()  # exclude the appended document EOS
            if len(chosen) < count:
                rows.append(D.TokenRun(len(rows), tokenized, min(256, len(tokenized.input_ids))))
                chosen.append({"example_id": doc.identity, "content_sha256": doc.content_sha256})
            if len(continuations) < generation_count and len(raw) >= PROMPT_TOKENS + CONTINUATION_TOKENS:
                continuations.append({"example_id": doc.identity, "content_sha256": doc.content_sha256,
                    "source": source, "prompt_ids": raw[:PROMPT_TOKENS],
                    "target_ids": raw[PROMPT_TOKENS:PROMPT_TOKENS + CONTINUATION_TOKENS]})
            if len(chosen) == count and len(continuations) == generation_count:
                break
        if len(chosen) != count or len(continuations) != generation_count:
            raise ValueError(f"insufficient {source} capability documents")
        result[source] = (rows, chosen, continuations)
    return result


def write_capability(populations, directory):
    """One independent context per source document, with variable loss counts."""
    directory = Path(directory)
    report = {}
    for source, (runs, documents, continuations) in populations.items():
        pieces = [D.pack_runs([r], packing="intact", position_policy="document")[0] for r in runs]
        arrays = {name: np.concatenate([a[name] for a in pieces]) for name in D.ARRAY_NAMES}
        files = {name: D._write_array(directory / source / "teacher_forced", name, value) for name, value in arrays.items()}
        D._atomic_jsonl(directory / source / "continuations.jsonl", continuations)
        # The generation input file contains no gold strings, ids, masks or metadata.
        D._atomic_jsonl(directory / source / "prompts.jsonl",
                       ({"example_id": r["example_id"], "input_ids": r["prompt_ids"]} for r in continuations))
        report[source] = {"documents": documents, "teacher_forced_rows": len(runs),
            "loss_tokens": int(np.count_nonzero(arrays["labels"][:, 1:] != -100)), "arrays": files,
            "continuations": len(continuations), "prompt_tokens_each": PROMPT_TOKENS,
            "reference_tokens_each": CONTINUATION_TOKENS,
            "generation_source_ids": [r["example_id"] for r in continuations]}
    return report


def prepare(cache, historical_data, register_preparation, output):
    started = time.perf_counter()
    cache, historical_data, register_preparation, output = map(Path, (cache, historical_data, register_preparation, output))
    output.mkdir(parents=True, exist_ok=False)
    D.verify_file(historical_data / "manifest.json", HISTORICAL_MANIFEST)
    D.verify_file(register_preparation / "protocol.json", REGISTER_PROTOCOL)
    protocol = json.loads((register_preparation / "protocol.json").read_text())
    for relative, digest in protocol["artifact_sha256"].items():
        D.verify_file(register_preparation / relative, digest)
    tokenizer = D.load_gpt2_tokenizer(cache / "tokenizer/tokenizer.json")
    paths = {source: [cache / "datasets" / source / Path(f["file"]).name for f in spec["files"]]
             for source, spec in D.SOURCE_SPECS.items()}
    documents, inventory = D.ingest_sources(paths)  # local pinned files only
    documents, dedup = D.deduplicate_documents(documents)
    train_docs, dev_docs, split = D.deterministic_dev_split(documents)
    manifest = {"format": FORMAT, "tokenizer": D.TOKENIZER_SPEC, "source_inventory": inventory,
        "historical_manifest_sha256": HISTORICAL_MANIFEST, "register_protocol_sha256": REGISTER_PROTOCOL,
        "packing": "intact", "position_policy": "document", "mixture": D.MIXTURE,
        "model_input_arrays": list(D.MODEL_INPUT_ARRAYS), "splits": {},
        "development_seed": SEED, "requested_loss_tokens_per_arm": REQUESTED_LOSS_TOKENS,
        "auxiliary_schedule": "before/after/after/after; existing validated alignment",
        "evaluation_role": "development monitoring, including reused historical diagnostics; not confirmation"}
    # Preserve the old structured programs and ordinary split; no new curriculum.
    old = json.loads((historical_data / "manifest.json").read_text())
    monitor_pairs = set()
    monitor_source = register_preparation / "multi_state"
    for path in sorted((monitor_source / "evaluation").glob("*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            monitor_pairs.add((tuple(row["program"]), tuple(row["initial"])))
    for name, docs, total in (("train", train_docs, TRAIN_INPUT_TOKENS), ("dev", dev_docs, DEV_INPUT_TOKENS)):
        path = historical_data / "structured" / f"{name}.jsonl"
        D.verify_file(path, old["structured_evaluation"]["files"][name]["sha256"])
        examples = list(read_examples(path))
        if any((tuple(map(str, e.program)), tuple(e.initial)) in monitor_pairs for e in examples):
            raise ValueError("fixed diagnostic program/state pair occurs in mixed training/dev inventory")
        budgets = {source: total * parts // 20 for source, parts in D.MIXTURE_PARTS.items()}
        runs, audit = bounded_runs(docs, examples, tokenizer, budgets)
        manifest["splits"][name] = {**write_packed(runs, output / name), "allocation": audit}
        del runs, examples
    manifest["capability"] = write_capability(capability_records(dev_docs, tokenizer), output / "capability")
    for name in ("evaluation", "prompts", "evaluation_arrays", "answer_masks", "dev"):
        shutil.copytree(monitor_source / name, output / "register_monitor" / name)
    shutil.copy2(monitor_source / "replay.json", output / "register_monitor/replay.json")
    D._atomic_json(output / "deduplication.json", dedup)
    D._atomic_json(output / "document_split.json", split)
    # Reuse the actual trainer's sampler to freeze atomic-batch overshoot and order.
    from .train import make_sampler
    dataset = D.PackedDataset(output / "train")
    sampler = make_sampler(dataset, 8, SEED)
    order_digest = hashlib.sha256()
    input_tokens = padded_slots = 0
    source_loss = dict.fromkeys(D.SOURCE_ORDER, 0)
    while sampler.consumed_loss_tokens < REQUESTED_LOSS_TOKENS:
        batch = sampler.next_batch()
        order_digest.update(batch.batch_identity.encode())
        indices = list(batch.indices)
        padding = dataset.arrays["key_padding_mask"][indices]
        input_tokens += int(np.count_nonzero(~padding))
        padded_slots += padding.size
        labels = dataset.arrays["labels"][indices, 1:]
        sources = dataset.arrays["source_ids"][indices, 1:]
        for source, index in D.SOURCE_IDS.items():
            source_loss[source] += int(np.count_nonzero((labels != -100) & (sources == index)))
    manifest["endpoint"] = {"requested_loss_tokens": REQUESTED_LOSS_TOKENS,
        "actual_loss_tokens": sampler.consumed_loss_tokens,
        "updates": sampler.batches_consumed, "cursor": sampler.cursor, "data_rows": len(dataset),
        "input_tokens": input_tokens, "padded_slots": padded_slots, "source_loss_tokens": source_loss,
        "data_identity": sampler.data_identity, "order_identity": sampler.order_identity,
        "order_prefix_sha256": order_digest.hexdigest(),
        "batch_size": 8, "selection": "final endpoint only; no intermediate selection"}
    manifest["preparation_wall_seconds"] = time.perf_counter() - started
    native = Path(__file__).parent
    manifest["preparation_source_sha256"] = {name: D.sha256_file(native / name) for name in
        ("mixed_learning.py", "data.py", "structured.py", "sampler.py", "train.py")}
    manifest["artifacts"] = {str(p.relative_to(output)): D.sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    D._atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("cache", "historical-data", "register-preparation", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = prepare(args.cache, args.historical_data, args.register_preparation, args.output)
    print(json.dumps({"format": FORMAT, "endpoint": report["endpoint"], "seconds": report["preparation_wall_seconds"]}))


if __name__ == "__main__":
    main()
