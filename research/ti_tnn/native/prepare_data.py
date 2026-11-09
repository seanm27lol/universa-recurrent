"""Build the pinned native train/dev data without using evaluation or gated corpora.

Full construction (from the repository root)::

    PYTHONPATH=research/ti_tnn python -m native.prepare_data \
      --cache runs/ti_tnn_native/cache --output runs/ti_tnn_native/data

Both complete Wikipedia shards and the complete CodeParrot shard are downloaded and
hash-checked.  ``--train-tokens`` and ``--dev-tokens`` optionally set finite budgets;
otherwise each split uses the largest 45/45/10 allocation supported by its available
streams.  Unconsumed suffixes are reported, never cycled.  ``--structured-only`` is an
explicit synthetic/baseline check and does not claim to build the language dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data
from .structured import baseline_report, generate_dataset, save_structured_jsonl


def build(args: argparse.Namespace) -> dict:
    structured = generate_dataset(train=args.structured_train, dev=args.structured_dev,
                                  unseen_composition=args.structured_unseen,
                                  longer_length=args.structured_longer, seed=args.seed)
    output = Path(args.output)
    structured_files = save_structured_jsonl(structured, output / "structured")
    report = baseline_report(structured)
    data._atomic_json(output / "structured" / "baselines.json", report)
    if args.structured_only:
        return {"structured_only": True, "structured_files": structured_files, "baselines": report}

    paths, tokenizer_path = data.download_pinned(args.cache)
    documents, inventory = data.ingest_sources(paths)
    inventory["sources"]["structured"] = {
        "generator": "native.structured.generate_dataset", "seed": args.seed,
        "counts": {split: len(examples) for split, examples in structured.items()},
        "auxiliary_policy": "execution traces supervise training only; never inference inputs",
    }
    inventory["tokenizer_file"] = {"path": str(tokenizer_path), "sha256": data.TOKENIZER_SPEC["sha256"]}
    tokenizer = data.load_gpt2_tokenizer(tokenizer_path)
    manifest = data.build_dataset(
        documents, structured["train"], structured["dev"], tokenizer, output,
        train_tokens=args.train_tokens, dev_tokens=args.dev_tokens,
        source_inventory=inventory, tokenizer_spec=data.TOKENIZER_SPEC,
        code_generation_count=args.code_generation_count,
        code_generation_target_chars=args.code_generation_target_chars,
    )
    manifest["structured_evaluation"] = {
        "files": structured_files,
        "baselines": report,
        "holdouts_not_packed_into_train_dev": ["unseen_composition", "longer_length"],
        "target_text_policy": "free generation predicts only target_text after prompt",
    }
    data._atomic_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.prepare_data", description=__doc__.split("\n\n")[0])
    result.add_argument("--cache", default="runs/ti_tnn_native/cache", help="pinned download cache")
    result.add_argument("--output", default="runs/ti_tnn_native/data", help="array and audit output directory")
    result.add_argument("--train-tokens", type=int, default=None, help="finite train tokens before right padding")
    result.add_argument("--dev-tokens", type=int, default=None, help="finite dev tokens before right padding")
    result.add_argument("--code-generation-count", type=int, default=data.CODE_GENERATION_DEFAULT_COUNT,
                        help=f"held-out dev code prompts, bounded to [0, {data.CODE_GENERATION_MAX_COUNT}]")
    result.add_argument("--code-generation-target-chars", type=int,
                        default=data.CODE_GENERATION_DEFAULT_TARGET_CHARS,
                        help=f"final-line target bound, up to {data.CODE_GENERATION_MAX_TARGET_CHARS} characters")
    result.add_argument("--structured-train", type=int, default=4096)
    result.add_argument("--structured-dev", type=int, default=512)
    result.add_argument("--structured-unseen", type=int, default=512)
    result.add_argument("--structured-longer", type=int, default=512)
    result.add_argument("--seed", type=int, default=20261108)
    result.add_argument("--structured-only", action="store_true", help="write synthetic scoring data, not LM arrays")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    manifest = build(args)
    if manifest.get("structured_only"):
        print(json.dumps({"structured_only": True,
                          "examples": {split: entry["examples"] for split, entry in
                                       manifest["structured_files"].items()}}, sort_keys=True))
    else:
        print(json.dumps({split: {"tokens": values["tokens"], "loss_tokens": values["loss_tokens"],
                                  "source_tokens": values["source_tokens"]}
                          for split, values in manifest["splits"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
