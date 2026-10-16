"""Prepare WikiText-103 (raw) for the TI-TNN language-modelling benchmark.

Downloads the four parquet files of `Salesforce/wikitext` (config wikitext-103-raw-v1)
from the Hugging Face Hub, joins each split's lines in order, tokenizes with GPT-2's
byte-pair encoding (tiktoken "gpt2", 50,257 tokens) and writes uint16 token files plus
meta.json with token counts and sha256 of every input and output.

It needs pyarrow and tiktoken, which the GPU environment does not have, so run it in a
throwaway environment:

    uv run --no-project --with pyarrow --with tiktoken --with huggingface_hub --with numpy \\
        python research/ti_tnn/bench/prep_wikitext.py --out runs/titnn_data/wikitext103
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "Salesforce/wikitext"
FILES = {
    "train": ["wikitext-103-raw-v1/train-00000-of-00002.parquet", "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
    "validation": ["wikitext-103-raw-v1/validation-00000-of-00001.parquet"],
    "test": ["wikitext-103-raw-v1/test-00000-of-00001.parquet"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    meta = {"source": REPO, "revision": args.revision, "tokenizer": "tiktoken gpt2", "vocab_size": enc.n_vocab,
            "splits": {}}
    for split, files in FILES.items():
        text_parts, inputs = [], {}
        for name in files:
            local = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=args.revision))
            inputs[name] = sha256(local)
            text_parts.extend(pq.read_table(local, columns=["text"]).column("text").to_pylist())
        ids = np.array(enc.encode_ordinary("".join(text_parts)), dtype=np.uint16)
        out = args.out / f"{split}.bin"
        ids.tofile(out)
        meta["splits"][split] = {"tokens": int(ids.size), "lines": len(text_parts), "inputs_sha256": inputs,
                                 "output": out.name, "output_sha256": sha256(out)}
        print(f"{split}: {ids.size:,} tokens from {len(text_parts):,} lines")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
