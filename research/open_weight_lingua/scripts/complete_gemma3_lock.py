"""One-time completion of the pending Gemma-3 lock after license acceptance.

configs/model-lock-gemma3-12b.json ships with exact revisions, sizes and
HF-API LFS hashes for every file, but google/gemma-3-12b-it is gated behind
manual license acceptance: the sha256 of its ten small non-LFS files cannot
be resolved without an authenticated download, so those entries carry null
hashes and read_lock refuses the lock (fail closed — nothing runs on
unverified sources). Once the maintainer has accepted the Gemma terms at
https://huggingface.co/google/gemma-3-12b-it and exported HF_TOKEN, this
script downloads exactly the pending files at the pinned revision, hashes
them, and writes the completed lock back over the same path. It refuses to
touch any lock that is already complete, changes nothing but null hash
fields, and proves the result by passing the completed lock through
read_lock before replacing the file.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs/model-lock-gemma3-12b.json",
    )
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported source lock")
    pending = {
        role: [name for name, file in entry["files"].items() if file["sha256"] is None]
        for role, entry in lock["models"].items()
    }
    pending = {role: names for role, names in pending.items() if names}
    if not pending:
        raise SystemExit("lock is already complete; refusing to rewrite it")
    if set(pending) != {"target"}:
        raise ValueError("only the gated target entry may carry pending hashes")
    entry = lock["models"]["target"]
    if entry.get("gated") != "manual" or not re.fullmatch(
        r"[0-9a-f]{40}", entry["revision"]
    ):
        raise ValueError("target entry is not the pinned gated Gemma-3 repository")
    from huggingface_hub import hf_hub_download

    filled = {}
    for name in sorted(pending["target"]):
        downloaded = Path(
            hf_hub_download(
                entry["repo_id"], filename=name, revision=entry["revision"]
            )
        ).read_bytes()
        if len(downloaded) != entry["files"][name]["bytes"]:
            raise ValueError(f"downloaded size disagrees with the lock: {name}")
        filled[name] = sha256_bytes(downloaded)
        print(f"{name}: {filled[name]}")
    completed = json.loads(args.lock.read_text())
    for name, digest in filled.items():
        completed["models"]["target"]["files"][name]["sha256"] = digest
    for role, original_entry in lock["models"].items():
        for name, file in original_entry["files"].items():
            new = completed["models"][role]["files"][name]
            if file["sha256"] is not None and new["sha256"] != file["sha256"]:
                raise ValueError(f"completion would change a pinned hash: {role}/{name}")
            if new["bytes"] != file["bytes"]:
                raise ValueError(f"completion would change a pinned size: {role}/{name}")
    # Prove the completed lock passes the real gate before replacing the file.
    from open_weight_lingua.preflight import read_lock

    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.lock.parent, suffix=".json", delete=False
    ) as stream:
        json.dump(completed, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    try:
        read_lock(temp_path)
        os.replace(temp_path, args.lock)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    print(f"completed lock written: {args.lock}")


if __name__ == "__main__":
    main()
