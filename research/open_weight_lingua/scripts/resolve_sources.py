"""Explicit maintainer operation: resolve upstream once, without model weights.

Writes a NEW lock and small metadata/config provenance files. Normal fetch and
inference use the committed lock exclusively and never resolve a moving branch.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def read(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing source lock")
    lock = {
        "schema_version": 1,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "sources": {},
        "license": "Apache-2.0",
    }
    repos = {
        "target": "Qwen/Qwen2.5-7B-Instruct",
        "av": "kitft/nla-qwen2.5-7b-L20-av",
        "ar": "kitft/nla-qwen2.5-7b-L20-ar",
    }
    for role, repo in repos.items():
        info = json.loads(read(f"https://huggingface.co/api/models/{repo}?blobs=true"))
        revision = info["sha"]
        folder = args.metadata_cache / role / revision
        folder.mkdir(parents=True, exist_ok=True)
        entry = {
            "repo_id": repo,
            "revision": revision,
            "license": info["cardData"]["license"],
            "files": {},
        }
        for file in info["siblings"]:
            name = file["rfilename"]
            if name == ".gitattributes":
                continue
            if name.endswith(".safetensors"):
                digest = file["lfs"]["sha256"]
            else:
                data = read(f"https://huggingface.co/{repo}/resolve/{revision}/{name}")
                (folder / name).write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
            entry["files"][name] = {"sha256": digest, "bytes": file["size"]}
        entry["download_bytes"] = sum(f["bytes"] for f in entry["files"].values())
        lock["models"][role] = entry
        print(role, revision, entry["download_bytes"])
    source_files = {
        "kitft/nla-inference": ["nla_inference.py", "README.md", "LICENSE"],
        "kitft/natural_language_autoencoders": [
            "nla/datagen/extractors.py",
            "docs/inference.md",
        ],
    }
    for repo, names in source_files.items():
        revision = json.loads(
            read(f"https://api.github.com/repos/{repo}/commits/main")
        )["sha"]
        files = {}
        for name in names:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{name}"
            files[name] = {"url": url, "sha256": hashlib.sha256(read(url)).hexdigest()}
        lock["sources"][repo] = {"revision": revision, "files": files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
