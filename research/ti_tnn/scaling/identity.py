"""Identities that bind every run: source, configuration, model, data and snapshot hashes, and the environment.

**Example.** Before the 16-question B4 run starts, its manifest records:
- the git commit, and that the research tree is clean;
- the sha256 of every `scaling/*.py` file;
- the run configuration's content hash;
- the model's pinned revision, with the blob hashes of its weight files;
- the inventory's hash and each snapshot's sha256.

A verifier later recomputes each of them. A run whose recorded identities do not match what it claims to have used is
rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from . import contracts as C

RESEARCH_PATHS = ("research/ti_tnn/scaling", "tests", "pyproject.toml")  # the tree a run's producer code lives in


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_identity(obj) -> str:
    """The sha256 of an object's canonical JSON: equal content, equal identity."""
    return sha256_bytes(canonical_json(obj).encode())


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=C.REPO, capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> dict:
    """The commit and whether the research tree (the scaling package, the tests, pyproject) has uncommitted changes."""
    status = _git("status", "--porcelain", "--", *RESEARCH_PATHS)
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "research_tree_dirty": bool(status), "dirty_paths": status.splitlines()[:20]}


def require_clean_tree() -> dict:
    state = git_state()
    if state["research_tree_dirty"]:
        raise RuntimeError(f"the research tree has uncommitted changes; commit the producer code first: "
                           f"{state['dirty_paths']}")
    return state


def source_hashes() -> dict:
    """sha256 of every Python file of the scaling package (the producer code a run executes)."""
    return {p.name: sha256_file(p) for p in sorted(C.SCALING.glob("*.py"))}


def environment() -> dict:
    out = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
           "machine": platform.machine(), "host": socket.gethostname()}
    import importlib.metadata as md
    for pkg in ("torch", "transformers", "tokenizers", "safetensors", "sqlglot", "numpy", "huggingface_hub", "accelerate",
                "flash-linear-attention", "triton"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch
        out["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:  # torch absent (CI core) or no device
        out["torch_error"] = repr(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30)
        out["driver"] = q.stdout.strip() or None
    except Exception:
        out["driver"] = None
    return out


def model_snapshot_dir(key: str) -> Path:
    """The local Hugging Face cache snapshot of a pinned model (no network)."""
    m = C.MODELS[key]
    root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    d = root / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
    if not d.is_dir():
        raise FileNotFoundError(f"{m['repo']}@{m['revision']} is not in the local cache ({d})")
    return d


def model_identity(key: str, verify: bool = False) -> dict:
    """The pinned repo and revision, and each file's content identity from the cache.

    In this cache layout a snapshot file links to `blobs/<name>`, and for large (LFS) files `<name>` is the file's
    sha256 (the LFS object id). That blob may link again into a shared store under another name, so only the
    *first* link's name is the LFS id. Small files are hashed directly. With `verify`, every large file is re-hashed
    as well (about a minute for the 27B)."""
    m = C.MODELS[key]
    d = model_snapshot_dir(key)
    files = {}
    for p in sorted(d.iterdir()):
        target = p.resolve()
        first = Path(os.readlink(p)).name if p.is_symlink() else ""
        entry = {"bytes": target.stat().st_size}
        if p.suffix == ".safetensors" and len(first) == 64:
            entry.update(sha256=first, source="lfs-object-id")
            if verify:
                entry["verified"] = sha256_file(target) == first
        else:
            entry.update(sha256=sha256_file(target), source="hashed")
        files[p.name] = entry
    return {"key": key, "repo": m["repo"], "revision": m["revision"], "files": files}


def write_manifest(run_dir: Path, *, config: dict, extra: dict | None = None, require_clean: bool = True) -> dict:
    """Bind a run before it starts: git state, sources, configuration, environment. Written once; never overwritten."""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} exists: manifests are immutable")
    state = require_clean_tree() if require_clean else git_state()
    manifest = {"study": C.STUDY, "study_version": C.STUDY_VERSION, "git": state, "sources": source_hashes(),
                "config": config, "config_identity": content_identity(config), "environment": environment(),
                **(extra or {})}
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    return manifest
