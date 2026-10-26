"""Identities that bind every run: source, configuration, model, data and snapshot hashes, and the environment.

**Example.** Before the 16-question B4 run starts, its manifest records:
- the git commit, and that the research tree is clean;
- the sha256 of every `scaling/*.py` file;
- the run configuration's content hash;
- the model's pinned revision, with every file checked against `pins/models_v1.json`;
- the inventory's hash and each snapshot's sha256.

A verifier recomputes each of them. A run whose recorded identities do not match what it claims to have used is
rejected.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path, PurePath

from . import contracts as C

RESEARCH_PATHS = ("research/ti_tnn/scaling", "tests", "pyproject.toml")  # the tree a run's producer code lives in
MODELS_PIN = C.PINS / "models_v1.json"


class IdentityError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _canon(obj):
    """A strictly JSON-shaped copy: string keys only; tuples and lists as lists; sets as sorted lists; dataclasses as
    dicts; paths as strings. Anything else raises, so an identity can never depend on a repr or on hash order."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise TypeError(f"non-finite float {obj!r} in an identity")
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r} in an identity")
            out[k] = _canon(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canon(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canon(dataclasses.asdict(obj))
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not allowed in an identity")


def canonical_json(obj) -> str:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_identity(obj) -> str:
    """The sha256 of an object's canonical JSON: equal content, equal identity."""
    return sha256_bytes(canonical_json(obj).encode())


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=C.REPO, capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> dict:
    """The commit and whether the research tree (the scaling package, the tests, pyproject) has uncommitted changes."""
    status = _git("status", "--porcelain", "--", *RESEARCH_PATHS)
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "worktree": str(C.REPO), "research_tree_dirty": bool(status), "dirty_paths": status.splitlines()[:20]}


def require_clean_tree() -> dict:
    state = git_state()
    if state["research_tree_dirty"]:
        raise IdentityError(f"the research tree has uncommitted changes; commit the producer code first: "
                            f"{state['dirty_paths']}")
    return state


def source_hashes() -> dict:
    """sha256 of every Python file of the scaling package (the producer code a run executes)."""
    return {p.name: sha256_file(p) for p in sorted(C.SCALING.glob("*.py"))}


def _nvidia_smi(query: str) -> str | None:
    try:
        q = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30)
        return q.stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    """Interpreter, packages and device facts. It never initialises CUDA: the GPU is read through nvidia-smi."""
    out = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
           "machine": platform.machine(), "host": socket.gethostname(),
           "gpu": _nvidia_smi("name"), "driver": _nvidia_smi("driver_version"),
           "linear_attention_kernels": os.environ.get("TI_TNN_SCALING_LINEAR_KERNELS", "fla")}
    import importlib.metadata as md
    for pkg in ("torch", "transformers", "tokenizers", "safetensors", "sqlglot", "numpy", "huggingface_hub",
                "accelerate", "flash-linear-attention", "causal-conv1d", "kernels", "triton"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def model_snapshot_dir(key: str) -> Path:
    """The local Hugging Face cache snapshot of a pinned model (no network)."""
    m = C.MODELS[key]
    root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    d = root / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
    if not d.is_dir():
        raise IdentityError(f"{m['repo']}@{m['revision']} is not in the local cache ({d})")
    return d


def model_identity(key: str, verify: bool = False) -> dict:
    """The pinned repo and revision, and each file checked against the committed pin (`pins/models_v1.json`, verified
    once against the Hugging Face API and by re-hashing).

    In this cache layout a snapshot file links to `blobs/<name>`; for large (LFS) files `<name>` is the file's sha256
    (the LFS object id). That blob may link again into a shared store under another name, so only the *first* link's
    name is the LFS id. Small files are hashed directly. Any missing, extra or mismatched file raises. With `verify`,
    every weight file is re-hashed too (about a minute for the 27B)."""
    m = C.MODELS[key]
    d = model_snapshot_dir(key)
    pinned = json.loads(MODELS_PIN.read_text())["models"][key]
    if pinned["repo"] != m["repo"] or pinned["revision"] != m["revision"]:
        raise IdentityError(f"{key}: the model pin is for {pinned['repo']}@{pinned['revision']}")
    files = {}
    for p in sorted(d.iterdir()):
        target = p.resolve()
        first = Path(os.readlink(p)).name if p.is_symlink() else ""
        entry = {"bytes": target.stat().st_size}
        if p.suffix == ".safetensors" and len(first) == 64:
            entry.update(sha256=first, source="lfs-object-id")
            if verify and sha256_file(target) != first:
                raise IdentityError(f"{key}: {p.name} does not hash to its LFS object id")
        else:
            entry.update(sha256=sha256_file(target), source="hashed")
        files[p.name] = entry
    expected = pinned["files"]
    missing, extra = sorted(set(expected) - set(files)), sorted(set(files) - set(expected))
    bad = sorted(n for n in set(expected) & set(files)
                 if (files[n]["sha256"], files[n]["bytes"]) != (expected[n]["sha256"], expected[n]["bytes"]))
    if missing or extra or bad:
        raise IdentityError(f"{key}: the local snapshot differs from pins/models_v1.json: missing {missing}, "
                            f"unexpected {extra}, mismatched {bad}")
    return {"key": key, "repo": m["repo"], "revision": m["revision"], "pin": "pins/models_v1.json",
            "pin_sha256": sha256_file(MODELS_PIN), "files": files}


def write_manifest(run_dir: Path, *, config: dict, extra: dict | None = None, require_clean: bool = True) -> dict:
    """Bind a run before it starts: git state, sources, configuration, environment. Written once; never overwritten."""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(f"{path} exists: manifests are immutable")
    state = require_clean_tree() if require_clean else git_state()
    manifest = {"study": C.STUDY, "study_version": C.STUDY_VERSION, "git": state, "sources": source_hashes(),
                "config": _canon(config), "config_identity": content_identity(config), "environment": environment(),
                **_canon(extra or {})}
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest
