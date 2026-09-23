"""Parent-side bridge to the vLLM worker subprocess (opt-in AV/AR backend).

The pipeline keeps model lock/hash verification, tokenization conventions and
the run evidence schema; the worker (vllm_worker.py, a different interpreter
with a conflicting transformers pin) only executes forwards. Results re-enter
the same row/evidence schema as the eager path. vLLM output is a DIFFERENT
measurement backend's evidence: no vllm/eager equivalence is claimed beyond the
measured gate in vllm_equivalence.py, and a gate FAIL forbids mixing these
outputs into eager-regime evidence.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from .artifacts import load_numeric, save_numeric, sha256_file, write_json
from .nla_adapter import ar_prompt, av_prompt

VLLM_PIN = "0.30.0"
PROJECT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = Path(__file__).resolve().with_name("vllm_worker.py")
DEFAULT_WORKER_PYTHON = PROJECT / ".venv-vllm" / "bin" / "python"


def worker_python():
    raw = os.environ.get("OWL_VLLM_PYTHON") or str(DEFAULT_WORKER_PYTHON)
    path = Path(raw)
    if not path.is_file():
        raise ValueError(
            f"vLLM worker python not found: {path}; run "
            "research/open_weight_lingua/scripts/setup_vllm_worker.sh or set "
            "OWL_VLLM_PYTHON to a pinned worker interpreter"
        )
    return path


def probe_worker(python=None):
    """Version facts for the manifest; refuses an unpinned worker interpreter."""
    python = Path(python) if python else worker_python()
    probe = (
        "import importlib.metadata as m, json, platform; "
        "print(json.dumps({'vllm': m.version('vllm'), 'torch': m.version('torch'), "
        "'transformers': m.version('transformers'), "
        "'python': platform.python_version()}))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"vLLM worker python probe failed: {exc}") from exc
    if completed.returncode != 0:
        raise ValueError(
            "vLLM worker python cannot import the pinned stack; run "
            f"scripts/setup_vllm_worker.sh. stderr tail: {completed.stderr[-2000:]}"
        )
    info = json.loads(completed.stdout.strip().splitlines()[-1])
    if info["vllm"] != VLLM_PIN:
        raise ValueError(
            f"vLLM worker pin mismatch: {info['vllm']} != {VLLM_PIN}; rebuild with "
            "scripts/setup_vllm_worker.sh"
        )
    return {"python_path": str(python), **info}


def _run_worker(mode, job_path):
    # A hung worker (e.g. engine-core deadlock) must fail the run loudly, not
    # hang it; the default ceiling is far above any measured stage time. The
    # worker runs in its own process group so a timeout also reaps the forked
    # engine core, which would otherwise linger holding GPU memory.
    timeout = float(os.environ.get("OWL_VLLM_WORKER_TIMEOUT_S", "10800"))
    proc = subprocess.Popen(
        [str(worker_python()), str(WORKER_SCRIPT), mode, "--job", str(job_path)],
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        raise RuntimeError(
            f"vLLM worker {mode} exceeded {timeout:.0f}s and was killed; "
            "inspect the worker job directory for a hung engine"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"vLLM worker {mode} failed (exit {proc.returncode}); stderr tail:\n"
            f"{stderr[-4000:]}"
        )


def _read_completion(output_dir, mode):
    try:
        completion = json.loads((output_dir / "completion.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"vLLM worker {mode} wrote no readable completion: {exc}"
        ) from exc
    if completion.get("status") not in ("COMPLETE", "COMPLETE_WITH_FAILURES"):
        raise RuntimeError(
            f"vLLM worker {mode} completion status: {completion.get('status')}"
        )
    for name, recorded in completion.get("outputs", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != recorded["sha256"]:
            raise RuntimeError(f"vLLM worker {mode} output hash mismatch: {name}")
    return completion


def verbalize_batch(
    *, av_path, av_meta, tokenizer, vectors, engine=None, work_dir=None
):
    """One worker invocation verbalizing every activation; DescriptionRecord rows.

    vectors: {row_id: 1-D activation tensor}. Returns (records, completion).
    """
    prompt_ids, position = av_prompt(tokenizer, av_meta)
    cleanup = work_dir is None
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="owl_vllm_av_"))
    try:
        work.mkdir(parents=True, exist_ok=True)
        save_numeric(work / "activations.safetensors", dict(vectors))
        job = {
            "schema_version": 1,
            "mode": "verbalize",
            "model_dir": str(av_path),
            "output_dir": str(work / "out"),
            "payload_safetensors": "activations.safetensors",
            "rows": [{"id": row_id} for row_id in vectors],
            "metadata": {
                "width": av_meta.width,
                "injection_scale": av_meta.injection_scale,
                "injection_id": av_meta.injection_id,
                "left_id": av_meta.left_id,
                "right_id": av_meta.right_id,
                "prompt_token_ids": prompt_ids,
                "injection_position": position,
                "eos_token_id": tokenizer.eos_token_id,
                "max_new_tokens": 200,
            },
            "engine": engine or {},
        }
        write_json(work / "job.json", job)
        _run_worker("verbalize", work / "job.json")
        completion = _read_completion(work / "out", "verbalize")
        try:
            records = json.loads((work / "out" / "descriptions.json").read_text())[
                "records"
            ]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"vLLM worker verbalize output unreadable: {exc}"
            ) from exc
        if set(records) != set(vectors):
            raise RuntimeError("vLLM worker returned a mismatched row set")
        for row_id, record in records.items():
            token_ids = record.get("token_ids")
            if token_ids is None:
                continue
            # The pipeline tokenizer (transformers 4.57) is the convention owner;
            # a worker decode disagreement is a loud cross-version failure.
            recomputed = tokenizer.decode(token_ids, skip_special_tokens=False)
            if recomputed != record.get("raw_text"):
                raise RuntimeError(
                    f"worker/parent tokenizer decode divergence on row {row_id}"
                )
        return records, completion
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


def reconstruct_batch(
    *, ar_path, ar_meta, tokenizer, head_path, items, engine=None, work_dir=None
):
    """One worker invocation reconstructing every description; direction vectors.

    items: [(key, description)]. Returns (vectors {key: tensor}, completion).
    """
    cleanup = work_dir is None
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="owl_vllm_ar_"))
    try:
        work.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, description in items:
            # Parent-computed ids carry the eager tokenizer convention; the
            # worker re-derives and must reproduce them exactly.
            rows.append(
                {
                    "id": key,
                    "description": description,
                    "prompt_token_ids": ar_prompt(tokenizer, ar_meta, description),
                }
            )
        job = {
            "schema_version": 1,
            "mode": "reconstruct",
            "model_dir": str(ar_path),
            "value_head": str(head_path),
            "output_dir": str(work / "out"),
            "rows": rows,
            "metadata": {
                "width": ar_meta.width,
                "ar_template": ar_meta.ar_template,
                "suffix_ids": list(ar_meta.suffix_ids),
            },
            "engine": engine or {},
        }
        write_json(work / "job.json", job)
        _run_worker("reconstruct", work / "job.json")
        completion = _read_completion(work / "out", "reconstruct")
        try:
            vectors = load_numeric(work / "out" / "directions.safetensors")
        except ValueError as exc:
            raise RuntimeError(
                f"vLLM worker reconstruct output unreadable: {exc}"
            ) from exc
        expected = {key for key, _ in items}
        if set(vectors) - expected:
            raise RuntimeError("vLLM worker returned unexpected direction keys")
        return vectors, completion
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)
