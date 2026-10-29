"""Verify a finished baseline or smoke run from its files: reconstruct, re-execute, re-derive, compare.

**Example.** `python -m scaling.cli verify runs/ti_tnn_scaling/baselines/<stamp>_B4_nonthinking_p0-16` checks:
1. **Identity.** The manifest's commit exists, the research tree was clean, and every recorded source hash matches
   the file at that commit. The configuration hashes to its recorded identity.
2. **Requests.** The requested ids are reconstructed from the pinned inventory, and every record answers exactly one of
   them (no missing, extra or duplicated request).
3. **Snapshots.** Every record's snapshot hash is the pinned hash of its database, and the file on disk still has it.
4. **Outcomes.** Each final SQL is re-executed against the gold SQL with the benchmark scorer. The re-derived correctness
   must equal the recorded one, and an abstention scores 0.
5. **Records.** The common record checks (`records.summarize`): hidden retries, turn and tool budgets, token sums,
   numbers that are finite and non-negative, one arm and one configuration.
6. **Summary.** Quality, the correct count, the elapsed total and the cost per correct request are recomputed and
   compared with `summary.json`.
7. **Ledger.** The run's reservation exists, has ended, and its status is recorded.
8. **Smoke only.** Each final checkpoint re-hashes to its name; its weight and optimizer digests match the recorded
   ones; the training log has one line per update; the reconstruction decoded the same actions and final output for
   every request.

Any failed check is a problem; a run with problems is rejected. The verifier changes nothing.

**Not independent.** This reuses the producer's scorer and record checks. The independent reconstruction (written
without the producer modules) is the reviewer's, under the review's scratch directory.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from . import contracts as C
from . import identity
from . import records as R

TOLERANCE = 1e-9


def _git_file_sha256(commit: str, relpath: str) -> "str | None":
    try:
        blob = subprocess.run(["git", "show", f"{commit}:{relpath}"], cwd=C.REPO, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    return identity.sha256_bytes(blob)


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def _check_identity(manifest: dict, problems: list) -> None:
    git = manifest.get("git", {})
    if git.get("research_tree_dirty"):
        problems.append(f"the research tree was dirty at the run: {git.get('dirty_paths')}")
    commit = git.get("commit")
    rel = C.SCALING.relative_to(C.REPO)
    for name, digest in manifest.get("sources", {}).items():
        at_commit = _git_file_sha256(commit, f"{rel}/{name}")
        if at_commit != digest:
            problems.append(f"source {name}: the recorded hash is not the file at {commit}")
    if identity.content_identity(manifest["config"]) != manifest.get("config_identity"):
        problems.append("the configuration does not hash to its recorded identity")


def _check_records(recs: list, requested: list, problems: list) -> "dict | None":
    from . import data, scorer
    pins = json.loads(Path(data.PIN_SNAPSHOTS).read_text())
    pinned = {db: v["sha256"] for db, v in (pins.get("databases") or pins.get("snapshots") or {}).items()}
    for rec in recs:
        tid = rec.get("task_id")
        db = rec.get("db_id")
        if pinned and pinned.get(db) != rec.get("snapshot_sha256"):
            problems.append(f"{tid}: snapshot {str(rec.get('snapshot_sha256'))[:12]} is not the pinned hash of {db}")
            continue
        snap = data.snapshot(db)
        if snap.sha256 != rec.get("snapshot_sha256"):
            problems.append(f"{tid}: the snapshot hash differs from data.snapshot({db})")
            continue
        from .tools import verify_snapshot
        try:
            verify_snapshot(snap, rehash=True)
        except Exception as e:  # the file on disk changed after the run
            problems.append(f"{tid}: {e}")
            continue
        final = rec.get("final") or {}
        rescored = scorer.score(snap, final.get("sql"), data.gold(tid).sql)
        if (rec.get("score") or {}).get("correct") != rescored.get("correct"):
            problems.append(f"{tid}: recorded correct={rec.get('score', {}).get('correct')!r}, "
                            f"re-executed {rescored.get('correct')!r} ({rescored.get('status')})")
    try:
        return R.summarize(recs, requested)
    except R.RecordError as e:
        problems.append(f"records: {e}")
        return None


def _check_summary(recomputed: dict, recorded: dict, problems: list, where: str) -> None:
    for key in ("requested", "correct", "answered", "abstained", "errors", "timeouts", "repairs", "tool_calls"):
        if recomputed.get(key) != recorded.get(key):
            problems.append(f"{where}: {key} recorded {recorded.get(key)!r}, recomputed {recomputed.get(key)!r}")
    for key in ("quality", "elapsed_seconds_total", "cost_seconds_per_correct"):
        if not _close(recomputed.get(key), recorded.get(key)):
            problems.append(f"{where}: {key} recorded {recorded.get(key)!r}, recomputed {recomputed.get(key)!r}")


def _check_ledger(run_dir: Path, problems: list) -> list:
    from . import ledger
    found = []
    for entry in ledger.read():
        if entry.get("event") == "start" and (entry.get("extra") or {}).get("run_dir") == str(run_dir):
            found.append(entry["reservation"])
    usage = {r["reservation"]: r for r in ledger.usage()["reservations"]}
    out = [usage[r] for r in found if r in usage]
    if not out:
        problems.append("no ledger reservation names this run directory")
    for r in out:
        if not r.get("ended"):
            problems.append(f"reservation {r['reservation']} has no end entry")
    return out


def _check_smoke(run_dir: Path, summary: dict, config: dict, problems: list) -> None:
    from . import train
    for arm, arm_out in summary.get("arms", {}).items():
        ckpt = arm_out["checkpoint"]
        try:
            loaded = train.load_checkpoint(ckpt["path"])
        except Exception as e:
            problems.append(f"{arm}: checkpoint {ckpt['path']}: {e}")
            continue
        if train.tensor_digest(loaded["model"]) != ckpt["model_digest"]:
            problems.append(f"{arm}: the checkpoint's weights do not match the recorded digest")
        if train.optimizer_digest(loaded["optimizer"]) != ckpt["optimizer_digest"]:
            problems.append(f"{arm}: the checkpoint's optimizer state does not match the recorded digest")
        if loaded["update"] != config["updates"] - 1:
            problems.append(f"{arm}: the final checkpoint is update {loaded['update']}, not {config['updates'] - 1}")
        log = [json.loads(x) for x in Path(arm_out["train_log"]).read_text().splitlines() if x.strip()]
        if [r["update"] for r in log] != list(range(config["updates"])):
            problems.append(f"{arm}: the training log does not hold updates 0..{config['updates'] - 1} once each")
        if any(not math.isfinite(r["loss"]) for r in log):
            problems.append(f"{arm}: a non-finite training loss")
        if not arm_out.get("reconstruction_all_equal"):
            bad = [r["task_id"] for r in arm_out.get("reconstruction", []) if not (r["same_actions"] and r["same_final"])]
            problems.append(f"{arm}: the reconstruction differs on {bad}")
        recs = R.load(run_dir / arm / "records.jsonl")
        recomputed = _check_records(recs, config["dev_task_ids"], problems)
        if recomputed is not None and "summary" in arm_out:
            _check_summary(recomputed, arm_out["summary"], problems, arm)


def verify_run(run_dir) -> dict:
    """All checks for one run directory; {"run_dir", "kind", "problems": [...], "summary": recomputed}."""
    from . import data
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = manifest["config"]
    summary = json.loads((run_dir / "summary.json").read_text())
    problems: list = []
    _check_identity(manifest, problems)
    out = {"run_dir": str(run_dir), "kind": config["command"], "problems": problems}
    if config["command"] == "baseline":
        lo, hi = config["positions"]
        requested = data.inventory_ids(C.INVENTORY["size"])[lo:hi]
        if requested != config["requested"]:
            problems.append("the requested ids differ from the pinned inventory at these positions")
        recomputed = _check_records(R.load(run_dir / "records.jsonl"), requested, problems)
        out["summary"] = recomputed
        if recomputed is not None:
            if "summary" not in summary:
                problems.append(f"summary.json has no summary: {summary.get('summary_error')}")
            else:
                _check_summary(recomputed, summary["summary"], problems, "summary")
    elif config["command"] == "smoke":
        if config["dev_task_ids"] != data.inventory_ids(C.INVENTORY["size"])[:16]:
            problems.append("the smoke's development ids are not the first 16 of the pinned inventory")
        pin = json.loads((C.PINS / "ir_coverage_v1.json").read_text())["smoke_train"]["task_ids"]
        if config["train_task_ids"] != pin:
            problems.append("the smoke's training ids are not the pinned smoke_train ids")
        _check_smoke(run_dir, summary, config, problems)
    else:
        problems.append(f"unknown run kind {config['command']!r}")
    if config["command"] != "baseline" or config.get("arm") != "L":
        out["reservations"] = _check_ledger(run_dir, problems)
    out["verdict"] = "accepted" if not problems else "rejected"
    return out
