"""The development evidence bundle: compact, weight-free copies of the runs, and the tables rebuilt from them.

**Example.** After the baselines and the smoke:

    python -m scaling.evidence bundle RUN_DIR [RUN_DIR ...]   # copy into evidence/dev_v1/, write SHA256SUMS
    python -m scaling.evidence tables                         # rebuild every report table from the bundle alone

For each run, `bundle` copies:
- the manifest and the summary;
- the records;
- for the smoke, each arm's config, training log and records. Checkpoints are not copied: weights are not
  evidence here, and their sha256, weight digest and optimizer digest are in the summary.

It also copies the GPU ledger and the verifier's output. `SHA256SUMS` lists every file.

`tables` reads only the bundle. It refuses to run if any file's hash differs from `SHA256SUMS`. It prints the
quality, cost, token and accounting tables used in `docs/REPORT_dev.md`.

`runs/` stays untracked; the bundle is the reviewable copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from . import contracts as C

BUNDLE = C.SCALING / "evidence" / "dev_v1"
COPY = ("manifest.json", "summary.json", "records.jsonl", "profile.json")
SMOKE_ARM_FILES = ("config.json", "train_log.jsonl", "train_log.superseded.jsonl", "records.jsonl")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(run_dirs, verification=None) -> list:
    """Copy each run's evidence files into BUNDLE/<kind>/<run name>/; then the ledger; then SHA256SUMS."""
    copied = []
    for d in map(Path, run_dirs):
        kind = d.parent.name  # baselines | profiles | smoke
        dest = BUNDLE / kind / d.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            if (d / name).exists():
                shutil.copy2(d / name, dest / name)
                copied.append(dest / name)
        if kind == "smoke":
            for arm_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                for name in SMOKE_ARM_FILES:
                    if (arm_dir / name).exists():
                        (dest / arm_dir.name).mkdir(exist_ok=True)
                        shutil.copy2(arm_dir / name, dest / arm_dir.name / name)
                        copied.append(dest / arm_dir.name / name)
    shutil.copy2(C.RUNS / "gpu_ledger.jsonl", BUNDLE / "gpu_ledger.jsonl")
    if verification is not None:
        shutil.copy2(verification, BUNDLE / "verification.json")
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (BUNDLE / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  {p.relative_to(BUNDLE)}\n" for p in files))
    return copied


def _checked() -> None:
    sums = (BUNDLE / "SHA256SUMS").read_text().splitlines()
    for line in sums:
        digest, rel = line.split("  ", 1)
        if _sha256(BUNDLE / rel) != digest:
            raise SystemExit(f"{rel}: its sha256 differs from SHA256SUMS")
    listed = {line.split("  ", 1)[1] for line in sums}
    present = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    if listed != present:
        raise SystemExit(f"files not in SHA256SUMS: {sorted(present - listed)}; missing: {sorted(listed - present)}")


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def tables() -> str:
    """Markdown tables, from the bundle only."""
    _checked()
    out = []
    rows = []
    for d in sorted((BUNDLE / "baselines").iterdir()) if (BUNDLE / "baselines").exists() else []:
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        s = json.loads((d / "summary.json").read_text()).get("summary")
        if not s:
            continue
        recs = _load(d / "records.jsonl")
        mode = cfg["mode"] + (f" ({cfg['reasoning_effort']})" if cfg.get("reasoning_effort") else "")
        t = s["tokens"]
        rows.append(f"| {d.name} | {cfg['arm']} | {mode} | {cfg['positions'][0]}:{cfg['positions'][1]} | "
                    f"{s['correct']}/{s['requested']} | {s['answered']} | {s['abstained']} | {s['tool_calls']} | "
                    f"{s['repairs']} | {s['elapsed_seconds_total']:.1f} | "
                    f"{(s['cost_seconds_per_correct'] or float('nan')):.1f} | {t.get('prefill', 0)} | "
                    f"{t.get('generated', 0)} | {t.get('reasoning', 0)} | "
                    f"{max((r['elapsed_seconds'] for r in recs), default=0):.1f} |")
    out.append("| run | arm | mode | positions | correct | answered | abstained | tool calls | repairs | elapsed s | "
               "s per correct | prefill tok | generated tok | reasoning tok | slowest request s |")
    out.append("|" + "---|" * 15)
    out += rows
    for d in sorted((BUNDLE / "smoke").iterdir()) if (BUNDLE / "smoke").exists() else []:
        s = json.loads((d / "summary.json").read_text())
        out.append("")
        out.append(f"Smoke {d.name}:")
        out.append("")
        out.append("| arm | correct | answered | abstained (reasons) | elapsed s | s per correct | train s | final loss | "
                   "reconstruction equal |")
        out.append("|" + "---|" * 9)
        for arm, a in s["arms"].items():
            sm = a.get("summary") or {}
            log = _load(d / arm / "train_log.jsonl")
            out.append(f"| {a['label']} | {sm.get('correct')}/{sm.get('requested')} | {sm.get('answered')} | "
                       f"{sm.get('abstained')} {json.dumps(sm.get('abstention_reasons'))} | "
                       f"{sm.get('elapsed_seconds_total', float('nan')):.1f} | {sm.get('cost_seconds_per_correct')} | "
                       f"{a['train_seconds']:.1f} | {log[-1]['loss']:.4f} | {a['reconstruction_all_equal']} |")
    ledger = _load(BUNDLE / "gpu_ledger.jsonl")
    out.append("")
    out.append("| reservation | command | status | projected h | charged h |")
    out.append("|---|---|---|---|---|")
    starts = {e["reservation"]: e for e in ledger if e.get("event") == "start"}
    ends = {e["reservation"]: e for e in ledger if e.get("event") == "end"}
    total = 0.0
    for rid, st in starts.items():
        en = ends.get(rid, {})
        hours = en.get("elapsed", 0.0) / 3600
        total += hours
        out.append(f"| {rid} | {st.get('command')} | {en.get('status', 'OPEN')} | {st.get('projected_hours')} | "
                   f"{hours:.3f} |")
    out.append(f"| total | | | | {total:.3f} of 8 |")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m scaling.evidence")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("run_dirs", nargs="+")
    b.add_argument("--verification")
    sub.add_parser("tables")
    args = p.parse_args(argv)
    if args.cmd == "bundle":
        for path in bundle(args.run_dirs, args.verification):
            print(path)
    else:
        print(tables())
    return 0


if __name__ == "__main__":
    sys.exit(main())
