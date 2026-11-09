# TI-TNN native bring-up: starting state

Inspection window: approximately **2026-11-08 23:20–23:35 America/New_York** (EST, UTC−05:00). This snapshot records the starting point for native work; it is not a training result.

## Repository and preservation boundary

- Isolated worktree: `/home/seanjazm27/projects/universa-recurrent-native`, branch `ti-tnn-native-bringup`.
- The worktree `HEAD` and actual remote `main` were both `e03d667b1a9a19b7e6b9f2769f429cc973972073`. The commit is the merge of PR #75; its reviewed head was `d910651e48c5c64705fd506419229fa0a07c2239`.
- `gh` confirmed PR #75's checks and post-merge `Tests` push run [42514326369](https://github.com/seanm27lol/universa-recurrent/actions/runs/42514326369) on the merge commit. The run completed with overall conclusion `success`; its substantive jobs passed and `commit-messages` was skipped.
- Canonical Git common directory: `/home/seanjazm27/projects/universa-recurrent-git/.git`.
- `git worktree list` showed 30 historical worktrees plus this new native worktree. Every historical worktree, producer, protocol, evidence file, manifest, and result was left untouched. Native work belongs only in the new worktree.

## Host, processes, and storage

- GPU: **NVIDIA GB10**.
- CPU: **20-core aarch64**.
- Memory: **121 GiB unified RAM**.
- Storage: about **1.6 TiB free** on `/dev/nvme0n1p2`, mounted at `/`.
- Historical shared runs root: `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling`; `du -sh` reported **181G** (about **181 GiB**).
- Unrelated GPU compute process: PID **514236**, command `.venv/bin/python ui_server.py`, working directory `/home/seanjazm27/rf-moe`, using **533 MiB** GPU memory in `nvidia-smi`. **Never terminate this process.** This identity is a time-stamped observation, so reverify it rather than relying on the PID later.
- No project training process was found in the process listing during inspection.

## Existing environment

Reuse `/home/seanjazm27/projects/universa-recurrent-scaling/.venv-scaling` as-is; **do not upgrade it**. Read-only package-metadata inspection confirmed:

- CPython **3.12.3**
- `torch` **2.13.0+cu130**
- `tokenizers` **0.23.3**
- `transformers` **5.20.2**
- `pytest` **9.1.1**

## GPU ledger and native authorization

The live ledger is `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling/gpu_ledger.jsonl`, with lock file `gpu.lock` beside it. Inspection used the existing ledger's accounting rule and made **no change** to either file.

- Historical reservations: **62**.
- Actual historical use: **23.951425708866115 h**.
- Historical cap: **32 h**, leaving **8.048574291133885 h** under that old cap.
- One old killed `pilot-encode train_1200-2400` reservation, `20261030T182008-e81ea738`, has no end entry. It remains charged **0.5586347438679594 h**: its last heartbeat elapsed value, **1981.085077924654 s**, plus one **30 s** heartbeat interval, divided by 3600.
- The latest reservation, DBE `20261107T182054-3c584e74` (`dbe run`), ended with status `ok`.
- A nonblocking `flock` probe acquired and released the lock, so there was **no active lock owner**. The old unended accounting record does not imply a live owner.

**Native authorization is version 3.** Its cumulative cap is:

```text
23.951425708866115 h preserved actual historical use
+ 12 h                  new native allowance
= 35.951425708866115 h cumulative cap
```

Thus native work starts with **exactly 12 additional hours**, not the old unused **8.048574291133885 h** plus 12 hours, and not the old 32 h cap plus 12 hours. The unused old allowance is not added. Version 3 authorizes subsequent native work; it does not retroactively rewrite the historical ledger, whose entries must remain intact.

## Evidence boundary

These checks establish repository provenance, host capacity, package versions, process state, and budget accounting. They do **not** establish learning. For example, constructing a native model and passing tensor-shape or unit tests would be a **construction check**; it would not show that training improves loss or downstream task performance. **Learning evidence** requires separately reported training and evaluation measurements under an explicit protocol. No native learning result is claimed here.

## Read-only reverification

The following commands reproduce the inspection without fetching, installing, training, terminating processes, or writing the ledger. Values are a snapshot and may legitimately change later.

```bash
NATIVE=/home/seanjazm27/projects/universa-recurrent-native
SCALING=/home/seanjazm27/projects/universa-recurrent-scaling
RUNS="$SCALING/runs/ti_tnn_scaling"

cd "$NATIVE"
git status --short --branch
git branch --show-current
git rev-parse HEAD origin/main
git rev-parse --path-format=absolute --git-common-dir
git show -s --format='%H%n%P%n%s' HEAD
git worktree list
gh api repos/seanm27lol/universa-recurrent/branches/main --jq '.commit.sha'
gh pr view 75 --json mergeCommit,headRefOid,statusCheckRollup,url
gh pr checks 75
gh run view 42514326369 --json event,status,conclusion,headSha,url,jobs

nvidia-smi
lscpu
free -h
df -h "$NATIVE"
du -sh "$RUNS"
ps -p 514236 -o pid=,ppid=,stat=,etime=,args=
readlink -f /proc/514236/cwd
ps -eo pid=,args= | awk 'BEGIN{IGNORECASE=1} /ti_tnn_train|torchrun|deepspeed|universa-recurrent.*(train|fit)|python.*(train|fit)/ && $0 !~ /awk/ {print}'

PYTHONDONTWRITEBYTECODE=1 "$SCALING/.venv-scaling/bin/python" -c \
  'import sys; from importlib.metadata import version; print(sys.version); [print(f"{name}=={version(name)}") for name in ("torch", "tokenizers", "transformers", "pytest")]'

cd "$SCALING"
PYTHONDONTWRITEBYTECODE=1 .venv-scaling/bin/python - <<'PY'
import json
from decimal import Decimal
from pathlib import Path
from research.ti_tnn.scaling.ledger import usage

path = Path("runs/ti_tnn_scaling/gpu_ledger.jsonl")
events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
summary = usage(path)
reservations = summary["reservations"]
print("reservations:", len(reservations))
print("used_hours:", repr(summary["used_hours"]))
print("unended:", [(r["reservation"], r["command"], repr(r["hours"]))
                   for r in reservations if r["open"]])
starts = [event for event in events if event["event"] == "start"]
print("latest_start:", {key: starts[-1].get(key) for key in
                        ("reservation", "command", "authorization", "cap_hours")})
print("latest_event:", events[-1])
print("native_v3_cap_hours:", Decimal(str(summary["used_hours"])) + Decimal("12"))
PY

# Success prints no-active-lock-owner and immediately releases the lock.
flock -n "$RUNS/gpu.lock" -c 'echo no-active-lock-owner'
sha256sum "$RUNS/gpu_ledger.jsonl" "$RUNS/gpu.lock"
```
