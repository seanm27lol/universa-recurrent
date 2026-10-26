"""The GPU-reservation ledger: one owner at a time, and an enforced cap on aggregate reservation hours.

**Example.** The work order allows 8 aggregate GPU-reservation hours. Suppose earlier reservations used:
- 0.4 h profiling the 4B;
- 1.1 h profiling the 27B, including one failed load;
- 2.0 h of baseline inference.

That is 3.5 h used. A paired smoke run projected at 1.2 h may start (3.5 + 1.2 ≤ 8). It is then held to
min(8, 3.5 + 1.2 × 1.5) = 5.3 h of aggregate use. Past that, the watchdog writes the end entry
(`budget_exceeded`) and terminates the process. A run projected at 5 h is refused before it touches the GPU.

**One ledger for every worktree.** GPU commands run from a clean worktree at the commit being run, and edits happen in
another worktree. A ledger inside each worktree would let the cap and the one-owner rule be split between them. So a
reservation requires `TI_TNN_SCALING_RUNS`, set to the single shared, **absolute** runs directory, and refuses
otherwise. The resolved ledger path is written into every start entry.

**What counts.** A reservation opens **before** model loading, so loading, warm-up, profiling, failed attempts, baseline
inference and smoke training are all charged. The log is append-only, `<runs>/gpu_ledger.jsonl`:
- a start entry, heartbeats, then one end entry;
- durations come from a monotonic clock, so a wall-clock step cannot shrink a charge;
- a reservation that never ended (its process was killed) is charged to its last heartbeat plus one interval;
- a heartbeat that cannot be written makes `check()` fail;
- foreign GPU processes are logged at start and end and never touched.

**One owner.** An exclusive `flock` on `<runs>/gpu.lock` is held for the whole reservation.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import contracts as C

CAP_HOURS = 8.0
HEARTBEAT_SECONDS = 30.0
PROJECTION_MARGIN = 0.5      # a run is held to its own projection × 1.5, within the aggregate cap
TERMINATION_GRACE_SECONDS = 90.0

_ACTIVE = None               # the reservation this process holds, if any


class BudgetExceeded(RuntimeError):
    pass


class LedgerCorrupt(RuntimeError):
    pass


def shared_runs() -> Path:
    """The one shared runs directory every worktree must use for GPU reservations."""
    value = os.environ.get("TI_TNN_SCALING_RUNS")
    if not value:
        raise RuntimeError("a GPU reservation needs TI_TNN_SCALING_RUNS set to the shared, absolute runs directory "
                           "(e.g. /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling)")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"TI_TNN_SCALING_RUNS must be absolute, not {value!r}")
    return path


def ledger_path() -> Path:
    """The shared ledger if configured; otherwise this worktree's (read-only use, e.g. preflight)."""
    try:
        return shared_runs() / "gpu_ledger.jsonl"
    except RuntimeError:
        return C.RUNS / "gpu_ledger.jsonl"


def active():
    return _ACTIVE


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")


def foreign_gpu_processes() -> list[dict]:
    """GPU compute processes that are not this process (logged, never touched)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except Exception as e:
        return [{"error": repr(e)}]
    out = []
    for line in q.stdout.strip().splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) != os.getpid():
            out.append({"pid": int(parts[0]), "name": parts[1], "used_memory": parts[2]})
    return out


def read(path: Path | None = None) -> list[dict]:
    path = Path(path) if path is not None else ledger_path()
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            last = i == len(lines) - 1 and not text.endswith("\n")
            raise LedgerCorrupt(f"{path}: line {i + 1} is not valid JSON ({e}). "
                                + ("It is a partial final line: a process died mid-write. Repair by hand: keep the line "
                                   "if a start can be recovered from it, otherwise remove it and charge that "
                                   "reservation conservatively (to now) in a manual end entry." if last else
                                   "Repair by hand before any further reservation.")) from e
    return out


def usage(path: Path | None = None, now: float | None = None) -> dict:
    """Charged hours per reservation and in total, from monotonic durations. A reservation with no end entry is
    charged to its last heartbeat's elapsed time plus one interval."""
    path = Path(path) if path is not None else ledger_path()
    res: dict[str, dict] = {}
    for e in read(path):
        r = res.setdefault(e["reservation"], {"reservation": e["reservation"], "start": None, "elapsed": 0.0,
                                              "ended": False, "status": None, "command": None})
        if e["event"] == "start":
            r.update(start=e["t"], command=e.get("command"), projected_hours=e.get("projected_hours"))
        elif e["event"] == "heartbeat":
            r["elapsed"] = max(r["elapsed"], float(e.get("elapsed", 0.0)))
        elif e["event"] == "end":
            r.update(ended=True, status=e.get("status"), elapsed=max(r["elapsed"], float(e.get("elapsed", 0.0))))
    total = 0.0
    for r in res.values():
        if r["start"] is None:
            continue
        seconds = r["elapsed"] if r["ended"] else r["elapsed"] + HEARTBEAT_SECONDS
        r["hours"] = seconds / 3600.0
        r["open"] = not r["ended"]
        total += r["hours"]
    return {"ledger": str(path), "cap_hours": CAP_HOURS, "used_hours": total, "remaining_hours": CAP_HOURS - total,
            "reservations": sorted(res.values(), key=lambda r: r["start"] or 0)}


class Reservation:
    """Hold the GPU for one command. Use as a context manager; exactly one end entry is written on any exit."""

    def __init__(self, command: str, projected_hours: float, cap_hours: float = CAP_HOURS, path: Path | None = None,
                 lock: Path | None = None, extra: dict | None = None, terminate: bool = True):
        if not isinstance(projected_hours, (int, float)) or isinstance(projected_hours, bool) \
                or not math.isfinite(projected_hours) or projected_hours <= 0:
            raise ValueError(f"projected_hours must be a finite positive number, not {projected_hours!r}")
        self.command, self.projected_hours, self.cap_hours = command, float(projected_hours), float(cap_hours)
        runs = shared_runs() if path is None or lock is None else None
        self.path = Path(path) if path is not None else runs / "gpu_ledger.jsonl"
        self.lock_path = Path(lock) if lock is not None else runs / "gpu.lock"
        self.extra, self.terminate = extra or {}, terminate
        self.id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self._lock_file = None
        self._stop = threading.Event()
        self._end_lock = threading.Lock()
        self._ended = False
        self.exceeded = threading.Event()
        self.heartbeat_failed = threading.Event()
        self.t_start = self.m_start = None
        self.used_before = 0.0
        self.limit_hours = None
        self._thread = None
        self._old_handler = None

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.m_start if self.m_start is not None else 0.0

    def elapsed_hours(self) -> float:
        return self.elapsed_seconds() / 3600.0

    def __enter__(self) -> "Reservation":
        global _ACTIVE
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError("another process of this study holds the GPU reservation lock")
        try:
            self.used_before = usage(self.path)["used_hours"]
        except Exception:
            self._release_lock()
            raise
        if self.used_before + self.projected_hours > self.cap_hours:
            self._release_lock()
            raise BudgetExceeded(f"{self.command}: {self.used_before:.3f} h used + {self.projected_hours:.3f} h projected "
                                 f"exceeds the {self.cap_hours} h cap")
        self.limit_hours = min(self.cap_hours, self.used_before + self.projected_hours * (1 + PROJECTION_MARGIN))
        self.t_start, self.m_start = time.time(), time.monotonic()
        self._append({"event": "start", "reservation": self.id, "t": self.t_start, "iso": _iso(self.t_start),
                      "elapsed": 0.0, "command": self.command, "projected_hours": self.projected_hours,
                      "limit_hours": self.limit_hours, "cap_hours": self.cap_hours,
                      "used_before_hours": self.used_before, "ledger": str(self.path), "pid": os.getpid(),
                      "host": socket.gethostname(), "foreign_gpu_processes": foreign_gpu_processes(),
                      "extra": self.extra})
        if threading.current_thread() is threading.main_thread():
            self._old_handler = signal.signal(signal.SIGTERM, self._on_sigterm)
        self._thread = threading.Thread(target=self._watchdog, daemon=True)
        self._thread.start()
        _ACTIVE = self
        return self

    def _on_sigterm(self, signum, frame):
        if self.exceeded.is_set():
            raise BudgetExceeded(f"{self.command}: terminated at the reservation limit ({self.limit_hours:.3f} h aggregate)")
        raise RuntimeError(f"{self.command}: terminated by SIGTERM")

    def _over(self) -> bool:
        return self.used_before + self.elapsed_hours() > self.limit_hours

    def _watchdog(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self._append({"event": "heartbeat", "reservation": self.id, "t": time.time(),
                              "elapsed": self.elapsed_seconds()})
            except Exception:
                self.heartbeat_failed.set()  # check() now fails closed
            if self._over() and not self.exceeded.is_set():
                self.exceeded.set()
                self._end("budget_exceeded", f"limit {self.limit_hours:.3f} h aggregate reached")
                if self.terminate:
                    os.kill(os.getpid(), signal.SIGTERM)
                    threading.Timer(TERMINATION_GRACE_SECONDS, lambda: os._exit(3)).start()

    def check(self) -> None:
        """Raise if this reservation reached its limit or its heartbeat failed. Call it between units of work."""
        if self.heartbeat_failed.is_set():
            raise BudgetExceeded(f"{self.command}: the ledger heartbeat could not be written; stopping (fail closed)")
        if self.exceeded.is_set() or self._over():
            self.exceeded.set()
            raise BudgetExceeded(f"{self.command}: the reservation limit ({self.limit_hours:.3f} h aggregate) is reached")

    def remaining_hours(self) -> float:
        return self.limit_hours - self.used_before - self.elapsed_hours()

    def _end(self, status: str, error: str | None) -> None:
        with self._end_lock:
            if self._ended:
                return
            self._ended = True
            t = time.time()
            self._append({"event": "end", "reservation": self.id, "t": t, "iso": _iso(t), "status": status,
                          "elapsed": self.elapsed_seconds(), "hours": self.elapsed_hours(),
                          "error": None if error is None else str(error)[:500],
                          "foreign_gpu_processes": foreign_gpu_processes()})

    def _release_lock(self) -> None:
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        finally:
            self._lock_file.close()

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.exceeded.is_set():
            status = "budget_exceeded"
        elif exc_type is None:
            status = "ok"
        else:
            status = "budget_exceeded" if exc_type is BudgetExceeded else "failed"
        self._end(status, None if exc is None else repr(exc))
        if self._old_handler is not None:
            signal.signal(signal.SIGTERM, self._old_handler)
        _ACTIVE = None
        self._release_lock()
        return False
