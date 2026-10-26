"""The GPU-reservation ledger: one owner at a time, and an enforced cap on aggregate reservation hours.

**Example.** The work order allows 8 aggregate GPU-reservation hours. Suppose earlier reservations already used:
- 0.4 h profiling the 4B model;
- 1.1 h profiling the 27B model, including one load that failed;
- 2.0 h of baseline inference.

That is 3.5 h. A paired smoke run projected at 1.2 h can start, since 3.5 + 1.2 ≤ 8. One projected at 5 h is
refused before it touches the GPU.

**What counts.** A reservation starts **before** model loading, so loading, warm-up, profiling, failed attempts, baseline
inference and smoke training are all charged. The log is append-only, `runs/ti_tnn_scaling/gpu_ledger.jsonl`:
- a start entry, then a heartbeat every 60 s, then an end entry with its status;
- a reservation that never ended (its process was killed) is charged until its last heartbeat plus one interval;
- foreign GPU processes are logged at start and end and never touched.

**One owner.** An exclusive `flock` on `runs/ti_tnn_scaling/gpu.lock` is held for the whole reservation, so two of this
study's processes cannot hold the GPU at once.

**Enforcement.** `Reservation.check()` raises `BudgetExceeded` once the running reservation would push the aggregate past
the cap. Long loops call it between units of work, and a watchdog thread sets the same flag on its own.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import contracts as C

LEDGER = C.RUNS / "gpu_ledger.jsonl"
LOCK = C.RUNS / "gpu.lock"
CAP_HOURS = 8.0
HEARTBEAT_SECONDS = 60.0


class BudgetExceeded(RuntimeError):
    pass


def _now() -> float:
    return time.time()


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


def read(path: Path = LEDGER) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def usage(path: Path = LEDGER, now: float | None = None) -> dict:
    """Charged hours per reservation and in total. A reservation with no end is charged to its last heartbeat plus one
    interval (or to `now`, if it is still alive and holds the lock)."""
    now = _now() if now is None else now
    res: dict[str, dict] = {}
    for e in read(path):
        r = res.setdefault(e["reservation"], {"reservation": e["reservation"], "start": None, "last": None, "end": None,
                                              "status": None, "command": None})
        if e["event"] == "start":
            r.update(start=e["t"], last=e["t"], command=e.get("command"), projected_hours=e.get("projected_hours"))
        elif e["event"] == "heartbeat":
            r["last"] = max(r["last"] or e["t"], e["t"])
        elif e["event"] == "end":
            r.update(end=e["t"], status=e.get("status"))
    total = 0.0
    for r in res.values():
        if r["start"] is None:
            continue
        stop = r["end"] if r["end"] is not None else min(now, (r["last"] or r["start"]) + HEARTBEAT_SECONDS)
        r["hours"] = max(0.0, stop - r["start"]) / 3600.0
        r["open"] = r["end"] is None
        total += r["hours"]
    return {"cap_hours": CAP_HOURS, "used_hours": total, "remaining_hours": CAP_HOURS - total,
            "reservations": sorted(res.values(), key=lambda r: r["start"] or 0)}


class Reservation:
    """Hold the GPU for one command. Use as a context manager; the end entry is written on any exit."""

    def __init__(self, command: str, projected_hours: float, cap_hours: float = CAP_HOURS, path: Path = LEDGER,
                 lock: Path = LOCK, extra: dict | None = None):
        self.command, self.projected_hours, self.cap_hours = command, float(projected_hours), float(cap_hours)
        self.path, self.lock_path, self.extra = Path(path), Path(lock), extra or {}
        self.id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self._lock_file = None
        self._stop = threading.Event()
        self.exceeded = threading.Event()
        self.t_start = None
        self.used_before = 0.0

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __enter__(self) -> "Reservation":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError("another process of this study holds the GPU reservation lock")
        u = usage(self.path)
        self.used_before = u["used_hours"]
        if self.used_before + self.projected_hours > self.cap_hours:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            raise BudgetExceeded(f"{self.command}: {self.used_before:.3f} h used + {self.projected_hours:.3f} h projected "
                                 f"exceeds the {self.cap_hours} h cap")
        self.t_start = _now()
        self._append({"event": "start", "reservation": self.id, "t": self.t_start, "iso": _iso(self.t_start),
                      "command": self.command, "projected_hours": self.projected_hours, "cap_hours": self.cap_hours,
                      "used_before_hours": self.used_before, "pid": os.getpid(), "host": socket.gethostname(),
                      "foreign_gpu_processes": foreign_gpu_processes(), **self.extra})
        threading.Thread(target=self._heartbeat, daemon=True).start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            t = _now()
            self._append({"event": "heartbeat", "reservation": self.id, "t": t})
            if self.used_before + (t - self.t_start) / 3600.0 > self.cap_hours:
                self.exceeded.set()

    def elapsed_hours(self) -> float:
        return (_now() - self.t_start) / 3600.0 if self.t_start else 0.0

    def check(self) -> None:
        """Raise once this reservation would take the aggregate past the cap. Call it between units of work."""
        if self.exceeded.is_set() or self.used_before + self.elapsed_hours() > self.cap_hours:
            self.exceeded.set()
            raise BudgetExceeded(f"{self.command}: the aggregate GPU-reservation cap ({self.cap_hours} h) is reached")

    def remaining_hours(self) -> float:
        return self.cap_hours - self.used_before - self.elapsed_hours()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        t = _now()
        status = "ok" if exc_type is None else ("budget_exceeded" if exc_type is BudgetExceeded else "failed")
        self._append({"event": "end", "reservation": self.id, "t": t, "iso": _iso(t), "status": status,
                      "hours": (t - self.t_start) / 3600.0, "error": None if exc is None else repr(exc)[:500],
                      "foreign_gpu_processes": foreign_gpu_processes()})
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        return False
