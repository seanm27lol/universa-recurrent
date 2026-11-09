"""Native bring-up accounting under authorization v3, using the tested scaling ledger unchanged.

The cap is cumulative. For example, 23.951425708866115 historical hours plus 2 native hours leave 10 native hours,
not 12 fresh hours after each run. Failed attempts and interrupted reservations remain charges. Superseding v2 means
only that this work order authorizes native bring-up; every historical line and charge is preserved.

A reservation uses the same anchored ``TI_TNN_SCALING_RUNS`` directory, lock, and JSONL ledger as the scaling study.
This module changes only the start metadata for the v3 cap and the interpretation of the additional 12-hour allowance.
The base class still reconstructs durations, enforces admission and watchdog limits, and writes all end records.
"""
from __future__ import annotations

from pathlib import Path

from scaling import ledger as _base

HISTORICAL_HOURS = 23.951425708866115
ADDITIONAL_HOURS = 12.0
CAP_HOURS = 35.951425708866115

NATIVE_AUTHORIZATION = {
    "version": 3,
    "cap_hours": CAP_HOURS,
    "historical_hours": HISTORICAL_HOURS,
    "additional_hours": ADDITIONAL_HOURS,
    "date": "2026-11-08",
    "supersedes": 2,
    "scope": "native randomly initialized model bringup: loading, warm-up, compilation, profiling, training, "
              "evaluation, failures, and GPU reproduction; 12 additional GPU-reservation hours beyond "
              "23.951425708866115 actual historical hours, for a 35.951425708866115-hour cumulative cap; "
              "supersedes v2 only as a new work-order authorization and preserves every historical charge",
}
AUTHORIZATION = NATIVE_AUTHORIZATION
AUTHORIZATIONS = (*_base.AUTHORIZATIONS, NATIVE_AUTHORIZATION)

BudgetExceeded = _base.BudgetExceeded
Terminated = _base.Terminated
LedgerCorrupt = _base.LedgerCorrupt
read = _base.read
shared_runs = _base.shared_runs
ledger_path = _base.ledger_path
anchor_file = _base.anchor_file


def cap_of(version: int) -> float:
    """Return the cumulative cap for any historical authorization or for native v3."""
    for authorization in AUTHORIZATIONS:
        if authorization["version"] == version:
            return authorization["cap_hours"]
    raise KeyError(f"no authorization version {version}")


def validate_shared_reservation_paths(path: Path | None = None, lock: Path | None = None) -> tuple[Path, Path]:
    """Validate production paths without creating an anchor, ledger, lock, or directory.

    Real CUDA/study callers must use the existing anchor and the canonical files below its resolved runs root.
    Ordinary CPU tests can still give the base ``Reservation`` both temporary paths directly.
    """
    runs = shared_runs().resolve()
    anchor = anchor_file()
    if not anchor.is_file():
        raise RuntimeError(f"a production GPU reservation needs the existing shared ledger anchor at {anchor}")
    anchored = anchor.read_text().strip()
    if anchored != str(runs):
        raise RuntimeError(f"the shared ledger anchor is {anchored!r}, not the resolved TI_TNN_SCALING_RUNS root "
                           f"{str(runs)!r}")
    ledger = runs / "gpu_ledger.jsonl"
    lock_path = runs / "gpu.lock"
    if path is not None and Path(path).resolve() != ledger:
        raise RuntimeError(f"the native ledger must be {ledger}, not {path}")
    if lock is not None and Path(lock).resolve() != lock_path:
        raise RuntimeError(f"the native reservation lock must be {lock_path}, not {lock}")
    if not ledger.is_file() or ledger.stat().st_size == 0:
        raise RuntimeError(f"the production GPU ledger must already contain the preserved historical charges: {ledger}")
    historical = _base.usage(ledger)["used_hours"]
    if historical + 1e-12 < HISTORICAL_HOURS:
        raise RuntimeError(
            f"the production GPU ledger reconstructs {historical:.15f} h, below the preserved "
            f"{HISTORICAL_HOURS:.15f} h historical baseline; refusing to reset accounting"
        )
    return ledger, lock_path


def usage(path: Path | None = None, now: float | None = None) -> dict:
    """Return base durations and every reservation, plus native use beyond the fixed historical baseline."""
    report = _base.usage(path=path, now=now)
    total = report["used_hours"]
    native_used = max(0.0, total - HISTORICAL_HOURS)
    report.update(
        cap_hours=CAP_HOURS,
        authorization=AUTHORIZATION["version"],
        remaining_hours=CAP_HOURS - total,
        historical_hours=HISTORICAL_HOURS,
        native_used_hours=native_used,
        native_remaining_hours=max(0.0, ADDITIONAL_HOURS - native_used),
    )
    return report


class Reservation(_base.Reservation):
    """A base reservation with the v3 cumulative cap and v3 start metadata; charging behavior is inherited."""

    def __init__(self, command: str, projected_hours: float, path: Path | None = None,
                 lock: Path | None = None, extra: dict | None = None, terminate: bool = True):
        super().__init__(command=command, projected_hours=projected_hours, cap_hours=CAP_HOURS, path=path, lock=lock,
                         extra=extra, terminate=terminate)

    def _append(self, entry: dict) -> None:
        if entry.get("event") == "start" and entry.get("cap_hours") == CAP_HOURS:
            entry = dict(entry)
            entry["authorization"] = AUTHORIZATION["version"]
            entry["native_authorization"] = dict(NATIVE_AUTHORIZATION)
        super()._append(entry)
