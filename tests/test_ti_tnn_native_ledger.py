"""CPU construction checks for native authorization v3; these are accounting checks, not learning evidence.

Every reservation uses a temporary ledger and lock. GPU process queries are replaced with an empty result, and no test
opens the repository's real anchored reservation. Historical JSONL bytes must remain an unchanged prefix of new output.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from native import ledger as native  # noqa: E402
from scaling import ledger as scaling  # noqa: E402


class _Clock:
    def __init__(self):
        self.wall = 1_800_000_000.0
        self.monotonic_value = 10_000.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.monotonic_value

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic_value += seconds


@pytest.fixture(autouse=True)
def _no_gpu_queries(monkeypatch):
    monkeypatch.setattr(scaling, "foreign_gpu_processes", lambda: [])


@pytest.fixture
def clock(monkeypatch):
    value = _Clock()
    monkeypatch.setattr(scaling, "time", value)
    return value


def _append_closed(path: Path, reservation: str, hours: float, cap_hours: float,
                   authorization: int | None, t: float) -> None:
    start = {"event": "start", "reservation": reservation, "t": t, "elapsed": 0.0,
             "command": reservation, "projected_hours": hours, "cap_hours": cap_hours,
             "limit_hours": cap_hours}
    if authorization is not None:
        start["authorization"] = authorization
    if authorization == 3:
        start["native_authorization"] = dict(native.NATIVE_AUTHORIZATION)
    end = {"event": "end", "reservation": reservation, "t": t + hours * 3600,
           "elapsed": hours * 3600, "status": "ok"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(start, sort_keys=True) + "\n")
        stream.write(json.dumps(end, sort_keys=True) + "\n")


def _historical(path: Path) -> bytes:
    _append_closed(path, "historical-v1", 6.878, 8.0, None, 1_790_000_000.0)
    _append_closed(path, "historical-v2", native.HISTORICAL_HOURS - 6.878, 32.0, 2, 1_790_100_000.0)
    return path.read_bytes()


def test_v3_is_a_new_work_order_without_mutating_scaling_globals():
    assert native.HISTORICAL_HOURS == 23.951425708866115
    assert native.ADDITIONAL_HOURS == 12.0
    assert native.CAP_HOURS == 35.951425708866115
    assert math.isclose(native.CAP_HOURS, native.HISTORICAL_HOURS + native.ADDITIONAL_HOURS,
                        rel_tol=0.0, abs_tol=1e-14)
    assert [a["version"] for a in native.AUTHORIZATIONS] == [1, 2, 3]
    assert [a["version"] for a in scaling.AUTHORIZATIONS] == [1, 2]
    assert scaling.AUTHORIZATION["version"] == 2 and scaling.CAP_HOURS == 32.0
    assert native.AUTHORIZATION["supersedes"] == 2
    assert "new work-order authorization" in native.AUTHORIZATION["scope"]
    assert "preserves every historical charge" in native.AUTHORIZATION["scope"]
    for work in ("randomly initialized", "loading", "warm-up", "compilation", "profiling", "training", "evaluation",
                 "failures", "GPU reproduction"):
        assert work in native.AUTHORIZATION["scope"]
    assert native.cap_of(1) == 8.0 and native.cap_of(2) == 32.0
    assert native.cap_of(3) == native.CAP_HOURS
    with pytest.raises(KeyError):
        native.cap_of(4)
    assert issubclass(native.Reservation, scaling.Reservation)
    assert native.Reservation.__enter__ is scaling.Reservation.__enter__
    assert native.Reservation.__exit__ is scaling.Reservation.__exit__


def test_usage_preserves_every_historical_reservation_and_base_duration(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    before = _historical(path)
    base = scaling.usage(path)
    report = native.usage(path)
    assert report["historical_hours"] == native.HISTORICAL_HOURS
    assert report["used_hours"] == base["used_hours"]
    assert math.isclose(report["used_hours"], native.HISTORICAL_HOURS)
    assert report["native_used_hours"] == 0.0
    assert report["native_remaining_hours"] == 12.0
    assert report["cap_hours"] == native.CAP_HOURS and report["authorization"] == 3
    assert report["reservations"] == base["reservations"]
    assert {r["authorization"] for r in report["reservations"]} == {1, 2}
    assert path.read_bytes() == before


def test_v3_start_metadata_and_historical_lines_are_append_only(tmp_path, clock):
    path, lock = tmp_path / "gpu_ledger.jsonl", tmp_path / "gpu.lock"
    before = _historical(path)
    with native.Reservation("native loading and warmup", projected_hours=0.01, path=path, lock=lock) as reservation:
        assert reservation.path == path and reservation.lock_path == lock
        assert reservation.cap_hours == native.CAP_HOURS
        assert math.isclose(reservation.used_before, native.HISTORICAL_HOURS)
        assert math.isclose(reservation.limit_hours, native.HISTORICAL_HOURS + 0.015)
        clock.advance(45.0)
    after = path.read_bytes()
    assert after.startswith(before)
    new_entries = [json.loads(line) for line in after[len(before):].decode().splitlines()]
    assert [entry["event"] for entry in new_entries] == ["start", "end"]
    start, end = new_entries
    assert start["authorization"] == 3 and start["cap_hours"] == native.CAP_HOURS
    assert start["native_authorization"] == native.NATIVE_AUTHORIZATION
    assert start["native_authorization"]["historical_hours"] == native.HISTORICAL_HOURS
    assert start["native_authorization"]["additional_hours"] == 12.0
    assert "native_authorization" not in end
    report = native.usage(path)
    current = report["reservations"][-1]
    assert current["authorization"] == 3 and current["status"] == "ok"
    assert current["hours"] == 45.0 / 3600.0
    assert math.isclose(report["native_used_hours"], current["hours"], rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(report["native_remaining_hours"], 12.0 - current["hours"], rel_tol=0.0, abs_tol=1e-14)


def test_exactly_twelve_additional_hours_are_available(tmp_path, clock):
    path, lock = tmp_path / "gpu_ledger.jsonl", tmp_path / "gpu.lock"
    _historical(path)
    _append_closed(path, "native-earlier", 11.9, native.CAP_HOURS, 3, 1_791_000_000.0)
    before_refusal = path.read_bytes()
    with pytest.raises(native.BudgetExceeded, match="exceeds the.*cap"):
        with native.Reservation("over the additional allowance", projected_hours=0.1001, path=path, lock=lock):
            pass
    assert path.read_bytes() == before_refusal
    with native.Reservation("remaining allowance", projected_hours=0.1, path=path, lock=lock) as reservation:
        assert math.isclose(reservation.used_before, native.HISTORICAL_HOURS + 11.9)
        assert reservation.limit_hours == native.CAP_HOURS
        clock.advance(0.1 * 3600.0)
    report = native.usage(path)
    assert math.isclose(report["native_used_hours"], 12.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(report["native_remaining_hours"], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert len(report["reservations"]) == 4


@pytest.mark.parametrize(("failure", "status"), [
    (RuntimeError("load failed"), "failed"),
    (native.Terminated("external interruption"), "terminated"),
])
def test_failed_and_interrupted_reservations_keep_base_charging(tmp_path, clock, failure, status):
    path, lock = tmp_path / "gpu_ledger.jsonl", tmp_path / "gpu.lock"
    _historical(path)
    with pytest.raises(type(failure)):
        with native.Reservation("native attempt", projected_hours=0.1, path=path, lock=lock):
            clock.advance(90.0)
            raise failure
    report = native.usage(path)
    current = report["reservations"][-1]
    assert current["status"] == status and current["ended"]
    assert current["hours"] == 90.0 / 3600.0
    assert report["used_hours"] == scaling.usage(path)["used_hours"]
    assert math.isclose(report["native_used_hours"], current["hours"], rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(report["native_remaining_hours"], 12.0 - current["hours"], rel_tol=0.0, abs_tol=1e-14)
    end = [entry for entry in native.read(path) if entry["event"] == "end"][-1]
    assert end["status"] == status and type(failure).__name__ in end["error"]


def test_killed_process_is_charged_to_heartbeat_plus_one_interval(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    _historical(path)
    start = {"event": "start", "reservation": "killed-native", "t": 1_791_000_000.0, "elapsed": 0.0,
             "command": "killed native process", "projected_hours": 1.0, "cap_hours": native.CAP_HOURS,
             "authorization": 3, "native_authorization": dict(native.NATIVE_AUTHORIZATION)}
    heartbeat = {"event": "heartbeat", "reservation": "killed-native", "t": 1_791_000_012.5, "elapsed": 12.5}
    with path.open("a") as stream:
        stream.write(json.dumps(start, sort_keys=True) + "\n")
        stream.write(json.dumps(heartbeat, sort_keys=True) + "\n")
    report = native.usage(path)
    current = report["reservations"][-1]
    expected_hours = (12.5 + scaling.HEARTBEAT_SECONDS) / 3600.0
    assert current["open"] and not current["ended"]
    assert current["hours"] == expected_hours
    assert report["reservations"] == scaling.usage(path)["reservations"]
    assert math.isclose(report["native_used_hours"], expected_hours, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(report["native_remaining_hours"], 12.0 - expected_hours, rel_tol=0.0, abs_tol=1e-14)
