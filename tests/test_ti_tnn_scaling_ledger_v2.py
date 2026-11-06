"""The GPU ledger under authorization v2 (2026-11-02): a 32-hour cumulative cap, with the history preserved.

- **The versions:** v1 (8 h) and v2 (32 h) are both recorded; v2 is in force; the frozen pilot's commands keep v1.
- **The history:** reservations made before the versioning (6.878 h, no `authorization` field) are read unchanged,
  attributed to v1 by their recorded cap, and count against the 32 h; new entries are appended, never rewritten.
- **One ledger across worktrees:** two git worktrees of one repository share the anchor in the git common
  directory, so a reservation from either sees the other's charges, cannot exceed the cap between them, and cannot
  hold the GPU while the other does. A worktree pointing at another runs directory is refused.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from scaling import contracts as C  # noqa: E402
from scaling import ledger, pilot_run  # noqa: E402

HISTORICAL_HOURS = 6.878


def _historical(path: Path, hours: float = HISTORICAL_HOURS, n: int = 4) -> list:
    """Ledger lines as the pilot wrote them: cap 8, no authorization field, ended reservations."""
    lines, t = [], 1_790_000_000.0  # September 2026, before any new entry
    for k in range(n):
        rid = f"2026103{k}T000000-old{k}"
        seconds = hours * 3600 / n
        lines.append({"event": "start", "reservation": rid, "t": t, "elapsed": 0.0, "command": f"pilot step {k}",
                      "projected_hours": 1.0, "cap_hours": 8.0, "limit_hours": 8.0})
        lines.append({"event": "end", "reservation": rid, "t": t + seconds, "elapsed": seconds, "status": "ok"})
        t += seconds + 60
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    return lines


def test_v2_is_in_force_and_v1_is_kept():
    versions = [a["version"] for a in ledger.AUTHORIZATIONS]
    assert versions == [1, 2]
    assert ledger.AUTHORIZATION["version"] == 2 and ledger.CAP_HOURS == 32.0
    assert ledger.cap_of(1) == 8.0 and ledger.cap_of(2) == 32.0
    assert ledger.AUTHORIZATIONS[1]["supersedes"] == 1
    with pytest.raises(KeyError):
        ledger.cap_of(3)
    # the frozen pilot is not extended by v2
    assert pilot_run.PILOT_CAP_HOURS == 8.0


def test_historical_entries_count_against_the_cumulative_cap_and_are_never_rewritten(tmp_path):
    path, lock = tmp_path / "gpu_ledger.jsonl", tmp_path / "gpu.lock"
    _historical(path)
    before = path.read_bytes()
    u = ledger.usage(path)
    assert math.isclose(u["used_hours"], HISTORICAL_HOURS)
    assert u["cap_hours"] == 32.0 and u["authorization"] == 2
    assert math.isclose(u["remaining_hours"], 32.0 - HISTORICAL_HOURS)
    assert {r["authorization"] for r in u["reservations"]} == {1}  # inferred from the recorded 8 h cap
    with ledger.Reservation("v2 work", projected_hours=0.01, path=path, lock=lock) as r:
        assert math.isclose(r.used_before, HISTORICAL_HOURS)
        assert math.isclose(r.limit_hours, HISTORICAL_HOURS + 0.015)
    after = path.read_bytes()
    assert after.startswith(before)  # append-only: the history is byte-identical
    new = [json.loads(x) for x in after[len(before):].decode().splitlines()]
    assert new[0]["event"] == "start" and new[0]["authorization"] == 2 and new[0]["cap_hours"] == 32.0
    assert ledger.usage(path)["reservations"][-1]["authorization"] == 2


def test_the_new_cap_cannot_be_exceeded(tmp_path):
    path, lock = tmp_path / "gpu_ledger.jsonl", tmp_path / "gpu.lock"
    _historical(path, hours=31.9)
    with pytest.raises(ledger.BudgetExceeded, match="32"):
        with ledger.Reservation("too big", projected_hours=0.2, path=path, lock=lock):
            pass
    with ledger.Reservation("fits", projected_hours=0.05, path=path, lock=lock) as r:
        assert r.limit_hours <= ledger.CAP_HOURS  # min(cap, used + 1.5 x projected)
        assert math.isclose(r.limit_hours, 31.975)
    _historical(path, hours=31.99)
    with ledger.Reservation("at the edge", projected_hours=0.009, path=path, lock=lock) as r:
        assert r.limit_hours == ledger.CAP_HOURS  # the margin never lifts the limit past the cap


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def two_worktrees(tmp_path, monkeypatch):
    """A repository with a second worktree; C.REPO is switched between them as each 'process' would see it."""
    main, other = tmp_path / "repo", tmp_path / "repo-run"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init",
         cwd=main)
    _git("worktree", "add", "-q", "--detach", str(other), cwd=main)
    runs = tmp_path / "shared_runs"
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", str(runs))

    def use(worktree: Path):
        monkeypatch.setattr(C, "REPO", worktree)
    return main, other, runs, use


def test_reservations_from_two_worktrees_share_one_ledger_and_one_cap(two_worktrees):
    main, other, runs, use = two_worktrees
    use(main)
    anchor_main = ledger.anchor_file()
    use(other)
    assert ledger.anchor_file() == anchor_main  # the anchor lives in the shared git common directory
    _historical(runs / "gpu_ledger.jsonl", hours=31.0)
    use(main)
    with ledger.Reservation("from the edit worktree", projected_hours=0.4) as r:
        assert r.path == runs / "gpu_ledger.jsonl"
    assert anchor_main.read_text().strip() == str(runs)
    used = ledger.usage(runs / "gpu_ledger.jsonl")["used_hours"]
    use(other)
    with pytest.raises(ledger.BudgetExceeded):  # the other worktree sees the same 31.0 h + the first reservation
        with ledger.Reservation("from the run worktree", projected_hours=32.0 - used + 0.01):
            pass
    with ledger.Reservation("from the run worktree", projected_hours=0.1) as r:
        assert math.isclose(r.used_before, used) and r.path == runs / "gpu_ledger.jsonl"
    starts = [e for e in ledger.read(runs / "gpu_ledger.jsonl") if e["event"] == "start"]
    assert [e["command"] for e in starts[-2:]] == ["from the edit worktree", "from the run worktree"]
    assert all(e["anchor"] == str(anchor_main) and e["authorization"] == 2 for e in starts[-2:])


def test_only_one_worktree_holds_the_gpu_at_a_time(two_worktrees):
    main, other, runs, use = two_worktrees
    use(main)
    with ledger.Reservation("owner", projected_hours=0.1):
        use(other)
        with pytest.raises(RuntimeError, match="holds the GPU"):
            with ledger.Reservation("second owner", projected_hours=0.1):
                pass
    starts = [e for e in ledger.read(runs / "gpu_ledger.jsonl") if e["event"] == "start"]
    assert [e["command"] for e in starts] == ["owner"]  # the refused one never started


def test_a_worktree_with_another_runs_directory_is_refused(two_worktrees, monkeypatch, tmp_path):
    main, other, runs, use = two_worktrees
    use(main)
    with ledger.Reservation("anchors it", projected_hours=0.01):
        pass
    use(other)
    elsewhere = tmp_path / "private_runs"
    elsewhere.mkdir()
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", str(elsewhere))
    with pytest.raises(RuntimeError, match="anchored"):  # a private ledger would split the cap
        ledger.Reservation("private ledger", projected_hours=0.01)
