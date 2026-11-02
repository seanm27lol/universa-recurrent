"""The independent pilot verifier's public arithmetic reconstruction (research/ti_tnn/scaling/independent/
verify_pilot.py) on the committed evidence bundle: no runs root, database, checkpoint or cached state.

**What it checks:**
- the bundle reproduces the verdict, in place and after the bundle is moved;
- a false reconstruction entry rejects its evaluation even though the producer's summary still says `passed`;
- a producer total that disagrees with its records is a failed check.

It needs torch (the seeded example stream) and the git history holding the freeze and run commits (`git show`);
without either it skips, as in a shallow CI checkout."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
sys.path.insert(0, str(REPO / "research" / "ti_tnn" / "scaling" / "independent"))
from scaling import evidence  # noqa: E402

pytest.importorskip("torch")
BUNDLE = evidence.PILOT_BUNDLE


def _verifier():
    if not (BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no pilot bundle")
    pin = json.loads((REPO / "research/ti_tnn/scaling/pins/pilot_protocol_v1.json").read_text())
    commit = json.loads(next(BUNDLE.glob("eval/s100_*/manifest.json")).read_text())["git"]["commit"]
    for c in (pin["frozen_at"], commit):
        if subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
            pytest.skip(f"commit {c[:8]} is not in this checkout's history (shallow clone)")
    import verify_pilot
    return verify_pilot


def _public(vp, bundle):
    out, rep, _ = vp.public_reconstruction(bundle, REPO)
    return out, rep


def test_the_public_reconstruction_reproduces_the_verdict_in_place_and_moved(tmp_path):
    vp = _verifier()
    out, rep = _public(vp, BUNDLE)
    assert not rep.fails, rep.fails[:5]
    r = out["result"]
    assert r["rule"] == "A3" and r["cleared_L_in_both_seeds"] == {"T4": True, "c4_params": False}
    assert r["counts"] == {"T4/100": 9, "T4/101": 9, "c4_params/100": 4, "c4_params/101": 10, "L": 4, "questions": 64}
    assert all(out["agreement_with_analysis_json"].values())
    assert round(out["serving"]["L"]["time_to_submission_mean"], 2) == 3.00
    moved = tmp_path / "another" / "place" / "pilot_v1"
    shutil.copytree(BUNDLE, moved)
    out2, rep2 = _public(vp, moved)
    assert not rep2.fails and out2["result"] == r and out2["evaluations"] == out["evaluations"]


def _tampered(tmp_path, edit):
    copy = tmp_path / "pilot_v1"
    shutil.copytree(BUNDLE, copy)
    edit(copy)
    evidence.write_sums(copy)  # the checksums pass; only the reconstruction can catch the edit
    return copy


def test_a_false_reconstruction_entry_rejects_the_evaluation_even_if_passed_stays_true(tmp_path):
    vp = _verifier()

    def edit(copy):
        path = next(copy.glob("eval/s100_*/summary.json"))
        s = json.loads(path.read_text())
        s["arms"]["c4_params"]["reconstruction"][7]["fresh_states_equal_anchor"] = False
        path.write_text(json.dumps(s))  # passed and every aggregate flag stay true
    out, rep = _public(vp, _tampered(tmp_path, edit))
    run = next(r for r in out["evaluations"] if r["seed"] == 100)
    assert run["eligible"] is False and run["producer_passed"] is True
    assert any("1 false inventory entries" in p for p in run["problems"])
    assert any("passed flag agrees with the inventory" in f for f in rep.fails)
    assert any("seed 100: exactly one eligible evaluation" in f for f in rep.fails)


def test_a_producer_total_that_disagrees_with_its_records_fails(tmp_path):
    vp = _verifier()

    def edit(copy):
        path = next(copy.glob("eval/s101_*/summary.json"))
        s = json.loads(path.read_text())
        s["arms"]["T4"]["summary"]["cost_seconds_per_correct"] = 5.0
        path.write_text(json.dumps(s))
    _out, rep = _public(vp, _tampered(tmp_path, edit))
    assert any("T4: cost_seconds_per_correct recomputed = producer summary" in f for f in rep.fails)
