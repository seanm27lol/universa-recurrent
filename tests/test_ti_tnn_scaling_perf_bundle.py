"""The performance study's evidence bundle (research/ti_tnn/scaling/evidence/perf_v2): the report's tables rebuild
from the bundles alone (no runs/ directory, no GPU), and a score that disagrees with the analysis is refused even
when the sums are rewritten to match."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
from scaling import evidence  # noqa: E402

ROWS = ("| T4-v2 | 10 | 16 | 10 | 36 of 192 |", "| C4-v2 | 12 | 11 | 12 | 35 of 192 |",
        "| T4: v2 - v1 | +4 | +9 | +2 | +15 | +11 .. +15 | I1: material improvement | 0.008 |",
        "| C4: v2 - v1 | +8 | +4 | +6 | +18 | +14 .. +19 | I1: material improvement | 0.010 |",
        "| v2: T4 - C4 | -2 | +5 | -2 | +1 | +0 .. +2 | A3: no consistent difference | 1.000 |",
        "| B4 | — | — | — | 41 of 64 (one run) |", "| Q27 | — | — | — | 50 of 64 (one run) |",
        "| B4 | — | — | — | 40 of 64 (one run) |", "| Q27 | — | — | — | 49 of 64 (one run) |")


def _perf_report():
    if not (evidence.PERF_BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no perf bundle")
    from scaling import perf_report
    return perf_report


def test_the_perf_bundle_rebuilds_the_report():
    text = _perf_report().tables()
    assert text.startswith("Recomputed from the bundled records and equal to analysis.json")
    for row in ROWS:
        assert row in text, row
    questions = _perf_report().questions()
    assert questions.count("| bird23tf-") == 128


def test_a_score_that_disagrees_with_the_analysis_is_refused(tmp_path, monkeypatch):
    perf_report = _perf_report()
    copy = tmp_path / "perf_v2"
    shutil.copytree(evidence.PERF_BUNDLE, copy)
    (records,) = sorted(copy.glob("eval/s100_*/T4-v2/records.jsonl"))
    lines = records.read_text().splitlines()
    first = json.loads(lines[0])
    first["score"]["correct"] = 1 - first["score"]["correct"]
    records.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    evidence.write_sums(copy)  # the sums match the tampered file, so only the recomputation can catch it
    monkeypatch.setattr(perf_report, "BUNDLE", copy)
    with pytest.raises(perf_report.Inconsistent, match="counts"):
        perf_report.tables()


def test_the_corrections_rebuild_from_the_bundle():
    pytest.importorskip("torch")  # the MAC formulas live in the controller module
    text = _perf_report().corrections()
    for row in ("| new | T4-v2 | 192 | 7.015 G | 7.057 G | +0.60% |",
                "| new | T4-v2 | 192 | 0.56 | 0.29 | 0.16 | 0.56 / 0.28 / 0.16 |",
                "| standalone setup | the training pool's backbone states, completed encode reservations (4) | 1.936 |",
                "| research | everything under authorization v2 (21 reservations) | 14.538 |"):
        assert row in text, row
    assert text.count("| B4 thinking, positions") == 2 and text.count("| Q27 thinking (low), positions") == 2


def test_executed_macs_refuse_a_record_they_cannot_reconstruct():
    pytest.importorskip("torch")
    _perf_report()
    from scaling import controller as K
    from scaling import perf_corrections as PC
    (records,) = sorted(evidence.PERF_BUNDLE.glob("eval/s100_*/T4-v2/records.jsonl"))
    record = json.loads(records.read_text().splitlines()[0])
    cfg = K.ControllerConfig(**json.loads((evidence.PERF_BUNDLE / "train" / "T4-v2_s100" / "config.json")
                                          .read_text())["controller"])
    m = PC.executed_macs(record, cfg)
    assert m["executed"] > m["recorded"] and m["pointer_recorded"] == record["controller_work"]["macs"]["pointer"]
    record["controller_work"]["macs"]["pointer"] += 1
    with pytest.raises(PC.MacMismatch, match="pointer"):
        PC.executed_macs(record, cfg)
