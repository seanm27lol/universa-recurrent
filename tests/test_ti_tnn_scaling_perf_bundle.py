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
