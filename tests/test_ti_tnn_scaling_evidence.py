"""The evidence bundles (research/ti_tnn/scaling/evidence/dev_v1 and repair_v1): every file matches SHA256SUMS, and the
report's headline numbers rebuild from the bundle alone (no runs/ directory, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
from scaling import evidence  # noqa: E402


def test_the_bundle_matches_its_sums_and_rebuilds_the_headline_numbers():
    if not (evidence.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no evidence bundle")
    text = evidence.tables()
    for row in ("| L | nonthinking | 0:16 | 1/16 |", "| B4 | thinking | 0:16 | 11/16 |",
                "| Q27 | thinking (low) | 0:16 | 13/16 |", "| T4 | 0/16 |", "| C4_params | 0/16 |",
                "2.861 of 8"):
        assert row in text, row


def test_a_changed_byte_is_refused(tmp_path, monkeypatch):
    if not (evidence.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no evidence bundle")
    import shutil
    copy = tmp_path / "dev_v1"
    shutil.copytree(evidence.BUNDLE, copy)
    target = copy / "gpu_ledger.jsonl"
    target.write_bytes(target.read_bytes().replace(b'"ok"', b'"OK"', 1))
    monkeypatch.setattr(evidence, "BUNDLE", copy)
    with pytest.raises(SystemExit, match="gpu_ledger.jsonl"):
        evidence.tables()


def test_the_repair_bundle_rebuilds_the_timing_table():
    if not (evidence.REPAIR_BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no repair bundle")
    text = evidence.timing_table()
    assert "| file | 8 | high | 12 | 0.516 |" in text and "| file | 8 | highest | 20 | 5.343 |" in text


PILOT_ROWS = ("| T4 | 9 | 9 | 18 |", "| C4_params | 4 | 10 | 14 |", "| L (deterministic, no fit) | 4 | 4 | 8 |",
              "**A3: no pilot difference by the pre-set rule**", "two-sided p = 0.3594", "| representable | 58 |",
              "6.878 of 8 (pilot: 3.899)",
              # time to SQL submission (total / mean) and generation cost per EX-correct submission, the frozen
              # metric; then the scorer's recorded execution of the submitted queries, descriptive
              "| 100 | T4 | 9 | 64 | 0 | 86.2 / 1.35 | 9.6 | 36.0 (1) | 13.6 |",
              "| 101 | C4_params | 10 | 64 | 0 | 85.2 / 1.33 | 8.5 | 94.1 (3) | 17.9 |",
              "| — | L | 4 | 64 | 0 | 192.3 / 3.00 | 48.1 | 4.5 (0) | 49.2 |")


def _pilot_bundle():
    if not (evidence.PILOT_BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no pilot bundle")
    from scaling import pilot_report
    return pilot_report


def test_the_pilot_bundle_rebuilds_the_pilot_report():
    text = _pilot_bundle().tables()
    for row in PILOT_ROWS:
        assert row in text, row
    assert text.startswith("Recomputed from the decision records and equal to analysis.json")


def test_the_pilot_report_rebuilds_from_a_relocated_bundle(tmp_path):
    """Runs are matched by their reference relative to the runs root, so a bundle moved anywhere (here, with a
    different absolute path and no runs/ directory) rebuilds the same tables."""
    pilot_report = _pilot_bundle()
    import shutil
    moved = tmp_path / "elsewhere" / "copy_of_pilot_v1"
    shutil.copytree(evidence.PILOT_BUNDLE, moved)
    assert pilot_report.tables(moved) == pilot_report.tables()


def _tampered(tmp_path, edit):
    """A copy of the pilot bundle with one file edited and SHA256SUMS regenerated, so only the consistency check
    between the records and analysis.json or the summaries can catch it."""
    import json
    import shutil
    copy = tmp_path / "pilot_v1"
    shutil.copytree(evidence.PILOT_BUNDLE, copy)
    edit(copy, json)
    evidence.write_sums(copy)
    evidence._checked(copy)  # the checksums pass
    return copy


def test_an_analysis_that_disagrees_with_its_records_is_refused(tmp_path):
    pilot_report = _pilot_bundle()

    def counts(copy, json):  # analysis.json claims three more T4 successes in seed 101 (pooled +7) and rule A1
        path = next(copy.glob("analysis/*/analysis.json"))
        a = json.loads(path.read_text())
        a["counts"]["T4/101"] += 3
        a["rule"] = "A1: T4 ahead in this pilot"
        path.write_text(json.dumps(a))
    with pytest.raises(SystemExit, match="disagrees with the decision records(.|\n)*counts(.|\n)*rule"):
        pilot_report.tables(_tampered(tmp_path / "a", counts))

    def blocks(copy, json):  # only the block statistic differs
        path = next(copy.glob("analysis/*/analysis.json"))
        a = json.loads(path.read_text())
        a["exploratory_sign_flip"]["p_two_sided"] = 0.01
        path.write_text(json.dumps(a))
    with pytest.raises(SystemExit, match="exploratory_sign_flip"):
        pilot_report.tables(_tampered(tmp_path / "b", blocks))


def test_a_record_that_disagrees_with_its_summary_is_refused(tmp_path):
    pilot_report = _pilot_bundle()

    def score(copy, json):  # one decision record's EX score flipped
        path = next(copy.glob("eval/s100_*/c4_params/records.jsonl"))
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        recs[0]["score"]["correct"] = 1 - recs[0]["score"]["correct"]
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match="c4_params/100: correct recorded 4"):
        pilot_report.tables(_tampered(tmp_path, score))


def test_a_false_reconstruction_entry_makes_an_evaluation_ineligible(tmp_path):
    """The summary still says passed and every aggregate flag is true, but one request's reconstruction differs."""
    pilot_report = _pilot_bundle()

    def entry(copy, json):
        path = next(copy.glob("eval/s101_*/summary.json"))
        s = json.loads(path.read_text())
        s["arms"]["T4"]["reconstruction"][5]["same_final"] = False
        assert s["passed"] is True and s["arms"]["T4"]["reconstruction_all_equal"] is True
        path.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="s101_.*records passed, but T4: 1 reconstruction entries are not true"):
        pilot_report.tables(_tampered(tmp_path, entry))
