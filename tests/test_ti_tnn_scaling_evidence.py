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
