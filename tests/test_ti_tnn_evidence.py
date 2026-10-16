"""The TI-TNN evidence bundle (experiments/results/ti_tnn) rebuilds the numbers the reports quote."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "experiments" / "results" / "ti_tnn"
spec = importlib.util.spec_from_file_location("ti_tnn_make_tables", BUNDLE / "make_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_every_file_matches_the_manifest_and_none_is_unlisted():
    manifest = tables.verify()
    assert len(manifest["files"]) == 81
    assert {f["experiment"] for f in manifest["files"]} == set(manifest["experiments"])
    assert not any(p.suffix in (".pt", ".bin", ".safetensors") for p in BUNDLE.rglob("*"))  # weight-free


def test_composition_counts_match_the_reports():
    comp = tables.composition_tables()
    first, controls = comp["functorial_composition"], comp["functorial_composition_controls"]
    assert first["functorial"]["two_hop_correct_total"] == (2560, 2560)
    assert first["linear"]["two_hop_correct_total"] == (2406, 2560)
    assert first["transformer"]["two_hop_correct_total"] == (425, 2560)
    assert controls["shared"]["two_hop_correct_total"] == (2559, 2560)  # one error, not two
    assert controls["ortho_untied"]["two_hop_correct_total"] == (2560, 2560)
    assert controls["soft"]["two_hop_correct_total"] == first["soft"]["two_hop_correct_total"] == (2251, 2560)


def test_toy_results_match_the_reports():
    toys = tables.toy_tables()
    assert toys["compositional_heldout_xor"]["routed_bridge"][0] == 1.0
    assert toys["compositional_heldout_xor"]["routed_nobridge"][0] == pytest.approx(0.492, abs=5e-4)
    assert toys["unsupervised"]["unsup_bridge"]["bridge_fire_rate"][0] == 0.0


def test_real_benchmarks_rebuild_the_bundled_summary():
    pytest.importorskip("torch")  # summarize.py imports the model module
    real = tables.real_benchmark_tables()  # raises if the rebuilt summary differs from summary.json
    assert real["lm"]["moe_top2"]["validation_perplexity"]["mean"] == pytest.approx(37.92, abs=0.005)
