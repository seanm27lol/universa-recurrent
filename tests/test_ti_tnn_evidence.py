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
    assert len(manifest["files"]) == 288
    assert not any("/home/" in (BUNDLE / f["path"]).read_text() for f in manifest["files"])  # no local paths
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


def test_adaptive_compute_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")  # frontier.py imports the model module
    a = tables.adaptive_tables()  # raises unless every evaluated file is bundled and every verdict matches
    assert a["lm"]["complete"] and a["hops"]["complete"]
    assert a["lm"]["validated"] is False
    assert a["lm"]["predictions"] == {"efficiency": False, "equal_compute": False, "matched_cap": True, "latency": None}
    assert a["hops"]["predictions"] == {"efficiency": False, "equal_compute": False, "matched_cap": False}
    assert a["lm"]["variants"]["ti_tnn_adaptive_c30"]["quality"]["mean"] == pytest.approx(49.09, abs=0.005)


def test_init_control_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    c = tables.init_control_tables()  # raises unless it matches the recorded analysis
    assert c["complete"] and c["holds"] is False and c["latency"]["holds"] is None
    assert c["families"]["ti_tnn_adaptive_c30_initdense"]["holds"] is True
    assert c["families"]["moe_top1_initdense"]["holds"] is False


def test_lookup_learning_work_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    learning = tables.learning_tables()  # raises unless part A v2's verdict matches the recorded one
    assert len(learning["init_check"]) == 3 and len(learning["diagnostics"]) == 75
    v2 = learning["part_a_v2"]
    assert v2["inventory"]["complete"] and v2["validity"]["holds"] is False and v2["fixed_moe"]["holds"] is True
    assert v2["learning_fixes_work"] is False and v2["ti_tnn_passes"] is None
    assert v2["learned_by_variant"]["moe_top1_initdense_sh50_s18000"] == 5
    assert v2["learned_by_variant"]["dense_s18000"] == 3


def test_wikitext_fixed_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    wf = tables.wikitext_fixed_tables()
    assert wf["fixes_help_on_text"]["holds"] is False and wf["beats_plain_skipping"]["holds"] is False
    assert wf["candidate"]["quality"]["mean"] == pytest.approx(57.21, abs=0.005)


def test_typed_composition_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")  # typed_composition.py imports the learned modules
    typed = tables.typed_composition_tables()  # raises unless it matches the recorded verdict.json
    assert typed["complete"] and all(typed["predictions"].values())
    assert all(typed["families"][f]["competent_runs"] == 15 for f in ("specialists", "shared_dense", "moe"))
    assert typed["families"]["endpoint"]["same_endpoint_min"] == pytest.approx(6 / 19)
    assert typed["families"]["answer_only"]["intermediate_min"] is None


def test_semantic_interventions_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")  # semantic_interventions.py imports the learned modules
    sem = tables.semantic_interventions_tables()  # raises unless it matches the recorded verdict.json
    assert sem["complete"] and all(sem["predictions"].values())
    assert (sem["executions"], sem["interventions"]) == (83616, 387264)
    assert sem["families"]["endpoint"]["law_defects_max"] == 0 and sem["families"]["endpoint"]["competent"] == 0
    assert sem["families"]["hidden_dependency"]["undeclared_calls_max"] == 477
