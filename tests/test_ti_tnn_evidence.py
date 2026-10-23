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
    assert len(manifest["files"]) == 516
    assert not any("/home/" in (BUNDLE / f["path"]).read_text() for f in manifest["files"]
                   if not f["path"].endswith(".gz"))  # no local paths
    import gzip
    assert not any("/home/" in gzip.open(BUNDLE / f["path"], "rt").read() for f in manifest["files"]
                   if f["path"].endswith(".gz"))
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


def test_typed_routing_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")  # typed_questions.py imports the interpreter
    routing = tables.typed_routing_tables()  # raises unless it matches the recorded verdict.json
    p = routing["predictions"]
    assert routing["complete"] and p["6_shift_accepted_error_bounded"] is False
    assert all(v for k, v in p.items() if not k.startswith("6_"))
    t = routing["systems"]["typed_specialists"]
    assert (t["test"]["accepted_errors_total"], t["test"]["answered_total"]) == (2, 13714)
    assert (t["shift"]["accepted_errors_total"], t["shift"]["answered_total"]) == (59, 100)


def test_typed_recurrence_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    recur = tables.typed_recurrence_tables()  # raises unless it matches the recorded verdict.json
    assert recur["complete"] and all(recur["predictions"].values())
    assert (recur["queries"], recur["pause_points"]) == (15480, 53970)
    assert recur["systems"]["recurrent"]["accuracy"] == [1.0, 1.0]


def test_local_updates_verdict_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    upd = tables.local_updates_tables()  # raises unless it matches the recorded verdict.json
    assert upd["complete"] and all(upd["predictions"].values())
    assert upd["families"]["specialists"]["activated"] == 150
    assert upd["families"]["shared_same_data"]["unaffected_changed_from_base_total"] == 4387
    assert upd["families"]["shared_same_data"]["atomic_changed_by_relation"] == {"largest_city_of": 373}


def test_the_post_repair_rerun_reproduces_the_module_replacement_verdict():
    pytest.importorskip("torch")
    assert tables.local_updates_post_repair_tables() == tables.local_updates_tables()  # raises on any difference

def test_shared_path_control_rebuilds_from_the_bundle():
    pytest.importorskip("torch")
    sc = tables.shared_control_tables()  # raises unless it matches the recorded output
    assert sc["complete"]
    assert sc["learned"] == {"dense_d128_l8_m2_s18000": 4, "ti_tnn_adaptive_c30_initdense_warm50_kv_sh50_e1_s18000": 1}
    assert sc["reference_counts"]["moe_top1_initdense_sh50_s18000"] == 5
    assert sc["shared_path_insufficient"]["holds"] and sc["routing_helps_ti_tnn"]["holds"]


def test_interpreter_robustness_unit_rebuilds_from_its_records():
    pytest.importorskip("torch")
    out = tables.interpreter_robustness_tables(quick=True)  # one unit re-derived; raises on any discrepancy
    assert out["verified"] and out["partial"]
    import json
    verdict = json.loads((BUNDLE / "interpreter_robustness/20261018T232057/verdict.json").read_text())
    assert verdict["predictions"] == {"1_sanity_every_arm": True, "2_taught_wording_learned_every_arm": True,
                                      "3_word_aug_joint_novelty": True}
    assert verdict["successes_by_world"]["joint_novelty"]["word_aug"] == {"700": 5, "701": 5, "702": 5}
    assert not any(verdict["separate_verdicts"]["s2_transfer"].values())
    assert not any(verdict["separate_verdicts"]["s1_structure"].values())


def test_s2_diagnosis_development_rebuilds_from_its_records():
    pytest.importorskip("torch")
    out = tables.s2_diagnosis_tables(quick=True)  # development part; raises on any discrepancy
    dev = out["development"]
    assert dev["decision_rule"]["proceed_to_confirmation"]
    assert {w: m["C"] > m["B"] for w, m in dev["decision_rule"]["S2_exact_world_means"].items()} == {"100": True,
                                                                                                    "101": True}
    assert all(dev["units"][f"D/{w}/{i}"]["S2_exact"] == 1.0 for w in (100, 101) for i in (100, 101, 102))
    import json
    conf = json.loads((BUNDLE / "s2_diagnosis/confirmation_analysis/s2_development_analysis.json").read_text())
    assert conf["confirmation"]["complete"] and not conf["confirmation"]["prediction_C_holds"]
    assert not conf["confirmation"]["secondary_C_above_B_every_world"]


def test_s2_trace_release_is_bound_to_the_committed_hashes(tmp_path):
    pytest.importorskip("torch")
    import json
    release = json.loads((BUNDLE / "s2_diagnosis/trace_release.json").read_text())
    committed = {f["original_path"]: f for f in json.loads((BUNDLE / "s2_diagnosis/local_traces.json").read_text())["files"]}
    assert len(release["assets"]) == len(committed) == 63
    assert all(committed[a["original_path"]]["sha256"] == a["sha256"] for a in release["assets"])
    assert len({a["name"] for a in release["assets"]}) == 63
    for a in release["assets"]:  # a downloaded set whose bytes differ from the committed hashes is refused
        (tmp_path / a["name"]).write_bytes(b"not the trace")
    with pytest.raises(ValueError, match="sha256"):
        tables.s2_diagnosis_tables(quick=True, traces_dir=tmp_path)


def test_the_integration_test_bundle_is_bound_to_its_session_release(tmp_path):
    import json
    root = BUNDLE / "integration_test"
    release = json.loads((root / "session_release.json").read_text())
    assert len(release["assets"]) == 15 and len({a["name"] for a in release["assets"]}) == 15
    verification = json.loads((root / "verification.json").read_text())
    assert verification["verified"] and verification["commit"].startswith("e03075a")
    assert verification["verdict"]["construction_every_unit"] and not verification["verdict"]["construction_not_exercised"]
    assert all(v == {"800": 5, "801": 5, "802": 5} for v in verification["verdict"]["learned_successes_by_world"].values())
    for a in release["assets"]:  # a downloaded set whose bytes differ from the committed hashes is refused
        (tmp_path / a["name"]).write_bytes(b"not the sessions")
    with pytest.raises(ValueError, match="sha256"):
        tables.integration_test_tables(sessions_dir=tmp_path)
