import json

import numpy as np
import pytest

from open_weight_lingua import state_nonlinear as sn
from open_weight_lingua import state_trace_survey as ts


def _sign_flipped(n, seed, d=16):
    """Each value v is coded as +c_v or -c_v with equal counts: its class mean is zero.

    A linear readout of v has nothing to use; a nonlinear one can read |c_v|.
    The same shape as Othello-GPT's board, which linear probes missed until
    a "mine/theirs" basis was found (Li et al. 2023; Nanda et al. 2023).
    """
    rng = np.random.default_rng(seed)
    code = np.random.default_rng(99).normal(size=(sn.VALUES, d))
    values = rng.integers(0, sn.VALUES, n)
    x = np.concatenate([code[values], -code[values]]) + 0.05 * rng.normal(size=(2 * n, d))
    return x, np.concatenate([values, values])


def test_mlp_reads_a_sign_flipped_code_that_a_linear_probe_cannot():
    train_x, train_y = _sign_flipped(250, 0)
    select_x, select_y = _sign_flipped(75, 1)
    test_x, test_y = _sign_flipped(75, 2)
    fit = sn.mlp_probe(train_x, train_y, select_x, select_y, test_x, device="cpu", seed=1)
    (linear,) = ts.fit_predict(train_x, train_y, test_x, (0.01,), "cpu")
    assert np.mean(fit["evaluate_pred"] == test_y) > 0.95
    assert np.mean(linear == test_y) < 0.2
    assert fit["weight_decay"] in sn.WEIGHT_DECAYS and fit["epoch"] % sn.EVAL_EVERY == 0


def test_permuted_control_uses_the_chosen_hyperparameters_and_stays_at_chance():
    train_x, train_y = _sign_flipped(250, 0)
    select_x, select_y = _sign_flipped(75, 1)
    test_x, test_y = _sign_flipped(75, 2)
    shuffled = np.random.default_rng(sn.PERMUTATION_SEED).permutation(train_y)
    fit = sn.mlp_probe(train_x, shuffled, select_x, select_y, test_x, device="cpu", seed=2, choice=(0.01, 120))
    assert (fit["weight_decay"], fit["epoch"]) == (0.01, 120)
    assert np.mean(fit["evaluate_pred"] == test_y) <= sn.CONTROL_MAX


def test_fits_are_reproducible_and_seeded_by_identity():
    x, y = _sign_flipped(60, 3)
    first = sn.mlp_probe(x[:80], y[:80], x[80:100], y[80:100], x[100:], device="cpu", seed=sn.fit_seed("A", "answer", 3, "x"))
    second = sn.mlp_probe(x[:80], y[:80], x[80:100], y[80:100], x[100:], device="cpu", seed=sn.fit_seed("A", "answer", 3, "x"))
    assert (first["evaluate_pred"] == second["evaluate_pred"]).all() and first["epoch"] == second["epoch"]
    assert sn.fit_seed("A", "answer", 3, "x") != sn.fit_seed("A", "answer", 3, "y")


def test_selection_ties_go_to_the_larger_decay_then_the_earlier_epoch():
    zeros, labels = np.zeros((40, 8)), np.full(40, 3)
    fit = sn.mlp_probe(zeros[:30], labels[:30], zeros[30:35], labels[30:35], zeros[35:], device="cpu", seed=5)
    assert fit["select_accuracy"] == 1.0  # every checkpoint of both decays ties
    assert (fit["weight_decay"], fit["epoch"]) == (max(sn.WEIGHT_DECAYS), sn.EVAL_EVERY)


def _final_state_rows(n_per_part):
    rows, rng = [], np.random.default_rng(7)
    for part, n in zip(("train", "select", "test"), n_per_part):
        for i in range(n):
            x, y = (int(v) for v in rng.integers(0, sn.VALUES, 2))
            rows.append({"id": f"{part}-{i:04d}-{'xy'[i % 2]}", "part": part,
                         "labels": {"x": {"value": x, "computed": i % 3 == 0}, "y": {"value": y, "computed": i % 3 == 1}}})
    return rows


def test_part_a_finds_a_nonlinear_site_and_applies_gate_g5a():
    rows = _final_state_rows((500, 100, 100))
    code = np.random.default_rng(99).normal(size=(sn.VALUES, 8))
    signs = np.random.default_rng(8).choice([-1.0, 1.0], size=(len(rows), 1))
    noise = np.random.default_rng(9).normal(size=(len(rows), 2, 16))

    def load(position):
        features = noise.copy()
        if position == "question_mark":  # both values, sign-flipped, in block 1 only
            x = np.array([r["labels"]["x"]["value"] for r in rows])
            y = np.array([r["labels"]["y"]["value"] for r in rows])
            features[:, 1] = np.concatenate([signs * code[x], signs * code[y]], axis=1)
        return features

    results, summary = sn.part_a(rows, load, "cpu")
    site = summary["selected_site"]
    assert (site["position"], site["layer"]) == ("question_mark", 1)
    assert summary["gate_G5a"]["passed"] is True
    assert max(site["control_test"].values()) <= sn.CONTROL_MAX
    assert len(results) == len(sn.POSITIONS) * 2
    assert {"test_asked", "test_not_asked"} <= set(results[0]["x"]["onehot"])


def _boundary_records(n_per_part):
    records, rng = [], np.random.default_rng(11)
    categories = ("literal", "arithmetic", "copy", "carried")
    for part, n in zip(("train", "select", "test"), n_per_part):
        for i in range(n):
            records.append({"part": part, **{
                v: {"value": int(rng.integers(0, sn.VALUES)), "category": categories[(i + k) % 4],
                    "computed": bool((i // 4) % 2), "copy_guess": 0}
                for k, v in enumerate(("x", "y"))}})
    return records


def test_part_b_readings_need_the_control_as_well_as_the_probe(monkeypatch):
    records = _boundary_records((400, 120, 120))
    code = np.random.default_rng(99).normal(size=(sn.VALUES, 8))
    signs = np.random.default_rng(12).choice([-1.0, 1.0], size=(len(records), 1))
    features = np.random.default_rng(13).normal(size=(len(records), 3, 16))
    x = np.array([r["x"]["value"] for r in records])
    y = np.array([r["y"]["value"] for r in records])
    features[:, 2] = np.concatenate([signs * code[x], signs * code[y]], axis=1)
    results, summary = sn.part_b(features, records, "cpu")
    assert len(results) == 3 and summary["R1"]["layer"] == summary["R2"]["layer"] == 2
    assert summary["gate_G5b"]["passed"] is True
    monkeypatch.setattr(sn, "CONTROL_MAX", -1.0)  # a control that "reads" disqualifies the reading
    _, strict = sn.part_b(features, records, "cpu")
    assert strict["R1"]["probe_supported"] is True and strict["R1"]["supported"] is False
    assert strict["gate_G5b"]["passed"] is False


def _fixture_inputs(tmp_path, monkeypatch, model_factory, tokenizer):
    """Phase Three and Phase Four fixture runs on a tiny random model, as their own tests build them."""
    from test_pilot_stages import TINY_COUNTS, _inputs, _stub_load_model
    import transformers

    from open_weight_lingua import state_survey as ss
    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": ss.sha256_file(lock)}))
    boundaries = lambda tok, ids, prompt: [min(i, len(ids) - 1) for i in range(len(ts.trace_labels(prompt)))]  # noqa: E731
    for module in (ss, ts, sn):
        monkeypatch.setattr(module, "load_model", _stub_load_model(model_factory))
        monkeypatch.setattr(module, "read_lock", lambda path: json.loads(path.read_text()))
        monkeypatch.setattr(module, "model_paths", lambda lock, cache: {"target": tmp_path})
        monkeypatch.setattr(module, "verify_models", lambda lock, paths: {})
        monkeypatch.setattr(module, "compatibility", lambda device: {})
    monkeypatch.setattr(ss, "TRAIN_GROUPS", 2)
    monkeypatch.setattr(ts, "TRAIN_GROUPS", 2)
    monkeypatch.setattr(ss, "positions", lambda tok, row: {"program_end": 0, "query_variable": 1, "question_mark": 2, "user_end": 2, "answer": 2})
    monkeypatch.setattr(ts, "boundary_positions", boundaries)
    monkeypatch.setattr(sn, "boundary_positions", boundaries)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: tokenizer)
    common = ["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"), "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"]
    survey, trace = RunDirectory(tmp_path / "survey"), RunDirectory(tmp_path / "trace")
    ss.run(ss.parse_args(common), survey, {})
    ts.run(ts.parse_args(common), trace, {})
    return common, survey.path, trace.path


def test_phase_five_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model, fixture tokens; the plumbing only, never a real-model result."""
    from open_weight_lingua.artifacts import RunDirectory

    common, survey, trace = _fixture_inputs(tmp_path, monkeypatch, model_factory, tokenizer)
    monkeypatch.setattr(sn, "EPOCHS", 20)
    args = sn.parse_args(common + ["--survey-run", str(survey), "--trace-run", str(trace)])
    run = RunDirectory(tmp_path / "nonlinear")
    results, summary = sn.run(args, run, {})
    assert summary["part_a"]["rows"] == {"train": 8, "select": 4, "test": 16}
    assert len(results["part_a"]) == len(sn.POSITIONS) * summary["layers"]
    assert len(results["part_b"]) == summary["layers"]
    assert set(summary["gate_G5"]) == {"passed"}
    assert set(summary["part_b"]) >= {"R1", "R2", "gate_G5b", "baselines"}
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["inputs"]["phase_four_run"] == "trace" and manifest["boundaries"] == sum(summary["part_b"]["rows"].values())
    assert (run.path / "raw" / "boundary_features.safetensors").is_file()


def test_recorded_boundaries_must_match_the_tokenizer(tmp_path, monkeypatch, model_factory, tokenizer):
    from open_weight_lingua.artifacts import RunDirectory

    common, survey, trace = _fixture_inputs(tmp_path, monkeypatch, model_factory, tokenizer)
    monkeypatch.setattr(sn, "boundary_positions", lambda tok, ids, prompt: [0] * len(ts.trace_labels(prompt)))
    args = sn.parse_args(common + ["--survey-run", str(survey), "--trace-run", str(trace)])
    with pytest.raises(ValueError, match="recorded boundaries differ"):
        sn.run(args, RunDirectory(tmp_path / "nonlinear"), {})
