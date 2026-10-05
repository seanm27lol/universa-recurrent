import numpy as np
import pytest

from open_weight_lingua import state_probe_audit as audit


def test_controls_expose_scalar_and_intercept_restrictions():
    controls = audit.positive_controls()
    np.testing.assert_allclose(controls["scalar_state"]["frozen_onehot"], .1)
    assert max(controls["scalar_state"]["affine_scalar"]) == 1.
    np.testing.assert_allclose(controls["imbalanced_affine_separable"]["frozen_onehot"], .65)
    np.testing.assert_allclose(controls["imbalanced_affine_separable"]["affine_onehot"], 1.)


def test_affine_predictions_match_independent_primal_ridge_with_intercept():
    rng = np.random.default_rng(41)
    x, e = rng.normal(size=(60, 7)), rng.normal(size=(30, 7))
    y = rng.integers(0, 20, 60)
    result = audit.ridge_predictions(x, e, {"y": y})["y"]
    xc, ec = x - x.mean(0), e - x.mean(0)
    onehot = np.eye(20)[y]
    scale = np.mean(np.sum(xc * xc, axis=1))
    for index, alpha in enumerate(audit.ALPHAS):
        operator = np.linalg.solve(xc.T @ xc + alpha * scale * np.eye(7), xc.T)
        scores = ec @ operator @ (onehot - onehot.mean(0)) + onehot.mean(0)
        scalar = ec @ operator @ (y - y.mean()) + y.mean()
        np.testing.assert_array_equal(result["affine_onehot"][index], scores.argmax(1))
        np.testing.assert_array_equal(result["affine_scalar"][index], np.clip(np.rint(scalar), 0, 19))


def test_frozen_replay_matches_existing_cpu_implementation():
    from open_weight_lingua.state_trace_survey import fit_predict
    rng = np.random.default_rng(12)
    x, e, y = rng.normal(size=(50, 9)), rng.normal(size=(20, 9)), rng.integers(0, 20, 50)
    expected = fit_predict(x, y, e, audit.ALPHAS, "cpu")
    actual = audit.ridge_predictions(x, e, {"y": y})["y"]["frozen_onehot"]
    np.testing.assert_array_equal(actual, expected)


def test_alpha_selection_uses_only_provided_selection_predictions_and_larger_tie():
    predictions = np.array([[1, 2], [1, 3], [1, 2]])
    assert audit.alpha_index(predictions, np.array([1, 2])) == 2


def test_layer_selection_ignores_held_out_scores_and_checks_both_variables():
    def entry(select, test):
        return {"asked": {v: {"select": {"carried": select, "arithmetic": select},
                                "test": {"carried": test, "carried_computed": test, "arithmetic": test},
                                "permuted_R1": .05, "permuted_R2": .05} for v in audit.VARIABLES}}
    grid = {0: entry(.7, .99), 1: entry(.8, .3), 2: entry(.8, .99)}
    result = audit.reading_summary(grid)
    assert result["R1"]["layer"] == result["R2"]["layer"] == 1
    assert not result["both_readings_meet_thresholds_and_controls"]


def test_constant_features_retain_priors_for_affine_readouts():
    x = np.ones((20, 3))
    y = np.array([7] * 18 + [9] * 2)
    result = audit.ridge_predictions(x, x[:2], {"y": y})["y"]
    np.testing.assert_array_equal(result["frozen_onehot"], 0)
    np.testing.assert_array_equal(result["affine_onehot"], 7)
    np.testing.assert_array_equal(result["affine_scalar"], 7)


def test_comparison_exposes_numeric_differences_instead_of_hiding_them():
    result = audit.comparison([{"layer": 0, "x": {"alpha": .1, "test": .5}}],
                              [{"layer": 0, "x": {"alpha": 1., "test": .51}}])
    assert result["changed_metrics"] == 2 and result["alpha_changes"]


def test_group_crossing_is_rejected_before_any_probe_fit():
    manifest = {"rows": [{"id": "a"}], "records": [{"group_id": "g", "part": "train"}, {"group_id": "g", "part": "test"}]}
    with pytest.raises(ValueError, match="crosses"):
        audit.metadata(manifest)
