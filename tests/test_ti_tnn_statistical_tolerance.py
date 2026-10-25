"""The one declared tolerance in reconstruction (research/ti_tnn/bench/statistical_tolerance.py), on real roundoff.

**Example.** The factored pilot's unit (world 100, init 100, full) stores A's historical S2 bound: 11 errors in 785
answered, upper bound 0.02308779216470323 (CPython 3.11.14, the producer). CPython 3.12.3 recomputes
0.02308779216470245 from the same counts. Exact equality rejects that authentic value. The tolerance accepts it
and still rejects:
- altered counts;
- materially changed statistics, including a change just beyond the tolerance;
- altered certificate settings: threshold, alpha and delta;
- changed acceptance decisions;
- any difference, however small, outside the statistical keys.

Only the pilot verifier's own certificate check needs torch. The rest runs wherever the core suite runs. The last
test recomputes every bound and p-value stored in that unit, on whichever runtime runs it, so each CI interpreter
is checked against the producer's stored values.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
sys.path.insert(0, str(REPO / "research" / "ti_tnn" / "bench"))
import statistical_tolerance as ST  # noqa: E402
from integrated import calibration as K  # noqa: E402

UNITS = REPO / "experiments" / "results" / "ti_tnn" / "factored_interpreter" / "pilot" / "20261025T051637" / "units"
STEP = ("threshold", "accepted", "errors", "p_value", "rejected")
SETTINGS = ("threshold", "certified", "stop_reason", "alpha", "delta")

# Recomputed by CPython 3.12.3 (Ubuntu's build, aarch64) from the stored counts; the stored values are CPython 3.11.14's.
ELSEWHERE_S2_TABLE = {"error_upper_95": 0.02308779216470245,
                      "abstention_interval_95": [0.010531261182425533, 0.030736950080853996]}
ELSEWHERE_AK_STEP_10_P = 0.904370717481734  # unit w101_i101_full: A+K's exact-program certificate, step 10


def _unit(name: str) -> dict:
    return json.loads((UNITS / name / "unit.json").read_text())


def _certificate(name: str = "w101_i101_full") -> dict:
    c = _unit(name)["systems"]["AK"]["certificate"]
    return {"steps": [{k: s[k] for k in STEP} for s in c["steps"]], **{k: c[k] for k in SETTINGS}}


@pytest.fixture(scope="module")
def s2_table():
    stored = _unit("w100_i100_full")["historical_A"]["certified"]["S2"]
    assert (stored["errors"], stored["answered"], stored["n"]) == (11, 785, 800)
    assert stored["error_upper_95"] == 0.02308779216470323  # the authentic stored value
    return stored


def test_the_real_cross_runtime_roundoff_in_a_bound_is_accepted(s2_table):
    elsewhere = {**s2_table, **ELSEWHERE_S2_TABLE}
    assert elsewhere != s2_table  # exact equality rejects the authentic recomputation
    agreement = ST.Agreement()
    assert agreement.compare(s2_table, elsewhere), agreement.mismatches
    # the bound and both interval ends differ; the largest difference is the bound's, 7.8e-16
    assert agreement.statistical_differing == 3 and agreement.statistical_values == 3
    assert agreement.largest_difference == pytest.approx(7.8e-16, rel=0.01) and agreement.exact_values > 10


def test_the_real_cross_runtime_roundoff_in_a_p_value_is_accepted():
    stored = _certificate()
    assert [stored["steps"][10][k] for k in STEP] == [0.0, 778, 11, 0.9043707174818075, False]
    elsewhere = copy.deepcopy(stored)
    elsewhere["steps"][10]["p_value"] = ELSEWHERE_AK_STEP_10_P
    assert elsewhere != stored
    agreement = ST.Agreement()
    assert agreement.compare(stored, elsewhere), agreement.mismatches
    assert agreement.statistical_differing == 1 and agreement.largest_difference == pytest.approx(7.35e-14, rel=0.01)


@pytest.mark.parametrize("change", [
    "an error count", "an answered count", "a bound changed materially", "a bound changed just beyond the tolerance",
    "an interval end changed materially", "a coverage changed by roundoff (not a statistical key)",
])
def test_altered_counts_and_material_statistics_still_fail(s2_table, change):
    elsewhere = copy.deepcopy({**s2_table, **ELSEWHERE_S2_TABLE})
    if change == "an error count":
        elsewhere["errors"] += 1
    elif change == "an answered count":
        elsewhere["answered"] -= 1
    elif change == "a bound changed materially":
        elsewhere["error_upper_95"] = 0.0241
    elif change == "a bound changed just beyond the tolerance":
        elsewhere["error_upper_95"] = s2_table["error_upper_95"] * (1 + 2 * ST.REL)
    elif change == "an interval end changed materially":
        elsewhere["abstention_interval_95"][1] += 1e-6
    else:
        elsewhere["coverage"] += 1e-16
        assert elsewhere["coverage"] != s2_table["coverage"]
    assert not ST.Agreement().compare(s2_table, elsewhere)
    with pytest.raises(ValueError, match="differs from the rebuilt value"):
        ST.require(s2_table, elsewhere, "the historical S2 table")


CERTIFICATE_CHANGES = {
    "the selected threshold": lambda c: c.__setitem__("threshold", 0.4),
    "alpha": lambda c: c.__setitem__("alpha", 0.02),
    "delta": lambda c: c.__setitem__("delta", 0.1),
    "the certification decision": lambda c: c.__setitem__("certified", not c["certified"]),
    "a step's rejection decision": lambda c: c["steps"][0].__setitem__("rejected", not c["steps"][0]["rejected"]),
    "a step's accepted count": lambda c: c["steps"][3].__setitem__("accepted", c["steps"][3]["accepted"] + 1),
    "a step's threshold": lambda c: c["steps"][2].__setitem__("threshold", 0.985),
    "a p-value changed materially": lambda c: c["steps"][10].__setitem__("p_value", 0.91),
}


@pytest.mark.parametrize("change", sorted(CERTIFICATE_CHANGES))
def test_altered_certificate_settings_and_decisions_still_fail(change):
    stored = _certificate()
    elsewhere = copy.deepcopy(stored)
    elsewhere["steps"][10]["p_value"] = ELSEWHERE_AK_STEP_10_P  # the authentic roundoff, accepted on its own
    assert ST.Agreement().compare(stored, elsewhere)
    CERTIFICATE_CHANGES[change](elsewhere)
    assert not ST.Agreement().compare(stored, elsewhere)


def test_the_pilot_verifier_compares_certificates_within_the_tolerance():
    """The verifier's own certificate comparison, on the stored certificate and the real recomputed p-value."""
    pytest.importorskip("torch")
    import factored_pilot_verify as PV
    stored = _unit("w101_i101_full")["systems"]["AK"]["certificate"]
    mine = {k: stored[k] for k in SETTINGS}
    mine["steps"] = [[s[k] for k in PV.STEP_FIELDS] for s in stored["steps"]]
    mine["steps"][10][3] = ELSEWHERE_AK_STEP_10_P
    agreement = ST.Agreement()
    assert PV.same_certificate(stored, mine, agreement) and agreement.statistical_differing == 1
    for edit in (lambda m: m.__setitem__("alpha", 0.02), lambda m: m.__setitem__("delta", 0.1),
                 lambda m: m["steps"][0].__setitem__(4, not m["steps"][0][4]), lambda m: m.__setitem__("threshold", 0.4),
                 lambda m: m["steps"][10].__setitem__(3, 0.91)):
        bad = copy.deepcopy(mine)
        edit(bad)
        assert not PV.same_certificate(stored, bad)


def test_every_stored_statistic_of_a_unit_agrees_with_this_runtime():
    """Every bound and p-value stored in the unit, recomputed here from its stored counts: equal on the producer's
    runtime, within the declared tolerance on any other (CI runs this on CPython 3.11, 3.12 and 3.13)."""
    agreement, recomputed = ST.Agreement(), 0

    def walk(x, alpha=0.01):
        nonlocal recomputed
        if isinstance(x, dict):
            alpha = x["alpha"] if isinstance(x.get("alpha"), float) else alpha
            mine = {}
            if {"errors", "answered", "error_upper_95"} <= set(x):
                mine["error_upper_95"] = K.clopper_pearson_upper(x["errors"], x["answered"])
            if {"n", "answered", "abstention_interval_95"} <= set(x):
                mine["abstention_interval_95"] = K.exact_interval(x["n"] - x["answered"], x["n"])
            if {"errors", "accepted", "p_value"} <= set(x):
                mine["p_value"] = K.binomial_p_value(x["errors"], x["accepted"], alpha)
            if mine:
                recomputed += len(mine)
                assert agreement.compare({k: x[k] for k in mine}, mine), agreement.mismatches[-1]
            for v in x.values():
                walk(v, alpha)
        elif isinstance(x, list):
            for v in x:
                walk(v, alpha)

    walk(_unit("w100_i100_full"))
    assert recomputed > 600 and agreement.largest_difference <= 1e-11
