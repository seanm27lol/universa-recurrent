"""Repeated updates: export -> independent re-derivation -> deliberate corruption (protocols/repeated_updates_v1.md).

One small deterministic run on a 6-country development world, with eight arms. They cover every replay
policy, both stopping rules, both references and two specialist arms. Every screened-out attempt is fully
checked here, since the protocol asks for exhaustive audit coverage on small fixtures.

- **The run.** `repeated_updates_verify.verify` must re-derive it, and sampled reproduction must give
  identical records.
- **Corruptions caught everywhere.** Each corruption of the export below must be rejected.
- **The coherent forgery.** An error removed with the digest recomputed passes re-derivation and is caught only
  by reproduction. The protocol states that boundary, and the test shows it.
- **The decision.** Its three outcomes are checked on hand-made aggregates.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
torch = pytest.importorskip("torch")
import repeated_updates as RU  # noqa: E402
import repeated_updates_verify as VF  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402

ARMS = ("shared/random/1/8/fixed", "shared/relation_first/1/16/early", "shared/signature/1/8/fixed",
        "shared/check_guided/1/4/early", "shared/none", "shared/full", "specialists/0", "specialists/1/4")
UNIT = Path("units") / "w100_i100"


def _small(fn):
    mp = pytest.MonkeyPatch()
    full = RU.arms
    mp.setattr(RU, "arms", lambda: [a for a in full() if a["name"] in ARMS])
    mp.setattr(RU, "AUDIT_MODULUS", 1)  # every screened-out attempt is fully checked
    try:
        return fn()
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    torch.set_num_threads(4)
    out = tmp_path_factory.mktemp("repeated")
    args = RU.parse_args(["--worlds", "100", "--inits", "100", "--countries", "6", "--steps", "600", "--out", str(out)])
    return _small(lambda: RU.run(args, out))


def _lines(run_dir):
    return [json.loads(x) for x in gzip.open(run_dir / UNIT / "proposals.jsonl.gz", "rt").read().splitlines()]


def _write(run_dir, lines):
    with gzip.open(run_dir / UNIT / "proposals.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))


def test_the_run_re_derives_and_its_sampled_chains_reproduce(run):
    out = _small(lambda: VF.verify(run, development=True, reproduce=True))
    assert out["verified"] and out["construction"]["screen_checker_agreement"]
    assert out["reproduction"]["all_identical"] and out["reproduction"]["chains"] == len(ARMS)
    lines = _lines(run)
    trained = [x for x in lines if x["duplicate_of"] is None]
    assert all(x["gates"] is not None for x in trained)  # exhaustive audit on the fixture
    assert all(set(x["gates"].values()) == {"pass"} for x in trained if x["accepted"])
    for ref in ("shared/none", "shared/full"):  # one distinct training set per request: attempts 2-8 are duplicates
        assert all(x["duplicate_of"] == 0 for x in lines if x["arm"] == ref and x["proposal"] > 0)
    arms = out["worlds"]["100"]["arms"]
    assert arms["shared/full"]["accepted"] == 12 and arms["shared/none"]["accepted"] < 12


def test_frozen_verification_requires_the_frozen_sources_and_design(run, tmp_path):
    """Frozen mode accepts only a run of the frozen sources (revision 3) and the frozen design: worlds 900-902 x
    inits 0-4 and the frozen configuration. The development fixture is neither."""
    with pytest.raises(VF.VerificationError, match="frozen"):
        _small(lambda: VF.verify(run))
    assert VF.FROZEN_SOURCES is not None and set(VF.FROZEN_SOURCES) == set(RU.SOURCES)
    bad = tmp_path / "run"
    shutil.copytree(run, bad)
    meta = json.loads((bad / "run.json").read_text())
    meta["provenance"]["sha256"] = {**VF.FROZEN_SOURCES, "repeated_updates.py": "0" * 64}
    (bad / "run.json").write_text(json.dumps(meta))
    with pytest.raises(VF.VerificationError, match="frozen source"):
        _small(lambda: VF.verify(bad))
    meta = json.loads((run / "run.json").read_text())
    assert VF.inventory_problems({**meta, "worlds": {}, "units": []}, development=True)
    assert _small(lambda: VF.inventory_problems({**meta, "arms": meta["arms"][:-1]}, development=True))


def test_the_harness_refuses_the_confirmation_worlds_without_the_frozen_protocol(tmp_path):
    with pytest.raises(SystemExit, match="confirmation"):
        RU.run(RU.parse_args(["--worlds", "900", "--inits", "0", "--out", str(tmp_path)]), tmp_path)


# --- corruptions caught everywhere ----------------------------------------------------------------------------------
def _first(lines, pred):
    return next(x for x in lines if pred(x))


def _drop_replay(lines):
    x = _first(lines, lambda x: x["duplicate_of"] is None and x["replay_size"] > 1)
    x["train"] = x["train"][:-1]


def _hide_error(lines):
    _first(lines, lambda x: x["errors"])["errors"].pop()


def _wrong_parent(lines):
    _first(lines, lambda x: x["update"] > 0)["parent"] = "0" * 64


def _refused_landed(lines):
    x = _first(lines, lambda x: x["errors"])
    x["deployed_world"] = x["target_world"]


def _flip_acceptance(lines):
    _first(lines, lambda x: x["accepted"])["accepted"] = False


def _accepted_with_a_failed_gate(lines):
    _first(lines, lambda x: x["accepted"])["gates"]["calibration_and_resources"] = "insufficient"


def _unflagged_duplicate(lines):
    _first(lines, lambda x: x["duplicate_of"] is not None)["duplicate_of"] = None


def _missing_check(lines):
    _first(lines, lambda x: x["screen"])["gates"] = None


def _not_wrong(lines):
    x = _first(lines, lambda x: x["errors"])
    x["errors"][0][1] = "country_0"  # an answer outside the relation's target type, or simply not its error


def _early_steps(lines):
    _first(lines, lambda x: x["arm"].endswith("/early") and x["duplicate_of"] is None)["steps"] = 7


def _check_guided_draw(lines):
    x = _first(lines, lambda x: x["arm"].startswith("shared/check_guided") and x["proposal"] > 0
               and x["duplicate_of"] is None)
    x["train"] = sorted(x["train"][:-1] + [min(set(range(len(x["train"]) + 1)) - set(x["train"]))])  # another draw


def _missing_attempt(lines):
    lines.remove(_first(lines, lambda x: x["proposal"] > 0))


def _extra_attempt(lines):
    lines.append(dict(lines[-1]))


CORRUPTIONS = {"replay set": _drop_replay, "hidden error": _hide_error, "wrong parent": _wrong_parent,
               "refused request landed": _refused_landed, "flipped acceptance": _flip_acceptance,
               "accepted with a failed gate": _accepted_with_a_failed_gate, "unflagged duplicate": _unflagged_duplicate,
               "missing full check": _missing_check, "error entry not an error": _not_wrong,
               "step count": _early_steps, "check_guided draw": _check_guided_draw,
               "missing attempt": _missing_attempt, "extra attempt": _extra_attempt}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_every_deliberate_corruption_is_rejected(run, tmp_path, name):
    bad = tmp_path / "run"
    shutil.copytree(run, bad)
    lines = _lines(bad)
    CORRUPTIONS[name](lines)
    _write(bad, lines)
    with pytest.raises(VF.VerificationError):
        _small(lambda: VF.verify(bad, development=True))


def test_a_unit_summary_that_does_not_re_derive_is_rejected(run, tmp_path):
    bad = tmp_path / "run"
    shutil.copytree(run, bad)
    unit = json.loads((bad / UNIT / "unit.json").read_text())
    unit["chains"]["shared/full"]["accepted"] -= 1
    (bad / UNIT / "unit.json").write_text(json.dumps(unit))
    with pytest.raises(VF.VerificationError, match="accepted count"):
        _small(lambda: VF.verify(bad, development=True))


def test_a_coherent_forgery_is_caught_only_by_reproduction(run, tmp_path):
    """One wrong answer removed from a rejected attempt of a `random` chain, with the table digest recomputed. The
    record re-derives: the attempt still fails its screen, and nothing downstream reads its errors. Only retraining
    the chain shows that the network did not behave that way."""
    from integrated import relations as W
    bad = tmp_path / "run"
    shutil.copytree(run, bad)
    lines = _lines(bad)
    arm = "shared/random/1/8/fixed"
    x = _first(lines, lambda y: y["arm"] == arm and len(y["errors"] or []) >= 2)
    meta = json.loads((bad / "run.json").read_text())
    world = W.generate_world(100, n_countries=meta["config"]["countries"])
    requests, keys = RU.schedule(world), RU.fact_keys(world)
    facts = {r: dict(t) for r, t in world.facts.items()}
    for y in (y for y in lines if y["arm"] == arm):  # the chain's accepted history up to x gives x's target world
        req = requests[y["update"]]
        target = {r: dict(t) for r, t in facts.items()}
        target[req["relation"]][req["source"]] = req["target"]
        if y is x:
            break
        if y["accepted"]:
            facts = target
    x["errors"].pop()
    table = [target[r][s_] for r, s_ in keys]
    for i, a in x["errors"]:
        table[i] = a
    x["table_digest"] = content_identity(table)
    _write(bad, lines)
    assert _small(lambda: VF.verify(bad, development=True))["verified"]
    rep = _small(lambda: VF.verify(bad, development=True, reproduce=True))["reproduction"]
    assert [r["arm"] for r in rep["results"] if r["identical"] is False] == [arm]


# --- the decision ---------------------------------------------------------------------------------------------------
def _agg(full: int, best: int, exact: int = 2, n: int = 24):
    arms = {a["name"]: {"accepted": 0, "requests": n} for a in RU.arms()}
    arms["shared/full"]["accepted"] = full
    for a in RU.arms():
        if a["policy"] in RU.POLICIES and a["beta"] == "1/8":
            arms[a["name"]]["accepted"] = best
    return {"shared_bases_exact": exact, "arms": arms}


@pytest.mark.parametrize("full, best, exact, outcome", [
    (24, 20, 2, "gap"),                # reference qualifies; ordinary below 90%
    (24, 22, 2, "no target gap"),      # ordinary at 22/24 >= 90%, though behind a perfect reference
    (19, 5, 2, "inconclusive"),        # the reference fails its bar: no conclusion about a gap
    (24, 20, 1, "inconclusive"),       # an inexact shared base: the world does not qualify
])
def test_the_decision_has_three_outcomes(full, best, exact, outcome):
    agg = {w: _agg(full, best, exact) for w in ("100", "101", "102")}
    assert VF.decision(agg, [100, 101])["1/8"]["outcome"] == outcome


# --- revision 3: H1 at beta = 1/4, and the descriptive maxima ------------------------------------------------------
def _world_agg(accepted: dict, n: int = 24, exact: int = 2) -> dict:
    """Every arm with `n` requests (half capital moves), accepting `accepted.get(name, 0)` of them, spread evenly
    over the two kinds (capital moves get the odd one)."""
    arms = {}
    for a in RU.arms():
        k = accepted.get(a["name"], 0)
        cap = accepted.get((a["name"], "capital_moves"), (k + 1) // 2)
        arms[a["name"]] = {"accepted": k, "requests": n,
                           "by_kind": {"capital_moves": [cap, n // 2], "operator_changes": [k - cap, n // 2]},
                           "costs": {"steps": 1000, "distinct_replay": 3 * n, "trained": n, "example_steps": 10 * n,
                                     "checker_runs": n}}
    return {"shared_bases_exact": exact, "arms": arms}


@pytest.mark.parametrize("gap_at, holds", [("1/4", True), ("1/8", False)])
def test_h1_is_evaluated_at_one_quarter_not_one_eighth(gap_at, holds):
    """Revision 3 moved H1 from 1/8 to 1/4. A 20-point capital-move difference at the other allowance must not
    count."""
    accepted = {"shared/full": 24}
    for beta in ("1/8", "1/4"):
        sig, rel = (12, 1) if beta == gap_at else (6, 6)
        accepted[(f"shared/signature/{beta}/fixed", "capital_moves")] = sig
        accepted[(f"shared/relation_first/{beta}/fixed", "capital_moves")] = rel
        accepted[f"shared/signature/{beta}/fixed"] = sig
        accepted[f"shared/relation_first/{beta}/fixed"] = rel
    agg = {w: _world_agg(accepted) for w in ("100", "101")}
    assert VF.predictions(agg)["H1_signature_over_relation_first_on_capital_moves"] is holds


def test_the_descriptive_maxima_leave_the_primary_decision_unchanged():
    """`check_guided` closes the target (23 of 24) while the other policies reach only 18: the decision is 'no
    target gap', and the summary excluding `check_guided`, below target, must not turn it into a gap."""
    import copy
    accepted = {"shared/full": 24, "shared/check_guided/1/8/fixed": 23}
    for p in ("random", "relation_first", "signature"):
        for s in ("fixed", "early"):
            accepted[f"shared/{p}/1/8/{s}"] = 18 if (p, s) in (("signature", "fixed"), ("random", "early")) else 10
    agg = {w: _world_agg(accepted) for w in ("100", "101", "102")}
    before, frozen = VF.decision(agg, [100, 101]), copy.deepcopy(agg)
    summary = VF.best_observed(agg, [100, 101])
    assert VF.decision(agg, [100, 101]) == before and agg == frozen
    assert before["1/8"]["outcome"] == "no target gap"
    row = summary["1/8"]["100"]
    assert row["qualifies"] and row["all_ordinary"]["rate"] == 23 / 24
    assert list(row["all_ordinary"]["winners"]) == ["shared/check_guided/1/8/fixed"]
    assert row["excluding_check_guided"]["rate"] == 18 / 24 < 0.9
    assert sorted(row["excluding_check_guided"]["winners"]) == ["shared/random/1/8/early", "shared/signature/1/8/fixed"]
    assert row["excluding_check_guided"]["winners"]["shared/signature/1/8/fixed"]["distinct_replay_per_request"] == 3


def test_the_verifier_reports_the_decision_and_the_summary_from_the_same_aggregates(run):
    out = _small(lambda: VF.verify(run, development=True))
    assert out["decision"] == VF.decision(out["worlds"], [100])
    assert out["best_observed"] == VF.best_observed(out["worlds"], [100])
