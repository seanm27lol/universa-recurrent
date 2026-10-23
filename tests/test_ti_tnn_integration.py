"""Export -> independent reconstruction -> deliberate corruption, on small deterministic integration-test runs
(protocols/integration_test_v1.md: "Before the freeze").

Two small runs on a 6-country development world, with three branches (one of each kind):
- **ceiling:** the protocol's recipe, with every learned part perfect;
- **non-ceiling:** two test-only patches to the producer's inputs, never to its outputs:
  - S0's `operator_of` specialist is trained for a single step, so S0 is wrong on some operators. An operator
    update's fine-tuning then corrects them as well: corrections in the actual difference, from a learned
    pipeline. A route update leaves them wrong, so its candidate is rejected at the gates and stays in the
    inventory;
  - the interpreter's scores are set deterministically by question text: 80% confident, 20% spread over
    [0.5, 0.99). Parses scoring below 0.8 get the wrong target operator. Errors then fall at low scores, and a
    certificate whose labels are otherwise right selects a non-zero threshold. C0 is uncertified, because S0's
    operator errors fall at high scores.

`integration_test_verify.verify` must rebuild both runs. Each corruption of the ceiling run's export must
then be rejected: an interpreter record, a label, a certificate, a table, a context, a session record (naive
or migrated), a release decision, an exhaustive digest, a missing or extra episode, a difference identity,
or the witness. After the reviews of #58 and #59, so must an internally valid record in the wrong place, and a
release record that differs from its re-derivation in any one field, or is missing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
torch = pytest.importorskip("torch")
import integration_test as IT  # noqa: E402
import integration_test_verify as VF  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import query_interpreter as QI  # noqa: E402

SMALL = {"calibration": 500, "test": 200, "default_diagnostic": 30}


def _args():
    return IT.parse_args(["--worlds", "100", "--inits", "100", "--countries", "6", "--exhaustive-budgets", "1", "16"])


def _spread(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(IT, "SIZES", {**IT.SIZES, **SMALL})
    mp.setattr(IT, "BRANCH_KINDS", {"operator": 1, "route": 1, "two_fact": 1})
    mp.setattr(IT, "EPISODE_UNAFFECTED", 5)
    mp.setattr(IT, "INTERPRETED_UNAFFECTED", 3)
    torch.set_num_threads(4)
    out = {}
    ceiling = tmp_path_factory.mktemp("ceiling")
    IT.run(_args(), ceiling)
    out["ceiling"] = ceiling
    train_family, interpret = M.train_family, QI.interpret

    def weak_operator(world, family, seed, d, hidden, steps, lr, *a, **k):
        trained = train_family(world, family, seed, d, hidden, steps, lr, *a, **k)
        weak = train_family(world, family, seed, d, hidden, 1, lr, *a, **k)
        trained["modules"]["operator_of"] = weak["modules"]["operator_of"]
        return trained

    def spread_scores(net, vocab, texts, batch=256):
        parses = interpret(net, vocab, texts, batch)
        out_ = []
        for text, p in zip(texts, parses):
            if p is None or p.outcome != "query":
                out_.append(p)
                continue
            u = _spread(text)  # 80% confident (above the grid's first rule), 20% spread over [0.5, 0.99)
            score = 0.9995 if u < 0.8 else 0.5 + 0.49 * (u - 0.8) / 0.2
            target = p.target
            if score < 0.8 and target.startswith("operator_"):
                target = "operator_0" if target != "operator_0" else "operator_1"
            s32 = float(torch.tensor(score, dtype=torch.float32))  # components that reproduce the score exactly
            probs = (s32,) + (1.0,) * (len(p.step_probabilities) - 1)
            out_.append(replace(p, score=float(torch.tensor(probs).prod()) * 1.0 * 1.0, target=target,
                                step_probabilities=probs, start_probability=1.0, target_probability=1.0))
        return out_

    mp.setattr(M, "train_family", weak_operator)
    mp.setattr(QI, "interpret", spread_scores)
    non_ceiling = tmp_path_factory.mktemp("non_ceiling")
    IT.run(_args(), non_ceiling)
    out["non_ceiling"] = non_ceiling
    mp.undo()
    yield out


def _with_small_constants(fn):
    mp = pytest.MonkeyPatch()
    mp.setattr(IT, "SIZES", {**IT.SIZES, **SMALL})
    mp.setattr(IT, "BRANCH_KINDS", {"operator": 1, "route": 1, "two_fact": 1})
    mp.setattr(IT, "EPISODE_UNAFFECTED", 5)
    mp.setattr(IT, "INTERPRETED_UNAFFECTED", 3)
    try:
        return fn()
    finally:
        mp.undo()


def test_the_ceiling_run_rebuilds_from_its_export(runs):
    out = _with_small_constants(lambda: VF.verify(runs["ceiling"], development=True))
    assert out["verified"] and out["verdict"]["construction_every_unit"]
    unit = out["units"][0]
    assert all(v is True for v in unit["construction"].values()), unit["construction"]
    assert unit["L2"] and all(b["activated"] for b in unit["branches"])
    assert unit["episodes"]["supplied"]["records"] > 0 and unit["episodes"]["interpreted"]["records"] > 0


def test_a_non_ceiling_run_rebuilds_and_its_learned_failures_are_counted(runs):
    out = _with_small_constants(lambda: VF.verify(runs["non_ceiling"], development=True))
    assert out["verified"] and out["verdict"]["construction_every_unit"]
    unit = out["units"][0]
    assert not unit["L2"] and not unit["L3"] and not unit["L4"]  # S0 is wrong on operators; a candidate is rejected
    by_kind = {b["index"]: b for b in unit["branches"]}
    assert any(not b["activated"] for b in by_kind.values())  # the route candidate stays in the inventory, rejected
    assert any(b["activated"] and b["corrections"] > 0 for b in by_kind.values())  # learned corrections
    certs = json.loads(gzip.open(runs["non_ceiling"] / "units" / "w100_i100" / "certificates.json.gz", "rt").read())
    assert any(c["certified"] and c["threshold"] not in (None, 0.0) for c in certs.values())  # a non-zero threshold
    rows = json.loads(gzip.open(runs["non_ceiling"] / "units" / "w100_i100" / "interpreter.json.gz", "rt").read())
    refused_low = [r for r in rows if r["split"] == "test" for rel in r["release"].values() for f in rel.values()
                   if f["reason"] in ("not admitted", "below the effective threshold")]
    assert refused_low  # releases refused under a non-zero threshold


# --- deliberate corruption -----------------------------------------------------------------------------------------
def _edit_json_gz(path, edit):
    data = json.loads(gzip.open(path, "rt").read())
    edit(data)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def _edit_lines(path, edit, select):
    lines = [json.loads(x) for x in gzip.open(path, "rt").read().splitlines()]
    done = False
    for i, line in enumerate(lines):
        if not done and select(line):
            result = edit(line)
            if result == "drop":
                lines.pop(i)
            done = True
            break
    assert done, "nothing to corrupt"
    with gzip.open(path, "wt") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))


def _first_executed(rows):
    return next(r for r in rows if r["split"] == "test" and r["decision"]["execute"])


def _naive_with_mixing(line):
    return line["policy"] == "naive" and line["k"] >= 1


CORRUPTIONS = {
    "interpreter score": ("interpreter.json.gz", lambda rows: _first_executed(rows).__setitem__("score", 0.123)),
    "parse field": ("interpreter.json.gz", lambda rows: _first_executed(rows)["parse"].__setitem__("budget", 1)),
    "correctness label": ("interpreter.json.gz", lambda rows: (lambda r: r["correct"].__setitem__(
        next(iter(r["correct"])), not next(iter(r["correct"].values()))))(_first_executed(rows))),
    "release decision": ("interpreter.json.gz", lambda rows: (lambda r: (lambda f: f.__setitem__(
        "released", not f["released"]))(next(iter(r["release"].values()))["new"]))(_first_executed(rows))),
    "certificate step": ("certificates.json.gz", lambda certs: next(iter(certs.values()))["steps"][0].__setitem__(
        3, 0.5)),
    "table entry": ("tables.json.gz", lambda tables: (lambda t: t["operator_of"].__setitem__(
        next(iter(t["operator_of"])), "operator_5"))(next(iter(tables.values())))),
    "context binding": ("contexts.json.gz", lambda ctx: next(iter(ctx.values())).__setitem__("certificate", "x" * 64)),
    "exhaustive digest": ("exhaustive.json.gz", lambda e: next(iter(e.values())).__setitem__("digest", "0" * 64)),
}
LINE_CORRUPTIONS = {
    "naive trace answer": (lambda line: line["record"]["state"].update(
        status="answered", answer={"type_id": "Airport", "value": line["record"]["state"]["visited"][0],
                                   "type_version": 1}), _naive_with_mixing),
    "naive trace visited": (lambda line: line["record"]["state"]["visited"].append("country_0_city_0_airport"),
                            _naive_with_mixing),
    "migration proof": (lambda line: line["record"]["migration"].__setitem__("revalidated", 999),
                        lambda line: line["policy"] == "migrate" and line["record"]["migration"]["migrated"]
                        and line["k"] >= 1),
    "pin context": (lambda line: line["record"]["segments"][0].__setitem__("context", "y" * 64),
                    lambda line: line["policy"] == "pin"),
    "interpreted release": (lambda line: line["release"].__setitem__("released", not line["release"]["released"]),
                            lambda line: line["condition"] == "interpreted" and line["release"] is not None),
    "missing episode": (lambda line: "drop", lambda line: line["policy"] == "restart"),
    "extra episode": (lambda line: line.__setitem__("k", 999), lambda line: line["policy"] == "migrate"),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS) + sorted(LINE_CORRUPTIONS) + ["difference identity", "witness"])
def test_every_deliberate_corruption_of_the_export_is_rejected(runs, tmp_path, name):
    bad = tmp_path / "run"
    shutil.copytree(runs["ceiling"], bad)
    unit = bad / "units" / "w100_i100"
    if name in CORRUPTIONS:
        file, edit = CORRUPTIONS[name]
        _edit_json_gz(unit / file, edit)
    elif name in LINE_CORRUPTIONS:
        edit, select = LINE_CORRUPTIONS[name]
        _edit_lines(unit / "sessions.jsonl.gz", edit, select)
    elif name == "difference identity":
        data = json.loads((unit / "unit.json").read_text())
        data["branches"][0]["difference"]["identity"] = "0" * 64
        (unit / "unit.json").write_text(json.dumps(data))
    else:
        data = json.loads((bad / "run.json").read_text())
        data["worlds"]["100"]["witness"]["route"] = list(reversed(data["worlds"]["100"]["witness"]["route"]))
        (bad / "run.json").write_text(json.dumps(data))
    try:
        out = _with_small_constants(lambda: VF.verify(bad, development=True))
    except VF.VerificationError:
        return  # the export does not re-derive
    # a self-consistent record of a forbidden or unfaithful execution: a construction failure, never a pass
    assert not out["verdict"]["construction_every_unit"], name


# --- review of #58: internally valid records in the wrong place, and certificate risk settings ------------------
def _rewrite_sessions(path, transform):
    lines = [json.loads(x) for x in gzip.open(path, "rt").read().splitlines()]
    transform(lines)
    with gzip.open(path, "wt") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))


@pytest.mark.parametrize("condition", ["supplied", "interpreted"])
def test_a_valid_session_from_another_case_is_rejected(runs, tmp_path, condition):
    """The review's counterexample: keep an episode's header, substitute an internally valid session (same policy,
    pause index and branch) from another question. Each record still verifies on its own."""
    bad = tmp_path / "run"
    shutil.copytree(runs["ceiling"], bad)

    def swap(lines):
        pins = [x for x in lines if x["policy"] == "pin" and x["condition"] == condition and x["k"] == 0]
        first = pins[0]
        other = next(x for x in pins if x["query"] != first["query"] and x["branch"] == first["branch"])
        first["record"] = other["record"]  # a valid pin session for another question
        if condition == "interpreted":
            first["release"] = other["release"]
    _rewrite_sessions(bad / "units" / "w100_i100" / "sessions.jsonl.gz", swap)
    with pytest.raises(VF.VerificationError, match="another question|proposed query"):
        _with_small_constants(lambda: VF.verify(bad, development=True))


def test_a_restart_session_from_another_pause_index_is_rejected(runs, tmp_path):
    bad = tmp_path / "run"
    shutil.copytree(runs["ceiling"], bad)

    def swap(lines):
        restarts = [x for x in lines if x["policy"] == "restart" and x["condition"] == "supplied"]
        first = next(x for x in restarts if any(y["query"] == x["query"] and y["branch"] == x["branch"]
                                                and y["k"] != x["k"] for y in restarts))
        other = next(y for y in restarts if y["query"] == first["query"] and y["branch"] == first["branch"]
                     and y["k"] != first["k"])
        first["record"] = other["record"]  # the same question restarted, but from another pause index
    _rewrite_sessions(bad / "units" / "w100_i100" / "sessions.jsonl.gz", swap)
    with pytest.raises(VF.VerificationError, match="pause index"):
        _with_small_constants(lambda: VF.verify(bad, development=True))


def test_a_certificate_for_another_risk_level_is_rejected(runs):
    """The review's counterexample: alpha changed from 0.01 to 0.10 with every statistical field recomputed
    consistently. The fixed-sequence result re-derives, but the risk level is not the one expected."""
    import interpreter_robustness_verify as RV
    unit = runs["ceiling"] / "units" / "w100_i100"
    certs = json.loads(gzip.open(unit / "certificates.json.gz", "rt").read())
    expected = json.loads((unit / "unit.json").read_text())["expectation"]
    cert = dict(next(iter(certs.values())))
    records = [list(r) for r in cert["records"]]
    again = RV.certificate([r[1] for r in records], [r[2] for r in records], tuple(cert["grid"]), 0.10, cert["delta"])
    forged = {**cert, "alpha": 0.10, "steps": [list(s) for s in again["steps"]], "threshold": again["threshold"],
              "certified": again["certified"], "stop_reason": again["stop_reason"]}
    assert VF.certificate_problems(cert, records, expected, cert["snapshot_digest"]) == []
    assert any("alpha" in p for p in VF.certificate_problems(forged, records, expected, cert["snapshot_digest"]))


# --- review of #59: release records compared field by field --------------------------------------------------------
def _released_line(line):
    return line["condition"] == "interpreted" and line["release"] is not None and line["release"]["released"]


SESSION_RELEASE_EDITS = {  # each keeps "released", which is all the verifier merged in #59 compared
    "answer": lambda rel: rel.__setitem__("answer", "country_0_city_0_airport" if rel["answer"] !=
                                          "country_0_city_0_airport" else "country_0_city_1_airport"),
    "result_reason": lambda rel: rel.__setitem__("result_reason", "dead end"),
    "effective_threshold": lambda rel: rel.__setitem__("effective_threshold", 0.9),
    "supporting_step": lambda rel: rel.__setitem__("supporting_step", None),
    "reason": lambda rel: rel.__setitem__("reason", "below the effective threshold"),
}


@pytest.mark.parametrize("field", sorted(SESSION_RELEASE_EDITS))
def test_an_interpreted_release_decision_that_is_not_its_sessions_is_rejected(runs, tmp_path, field):
    """A released interpreted episode whose release decision disagrees with its own session in one field. The
    decision still says "released", under the right contexts, certificates and score; for "answer", it releases
    an answer the run never produced."""
    bad = tmp_path / "run"
    shutil.copytree(runs["ceiling"], bad)
    _edit_lines(bad / "units" / "w100_i100" / "sessions.jsonl.gz",
                lambda line: SESSION_RELEASE_EDITS[field](line["release"]), _released_line)
    try:
        out = _with_small_constants(lambda: VF.verify(bad, development=True))
    except VF.VerificationError:
        return
    assert not out["units"][0]["construction"]["release"] and not out["verdict"]["construction_every_unit"]


def _release_record(rows, select):
    """The first non-calibration release record that `select(row, field, record)` picks: (its parent, its field)."""
    for r in rows:
        if r["split"] == "calibration":
            continue
        for fields in r["release"].values():
            for field, rel in fields.items():
                if select(r, field, rel):
                    return fields, field
    raise AssertionError("nothing to corrupt")


INTERPRETER_RELEASE_EDITS = {  # each keeps "admitted" and "released" of every remaining record
    "reason of a pre-execution refusal": (lambda r, f, rel: not r["decision"]["execute"],
                                          lambda fields, f: fields[f].__setitem__("reason", "not admitted")),
    "threshold of a refusal": (lambda r, f, rel: not rel["released"],
                               lambda fields, f: fields[f].__setitem__("threshold", 0.0)),
    "supporting step of a release": (lambda r, f, rel: rel["released"],
                                     lambda fields, f: fields[f].__setitem__("supporting_step", None)),
    "a missing resumed record": (lambda r, f, rel: f == "resumed", lambda fields, f: fields.pop(f)),
}


@pytest.mark.parametrize("name", sorted(INTERPRETER_RELEASE_EDITS))
def test_an_interpreter_release_record_that_does_not_re_derive_is_rejected(runs, tmp_path, name):
    """The verifier merged in #59 compared only admission, release and (when released) the threshold, and only
    the records present. A missing record counted as "not released"."""
    bad = tmp_path / "run"
    shutil.copytree(runs["ceiling"], bad)
    select, edit = INTERPRETER_RELEASE_EDITS[name]
    _edit_json_gz(bad / "units" / "w100_i100" / "interpreter.json.gz", lambda rows: edit(*_release_record(rows, select)))
    with pytest.raises(VF.VerificationError, match="release record"):
        _with_small_constants(lambda: VF.verify(bad, development=True))


def test_the_verifiers_release_rule_is_the_adapters_on_every_path():
    """The verifier re-derives the rule without calling `certificates`; on every path, including the ones the
    confirmation never reached (no shared interpreter, an uncertified side), it must give the producer's rule."""
    from integrated import certificates as Q
    grid = (0.999, 0.99, 0.9, 0.0)

    def expectation(**changes):
        return Q.Expectation(**{**dict(interpreter_identity="i", policy_version="p", event_version="e",
                                       distribution_identity="d", grid=grid), **changes})

    def records(n=400, wrong=0, low=0):
        return ([(f"q{i}", 0.9995, True) for i in range(n)] + [(f"w{i}", 0.95, False) for i in range(wrong)]
                + [(f"l{i}", 0.5, True) for i in range(low)])
    certs = {"strict": Q.certify(records(400, wrong=1, low=50), expectation(), "s0"),  # selects 0.99
             "loose": Q.certify(records(400), expectation(), "s1"),  # selects 0.0
             "other interpreter": Q.certify(records(), expectation(interpreter_identity="j"), "s1"),
             "uncertified": Q.certify(records(100), expectation(), "s1")}
    assert certs["strict"].threshold == 0.99 and not certs["uncertified"].certified
    for a in certs.values():
        for b in certs.values():
            rule = Q.effective_release(a, b)
            assert VF.release_rule(a.as_dict(), b.as_dict(), False) == (
                rule["threshold"], rule["supporting_step"], rule["problems"])
        t = a.threshold
        assert VF.release_rule(a.as_dict(), a.as_dict(), True) == (t, list(a.supporting_step(t) or []) or None, [])
