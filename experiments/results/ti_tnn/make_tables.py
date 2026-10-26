"""Rebuild the TI-TNN reports' tables from this evidence bundle alone.

Checks every file against manifest.json first (sha256), then prints:
- the toy benchmarks' held-out results;
- the real benchmarks, recomputed with research/ti_tnn/bench/summarize.py and checked against the
  bundled summary.json;
- both composition runs, as exact correct/total counts per path;
- the adaptive-compute verdict, recomputed with research/ti_tnn/bench/frontier.py from the bundled runs
  and checked against the bundled frontier.json. Every hash in that file's run manifest must match a
  bundled run, so the bundle holds exactly the files that were evaluated.
- the learned-relation-operator verdict, recomputed with research/ti_tnn/bench/typed_composition.py
  and checked against the bundled verdict.json;
- the execution-record and intervention verdict, recomputed with
  research/ti_tnn/bench/semantic_interventions.py and checked against its bundled verdict.json;
- the learned-interpreter verdict, recomputed with research/ti_tnn/bench/typed_questions.py and checked
  against its bundled verdict.json;
- the bounded-recurrence verdict, recomputed with research/ti_tnn/bench/typed_recurrence.py and checked
  against its bundled verdict.json;
- the module-replacement verdict, recomputed with research/ti_tnn/bench/local_updates.py and checked
  against its bundled verdict.json;
- the interpreter robustness confirmation, re-derived from its bundled decision records by
  research/ti_tnn/bench/interpreter_robustness_verify.py (about two minutes).
  and checked against the bundled verdict.json.
- the lookup task's shared-path control, recomputed with frontier.py --shared-control and checked
  against the recorded output.
- the integration-test confirmation: its bundled verifier output, or, with the session release downloaded,
  every number rebuilt by research/ti_tnn/bench/integration_test_verify.py (about five minutes):

      gh release download ti-tnn-integration-test-sessions-v1 --repo seanm27lol/universa-recurrent --dir integration_sessions
      python experiments/results/ti_tnn/make_tables.py --integration-sessions integration_sessions
- the S2 diagnosis: the historical classification of every S2 question, the development fits and the fresh
  confirmation, re-derived from the bundled records with research/ti_tnn/bench/s2_diagnosis.py and
  s2_development_analysis.py and checked against the bundled analyses (about five minutes). Arm A's
  development trace tables are rebuilt from the bundle. The other arms' trace tables need the trace release
  (s2_diagnosis/trace_release.json):

      gh release download ti-tnn-s2-diagnosis-traces-v1 --repo seanm27lol/universa-recurrent --dir s2_traces
      python experiments/results/ti_tnn/make_tables.py --s2-traces s2_traces
- the factored interpreter (development; closed without a confirmation): the no-fit controls re-verified from the
  bundle; with the A+K trace release downloaded, the pilot re-verified and the A+K diagnosis re-run:
      gh release download ti-tnn-factored-pilot-traces-v1 --repo seanm27lol/universa-recurrent --dir factored_traces
      python experiments/results/ti_tnn/make_tables.py --factored-traces factored_traces
  Here and in the S2 rebuild, comparisons are exact except recomputed binomial bounds and p-values, which may differ
  by the declared rounding tolerance (research/ti_tnn/bench/statistical_tolerance.py).

    python experiments/results/ti_tnn/make_tables.py            # from the repository root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "research" / "ti_tnn" / "bench"


def verify(bundle: Path = HERE) -> dict:
    """Every listed file exists with its recorded hash, and no JSON file is unlisted."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    listed = {f["path"] for f in manifest["files"]}
    for f in manifest["files"]:
        digest = hashlib.sha256((bundle / f["path"]).read_bytes()).hexdigest()
        if digest != f["sha256"]:
            raise ValueError(f"{f['path']}: sha256 {digest} does not match the manifest")
    stray = {str(p.relative_to(bundle)) for pattern in ("*.json", "*.json.gz") for p in bundle.rglob(pattern)} \
        - listed - {"manifest.json"}
    if stray:
        raise ValueError(f"files not in the manifest: {sorted(stray)}")
    return manifest


def mean_sd(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def toy_tables(bundle: Path = HERE) -> dict:
    comp = json.loads((bundle / "toy_benchmarks/20261015T002259/compositional.json").read_text())
    route = json.loads((bundle / "toy_benchmarks/20261015T002259/routing.json").read_text())
    unsup = json.loads((bundle / "unsupervised_routing/20261015T152306/unsupervised.json").read_text())
    by = lambda doc: defaultdict(list, {})
    out = {"compositional_heldout_xor": {}, "routing_task_accuracy": {}, "unsupervised": {}}
    groups = defaultdict(list)
    for r in comp["runs"]:
        groups[r["variant"]].append(r["eval"]["combined_test"]["acc"])
    out["compositional_heldout_xor"] = {v: mean_sd(a) for v, a in groups.items()}
    groups = defaultdict(list)
    for r in route["runs"]:
        groups[r["variant"]].append(r["eval"]["task_acc"])
    out["routing_task_accuracy"] = {v: mean_sd(a) for v, a in groups.items()}
    groups = defaultdict(list)
    for r in unsup["runs"]:
        groups[r["variant"]].append((r["structure"]["task_acc"], r["structure"]["purity"], r["structure"]["bridge_fire_rate"]))
    out["unsupervised"] = {v: {"task_accuracy": mean_sd([x[0] for x in a]), "purity": mean_sd([x[1] for x in a]),
                               "bridge_fire_rate": mean_sd([x[2] for x in a])} for v, a in groups.items()}
    return out


def strip_paths(summary):
    """summarize.py records each run's own path; compare everything else."""
    if isinstance(summary, dict):
        return {k: strip_paths(v) for k, v in summary.items() if k != "runs"}
    return summary


def real_benchmark_tables(bundle: Path = HERE) -> dict:
    sys.path.insert(0, str(BENCH))
    import summarize  # research/ti_tnn/bench/summarize.py
    lm, scan = summarize.load([bundle / "real_benchmarks/lm", bundle / "real_benchmarks/scan"])
    rebuilt = summarize.summarize(lm, scan)
    recorded = json.loads((bundle / "real_benchmarks/summary.json").read_text())
    if strip_paths(rebuilt) != strip_paths(recorded):
        raise ValueError("the rebuilt real-benchmark summary differs from the bundled summary.json")
    return rebuilt


def composition_tables(bundle: Path = HERE) -> dict:
    out = {}
    for name in ("functorial_composition/20261016T152240", "functorial_composition_controls/20261016T173618"):
        doc = json.loads((bundle / name / "composition.json").read_text())
        counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        acc = defaultdict(lambda: defaultdict(list))
        for r in doc["runs"]:
            for family, res in r["eval"].items():
                if not isinstance(res, dict) or "paths" not in res:
                    continue
                acc[r["variant"]][family].append(res["accuracy"])
                for path, p in res["paths"].items():
                    correct = round(p["accuracy"] * p["n"])
                    counts[r["variant"]][f"{family}:{path}"][0] += correct
                    counts[r["variant"]][f"{family}:{path}"][1] += p["n"]
        out[name.split("/")[0]] = {
            v: {"mean_sd": {f: mean_sd(a) for f, a in acc[v].items()},
                "correct_total": {k: tuple(c) for k, c in counts[v].items()},
                "two_hop_correct_total": tuple(map(sum, zip(*[c for k, c in counts[v].items() if k.startswith("two_hop:")])))}
            for v in acc}
    return out


def adaptive_tables(bundle: Path = HERE) -> dict:
    """Rebuild the frozen adaptive-compute verdict from the bundle and check it against frontier.json."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())
    original = {str(bundle / f["path"]): f["original_path"] for f in manifest["files"]}
    by_original = {f["original_path"]: f.get("original_sha256", f["sha256"]) for f in manifest["files"]}
    recorded = json.loads((bundle / "adaptive_compute/frontier.json").read_text())
    for part in ("hops", "lm"):  # the evaluated files are exactly the bundled ones
        for row in recorded[part]["manifest"]:
            if by_original.get(row["path"]) != row["sha256"]:
                raise ValueError(f"{row['path']}: evaluated file is not the bundled one")
    sources = {str(bundle / "real_benchmarks/lm"): "316c18f", str(bundle / "adaptive_compute"): "64ecc86"}
    A, mA = frontier.load([bundle / "adaptive_compute/hops"], "hops", sources)
    B, mB = frontier.load([bundle / "real_benchmarks/lm", bundle / "adaptive_compute/lm"], "lm", sources)
    for m in mA + mB:  # judge with the paths the evaluator saw
        m["path"] = original[str(Path(m["path"]))]
    out = {}
    for part, V, man, ref, higher, close, ratio in (("hops", A, mA, "dense", True, 0.02, False),
                                                   ("lm", B, mB, "dense", False, 0.02, True)):
        inv = frontier.inventory(V, part, man)
        P = frontier.planned_only(V, part)
        pred = frontier.judge(P, ref, higher, close, ratio)
        if part == "lm":
            pred["latency"] = frontier.latency_prediction(
                json.loads((bundle / "adaptive_compute/latency.json").read_text()), man)
        for k in ("efficiency", "equal_compute", "matched_cap") + (("latency",) if part == "lm" else ()):
            if pred[k]["holds"] != recorded[part]["predictions"][k]["holds"]:
                raise ValueError(f"{part} {k}: rebuilt {pred[k]['holds']}, recorded {recorded[part]['predictions'][k]['holds']}")
        for n, v in P.items():
            if abs(v["quality"]["mean"] - recorded[part]["variants"][n]["quality"]["mean"]) > 1e-12:
                raise ValueError(f"{part} {n}: rebuilt quality differs")
        out[part] = {"complete": inv["complete"], "variants": P,
                     "predictions": {k: pred[k]["holds"] for k in pred if isinstance(pred[k], dict) and "holds" in pred[k]}}
    out["lm"]["validated"] = frontier.all_of([out["lm"]["predictions"]["efficiency"], out["lm"]["predictions"]["equal_compute"]])
    if out["lm"]["validated"] != recorded["lm"]["validated"]:
        raise ValueError("the rebuilt decision differs from the recorded one")
    return out


def init_control_tables(bundle: Path = HERE) -> dict:
    """Rebuild the WikiText-103 initialization control's verdict from the bundle and check it against the
    recorded analysis (frontier_ic_v2_init_control.json)."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())
    original = {str(bundle / f["path"]): f["original_path"] for f in manifest["files"]}
    V, rows = frontier.load([bundle / "real_benchmarks/lm", bundle / "adaptive_compute/lm",
                             bundle / "adaptive_compute_init_control/lm"], "lm")
    for m in rows:
        m["path"] = original[str(Path(m["path"]))]
    doc = json.loads((bundle / "adaptive_compute_init_control/latency_init_control.json").read_text())
    rebuilt = frontier.init_control(V, rows, doc)
    recorded = json.loads((bundle / "adaptive_compute_init_control/frontier_ic_v2_init_control.json").read_text())
    for n, f in recorded["families"].items():
        if rebuilt["families"][n]["holds"] != f["holds"]:
            raise ValueError(f"{n}: rebuilt {rebuilt['families'][n]['holds']}, recorded {f['holds']}")
    if rebuilt["latency"]["holds"] != recorded["latency"]["holds"]:
        raise ValueError("the rebuilt latency verdict differs from the recorded one")
    return rebuilt


def learning_tables(bundle: Path = HERE) -> dict:
    """The lookup-task learning work: the frozen initialization check, every exploratory diagnostic run,
    and part A v2's verdict rebuilt with frontier.py --hops-v2 and checked against the recorded one."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())

    def runs(root):
        out = []
        for f in sorted((bundle / root).rglob("result.json")):
            r = json.loads(f.read_text())
            c = r["config"]
            knobs = "".join(f" {k}={c[k]}" for k, d in (("lr", 1e-3), ("balance_coef", 0.01)) if c.get(k, d) != d)
            out.append({"round": f.parent.parent.name, "name": frontier.v2_name(r) + knobs, "seed": c["seed"],
                        "hard_accuracy": r["eval"]["hard_accuracy"], "by_hops": r["eval"]["accuracy_by_hops"]})
        return out
    by_original = {f["original_path"]: f.get("original_sha256", f["sha256"]) for f in manifest["files"]}
    recorded = json.loads((bundle / "adaptive_compute_part_a_v2/frontier_part_a_v2.json").read_text())
    for row in recorded["manifest"]:  # the evaluated files are exactly the bundled ones
        if by_original.get(row["path"]) != row["sha256"]:
            raise ValueError(f"{row['path']}: evaluated file is not the bundled one")
    V, rows = frontier.load([bundle / "adaptive_compute_part_a_v2/hops_v2"], "hops", namer=frontier.v2_name,
                            spec=frontier.V2_PROTOCOL)
    rebuilt = frontier.evaluate_v2(V, rows)
    for key in ("learning_fixes_work", "ti_tnn_passes"):
        if rebuilt[key] != recorded[key]:
            raise ValueError(f"part A v2 {key}: rebuilt {rebuilt[key]}, recorded {recorded[key]}")
    for key in ("validity", "fixed_moe"):
        if rebuilt[key]["holds"] != recorded[key]["holds"]:
            raise ValueError(f"part A v2 {key}: rebuilt differs from recorded")
    for key in ("efficiency", "equal_compute", "matched_cap"):
        if rebuilt["predictions"][key]["holds"] != recorded["predictions"][key]["holds"]:
            raise ValueError(f"part A v2 prediction {key}: rebuilt differs from recorded")
    return {"init_check": runs("lookup_init_check"), "diagnostics": runs("lookup_learning_diagnostics"),
            "part_a_v2": rebuilt, "part_a_v2_variants": V}


def typed_composition_tables(bundle: Path = HERE) -> dict:
    """Learned relation operators (integration plan PR 4): the frozen verdict rebuilt with
    typed_composition.evaluate() from the bundled runs and checked against the recorded verdict.json."""
    sys.path.insert(0, str(BENCH))
    import typed_composition  # research/ti_tnn/bench/typed_composition.py
    root = bundle / "typed_composition/20261018T145735"
    result = json.loads((root / "typed_composition.json").read_text())
    recorded = json.loads((root / "verdict.json").read_text())
    rebuilt = typed_composition.evaluate(result)
    if rebuilt != recorded:
        raise ValueError("the rebuilt typed-composition verdict differs from verdict.json")
    return rebuilt

def shared_control_tables(bundle: Path = HERE) -> dict:
    """The lookup task's shared-path control, rebuilt with frontier.py --shared-control from the bundled control
    runs and part A v2's bundled runs, and checked against the recorded output."""
    sys.path.insert(0, str(BENCH))
    import frontier
    root = bundle / "adaptive_compute_shared_path_control"
    recorded = json.loads((root / "frontier_sc_shared_control.json").read_text())
    V2, _ = frontier.load([bundle / "adaptive_compute_part_a_v2/hops_v2"], "hops", namer=frontier.v2_name,
                          spec=frontier.V2_PROTOCOL)
    C, _ = frontier.load([root / "hops_shared_control"], "hops", namer=frontier.v2_name, spec=frontier.V2_PROTOCOL)
    rebuilt = frontier.evaluate_shared_control(C, V2)
    if dict(rebuilt, variants=C) != recorded:
        raise ValueError("the rebuilt shared-path control differs from the recorded output")
    return dict(rebuilt, variants=C)


def semantic_interventions_tables(bundle: Path = HERE) -> dict:
    """Execution records, replay and interventions (integration plan PR 5): the frozen verdict rebuilt with
    semantic_interventions.evaluate() from the bundled runs and checked against the recorded verdict.json."""
    sys.path.insert(0, str(BENCH))
    import semantic_interventions  # research/ti_tnn/bench/semantic_interventions.py
    root = bundle / "semantic_interventions/20261018T153819"
    result = json.loads((root / "semantic_interventions.json").read_text())
    recorded = json.loads((root / "verdict.json").read_text())
    rebuilt = semantic_interventions.evaluate(result)
    if rebuilt != recorded:
        raise ValueError("the rebuilt semantic-interventions verdict differs from verdict.json")
    return dict(rebuilt, executions=sum(r["executions"] for r in result["runs"]),
                interventions=sum(r["interventions"]["n"] for r in result["runs"]))


def typed_routing_tables(bundle: Path = HERE) -> dict:
    """The learned interpreter with calibrated abstention (integration plan PR 6): the frozen verdict rebuilt with
    typed_questions.evaluate() from the bundled runs and checked against the recorded verdict.json."""
    sys.path.insert(0, str(BENCH))
    import typed_questions  # research/ti_tnn/bench/typed_questions.py
    root = bundle / "typed_routing/20261018T160927"
    result = json.loads((root / "typed_questions.json").read_text())
    recorded = json.loads((root / "verdict.json").read_text())
    rebuilt = typed_questions.evaluate(result)
    if rebuilt != recorded:
        raise ValueError("the rebuilt typed-routing verdict differs from verdict.json")
    return rebuilt


def typed_recurrence_tables(bundle: Path = HERE) -> dict:
    """Bounded recurrence and cost accounting (integration plan PR 7): the frozen verdict rebuilt with
    typed_recurrence.evaluate() from the bundled runs and checked against the recorded verdict.json."""
    sys.path.insert(0, str(BENCH))
    import typed_recurrence  # research/ti_tnn/bench/typed_recurrence.py
    root = bundle / "typed_recurrence/20261018T162814"
    result = json.loads((root / "typed_recurrence.json").read_text())
    recorded = json.loads((root / "verdict.json").read_text())
    rebuilt = typed_recurrence.evaluate(result)
    if rebuilt != recorded:
        raise ValueError("the rebuilt typed-recurrence verdict differs from verdict.json")
    rec = [r for r in result["runs"] if r["system"] == "recurrent"]
    return dict(rebuilt, queries=sum(r["queries"] for r in rec),
                pause_points=sum(r["pause_resume"]["pause_points"] for r in rec))


def local_updates_tables(bundle: Path = HERE) -> dict:
    """Offline module replacement (integration plan PR 8): the frozen verdict rebuilt with local_updates.evaluate()
    from the bundled runs and checked against the recorded verdict.json."""
    sys.path.insert(0, str(BENCH))
    import local_updates  # research/ti_tnn/bench/local_updates.py
    root = bundle / "local_updates/20261018T164944"
    result = json.loads((root / "local_updates.json").read_text())
    recorded = json.loads((root / "verdict.json").read_text())
    rebuilt = local_updates.evaluate(result)
    if rebuilt != recorded:
        raise ValueError("the rebuilt local-updates verdict differs from verdict.json")
    return rebuilt


def local_updates_post_repair_tables(bundle: Path = HERE) -> dict:
    """The module-replacement rerun after the contract repairs: its verdict rebuilt from its own runs, checked
    against its recorded verdict and against the original run's verdict (they must be identical)."""
    sys.path.insert(0, str(BENCH))
    import local_updates
    root = bundle / "local_updates_post_repair/20261018T182038"
    rebuilt = local_updates.evaluate(json.loads((root / "local_updates.json").read_text()))
    if rebuilt != json.loads((root / "verdict.json").read_text()):
        raise ValueError("the rebuilt post-repair verdict differs from its verdict.json")
    if rebuilt != json.loads((bundle / "local_updates/20261018T164944/verdict.json").read_text()):
        raise ValueError("the post-repair rerun does not reproduce the original verdict")
    return rebuilt


def interpreter_robustness_tables(bundle: Path = HERE, quick: bool = False) -> dict:
    """The interpreter robustness confirmation, re-derived from its bundled records by the independent verifier
    (which regenerates every study from its seed) and checked against the stored summaries and verdict. `quick`
    re-derives one unit only and skips the verdict (for tests)."""
    sys.path.insert(0, str(BENCH))
    import interpreter_robustness_verify as V  # research/ti_tnn/bench/interpreter_robustness_verify.py
    run = bundle / "interpreter_robustness/20261018T232057"
    out = V.verify(run, require_confirmation=not quick, only={(700, 0)} if quick else None)
    return out


def s2_diagnosis_tables(bundle: Path = HERE, quick: bool = False, traces_dir: Path | None = None) -> dict:
    """The S2 diagnosis, rebuilt from this bundle and checked against its bundled analyses.
    - historical: every S2 question of the interpreter robustness confirmation classified again from its bundled
      records (the robustness verifier runs first; about four minutes);
    - development (worlds 100-101) and confirmation (worlds 710-712): every metric, certificate, class, unit outcome
      and verdict re-derived from the bundled decision records, with each run's training additions rebuilt and its
      source bound to the frozen one;
    - trace-derived tables (teacher-forced stopping, greedy failure kinds): arm A's development traces are bundled.
      The other 63 trace files are a release asset (`s2_diagnosis/trace_release.json`). With `traces_dir` (where
      they were downloaded), each is checked against the sha256 committed in `s2_diagnosis/local_traces.json`, every
      trace is checked against its decision records, and every arm's trace tables are rebuilt and compared.
    `quick` rebuilds the development part only (for tests). Every comparison with the bundled analyses is exact, except
    recomputed bounds and p-values, which may differ within the declared statistical tolerance
    (research/ti_tnn/bench/statistical_tolerance.py): another runtime rounds the binomial tail differently.
    Trace files are checked against their committed sha256 before anything that needs torch is imported."""
    import shutil
    import tempfile
    sys.path.insert(0, str(BENCH))
    import statistical_tolerance as ST  # research/ti_tnn/bench/statistical_tolerance.py (no torch)
    root = bundle / "s2_diagnosis"
    agreement = ST.Agreement()

    def check(rebuilt: dict, recorded_path: str, keys, traced=()) -> dict:
        """Exact, except recomputed bounds and p-values, within the declared statistical tolerance."""
        recorded = json.loads((root / recorded_path).read_text())
        rebuilt = json.loads(json.dumps(rebuilt, sort_keys=True))  # the stored file's JSON round trip
        for k in keys:
            ST.require(recorded[k], rebuilt[k], f"{recorded_path}: {k}", agreement)
        for k in ("stopping", "failure_kinds", "failure_kind_by_class"):
            for arm in traced:
                ST.require(recorded[k][arm], rebuilt[k][arm], f"{recorded_path}: {k} for arm {arm}", agreement)
        return rebuilt

    runs = {part: sorted(p for p in (root / part).iterdir() if p.is_dir()) for part in ("development", "confirmation")}
    staging = None
    if traces_dir is not None:  # stage each run with its full set of traces (links; nothing is copied)
        release = json.loads((root / "trace_release.json").read_text())
        committed = {f["original_path"]: f for f in json.loads((root / "local_traces.json").read_text())["files"]}
        staging = Path(tempfile.mkdtemp(prefix="s2_traces_"))
        staged = {}
        for part, dirs in runs.items():
            for run in dirs:
                target = staging / part / run.name
                (target / "traces").mkdir(parents=True)
                for name in ("s2_development.json", "records"):
                    (target / name).symlink_to((run / name).resolve())
                for f in sorted((run / "traces").glob("*.json.gz")) if (run / "traces").exists() else ():
                    (target / "traces" / f.name).symlink_to(f.resolve())
                staged.setdefault(part, []).append(target)
        for asset in release["assets"]:
            expected = committed[asset["original_path"]]
            data = (Path(traces_dir) / asset["name"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected["sha256"] or len(data) != expected["bytes"]:
                raise ValueError(f"{asset['name']}: does not match the sha256 committed in local_traces.json")
            part, run, name = asset["name"].split(".", 2)
            (staging / part / run / "traces" / name).symlink_to((Path(traces_dir) / asset["name"]).resolve())
        runs = staged
    import s2_development_analysis as DA  # research/ti_tnn/bench/s2_development_analysis.py (torch)
    import s2_diagnosis as D2  # research/ti_tnn/bench/s2_diagnosis.py
    try:
        dev = DA.analyse(runs["development"])
        out = {"development": check(dev, "development_analysis/s2_development_analysis.json",
                                    ("units", "classes", "decision_rule", "runs"), traced=dev["traced_arms"])}
        if not quick:
            summary, _ = D2.analyse_run(bundle / "interpreter_robustness/20261018T232057")
            out["historical"] = check(summary, "historical/s2_diagnosis.json",
                                      ("counts", "details", "strata", "per_unit", "examples", "source"))
            conf = DA.analyse(runs["confirmation"])
            out["confirmation"] = check(conf, "confirmation_analysis/s2_development_analysis.json",
                                        ("units", "classes", "confirmation", "runs"), traced=conf["traced_arms"])
    finally:
        if staging is not None:
            shutil.rmtree(staging)
    out["statistical_agreement"] = agreement.summary()
    return out

def integration_test_tables(bundle: Path = HERE, sessions_dir: Path | None = None) -> dict:
    """The integration-test confirmation (protocols/integration_test_v1.md, frozen revision 4).

    Without `sessions_dir`, it returns the bundled verifier output; the manifest check has already covered it.
    With `sessions_dir` (the downloaded release `ti-tnn-integration-test-sessions-v1`), it proceeds in four steps:
    - every session file is checked against the sha256 in `integration_test/session_release.json`;
    - each file is staged next to the bundled export;
    - the frozen independent verifier (`research/ti_tnn/bench/integration_test_verify.py`) rebuilds every number,
      construction check and learned verdict, enforcing the frozen source binding;
    - the result must equal the bundled `verification.json`. This takes about five minutes."""
    import shutil
    import tempfile
    root = bundle / "integration_test"
    recorded = json.loads((root / "verification.json").read_text())
    if sessions_dir is None:
        return {"rebuilt": False, **recorded}
    release = json.loads((root / "session_release.json").read_text())
    for asset in release["assets"]:  # every downloaded file, before anything that needs torch is imported
        data = (Path(sessions_dir) / asset["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset["sha256"] or len(data) != asset["bytes"]:
            raise ValueError(f"{asset['name']}: does not match the sha256 in session_release.json")
    sys.path.insert(0, str(BENCH))
    import integration_test_verify as VF  # research/ti_tnn/bench/integration_test_verify.py (torch)
    run = root / release["run"]
    staging = Path(tempfile.mkdtemp(prefix="integration_"))
    try:
        target = staging / release["run"]
        (target / "units").mkdir(parents=True)
        (target / "run.json").symlink_to((run / "run.json").resolve())
        for asset in release["assets"]:
            source = (Path(sessions_dir) / asset["name"]).resolve()
            unit = target / "units" / asset["unit"]
            unit.mkdir()
            for f in (run / "units" / asset["unit"]).iterdir():
                (unit / f.name).symlink_to(f.resolve())
            (unit / "sessions.jsonl.gz").symlink_to(source)
        rebuilt = json.loads(json.dumps(VF.verify(target), sort_keys=True))
    finally:
        shutil.rmtree(staging)
    for key in ("verified", "commit", "verdict", "worlds", "units"):
        if rebuilt[key] != recorded[key]:
            raise ValueError(f"integration test: rebuilt {key} differs from the bundled verification.json")
    return {"rebuilt": True, **rebuilt}

def repeated_updates_tables(bundle: Path = HERE, rebuild: bool = False, reproduce: bool = False) -> dict:
    """Repeated updates under a bounded replay allowance (protocols/repeated_updates_v1.md, frozen revision 3).

    The whole weight-free export is bundled: run.json, and per unit unit.json and proposals.jsonl.gz. Without
    `rebuild` it returns the bundled verifier output; the manifest check has already covered it. With `rebuild`
    the frozen independent verifier (`research/ti_tnn/bench/repeated_updates_verify.py`) re-derives every attempt
    of every chain, enforcing the frozen source binding and inventory, and the result must equal the bundled
    `verification.json`: every aggregate, prediction, the PR 9 decision and the descriptive maxima. With
    `reproduce` it also retrains the frozen sample of 114 chains (about 20 minutes) and requires identical records."""
    root = bundle / "repeated_updates"
    recorded = json.loads((root / "verification.json").read_text())
    if not rebuild:
        return {"rebuilt": False, **recorded}
    sys.path.insert(0, str(BENCH))
    import repeated_updates_verify as VF  # research/ti_tnn/bench/repeated_updates_verify.py
    run = next(d for d in sorted(root.iterdir()) if d.is_dir())
    rebuilt = json.loads(json.dumps(VF.verify(run, reproduce=reproduce), sort_keys=True))
    keys = ("verified", "commit", "worlds", "construction", "predictions", "decision", "best_observed") + (
        ("reproduction",) if reproduce else ())
    for key in keys:
        if rebuilt[key] != recorded[key]:
            raise ValueError(f"repeated updates: rebuilt {key} differs from the bundled verification.json")
    return {"rebuilt": True, **recorded}


def factored_interpreter_tables(bundle: Path = HERE, traces_dir: Path | None = None, rebuild: bool = True) -> dict:
    """The factored interpreter's development engineering result (protocols/factored_interpreter_v1.md), recorded at
    0e1a3c9 and closed without a confirmation.

    - **Table only** (`rebuild=False`, no traces): the bundled verifier outputs and diagnosis, which the manifest
      check has already covered. Nothing here needs torch.
    - **The controls** (`rebuild`, the default): the no-fit controls run is bundled whole and re-verified by
      `factored_interpreter_verify`. The result must agree with `controls_verification.json`.
    - **The pilot** (`traces_dir`, where the release `ti-tnn-factored-pilot-traces-v1` was downloaded):
      - every trace file is checked against the sha256 in `pilot_trace_release.json`, before anything that needs
        torch is imported;
      - each file is staged next to the bundled compact export;
      - the pilot verifier and the A+K diagnosis re-run, and must agree with `pilot_verification.json` and
        `ak_diagnosis.json`.

    **"Agree"** means exactly, except recomputed bounds and p-values, within the declared statistical tolerance
    (research/ti_tnn/bench/statistical_tolerance.py). `statistical_agreement` reports what was compared at two
    levels: inside the verifiers (stored records against recomputation), and rebuilt outputs against bundled ones."""
    import shutil
    import tempfile
    root = bundle / "factored_interpreter"
    sys.path.insert(0, str(BENCH))
    import statistical_tolerance as ST  # research/ti_tnn/bench/statistical_tolerance.py (no torch)
    out = {"controls": json.loads((root / "controls_verification.json").read_text()),
           "pilot": json.loads((root / "pilot_verification.json").read_text()),
           "diagnosis": json.loads((root / "ak_diagnosis.json").read_text()),
           "controls_rebuilt": False, "pilot_rebuilt": False}
    release = json.loads((root / "pilot_trace_release.json").read_text())
    traces = {}
    if traces_dir is not None:  # every downloaded file, before anything that needs torch is imported
        for asset in release["assets"]:
            data = (Path(traces_dir) / asset["name"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != asset["sha256"] or len(data) != asset["bytes"]:
                raise ValueError(f"{asset['name']}: does not match the sha256 in pilot_trace_release.json")
            traces[asset["path"]] = data
    if not rebuild and traces_dir is None:
        return out
    import factored_interpreter_verify as FV  # research/ti_tnn/bench/factored_interpreter_verify.py (torch)
    inner, outer = ST.Agreement(), ST.Agreement()
    controls = next(d for d in sorted((root / "controls").iterdir()) if d.is_dir())
    rebuilt = json.loads(json.dumps(FV.verify(controls, agreement=inner), sort_keys=True, default=str))
    ST.require(out["controls"], rebuilt, "factored interpreter: the controls re-verification (controls_verification.json)",
               outer)
    out.update(controls=rebuilt, controls_rebuilt=True)
    if traces_dir is not None:
        import factored_pilot_diagnose as FD  # research/ti_tnn/bench/factored_pilot_diagnose.py
        import factored_pilot_verify as PV  # research/ti_tnn/bench/factored_pilot_verify.py
        staging = Path(tempfile.mkdtemp(prefix="factored_"))
        try:
            target = staging / release["run"]
            shutil.copytree(root / "pilot" / release["run"], target, symlinks=True)
            for path, data in traces.items():
                (target / path).write_bytes(data)
            pv = json.loads(json.dumps(PV.verify(target, agreement=inner), sort_keys=True, default=str))
            run = json.loads((target / "run.json").read_text())
            cfg = run["config"]
            dg = {"run": target.name, "commit": run["provenance"]["commit"],
                  "units": [FD.diagnose_unit(target, info, cfg["countries"], cfg["updates"], cfg["batch"])
                            for info in run["units"]]}
            dg = json.loads(json.dumps(dg, sort_keys=True))
        finally:
            shutil.rmtree(staging)
        ST.require(out["pilot"], pv, "factored interpreter: the pilot re-verification (pilot_verification.json)", outer)
        ST.require(out["diagnosis"], dg, "factored interpreter: the A+K diagnosis (ak_diagnosis.json)", outer)
        out.update(pilot=pv, diagnosis=dg, pilot_rebuilt=True)
    out["statistical_agreement"] = {"inside_the_verifiers": inner.summary(), "rebuilt_against_bundled": outer.summary()}
    return out


def frontier_names(learning: dict) -> list:
    sys.path.insert(0, str(BENCH))
    import frontier
    return [n for n in frontier.V2_EXPECTED if n in learning["part_a_v2_variants"]]


def wikitext_fixed_tables(bundle: Path = HERE) -> dict:
    """Rebuild the WikiText-103 fixed-recipe verdict from the bundle and check it against the recorded one."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    V, rows = frontier.load([bundle / "real_benchmarks/lm", bundle / "adaptive_compute/lm",
                             bundle / "adaptive_compute_init_control/lm", bundle / "adaptive_compute_wikitext_fixed/lm"],
                            "lm", namer=frontier.lm_name)
    rebuilt = frontier.evaluate_wikitext_fixed(V, rows)
    recorded = json.loads((bundle / "adaptive_compute_wikitext_fixed/frontier_wf_wikitext_fixed.json").read_text())
    for key in ("fixes_help_on_text", "beats_plain_skipping", "efficiency", "equal_compute"):
        if rebuilt[key]["holds"] != recorded[key]["holds"]:
            raise ValueError(f"wikitext fixed {key}: rebuilt {rebuilt[key]['holds']}, recorded {recorded[key]['holds']}")
    return rebuilt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", type=Path, help="also write every table as JSON")
    p.add_argument("--s2-traces", type=Path, help="the downloaded S2 trace release (s2_diagnosis/trace_release.json); "
                   "rebuilds every arm's trace-derived tables too")
    p.add_argument("--integration-sessions", type=Path, help="the downloaded integration-test session release "
                   "(integration_test/session_release.json); re-runs the frozen verifier on the full export")
    p.add_argument("--repeated-updates", action="store_true", help="re-run the frozen repeated-updates verifier on the "
                   "bundled export (a few minutes)")
    p.add_argument("--repeated-reproduce", action="store_true", help="with --repeated-updates, also retrain the frozen "
                   "sample of 114 chains (about 20 minutes)")
    p.add_argument("--factored-traces", type=Path, help="the factored interpreter pilot's A+K trace files "
                   "(factored_interpreter/pilot_trace_release.json); re-verifies the pilot and re-runs the A+K diagnosis")
    args = p.parse_args(argv)
    manifest = verify()
    print(f"{len(manifest['files'])} files match manifest.json\n")
    toys = toy_tables()
    print("Toy benchmarks (3 seeds; mean ± sd)")
    for v, (m, s) in toys["compositional_heldout_xor"].items():
        print(f"  held-out XOR, {v:20s} {m:.3f} ± {s:.3f}")
    for v, (m, s) in toys["routing_task_accuracy"].items():
        print(f"  routing task accuracy, {v:15s} {m:.3f} ± {s:.3f}")
    for v, d in toys["unsupervised"].items():
        print(f"  unsupervised, {v:15s} accuracy {d['task_accuracy'][0]:.3f}, purity {d['purity'][0]:.3f}, "
              f"bridge fires {d['bridge_fire_rate'][0]:.3f}")
    real = real_benchmark_tables()
    print("\nWikiText-103 (3 seeds; validation and test perplexity), rebuilt and matching summary.json")
    for v, s in real["lm"].items():
        print(f"  {v:15s} {s['validation_perplexity']['mean']:.2f} ± {s['validation_perplexity']['sd']:.2f}   "
              f"{s['test_perplexity']['mean']:.2f} ± {s['test_perplexity']['sd']:.2f}")
    print("SCAN (3 seeds; test exact match)")
    for split, variants in real["scan"].items():
        print(f"  {split:18s} " + "  ".join(f"{v} {s['test_exact_match']['mean']:.4f}" for v, s in variants.items()))
    comp = composition_tables()
    for run, variants in comp.items():
        print(f"\n{run} (5 seeds; two-hop correct / total)")
        for v, d in variants.items():
            c, n = d["two_hop_correct_total"]
            paths = "  ".join(f"{k.split(':')[1]} {a}/{b}" for k, (a, b) in d["correct_total"].items() if k.startswith("two_hop:"))
            print(f"  {v:12s} {c}/{n}   ({paths})")
    adaptive = adaptive_tables()
    print("\nAdaptive compute (3 seeds), rebuilt and matching frontier.json")
    for part, label, fmt in (("lm", "WikiText-103 validation perplexity", ".2f"), ("hops", "lookup hard accuracy", ".3f")):
        print(f"  {label}; inventory complete = {adaptive[part]['complete']}")
        for n, v in sorted(adaptive[part]["variants"].items(), key=lambda kv: kv[1]["compute"]["mean"]):
            print(f"    {n:22s} {v['quality']['mean']:{fmt}}  decoder MACs {v['compute']['mean'] / 1e6:.2f}M")
        print(f"    predictions: {adaptive[part]['predictions']}")
    print(f"  efficiency validated (predictions 5 and 6): {adaptive['lm']['validated']}")
    learning = learning_tables()
    print("\nLookup task: the frozen initialization check (moe_top1, dense-matched experts)")
    for r in learning["init_check"]:
        print(f"  seed {r['seed']}: {r['hard_accuracy']:.3f}  by hops {r['by_hops']}")
    print("Lookup task: exploratory diagnostics on development seeds (every run)")
    for r in learning["diagnostics"]:
        hops = " / ".join(f"{v:.2f}" for v in r["by_hops"].values())
        print(f"  {r['round']:10s} {r['name']:58s} seed {r['seed']}: {r['hard_accuracy']:.3f}  ({hops})")
    v2 = learning["part_a_v2"]
    print("Part A v2, rebuilt and matching the recorded verdict")
    for n in frontier_names(learning):
        v = learning["part_a_v2_variants"][n]
        print(f"  {n:58s} {v['quality']['mean']:.3f}  learned {v2['learned_by_variant'][n]}/5")
    print(f"  validity {v2['validity']['holds']}; fixed mixture {v2['fixed_moe']['holds']}; "
          f"learning fixes work {v2['learning_fixes_work']}; TI-TNN passes {v2['ti_tnn_passes']}")
    sc = shared_control_tables()
    print("Shared-path control (seeds 200-204), rebuilt and matching the recorded output")
    for n, v in sc["variants"].items():
        print(f"  {n:58s} {v['quality']['mean']:.3f}  learned {sc['learned'][n]}/5  3-hop "
              + ", ".join(f"{x:.3f}" for x in v["by_hops"]["3"]["values"]))
    print(f"  reference counts {sc['reference_counts']}; shared path insufficient "
          f"{sc['shared_path_insufficient']['holds']}; routing helps TI-TNN {sc['routing_helps_ti_tnn']['holds']}")
    wf = wikitext_fixed_tables()
    print(f"\nWikiText-103 with the lookup fixes, rebuilt and matching the recorded verdict: "
          f"{wf['candidate']['quality']['mean']:.2f} (initialization only {wf['fixes_help_on_text']['init_only']:.2f}); "
          f"fixes help = {wf['fixes_help_on_text']['holds']}, beats plain skipping = {wf['beats_plain_skipping']['holds']}")
    control = init_control_tables()
    print("\nWikiText-103 initialization control, rebuilt and matching the recorded analysis")
    for n, f in control["families"].items():
        print(f"  {n:32s} comparison keeps its sign = {f['holds']}; perplexity "
              + ", ".join(f"{k} {v:.2f}" for k, v in f["mean_perplexity"].items()))
    print(f"  latency: {control['latency']['holds']}")
    typed = typed_composition_tables()
    print("\nLearned relation operators (3 worlds x 5 inits), rebuilt and matching verdict.json; lowest value per family")
    for n, f in typed["families"].items():
        paths = "  ".join(f"{k} {v:.3f}" for k, v in sorted(f["paths_min"].items()))
        print(f"  {n:13s} atomic {f['atomic_min']:.3f}  same-endpoint {f['same_endpoint_min']:.3f}  {paths}  "
              f"competent runs {f['competent_runs']}/15")
    print(f"  predictions: {typed['predictions']}")
    sem = semantic_interventions_tables()
    print(f"\nExecution records and interventions ({sem['executions']} executions, {sem['interventions']} interventions), "
          "rebuilt and matching verdict.json; lowest value per family")
    for n, f in sem["families"].items():
        print(f"  {n:17s} answers {f['answer_accuracy_min']:.3f}  history independence {f['history_independence_min']:.3f}  "
              f"counterfactual {f['counterfactual_accuracy_min']:.3f}  undeclared calls (max) {f['undeclared_calls_max']}  "
              f"disabled unchanged {f['disabled_unchanged_min']:.3f}  law defects (max) {f['law_defects_max']}")
    print(f"  predictions: {sem['predictions']}")
    routing = typed_routing_tables()
    print("\nLearned interpreter with calibrated abstention (15 units), rebuilt and matching verdict.json")
    for n, f in routing["systems"].items():
        print(f"  {n:18s} " + "  ".join(f"{s}: coverage {v['coverage_min']:.3f}-{v['coverage_max']:.3f}, accepted errors "
                                         f"{v['accepted_errors_total']}/{v['answered_total']}" for s, v in f.items()))
    print(f"  predictions: {routing['predictions']}")
    recur = typed_recurrence_tables()
    print(f"\nBounded recurrence ({recur['queries']} queries, {recur['pause_points']} pause points), rebuilt and "
          "matching verdict.json; ranges over runs")
    for n, f in recur["systems"].items():
        print(f"  {n:15s} accuracy {f['accuracy'][0]:.3f}  applications {f['module_applications_mean'][0]:.2f}-"
              f"{f['module_applications_mean'][1]:.2f}  wall {f['wall_us_median'][0]:.0f}-{f['wall_us_median'][1]:.0f} us  "
              f"bookkeeping {f['bookkeeping_share_median'][0]:.2f}-{f['bookkeeping_share_median'][1]:.2f}")
    print(f"  predictions: {recur['predictions']}")
    upd = local_updates_tables()
    print("\nOffline module replacement (150 updates per system), rebuilt and matching verdict.json")
    for n, f in upd["families"].items():
        print(f"  {n:17s} activated {f['activated']}/{f['cases']}  write locality {f['gate_pass']['identity_and_scope']}  "
              f"unaffected kept {f['gate_pass']['unaffected_behavior']}  unaffected answers changed "
              f"{f['unaffected_changed_from_base_total']}  atomic moves {f['atomic_changed_by_relation']}")
    print(f"  faulty suite (as designed, per case): {upd['faulty_suite']}")
    print(f"  predictions: {upd['predictions']}")
    local_updates_post_repair_tables()
    print("  rerun after the contract repairs: identical verdict")
    robust = interpreter_robustness_tables()
    print("\nInterpreter robustness (worlds 700-702 x 5 inits), every number re-derived from the bundled records")
    for key, arms in robust["verdict"]["successes_by_world"].items():
        print(f"  {key:22s} " + "  ".join(f"{a} {'/'.join(str(v) for v in w.values())}" for a, w in arms.items()))
    s2 = s2_diagnosis_tables(traces_dir=args.s2_traces)
    print("\nS2 diagnosis, rebuilt from the bundled records and matching the bundled analyses")
    print("  historical S2 classes (12,000 questions per arm): " + "; ".join(
        f"{arm} exact {c['exact']['n']}, single relation {c.get('dropped:taught', {}).get('n', 0) + c.get('dropped:other', {}).get('n', 0)}"
        for arm, c in s2["historical"]["counts"].items()))
    dr = s2["development"]["decision_rule"]
    print(f"  development S2 exact means {dr['S2_exact_world_means']}; proceed to confirmation {dr['proceed_to_confirmation']}")
    conf = s2["confirmation"]["confirmation"]
    print(f"  confirmation: S2 and S3/S4 in the same unit {conf['successes_by_world']}; C holds {conf['prediction_C_holds']}; "
          f"C above B in every world {conf['secondary_C_above_B_every_world']}")
    a = s2["statistical_agreement"]
    print(f"  rebuilt against bundled: {a['exact_values']} values equal exactly; {a['statistical_values']} statistical "
          f"values, {a['statistical_differing']} differing, largest difference {a['largest_difference']:.1e}")
    it = integration_test_tables(sessions_dir=args.integration_sessions)
    v = it["verdict"]
    print(f"\nIntegration test (worlds 800-802 x 5 inits) {'rebuilt by the frozen verifier' if it['rebuilt'] else 'as verified at bundling (pass --integration-sessions to rebuild)'}: "
          f"construction every unit {v['construction_every_unit']}; learned successes by world {v['learned_successes_by_world']}")
    ru = repeated_updates_tables(rebuild=args.repeated_updates, reproduce=args.repeated_reproduce)
    print(f"\nRepeated updates (worlds 900-902 x 5 inits) "
          f"{'rebuilt by the frozen verifier' if ru['rebuilt'] else 'as verified at bundling (pass --repeated-updates to rebuild)'}: "
          f"predictions {ru['predictions']}; PR 9 decision {({b: d['outcome'] for b, d in ru['decision'].items()})}; "
          f"sampled reproduction {ru['reproduction']['identical']}/{ru['reproduction']['chains']} identical")
    fi = factored_interpreter_tables(traces_dir=args.factored_traces)
    print(f"\nFactored interpreter (development; closed without confirmation): controls re-verified "
          f"({len(fi['controls']['units'])} units); pilot {'re-verified with its A+K traces' if fi['pilot_rebuilt'] else 'as verified at bundling (pass --factored-traces to rebuild)'}"
          f" ({len(fi['pilot']['units'])} units, rules {fi['pilot']['rules']})")
    for level, a in fi["statistical_agreement"].items():
        print(f"  {level.replace('_', ' ')}: {a['exact_values']} values equal exactly; {a['statistical_values']} "
              f"statistical values, {a['statistical_differing']} differing, largest difference {a['largest_difference']:.1e} "
              f"(tolerance {a['tolerance']['rel']:g} relative, {a['tolerance']['abs']:g} absolute)")
    for part in ("development", "confirmation"):
        a = s2[part]
        print(f"  {part} trace tables rebuilt for arms {a['traced_arms']}"
              + ("" if args.s2_traces else " (the other arms need --s2-traces)"))
        for arm in a["traced_arms"]:
            e = a["stopping"][arm]["all"]["step1_p_end"]
            print(f"    {arm}: P(<end>) after the first relation, S2 mean {e['S2']['mean']:.4f} "
                  f"(share > 0.5: {e['S2']['share_above_half']:.4f}); S2_base mean {e['S2_base']['mean']:.2e}; "
                  f"greedy failure kinds {a['failure_kinds'][arm]['s2']['all']}")
    print(f"  predictions: {robust['verdict']['predictions']}")
    if args.json:
        args.json.write_text(json.dumps({"toys": toys, "real_benchmarks": real, "composition": comp,
                                         "adaptive_compute": adaptive, "typed_composition": typed},
                                        indent=2, default=list))


if __name__ == "__main__":
    main()
