"""Reviewer's corruption probes of dbe_report.reconstruct (d304831), each on a fresh temporary copy of the bundle with
SHA256SUMS regenerated. Imports the code from the reviewer's own clean clone (CLONE), never the worktree."""
import json, math, shutil, sys, tempfile, copy
from pathlib import Path
CLONE = Path(sys.argv[1]); WORK = Path(sys.argv[2])
sys.path.insert(0, str(CLONE / "research/ti_tnn"))
from scaling import dbe_report as DR, evidence  # noqa
ORIG = DR.BUNDLE
assert str(ORIG).startswith(str(CLONE)), ORIG
RUN = "run/20261107T182054Z"; AN = "analysis/20261107T184900Z"
NAN, INF = float("nan"), float("inf")
RES_ID = "20261107T182054-3c584e74"


def jl_edit(rel, fn):
    def go(b):
        p = b / rel
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        rows = fn(rows) or rows
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return go


def js_edit(rel, fn):
    def go(b):
        p = b / rel
        o = json.loads(p.read_text())
        o = fn(o) or o
        p.write_text(json.dumps(o, indent=1, sort_keys=True) + "\n")
    return go


def setp(o, path, v):
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = v


def mul(o, path, f):
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = o[path[-1]] * f


def req(path, v, i=0):
    return jl_edit(f"{RUN}/requests.jsonl", lambda rows: setp(rows[i], path, v))


def sco(path, v, i=0):
    return jl_edit(f"{RUN}/scores.jsonl", lambda rows: setp(rows[i], path, v))


def ana(path, v):
    return js_edit(f"{AN}/analysis.json", lambda o: setp(o, path, v))


def man(which, path, v):
    return js_edit(f"{RUN if which == 'run' else AN}/manifest.json", lambda o: setp(o, path, v))


def led(fn):
    return jl_edit("gpu_ledger.jsonl", fn)


def dbe_entries(rows, ev):
    return [r for r in rows if r.get("reservation") == RES_ID and r["event"] == ev]


def multi(*edits):
    def go(b):
        for e in edits:
            e(b)
    return go


def rename_run(new, update_refs):
    def go(b):
        (b / RUN).rename(b / "run" / new)
        if update_refs:
            js_edit(f"{AN}/analysis.json", lambda o: setp(o, ["run"], f"dbe/run/{new}"))(b)
            js_edit(f"{AN}/manifest.json", lambda o: setp(o, ["config", "run"], f"dbe/run/{new}"))(b)
    return go


CASES = []
def case(name, edit, expect):   # expect: True refuse, False accept, None judgement
    CASES.append((name, edit, expect))

# requirement 1: comparison coverage
case("analysis: extra top-level key", ana(["extra"], 1), True)
case("analysis: missing key 'cost'", js_edit(f"{AN}/analysis.json", lambda o: o.pop("cost") and None), True)
case("analysis: protocol sha changed", ana(["protocol", "sha256"], "0" * 64), True)
case("analysis: protocol frozen_at changed", ana(["protocol", "frozen_at"], "0" * 40), True)
case("analysis: run path changed", ana(["run"], "dbe/run/20261107T000000Z"), True)
case("analysis: gpu_reservations hours +0.01", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["gpu_reservations", 0, "hours"], 1.02)), True)
case("analysis: gpu_reservations status 'failed'", ana(["gpu_reservations", 0, "status"], "failed"), True)
case("analysis: gpu_reservations id changed", ana(["gpu_reservations", 0, "id"], "x"), True)
case("analysis: a second gpu reservation", js_edit(f"{AN}/analysis.json", lambda o: o["gpu_reservations"].append(dict(o["gpu_reservations"][0]))), True)
case("analysis: ledger used_hours +0.1", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["ledger", "used_hours"], 1.01)), True)
case("analysis: ledger used_hours NaN", ana(["ledger", "used_hours"], NAN), True)
case("analysis: ledger remaining_hours changed", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["ledger", "remaining_hours"], 0.9)), True)
case("analysis: ledger cap_hours 32 (int)", ana(["ledger", "cap_hours"], 32), True)
case("analysis: ledger authorization 2.0", ana(["ledger", "authorization"], 2.0), True)
case("analysis: ledger.ledger not a string", ana(["ledger", "ledger"], 5), True)
case("analysis: an analysed value changed (pooled V1 81)", ana(["pooled", "V1"], 81), True)
case("run manifest: commit changed", man("run", ["git", "commit"], "1" * 40), True)
case("analysis manifest: commit changed", man("an", ["git", "commit"], "1" * 40), True)
case("both manifests: same other commit", multi(man("run", ["git", "commit"], "1" * 40), man("an", ["git", "commit"], "1" * 40)), None)
case("run manifest: short commit", man("run", ["git", "commit"], "8e1ee1b"), True)
case("run manifest: research_tree_dirty True", man("run", ["git", "research_tree_dirty"], True), True)
case("run manifest: research_tree_dirty 0", man("run", ["git", "research_tree_dirty"], 0), True)
case("analysis manifest: dirty_paths non-empty", man("an", ["git", "dirty_paths"], ["x.py"]), True)
case("run manifest: requested list reordered", js_edit(f"{RUN}/manifest.json", lambda o: o["config"]["requested"].reverse()), True)
case("run manifest: one requested id replaced", js_edit(f"{RUN}/manifest.json", lambda o: setp(o, ["config", "requested", 0], "bird23tf-00000")), True)
case("run manifest: seeds [200, 202]", man("run", ["config", "seeds"], [200, 202]), True)
case("run manifest: seeds [200.0, 201]", man("run", ["config", "seeds"], [200.0, 201]), True)
case("run manifest: lookup version", man("run", ["config", "lookup"], "db-evidence-v1.3"), True)
case("run manifest: protocol sha", man("run", ["config", "protocol", "sha256"], "0" * 64), True)
case("analysis manifest: protocol frozen_at", man("an", ["config", "protocol", "frozen_at"], "0" * 40), True)
case("analysis manifest: run changed", man("an", ["config", "run"], "dbe/run/x"), True)
case("run dir moved (renamed, references not updated)", rename_run("20261107T182055Z", False), True)
case("run dir renamed, analysis and its manifest updated to match", rename_run("20261107T182055Z", True), True)
case("a second run directory", lambda b: shutil.copytree(b / RUN, b / "run/20261107T190000Z"), True)
case("a second analysis directory", lambda b: shutil.copytree(b / AN, b / "analysis/20261107T190000Z"), True)
case("request db_id changed", req(["db_id"], "world"), True)
case("score db_id changed", sco(["db_id"], "world"), True)
# the bundled ledger
case("ledger: dbe end entry elapsed +60 s", led(lambda rows: setp(dbe_entries(rows, "end")[0], ["elapsed"], dbe_entries(rows, "end")[0]["elapsed"] + 60)), True)
case("ledger: dbe end entry hours field changed (usage ignores it)", led(lambda rows: setp(dbe_entries(rows, "end")[0], ["hours"], 0.9)), None)
case("ledger: dbe end entry status 'failed'", led(lambda rows: setp(dbe_entries(rows, "end")[0], ["status"], "failed")), True)
case("ledger: dbe end entry removed", led(lambda rows: [r for r in rows if not (r.get("reservation") == RES_ID and r["event"] == "end")]), True)
case("ledger: a mid heartbeat of the dbe run changed (below the max)", led(lambda rows: setp(dbe_entries(rows, "heartbeat")[3], ["elapsed"], 1.0)), None)
case("ledger: a heartbeat raised above the end elapsed", led(lambda rows: setp(dbe_entries(rows, "heartbeat")[3], ["elapsed"], 5000.0)), True)
case("ledger: an earlier (non-dbe) reservation's end elapsed +60 s", led(lambda rows: setp(next(r for r in rows if r["event"] == "end" and r["reservation"] != RES_ID), ["elapsed"], next(r for r in rows if r["event"] == "end" and r["reservation"] != RES_ID)["elapsed"] + 60)), True)
def second_dbe(rows):
    s = dict(dbe_entries(rows, "start")[0]); e = dict(dbe_entries(rows, "end")[0])
    s["reservation"] = e["reservation"] = "20261107T200000-deadbeef"; s["t"] += 7200; e["t"] += 7200
    return rows + [s, e]
case("ledger: a second 'dbe ' reservation", led(second_dbe), True)
def second_dbe_zero(rows):
    s = dict(dbe_entries(rows, "start")[0]); e = dict(dbe_entries(rows, "end")[0])
    s["reservation"] = e["reservation"] = "20261107T200000-deadbeef"; s["command"] = "dbe analyze"; e["elapsed"] = 0.0
    return rows + [s, e]
case("ledger: a second 'dbe ' reservation of zero hours", led(second_dbe_zero), True)
case("ledger: the dbe run's start extra.run_dir changed", led(lambda rows: setp(dbe_entries(rows, "start")[0], ["extra", "run_dir"], "dbe/run/x")), None)
case("summary: reservation id changed", js_edit(f"{RUN}/summary.json", lambda o: setp(o, ["reservation", "id"], "x")), True)
case("summary: reservation hours changed", js_edit(f"{RUN}/summary.json", lambda o: setp(o, ["reservation", "hours"], 0.9)), None)
case("summary: load_seconds NaN", js_edit(f"{RUN}/summary.json", lambda o: setp(o, ["load_seconds"], NAN)), None)
# requirement 2: raw values
TIMES = [("generation (original.call.seconds)", "req", ["original", "call", "seconds"]),
         ("generation (D1 revision)", "req", ["D1", "revised", "call", "seconds"]),
         ("generation (V1 revision)", "req", ["V1", "revised", "call", "seconds"]),
         ("tool (original.execution.seconds)", "req", ["original", "execution", "seconds"]),
         ("tool (V1 revision execution)", "req", ["V1", "revised", "execution", "seconds"]),
         ("lookup (evidence.seconds)", "req", ["evidence", "seconds"]),
         ("lookup package latency_seconds", "req", ["evidence", "package", "latency_seconds"]),
         ("revision_seconds_allowed", "req", ["revision_seconds_allowed"]),
         ("submitted execution (score predicted.elapsed_seconds)", "sco", ["score", "predicted", "elapsed_seconds"])]
for label, kind, path in TIMES:
    f = req if kind == "req" else sco
    for v, vn in ((NAN, "NaN"), (INF, "+Infinity"), (-INF, "-Infinity"), (-1e-9, "-1e-9"), (True, "True")):
        case(f"time {label} = {vn}", f(path, v), True)
    case(f"time {label} = -0.0 (replacing a nonzero value)", f(path, -0.0), True if "latency" not in label and "allowed" not in label else None)
    case(f"time {label} = 1e308", f(path, 1e308), True if "latency" not in label and "allowed" not in label else None)
COUNTS = [("original prompt_tokens", "req", ["original", "call", "prompt_tokens"]),
          ("V1 generated_tokens", "req", ["V1", "revised", "call", "generated_tokens"]),
          ("D1 peak_memory_bytes", "req", ["D1", "revised", "call", "peak_memory_bytes"]),
          ("V1 execution.row_count", "req", ["V1", "revised", "execution", "row_count"]),
          ("package payload_chars", "req", ["evidence", "package", "payload_chars"]),
          ("package queries", "req", ["evidence", "package", "queries"]),
          ("package instructions", "req", ["evidence", "package", "instructions"]),
          ("package payload_chars_untruncated", "req", ["evidence", "package", "payload_chars_untruncated"]),
          ("score predicted.row_count", "sco", ["score", "predicted", "row_count"]),
          ("score predicted.distinct_rows", "sco", ["score", "predicted", "distinct_rows"])]
for label, kind, path in COUNTS:
    f = req if kind == "req" else sco
    for v, vn in ((True, "True"), (False, "False"), (1.0, "1.0"), (-1, "-1"), (NAN, "NaN")):
        case(f"count {label} = {vn}", f(path, v, 1), True)
BOOLS = [("V1 execution.truncated", ["V1", "revised", "execution", "truncated"]), ("evidence.ok", ["evidence", "ok"]),
         ("package.truncated", ["evidence", "package", "truncated"]), ("package.wall_safety_stop", ["evidence", "package", "wall_safety_stop"])]
for label, path in BOOLS:
    for v, vn in ((0, "0"), (1, "1"), ("false", "'false'"), (None, "None")):
        case(f"Boolean {label} = {vn}", req(path, v, 2), True)
for v, vn in ((True, "True"), (1.0, "1.0"), (2, "2"), (NAN, "NaN")):
    case(f"correct = {vn}", sco(["score", "correct"], v, 3), True)
for v, vn in ((200.0, "200.0"), (True, "True"), ("200", "'200'")):
    case(f"request seed = {vn}", req(["seed"], v), True)
    case(f"score seed = {vn}", sco(["seed"], v), True)
# requirement 3: legitimate round-off
case("LEGIT: unchanged copy", lambda b: None, False)
case("LEGIT: analysis cost V1 total_seconds * (1+1e-12)", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["cost", "V1", "total_seconds"], 1 + 1e-12)), False)
case("LEGIT: analysis cost D0 per_request_seconds * (1-1e-12)", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["cost", "D0", "per_request_seconds"], 1 - 1e-12)), False)
case("LEGIT: analysis ledger used_hours * (1+1e-12)", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["ledger", "used_hours"], 1 + 1e-12)), False)
case("LEGIT: analysis gpu_reservations hours * (1+1e-12)", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["gpu_reservations", 0, "hours"], 1 + 1e-12)), False)
case("LEGIT: analysis lookup seconds_total * (1+1e-12)", js_edit(f"{AN}/analysis.json", lambda o: mul(o, ["lookup", "seconds_total"], 1 + 1e-12)), False)

results = []
for name, edit, expect in CASES:
    tmp = Path(tempfile.mkdtemp(dir=WORK)) / "dbe_v1"
    shutil.copytree(ORIG, tmp)
    edit(tmp)
    evidence.write_sums(tmp)
    DR.BUNDLE = tmp
    try:
        DR.reconstruct()
        out = "accepted"
    except DR.Inconsistent as e:
        out = "refused: " + str(e)[:160]
    except BaseException as e:  # noqa: BLE001
        out = f"CRASH {type(e).__name__}: {str(e)[:160]}"
    shutil.rmtree(tmp.parent)
    ok = None if expect is None else (out.startswith("refused") == expect)
    results.append({"case": name, "expected": {True: "refuse", False: "accept", None: "judgement"}[expect],
                    "outcome": out, "as_expected": ok})
    print(f"{'OK ' if ok else ('?? ' if ok is None else 'BAD')} {name}: {out}")
DR.BUNDLE = ORIG
summary = {"cases": len(results), "as_expected": sum(r["as_expected"] is True for r in results),
           "unexpected": [r for r in results if r["as_expected"] is False],
           "judgement": [r for r in results if r["as_expected"] is None]}
(WORK / "probe_results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
print("SUMMARY", json.dumps({k: (v if isinstance(v, int) else len(v)) for k, v in summary.items()}))
