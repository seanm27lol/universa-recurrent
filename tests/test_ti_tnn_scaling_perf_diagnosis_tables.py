"""The post-evaluation diagnosis tables (`diagnose.perf_tables`) pool each arm over its seeds, per population.

- Counts are summed over the seeds; categories and first divergences too.
- A teacher-forced error rate is pooled wrong steps over pooled steps, not a mean of per-seed rates.
- An EX-correct request with a same-reference predicate is counted as suspicious.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from scaling import diagnose as D  # noqa: E402
from scaling import perf  # noqa: E402


def _summary(correct: int, timeouts: int) -> dict:
    s = {k: 0 for k in D.PERF_COUNTS}
    s.update(correct=correct, timeouts=timeouts, requests=2)
    s["categories"] = {"table grounding": 1}
    s["first_divergence"] = {"FROM table": 1}
    return s


def _row(correct: int, same_reference: bool) -> dict:
    return {"correct": correct, "task_id": "t", "db_id": "d", "same_reference_predicates": [1] if same_reference else [],
            "plans": {"comparable": False}}


def test_perf_tables_pool_over_seeds(tmp_path):
    seeds = ("100", "101")
    x = {"evaluations": {s: f"s{s}" for s in seeds}, "summary": {}, "requests": {}, "teacher_forced": {}}
    for arm, g in perf.ARMS:
        lab = perf.label(arm, g)
        for i, s in enumerate(seeds):
            for pop in perf.POPULATIONS:
                k = f"{lab}/{s}/{pop}"
                x["summary"][k] = _summary(correct=1 + i, timeouts=i)
                x["requests"][k] = [_row(1, same_reference=(i == 0)), _row(0, same_reference=True)]
                # seed 100: 1 wrong of 1; seed 101: 0 wrong of 3 -> pooled 1/4 = 0.25 (a mean of rates would be 0.50)
                x["teacher_forced"][k] = {"representable": 2, "mean_step_accuracy": 0.5 + 0.25 * i,
                                          "by_decision": {"FROM table": {"any_wrong": 1 - i, "steps": 1 + 2 * i}}}
    (tmp_path / "perf_diagnosis.json").write_text(json.dumps(x))
    out = D.perf_tables(SimpleNamespace(out=str(tmp_path)))
    labels = " | ".join(perf.label(*a) for a in perf.ARMS)
    assert f"| free decoding | {labels} |" in out
    assert "| correct | 3 | 3 | 3 | 3 |" in out and "| timeouts | 1 | 1 | 1 | 1 |" in out
    assert "| table grounding | 2 | 2 | 2 | 2 |" in out and "| FROM table | 2 | 2 | 2 | 2 |" in out
    assert "| FROM table | 4 | 0.25 | 0.25 | 0.25 | 0.25 |" in out
    assert "| EX-correct with a suspicious program | 1 | 1 | 1 | 1 |" in out
    assert "| mean step accuracy (representable requests) | 4 | 0.625 | 0.625 | 0.625 | 0.625 |" in out
    assert out.count("### ") == len(perf.POPULATIONS)
