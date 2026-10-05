"""Independent recount of a Phase Seven family run (does not import the frozen evaluator).

Usage:
    python scripts/prompt_factorial_recount.py RUN_DIR PHASE_SIX_RUN_DIR CONVENTION

Example: Gemma's expanded question placed before `x = 7 / y = 16 / y = y + 1 /
y = y - 3` answers 14, y's value. This script rescores every saved answer
with `answer_text_matches` under the family's convention (`rstrip` for Gemma,
`raw` for Qwen) and prints:
- the five cell accuracies and the five factorial contrasts, with a
  whole-group bootstrap (3,000 resamples, seed 12345: a check on the frozen
  evaluator's 10,000-resample intervals, not a replacement);
- accuracy when the program's last line assigns the asked or the other
  variable, and how often answers then equal the other variable's value;
- accuracy by asked variable and by the asked variable's last update;
- whether the two formats Phase Six already ran reproduce its saved texts.
Descriptive; it can never change a frozen result.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from open_weight_lingua.metrics import answer_text_matches

CONDITIONS = ("original_after", "original_before", "expanded_after", "expanded_before", "original_repeat")


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    run, phase_six, convention = Path(argv[0]), Path(argv[1]), argv[2]
    rows = [json.loads(line) for line in (run / "generations.jsonl").open()]
    rescored = [answer_text_matches(r["text"], r["answer"], convention) for r in rows]
    print("rows", len(rows), "| disagreements with stored correctness:", sum(a != r["correct"] for a, r in zip(rescored, rows)))
    by = defaultdict(dict)
    for row, correct in zip(rows, rescored):
        by[row["base_id"]][row["condition"]] = (correct, row)
    bases = sorted(by)
    if any(len(by[b]) != len(CONDITIONS) for b in bases):
        raise ValueError("a base prompt is missing a condition")
    print("cells:", {c: f"{np.mean([by[b][c][0] for b in bases]):.3f} ({sum(by[b][c][0] for b in bases)}/{len(bases)})" for c in CONDITIONS})

    def score(condition):
        return lambda b: by[b][condition][0]

    def contrast(function):
        per_group = defaultdict(list)
        for b in bases:
            per_group[by[b]["original_after"][1]["group_id"]].append(function(b))
        groups = [per_group[k] for k in sorted(per_group)]
        rng = np.random.default_rng(12345)
        draws = [np.mean([x for i in rng.integers(0, len(groups), len(groups)) for x in groups[i]]) for _ in range(3000)]
        return np.mean([x for g in groups for x in g]), np.quantile(draws, [0.0025, 0.9975])

    contrasts = {
        "position, original wording (before - after)": lambda b: score("original_before")(b) - score("original_after")(b),
        "position, expanded wording (before - after)": lambda b: score("expanded_before")(b) - score("expanded_after")(b),
        "wording, question before (expanded - original)": lambda b: score("expanded_before")(b) - score("original_before")(b),
        "wording, question after (expanded - original)": lambda b: score("expanded_after")(b) - score("original_after")(b),
        "interaction": lambda b: (score("expanded_before")(b) - score("expanded_after")(b)) - (score("original_before")(b) - score("original_after")(b)),
        "repeat - original_before": lambda b: score("original_repeat")(b) - score("original_before")(b),
        "repeat - original_after": lambda b: score("original_repeat")(b) - score("original_after")(b),
    }
    for name, function in contrasts.items():
        point, interval = contrast(function)
        print(f"  {name:48s} {point:+.3f}  99.5% [{interval[0]:+.3f}, {interval[1]:+.3f}]")
    print("by whether the last line assigns the asked variable:")
    for c in CONDITIONS:
        asked_last = np.mean([by[b][c][0] for b in bases if by[b][c][1]["last_line_assigns_asked"]])
        other = [by[b][c][1] for b in bases if not by[b][c][1]["last_line_assigns_asked"]]
        other_last = np.mean([by[b][c][0] for b in bases if not by[b][c][1]["last_line_assigns_asked"]])
        gave_other = np.mean([r["text"].strip() == r["other_answer"] and r["other_answer"] != r["answer"] for r in other])
        print(f"  {c:16s} asked-last {asked_last:.3f}  other-last {other_last:.3f}  (answers equal the other value {gave_other:.3f})")
    print("by asked variable:", {c: {v: round(float(np.mean([by[b][c][0] for b in bases if by[b][c][1]["variable"] == v])), 3) for v in "xy"} for c in CONDITIONS})
    print("by the asked variable's last update:")
    for kind in ("literal", "arithmetic", "copy"):
        chosen = [b for b in bases if by[b]["original_after"][1]["asked_last_update_kind"] == kind]
        print(f"  {kind:10s} n={len(chosen):4d}", {c: round(float(np.mean([by[b][c][0] for b in chosen])), 3) for c in CONDITIONS})
    saved = {g["id"]: g for g in json.loads((phase_six / "results.json").read_text())["generations"]}
    for condition, key in (("original_after", "question_after"), ("expanded_before", "question_first")):
        differ = [(b, saved[b][key]["text"], by[b][condition][1]["text"]) for b in bases if saved[b][key]["text"] != by[b][condition][1]["text"]]
        print(f"Phase Six reproduction, {condition} vs {key}: identical {len(bases) - len(differ)}, differ {len(differ)}", differ[:3])


if __name__ == "__main__":
    main(sys.argv[1:])
