"""Post-hoc structure of edit eligibility across closed pilot runs (read-only).

Usage:
    python scripts/edit_structure_analysis.py LABEL=RUN_DIR[:CONVENTION] ...

CONVENTION is ``raw`` (default) or ``rstrip`` and must match the run's own
answer convention. The script reads each run's saved manifest and results,
re-parses the saved AV descriptions, and prints the tables in
reports/edit_eligibility_structure.md as JSON. It loads no model and changes no
file. "Receivers" are the rows the frozen edit rule parses: side A, querying
the affected variable.
"""

from collections import Counter
import json
from pathlib import Path
import re
import sys

from open_weight_lingua import answer_slot
from open_weight_lingua.tasks import Statement, interpret
from open_weight_lingua.text_edits import parse as frozen_parse

NUMBER = re.compile(r"(?<![0-9A-Za-z_.])(\d{1,2})(?![0-9])")
GUARD = r"(?<![A-Za-z0-9_])"
MARK = r"[`'\"*]*"
FROZEN_FORMS = ("{v} is currently", "{v} is now", "the current value of {v} is")


def program(prompt: str) -> tuple[Statement, ...]:
    statements = []
    for line in prompt.split("\nWhat is")[0].split("\n"):
        variable, rhs = (part.strip() for part in line.split("="))
        arithmetic = re.fullmatch(r"([xy]) ([+-]) (\d+)", rhs)
        if arithmetic:
            operation = "add" if arithmetic.group(2) == "+" else "subtract"
            statements.append(Statement(variable, operation, int(arithmetic.group(3))))
        elif rhs in ("x", "y"):
            statements.append(Statement(variable, "copy", rhs))
        else:
            statements.append(Statement(variable, "assign", int(rhs)))
    return tuple(statements)


def answer(generation: dict | None, convention: str) -> int | None:
    if not generation or not generation.get("terminated"):
        return None
    text = generation["text"].rstrip() if convention == "rstrip" else generation["text"]
    return int(text) if re.fullmatch(r"0|[1-9][0-9]*", text) else None


def numbers(text: str) -> set[int]:
    return {int(n) for n in NUMBER.findall(text)}


def bound_values(text: str, variable: str) -> list[int]:
    pattern = re.compile(
        GUARD + MARK + variable + MARK
        + r"\s*(?:=|==|:|is now|is currently|is|equals|becomes|holds)\s*"
        + MARK + r"(\d{1,2})(?![0-9])"
    )
    return [int(m.group(1)) for m in pattern.finditer(text)]


def relaxed_frozen_hit(text: str) -> bool:
    flat = re.sub(r"[`*\"']", "", text).lower()
    return any(
        re.search(GUARD + form.format(v=v) + r"\s*(\d{1,2})(?![0-9])", flat)
        for v in ("x", "y")
        for form in FROZEN_FORMS
    )


def analyze(run: Path, convention: str) -> dict:
    rows = {r["id"]: r for r in json.loads((run / "results.json").read_text())}
    inputs = {r["id"]: r for r in json.loads((run / "manifest.json").read_text())["inputs"]}
    text = {i: rows[i]["description"].get("description") or "" for i in inputs}
    census = {"all": Counter(), "receivers": Counter()}
    for rid, row in inputs.items():
        prog = program(row["prompt"])
        state = interpret(prog)
        query = row["variable"]
        other = "y" if query == "x" else "x"
        assert str(state[query]) == row["answer"]
        literals = {int(n) for n in re.findall(r"\d+", row["prompt"])}
        counterpart = rid.replace("-A-", "-B-") if "-A-" in rid else rid.replace("-B-", "-A-")
        cf = int(inputs[counterpart]["answer"])
        unrelated = inputs[rows[rows[rid]["controls"]["shuffled_description_id"]]["controls"]["shuffled_description_id"]]
        desc, nums, slot = text[rid], numbers(text[rid]), answer_slot.parse(text[rid])
        scopes = ["all"] + (["receivers"] if row["side"] == "A" and row["affected"] else [])
        for scope in scopes:
            c = census[scope]
            c["rows"] += 1
            frozen = frozen_parse(desc)
            c["frozen_hits_any_variable"] += any(frozen[v].status != "absent" for v in ("x", "y"))
            c["frozen_hits_queried_variable"] += frozen[query].status != "absent"
            c["frozen_relaxed_hits"] += relaxed_frozen_hit(desc)
            c["names_queried_variable"] += bool(re.search(GUARD + MARK + query + MARK + r"(?![A-Za-z0-9_])", desc))
            c["contains_true_answer"] += state[query] in nums
            c["contains_unrelated_answer"] += int(unrelated["answer"]) in nums and unrelated["answer"] != row["answer"]
            if state[query] not in literals:
                c["computed_answer_rows"] += 1
                c["computed_answer_in_text"] += state[query] in nums
            if state[other] != state[query] and state[other] not in literals:
                c["computed_other_rows"] += 1
                c["computed_other_in_text"] += state[other] in nums
            bound = bound_values(desc, query)
            c["bound_statement_rows"] += bool(bound)
            c["bound_statement_true"] += state[query] in bound
            c[f"slot_{slot.status}"] += 1
            if slot.lead is not None:
                c["slot_lead_is_answer"] += slot.lead == state[query]
                c["slot_lists_answer"] += state[query] in slot.candidates
                if cf != state[query]:
                    c["slot_lists_counterfactual"] += cf in slot.candidates
            c["mentions_final_token"] += "final token" in desc.lower()
    channel = Counter()
    for rid, row in inputs.items():
        foreign = rows[rid]["controls"]["shuffled_description_id"]
        lead = answer_slot.parse(text[foreign]).lead
        own, foreign_answer = int(row["answer"]), int(inputs[foreign]["answer"])
        p3 = answer(rows[rid]["conditions"]["P3"]["generation"], convention)
        if lead is None or p3 is None:
            channel["unscored"] += 1
            continue
        if lead != own:
            channel["foreign_lead_differs"] += 1
            channel["p3_takes_foreign_lead"] += p3 == lead
            channel["p3_keeps_own_answer"] += p3 == own
            unrelated = int(inputs[rows[foreign]["controls"]["shuffled_description_id"]]["answer"])
            if unrelated not in (own, lead):
                channel["unrelated_baseline_rows"] += 1
                channel["p3_takes_unrelated_answer"] += p3 == unrelated
        if lead not in (own, foreign_answer) and foreign_answer != own:
            channel["foreign_lead_wrong"] += 1
            channel["wrong_p3_takes_lead"] += p3 == lead
            channel["wrong_p3_takes_foreign_true_answer"] += p3 == foreign_answer
            channel["wrong_p3_keeps_own"] += p3 == own
    openers = Counter(" ".join(text[i].split()[:3]) for i in inputs)
    return {
        "run": run.name,
        "convention": convention,
        "census": {k: dict(v) for k, v in census.items()},
        "channel": dict(channel),
        "top_openers": openers.most_common(3),
        "answer_slot_rule": {"version": answer_slot.RULE_VERSION, "sha256": answer_slot.RULE_SHA256},
    }


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    out = {}
    for spec in argv:
        label, _, location = spec.partition("=")
        path, _, convention = location.partition(":")
        out[label] = analyze(Path(path), convention or "raw")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
