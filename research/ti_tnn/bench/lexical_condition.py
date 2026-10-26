"""The lexical-generalization condition's phrase inventory and question sampler (protocols/factored_interpreter_v1.md,
"A separate lexical-generalization condition"). A generator only: it defines what is asked, and decides nothing.
Specification and no-fit controls only; no neural model is fitted on it.

**Example.** Training taught "the airfield of" (an `airport_of` phrasing) in one-step questions, and the Base phrasing
"the airport serving". The condition asks:

> what is the operator of the airfield serving country_12_city_1 ?

Here "the airfield serving" occurs nowhere in training as a string. Its words do:
- "airfield" only in `airport_of` phrasings;
- "serving" only in "the airport serving".

So training identifies its meaning. The training-derived word-association baseline W grounds it to `airport_of`.
The exact-segment lookup L refuses it, because the segment is unseen.

**The phrase inventory** (`inventory`) is complete under one declared rule. Each relation's taught head (the words
between "the" and the taught phrasing's last word) is combined with each connector (last word) of the relation's
Base prefix phrasings other than "of". Strings that already occur among the declared phrasings are dropped. That
gives four forms, one per relation except `capital_of`, whose Base phrasings use only "of":

| Relation | Taught | Base phrasing with another connector | New form |
|---|---|---|---|
| `largest_city_of` | the most populous city of | the biggest city in | the most populous city in |
| `country_of` | the nation of | the country containing | the nation containing |
| `airport_of` | the airfield of | the airport serving | the airfield serving |
| `operator_of` | the firm running | the company operating | the firm operating |

**The sampler** (`lexical_items`) draws from the robustness study's held-out population (instances not used in
training), independently and with replacement, in prefix style. It builds two designs of matched variants:

| Design | Base case | Variants |
|---|---|---|
| LG1 | a one-step held-out instance whose relation has a form | `LG1` (the form: answer), `LG1_taught` (answer), `LG1_base` (answer) |
| LG2 | a two-step held-out instance, with position j among those whose relation has a form | `LG2` (the form at j: answer), `LG2_taught` (taught at j: answer), `LG2_base` (answer), `LG2_held` (the held-out phrasing at j: abstain) |

`LG2_held` is a matched control at the same position. It is reported with the condition and never pooled with
S4.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as IR  # noqa: E402  (the robustness study's phrasings, renderer and frames)
from integrated import reference as R  # noqa: E402

GENERATOR_VERSION = "ti_tnn_factored_interpreter_v1"
LG_SIZES = {"LG1": 400, "LG2": 800}
LG1_VARIANTS = ("LG1", "LG1_taught", "LG1_base")
LG2_VARIANTS = ("LG2", "LG2_taught", "LG2_base", "LG2_held")


def phrase_words(template: str) -> tuple[str, ...]:
    return tuple(template.replace("{x}", "").split())


def inventory() -> dict[str, list[str]]:
    """relation -> the new forms (templates), under the declared rule."""
    declared = {phrase_words(t) for r in IR.BASE_PREFIX
                for t in list(IR.BASE_PREFIX[r]) + [IR.TAUGHT[r], IR.HELD[r], IR.BASE_POSSESSIVE[r]]}
    forms = {}
    for r in sorted(IR.BASE_PREFIX):
        head = phrase_words(IR.TAUGHT[r])[1:-1]
        for base in IR.BASE_PREFIX[r]:
            connector = phrase_words(base)[-1]
            form = ("the",) + head + (connector,)
            if connector != "of" and form not in declared:
                forms.setdefault(r, []).append(" ".join(form) + " {x}")
    return forms


def held_instances(world, study) -> list[tuple[tuple[str, ...], str]]:
    """The robustness study's held-out population of instances with at most two relations, in a fixed order."""
    train = {(tuple(p), s) for p, s in study["train_instances"]}
    return sorted((p, s) for p in R.well_typed_paths(world, 2)
                  for s in world.types[world.relations[p[0]].source].values if (tuple(p), s) not in train)


def lexical_items(world, study, seed: int) -> list[dict]:
    rng = random.Random(f"{GENERATOR_VERSION}|lexical|{world.identity}|{seed}")
    forms = inventory()
    held = held_instances(world, study)
    one = [(p, s) for p, s in held if len(p) == 1 and p[0] in forms]
    two = [(p, s) for p, s in held if len(p) == 2 and any(r in forms for r in p)]
    items = []

    def add(category, p, s, phrases, frame, group, j, form, expected="answer"):
        items.append({"id": f"{world.seed}-lg-{len(items)}", "split": "test", "category": category, "kind": "supported",
                      "text": IR.render(s, phrases, frame), "program": list(p), "entity": s, "expected": expected,
                      "answer": R.reference_answer(world, p, s), "style": "prefix", "group": group, "frame": frame,
                      "lg_position": j, "lg_form": form})

    for g in range(LG_SIZES["LG1"]):
        p, s = rng.choice(one)
        frame, form = rng.choice(IR.BASE_FRAMES), rng.choice(forms[p[0]])
        base = IR.base_phrases(p, "prefix", rng)
        add("LG1", p, s, [form], frame, f"lg1-{g}", 0, form)
        add("LG1_taught", p, s, [IR.TAUGHT[p[0]]], frame, f"lg1-{g}", 0, form)
        add("LG1_base", p, s, base, frame, f"lg1-{g}", 0, form)
    for g in range(LG_SIZES["LG2"]):
        p, s = rng.choice(two)
        j = rng.choice([i for i, r in enumerate(p) if r in forms])
        frame, form = rng.choice(IR.BASE_FRAMES), rng.choice(forms[p[j]])
        base = IR.base_phrases(p, "prefix", rng)
        at = lambda tpl: [tpl if i == j else base[i] for i in range(len(p))]  # noqa: E731
        add("LG2", p, s, at(form), frame, f"lg2-{g}", j, form)
        add("LG2_taught", p, s, at(IR.TAUGHT[p[j]]), frame, f"lg2-{g}", j, form)
        add("LG2_base", p, s, base, frame, f"lg2-{g}", j, form)
        add("LG2_held", p, s, at(IR.HELD[p[j]]), frame, f"lg2-{g}", j, form, expected="abstain")
    return items
