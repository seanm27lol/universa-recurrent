# Why no description is edit-eligible: the structure behind 0/128 — 2026-09-29

**Frame.** This is a **post-hoc, descriptive, not preregistered** reading of
saved evidence from three closed runs: the Qwen2.5-7B pilot
`pilot-20260921T235825Z-6164d210` (raw answer convention) and the Gemma-3-12B
and Gemma-3-27B confirmation pilots `pilot-20260927T202455Z-f5ec3892` and
`pilot-20260929T034532Z-4e4d655f` (rstrip). It loads no model. **Every
recorded outcome stands: 0/128 groups are edit-eligible on every family under
the frozen rule v1.0.0, and the edit hypothesis is untested.** It extends
[pilot_description_analysis.md](pilot_description_analysis.md), a phrasing
census of the Qwen pilot alone, to both Gemma pairs and to the structure of the
assay.

## A concrete case

Program `x = 9; y = 8; y = 3; y = y + 3; x = x + 1`, question "What is x?",
answer 10. The frozen rule edits a group only if the description of this row
states the value in one of three forms: `x is currently 10`, `x is now 10`, or
`the current value of x is 10`. The Gemma-3-27B description
(`pilot-0006-A-x`) ends:

> Final token "\n" ends a final answer label ("the result is..."), immediately
> expecting a numeric answer like "10" or "11" to specify the updated value of
> the number 9. or "10" or "11" or "10" or "11" or "10 after increment." or "1."

The value is there, but as a list of predicted next tokens, and not attached
to `x` ("the number 9"). The rule finds nothing to edit.

## Intuition

The activation is captured at the newline that opens the assistant's turn: the
model is about to write the answer digit. The plainest thing that state
carries is the next token. The AV is asked only to "describe the semantic
content" in 2–3 snippets, and on every family it writes the same three-part
genre: a structural guess ("Structured … format"), a quoted phrase that
"signals" something, and a "Final token" paragraph listing predicted
continuations. The frozen rule asks for a statement about program state bound
to a variable name. The rule targets state; the site and the AV produce a
prediction. That mismatch, not a regex detail, is why coverage is zero.

## Definitions

- **Receiver.** The one row per group that the frozen rule parses,
  `{group}-A-{affected}` (`runner._edit_plan`). Its answer is the affected
  variable's current value, so for receivers "the answer" and "the value the
  edit should change" are the same number.
- **Computed value.** A value that occurs nowhere in the prompt as an integer
  literal, so a description cannot contain it by copying the prompt.
- **Answer slot (AS-1.0.0).** From the first line that begins `Final token` to
  the next blank line; its candidates are the quoted canonical integers 0–19
  after the first `expect…`/`likely`/`like`. One distinct value is
  `eligible`, several are `ambiguous`, none is `absent`. Implemented in
  [`src/open_weight_lingua/answer_slot.py`](../src/open_weight_lingua/answer_slot.py)
  (rule sha256 `5ef9b14c5d8d…`). It is a reading aid here, not a frozen
  Phase Two rule.
- **Natural edit.** P3 patches in the description of another group's row (the
  next group, same variant ordinal). When that description's slot lead differs
  from the receiver's own answer, P3 is a replacement whose text states a
  different number — the closest thing to an edit the saved evidence contains.

## Code

```bash
.venv-phase2/bin/python research/open_weight_lingua/scripts/edit_structure_analysis.py \
  Qwen2.5-7B=<qwen-pilot-run>:raw \
  Gemma-3-12B=research/open_weight_lingua/runs/pilot-20260927T202455Z-f5ec3892:rstrip \
  Gemma-3-27B=research/open_weight_lingua/runs/pilot-20260929T034532Z-4e4d655f:rstrip
```

The script reads saved `manifest.json` and `results.json` only, recomputes
each program's state with `tasks.interpret` (which reproduces every saved
answer), and prints every number below.

## Evidence

### What the receiver descriptions contain (128 receivers per family)

| | Qwen2.5-7B L20 | Gemma-3-12B L32 | Gemma-3-27B L41 |
|---|---:|---:|---:|
| Frozen-rule hit on the affected variable | 0 | 0 | 0 |
| Frozen forms, case- and markdown-insensitive, all 512 rows | 3 | 0 | 0 |
| Names the queried variable | 46 | 50 | 116 |
| Contains the true answer anywhere | 38 | 120 | 127 |
| Contains an unrelated row's answer (chance baseline) | 36 | 13 | 18 |
| Computed answer appears in the text | 13/50 | 44/50 | 49/50 |
| The *other* variable's computed value appears | 5/45 | 4/45 | 2/45 |
| Variable-bound statement (`x = N`, `x is N`, …) of the queried variable | 26 | 33 | 9 |
| … stating the true value | 3 | 20 | 2 |
| Answer slot eligible / ambiguous / absent | 11 / 112 / 5 | 65 / 63 / 0 | 74 / 53 / 1 |
| Slot lead is the true answer | 12 | 108 | 111 |
| Slot lists the true answer | 25 | 120 | 126 |
| Slot already lists the ±1 counterfactual value | 23 | 19 | 18 |

The three Qwen frozen-form hits are the ones the earlier census found: all
state false values inside quoted, speculative mini-narratives.

### Does a description's number move behavior? (P3 natural edits, all 512 rows)

| | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B |
|---|---:|---:|---:|
| Rows whose foreign slot lead differs from the own answer | 440 | 477 | 470 |
| … P3 answers the foreign lead | 31 (7.0%) | **220 (46.1%)** | 35 (7.4%) |
| … P3 keeps its own answer | 123 | 203 | 337 |
| Unrelated-answer baseline: P3 answers a third row's answer | 13/377 | 0/401 | 3/403 |
| Rows whose foreign lead is wrong about its own program | 355 | 60 | 59 |
| … P3 answers that lead / the foreign true answer / its own | 22 / 18 / 106 | 6 / 3 / 47 | 1 / 0 / 50 |

## What this establishes

- **The 0/128 is structural.** Even with case and markdown ignored, the frozen
  forms never occur in 1,024 Gemma descriptions, and Qwen's three occurrences
  are false, quoted narratives. A looser regex would not recover a
  state-statement channel that the text does not contain.
- **Gemma descriptions carry the queried value, as a next-token prediction.**
  It appears in 120/128 and 127/128 receivers against a 13–18 chance
  baseline, including computed values the prompt never shows (44/50, 49/50),
  and it leads the answer slot in 108 and 111 receivers. Qwen's descriptions
  mostly do not carry it (12 leads; containment near its baseline).
- **The other variable's state is absent.** Its computed value appears in
  4/45 and 2/45 Gemma receivers. At this site no text rule can support the
  pilot's wrong-variable control, because there is nothing to edit.
- **An answer-slot rule would clear the 32-group floor only on paper.** 65/128
  (12B) and 74/128 (27B) receivers have a single-candidate slot. But such a
  rule tests a different hypothesis — editing a described *prediction*, not a
  variable's *state* — about half the receivers are ambiguous, and in 18–19
  receivers the ±1 counterfactual is already a candidate.
- **The text channel into behavior differs by family.** Replacing Gemma-12B's
  activation with another group's description-derived direction yields that
  description's lead number 46% of the time (unrelated baseline 0/401). On
  Gemma-27B the patched model mostly keeps its own answer (7%), and on Qwen
  adoption is 7% against a 3.4% baseline.

## What this does not establish

- It relabels nothing: every family's 0/128 coverage and STOP decision stand,
  and the edit hypothesis remains untested.
- A candidate in the answer slot is the AV's guess about the next token. That
  it matches the model's answer is not evidence that the description reads the
  activation, nor that editing it would steer the model.
- P3 natural edits change the whole text, not one number. The subset where the
  foreign lead is wrong (about 60 rows on each Gemma family) is too small and
  too mixed to separate the number from the rest of the description. Whether
  editing only the slot moves behavior needs model forwards; that measurement
  is not part of this report.
- The candidate and statement patterns are regex heuristics chosen after
  reading the text. One task, one site and one pilot split per family; the
  27B AV numbers carry its unauditable BF16 serving cast.
