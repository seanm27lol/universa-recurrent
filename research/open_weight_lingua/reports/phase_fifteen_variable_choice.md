# Phase Fifteen: question-first errors are the late read choosing the wrong statements — exploratory — 2026-10-12

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

**What was known.**
- **Phases Six and Seven.** Asking the question before the program costs
  accuracy (Gemma-3-12B 0.944 → 0.847, Qwen2.5-7B 0.810 → 0.508). The loss
  concentrates on programs that end on the *other* variable, and wrong
  answers there are often that variable's value.
- **Phase Fourteen.** The question order does not change which statements
  the late read uses.

Phase Fifteen ([brief](../protocols/phase_fifteen_variable_choice_brief.md))
asked whether the question-first errors come from the late read taking the
wrong statements. It hid candidate statements from the post-program
positions, at every layer:
- the other variable's lines (`block_other`);
- the asked variable's superseded lines, before a literal final assignment
  (`block_superseded`);
- or both (`block_both`).

**They do.** With the question first, on programs that end on the other
variable, hiding that variable's lines restores question-after accuracy
completely:
- Gemma: 0.813 → 0.971, against 0.974 with the question after.
- Qwen: 0.393 → 0.879, against 0.877.

Answers equal to the other variable's value fall to the question-after rate
(Gemma 14.6% → 4.2%, Qwen 24.1% → 4.6%). Qwen also takes superseded values
of the asked variable. Hiding those as well brings it to 0.971 on its
literal finals, above its question-after accuracy (0.896).

With the question after, the same blocks cost nothing.

Four of the five predictions held. The fifth failed as worded, for Qwen,
because accuracy *exceeded* the predicted band. These are exploratory
results on opened programs.

## What ran

- **Freeze.** The brief, module, tests and launcher were committed and
  pushed at `aa14cd3` before any forward. Both manifests record that commit
  and the brief's hash `0a8b51368fc9…`. All 112 source hashes equal the
  files at `aa14cd3`.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, each in both orders, as in Phase Fourteen: 3,072 per family.
- **Strata.**
  - Main stratum: programs whose last line assigns the other variable,
    excluding 12 prompts per order whose answer passes through a copy of
    the other variable. That leaves 758 per order.
  - Superseded literals: a literal final assignment with an earlier one.
    280 per order.
- **Scoring.** Teacher-forced, for the answer, the other variable's final
  value and the asked variable's previous value. The bitwise no-op control
  held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p15-choice-gemma3-12b-20261012T172610Z-99772208` | `894e14ed…` / `578dbe31…` / `6d50072a…` | 4,047 s / 4,195 s |
| Qwen2.5-7B | `p15-choice-qwen2.5-7b-20261012T184537Z-00a46137` | `a4a66e90…` / `783e52ea…` / `5ed141a1…` | 2,295 s / 2,382 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB).

## Results

**Main stratum: the program ends on the other variable** (758 prompts per
order). The cells give accuracy, with the rate of saying the other
variable's final value in parentheses.

| Order | Gemma baseline | Gemma `block_other` | Qwen baseline | Qwen `block_other` |
|---|---|---|---|---|
| Question after | 0.974 (4.2%) | 0.980 (4.2%) | 0.877 (5.4%) | 0.897 (4.9%) |
| Question first | **0.813 (14.6%)** | **0.971 (4.2%)** | **0.393 (24.1%)** | **0.879 (4.6%)** |

**Superseded literals** (280 prompts per order). The cells give accuracy,
with the rate of saying the asked variable's previous value in parentheses.

| Order | baseline | `block_other` | `block_superseded` | `block_both` |
|---|---|---|---|---|
| Gemma, question after | 0.968 (1.7%) | 0.982 | 0.996 (0.0%) | 1.000 |
| Gemma, question first | 0.957 (3.5%) | 0.964 | 1.000 (0.0%) | 1.000 |
| Qwen, question after | 0.896 (6.6%) | 0.932 | 0.950 (0.0%) | 0.986 |
| Qwen, question first | **0.514 (30.6%)** | 0.621 | **0.757 (3.5%)** | **0.971** |

- When Qwen's superseded-literal program ends on the asked variable (208
  prompts), its question-first errors are mainly the previous value.
  Hiding the superseded lines takes accuracy from 0.587 to 0.885.
- When the program ends on the other variable (72 prompts), both kinds of
  error occur. Only `block_both` restores it: 0.306 → 0.972.

**Programs whose answer depends on the other variable** (12 per order,
through a copy). `block_other` removes the answer, as expected (Gemma
0.417 → 0.000 with the question after).

**Predictions.**
1. The no-op control is exact. *Held.*
2. With the question after, `block_other` and `block_superseded` each cost
   at most 3 points. *Held:* both raise accuracy slightly. `block_other`
   changes it by +1.4 (Gemma) and +2.0 (Qwen) points on the 1,524
   independent prompts. `block_superseded` changes it by +2.9 and +5.4 on
   its prompts.
3. With the question first, on the main stratum, `block_other` raises
   accuracy by at least 10 points and at least halves the other-value rate.
   *Held:* +15.8 points (Gemma) and +48.5 (Qwen); the rate fell from 14.6%
   to 4.2% and from 24.1% to 4.6%.
4. It closes at least half of the gap to the question-after baseline.
   *Held:* 98% (Gemma) and 100% (Qwen).
5. With the question first, `block_both` on superseded literals is within
   5 points of the question-after baseline. *Held for Gemma* (1.000 against
   0.968). *Failed as worded for Qwen:* 0.971 against 0.896 is 7.5 points
   *above*. Like Phase Fourteen's prediction 3, it was meant to rule out a
   shortfall, and none occurred. Future predictions of this kind will be
   one-sided.

## What this establishes (exploratory)

- **The question-first accuracy loss is a selection failure in the late
  read, not missing information.**
  - The asked variable's statements still carry what the answer needs:
    hide the distractors, and the post-program positions read the right
    answer as often as with the question after.
  - What fails with the question first is the read's *choice*. It takes
    the other variable's statements (both models) or the asked variable's
    superseded ones (mainly Qwen).
- **Both selection rules of the account weaken together.** The account has
  the read pick the asked variable's statements (Phase Nine) and the most
  recent intact one among them (Phase Twelve). With the question first,
  both choices become unreliable, Qwen's much more than Gemma's.
- **A likely reason, untested.** With the question after, the asked
  variable's name sits among the post-program positions that do the
  reading. With it first, the only post-program positions are the chat
  template's, and the name is far behind the program. Phase Seven's repeat
  diagnostic fits this: asking again after the program restores most of
  the accuracy.

## What this does not establish

- **Confirmation.** Opened programs only.
- **Normal operation.** Hiding statements at every layer is an
  off-distribution intervention. It shows the read *can* answer without the
  distractors, not how it weighs them normally.
- **Why selection weakens.** The distance to the question is a
  hypothesis, not a result.
- **Scope.** One wording; two models.

Code:
[variable_choice_knockout.py](../src/open_weight_lingua/variable_choice_knockout.py),
launched with `scripts/run_variable_choice_knockout.sh`.
