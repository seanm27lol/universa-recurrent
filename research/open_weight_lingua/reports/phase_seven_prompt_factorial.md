# Phase Seven: separating question position from wording — 2026-10-05

```text
x = 7
y = 16
y = y + 1
y = y - 3          x is 7; the program ends on y (14)

original wording:  What is x? Reply with only the integer.
expanded wording:  What is x at the end of this program? Reply with only the integer.
```

Asked the expanded question *before* this program, Gemma-3-12B answers 14,
which is y's value. Asked the original question *after* it, Gemma answers 7.

Phase Six changed two things at once: it moved the question before the
program **and** added "at the end of this program". Its accuracy loss
(Gemma 0.944 → 0.729, Qwen 0.810 → 0.502) could therefore come from position,
wording or both ([review addendum](phase_six_question_first.md)). Phase Seven
([protocol](../protocols/phase_seven_prompt_factorial_v1.md)) crosses the two
factors on the same programs. It also adds a diagnostic that asks the
original question both before and after the program. It measures answers
only: no activations, probes or edits.

**Result.**
- **Qwen2.5-7B:** the whole Phase Six loss comes from position. Moving the
  question first costs about 30 points with either wording, and wording has
  no resolved effect.
- **Gemma-3-12B:** position costs about 10 points, and the expanded wording
  costs about 12 more, but only when the question comes first. That
  interaction is resolved and larger than the protocol's 5-point practical
  reference.

These are **exploratory** results on reused calibration and pilot programs.

## What ran

- **Design.**
  - **Formats:** five, each with the same 1,536 program/question pairs per
    family (384 groups, both sides, both asked variables): the four 2×2
    cells plus `original_repeat`. That is 7,680 answers per family.
  - **Decoding:** greedy, at most 8 new tokens, the pinned 128-token bucket,
    no KV cache.
  - **Stopping:** at any end-of-sequence token the model declares. This is
    the multi-EOS fix: Gemma `{1, 106}`, Qwen `{151643, 151645}`.
  - **Scoring:** each family's frozen convention (`rstrip` Gemma, `raw`
    Qwen).
- **Freeze.**
  - The protocol (`fb4ca286081b…`), code and prepared stimuli were committed
    at `210bffb`. That commit was pushed before any model forward.
  - Launch freeze v1: `7fc2689eda42…`.
- **Reserved validation splits:** never read.

| Family | Campaign / stage | Freeze | Outcome | Wall clock |
|---|---|---|---|---:|
| Gemma-3-12B | v1 `phase-seven-campaign-v1-20261005/gemma` | v1 | **FAILED, 0 rows** (see below) | 30 s |
| Qwen2.5-7B | v1 `phase-seven-campaign-v1-20261005/qwen` | v1 | COMPLETE, 7,680 / 7,680 | 1,780 s (generation 1,672 s) |
| Gemma-3-12B | v1.1 `phase-seven-campaign-v1.1-20261005/gemma` | v1.1 `e681789b2e2f…` | COMPLETE, 7,680 / 7,680 | 3,440 s |

**Operational record**, kept in full because the protocol forbids silent
retries:

- **The v1 Gemma stage failed after 30 seconds, before generating
  anything.**
  - The launch supervisor around the frozen runner stopped its own stage
    when a 375 MiB process from another project appeared on the GPU.
  - The interrupt landed while checkpoint shards were loading. That
    surfaced as `ValueError: could not determine the shape of object type
    'torch.storage.UntypedStorage'`.
  - Every setup step was later re-run in isolation and succeeded: tokenizer,
    re-tokenization of all 7,680 prompts, and model loading on CPU and CUDA.
  - The runner had stored only the error message. The full traceback was in
    the supervisor log. The v1 `completion.json` was later annotated with
    that traceback; its original is preserved beside it as
    `completion.before-traceback.json`.
- **Gemma was rerun as an operational restart, v1.1.**
  - All 144 frozen scientific files (source, tests, stimuli, locks,
    protocol) are byte-identical to v1.
  - Only three things changed: a new campaign directory, a Gemma-only run
    order, and an amended supervisor rule. That rule logs small foreign GPU
    processes instead of stopping the stage, and stops only on memory
    pressure.
  - The v1.1 stage stayed inside the original shared 3-hour deadline.
- **Other GPU activity.**
  - During the Qwen stage, the other project's 372–379 MiB processes
    appeared five times and were logged.
  - A diagnostic model load by the reviewing session (03:31–03:32 UTC)
    delayed Qwen's admission by about 90 seconds.
  - Greedy outputs here do not depend on concurrent GPU use, so only timing
    could be affected. The Phase Six reproduction below confirms it.
- **The traceback-capture fix** was written during the run but applied to
  the source only afterwards, for future runs. The frozen runs used the
  unmodified runner.

**Checks.**
- **Independent rescoring.** All 15,360 saved answers were rescored with
  `answer_text_matches` by `scripts/prompt_factorial_recount.py`. That
  script does not import the frozen evaluator, and it agrees with the stored
  correctness everywhere.
- **Recount against the evaluator.** Its cell counts and all seven contrast
  point estimates equal the frozen evaluator's. Its 3,000-resample
  intervals are within about 0.01 of the evaluator's 10,000-resample ones.
- **Reproduction of Phase Six.** Two formats are exactly Phase Six's
  prompts.
  - Gemma reproduces Phase Six's saved answers byte for byte: 1,536 of
    1,536 in both.
  - Qwen matches except for 13 answers (9 and 4). Phase Six had decoded
    those past `<|endoftext|>`, as in `10\n<|endoftext|>Human Resources…`.
    With the multi-EOS fix they stop at `10\n`.
  - None of the 13 changes score under `raw`.

## Results

Accuracy on the 1,536 base prompts, correct / 1,536:

| Format | Gemma-3-12B | Qwen2.5-7B |
|---|---:|---:|
| Original question, after | **0.944** (1,450) | **0.810** (1,244) |
| Expanded question, after | 0.930 (1,429) | 0.800 (1,229) |
| Original question, before | 0.847 (1,301) | 0.508 (781) |
| Expanded question, before (Phase Six's format) | 0.729 (1,119) | 0.502 (771) |
| Original question, before *and* after (diagnostic) | 0.928 (1,426) | 0.706 (1,085) |

Every Gemma answer terminated. Qwen's terminated in 1,514–1,535 of 1,536
answers per format.

**The five planned contrasts.** These are differences in accuracy on the
same prompts. Each interval is a percentile 99.5% interval from 10,000
whole-group resamples (seed 701100 Gemma, 701101 Qwen), Bonferroni across
the ten model-by-contrast comparisons.

| Contrast | Gemma-3-12B | Qwen2.5-7B |
|---|---|---|
| Position, original wording (before − after) | **−0.097** [−0.130, −0.065] | **−0.301** [−0.345, −0.258] |
| Position, expanded wording | **−0.202** [−0.241, −0.163] | **−0.298** [−0.340, −0.255] |
| Wording, question before (expanded − original) | **−0.118** [−0.155, −0.083] | −0.007 [−0.043, +0.029] |
| Wording, question after | −0.014 [−0.028, −0.001] | −0.010 [−0.024, +0.005] |
| Interaction (position effect, expanded − original) | **−0.105** [−0.145, −0.066] | +0.003 [−0.036, +0.042] |

Bold marks intervals that lie entirely beyond the 5-point reference.

- Gemma's wording-after effect has a resolved sign, but it is smaller than
  5 points.
- Qwen's wording effects and interaction are unresolved. Their intervals
  stay within about ±4.5 points.

**The repeat diagnostic.** This has 95% intervals and is outside the primary
family.

| Original question asked before and after, minus | Gemma | Qwen |
|---|---|---|
| … asked only before | +0.081 [+0.061, +0.103] | +0.198 [+0.168, +0.229] |
| … asked only after | −0.016 [−0.030, −0.002] | −0.104 [−0.128, −0.080] |

Repeating the question after the program almost restores Gemma's accuracy.
It restores only part of Qwen's: an early question still costs Qwen 10
points when the question is repeated.

## Breakdowns (descriptive)

**By the program's last assignment.** Each program is asked about both
variables, so half the prompts end on an assignment to the asked variable.
The table gives accuracy, and in parentheses how often answers equal the
other variable's final value when it differs. The parentheses come from the
recount script.

| Last line assigns | Gemma, original after | Gemma, original before | Gemma, expanded before | Qwen, original after | Qwen, original before | Qwen, expanded before |
|---|---|---|---|---|---|---|
| The asked variable (768) | 0.923 | 0.896 | 0.918 | 0.759 | 0.650 | 0.667 |
| The other variable (768) | 0.965 (0.1%) | **0.798 (11.3%)** | **0.539 (35.2%)** | 0.861 (2.0%) | **0.367 (19.1%)** | **0.337 (25.4%)** |

The loss from a question placed first concentrates where the program ends
on the other variable, in both models.
- **Gemma:** the expanded wording triples the other-value answers (11.3% to
  35.2%). That is the source of Gemma's interaction.
- **Qwen:** the pattern is already present with the original wording.
- **Hardest cell:** asking for x when the program ends on y. With the
  question first, Gemma scores 0.789 (original) and 0.377 (expanded) on
  those 398 prompts. Qwen scores 0.259 and 0.168.

**By the asked variable's last update.**

| Last update of the asked variable | n | Gemma: original after / before / expanded before | Qwen: original after / before / expanded before |
|---|---:|---|---|
| A literal (`x = 7`) | 706 | 0.987 / 0.816 / 0.550 | 0.918 / 0.344 / 0.422 |
| Arithmetic (`x = x + 2`) | 674 | 0.895 / 0.866 / 0.875 | 0.691 / 0.610 / 0.540 |
| A copy (`x = y`) | 156 | 0.962 / 0.904 / 0.904 | 0.833 / 0.814 / 0.699 |

A question placed first hurts most when the asked variable was last set by a
literal. That is also when the other variable is more often the last writer.

**By asked variable.**
- **Gemma** with the question first: x 0.846 and y 0.848 with the original
  wording, x 0.647 and y 0.810 with the expanded wording.
- **Qwen** with the original wording first: x 0.444 and y 0.573. With it
  after: 0.828 and 0.792.

The evaluator's summary also has the full per-cell counts: the joint
asked-variable × last-writer table, operation subtypes, source split and
termination strata.

## What this establishes

- **Phase Six's loss had different sources in the two models.**
  - For Qwen it is a **position** effect, with wording ruled out within
    about ±4.5 points.
  - For Gemma it is a **position** effect of about 10 points plus a
    **wording × position interaction** of about 10 points. "At the end of
    this program" hurts only when it precedes the program, where it pulls
    answers toward the last-written variable.
  - The Phase Six report's caveat about this phrase was therefore correct
    for Gemma.
- **A question placed first is a weaker cue on this task** for both models,
  especially when the program ends on the other variable. Restating it
  after the program largely removes the problem for Gemma, but not for
  Qwen.

## What this does not establish

- **Behaviour, not mechanism.** These contrasts concern answer accuracy
  under these prompt formats. They do not show where, whether or how the
  models store variable values. Errors that match the other variable's value
  do not prove the asked value was unavailable internally.
- **Exploratory, reused data.** The programs have informed every phase since
  Phase Two. A confirmatory test would need data the analysis has not seen.
- **One pair of wordings, two models, one task family** (integers 0–19,
  short programs). The interaction concerns this particular phrase.
- **Gemma's estimate comes from an operational restart.** It used the same
  frozen files and no rows were generated before the failure. It is still a
  second launch, recorded as such.

## Next

The exploratory results now name specific contrasts worth confirming:
- Gemma's wording × position interaction;
- Qwen's position effect;
- the other-variable answers when the program ends on the other variable.

Following the review, freeze those contrasts and their analysis first. Only
then open `validation_a` (256 unseen groups, 1,024 prompts per format) for a
confirmatory 2×2. `validation_b` stays reserved.

Measured generation cost: 0.425 s per answer on Gemma (3,263 s for 7,680)
and 0.218 s on Qwen (1,672 s). A four-format validation_a run is 4,096
answers per family: about 29 minutes of generation for Gemma and 15 for
Qwen. Loading and hash checks add about 3 minutes per family.

Code:
- [prompt_factorial.py](../src/open_weight_lingua/prompt_factorial.py)
  (stimuli);
- [prompt_factorial_run.py](../src/open_weight_lingua/prompt_factorial_run.py)
  (runner and frozen evaluator);
- `scripts/prompt_factorial_recount.py` (independent recount).
