# Post-hoc answer-lens re-analysis of the closed pilots — 2026-09-27

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. This page is a
**post-hoc, descriptive** re-read of saved generations under one declared lens —
it changes no frozen outcome. The Gemma-12B pilot's usability-gate failure was
traced to the instrument, not the model (failure taxonomy below): the mirror
target answers `digits + "\n" + <end_of_turn>`, and the frozen Qwen-era metric
(`exact_integer`, no stripping) rejects the trailing newline. The lens asks only
what the closed pilots' numbers look like after stripping trailing ASCII
whitespace (Python `str.rstrip()` semantics: space, `\t`, `\n`, `\r`, `\f`,
`\v`). Nothing is recomputed from weights; no run is relabeled.

## The instrument-fix evidence (12B pilot failure taxonomy, 512 variants)

From `pilot-20260923T043613Z-f7e71d7b` saved generations: correct 266 (0.5195);
**trailing-whitespace rejections 228 (0.4453), of which 221 had the correct
value after stripping**; wrong value 13; other format 4 (three negative answers
like `-6`, which can never be in range); near-miss 1; non-terminated 0. The Qwen
pilot has **zero** whitespace-affected rows (verified again here), so its frozen
numbers stand untouched. No accuracy gradient by program length (3–6
statements: 0.35/0.66/0.53/0.61), no x/y or A/B asymmetry beyond noise. A
difficulty probe (`/tmp/gemma12_taskprobe/`, 64 fresh prompts per config,
probe seeds 900001+, disjoint from all split programs) then showed content
accuracy 0.98–1.00 across difficulty levels with frozen-metric accuracy
*falling* on easier prompts (more trailing newlines) — difficulty is not the
driver.

## Lens tables (saved generations, rstrip-only)

### Gemma-3-12B pilot (`pilot-20260923T043613Z-f7e71d7b`)

| Condition | Lens exact (of 512) | Lens accuracy | Lens agreement with P0 | (Frozen accuracy) |
|---|---:|---:|---:|---:|
| P0 | 487 | 0.9512 | 1.000 | 0.5195 |
| P2 | 478 | 0.9336 | 0.945 | 0.5527 |
| P3 | 232 | 0.4531 | 0.451 | 0.3223 |
| P4 | 488 | 0.9531 | 0.998 | 0.5273 |
| P5 | 0 | 0.0000 | 0.000 | 0.0000 |
| donor | 480 | 0.9375 | 0.977 | 0.5312 |
| calibration_median_norm | 478 | 0.9336 | 0.949 | 0.5664 |

Whole-group (ITT) correctness under the lens: P0 0.8750, P2 0.8203. Paired P2
accuracy loss under the lens (frozen bootstrap reused: 3,000 resamples, seed
203100): **point 5.47 pp, one-sided upper 10.16 pp** — above the frozen 5 pp
limit.

### Gemma-3-27B pilot (`pilot-20260926T185550Z-d085d53c`; AV served BF16-cast)

| Condition | Lens exact (of 512) | Lens accuracy | Lens agreement with P0 | (Frozen accuracy) |
|---|---:|---:|---:|---:|
| P0 | 494 | 0.9648 | 1.000 | 0.2539 |
| P2 | 488 | 0.9531 | 0.967 | 0.2266 |
| P3 | 367 | 0.7168 | 0.725 | 0.2227 |
| P4 | 494 | 0.9648 | 0.996 | 0.2617 |
| P5 | 3 | 0.0059 | 0.004 | 0.0039 |
| donor | 492 | 0.9609 | 0.984 | 0.2656 |
| calibration_median_norm | 489 | 0.9551 | 0.969 | 0.2305 |

Whole-group (ITT) correctness under the lens: P0 0.8984, P2 0.8750. Paired P2
accuracy loss under the lens: **point 2.34 pp, one-sided upper 6.25 pp** — above
the frozen 5 pp limit.

### Qwen2.5-7B pilot (`pilot-20260921T235825Z-6164d210`, read-only control)

Zero whitespace-affected rows; the lens reproduces the frozen record exactly
(P0 415/512 = 0.8105; P2 289/512 = 0.5645; group P0 0.5469 / P2 0.2031; loss
point 34.38 pp, upper 41.41 pp — identical to the closed phase's frozen
statistics). This validates the lens harness and leaves the Qwen outcome
untouched.

## What the lens does and does not say

Under the corrected answer instrument, both Gemma pilots' usability gate moves
from failed to passed (P0 0.951 / 0.965) — and their preservation criterion
moves from met to **not met** (P2-loss uppers 10.16 pp and 6.25 pp against the
5 pp limit), aligning them with the usable-task Qwen outcome instead of
contradicting it: the earlier "preservation passes" were floor effects of the
format rejections. The edit-coverage result is text-independent and unchanged
(0/128 on both). These are post-hoc recomputations over saved evidence; the
closed pilots' frozen decisions (STOP on each family) stand, because thresholds
are frozen per phase and instruments are not amended retroactively. The lens
exists to design the confirmation run: a frozen metric amendment
(`protocols/gemma_answer_convention.md`) and a fresh 12B confirmation pilot
computed under it. Any number the confirmation run reports supersedes the
corresponding lens preview; the lens numbers themselves are descriptive.
