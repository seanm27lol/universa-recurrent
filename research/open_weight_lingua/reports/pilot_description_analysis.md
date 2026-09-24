# Post-hoc analysis of pilot AV descriptions — what the 512 saved descriptions contain

**Frame.** This is a **post-hoc, descriptive, not preregistered** analysis of saved
Phase Two pilot evidence (run `pilot-20260921T235825Z-6164d210`, branch
`phase-two-milestone-two`). Its purpose is to describe what the AV's 512 saved
descriptions actually say, to ground the design of any **future** frozen edit rule or
future phase. It does **not** relabel, amend, or reinterpret the completed pilot's
recorded outcomes: the pilot found **0/128 edit-eligible groups** (all "absent") under
the frozen rule `text_edits` v1.0.0 and recommended **stop**; that stands. Every
"wider rule" number below measures what a hypothetical future rule *could have
matched*; any such rule must be **frozen before that future phase's own pilot**, and
matching a statement in a description is never evidence that the statement is true of
an activation. All evidence is local; the raw analysis script and full dumps live only
in `/tmp/owl_desc_analysis/` (not committed).

## Data and method

- `runs/pilot-20260921T235825Z-6164d210/results.json` (512 rows = 128 groups x 4
  variants) joined by row `id` to `manifest.json` `inputs` (prompt, answer, variable,
  side, affected). All 512 `description.status` are `ok`.
- True final values of `x` and `y` were recomputed by parsing each manifest prompt
  program and running `open_weight_lingua.tasks.interpret`. Sanity check: the
  interpreted value of the queried variable reproduces the manifest `answer` for
  **512/512** rows.
- Re-running the frozen parser on the 128 rows that carry a saved `edit.parse`
  reproduces all **256/256** saved per-variable entries exactly (0 mismatches), so the
  post-hoc re-parse below is consistent with the pilot's own accounting.

## 1. Description census

**Shape.** Every description is exactly three paragraphs; 512/512 begin with the word
"Structured"; the injection position is 111 for every row.

| Length | min | p25 | median | mean | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| Characters | 477 | 561 | 577 | 577.7 | 594 | 646 |
| Words | 83 | 92 | 95 | 94.7 | 97 | 104 |
| AV tokens (`description.token_ids`) | 128 | 139 | 142 | 141.0 | 143 | 147 |

**Variable mentions** (standalone token `x` / `y`, alnum-guarded):

| Mention | rows | fraction |
|---|---:|---:|
| x only | 231 | 45.1% |
| both x and y | 55 | 10.7% |
| y only | 0 | 0.0% |
| neither | 226 | 44.1% |

**Phrasing census** (rows with at least one occurrence; total occurrences in
parentheses). Occurrences are counted regardless of quoting or truth.

| Pattern | x rows (occ) | y rows (occ) |
|---|---:|---:|
| `x = N` (assign-like) | 97 (108) | 32 (34) |
| `x is N` | 49 (55) | 2 (2) |
| `x is now N` | 3 (3) | 0 |
| `x is currently N` | 0 | 0 |
| `the current value of x is N` | 0 | 0 |
| `value of x is now N` | 1 (1) | 0 |
| `x becomes N` | 0 | 1 (1) |
| `x equals N` | 0 | 0 |

Keyword rows: `currently` **0/512**; `initial`/`initially` 192/512; `final`/`finally`
512/512; `result` 484/512. A digit appears within 12 characters of a standalone `x` in
198 rows, of `y` in 43 rows. Backtick-quoted program-like lines (e.g. `` `x = 'apple'` ``)
appear in only 13/512 rows.

**Frozen rule re-run over all 512 descriptions (post-hoc):** x: 509 absent, 3
eligible, 0 ambiguous; y: 512 absent. The pilot's own eligibility accounting operated
only on the 128 side-A affected-query rows (one pair per group) and recorded
**128/128 absent for the affected variable → 0/128 eligible groups**; exactly one
saved non-absent parse exists in the run: **pilot-0077-A-y**, x eligible with value 10
via the `is-now` template (the affected variable y is absent, so the group is not
eligible). The post-hoc re-parse finds two further x-eligible rows that were never in
the pilot's eligibility path: `pilot-0019-B-y` (side B; no edit record) and
`pilot-0125-A-x` (side-A control row; minimal edit record). All three hits are the
"x is now N" form, and **all three parsed values are false of their programs**
(10 vs true x = 6; 3 vs true x = 0; 3 vs true x = 13).

Verbatim description of the pilot's single x-eligible row `pilot-0077-A-y`:

> Structured math format with code blocks showing sequence operations, implying a pattern or answer to a question about variable values after an operation.
>
> The answer " "12" after the operation 5+7 returns" is given with a second attempt showing "The value of x is now 10," suggesting the final answer mirrors the initial value check, implying "5" or similar.
>
> Final token "value
> " closes a numeric answer ("Result: 5"), part of a concluding example check ("x = 12..."), strongly expecting "5" or "12" to complete the answer, likely followed by "12" or "not changed" to close the clause.

The frozen rule's hit comes from the substring "x is now 10" inside a *quoted,
speculative* mini-narrative ("a second attempt showing ..."), not an assertion.

## 2. Correctness census (post-hoc; what a wider future rule could have matched)

For each row, we checked whether the description states the **true** final value of
the queried variable (and of the other variable) in any of the eight phrasings above.

| Role | Any value statement present (any value, right or wrong) | States the TRUE value |
|---|---:|---:|
| Queried variable | 106/512 (20.7%) | **16/512 (3.1%)** — `x is N`: 10, `x = N`: 6 |
| Other variable | 66/512 (12.9%) | **6/512 (1.2%)** — `x = N`: 5, `x is N`: 1 |

**Hypothetical widened rule on the pilot's own 128 coverage pairs** (side-A
affected-query rows; a "hit" is any of the eight phrasings on the affected variable):

| Rule | Unambiguous single-hit groups | Notes |
|---|---:|---|
| Frozen v1.0.0 (three phrasings) | **0/128** | the recorded pilot outcome |
| + `x = N`, `x is N`, `x becomes N`, `x equals N` | **20/128 (15.6%)** | 24/128 have any hit; 4/128 ambiguous (multiple hits); of the 20 single hits, **1/20 states the true value** (`pilot-0079-A-y`, `y=3` inside a quoted illustrative snippet `"x=5" and "y=3"`) |

Single hits by phrasing: `x = N` 17, `x is N` 3. 15.6% would still fall short of the
pilot's frozen 32-group (25%) eligibility floor, and 19 of the 20 matched statements
are false of the program — so this measures *syntactic* match potential only, not
usable edits. This table is descriptive of saved text; it is not a proposed rule.

## 3. Reconstruction quality

All 512 reconstructions have status `ok` and all six conditions completed (`ok` 512/512
each). Cosine similarity of the AR reconstruction:

| min | p25 | median | mean | p75 | max |
|---:|---:|---:|---:|---:|---:|
| 0.5467 | 0.8193 | 0.8411 | 0.8283 | 0.8533 | 0.8774 |

Cosine vs P2 behavioral preservation (P2 = own description → AR direction + original
norm):

| Split | n | mean cosine | median |
|---|---:|---:|---:|
| P2 generation == P0 generation | 315 (61.5%) | 0.8376 | 0.8441 |
| P2 generation != P0 generation | 197 | 0.8134 | 0.8333 |
| P2 exact answer correct | 289 (56.4%) | 0.8387 | 0.8442 |
| P2 exact answer wrong | 223 | 0.8147 | 0.8323 |

Cosine carries only a weak signal for behavioral preservation (~0.024 mean gap between
preserved and broken rows, heavily overlapping distributions; the cosine range is
compressed into [0.55, 0.88]).

P4 (generic PCA direction under the no-more-than-budget rule) vs P2 agreement:

| Pattern | rows |
|---|---:|
| P4 text == P0 text | **495/512 (96.7%)** |
| P2 text == P0 text | 315/512 (61.5%) |
| P4 text == P2 text | 312/512 (60.9%) |
| Both == P0 | 308 |
| P4 == P0 only | 187 |
| P2 == P0 only | 7 |
| Neither == P0 | 10 |

Exact-answer accuracy from saved generations (P0/P2/P3 match the pilot's summary;
P4/donor/P5 computed here the same way): P0 0.8105, P2 0.5645, P3 0.2676, P4 0.8027,
P5 0.0527, donor 0.7852. The language-description direction (P2) loses ~25 accuracy
points relative to P0, while the generic PCA direction (P4) stays within ~1 point —
on this site, the description-derived direction is behaviorally much further from the
identity than a budget-matched generic direction, consistent with the pilot's
P2-versus-P3 log-prob finding and its stop decision.

## 4. Style clusters

Normalized first-line clusters (lowercased, punctuation stripped):

| First-line prefix | rows |
|---|---:|
| "structured math format with co[de block]" | 157 |
| "structured math format with nu[mbered steps]" | 78 |
| "structured math expression for[mat]" | 74 |
| "structured python code format" | 68 |
| "structured format with numbere[d steps]" | 54 |

The top five clusters cover 431/512 (84%); every description begins "Structured" and
follows the same three-paragraph template: (i) a structural guess at the context,
(ii) a quoted speculative "answer" continuation, (iii) a closing paragraph about the
final token (348/512 contain the literal `Final token "value` opener). The AV freely
narrates variables that do not exist in any prompt — backticked single-letter mentions:
`` `x` `` 133, `` `n` `` 66, `` `b` `` 45, `` `a` `` 24, `` `y` `` 16 occurrences.
`milestone_two.md` records only aggregate smoke/pilot tables and quotes no AV
description verbatim, so the style evidence above comes from the pilot's saved
descriptions themselves; the dominant style is the "Structured math …" meta-narrative
visible throughout the run artifacts.

Representative verbatim excerpts (complete paragraphs; truncation marked with [...]):

`pilot-0000-A-y` (cluster "structured math format with co", n=157):

> Structured math format with code block and variable output pattern, showing a sequence of operations with "x=5" and "y=4" implying a result check.
>
> The answer "Since the operation does not change x, the value of x is" mirrors the question's answer format, strongly implying the final answer is "4" to confirm the initial value of x. [...]

`pilot-0001-A-y` (cluster "structured python code format", n=68):

> Structured Python code format with formatted output using `print` statements showing sequence operations, establishing a pattern of results for `a` and `b`.

`pilot-0002-B-y` (cluster "structured format with numbere", n=54):

> Structured format with numbered steps and variable assignments in a math expression context, showing "5-3" calculation.

`pilot-0079-A-y` (the single widened-rule hit that is true; note the value appears only
inside a quoted illustrative snippet):

> Structured math format with code block and variable output pattern, showing a sequence of operations with "x=5" and "y=3" results.

The style is structural and speculative — the AV describes the *shape* of the context
and guesses which numbers might complete the answer ("strongly expecting "5" or "12""),
rather than asserting the current value of `x` or `y`.

## 5. Would modest phrasing widening have produced meaningful coverage on this site/task?

Syntactically a little, semantically no. Widening the frozen three-phrasing rule to
also accept `x = N` and `x is N` would have moved group coverage from 0/128 to 20/128
unambiguous single-statement groups (15.6%) — still below the pilot's frozen 32-group
floor — with 4 more groups ambiguous. But only 1 of those 20 statements is true of the
program; across all 512 rows only 16 descriptions state the queried variable's true
final value in any of eight phrasings (3.1%); and the three descriptions that satisfy
even the *current* frozen rule all state false values inside quoted, speculative
mini-narratives (e.g. pilot-0077-A-y's "a second attempt showing 'The value of x is
now 10'" where the true x is 6). 226/512 descriptions never mention `x` or `y` as
standalone tokens at all, and the AV narrates nonexistent variables (`` `n` ``,
`` `b` ``, `` `a` ``: 135 backticked occurrences, versus 16 for `y`). The coverage
failure is therefore not primarily a phrasing mismatch: on this site and task the AV
does not verbalize per-variable current values — it produces structural meta-commentary
and answer-shape guesses. Any future edit rule would need different description
behavior (different AV prompting, model, site, or task), must be frozen before that
future phase's own pilot, and nothing here changes the completed pilot's recorded
0/128 outcome.

## Artifacts

- Analysis script: `/tmp/owl_desc_analysis/analyze.py` (not committed)
- Full numeric dump: `/tmp/owl_desc_analysis/full_dump.json`; verbatim excerpts:
  `/tmp/owl_desc_analysis/excerpts.txt`; stdout: `/tmp/owl_desc_analysis/run_stdout.txt`
- Source evidence: `research/open_weight_lingua/runs/pilot-20260921T235825Z-6164d210/`
  (`results.json`, `manifest.json`, `summary.json`, `report.md`) — untouched by this
  analysis.
