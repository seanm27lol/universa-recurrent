# Steering assay: do description *differences* steer behavior? (frozen pre-run protocol)

**Status: FROZEN before any steering run, 2026-09-22. Branch `steering-assay`,
based on the closed Phase Two tip `3471d7a`. No steering outcome has been read;
every threshold below is a design choice locked here, not a paper claim and not
a result.**

This is a **new post-Phase-Two measurement**, not a reopening of the closed
pilot. The pilot (`pilot-20260921T235825Z-6164d210`, decision STOP) measured
*preservation* and *edit coverage*: it found P2 absolute reconstruction
behavior-lossy (56.4% vs P0 81.1%), P2≫P3 on correct-answer log probability
(directional signal present), and 0/128 edit-eligible descriptions. All pilot
outcomes stand unchanged. This assay asks a different, weaker question:

> **Does a description difference, turned into a norm-scaled activation
> difference and added to the receiver's activation, produce a *targeted*
> behavioral change — even where absolute reconstruction is lossy?**

The recipe is the NLA paper's reconstructed-difference steering (Planning in
Poetry; Fraser-Taliente, Kantamneni, Ong et al. 2026,
https://transformer-circuits.pub/2026/nla/): for original and edited
descriptions `d_orig`, `d_steered`, take `Δ = AR(d_steered) − AR(d_orig)` and
patch the target's residual stream with

```text
h′ = h + α · ‖h‖₂ · Δ / ‖Δ‖₂
```

at the NLA's layer and token. The paper reports steering success only ~50% of
the time with sometimes-incoherent completions on a much stronger target; our
expectations are calibrated accordingly. The method is the paper's; this assay
is a measurement of it on our released Qwen2.5-7B pair, not a new method and
not evidence of semantic read-out.

## 1. Task, split, receiver

- **Task**: the same paired-program generator (`tasks.py`), unchanged. Example:
  A: `x = 3; y = 8; x = x + 2` vs B: `x = 4; y = 8; x = x + 2`; the affected
  variable's answer is 5 vs 6, the other variable's answer is 8 in both.
- **Split**: the **reused pilot split** — the same 128 groups, same plan hash
  (checked against the pilot run's manifest at run time). This is deliberate:
  the pilot's saved per-row AR direction vectors and P0 behavior enable direct
  comparison. **This is a new measurement on a reused split, not a new split
  and not a pilot rerun; no pilot number is recomputed or amended.** Reuse is
  not leakage here because nothing is selected on steering outcomes — arms,
  controls, templates, the α grid and the criteria are frozen in this document
  before the run, and the run executes exactly once (§7).
- **Receiver rows**: per group, the two side-A rows — the row querying the
  affected variable (`{group}-A-{affected}`, "x-row" in the running example)
  and the row querying the unaffected variable (`{group}-A-{other}`, "y-row").
  256 executed rows. The site is the Phase Two convention: block-20 output,
  last non-padding token of each row's own prompt.

## 2. Arms

- **Arm 1 — AV-difference (pure released-model)**. Per group, from the pinned
  pilot run's saved artifacts (accessed read-only at run time via
  `--pilot-run`, never copied into the repository):

  ```text
  Δ_av = ar_direction({group}-B-{affected}) − ar_direction({group}-A-{affected})
  ```

  i.e. the difference of the AR reconstructions of the group's own saved AV
  descriptions of the two counterfactual activations. **The AV and AR models
  are not loaded for arm 1**; only the saved float32 direction files are read.

- **Arm 2 — oracle-template difference**. Per group, two hand-written template
  descriptions per frozen style (§3) asserting the two values of the affected
  variable (the A value and the B value), passed through the **live AR** (the
  AR model *is* loaded for arm 2):

  ```text
  Δ_oracle,style = AR(template_style(affected, B value)) − AR(template_style(affected, A value))
  ```

  Oracle text is an **intervention instrument**: it asserts the counterfactual
  value on purpose. It is never presented as discovered semantics or as
  something read out of an activation.

Both arms patch identically: `h′ = h + α·n·u(Δ)`, where `h`, `n` are the
receiver row's freshly captured activation and its retained float32 norm,
`u(Δ) = Δ/‖Δ‖₂` (`geometry.unit_direction`; a nonfinite or near-zero Δ fails
closed and counts against the arm, intention-to-test). The patched norm is
*not* restored to `n` — the patch is additive, exactly as in the paper recipe;
the resulting norm and norm ratio are recorded.

## 3. Frozen oracle templates (verbatim)

Two styles, frozen here. `{variable}` ∈ {x, y} and `{value}` ∈ 0..19 are the
only substitutions. Style 2 mimics the AV's observed three-paragraph
"Structured …" register documented in
`reports/pilot_description_analysis.md` (512/512 pilot descriptions begin
"Structured", with a structural opening, a speculative answer paragraph, and a
"Final token" closing paragraph) — the oracle borrows the register so the AR
sees familiar text, while *asserting* the value the AV never asserted.

**Style `terse`** (one sentence):

```text
The current value of {variable} is {value}.
```

**Style `structured`** (three paragraphs, AV register):

```text
Structured math format with code block showing a short sequence of variable assignments and arithmetic on the variables x and y, ending with a question asking for the value of {variable}.

Tracing the assignments in order, the value of {variable} is {value} when the question is asked, so the expected reply is the single integer "{value}".

Final token of the prompt closes the question "What is {variable}?", strongly expecting "{value}" to complete the answer.
```

Templates are rendered deterministically; the manifest records both templates
verbatim plus their sha256, and every rendered per-group text with its sha256.
No template text is tuned after the run.

## 4. Controls (all recorded, all on the full α grid)

- **Wrong-variable oracle δ** (arm-2 specificity control), per style:
  `Δ_wv,style = AR(template_style(other, candidate)) − AR(template_style(other, other value))`
  where `candidate = other value + Δvalue`, `Δvalue = B affected answer − A
  affected answer` (±1 in every accepted group), mirrored (`−Δvalue`) when the
  direct candidate falls outside 0..19, following the pilot's frozen
  wrong-variable convention; out-of-range on both sides is reported, never
  forced. The x-row answer must not move (§5, C3). The y-row is also scored
  against the candidate value.
- **Same-value AV-difference δ** (arm-1 control):
  `Δ_av,ctl = ar_direction({group}-B-{other}) − ar_direction({group}-A-{other})`
  — the description difference across the counterfactual pair on the query
  whose answer does *not* change. It should not move the x-row answer.
- **α = 0**: no-op, implied by the in-run P0 reference row; not separately
  executed.
- **Sign flip**: α = −1 is in the frozen grid, so every δ is also applied in
  the opposite direction.

## 5. Frozen α grid, metrics, and success criteria

**α grid: {−1, 0.5, 1, 2}** (α = 0 implied by P0). Criteria are evaluated at
the **primary α = 1**; the rest of the grid is dose-response evidence, and the
sign flip tests direction dependence.

Per group, per arm/control × α, on the freshly patched receiver rows:

- `flip_to_B`: x-row steered greedy answer terminates and exactly equals the
  group's B-side affected answer (the answer "5 → 6" question).
- `retained_A`: x-row steered greedy answer exactly equals the A-side answer.
- `moved_x`: x-row steered generation text differs from the same run's P0
  x-row generation text. A failed or non-terminating steered generation counts
  as moved; a failed P0 counts the group as moved against the arm
  (conservative).
- `y_intact`: y-row steered generation text equals the same run's P0 y-row
  generation text (agreement — the patch did not disturb the y answer,
  regardless of whether P0's y answer was right). Exact-true-y rates are also
  reported.
- `L = logP(B affected answer) − logP(A affected answer)` on the x-row
  (teacher-forced, same frozen answer tokenization as the pilot); groups with
  incomplete scores are excluded from L means with the exclusion counted.
- Next-token KL vs P0 on both rows (valid cases; failures never manufacture
  finite KL).

All rates are over **all 128 groups (intention-to-test)**: any failure,
missing direction, failed AR reconstruction, or crashed generation counts
against the arm. Uncertainty: whole-group bootstrap, 3,000 resamples, **fixed
seed 205100** (new, distinct from the pilot's 203100).

**Frozen success criteria (design choices, declared pre-run; per arm, arm 2
per style), all at α = 1:**

| ID | Criterion | Threshold |
|---|---|---|
| C1 | flip_to_B rate over all 128 groups | **≥ 0.30** |
| C2 | y_intact rate over all 128 groups, same condition | **≥ 0.90** |
| C3 | moved_x rate of the arm's own control δ (arm 1: same-value AV control; arm 2: same-style wrong-variable oracle) | **< 0.10** |
| C4 (supporting, non-gating) | flip_to_B at α = −1 ≤ flip_to_B at α = 1 | reported |

**An arm shows targeted steering iff C1 ∧ C2 ∧ C3 all hold at α = 1.** C4 is
reported as supporting evidence, never as a gate. The 0.30 flip bar sits below
the paper's ~50% self-reported success (different, stronger target; messy
completions tolerated there — we require exact terminated integer answers);
0.90 integrity permits modest collateral damage from a single-site additive
patch; 0.10 is a conventional specificity ceiling for the control. These are
choices made before seeing any steering outcome.

## 6. Carried-over Phase Two numerics

Unchanged from the pinned pilot: `TARGET_BUCKET = 128` kernel-shape pinning
with the loud pre-inference fit check; the same-length dummy-suffix bitwise
gate inside `score_answer`; `SUFFIX_DRIFT_BOUND = 1e-1` as untouched backstop;
identity gates (repeat/noop/raw_restore at atol/rtol 1e-5) and the P0/P1
greedy identity backstop per executed row; ITT accounting; fresh run
directory with manifest-before-inference, source and protocol hashes, safe
numeric artifacts, reports-only ZIP; independent `audit.py` replay. The
manifest additionally records: both arms, both templates verbatim with sha256,
the α grid, the controls, the frozen criteria verbatim, the seeds, and the
consumed pilot run's **path hash, manifest sha256, plan hash and completion
status** (the pilot path itself is user-local and is never written to the
manifest or the repository).

## 7. Cost projection and stopping rule

Measured pinned-pilot rates (milestone_two.md): target identity ≈ 0.17 s per
forward, behavior ≈ 0.09 s per forward, AR ≈ 0.21 s per forward, greedy
terminates early (≈1–2 tokens). Projected for 128 groups × 2 receiver rows:

| Stage | Projected forwards | Projected time |
|---|---:|---:|
| Target load + identity gates (256 rows) | ≈ 2,500 | ≈ 7 min |
| AR load + arm-2/control reconstructions (8 per group) | 1,024 | ≈ 4 min |
| Target reload + behavior (26 conditions × 2 rows × 128 groups) | ≈ 45,000 | ≈ 70 min |
| Loads, hashing, suite, bundling | — | ≈ 20 min |
| **Total** | | **≈ 1.5–2 h GPU** |

Well inside the inherited eight-hour per-stage budget. If measured throughput
on the first groups projects past four hours, stop and report rather than
shrinking groups, conditions, or the grid mid-run.

**Stopping rule: exactly one run of this frozen protocol; report whatever
happens.** No α re-tuning, no template edits, no arm additions, no
checkpoint/site/layer changes after outcomes are read. A source bug permits
only a regression test plus a visibly versioned rerun, as in Phase Two.

## 8. Honesty limits (carried into every report of this assay)

- Steering rates are **behavioral measurements** on one checkpoint, one task
  family, one site. They are not semantic proofs: a flip toward B does not
  show the description "encoded" the value, and a null result does not show
  the site lacks a representation.
- Arm-1 descriptions come from the released AV; the pilot showed they assert
  no current values (0/128 coverage) and are behavior-lossy under replacement.
  Any arm-1 effect is therefore a *difference*-space measurement, credited to
  the paper's recipe, not to a reading of the activation.
- Arm-2 oracle text is written by us; it is an intervention instrument and is
  never claimed as read-out semantics.
- The NLA paper reports ~50% steering success with messy completions on its
  own target; nothing here imports that rate. Our criteria are local design
  choices.
- The auditor replays saved counts/KL/statistics; it does not authenticate
  execution and does not re-read the pilot run (only its hash is recorded).
- This run reuses the pilot split. The pilot's recorded outcomes stand; this
  assay's outcomes are reported separately and never blended into the closed
  Phase Two record.
