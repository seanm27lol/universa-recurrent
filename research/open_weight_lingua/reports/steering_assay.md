# Steering assay closeout — 2026-09-22

The single frozen reconstructed-difference steering run
**`steering-20260922T160432Z-d3b9e4d7` COMPLETED** (128/128 groups, 256
receiver rows, 0 failed/skipped, independent auditor PASS) and **all three
arms failed all frozen criteria**: `successful_arms: []`. Per the frozen
protocol's stopping rule (one run, report whatever happens) this assay is
**closed with a negative result**. The closed Phase Two pilot's recorded
outcomes stand unchanged; this was a new measurement on the reused pilot
split, not a pilot rerun.

## Question and frozen protocol

Does a description *difference* — the pilot's own saved AV descriptions
(arm 1) or frozen oracle templates asserting the counterfactual values
(arm 2) — produce *targeted* behavioral change under the NLA paper's
reconstructed-difference steering recipe (`Δ = AR(d_steered) − AR(d_orig)`,
`h′ = h + α·‖h‖·Δ/‖Δ‖` at the block-20 last-prompt-token site), even where
absolute reconstruction is lossy? Arms, controls (wrong-variable oracle δ per
style, same-value AV-difference δ, α=0 implied by P0, sign flip at α=−1), the
α grid {−1, 0.5, 1, 2}, metrics, and the success criteria below were frozen
in [protocols/steering_assay_brief.md](../protocols/steering_assay_brief.md)
on 2026-09-22 **before the run and never weakened after reading outcomes**.
They are design choices, not paper claims.

## Decision: every criterion unmet at the primary α = 1 (128 groups, ITT)

Copied from the run's `completion.json`/`summary.json` (whole-group bootstrap,
3,000 resamples, seed 205100; intervals in the run's `summary.json`):

| Arm | C1 flip_to_B ≥ 0.30 | C2 y_intact ≥ 0.90 | C3 control moved_x < 0.10 | C4 (supporting, non-gating) | Success |
|---|---|---|---|---|---|
| av_difference | **0.0781 — unmet** | **0.5703 — unmet** | **0.4453 — unmet** | met (0.0781 at α=1 vs 0.0078 at α=−1) | **no** |
| oracle_terse | **0.0313 — unmet** | **0.4219 — unmet** | **0.4844 — unmet** | met by the frozen ≤ rule at equality (0.0313 vs 0.0313): no direction dependence | **no** |
| oracle_structured | **0.0234 — unmet** | **0.3594 — unmet** | **0.5625 — unmet** | met (0.0234 vs 0.0156) | **no** |

C4 is supporting evidence only and never gates; only av_difference and
oracle_structured show even the weak direction dependence it measures, and
oracle_terse's flip rate is identical under sign flip. The headline is
unchanged either way: **no arm came close to any gating criterion.**

## Dose response across the frozen α grid (point rates; disruption grows, targeting does not)

| Condition | α=−1 flip / intact | α=0.5 flip / intact | α=1 flip / intact | α=2 flip / intact |
|---|---|---|---|---|
| av_diff (arm 1) | 0.0078 / 0.4453 | 0.0234 / 0.8125 | 0.0781 / 0.5703 | 0.0156 / 0.2891 |
| av_diff_control | 0.0469 / 0.4531 | 0.0312 / 0.7734 | 0.0234 / 0.5000 | 0.0469 / 0.2578 |
| oracle_terse (arm 2) | 0.0312 / 0.5781 | 0.0156 / 0.9062 | 0.0312 / 0.4219 | 0.0234 / 0.1562 |
| wrongvar_terse | 0.0234 / 0.4766 | 0.0000 / 0.8594 | 0.0156 / 0.5234 | 0.0391 / 0.1875 |
| oracle_structured (arm 2) | 0.0156 / 0.4609 | 0.0156 / 0.7578 | 0.0234 / 0.3594 | 0.0156 / 0.1484 |
| wrongvar_structured | 0.0156 / 0.4219 | 0.0156 / 0.7266 | 0.0234 / 0.3516 | 0.0391 / 0.1641 |

At α=1 the intended deltas move the x-row answer off P0 in 43–63% of groups
(av_diff 0.4297, oracle_terse 0.4766, oracle_structured 0.6328) while the
matched control deltas move it in 44–56% (av_diff_control 0.4453,
wrongvar_terse 0.4844, wrongvar_structured 0.5625) — the controls disturb the
target answer about as much as the intended interventions. Flip rates toward
B stay in 0–8% at every strength, including the controls', while y-integrity
collapses monotonically with |α| (e.g. oracle_terse 0.9062 → 0.4219 → 0.1562
from α=0.5 to 2) and mean next-token KL rises steeply (oracle_terse 0.15 at
α=0.5 → 4.35 at α=1 → 9.16 at α=2; oracle_structured up to 13.50 at α=2).

## Continuous measures: L and KL

L = logP(B answer) − logP(A answer) on the x-row, valid for all 128 groups in
every condition. P0's mean L is **−16.08** (computed post-hoc from the run's
saved P0 answer scores): the unpatched model overwhelmingly prefers A's
answer. Steering shifts L toward B (e.g. av_diff@1 **−9.29**,
oracle_terse@1 **−9.00**, oracle_structured@1 **−7.15**) — but the matched
controls shift it comparably (av_diff_control@1 −10.16, wrongvar_terse@1
−11.23, wrongvar_structured@1 −9.88), so the preference shift is as
non-specific as the greedy flips. The continuous and discrete measures agree:
**a real but indiscriminate push, not targeted control.**

## Gates, cross-run determinism, and resources

* Identity gates bitwise on all 256 rows; greedy identity backstop **exact
  256/256** in both stages; P0/P1 references valid 256/256; every
  `suffix_drift_relative` measurement exactly 0.0 under kernel-shape pinning.
* Cross-run versus the pinned pilot's saved records (descriptive, never a
  gate): fresh P0 generations match the pilot's saved P0 records **256/256**,
  and fresh captured originals are bitwise equal to the pilot's saved
  originals **256/256** — the pinned regime reproduces the pilot's P0
  behavior exactly on this machine.
* Measured resources: wall 6,114.3 s through bundle (101.9 min; setup 409.0 s
  recorded separately) — inside the frozen 1.5–2 h projection; stages: target
  identity 323.6 s (2,496 forwards), AR 134.3 s (1,024 forwards), behavior
  5,596.2 s (59,275 forwards), fetch/hash verification 49.2 s; GPU peak
  15.91 GB allocated / 16.11 GB reserved.
* Independent auditor: **PASS** — it replays saved counts, next-token KL, the
  frozen statistics and the decision, and verifies the manifest's frozen
  criteria and oracle templates against the audited code; it cannot replay
  AR/target forwards and cannot re-read the pilot run (only its recorded
  hashes).

## Reading (bounded negative result)

At this site, on this task family, with this released AV/AR pair,
reconstructed-difference steering is **non-specific disruption, not targeted
control**: intended deltas flip the affected answer in 2–8% of groups while
disturbing the unaffected answer in 43–64%, wrong-variable and same-value
control deltas move the affected answer as much as the intended ones, and
increasing strength buys disruption rather than targeting. This is consistent
with the closed pilot's preservation finding (P2 56.4% vs P0 81.1%; 0/128
editable descriptions): the released pair's L20 reconstructions do not carry
behaviorally specific per-variable information usable for either preservation
or steering here.

**Honesty limits, carried from the protocol.** These are behavioral
measurements on one checkpoint, one task family, and one site — not a
semantic proof in either direction. Oracle texts were hand-written
intervention instruments, never read-out semantics; arm-1 descriptions are
known from the pilot to assert no current values. This negative result does
**not** refute NLA steering generally: the NLA authors' poetry-planning
steering (≈50% success with messy completions) used a different, stronger
target, a different site, and a different task. No ratio-style "fraction
recovered" was computed.

## Evidence and closure

* Run directory (local, git-ignored):
  `research/open_weight_lingua/runs/steering-20260922T160432Z-d3b9e4d7/`
  (`manifest.json`, `results.json`, `summary.json`, `completion.json`,
  `report.md`, `compatibility.json`, `inventory.json`, `reports.zip`, raw
  safetensors retained locally).
* Manifest identity: `git_head` `aabcef5be7f98e4b74eedba01b4eac2ee8cd6d1f`,
  25 source files hashed; plan hash
  `ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f` (the
  reused pilot split); consumed pilot run manifest sha256
  `cb3ec24ff3187813eb0f2a25ee8397ec7bd94e5b6a8b615588c5615df38b6176`, pilot
  path sha256 `69887207…` (path itself never recorded); frozen criteria and
  both oracle templates verbatim with sha256 in the manifest.
* **Closure:** one run per the frozen stopping rule; the decision is
  recorded, and no α sweep, template revision, checkpoint/site/layer change,
  or follow-up steering stage follows. The assay is closed.
