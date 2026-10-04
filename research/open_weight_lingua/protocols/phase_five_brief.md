# Phase Five: is the state there, but nonlinear?

**Status: FROZEN 2026-10-04 (UTC), before any Phase Five probe fit or model
forward. Branch `phase-five-nonlinear`, based on `main` at `310b085`. Every
threshold, gate and stopping rule below is a design choice locked here, not a
result.**

## The question

Phases Three and Four found no *linearly* readable program state, at the
end of the program or at any statement boundary. Take
`x = 9; y = 8; y = 3; y = y + 3; x = x + 1`. No linear probe reads x = 10
and y = 6 together anywhere, and after `y = y + 3` the result 6 reads at
only about 0.2
([reports/phase_three_stage0.md](../reports/phase_three_stage0.md),
[reports/phase_four_trace.md](../reports/phase_four_trace.md)). Both
reports named the gap: a nonlinear code is untested.

The field has a famous case where that gap mattered. On Othello-GPT, linear
probes failed to read the board while nonlinear probes succeeded (Li et al.
2023, *Emergent World Representations*, ICLR). A linear code was found later,
in a relative "mine/theirs" basis (Nanda, Lee & Wattenberg 2023). Phase
Five gives the state the same chance here, with a small neural probe. If it
reads what linear probes could not, the "no readable state" conclusion
reverses.

## Design

**Probe (frozen).**

- **Inputs:** features standardized per dimension on the training split
  (standard deviation clamped at 1e-6).
- **Architecture:** `Linear(d, 256) → GELU → Dropout(0.1) → Linear(256, 20)`,
  float32, on the GPU.
- **Training:** full-batch AdamW at learning rate 1e-3 for 300 epochs, with
  weight decay ∈ {0.01, 0.1}. Selection-split accuracy is checked every 10
  epochs, and the (weight decay, epoch) with the best selection accuracy is
  used. Ties go to the larger decay, then the earlier epoch.
- **Seed:** fixed per fit, derived from 501100 and the fit's identity.
- **Can it read what a linear probe cannot?** Yes, on synthetic data. In
  `tests/test_state_nonlinear.py` each value v is coded as +c_v or −c_v with
  equal counts, so its class mean is zero. A linear probe then reads under
  0.2 and this MLP over 0.95. That is the Othello-GPT situation in miniature.

**Permuted-label control** (in the spirit of Hewitt & Liang 2019's control
tasks). A flexible probe can memorize labels or exploit a leak between
splits. At every selected site, the same MLP is trained with the chosen
weight decay and number of epochs, but on training labels shuffled with
Phase Three's seed 301100. It is scored against the true held-out labels. A
reading counts only if this control stays at or below 0.20 for both
variables.

**Part A: the final state** (Phase Three's question).
- **Data:** Phase Three's saved activations, read-only (five positions,
  every block; the same train / select / held-out splits).
  - Gemma-3-12B `p3-stage0-gemma3-12b-20261004T062811Z-745ea17c`;
  - Qwen2.5-7B `p3-stage0-qwen2.5-7b-20261004T074740Z-840871a1`.
- **Site selection:** Phase Three's rule (`state_survey.select_site`),
  now with the MLP.
- **Gate G5a:** at the selected site, Phase Three's gate G0
  (`state_survey.gate_g0`) holds with the MLP: held-out
  `min(acc_x, acc_y) ≥ 0.80` and computed-subset minimum ≥ 0.50. The control
  accuracy must also be ≤ 0.20 for both variables.
- **Descriptive (never gating):** held-out accuracy split by whether the
  variable was the one asked, at every site. This is the check Phase Three
  ran post hoc for its linear probe.

**Part B: the running state** (Phase Four's question).
- **Data:** Phase Four's 2,548 statement boundaries. The positions and
  labels come from its manifests, read-only. The activations are captured
  again at exactly those positions, because Phase Four did not keep them.
  - Each recorded position is first re-derived with the tokenizer, and the
    run stops on any mismatch.
  - This run keeps the re-captured activations (float32) under its local
    `raw/`.
- **Probe:** one MLP per layer and variable, pooled over boundaries. The MLP
  can use position information, so it can also read a code that differs from
  one line to the next.
- **Readings:**
  - **R1 (carried):** carried accuracy ≥ 0.80 and carried-computed ≥ 0.50,
    both variables.
  - **R2 (arithmetic):** arithmetic accuracy ≥ 0.80, both variables.
  - Each reading uses its layer selected on the selection split by Phase
    Four's rule (`state_trace_survey.readings`), with the control at that
    layer ≤ 0.20.
- **Gate G5b = R1 and R2.**

**Gate G5:** a nonlinear state is readable iff G5a **or** G5b holds on
Gemma-3-12B. Qwen2.5-7B is reported only.

- **If G5 passes,** the "no readable state" conclusion of Phases Three and
  Four is reversed in the ledger. A causal test would follow under its own
  brief.
- **If G5 fails,** the conclusion stands, now for linear *and* small
  nonlinear readouts.

**Predictions (not thresholds), recorded before any fit:** G5 fails.

- At the answer position, the MLP reads the asked variable well and the
  other one near chance, so `min(x, y)` stays below 0.80.
- At the end of the program, computed values stay low.
- At statement boundaries, arithmetic results improve on the linear 0.2 but
  stay well below 0.80.

## Budget, stopping, honesty

- **Budget:** at most 4 GPU-hours. One run per family. A source bug permits
  only a regression test plus a visibly versioned rerun.
- **Start:** each run starts only when the GPU has no other workload.
- **Data:** the validation splits stay unopened.
- **Limits:**
  - One small MLP architecture. "Not readable by this probe" is weaker than
    "not represented".
  - Probes measure what is decodable, not what the model uses.
  - **A pass would not settle the question either.** A nonlinear probe can
    do the arithmetic itself, from operands the activation encodes, so a
    readable result need not be a value the model computed. Only a causal
    test, under its own brief, could tell those apart (Belinkov 2022,
    *Probing Classifiers*, Computational Linguistics).
  - One task family.

**Code:** `src/open_weight_lingua/state_nonlinear.py`, launched with
`scripts/run_state_nonlinear.sh`.
