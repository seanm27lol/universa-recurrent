# Edit-channel diagnostics on the Gemma pairs (frozen pre-run protocol)

**Status: FROZEN 2026-09-29 (UTC), before any diagnostic model forward. No
diagnostic outcome has been read; every rule, condition, statistic and reading
below is a design choice locked here, not a result.**

For `x = 9; y = 8; y = 3; y = y + 3; x = x + 1` and "What is x?", the answer
is 10. Gemma-3-27B's saved description of that activation ends "expecting a
numeric answer like "10" or "11"". The post-hoc structure report
([reports/edit_eligibility_structure.md](../reports/edit_eligibility_structure.md))
found that on both Gemma pairs the value sits in this "Final token" answer
slot and never in the frozen rule's variable-state forms, which is why 0/128
groups were edit-eligible. This protocol runs two diagnostics that report
could not: whether rewriting the number in that slot moves the model's answer,
and whether a different capture position produces variable-state statements.

**These are new diagnostic measurements on the reused pilot split. They are
not pilot reruns, not Phase Two validation, and not a reopening of the closed
steering assay.** Every closed outcome stands: 0/128 edit coverage under the
frozen rule v1.0.0 on every family, the STOP decisions, and the negative
steering-assay result on Qwen. Nothing measured here is blended into those
records.

## D1 — does rewriting the answer slot move the answer? (Gemma-3-12B primary, Gemma-3-27B secondary)

**Inputs, read-only.** The confirmation pilots
`pilot-20260927T202455Z-f5ec3892` (12B) and `pilot-20260929T034532Z-4e4d655f`
(27B): their manifests, saved descriptions, saved original activations and
retained norms, saved AR directions, and saved P0/P2 behavior. Models are the
locked ones (`configs/model-lock-gemma3-12b.json`,
`configs/model-lock-gemma3-27b.json`), hash-verified at run start. The AV is
not loaded; descriptions are the saved ones.

**Receivers and populations.** The 128 receivers are the rows the frozen rule
parses, `{group}-A-{affected}`; their answer `a` is the affected variable's
current value. Rule AS-1.0.0 ([`answer_slot.py`](../src/open_weight_lingua/answer_slot.py),
sha256 recorded in the manifest) parses each saved description.

- *Primary population*: slot `eligible` and its lead equals `a`.
- *Secondary population*: slot `eligible` with a lead other than `a`.
  Reported descriptively only.

The population sizes are already known from saved text, and they are
syntactic, not a result of this diagnostic:

- 12B: 64 primary receivers, 1 secondary, 63 excluded (323 conditions);
- 27B: 74 primary, 0 secondary, 54 excluded (370 conditions).

**Requested values.** `c` = the group's counterfactual answer (the B-side
affected answer, `a ± 1`); `o = 2a − c`, the other neighbor; `d = (a + 10) mod
20`, a distant value.

**Conditions, per receiver.**

| Condition | Text given to the AR | Role |
|---|---|---|
| E0 | the saved description, unedited | harness check; must reproduce the saved P2 |
| E1 | `edit_slot(description, c)` | **primary edit** |
| E1g | `edit_everywhere(description, c)` | secondary: the value also rewritten outside the slot |
| E2 | `edit_slot(description, o)`, only when `0 ≤ o ≤ 19` | is the effect specific to the counterfactual? |
| E3 | `edit_slot(description, d)` | does a distant written value also move the answer? |

Every condition is patched exactly as the pilot's P2: `restore_norm(AR(text),
retained_norm)` replaces the block output at the pilot site (the last
assistant-prefix token), and all target forwards use the pinned
`TARGET_BUCKET = 128` shapes. For each condition the run records:

- the greedy generation (at most 8 tokens) and its integer answer under the
  `rstrip` convention;
- teacher-forced log-probabilities of `a`, `c`, `d` and, when valid, `o`;
- next-token KL against the saved P0 logits;
- the direction's cosine to the original activation.

**Harness gates, recorded per receiver.**

- G0: the fresh unpatched next-token logits equal the saved P0 logits
  bitwise, and the fresh P0 greedy generation equals the saved one.
- G1: the E0 AR direction equals the saved AR direction bitwise, the E0
  replacement equals the saved P2 replacement bitwise, and the E0 greedy
  generation equals the saved P2 generation.

**D1's statistics are interpreted only if G0 and G1 hold on every executed
receiver.** Any failure is reported first, and the statistics are then marked
"harness not reproduced".

**Statistics** (primary population, receiver-level bootstrap, 3,000 resamples,
seed 206100, one-sided 95% lower bounds). `hit(E, v)` means the greedy answer
equals `v`.

- `Δ1 = mean hit(E1, c) − mean hit(E0, c)`, the rise in answering `c`.
- `L1` = mean over receivers of `[logP(c) − logP(a)]` under E1, minus the same
  quantity under E0.
- The same pair for E1g (value `c`), for E2 (value `o`, over receivers where
  it is defined) and for E3 (value `d`).
- Descriptive: retention (answer = `a`) and off-target answers (neither `a`
  nor the requested value) under every condition.

**Readings, declared now.**

- *The slot edit moves behavior toward the written value* on a family iff the
  lower bounds of both `Δ1` and `L1` are above 0.
- *Number-general*, iff the same holds for E2 and E3 on their own values.
- *Comparator only*: whether `mean hit(E1, c) ≥ 0.30`, the bar the closed
  steering assay used for C1. It gates nothing.

**Predictions, recorded before the run — not thresholds.** The P3 natural
edits adopted another group's lead number in 46% of Gemma-12B rows and 7% of
Gemma-27B rows. So for 12B we expect the slot edit to move behavior
(`mean hit(E1, c)` around 0.3–0.5). For 27B we expect a weak effect
(`mean hit(E1, c)` at most about 0.15). A whole-text swap is a stronger push
than a one-number edit, so both could come in lower.

## D2 — does an end-of-program capture yield variable-state descriptions? (Gemma-3-12B)

**Rows.** The same 128 receiver prompts, with the same tokens as the pilot.

**New position.** `p_end` is the first token position at which the decoded
prompt prefix contains the whole program text, i.e. the token that completes
the program's last character, before the question is asked. It uses the same
layer (block 32) and the same model and AV as the pilot. Only the token
position changes.

**Gates.**

- G2: capturing at the pilot's own site reproduces every saved original
  activation bitwise.
- G3: re-verbalizing the saved originals of the first eight receivers
  (`pilot-0000` to `pilot-0007`) reproduces their saved description token ids
  bitwise.
- Descriptive only: `p_end` captures for the `A-{affected}` and `A-{other}`
  prompts, which share every token before the question, are compared for
  bitwise equality.

**Measurement.** One greedy AV description per receiver at `p_end` (same
decoding as the pilot: greedy, at most 200 tokens, the checkpoint's declared
eos set). Text-only statistics, computed after the run:

- frozen-rule v1.0.0 statuses for `x` and `y`, and the resulting coverage of
  the affected variable;
- whether the affected variable's *computed* final value (one that occurs
  nowhere in the prompt) appears bound to that variable (`x = N`, `x is N`,
  `x is now N`, and the other bound forms of the structure report);
- whether the true final values of `x` and `y` appear anywhere;
- the answer-slot status;
- the same census for the saved pilot-site descriptions of the same
  receivers.

**Reading, declared now.** End-of-program descriptions *state variable
state* iff the affected variable's computed final value appears bound to that
variable in at least 32 of 128 descriptions, mirroring the pilot's 32-group
floor. Frozen-rule coverage is reported alongside.

**Prediction, recorded before the run — not a threshold.** The reading is not
met. We expect descriptions of the program's surface — its assignments, their
literals and the likely next token — rather than computed state.

## Code and commands

The implementation is
[`src/open_weight_lingua/edit_diagnostics.py`](../src/open_weight_lingua/edit_diagnostics.py),
with text measures in
[`description_census.py`](../src/open_weight_lingua/description_census.py).
Its end-to-end tests in `tests/test_edit_diagnostics.py` run both parts on tiny
random fixture models. On fixtures every harness gate reproduces the pilot's
saved P0/P2 bitwise; that establishes the plumbing, not any real-model result.
From the repository root:

```bash
bash research/open_weight_lingua/scripts/run_edit_diagnostics.sh --part d1 \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --pilot-run research/open_weight_lingua/runs/pilot-20260927T202455Z-f5ec3892
bash research/open_weight_lingua/scripts/run_edit_diagnostics.sh --part d1 \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-27b.json \
  --pilot-run research/open_weight_lingua/runs/pilot-20260929T034532Z-4e4d655f
bash research/open_weight_lingua/scripts/run_edit_diagnostics.sh --part d2 \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --pilot-run research/open_weight_lingua/runs/pilot-20260927T202455Z-f5ec3892
.venv-phase2/bin/python -m open_weight_lingua.edit_diagnostics --audit <run-dir>
```

The audit recounts D1 hit rates, or D2 frozen-rule coverage, from saved
results without calling the summarizers.

## Cost, order, stopping rule

- **Order.** D1 on 12B, then D1 on 27B, then D2 on 12B. Each part starts only
  when no other GPU workload is active and at least 60 GB (12B) or 90 GB (27B)
  of unified memory is available; otherwise it waits. An idle resident process
  (such as the rf-moe dashboard, about 0.5 GB and 0% SM, present during both
  confirmation pilots) does not count as a workload; it is recorded. This
  clarification was added before any diagnostic forward.
- **Projected cost.** About 25 min (D1 12B), 45 min (D1 27B) and 80 min
  (D2 12B), plus hash verification and loads.
- **Throughput stop.** If measured throughput on the first receivers projects
  a part beyond three times its estimate, that part stops and is reported. It
  is not shrunk mid-run.
- **Stopping rule.** Exactly one run per part. Whatever happens is reported.
  No rule, text, value or condition changes after outcomes are read. A source
  bug permits only a regression test plus a visibly versioned rerun.
- **Artifacts.** Run directories stay local (`runs/editdiag-*`). Reports cite
  run IDs and manifest hashes.

## Limits carried into every report of these diagnostics

- An answer-slot edit rewrites the AV's guess about the next token. A moved
  answer shows that the AR-patch path carries that number to the output. It
  does not show that the description reads the activation, or that the edit
  changes a variable's represented state.
- At the pilot site the answer is the next token, so "editing the answer" and
  "editing the affected variable's value" are the same number for receivers,
  but not the same hypothesis. The pilot's wrong-variable control has no
  counterpart here: receiver descriptions do not carry the other variable.
- D2 changes only the token position. Its descriptions are measured as text;
  no behavior is measured at `p_end`.
- One task family, one pilot split, two checkpoints. D1 runs no AV, but the
  27B descriptions it edits were produced under that AV's BF16 serving cast,
  which remains unauditable locally. D2 is 12B only.
