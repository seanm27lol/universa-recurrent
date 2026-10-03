# Receiver-specificity control for the consistent edit (frozen pre-run protocol)

**Status: FROZEN 2026-10-03 (UTC), before any D5 model forward, after D1–D4
were read. Every rule and reading below is a design choice locked here, not a
result.**

## The question

On Gemma-3-12B, rewriting every mention of the answer in the AV's own
description of a receiver moves the patched answer to the written number in
42 of 121 fresh receivers (D3), and four hand-written texts asserting the same
number do not (D4, 0–1/121). That separates "AV description" from "any text
with a number". It does not separate:

1. **This activation:** content specific to *this* receiver's activation,
   which the AR uses together with the written number; from
2. **This kind of text:** any AV-register description of this task. The
   description of a different program would work as well once it states the
   same number.

D5 patches each receiver with a different receiver's AV description,
consistently rewritten to state this receiver's counterfactual.

**The prediction is against the D4 report's reading.** The pilot's P3
condition patched in the next group's unedited description, and the 12B
answer followed that description's lead number in 220 of 477 rows (46%)
whenever it differed (reports/edit_eligibility_structure.md). So we expect a
foreign description to work about as well as the receiver's own, which would
mean the D3/D4 effect belongs to AV-register text, not to this activation.

## Design

- **Receivers.** D3's 121 primary receivers (Gemma-3-12B calibration split).
  Each receiver `i` has a true answer `a_i` and a counterfactual `c_i`.
- **Inputs, all read-only.** The D3 run (plan, results and saved P0 logits),
  the calibration run, and the Gemma-3-12B lock (`d49ac3ea…`), hash-verified
  at start. The AV is not loaded.
- **Donor (frozen rule).** For receiver `i`, the donor `j` is the next
  receiver after `i`, cyclically in D3's primary order, that queries the same
  variable. Its saved D3 description has a single-candidate answer slot whose
  lead is its own answer `a_j`.
- **Patch.** As P2, D3 and D4: `restore_norm(AR(text), retained_norm_i)`
  replaces block 32 at receiver `i`'s last assistant-prefix token. Greedy
  answer (`rstrip`), log-probabilities of D3's answer set plus `a_j`,
  next-token KL against D3's saved P0.

| Condition | Text | Role |
|---|---|---|
| F0 | `i`'s own description, every mention rewritten to `c_i` (D3's E1g, replayed) | comparator; harness gate |
| F1 | `j`'s description, every mention of `a_j` rewritten to `c_i` (`answer_slot.edit_everywhere`) | **primary** |
| F2 | `j`'s description, unedited (states `a_j`) | base: what a foreign description does without the edit |

When `a_j = c_i`, F1 equals F2 by construction. Those receivers are counted
and kept, intention-to-test.

Known from saved text before the run:

- all 121 receivers have a same-variable donor, never themselves;
- F1's answer slot leads with `c_i` in all 121;
- in 7, the donor already states `c_i`;
- in 8, it states the receiver's own `a_i`.

**Harness gate.** F0 must reproduce D3's saved E1g record bitwise: the
generation, and the log-probabilities of D3's answer set. D5's statistics are
interpreted only if it does on all 121 receivers.

## Statistics and readings (declared now)

Paired over the 121 receivers; receiver-level bootstrap, 3,000 resamples, new
seed 208100; one-sided 95% lower bounds. `hit(X, v)` means the greedy answer
equals `v`.

- **R5a — the receiver's own description matters** iff the lower bound of
  `mean[hit(F0, c) − hit(F1, c)]` is above 0.
- **R5b — a foreign AV description works as well** iff
  `mean hit(F1, c) ≥ mean hit(F0, c)` and the lower bound of
  `mean[hit(F1, c) − hit(F2, c)]` is above 0. The edit moves the answer
  beyond what the unedited foreign text already does.
- **Descriptive:**
  - adoption of the donor's number, `mean hit(F2, a_j)`, over receivers with
    `a_j ∉ {a_i, c_i}`;
  - retained and off-target rates;
  - cosines to the original;
  - `L = mean[logP(c) − logP(a)]` per condition against D3's E0.

**How the readings will be read.**

- R5a met: content specific to this activation matters, beyond AV register
  plus number.
- R5b met: the D3/D4 effect belongs to AV-register text stating a number,
  not to this activation. The handoff and reports must then drop the
  "receiver-specific" reading.
- Neither: report the paired intervals and call it unresolved.

**Predictions (not thresholds), recorded before any D5 forward:**

| Quantity | Prediction |
|---|---|
| F1 hit rate | about the same as F0 (0.3–0.5) |
| R5b | met |
| R5a | not met |
| F2 adoption of `a_j` | about 0.4 (pilot P3: 46%) |

## Cost, stopping rule, limits

- **Projected cost.** About 40 min: AR about 360 reconstructions, target
  3 × 121 patched conditions, plus loads, hash verification and the CPU
  suite.
- **Start condition.** Only on an idle GPU with at least 60 GiB available.
  If the target stage projects beyond three times one hour, the run stops and
  is reported.
- **Stopping rule.** Exactly one run, and whatever happens is reported. No
  donor rule, text or condition changes after outcomes are read. A source bug
  permits only a regression test plus a visibly versioned rerun.
- **Limits.**
  - One donor per receiver, under one fixed rule.
  - The donor's program differs from the receiver's in its numbers and
    statements, but it comes from the same task family and the same
    variable query.
  - "Receiver-specific" here means specific to this program's activation
    versus another program of the same task. Nothing here tests
    cross-task transfer or faithfulness.
