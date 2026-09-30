# Out-of-sample replication of the consistent answer-slot edit (frozen pre-run protocol)

**Status: FROZEN 2026-09-30 (UTC), before any D3 model forward. The order,
exactly:**

1. The design, every rule, the readings and the choice of family
   (Gemma-3-12B only) were drafted after the D1 Gemma-3-12B result was read.
2. The D1 Gemma-3-27B result was read next. The draft was not changed
   because of it, except for this status paragraph.
3. This protocol was committed while D2 was running, before its result was
   read.

**Every rule and reading below is a design choice locked here, not a
result.**

## Why this run

For `y = 3` in a Gemma-3-12B pilot receiver, the AV's description says
"Result: 3", "y = 3", "requiring the value 3" and, in its "Final token"
answer slot, "3" three more times. The D1 diagnostic
([edit_channel_diagnostics.md](edit_channel_diagnostics.md), run
`editdiag-d1-gemma3-12b-20260930T033853Z-d131ee6e`, harness reproduced
bitwise on 65/65 receivers, audit PASS) found:

- **E1, slot-only edit** (only the quoted slot candidates changed to the
  counterfactual 2): the greedy answer moved in 2/64 receivers. The frozen
  reading was not met, against a prediction of 0.3–0.5.
- **E1g, every standalone mention rewritten** (a secondary condition): the
  answer moved to the written value in 25/64 receivers (0.39; bootstrap
  lower bound 0.28).
- **Log-probability shifts were number-specific.** Each edit raised the
  log-probability of its own written value by 7–15 nats and the other
  candidate values by 1.3–3.2.

All 64 primary receivers mention the value outside the quoted slot, so a
slot-only edit leaves a self-contradicting text.

E1g was a secondary condition, found on the same 128 pilot groups that
motivated the rule, and it had no everywhere-edit specificity controls. D3
tests it out of sample, with those controls, once.

## Design

- **Groups.** The 256 groups of the Gemma-3-12B calibration split, from run
  `calibration-20260923T042912Z-d0e9499f` (plan hash `ff060012…`). Before
  this protocol, that split was used only for the target-only P4 PCA fit and
  the frozen median norm; no AV description or behavior was ever produced on
  it. The locked validation splits are not touched.
- **Receivers.** `{group}-A-{affected}` (256), at the pilot site: block 32,
  last assistant-prefix token.
- **Model.** The Gemma-3-12B lock (`d49ac3ea…`), hash-verified at start.
- **Descriptions.** One greedy AV description per receiver of the saved
  calibration original, with the pilot's decoding. Rule AS-1.0.0 defines the
  populations exactly as in D1: primary = eligible slot whose lead equals the
  true answer `a`; secondary = eligible with another lead; else excluded.
- **Values.** `c` = the counterfactual answer (`a ± 1`); `o = 2a − c`;
  `d = (a + 10) mod 20`.
- **Conditions** (patched exactly as the pilot's P2, `restore_norm(AR(text),
  retained_norm)`):

| Condition | Text | Role |
|---|---|---|
| E0 | unedited | base rates |
| E1 | slot-only edit to `c` | replicates D1's null |
| E1g | every standalone mention of the lead rewritten to `c` | **primary** |
| E2g | every mention rewritten to `o` (when `0 ≤ o ≤ 19`) | specificity |
| E3g | every mention rewritten to `d` | specificity, distant value |

- **Measurements.** Greedy answer (`rstrip`), log-probabilities of `a`, `c`,
  `o`, `d`, next-token KL against the fresh P0.

## Harness gates

- Every re-captured receiver activation equals its saved calibration original
  bitwise.
- **D1 replay.** The first four primary receivers of the D1 12B run are
  re-executed through D3's code path (E0 and E1g, with the pilot's saved
  inputs and originals). Their greedy generations and answer log-probabilities
  must equal D1's saved records bitwise.

D3's statistics are interpreted only if both gates hold on every row.

## Statistics and readings (declared now)

Primary population, receiver-level bootstrap, 3,000 resamples, seed 206100,
one-sided 95% lower bounds; definitions identical to D1:

- `Δ(E, v) = mean hit(E, v) − mean hit(E0, v)`;
- `L(E, v)` = mean shift, from E0 to E, of `logP(v) − logP(a)`.

Readings:

- **R3a: the consistent edit moves behavior out of sample** iff the lower
  bounds of both `Δ(E1g, c)` and `L(E1g, c)` are above 0.
- **R3b: number-general** iff R3a holds and the same holds for E2g (value `o`)
  and E3g (value `d`).
- **Comparator only:** whether `mean hit(E1g, c) ≥ 0.30`. It gates nothing.

**Predictions (from D1 12B; not thresholds):**

| Quantity | Prediction |
|---|---|
| Primary population | about half of the 256 receivers (D1: 64/128) |
| E1g hit rate | about 0.39 (D1: 0.39, lower 0.28); R3a met |
| E1 hit rate | at most about 0.06 (D1: 0.03) |
| E2g and E3g | lower bounds above 0 (D1's log-probability shifts were number-specific); R3b met — the least certain prediction |

## Cost, stopping rule, limits

- **Projected cost.** About 3.2 h: AV 256 × ~33 s, AR about 5 min, behavior
  about 40 min, plus loads and hash verification. It starts only when the
  GPU has no other active workload and at least 60 GiB is available. If the
  AV stage projects beyond three times the 12,000 s estimate, the run stops
  and is reported.
- **Stopping rule.** Exactly one run, and whatever happens is reported. No
  rule, text, value or condition changes after outcomes are read. A source
  bug permits only a regression test plus a visibly versioned rerun.
- **Limits.** The same as D1: an answer-slot edit rewrites the AV's
  next-token guess. A moved answer shows the AR-patch path carries the
  written number to the output. It does not show that the description reads
  the activation or changes a represented variable state. The everywhere edit
  can also rewrite an unrelated number that happens to equal the lead. One
  model, one task family, one site.
