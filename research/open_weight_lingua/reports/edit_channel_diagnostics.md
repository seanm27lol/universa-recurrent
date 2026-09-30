# Edit-channel diagnostics on the Gemma pairs — 2026-09-30

For `y = 3`, a Gemma-3-12B receiver's saved AV description says "Result: 3",
"y = 3", "requiring the value 3", and in its closing answer slot, "3" three
more times. Change only the three quoted slot numbers to 2, reconstruct the
text with the AR and patch it in exactly as the pilot's P2, and the model
still answers 3. Change every mention to 2, and in about a third of such
receivers it answers 2. This page records the four frozen diagnostic runs
behind that sentence. **They are new measurements on reused or fresh
calibration and pilot splits, not pilot reruns, not validation, and not a
reopening of the closed steering assay. Every closed outcome stands: 0/128
edit coverage under the frozen rule v1.0.0 on every family, the STOP
decisions, and the negative Qwen steering assay.**

## Why these runs, and what was frozen

The post-hoc structure report
([edit_eligibility_structure.md](edit_eligibility_structure.md)) found that
the Gemma descriptions carry the answer value in the AV's own "Final token"
answer slot, never in the frozen rule's variable-state forms. Two protocols
were committed and pushed before any forward they govern:

- [edit_channel_diagnostics.md](../protocols/edit_channel_diagnostics.md)
  (commits `d91e5fd`, clarified `95a050a`) covers D1 on Gemma-3-12B and 27B
  (rewrite the answer slot) and D2 on 12B (capture at the end of the program).
- [edit_replication.md](../protocols/edit_replication.md) (commit `440c41f`)
  covers D3: an out-of-sample replication on fresh 12B groups. It was drafted
  after the D1 12B result, committed after the D1 27B result and while D2
  was running; its status paragraph records that order.

Each part ran exactly once. Predictions were recorded before each run; the
table at the end compares them with the outcomes.

## Definitions

- **Receiver.** `{group}-A-{affected}`, the row the frozen edit rule parses.
  Its answer `a` is the affected variable's current value.
- **Values.** `c` = the group's counterfactual answer (`a ± 1`),
  `o = 2a − c`, `d = (a + 10) mod 20`.
- **Primary population.** Receivers whose answer slot (rule AS-1.0.0) has
  one distinct candidate, and that candidate is `a`.
- **Conditions.** Each text is reconstructed by the AR and patched as P2:
  `restore_norm(AR(text), retained_norm)` replaces block 32 (12B) or 41
  (27B) at the last assistant-prefix token.
  - E0: unedited.
  - E1: slot candidates rewritten to `c`.
  - E1g: every standalone mention of `a` rewritten to `c`.
  - E2 / E2g: the same two edits, with `o` as the value.
  - E3 / E3g: the same two edits, with `d` as the value.
- **Statistics.** A hit is a greedy (`rstrip`) answer equal to the written
  value. `Δ` is the hit rate minus E0's rate for the same value. `L` is the
  mean shift, from E0, of `logP(written) − logP(a)`. Both use one-sided 95%
  lower bounds from a receiver bootstrap (3,000 resamples, seed 206100).

Code: [`edit_diagnostics.py`](../src/open_weight_lingua/edit_diagnostics.py),
[`edit_replication.py`](../src/open_weight_lingua/edit_replication.py),
[`answer_slot.py`](../src/open_weight_lingua/answer_slot.py); launchers
`scripts/run_edit_diagnostics.sh` and `scripts/run_edit_replication.sh`; audits
`python -m open_weight_lingua.edit_diagnostics --audit <run>` and
`python -m open_weight_lingua.edit_replication --audit <run>`.

## Run identity and harness

| Part | Run | Manifest sha256 | `git_head` | Harness gates | Audit | Wall clock |
|---|---|---|---|---|---|---:|
| D1 12B | `editdiag-d1-gemma3-12b-20260930T033853Z-d131ee6e` | `d1b8f988c1721a6f…` | `95a050a` | 5 gates bitwise on 65/65 receivers | PASS | 1,272 s |
| D1 27B | `editdiag-d1-gemma3-27b-20260930T040529Z-2b8da593` | `4156764e99032b43…` | `95a050a` | 5 gates bitwise on 74/74 | PASS | 3,084 s |
| D2 12B | `editdiag-d2-gemma3-12b-20260930T050252Z-41d2b801` | `3e076138bce0be68…` | `95a050a` | re-capture 128/128, AV replay 8/8 bitwise | PASS | 4,027 s |
| D3 12B | `editrep-d3-gemma3-12b-20260930T061624Z-587ec9e4` | `1f9079dc1c7a0550…` | `440c41f` | re-capture 256/256, D1 replay 4/4 bitwise | PASS | 9,039 s |

The D1 gates are:

- fresh P0 logits and greedy output;
- the E0 AR direction;
- the E0 replacement;
- the E0 greedy output against the saved P2.

All held bitwise, so D1's unedited condition *is* the pilot's P2. D3's
replay re-executed four D1 receivers through D3's code and matched D1's saved
records bitwise.

**Environment.** Each part started only after three consecutive idle-GPU
checks. The only other GPU process was the idle rf-moe dashboard (0.5 GB,
0% SM). Peak reserved memory for D3 was 28.8 GiB.

## D1 — rewriting the answer slot on the pilot receivers

| Condition (written value) | 12B hits / 64 | 12B `L` (lower) | 27B hits / 74 | 27B `L` (lower) |
|---|---:|---:|---:|---:|
| E1, slot only (`c`) | 2 (Δ lower 0.0) | +6.98 (+6.08) | 0 | +3.41 (+2.80) |
| E1g, everywhere (`c`) | **25** (Δ lower 0.281) | +14.58 (+12.84) | 1 | +3.45 (+2.70) |
| E2, slot only (`o`) | 2 / 63 | +7.04 (+6.14) | 1 | +3.97 (+3.31) |
| E3, slot only (`d`) | 1 | +9.38 (+8.33) | 1 | +5.50 (+4.65) |

**Frozen readings.** "Slot edit moves behavior" is **not met** on either
family: the E1 `Δ` lower bound is 0.0. "Number-general" is not met, and the
0.30 comparator is not met.

**The log-probability shift is specific to the written number** (mean
shift of `logP(v) − logP(a)`; a descriptive table, not a frozen reading):

| Edit | toward `c` | toward `o` | toward `d` |
|---|---:|---:|---:|
| 12B E1 (writes `c`) | **+6.98** | +1.35 | +2.01 |
| 12B E2 (writes `o`) | +1.53 | **+7.04** | +2.01 |
| 12B E3 (writes `d`) | +1.76 | +1.79 | **+9.38** |
| 27B E1 (writes `c`) | **+3.41** | −2.09 | −1.33 |
| 27B E2 (writes `o`) | −1.60 | **+3.97** | −1.20 |
| 27B E3 (writes `d`) | −0.83 | −1.53 | **+5.50** |

In all 64 12B primary receivers the lead value also appears outside the
quoted slot candidates, so a slot-only edit leaves a self-contradicting text.
The everywhere edit removes the contradiction.

## D3 — the everywhere edit on 256 fresh groups (Gemma-3-12B)

The calibration split had never been given AV descriptions or behavior. All
256 fresh descriptions were `ok`. The primary population is 121 receivers
(2 secondary, 133 excluded).

| Condition (written value) | Hits | Hit rate | Δ lower | `L` (lower) | Answer kept |
|---|---:|---:|---:|---:|---:|
| E1, slot only (`c`) | 5 / 121 | 0.041 | 0.017 | +6.27 (+5.48) | 0.950 |
| **E1g, everywhere (`c`)** | **42 / 121** | **0.347** | **0.281** | +12.51 (+11.04) | 0.628 |
| E2g, everywhere (`o`) | 43 / 120 | 0.358 | 0.283 | +11.89 (+10.54) | 0.633 |
| E3g, everywhere (`d`) | 32 / 121 | 0.264 | 0.198 | +17.64 (+16.52) | 0.570 |

The E0 base rate is 0 for every written value.

**Frozen readings.**

- **R3a, the consistent edit moves behavior out of sample: met.**
- **R3b, number-general: met.**
- The 0.30 comparator is met (E1g 0.347).

The log-probability shifts are again diagonal: +11.9 to +17.6 toward the
written value, +1.2 to +2.6 toward the others.

## D2 — capturing at the end of the program (Gemma-3-12B)

Each receiver was captured at the token that completes the program (e.g. the
final `1` of `x = x + 1`), before the question is asked. The capture is
bitwise identical for the two queries that share that prefix, 128/128.

| 128 receivers | End of program | Answer position (pilot) |
|---|---:|---:|
| Frozen-rule hits | 0 | 0 |
| Names both variables | 54 | 1 |
| Affected variable bound to its true value | 14 | 20 |
| Computed value (never written in the prompt) appears | 4 / 50 | 44 / 50 |
| Computed value bound to its variable | 1 / 50 | 7 / 50 |
| Answer slot eligible / ambiguous / absent | 17 / 20 / 91 | 65 / 63 / 0 |

**Frozen reading: "states variable state" — not met.** R2a is 14 < 32, and
R2b is 0.02 < 0.25. The descriptions narrate the program's surface, for
example:

> Q&A forum structure with numbered list of arithmetic/programming
> questions … Final token "1" ends a list of arithmetic operations …

For `x = 11; y = 3; x = x - 1; x = x + 1` the value 11 is never mentioned.

## Predictions against outcomes

| Part | Recorded prediction | Outcome | Match |
|---|---|---|---|
| D1 12B | E1 hit rate about 0.3–0.5; slot edit moves behavior | 0.031; not met | **no** |
| D1 27B | E1 hit rate at most about 0.15 | 0.0 | yes |
| D2 12B | state reading not met; surface narration | not met (14/128; 1/50) | yes |
| D3 12B | primary about half; E1g about 0.39 and R3a met; E1 at most about 0.06; R3b met | 121/256; 0.347, met; 0.041; met | yes |

## What these runs establish

- **On Gemma-3-12B, a written number controls the answer through the
  reconstruction path, if the text is consistent.** Rewriting every mention of
  the value moves the greedy answer to the written value in about a third of
  receivers. Fresh groups, any written value (a neighbour, the counterfactual,
  or a distant number) and a pre-registered replication all agree.
- **Rewriting only the answer slot does not work.** The rest of the
  description still states the old value. This is why a rule that edits one
  statement — the frozen v1.0.0 design, applied to this genre — could not
  have shown an edit effect here even with coverage.
- **On Gemma-3-27B the written number shifts the preference but never the
  answer.** +3.4 to +5.5 nats toward the written value, 0–1 of 74 flips.
  This agrees with the P3 natural-edit reading from the structure report
  (7% adoption).
- **Moving the capture to the end of the program does not make the AV state
  variable values.** It names the variables more, and the computed values
  almost disappear (44/50 → 4/50). At that position the model has not been
  asked anything yet.

## What they do not establish

- **Nothing is relabeled.** Every family's 0/128 coverage and STOP decision
  stand. The Phase Two edit hypothesis as frozen — editing a stated
  variable-state value — remains untested. D1/D3 test a different object:
  the AV's next-token answer guess, which at this site is the same number.
- **The flip does not show that the description reads the activation.** The
  patch replaces the site activation with a reconstruction of the text, so
  the flip shows that the AR maps a consistently written number to a
  direction the target reads as that answer. That is a property of the AR
  and target, not evidence that the AV verbalized the activation faithfully,
  and not a change to a represented variable state.
- **No scaling claim.** 12B's flips and 27B's absence of flips do not rank
  model sizes: the sites differ (block 32 of 48, sliding-window; block 41 of
  62, full-attention), the NLA pairs were trained separately, and 27B's
  descriptions came from its BF16-cast AV.
- **Not a contradiction of the Qwen steering assay.** That assay used a
  different recipe (additive difference steering), a different family and
  hand-written templates.
- **Scope.** The everywhere edit can also rewrite an unrelated number that
  happens to equal the value. D1 reused the pilot split. One task family, one
  site per family, one run per part.
