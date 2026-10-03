# Consistency dose for the AV-description edit (frozen pre-run protocol)

**Status: FROZEN 2026-10-03 (UTC), before any D6 model forward, after D1–D5
were read. Every rule and reading below is a design choice locked here, not a
result.**

## The question

A Gemma-3-12B receiver's AV description mentions its answer many times: 5 to
15 standalone mentions, median 11, on D3's 121 primary receivers. D1, D3 and
D5 found:

| Mentions rewritten to the counterfactual `c` | Answer moves to `c` |
|---|---:|
| Only the quoted "Final token" candidates | 5 / 121 |
| All of them | 42 / 121 |
| All of them, in another program's description | 46 / 121 |

So consistency matters, but how? Two readings:

1. **Majority:** the answer flips once enough of the text agrees on the new
   number, wherever the mentions sit.
2. **Location:** some part of the text — the answer slot, or the
   description's earlier narrative — carries the number on its own.

D6 rewrites a growing fraction of the mentions, and separately everything
except the slot.

## Design

- **Receivers.** D3's 121 primary receivers (Gemma-3-12B calibration split),
  with D3's values `a` and `c`.
- **Inputs, all read-only.** The D3 run (plan, results and saved P0 logits),
  the calibration run, and the 12B lock (`d49ac3ea…`), hash-verified at
  start. The AV is not loaded.
- **Patch.** As P2 and D3–D5: `restore_norm(AR(text), retained_norm)` at
  block 32, last assistant-prefix token. Greedy answer (`rstrip`),
  log-probabilities of D3's answer set, next-token KL against D3's saved P0.
- **Mentions.** A mention is a standalone occurrence of `a` in the receiver's
  saved D3 description: the same pattern `answer_slot.edit_everywhere` uses.
  Mentions are listed in text order.

| Condition | Text (all edits write `c`) | Role |
|---|---|---|
| K100 | every mention rewritten (D3's E1g, replayed) | full dose; harness gate |
| K25 | the first ⌈0.25·n⌉ mentions rewritten | dose |
| K50 | the first ⌈0.50·n⌉ mentions | dose |
| K75 | the first ⌈0.75·n⌉ mentions | dose |
| NS | every mention except the quoted answer-slot candidates | location: the slot left unchanged |

Text order puts the narrative paragraphs before the "Final token" slot. So
K25 and K50 mostly rewrite narrative mentions, and K75 usually reaches into
the slot. On these descriptions no partial dose equals the full rewrite (none
of the 121 receivers has `⌈0.75·n⌉ = n`). Every receiver has at least one
mention outside the slot. When the first mentions in text order are exactly
the non-slot ones, NS coincides with a partial dose. That is known from saved
text before the run: NS equals K25 in 4 receivers and K50 in 3. These are
kept, intention-to-test. D3 already measured the zero dose (E0, 0/121) and
the slot-only edit (E1, 5/121); they are reported as reference points, not
rerun.

**Harness gate.** K100 must reproduce D3's saved E1g record (generation and
answer log-probabilities) bitwise on all 121 receivers. D6's statistics are
interpreted only if it does.

## Statistics and readings (declared now)

Paired over receivers; bootstrap, 3,000 resamples, new seed 209100; one-sided
95% lower bounds. `hit(X)` means the greedy answer equals `c`.

- **R6a — more agreement, more flips** iff the lower bound of
  `mean[hit(K75) − hit(K25)]` is above 0.
- **R6b — the slot is not needed** iff the lower bound of `mean hit(NS)` is
  above 0 (against D3's E0 base rate of 0) and `mean hit(NS) ≥ 0.5 · mean
  hit(K100)`.
- **Descriptive:**
  - the dose curve E0 → K25 → K50 → K75 → K100, with D3's slot-only E1 shown
    beside it;
  - `L = mean[logP(c) − logP(a)]` per condition against D3's E0;
  - retained and off-target rates.

**How the readings will be read.**

- R6a met: the flip is driven by how much of the description agrees.
- R6b met: the narrative mentions carry the number without the slot.
- R6a met but R6b not: agreement matters, and the slot's mentions are part
  of what must agree.

**Predictions (not thresholds), recorded before any D6 forward:**

| Quantity | Prediction |
|---|---|
| Hit rates | rise with the dose: K25 at most about 0.1, K50 about 0.1–0.2, K75 about 0.2–0.3, K100 0.347 (known) |
| R6a | met |
| NS | about 0.2–0.3; R6b met |

The R6b prediction is the least certain. Its basis is that slot-only edits
barely work (5/121) while the slot holds only about half the mentions.

## Cost, start, stopping rule, limits

- **Projected cost.** About 45 min: AR 605 reconstructions, target 5 × 121
  patched conditions, plus loads, hash verification and the CPU suite.
- **Start condition.** Only when the GPU has no other active workload and at
  least 60 GiB is available. On 2026-10-03 a 25-epoch training job owned by
  someone else was running, and D6 waits for it rather than sharing.
- **Throughput stop.** If the target stage projects beyond three times one
  hour, the run stops and is reported.
- **Stopping rule.** Exactly one run, and whatever happens is reported. No
  dose, order, text or condition changes after outcomes are read. A source
  bug permits only a regression test plus a visibly versioned rerun.
- **Limits.**
  - Text order confounds dose with location: early doses rewrite the
    narrative first. NS is the only condition that separates them.
  - A standalone match of `a` can also be an unrelated number, such as an
    operand that happens to equal it. This is the same caveat as the
    everywhere edit.
  - One model, one site, one split.
