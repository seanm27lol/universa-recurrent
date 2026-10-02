# Oracle-text control for the consistent edit (frozen pre-run protocol)

**Status: FROZEN 2026-10-02 (UTC), before any D4 model forward, and after
all four earlier diagnostic results (D1 12B and 27B, D2, D3) were read.
Every rule and reading below is a design choice locked here, not a result.**

## The question

D3 ([edit_replication.md](edit_replication.md), run
`editrep-d3-gemma3-12b-20260930T061624Z-587ec9e4`) found that on
Gemma-3-12B, rewriting every mention of the answer in the AV's own description
of `x = 9; …; x = x + 1` from 10 to 11 makes the patched model answer 11 in
42 of 121 fresh receivers. That could mean two different things:

1. The AV's description carries something about this activation, which the
   AR uses together with the written number.
2. The AR maps *any* text that asserts a number to a direction the target
   reads as "answer that number". The AV's description would then add nothing
   but the number.

D4 separates the two. It patches the same receivers through the same path with
hand-written texts that assert the same number and contain nothing from the
AV.

## Design

- **Receivers.** D3's 121 primary receivers (Gemma-3-12B calibration split),
  with D3's values: `a` = the true answer, `c` = the counterfactual.
- **Inputs, all read-only.** The D3 run (plan, results and saved P0 logits),
  the calibration run (inputs, saved originals and retained norms) and the
  Gemma-3-12B lock (`d49ac3ea…`), hash-verified at start. The AV is not
  loaded.
- **Patch.** Every condition is patched as P2 and as D3:
  `restore_norm(AR(text), retained_norm)` replaces block 32 at the last
  assistant-prefix token. Greedy answer (`rstrip`), log-probabilities of `a`,
  `c`, `o` and `d` (the D3 answer set), next-token KL against D3's saved P0.

**Conditions** (all assert `c`; `{variable}` is the receiver's queried
variable):

| Condition | Text (rendered with `value = c`) | Source |
|---|---|---|
| E1g | D3's edited AV description, every mention of `a` rewritten to `c` | D3 plan, replayed |
| O_terse | `The current value of {variable} is {value}.` | `steering.TEMPLATES["terse"]`, frozen 2026-09-22 |
| O_structured | the three-paragraph AV-register template | `steering.TEMPLATES["structured"]`, frozen 2026-09-22 |
| O_slot | `Final token "⏎" ends the question, immediately expecting a numeric answer like "{value}".` (`⏎` = a literal newline, as the AV writes it) | new, frozen here |
| O_number | `A short note that mentions the number {value}.` | new, frozen here |

O_terse states the variable's value in words close to the frozen v1.0.0
rule's `the current value of {var} is {value}` form. Its capital "The"
means the case-sensitive rule does not match it (a test pins this). So it
is a state statement in substance, not a v1.0.0-eligible one.

**Harness gate.** The replayed E1g condition must reproduce D3's saved E1g
records (generation and every answer log-probability) bitwise on all 121
receivers. D4's statistics are interpreted only if it does.

## Statistics and readings (declared now)

Paired over the 121 receivers; receiver-level bootstrap, 3,000 resamples, new
seed 207100; one-sided 95% lower bounds. `hit(X)` means the greedy answer
equals `c`. The base rate is D3's unedited E0, which hit `c` in 0/121.

- **R4a — the edited AV description adds beyond the written number** iff,
  for every oracle `k`, the lower bound of `mean[hit(E1g) − hit(O_k)]` is
  above 0.
- **R4b — a hand-written text moves the answer as often** iff, for at least
  one oracle `k`, `mean hit(O_k) ≥ mean hit(E1g)` and the lower bound of
  `mean hit(O_k)` (against D3's E0 base rate of 0) is above 0.
- **R4c(k) — hand-written text `k` moves the answer**, reported per oracle,
  iff the lower bounds of both `mean hit(O_k)` and
  `L(O_k) = mean[logP(c) − logP(a)]` under `O_k`, minus the same under D3's
  E0, are above 0.

**How the readings will be read.**

- R4a met: the AV's wording carries something the AR uses beyond the number.
- R4b met (R4a then cannot be): D3's flips are explained by AR
  controllability; they say nothing about the descriptions reading the
  activation.
- Neither: report the paired intervals and call it unresolved.

Also reported, descriptively:

- each condition's retained-answer and off-target rates;
- the cosine between `AR(text)` and the original activation;
- the paired `hit(E1g) − hit(O_k)` with its lower bound in both directions.

**Predictions (not thresholds), recorded before any D4 forward:**

| Quantity | Prediction |
|---|---|
| O_structured hit rate | comparable to E1g (about 0.25–0.4); its register is closest to the AV's |
| R4b | met, by O_structured |
| R4a | not met |
| O_number | at most about 0.1 |
| O_slot, O_terse | uncertain |

## Cost, stopping rule, limits

- **Projected cost.** About 1 h: AR about 605 reconstructions in a few
  minutes, target 5 × 121 patched conditions at about 3–4 s each, plus loads
  and hash verification.
- **Start condition.** Only on an idle GPU with at least 60 GiB available.
- **Throughput stop.** If the target stage projects beyond three times one
  hour, the run stops and is reported.
- **Stopping rule.** Exactly one run, and whatever happens is reported. No
  text, value or condition changes after outcomes are read. A source bug
  permits only a regression test plus a visibly versioned rerun.
- **Limits.**
  - Oracle texts are hand-written intervention instruments, never read-out
    semantics.
  - One model, one site, one task family, one split.
  - D4 compares texts at a fixed number of mentions per text; it does not
    vary repetition.
  - A difference between E1g and an oracle can come from anything in the
    text: length, register, repetition or content. Only R4b (an oracle doing
    as well) supports a clean conclusion.
