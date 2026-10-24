# Repeated updates under a bounded replay allowance: no prespecified gap for evolutionary proposals — 2026-10-24

**Protocol:** [`protocols/repeated_updates_v1.md`](../protocols/repeated_updates_v1.md), revision 3, frozen at
`e8c6f99` before any run on confirmation worlds 900–902.

**Raw results:** [`experiments/results/ti_tnn/repeated_updates/`](../../../experiments/results/ti_tnn/repeated_updates/).
It holds the complete weight-free export of all 15 units (27,580 attempts) and the verifier's output.

**Verification:** `bench/repeated_updates_verify.py`.
- **What it re-derived:** every attempt of every chain, and every number below, without calling the
  producer's training, selection or gate code.
- **Source binding:** the run recorded `e8c6f99` with a clean tree. Its producer-file hashes equal those fixed in
  the freezing commit.
- **Sampled reproduction:** 114 complete chains (one per arm and world, chosen by a hash fixed before the run)
  were retrained. All 114 gave identical records.

## The question, by example

World 900 has 255 facts in five relations, and a shared network answers every one of them right. Twelve
one-fact requests arrive in a fixed order: six capital moves and six operator changes.

**How a request is handled.** For each request the deployed network is fine-tuned on the changed fact plus a
replay set of at most B unchanged facts.
- **The allowance:** B = ⌊β·|F|⌋ facts per attempt. At β = 1/16 that is 15 facts in world 900.
- **The gates:** a candidate must answer every fact of the updated world right and pass every mandatory gate.
  Then it is activated.
- **Retries:** up to 8 attempts per request, each with a fresh replay draw. Otherwise the request is refused.

**Why replay matters.** PR 8 showed that without replay, updating `capital_of` damages `largest_city_of`, the
relation with the same type signature (Country → City).

**What the study asks.** Which ordinary ways of choosing replay need the least of it? And is there a
prespecified gap, between the best ordinary method and the 90% target, that evolutionary proposals (plan PR 9)
could aim at?

## Answer

**There is no prespecified gap at any allowance.** The best ordinary method reached the 90% target in at least
two of the three worlds at every β. So PR 9 stays deferred, as the protocol fixed in advance.

**The method that closes the gap reads exhaustive error lists.** It is `check_guided`, which replays the facts
the earlier attempts got wrong.
- **At β = 1/16 the outcome rests on the threshold itself:** world 900 reached exactly 54 of 60 (90.0%), world
  901 56, world 902 50.
- **The other three policies,** reported descriptively, stayed below 90% in at least two worlds through
  β = 1/4. Only at 1/2 did they reach every request.

**All four predictions hold, in every world:**
- full rehearsal is a valid reference;
- no replay fails;
- early stopping limits drift cheaply;
- at β = 1/4, replaying the complete same-signature pool first beats replaying the updated relation first, on
  capital moves.

## Results

**Bases:** all 15 shared and all 15 specialist bases were exact, so every world qualified for the decision.

### Predictions (frozen; one-sided; must hold in every world)

| # | Prediction | World 900 | World 901 | World 902 | Holds |
|---|---|---|---|---|---|
| R1 | Full rehearsal accepts ≥ 95% (57 of 60) | 60 | 58 | 60 | yes |
| R2 | No replay accepts ≤ 50% (30 of 60) | 7 | 5 | 1 | yes |
| H1 | Capital moves at β = 1/4, fixed stopping: `signature` − `relation_first` ≥ 20 points (of 30) | 29 v 6 | 29 v 4 | 23 v 4 | yes |
| H2 | At β = 1/16, early stopping raises `random` by ≥ 20 points with ≤ 10% of the steps (of 60) | 25 v 7, 1.9% | 22 v 8, 2.3% | 20 v 5, 2.5% | yes |

**H1 is a pilot-informed revision.**
- **Revision 2** placed it at β = 1/8, where it failed on development.
- **Revision 3 moved it to 1/4**, where the 47-fact signature pool fits the allowance.
- **The confirmation tests the revised hypothesis.** It does not rescue the original.
- **At 1/8, descriptively:** 19 v 3, 24 v 3 and 16 v 3 of 30. Those differences are large too, but there the
  pool is subsampled.

### The PR 9 decision (frozen rule; best ordinary arm, accepted of 60)

| β | World 900 | World 901 | World 902 | Outcome |
|---|---|---|---|---|
| 1/16 | 54 (90.0%) | 56 | 50 | no prespecified target gap |
| 1/8 | 59 | 59 | 56 | no prespecified target gap |
| 1/4 | 59 | 60 | 59 | no prespecified target gap |
| 1/2 | 60 | 60 | 60 | no prespecified target gap |

- **Which arm was best:** `check_guided` at β ≤ 1/4. At 1/2 it was `signature` with fixed stopping, tied with
  `check_guided` in world 901.
- **The rule:** "no prespecified target gap" needs the best ordinary arm at ≥ 90% in at least 2 qualifying worlds.
  At 1/16, worlds 900 and 901 met it; world 900 met it exactly.

### Descriptive: the best ordinary arm without `check_guided` (accepted of 60)

These three policies do not use earlier attempts' error lists to choose replay. They still pass every check.
This summary does not enter the decision.

| β | World 900 | World 901 | World 902 | Winning arm |
|---|---|---|---|---|
| 1/16 | 38 | 40 | 27 | `relation_first` or `signature`, early stopping |
| 1/8 | 45 | 48 | 43 | `signature` |
| 1/4 | 53 | 54 | 46 | `signature`, fixed |
| 1/2 | 60 | 60 | 60 | `signature`, fixed |

### The frontier (accepted of 180 over the three worlds; fixed / early stopping)

| Policy | β = 1/16 | 1/8 | 1/4 | 1/2 |
|---|---|---|---|---|
| `random` | 20 / 67 | 36 / 76 | 53 / 101 | 108 / 129 |
| `relation_first` | 49 / 98 | 78 / 101 | 88 / 124 | 123 / 144 |
| `signature` | 55 / 104 | 136 / 127 | 153 / 143 | 180 / 165 |
| `check_guided` | 154 / 160 | 172 / 159 | 178 / 162 | 178 / 165 |

**References:** no replay 13; full rehearsal 178.

**What the frontier shows:**
- **Targeting helps at every allowance.** `relation_first` beats `random`, and `signature` beats `relation_first`.
- **The gain is in capital moves.** With fixed stopping at β = 1/4, capital moves were accepted 6 of 90 times
  under `random`, 14 under `relation_first`, and 81 under `signature`.
- **The operator-change control behaves as predicted.** For operator changes, `signature` and `relation_first`
  draw from the same pool, and they accepted 72 and 74 of 90.
- **Early stopping helps the weak policies at every β,** and costs the strong ones at large β. For example,
  `signature` at 1/2 accepted 180 with fixed stopping and 165 with early stopping.

### Costs per request (three worlds pooled)

| Arm | Accepted of 180 | Distinct replay facts | Trained attempts | Optimizer steps | Example-steps | Full checks |
|---|---:|---:|---:|---:|---:|---:|
| full rehearsal | 178 | 283 | 1.0 | 300 | 85,200 | 1.0 |
| `signature` 1/2, fixed | 180 | 151 | 1.2 | 360 | 51,433 | 1.1 |
| `signature` 1/4, fixed | 153 | 84 | 2.5 | 745 | 53,692 | 1.1 |
| `check_guided` 1/4, fixed | 178 | 174 | 3.8 | 1,145 | 82,230 | 1.3 |
| `check_guided` 1/8, fixed | 172 | 108 | 4.5 | 1,340 | 48,300 | 1.3 |
| `check_guided` 1/16, early | 160 | 41 | 3.0 | 26 | 476 | 1.2 |
| `random` 1/16, early | 67 | 83 | 5.9 | 51 | 919 | 0.8 |

**`check_guided`'s per-attempt allowance understates what it sees.** At β = 1/8 an attempt replays at most
31–37 facts, but a request saw 108 distinct replay facts on average. After every attempt the method read an
exhaustive error list over all facts. In a larger domain only sampled checks would be available. So this arm is
an optimistic ordinary method, as the protocol says.

### The specialist contrast (descriptive)

Only the updated relation's network is trained, with replay drawn from that relation's own facts:

| Replay (of its relation) | Facts per attempt | Accepted of 180 |
|---|---|---:|
| none | 0 | 143 |
| 1/8 | 3 (capital) / 8–10 (operator) | 168 |
| 1/4 | 6 / 17–21 | 176 |
| 1/2 | 12 / 34–42 | 179 |

**The networks differ in size.** A specialist has 7,365 parameters (`capital_of`) or 4,710 (`operator_of`); the
shared network has 20,616.
- **Specialists cannot interfere across relations, by construction.** Their failures are drift within the
  updated relation, mostly operator changes.
- **The β values are not comparable** with the shared ones. A specialist's β is a fraction of one relation; the
  shared β is a fraction of the world.

### Construction checks (rebuilt by the verifier)

| Check | Exercised | Failures |
|---|---:|---:|
| Every attempt re-derived: parent, worlds, training set, duplicates, table digest, screen, stopping, audit rule, acceptance | 27,580 attempts (1,446 duplicates) | 0 |
| Screen-passing attempts that failed a mandatory gate | 4,643 | 0 |
| Screened-out attempts fully checked, and disagreeing with the screen | 2,032 of 21,491 (9.5%) | 0 |
| Sampled reproduction: retrained chains with identical records | 114 | 0 |

## What this establishes, and what it does not

**It establishes**, in a finite synthetic world of about 250–300 facts with single-fact requests and one small
shared network:
- **The problem.** Sequential updates of a shared network fail without replay and succeed with full rehearsal.
- **Targeted replay.** Choosing replay by relation, and then by declared type signature, lowers the replay
  needed. On capital moves at β = 1/4, the complete same-signature pool beat the updated relation alone in every
  world. That is the confirmed H1.
- **No prespecified target gap for PR 9** at any tested allowance. The best ordinary method reached the 90% target
  in at least two of three worlds at every β, so PR 9 stays deferred.

**It does not establish:**
- **Why signature replay helps.** A larger β also changes total replay and the training mixture. The confirmation
  supports the revised H1; it does not isolate pool coverage as the cause.
- **That ordinary methods would suffice without exhaustive error lists.** `check_guided` closes the gap because it
  reads a complete error list after every attempt. Without it, the best ordinary arm stayed below 90% in at least
  two worlds through β = 1/4. That is descriptive, and the frozen decision does not count it. A study where only
  sampled checks are available would need its own protocol.
- **That the decision at β = 1/16 is robust.** It turned on world 900's 54 of 60, exactly the threshold. The
  development pilot, on two other worlds, had found a gap at 1/16.
- **Anything beyond this world.** The allowance is a stand-in for the cost of rehearsal when facts are many. The
  study shows what each method needs at each fraction, not that the fractions carry over. Natural language,
  larger domains and imperfect deployed models are out of scope: every accepted candidate was exact.
- **That evolutionary proposals would or would not help.** No proposal method was tested. The rule only decides
  that this study gives PR 9 no prespecified target.

## How it got here

1. **Revision 1** ([#61](https://github.com/seanm27lol/universa-recurrent/pull/61)), with two exploratory
   development probes.
2. **Revision 2** ([#63](https://github.com/seanm27lol/universa-recurrent/pull/63)), after its review:
   - a three-outcome decision;
   - the `relation_first` control;
   - an immutable schedule, a per-attempt allowance and a duplicate rule;
   - acceptance as the screen plus every mandatory gate;
   - an explicit verification boundary.

   Implementing it exposed that the gates could not read a shared network's identity or cost. That was
   repaired first ([#62](https://github.com/seanm27lol/universa-recurrent/pull/62)).
3. **The development pilot** (worlds 100–101 × inits 100–101):
   - H1 failed at β = 1/8;
   - the decision found a gap at 1/16 only.
4. **Revision 3,** after the review of revision 2, then frozen:
   - H1 moved to 1/4;
   - the descriptive maxima were added, with and without `check_guided`.

## Evidence

**In the bundle,** at `experiments/results/ti_tnn/repeated_updates/`:
- `20261024T035329/run.json`;
- the 15 units' `unit.json` and `proposals.jsonl.gz` (3.5 MB in all);
- `verification.json`, the verifier's complete output, including the sampled reproduction.

**Rebuilding:**

```bash
python experiments/results/ti_tnn/make_tables.py --repeated-updates                       # re-derives everything, seconds
python experiments/results/ti_tnn/make_tables.py --repeated-updates --repeated-reproduce  # also retrains 114 chains, ~20 min
```

Both require the result to equal `verification.json`.

**The run:**
- **source:** `e8c6f99`, with a clean tree;
- **software:** PyTorch 2.10.0, Python 3.11.14, aarch64;
- **compute:** CPU, 4 threads, low priority;
- **wall time:** chains 85 minutes, 2026-10-24 03:53–05:19 UTC; verification with sampled reproduction about
  19 minutes.
