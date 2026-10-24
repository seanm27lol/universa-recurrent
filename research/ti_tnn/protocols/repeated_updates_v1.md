# Repeated updates under a bounded replay allowance (draft, revision 2)

**Status: DRAFT, revision 2 (2026-10-23), after the review of revision 1, with the development pilot appended.**
Nothing here is frozen, and no run has touched confirmation worlds 900–902.
- **What is authoritative:** the specification below. "History" keeps revision 1, the review's decisions,
  the exploratory probes and the development pilot.
- **The code it specifies:** `bench/repeated_updates.py` (the harness) and `bench/repeated_updates_verify.py`
  (the independent verifier).
- **What it builds on:** PR 8's gates and deployment, with the prerequisite repair in #62. That repair lets
  the gates read a shared network's identity and cost.

This is item 4 of the [integration plan's section 12](../integration_plan.md#12-second-review-amendments-2026-10-18-after-prs-4150)
order. The plan defers evolutionary proposals (plan PR 9) until there is a measurable objective, and names one
candidate: "repeated updates with a bounded replay allowance". Any proposal method would be compared with
"ordinary fine-tuning, early stopping and matched random search".

**This study is a small diagnostic of update and rehearsal failure.** It measures whether ordinary methods
leave a gap, and it tests no proposal method. Every deployed model in it is exact by construction (only exact
candidates are accepted). So it is not the imperfect-component serving experiment; that follows the
interpreter work (see the plan).

## A concrete example

World 100 has 276 facts in five relations. One shared network answers all of them, and every answer is right.
Twelve one-fact requests arrive in a fixed order. The first moves `country_6`'s capital from
`country_6_city_3` to `country_6_city_0`.

**The candidate.** The deployed network is fine-tuned on two things:
- the changed fact;
- a **replay set** of at most B = ⌊276/8⌋ = 34 unchanged facts, for allowance β = 1/8.

**The policies.**
- **`relation_first`:** replays other `capital_of` facts first.
- **`signature`:** also gives priority to `largest_city_of`, which has the same declared type signature
  (Country → City).

**The gates.** The candidate is screened: every one of the 276 facts of the updated world must be answered
right. It is then judged by every mandatory gate of PR 8's checker.
- **Accepted:** it is activated, and the next request starts from it.
- **Rejected:** another proposal is made with a fresh replay draw, up to 8 in all. If none is accepted, the
  request is refused, and the deployed world keeps the old capital.

In a development probe (one world, one init; "History"), replaying 38 facts gave 11 of 12 requests accepted
with signature-first replay, and 2 of 12 with uniform replay. The study asks whether such differences hold on
fresh worlds, what they are due to, and whether any ordinary method reaches full rehearsal's acceptance under
a small allowance.

## The question

**Under a bounded per-proposal replay allowance, how many of twelve requests can a shared network absorb while
passing every gate, and which ordinary methods need the least replay?**

1. **Does the problem exist** on fresh worlds? Without replay most requests should fail; with full rehearsal
   almost none should. PR 8 showed this for single updates.
2. **What targets replay?**
   - **The task being changed.** Does `relation_first` beat uniform replay?
   - **Cross-relation type information.** Does `signature` beat `relation_first`? The two differ only on capital
     moves: `operator_of` has no other relation with its signature, so for operator changes both policies
     draw from the same pool. Operator changes are then a control.
3. **Is there a gap for PR 9?** Across the whole replay frontier, does the best ordinary method fall short
   of the declared target where the reference meets its bar?

## Definitions

**Facts and keys.** F(w) is every fact (relation, source, target) of world w over the five relations of
`generate_world`, and every relation is total on its source type. The **canonical key list** is every
(relation, source), with relations and sources sorted; indices into it name facts. Its identity is exported.

**The schedule** is immutable: the same for every arm and init of a world.
- **Twelve requests:** six distinct countries' capitals and six distinct airports' operators. Each
  destination is drawn once against the original world, and the twelve come in one shuffled order (`schedule`).
- **Refusals change the history, not the schedule.** A refused request does not land, so later requests
  start from a different deployed history. They are still the same requests.
- **Every request is a real change.** Sources are distinct, so the deployed value at a request's turn is
  always the original.

**A candidate** is trained from the deployed network:
- **the training set:** the edit plus a replay set R of unchanged facts, in canonical order;
- **the recipe:** full-batch Adam at learning rate 3e-3, PR 8's recipe;
- **the allowance is per proposal:** |R| ≤ B = ⌊β·|F|⌋, with β ∈ {1/16, 1/8, 1/4, 1/2}.

The shared network has no dropout. So a candidate is a deterministic function of three things: its parent,
its training set and its stopping rule.

**Proposals.** Each request gets up to N = 8 attempts.
- **What varies between attempts is only the replay draw**, seeded from (world, init, arm, request,
  attempt). This is **replay-resampling search**: a candidate is not randomized beyond its training set.
- **Duplicates.** An attempt whose training set repeats an earlier one for the same request is not trained or
  checked. It is recorded as a duplicate and still uses an attempt. For no replay and full rehearsal, attempts
  2–8 are always duplicates.
- **The first accepted attempt is activated.** If none is accepted, the request is refused.
- **Recorded per attempt:** the replay size, the cumulative distinct replay facts of the request, the steps and
  example-steps, and the screen and checker work.

**Replay policies** (shared network):

| Policy | Priority pool (unchanged facts), then uniform fill up to B |
|---|---|
| `random` | none |
| `relation_first` | the updated relation's own facts |
| `signature` | facts of every relation with the updated relation's declared (source, target) signature, itself included |
| `check_guided` | facts that any earlier attempt for this request got wrong; the first attempt is therefore `random` |

**How the pool is used.**
- **A pool larger than B** is sampled uniformly; that is the seeded selection.
- **A smaller pool** is taken whole and filled uniformly from the remaining unchanged facts.
- **`check_guided` uses exhaustive error lists.** It is counted as an ordinary method, and its checking work is
  part of its cost.

**Stopping.**
- **`fixed`:** 300 steps, as in PR 8.
- **`early`:** stop at the first multiple of 5 steps at which the edit is answered right, at most 300.

**Arms.**
- **Ordinary arms:** 4 policies × 4 allowances × 2 stopping rules = 32.
- **References** (fixed stopping):
  - **no replay** (β = 0);
  - **full rehearsal:** every unchanged fact.
- **Specialists** (a descriptive contrast):
  - only the updated relation's network is trained;
  - its replay is drawn uniformly from that relation's own unchanged facts, at β ∈ {0, 1/8, 1/4, 1/2} of the
    relation's size;
  - fixed stopping, N = 8, the same duplicate rule.

  Its β is relative to one relation, the shared β to the whole world. So it is reported in absolute replay
  counts, with parameter counts and training cost beside it, never compared by the value of β alone.

**Acceptance requires three things:**
- **the screen:** atomic exactness, every fact of the target world answered right;
- **every mandatory gate** of `update_checks.Checker` passing against the deployed and target worlds:
  record context, identity and scope, types and laws, intended and unaffected behaviour, trace and
  continuation, and calibration and resources;
- **activation** by the deployment.

**The declared scope** of a shared candidate is the whole network. A shared network cannot be write-local
(PR 8), so it is judged on behaviour. A specialist candidate's scope is its relation.

**Checking work:**
- **Every screen-passing attempt** gets the full check.
- **A screened-out attempt** gets the full check when either holds:
  - the frozen audit rule selects it: sha256 of (world, init, arm, request, attempt) ≡ 0 mod 20, about 5%;
  - it is the first screened-out attempt of its update kind in its chain.
- **Every gate's outcome is exported separately.**

**The screen and the behavioural gates.** Every relation is total, and a module answers by argmax over its
target type. So the screen agrees with the behavioural gates, intended and unaffected behaviour: both pass
exactly when every fact is right. Any disagreement on an attempt that was fully checked is a construction
failure. The audit tests the screening shortcut's implementation. It does not prove that a rare disagreement
cannot occur.

The screen does not stand in for the other gates. In the review, a candidate exact on every fact whose record
named the wrong world failed the record-context gate.

**Costs, per request and per arm:**
- attempts, duplicates and trained attempts;
- replay size per attempt, and cumulative distinct replay facts per request;
- optimizer steps and example-steps (training-set size × steps);
- screens, full checks and audits;
- wall time.

## Units and procedure

- **Worlds:** confirmation worlds **900, 901, 902**, used by no earlier run. Development uses worlds 100–101.
- **Inits:** confirmation 0–4; development 100–101.
- **Bases.** Each unit trains a shared dense network and a specialist family as in PR 4 (1,500 steps,
  lr 1e-2).
  - **Every init stays in the inventory.** A failed base is never replaced with another seed.
  - **An inexact shared base:** that unit's shared arms do not run, and its world does not qualify for the
    decision.
  - **An inexact specialist base:** only that unit's specialist contrast is missing.
- **Per world:** 5 inits × 12 requests = 60 requests per arm. The twelve requests of a chain are dependent, so
  the 60 are not independent trials. The bars below are research-priority thresholds, not confidence
  statements.

## Predictions and the PR 9 decision (one-sided; bars confirmed by the development pilot before freezing)

**When a prediction holds.** It must hold in every world, on the per-world aggregate over the chains that
ran.

| # | Prediction | Kind |
|---|---|---|
| R1 | Full rehearsal accepts ≥ 95% of requests | Empirical; sanity, replicating PR 8 |
| R2 | No replay accepts ≤ 50% of requests | Empirical: the problem exists |
| H1 | On capital moves, at β = 1/8 with fixed stopping, `signature` accepts at least 20 points more requests than `relation_first` | Empirical: cross-relation type information targets replay |
| H2 | At β = 1/16, early stopping raises `random`'s acceptance by at least 20 points, using at most 10% of fixed stopping's optimizer steps | Empirical: ordinary early stopping limits drift |

**Reported without a prediction:**
- `relation_first` against `random`;
- operator changes as H1's control;
- the whole frontier: every arm, every β, by update kind;
- the attempt at which each request was accepted, within the N = 8 history. That is not a re-run with a
  smaller N, because a different N changes the deployed history;
- costs;
- the specialist contrast.

**The PR 9 decision, recorded before any confirmation run.** It is evaluated for each β.
- **Ordinary arms:** every policy (`random`, `relation_first`, `signature`, `check_guided`) with either stopping
  rule. The best of them, per world, is the one with the highest acceptance.
- **A world qualifies** when two things hold:
  - all 5 of its shared bases are exact;
  - full rehearsal accepts ≥ 95% of its requests, that is at least 57 of 60.

| Outcome at β | Rule |
|---|---|
| **Prespecified gap** | In at least 2 qualifying worlds, the best ordinary arm accepts < 90% (at most 53 of 60) |
| **No prespecified target gap** | In at least 2 qualifying worlds, the best ordinary arm accepts ≥ 90% |
| **Inconclusive** | Neither: fewer than 2 qualifying worlds, or the qualifying worlds split |

**What follows.**
- **A prespecified gap at some β makes PR 9 eligible, not automatic.** Its target is chosen by reviewing the
  whole frontier and costs. The smallest failing allowance is not automatically the most valuable target.
  PR 9's random-search baseline must search the same recipe space PR 9 searches: if PR 9 varies learning
  rates, loss weights or recipes, so does its baseline.
- **No prespecified target gap at any β:** PR 9 stays deferred, with that reason recorded.
- **Inconclusive results** are reported with their cause. PR 9 stays deferred.

## Export and independent verification

**Export** (weight-free JSON).
- **`run.json`:**
  - the provenance: producer-file hashes, commit, tree state, and Python, PyTorch and thread count;
  - the configuration and constants;
  - the arms;
  - each world's identity, schedule and key-list identity.
- **`unit.json`:**
  - for each base: its exactness, error list, table digest and snapshot digest;
  - the key-list and schedule identities;
  - each chain's status.
- **`proposals.jsonl.gz`,** one line per attempt:
  - arm, request and attempt;
  - the parent snapshot digest;
  - the deployed and target world identities;
  - the training set as key indices, and the replay size;
  - `duplicate_of`;
  - steps;
  - every wrong fact with its answer, and the table digest;
  - the screen;
  - whether it was audited, and every gate's outcome where checked;
  - the candidate snapshot digest;
  - the activation decision and the active snapshot after it;
  - wall time.

**The verifier** (`bench/repeated_updates_verify.py`) does not call the producer's training, selection or gate
code. It regenerates worlds, schedules and key lists, and implements the replay draws and the audit rule from
this text. For every attempt of every chain it re-derives:
- **the worlds and the parent:**
  - the deployed and target worlds: a refused request never lands;
  - the parent: the last **accepted** candidate, never a rejected attempt;
- **the training set:** the training set and its duplicate status. For `check_guided`, it is rebuilt from the
  earlier attempts' exported errors;
- **the table:**
  - every error entry: in range, not repeated, actually wrong, inside its type;
  - the candidate's table: the target world's facts with exactly those entries substituted, whose digest must
    match;
- **the checks:**
  - the screen;
  - consistency with the stopping rule: fixed attempts take 300 steps; an early stop is at a multiple of 5 with
    the edit right;
  - which attempts had to be fully checked;
  - the screen's agreement with the behavioural gates;
  - acceptance: screen, every mandatory gate and activation of that candidate;
- **the summaries:** every count, cost, rate, prediction and the decision.

It also enforces a frozen inventory: worlds, inits and arms, none missing or repeated. It binds the run to the
frozen sources, and compares every record field by field.

**Sampled reproduction.** `--reproduce` reruns a sample of complete chains with the producer's code:
- **the sample:** one complete chain per arm and world. The init is chosen by sha256 of (world, arm), which is
  fixed before the run and independent of any outcome;
- **the environment:** pinned, with Python, PyTorch and the thread count recorded;
- **the requirement:** identical records in every field but wall time. That covers error lists, table digests,
  step counts, accepted attempts, and the deployment history with its snapshot digests.

This is sampled reproduction, not independent reproduction of every trained model.

**What the verification can and cannot catch:**
- **Caught everywhere:** an error hidden without updating the digest; an altered replay set; a wrong parent; a
  refused request treated as landed; a flipped acceptance; an unflagged duplicate; a missing full check; a
  missing or extra attempt; a malformed error entry; an inconsistent step count; an incomplete inventory.
- **Caught only in reproduced chains:** a coherently altered behaviour report, with the errors and digest
  changed together. Elsewhere, an exported table remains the producer's record of the network's behaviour.

The corruption tests demonstrate both.

## Interpretation limits

- **A finite synthetic world:** about 280 facts in five relations, single-fact requests, one small shared
  network and one optimizer recipe.
- **The allowance is a stand-in.** Full rehearsal is cheap here. β models the cost of rehearsal when facts
  are many; the study shows what each method needs at each fraction, not that the fractions carry over.
- **`check_guided` relies on exhaustive error lists.** A larger domain would have only sampled checks, so this
  arm is an optimistic ordinary method.
- **`signature` relies on declared signatures,** which is the typed interface this project supplies.
- **Acceptance is strict.** One wrong fact anywhere rejects a candidate. That is the gates' definition, not a
  tolerance chosen here.
- **Exact deployed models.** Every accepted candidate is exact, so this diagnoses update failure. It does not
  measure serving with imperfect components.
- **The PR 9 decision is a research-priority gate, not a result about proposals.**

## Decisions (from the review of revision 1)

| # | Decision | As specified above |
|---|---|---|
| 1 | Specialist contrast | Included, descriptive. It is reported in absolute replay counts with parameter counts and costs; fixed stopping; N = 8 |
| 2 | β grid and N = 8 | Kept, with the allowance per proposal, cumulative distinct replay reported, and only the replay draw varying between attempts |
| 3 | 95% / 90% / two of three worlds | Kept as research-priority thresholds, with an explicit inconclusive outcome and a base-validity rule |
| 4 | `check_guided` | Kept as an ordinary method, with its checking counted. `relation_first` is added and counts among the ordinary methods |
| 5 | Sampled reproduction | One complete chain per arm and world, chosen independently of outcomes, with exact agreement required |
| 6 | Update kinds | PR 8's capital moves and operator changes, as one immutable schedule per world, reported by kind |
| 7 | Audit of screened-out attempts | The frozen 5% rule plus a minimum of the first screened-out attempt per update kind per chain. Every screen-passing attempt is fully checked, and the audited denominator is reported. Tests audit every attempt |

## History

### Revision 1 (2026-10-23, reviewed at `7b8c207`)

Revision 1 had the same question and grid. The review kept the design and asked for six changes, all made
above:
- **A three-outcome decision,** with an inconclusive outcome. Revision 1's "no gap" did not follow from its
  rule: a failed reference, or ordinary methods 10 points behind a perfect reference, both failed the gap
  predicate.
- **A denominator rule for inexact bases.**
- **The `relation_first` control,** so that a benefit is not attributed to signatures when it may come from
  rehearsing the task being changed. H1 now compares `signature` with `relation_first` on capital moves.
- **The schedule and the allowance.** An immutable schedule, an allowance per proposal, and a duplicate rule:
  with full-batch training and no dropout, a fresh seed alone does not change the candidate.
- **Acceptance.** It is the screen and every mandatory gate, not the screen alone.
- **An explicit verification boundary.** Every table is rebuilt from its errors. Reproduction is sampled and
  chosen independently of outcomes.

**The repair it led to.** Implementing revision 2 showed that the gates could not read a shared network's
identity or cost: both gates reported insufficient evidence for every shared candidate. That is repaired in
#62.

### Exploratory development probes (2026-10-23, world 100, init 100; disclosed)

Both probes ran before revision 1, to choose a regime that is neither at ceiling nor at floor. They used
scratch code, not this harness, one sequence each, and absolute replay sizes: 19, 38, 75 and 150 facts, that
is 7%, 14%, 27% and 54% of world 100's 276.

**Probe 1: ungated, 12 updates, every candidate applied, two replay draws per arm.** Exact candidates, of 12:

| Arm | Exact |
|---|---:|
| no replay | 1 |
| no replay, early stopping | 1, with unaffected errors 2–27 instead of 29–71, at 3% of the optimizer steps |
| `random`, 128 facts (46%) | 0–1 |
| signature-first, 128 facts (46%) | 10–12 |
| same-relation-first, 128 facts (46%) | 1–3 |
| full rehearsal | 12 |

**Probe 2: gated, with up to 8 proposals.** Accepted, of 12 (fixed / early stopping):

| Replay facts (of 276) | `random` | `signature` | `check_guided` (previous attempt's errors) |
|---|---|---|---|
| 19 (7%) | 1 / 6 | 5 / 9 | 3 / 8 |
| 38 (14%) | 2 / 7 | 11 / 9 | 4 / 9 |
| 75 (27%) | 4 / 6 | 10 / 9 | 7 / 9 |
| 150 (54%) | 8 / 9 | 12 / 10 | 12 / 10 |

- **References:** no replay 1; full rehearsal 12, each on its first proposal.
- **Specialists,** with own-relation replay of 0, 2, 4 and 8 facts and of all of them: 9, 9, 11, 11 and 12.
- **The probes' schedule** was drawn sequentially, not as revision 2's immutable schedule. Their proposals
  reseeded identical training sets, so some of their retries were duplicates.

### Development pilot (2026-10-24 UTC, worlds 100–101 × inits 100–101; revision 2's harness at `e67126e`)

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python research/ti_tnn/bench/repeated_updates.py \
        --worlds 100 101 --inits 100 101 --out <dir>

The harness has changed since only in one docstring word.

**Verification** (`repeated_updates_verify.verify(..., development=True, reproduce=True)`):
- **Re-derivation:** every one of 7,504 attempts re-derived. Of them, 392 were duplicates and 7,112 were trained.
- **Screen-passing attempts:** 1,212. Every one passed every mandatory gate and was activated.
- **Screened-out attempts:** 540 of 5,900 were fully checked (9.2%: the 5% rule plus the per-kind minimum).
  None disagreed with the screen.
- **Sampled reproduction:** 76 of 76 chains identical, one per arm and world.
- **Bases:** all shared and specialist bases were exact.

**Accepted requests, both worlds (of 48), with fixed / early stopping:**

| Policy | β = 1/16 | 1/8 | 1/4 | 1/2 |
|---|---|---|---|---|
| `random` | 3 / 20 | 7 / 20 | 15 / 27 | 25 / 35 |
| `relation_first` | 15 / 25 | 20 / 24 | 23 / 34 | 32 / 39 |
| `signature` | 11 / 28 | 24 / 38 | 43 / 36 | 47 / 43 |
| `check_guided` | 38 / 39 | 47 / 45 | 48 / 44 | 48 / 41 |

**References and contrast:**
- no replay: 2;
- full rehearsal: 48;
- specialists at β = 0, 1/8, 1/4 and 1/2 of their relation: 38, 44, 48 and 48. Those β mean 0, 3–10, 6–20
  and 12–40 replayed facts.

**Predictions on development:**
- **R1 holds:** full rehearsal 48 of 48.
- **R2 holds:** no replay 2 of 48.
- **H2 holds.** At 1/16, early stopping raised `random` from 1 to 11 of 24 in world 100 and from 2 to 9 in world
  101, with 2% of fixed stopping's optimizer steps.
- **H1 fails in world 100.** On capital moves at 1/8, fixed stopping, `signature` against `relation_first` was
  0 v 0 of 12 in world 100 and 3 v 0 in world 101.

**The decision on development:**
- **β = 1/16: a prespecified gap.** The best ordinary arm accepted 21 of 24 (87.5%) in world 100 and 19 of 24
  (79.2%) in world 101; both were `check_guided`.
- **β ≥ 1/8: no prespecified target gap.** `check_guided` with fixed stopping accepted 23–24 of 24.

**What the pilot shows:**
1. **Fixed stopping drifts on capital moves unless the same-signature relation is rehearsed.** With fixed
   stopping, `random` and `relation_first` accepted at most 2 of 24 capital moves at every β up to 1/4. At
   1/2 they accepted 4 and 8 of 24.
2. **`signature` works when its pool fits.** It went from 3 of 24 capital moves at 1/8 to 23 of 24 at 1/4.
   That is where its pool fits the allowance:
   - the pool is `capital_of` plus `largest_city_of` minus the edit, 47 facts in every world;
   - B is 34–36 at 1/8 and 69–72 at 1/4.

   **H1 failed because at 1/8 the pool is subsampled.** At 1/4, capital moves are 23 v 1 of 24 against
   `relation_first`.
3. **`check_guided` reaches the target from β = 1/8.** Its per-attempt allowance understates what it sees:
   - **cumulative distinct replay:** at 1/8 about 79–105 facts per request, 29–38% of the facts;
   - **exhaustive error lists:** it reads one after every attempt.
4. **Early stopping helps the weak policies and costs the strong ones.** It helps `random` and `relation_first`
   at every β. It lowers `signature` and `check_guided` at large β.

**Cost.**
- **Chains:** 337 s per unit for all 38 arms.
- **Verification:** about 70 s for 4 units. Reproduction takes about 9 s per chain.
- **Estimated confirmation:** 15 units about 85 min, verification with reproduction of 114 chains about 20 min.
  About 1 h 45 min of CPU in all, at 4 threads.

**Proposed for revision 3, for the reviewer to decide before freezing:**
1. **Move H1 from β = 1/8 to β = 1/4.**
   - **At 1/8,** the 47-fact signature pool exceeds B in every world, including 900–902: B = 31–37. That comes
     from the generator's fact counts, with no training or run on those worlds.
   - **At 1/4,** it fits: B = 63–75.
   - **The other option** is to keep 1/8 as revision 2 states. H1 would then test subsampled pools, and the
     pilot predicts it fails.
2. **Report descriptively the best ordinary arm without `check_guided`,** beside the decision. The decision
   itself is unchanged: `check_guided` stays an ordinary method, as decided.
