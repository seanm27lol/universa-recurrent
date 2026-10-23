# Repeated updates under a bounded replay allowance (draft, revision 1)

**Status: DRAFT for review, revision 1 (2026-10-23).** Nothing here is frozen, and no run has touched
confirmation worlds 900–902. Two exploratory development probes on world 100 informed the draft; they are
disclosed in "History".

This is item 4 of the [integration plan's section 12](../integration_plan.md#12-second-review-amendments-2026-10-18-after-prs-4150)
order. The plan defers evolutionary proposals (plan PR 9) until there is a measurable objective, and names one
candidate: "repeated updates with a bounded replay allowance". Any proposal method would be compared with
"ordinary fine-tuning, early stopping and matched random search, counting every proposed and rejected
candidate and every check". **This study measures whether those ordinary methods leave a gap.** It does not
test proposals.

## A concrete example

A world has 24 countries and about 280 facts in five relations. One shared network answers all of them, and
every answer is right. A sequence of twelve one-fact updates arrives, one at a time. The first moves
`country_7`'s capital to another of its cities.

**The candidate.** The deployed network is fine-tuned on two things:
- the changed fact;
- a **replay set**: at most B of the facts that did not change, here B = 38 (14% of world
  100's 276 facts).

**The gates.** The candidate is checked against the updated world by PR 8's gates. Their exhaustive behaviour
gates require every answer to be right: the changed fact, and all ~280 others.
- **Accepted:** it is activated, and the next update starts from it.
- **Rejected:** another proposal is made, with a fresh seed and a fresh replay draw, up to 8 in all. If none
  passes, the update is refused, and the world keeps the old capital.

**What the replay set changes,** in one development sequence on world 100 (disclosed in "History"; one world
and one init, so noisy):

| Replay chosen | Updates accepted of 12, within 8 proposals |
|---|---:|
| none | 1 |
| 38 facts drawn uniformly | 2 |
| 38 facts, the updated relation's signature first (`capital_of` and `largest_city_of` are both Country → City) | 11 |
| all ~280 facts (full rehearsal) | 12, each on its first proposal |

The question is whether that pattern holds on fresh worlds, and whether any ordinary method reaches full
rehearsal's acceptance under a small allowance.

## The question

**Under a bounded replay allowance, how many updates in a sequence can a shared network absorb while passing
every gate? Which ordinary methods need the least replay, and do they reach full rehearsal's acceptance?**

Three sub-questions:
1. **Does the problem exist** on fresh worlds? Without replay, most updates should fail; with full
   rehearsal, almost none (PR 8 showed this for single updates).
2. **Do the declared relation signatures target replay?** PR 8 found that interference follows relation
   identity: updating `capital_of` damaged `largest_city_of`, the relation with the same endpoints. If the
   typed interface predicts where damage falls, replaying same-signature facts first should need less replay
   than uniform draws.
3. **Is there a gap for PR 9?** Does the best ordinary method fall short of full rehearsal at some allowance
   well below full rehearsal? If not, PR 9 stays deferred with a stated reason.

## Definitions

**Facts.** F(w) is every (relation, source, target) fact of world w, over the five relations of
`generate_world`. Every relation is total on its source type.

**An update** changes one fact, drawn as in PR 8: a capital moves to another city of the same country, or an
airport changes operator.

**A sequence** is 12 updates, 6 of each kind in a drawn order. Each is drawn against the deployed world at
its turn. Its target world is the deployed world plus that one change. A refused update does not land; the
next update is drawn against the world as deployed.

**A candidate** fine-tunes the deployed network with Adam at learning rate 3e-3 (PR 8's recipe) on:
- the edit;
- a replay set R of unchanged facts of the target world, with |R| ≤ B = ⌊β·|F|⌋.

**The allowance** β ∈ {1/16, 1/8, 1/4, 1/2}.

**Acceptance.** Every mandatory gate of `update_checks.Checker` passes against the target world.
- **Declared scope:** the whole shared network. A shared network cannot be write-local (PR 8); its candidates
  are judged on behaviour.
- **The screen.** Every relation is total, and modules answer by argmax over the target type. So the
  behavioural gates pass exactly when every fact of the target world is answered correctly, which is atomic
  exactness. The harness screens each proposal by atomic exactness. It runs the full Checker on every
  proposal that passes the screen, and on a declared sample of those that fail it.
- **The equivalence** is tested: a screen and Checker disagreement on any checked proposal is a construction
  failure.

**Proposals.** Up to N = 8 per update. Each proposal draws a fresh seed and a fresh replay set. The first
accepted proposal is activated. Acceptance within the first n proposals, for every n ≤ 8, comes from the same
records, which gives the matched random-search curve.

**Arms** (shared network):

| Factor | Levels |
|---|---|
| **Replay policy** | `random`: uniform over the unchanged facts. `signature`: unchanged facts of relations with the updated relation's declared (source, target) signature first, the relation itself included, then uniform. `check_guided`: the first proposal is `random`; proposal n+1 replays the unchanged facts proposal n got wrong, then fills uniformly |
| **Allowance β** | 1/16, 1/8, 1/4, 1/2 |
| **Stopping** | `fixed`: 300 steps (PR 8). `early`: stop at the first multiple of 5 steps at which the edit is answered correctly, at most 300 |

That gives 3 × 4 × 2 = 24 arms. Two references are added, each with fixed stopping and N = 8:
- **no replay** (β = 0);
- **full rehearsal** (every unchanged fact).

**Specialists (contrast).** Only the updated relation's network is fine-tuned. Its replay set is drawn
uniformly from that relation's own unchanged facts, with β ∈ {0, 1/8, 1/4, 1/2} of the relation's size.
Cross-relation interference is impossible by construction; within-relation drift is not.

**Costs, per update:** proposals; screens and Checker runs; optimizer steps; example-steps (training examples
× steps); wall time.

## Units and procedure

- **Worlds:** confirmation worlds **900, 901, 902**, used by no earlier run. Development uses worlds 100–101.
- **Inits:** confirmation 0–4; development 100–101.
- **Bases:** each unit trains a shared dense network and a specialist family as in PR 4 (1,500 steps,
  lr 1e-2). A base must be exact on its world; an inexact base is reported and excluded, with its count.
- **Sequences:** one per world, drawn from the world seed, so every init and arm sees the same intended
  updates. The deployed sequences diverge after a refusal.
- **Seeds:** every proposal's seed and replay draw derive from (world, init, arm, update, proposal), so the
  verifier regenerates them.
- **The arm-level unit:** per world, 5 inits × 12 updates = 60 updates per arm.

## Predictions and the PR 9 decision (one-sided; bars to be confirmed by the development pilot)

A prediction **holds** when it holds in every world, on the per-world aggregate of 60 updates.

| # | Prediction | Kind |
|---|---|---|
| R1 | Full rehearsal accepts ≥ 95% of updates | Empirical; sanity, replicating PR 8 |
| R2 | No replay accepts ≤ 50% of updates within 8 proposals | Empirical: the problem exists |
| H1 | At β = 1/8 with fixed stopping, `signature` accepts at least 20 points more updates than `random` | Empirical: declared signatures target replay |
| H2 | At β = 1/16, early stopping raises `random`'s acceptance by at least 20 points, using at most 10% of fixed stopping's optimizer steps | Empirical: ordinary early stopping limits drift |

**Reported without a prediction:**
- the acceptance-by-proposal curves;
- `check_guided` against the other policies;
- costs;
- the specialist contrast.

**The PR 9 decision, recorded before any confirmation run.** A **gap exists at allowance β** when both hold
in at least 2 of 3 worlds:
- full rehearsal accepts ≥ 95% of updates;
- the best ordinary arm at β accepts < 90% of updates within N = 8 proposals. "Best" means the maximum over
  policy (`random`, `signature`, `check_guided`) and stopping.

| Outcome | What follows |
|---|---|
| A gap exists at some β ≤ 1/4 | PR 9's objective is fixed: at the smallest such β and N = 8, raise acceptance above the best ordinary arm, or match it with fewer proposals or example-steps. The baselines are this study's arms, run on fresh worlds |
| No gap exists | PR 9 stays deferred. The reason is recorded: ordinary methods already reach full rehearsal's acceptance at that allowance |

## Export and independent verification

**Export** (weight-free JSON, as in #58):
- **per unit:** the base's exactness; the drawn sequence; for each arm, the deployment chain;
- **per proposal:**
  - unit, arm, update and proposal index;
  - seed;
  - replay set, as indices into the target world's sorted fact list;
  - steps used;
  - every wrong fact with its predicted target;
  - a digest of the full atomic table;
  - the screen result;
  - Checker outcomes where run;
  - wall time.

**The verifier** (`bench/repeated_updates_verify.py`) does not call the producer's training or gate code. It:
- regenerates worlds, sequences and every replay draw from the declared samplers;
- re-derives each `check_guided` set from the previous proposal's exported errors;
- re-derives each proposal's acceptance from its exported errors against its target world;
- re-derives the deployment chain: each target is the deployed world plus the edit, each base is the last
  accepted candidate, and a refused update does not land;
- re-derives every rate, cost total, prediction and the PR 9 decision;
- requires the Checker to agree with the screen wherever it ran.

**Re-execution sample.** An exported error list would otherwise be the producer's record of a network's
behaviour. The verifier retrains a declared random sample of complete (unit, arm) chains from the base
recipe, on CPU, and requires identical error lists, table digests and step counts. Determinism is checked in
the pilot.

**What it carries over from #58–#60:**
- frozen source binding;
- a frozen inventory (every world, init and arm, none missing or repeated);
- every record compared field by field;
- deliberate-corruption tests, each rejected. They include:
  - an altered replay set;
  - a hidden error;
  - a flipped acceptance;
  - a candidate built on the wrong base;
  - a refused update deployed;
  - a missing or extra proposal;
  - a `check_guided` set not drawn from the previous errors;
  - an empty or incomplete frozen run.

## Interpretation limits

- **A finite synthetic world:** about 280 facts in five relations, single-fact updates, one small shared
  network and one optimizer recipe.
- **The allowance is a stand-in.** Full rehearsal is cheap here. β models the cost of rehearsal when facts
  are many; the study shows what each method needs at each fraction, not that the fractions carry over.
- **`check_guided` relies on an exhaustive check.** A larger domain would have only sampled checks, so this
  arm is an optimistic ordinary method.
- **`signature` relies on declared signatures.** That is the typed interface this project supplies; a system
  without it could not use the policy.
- **Acceptance is strict.** One wrong fact anywhere rejects a candidate. That is the gates' definition, not a
  tolerance chosen here.
- **The PR 9 decision is a gate, not a result about proposals.** A gap shows that ordinary methods fall short;
  it does not show that anything closes it.

## Decisions for the reviewer (revision 1)

| # | Decision | Recommendation |
|---|---|---|
| 1 | Include the specialist contrast | Yes: cheap, and it separates within-relation drift from cross-relation interference |
| 2 | β grid {1/16, 1/8, 1/4, 1/2} and N = 8 | Yes, unless the pilot puts every arm at a ceiling or floor |
| 3 | Gap thresholds: full ≥ 95%, best ordinary < 90%, in 2 of 3 worlds, at β ≤ 1/4 | Yes |
| 4 | Keep `check_guided` as an ordinary method | Yes: it is the strongest simple baseline, and leaving it out could manufacture a gap |
| 5 | Re-execution sample: one complete chain per arm per world (about 20% of compute) | Yes |
| 6 | Update kinds: PR 8's capital moves and operator changes | Yes, for comparability |
| 7 | Full Checker on a sample of screened-out proposals: 5% | Yes |

**Before freezing:**
1. Implement the harness, export, verifier and corruption tests.
2. Run the development pilot on worlds 100–101 × inits 100–101. Report runtimes, re-execution determinism and
   screen and Checker agreement.
3. Confirm the bars and the β grid, or revise them, with the pilot's numbers.

**Estimated cost.** The probes ran 30 arms × 12 updates × up to 8 proposals in about 2 minutes per unit with
the screen alone. With Checker runs and the re-execution sample: about 5–8 minutes per unit, about 2 hours for
15 units, CPU only.

## History

### Exploratory development probes (2026-10-23, world 100, init 100; disclosed)

Both probes were run before this draft, to choose a regime that is neither at ceiling nor at floor. They used
scratch code, not this protocol's harness, one sequence each, and absolute replay sizes (19, 38, 75 and 150
facts: 7%, 14%, 27% and 54% of world 100's 276). The pilot replaces them. Development worlds only.

**Probe 1: ungated, 12 updates, every candidate applied, two replay draws per arm.** Exact candidates, of
12:

| Arm | Exact |
|---|---:|
| no replay | 1 |
| no replay, early stopping | 1, with unaffected errors 2–27 instead of 29–71, at 3% of the optimizer steps |
| `random`, 128 facts (46%) | 0–1 |
| signature-first, 128 facts (46%) | 10–12 |
| full rehearsal | 12 |

**Probe 2: gated, with up to 8 proposals.** Accepted, of 12 (fixed / early stopping):

| Replay facts (of 276) | `random` | `signature` | `check_guided` |
|---|---|---|---|
| 19 (7%) | 1 / 6 | 5 / 9 | 3 / 8 |
| 38 (14%) | 2 / 7 | 11 / 9 | 4 / 9 |
| 75 (27%) | 4 / 6 | 10 / 9 | 7 / 9 |
| 150 (54%) | 8 / 9 | 12 / 10 | 12 / 10 |

- **References:** no replay 1; full rehearsal 12, each on the first proposal.
- **Specialists,** with own-relation replay of 0, 2, 4, 8 facts and all: 9, 9, 11, 11 and 12.
