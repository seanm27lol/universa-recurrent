# Interpreter robustness: new structure, taught wording, unfamiliar wording (frozen brief, revision 4)

**Status: FROZEN 2026-10-18 (UTC), revision 4, before any run on confirmation worlds 700–702.**
- **What is authoritative.** The specification immediately below. Everything after "History" is
  superseded and kept as it was written: the drafts, both development pilots and their reviews.
- **The frozen code** is the commit that adds this revision:
  - `bench/interpreter_robustness.py` (the generator, audit, arms, policies and `evaluate()`);
  - `bench/interpreter_robustness_verify.py`;
  - `integrated/interpreters.py`, `integrated/grammar.py`, `integrated/calibration.py`.
- **No training was done for revision 4.** Its changes were verified with development fixtures and
  synthetic records only (`tests/test_ti_tnn_interpreter_robustness.py`).

## The question

In [#48](../reports/typed_routing_v1.md), a learned interpreter failed under unseen phrasings. This study
separates what that failure mixed:
- **structure:** can familiar words compose into a three-step question when training had at most two?
- **taught wording:** does a synonym taught only in one-step questions transfer into two-step
  composition?
- **unfamiliar wording:** does the interpreter answer when the unfamiliar words are inessential (a novel
  frame or filler), and refuse when they carry the relation?

Simple controls run alongside, and any competence claim requires coverage as well as bounded error.

## Frozen specification

### Worlds, units, arms

- **Worlds and units.** `generate_world` seeds **700, 701, 702**, with 96 countries; inits **0–4**. A unit
  is one (world, init).
- **Modules.** The relation modules are the table modules, so the interpreter is the only learned part.
- **Arms.** A 2 × 2 factorial (word or character input × with or without novelty augmentation): `word`,
  `word_aug`, `char`, `char_aug`. Every arm trains for exactly 1,500 Adam updates (learning rate 3e-3,
  batches of 64) with #48's encoder and decoder. The arms are compared as complete configurations:
  - **word arms:** a vocabulary from the ordinary training questions. Augmentation pseudo-words and
    unfamiliar test words are both `<unk>`. Only the entity token is dropped (probability 0.1).
  - **character arms:** a character-level bidirectional GRU per word, with no dropout. The input
    contract is lowercase, the alphabet `a–z 0–9 _ ' ? . ,`, words of at most 48 characters, questions of
    at most 40 words, no truncation and no shared unknown symbol.
  - **augmented arms:** novelty augmentation (declared extra supervision), adding 25% of the Base
    training count in each form:
    - essential: one relation phrase replaced by pseudo-words, with target `<unsupported>`;
    - inessential: a pseudo-word frame or fillers, with the target unchanged.

    One pseudo-word pool and one count distribution (1–3 words) serve both forms.
- **The grammar control** is not learned. It knows the Base and taught lexicon and one declared rule:
  a prefix noun phrase starts at the first "the", and a possessive one at the entity. Unknown material
  outside the noun phrase is allowed, and an unparsable noun phrase is refused. It never sees held-out or
  novel words. It handles every category by construction and serves as the standard, not as a finding.

### Language

| Set | Contents | Where |
|---|---|---|
| **Base** | #48's prefix and possessive phrasings; frames "what is … ?", "name … ." | Everywhere |
| **Taught** | "the administrative centre of", "the most populous city of", "the nation of", "the airfield of", "the firm running" | Training only in one-step questions; T1, S2, S2_L3 |
| **Held-out (H)** | "the seat of government of", "the metropolis of", "the state that holds", "the aerodrome near", "the outfit managing" | S4 only |
| **Novel (N)** | Frames "could you tell me … ?", "i would like to know … ."; fillers "exactly", "precisely" | S3, S4b, S3_L3 only |

- **What the protocol requires.** H wording was never taught, and the protocol requires abstention on
  it. No word overlap rules out one shortcut. It does not rule out inference from morphology, position,
  entity types or the small set of relations; the S4 strata below control for type inference.
- **One style per question.** Each question is all prefix or all possessive.

### Training (fixed first; not sampled for any claim)

- **Instances.** 30% of the instances of at most two relations (a program and an entity) are drawn at
  random for training.
- **What training contains:**
  - their Base questions;
  - the taught phrasing of every one-step training instance;
  - refusals, at 6% of training instances per kind (ill-typed, unsupported, unknown entity,
    ambiguous), deduplicated within training;
  - the augmentation, in the augmented arms.
- **The structural limit.** Every training target has at most two relations.

### Evaluation: independent draws from declared populations

**The held-out population** has two parts:
- every instance of at most two relations not drawn for training;
- every three-relation instance (none is trained).

**Sampling rules:**
- every evaluated question is an independent draw with replacement, under fixed probabilities;
- repeated strings within or between samples are allowed, as independent draws imply;
- every item has a unique draw id;
- nothing evaluated is a training instance, and no evaluated refusal is a training question.

**The ID distribution:**
- **a refusal**, with probability 0.24. Its kind is uniform, and it is drawn from the refusal generator,
  conditioned on not being a training question.
- **otherwise a supported question:** a held-out instance of at most two relations, uniform, with a
  uniform style, frame and Base phrasing.

**Calibration** is 1,000 independent draws from the ID distribution. **The ID test** is a further 1,000
independent draws: supported draws are scored as **ID**, refusals as **R**.

**Matched designs.** Each draw of a design is a base case with matched variants. Each variant is its own
category and is never pooled with the others as independent evidence.

| Design | Draws | Base case | Variants (correct behavior) |
|---|---:|---|---|
| T1 | 500 | one-step held-out instance, a frame | `T1`: the taught phrase (answer); `T1_base` (answer) |
| S2 | 800 | two-step held-out instance, a frame, position j | `S2`: the taught phrase at j (answer); `S2_base` (answer) |
| S3 | 1,000 | short held-out instance, a frame | `S3`: a novel wrapper, uniform over two novel frames or the frame plus one of two fillers (answer); `S3_base` (answer) |
| S4 population | 1,000 | short held-out instance, position j | `S4a`: the H phrase at j (abstain); `S4b`: the same with a novel wrapper (abstain); `S4_base` (answer) |
| **S4 primary** | **2,000** | as above, **conditioned on the answer-ambiguous stratum** | `S4a_primary`, `S4b_primary` (abstain); `S4_primary_base` (answer) |
| S1 | 800 + 300 | three-relation instances: non-cancelling chains (800) and round trips (300) | `S1` (answer) |
| Length-3 diagnostics | 300 | non-cancelling three-relation instance, position j | `L3_base`, `S2_L3`, `S3_L3` (answer); `S4_L3` (abstain) |

S3 keeps its broad declared distribution; it is not restricted to the cases selected for S4.

**The S4 strata.**
- **How a case is classified.** At the held-out position, every relation is tried while the rest of the
  program stays fixed. An alternative is compatible if the whole program is well typed from the entity.
- **The three strata:**
  - **answer-ambiguous:** two or more compatible programs give two or more answers. This is the primary
    test.
  - **program-ambiguous with one answer:** several compatible programs, one answer.
  - **type-determined:** only one compatible program.
- **Where they come from.** The population design reports all three. The primary design samples only
  answer-ambiguous cases.

### Audit (the run refuses to start unless every check holds)

It checks the declared inventory: every category with its declared count, and S1's split by family. It
also checks:
- **variant groups:** complete and matched, with the same program and entity, and frames as declared;
- **S4 strata:** each recomputed and equal to the stored one; the primary designs are all
  answer-ambiguous;
- **ids and separation:** unique draw ids, and separation of training from evaluation;
- **training limits:** every training target has at most two relations, and taught words appear in
  training only in one-step questions;
- **vocabulary:** no H or N word in training; pseudo-words disjoint from the lexicon and entities; H
  sharing no word with Base or taught wording, apart from "the" and "of";
- **questions:** one style each, and every one within the character contract.

### Policies and certification

**Policies.**
- `none`: answer whenever the interpreter proposes a program the runtime accepts and runs.
- `certified`: answer when the raw score is at least the certified threshold, and never without a
  certificate.
- `novelty`: like `none`, but abstain if any token other than the copied entity is outside the training
  vocabulary.
- `novelty_certified`: both filters.

Platt probabilities, fitted on calibration with a positive slope, are for reporting only. A monotone
transform accepts the same questions at mapped thresholds, so there is no separate calibrated policy.

**Certification.** The rule family is "answer when the raw score is at least t". The grid, fixed in
advance and run strictest first, is 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.
- **The test.** For each t in order, the null hypothesis is "accepted error ≥ 1%", with p-value
  P(Binomial(n_t, 0.01) ≤ e_t). Testing stops at the first non-rejection at δ = 0.05, and the loosest
  rejected t is chosen.
- **The guarantee.** Calibration questions are independent draws from the ID distribution. So, given the
  world and the trained model, there is at most a 5% chance that the selected rule's true error on
  future questions from the ID distribution exceeds 1%.
- **What it does not cover:** other phrasings, longer programs, or other worlds.

**Power.** It is stated for a fixed rule, as a function of the number of accepted questions n and an
assumed true error rate. At n = 760 it is about 0.96 for a true error of 0.1%, 0.70 for 0.25%, and 0.27
for 0.5%. The allowed error count is 0 below 473, 1 below 628, and 2 below 773.

### Measurements, records and verification

**Measurements.** For each unit, arm, policy and category:
- coverage, accepted errors and the 95% upper error bound;
- abstention, with an exact two-sided 95% interval;
- exact program-and-entity accuracy, and right answers from wrong parses;
- reasons for not answering, and wrong-entity versus wrong-program errors.

**Records.** For calibration and test items, every unit keeps compact full-precision records: decision,
reason, score, proposed program, copied entity and produced answer.

**Verification.** `bench/interpreter_robustness_verify.py` regenerates each study from its seed and
re-audits it. From the records alone it re-derives correctness, novelty, every policy's acceptance, the
certificate, every metric and the verdict, compares each with the stored values, and rejects any
discrepancy or incomplete inventory.

### Criteria, joint rule and predictions

**Per-unit criteria:**
- **answers where it should:** a certificate exists, coverage is at least **0.80**, and the 95% upper
  error bound is at most **1%**;
- **abstains where it should:** observed abstention is at least **0.95**, reported with its exact
  interval. This is a benchmark criterion. A statistical claim that the population refusal rate is at
  least 95% would need the lower bound itself to reach 0.95.

**The joint rule:**
- **What a unit must meet.** A unit succeeds on a criterion set only when that **same** unit meets every
  part of it.
- **What a world must meet.** A world holds when at least **4 of its 5 fixed inits** succeed jointly.
- **What a prediction needs.** A prediction holds only when every confirmation world holds.
- **No certificate.** An uncertified unit is recorded as such, counts as not successful, and stays in the
  denominator.

**The predictions:**
1. **Sanity, every arm.** `certified` gives a certificate, ID coverage of at least 0.90 with a 1% bound,
   and R abstention of at least 0.95.
2. **Taught wording learned, every arm.** T1 accuracy under `none` is at least 0.95.
3. **The lead hypothesis, `word_aug`** (chosen after the development pilots, disclosed). With `certified`,
   in the same unit:
   - a certificate;
   - S3 coverage of at least 0.80;
   - an S3 error bound of at most 1%;
   - `S4a_primary` abstention of at least 0.95;
   - `S4b_primary` abstention of at least 0.95.

**Separate verdicts,** with no directional prediction, under the same joint rule and recorded for each
arm:
- **joint novelty,** for the other three arms;
- **S2 transfer:** answers where it should on S2;
- **S1 structure:** answers where it should on S1's non-cancelling chains. I expect every arm to fail.

**Reported, not predicted:**
- the other S4 strata;
- the S4 population;
- the length-3 diagnostics;
- the grammar control;
- collisions and probability calibration.

### Interpretation limits (frozen with the specification)

- **Thresholding is not refusal.** A certified threshold of 0 filters nothing, but the interpreter and
  runtime still refuse. S3 and S4 behavior is evidence about the learned interpreter's decisions. It is
  not evidence that certification produced robustness.
- **Augmentation.** A success for an augmented arm shows the property can be taught by declared extra
  supervision, not that it emerges unaided. The development result supports a narrower statement: among
  the learned configurations tested, joint novelty successes occurred only in `word_aug`. The grammar
  control succeeds without any augmentation, and four development units cannot establish necessity.
- **Scope.** The guarantee and every result are conditional on these worlds, models and declared
  distributions. More evaluation questions reduce question-sampling noise, not variation across
  trained models.

### Compute

CPU only. 60 fits at 1,500 updates (word arms about 17 s, character arms about 40 s), plus evaluation of
about 18,000 items per arm and unit. The estimate is about 1 hour.

## History (superseded; kept as written)

### Corrections to the history (revision 4)

- **Round 2's certificates are diagnostic only.** "The certificate bounds in-distribution risk" contradicts
  the partition sampler's unresolved justification. The defensible statement is that the certification
  algorithm selected threshold 0 under the working assumptions. Revision 4's independent sampler supports
  the confirmation guarantee; it does not establish round 2's retrospectively.
- **The S2 arithmetic was wrong.** At about 550 questions and 80% coverage (440 accepted), the 1% bound
  allows **zero** errors: one error among 440 gives an upper bound of 1.07%. 800 questions at 80%
  coverage (640 accepted) allow two errors (0.98%).
- **"Augmentation is necessary but not sufficient" is replaced.** Among the tested learned configurations,
  joint novelty successes occurred only in `word_aug`. Augmentation improved the word arm, but did not
  ensure success across the development units.
- **Power is stated at a given accepted count and assumed true error,** not "with zero errors". Power
  averages over the possible observed error counts.
- **Round 2's 0.94 unit.** The pilot does not establish that this unit's true refusal rate is 96%; the
  0.94 could reflect an init-dependent weakness. The development record stands at 3 joint successes out
  of 4 units under the original bars.

**Status: DRAFT, not frozen.** Revision 2 followed the review of the first draft (dfe142e). Revision 3
applies the changes approved after development pilot round 1 (see "Round 2 design"); round 2 runs on
development worlds and inits only. The grid start, world size, bars, sample sizes, aggregation rule and
the sampling justification stay provisional until round 2 shows the evaluation has enough evidence.
Confirmation worlds 700–702 stay untouched.

### Why

In [#48](../reports/typed_routing_v1.md) the interpreter failed under unseen phrasings, but that result
mixed three things:
- **information loss:** every unfamiliar word became `<unk>`;
- **a trivial control:** rejecting unfamiliar words would have refused the whole shift;
- **calibration:** it was barely needed, and its threshold had no selection-corrected guarantee.

This study separates the questions, runs simple controls first, and requires coverage for any
competence claim.

### Design

Code: `bench/interpreter_robustness.py`, `integrated/interpreters.py`, `integrated/grammar.py`, and the
certification functions in `integrated/calibration.py`.

#### Worlds, units, modules

- **Worlds.** `generate_world`, 48 countries. Development: worlds 100–101, inits 100–101. Confirmation:
  worlds **700–702**, inits 0–4. Each (world, init) is a unit.
- **Modules.** Table modules, to isolate the interpreter. In #48 the module registries gave identical
  results.

#### Language

The four phrase sets are disjoint except for the function words "the" and "of":

| Set | Contents | Where it appears |
|---|---|---|
| **Base** | #48's phrasings, in prefix ("the capital of X") and possessive ("X 's capital") style; frames "what is … ?", "name … ." | Everywhere |
| **Taught (T)** | Prefix only: "the administrative centre of X" (capital), "the most populous city of X" (largest city), "the nation of X" (country), "the airfield of X" (airport), "the firm running X" (operator) | Training **only in one-step questions**; test T1, S2, S2-L3 |
| **Held-out essential (H)** | Prefix only: "the seat of government of X", "the metropolis of X", "the state that holds X", "the aerodrome near X", "the outfit managing X" | Test only (S4) |
| **Novel inessential (N)** | Frames "could you tell me … ?", "i would like to know … ."; fillers "exactly", "precisely", placed after a Base frame's opening words | Test only (S3, S4b) |

- **What H means.** H wording was never taught, and the protocol requires abstention on it. Having no
  word in common with the taught sets rules out one shortcut (shared words). It does not rule out
  inference from morphology, position, entity types or the small set of relations.
- **One style per question.** Each question is all prefix or all possessive, so it has exactly one
  reading.

#### Items

Programs of length 1–2 are split by (program, entity) instance: training 30%, calibration 30%, test 40%,
one rendering per instance.

**Training:**
- Base questions;
- the taught wording, for every one-step training instance;
- refusals (6% of instances each): ill-typed, unknown entity, unsupported (Base wording), ambiguous;
- novelty augmentation, in the augmented arms only.

**The structural limit applies to every training target, refusals included.** In #48's generator,
ill-typed and unknown-entity refusals were drawn from paths of up to three relations, and the
interpreter's target is the full relation sequence. Here every training target has at most two
relations, and an audit checks the actual targets before any training (see Audit).

**Test categories.** Variants of one instance share its program, entity, style and outer frame, and
differ only where stated.

| Category | Construction | Correct | Question it answers |
|---|---|---|---|
| **ID** | Base, length 1–2, either style | Answer | Sanity |
| **R** | Refusals of the four kinds, Base wording, length ≤ 2 | Abstain | Sanity |
| **T1** (+ `T1_base`) | One-step, taught phrase (matched Base version) | Answer | Was the taught wording learned at all? Reported per phrase |
| **S2** (+ `S2_base`) | Two-step, prefix, one Base phrase replaced by its T synonym | Answer | Does taught wording transfer into composition at a trained length? |
| **Q_base** | Two-step or one-step prefix, Base wording, Base frame | Answer | Matched reference for S3/S4 |
| **S3** | Q_base with a novel frame or filler (N) | Answer | Inessential novelty |
| **S4a** | Q_base with one relation phrase replaced by its H synonym | Abstain | Essential novelty |
| **S4b** | S4a with the S3 wrapper as well | Abstain | Essential novelty despite a novel wrapper |
| **S1** | Length 3, Base, either style; non-cancelling chains and round trips reported apart | Answer | Can familiar words compose in a longer structure? |
| **L3_base**, **S2_L3**, **S3_L3**, **S4_L3** | Length-3 diagnostics: Base; with one T phrase; with a novel frame; with one H phrase | Answer, answer, answer, abstain | Interactions with the length shift; secondary only |

S3 and S4 are matched one- and two-step questions, so a model that fails at length three earns no credit
for refusing S4. Refusal strings are never repeated within or across splits.

#### Audit (before any training; the run refuses to start if any check fails)

The audit checks that:
- every training target, in every source, has at most two relations;
- taught words appear in training only in one-step questions;
- no H or N word appears in training;
- pseudo-words are disjoint from every lexicon word and entity;
- H shares no word with Base or T, apart from "the" and "of";
- each question has one style;
- every question meets the character contract;
- splits are disjoint by instance;
- refusal strings are unique.

On the development worlds, the inventory is 374 and 378 calibration questions, and 500 and 504 ID test
questions (plus refusals).

#### Interpreters: a 2 × 2 factorial, and a grammar control

| | No novelty augmentation | Novelty augmentation (declared extra supervision) |
|---|---|---|
| **Word embeddings** | `word` | `word_aug` |
| **Character word encoder** | `char` | `char_aug` |

**Encoder and decoder.** All four share #48's encoder and decoder, and differ only in how a word becomes
a vector.

**Word arms.**
- **Vocabulary:** the ordinary training questions. Augmentation pseudo-words are **not** added, so a word
  model sees them as `<unk>`, exactly as it sees unfamiliar test words.
- **Dropout:** only the entity token is replaced by `<unk>` (probability 0.1), so the entity head learns
  to copy unseen entities. Relation words are never dropped. #48 dropped any token, which trains the
  decoder to output the original program when a relation word is `<unk>`: in effect, to guess.

**Character arms.** A bidirectional GRU over the word's characters, under a declared input contract:
- questions are lowercased and split on whitespace;
- the alphabet is `a–z 0–9 _ ' ? . ,`;
- a word is at most 48 characters and a question at most 40 words;
- no truncation, and no shared unknown-character symbol. A question outside the contract is refused as
  "unsupported characters".

Distinct strings are therefore distinct input sequences. A finite-dimensional encoder is not guaranteed
to keep every distinction in its vectors.

**Novelty augmentation** (the `_aug` arms). It adds 25% of the Base training count in each of two forms,
built from Base training questions only:
- **essential:** one relation phrase is replaced by pseudo-words ("the *wosa cilu* of X", "X 's *mucit*"),
  with target `<unsupported>`;
- **inessential:** a pseudo-word frame, or pseudo-word fillers after a Base frame, with the target
  unchanged.

Both forms draw on **one pseudo-word pool, with the same count distribution (1–3 words)**. So only where
the pseudo-words sit tells an essential use from an inessential one.

**Matched budgets.** Every arm trains for the same number of optimizer updates on batches of 64, so
smaller sets repeat more. The pilot uses 1,500 updates.

**The grammar control** is not learned.
- **What it knows:** the Base and T lexicon, plus one declared structural rule. A prefix noun phrase
  starts at the first "the" and is a sequence of lexicon phrases, then the entity; a possessive one is
  the entity, then "'s" phrases.
- **What it does:** unknown material outside the noun phrase (frames, fillers) is allowed. A noun phrase
  that cannot be parsed, or parses more than one way, is refused.
- **What it is not given:** the H or N word lists, or any test label. It handles every category by
  construction, and serves as the standard to compare against, not as a finding.

#### Answering policies

These apply to each learned arm:

| Policy | Rule |
|---|---|
| `none` | Answer whenever the interpreter proposes a program that the runtime accepts and runs |
| `certified` | Answer when the raw score is at least the certified threshold (below); never, if no certificate |
| `novelty` | `none`, but abstain when any token other than the copied entity is outside the training vocabulary |
| `novelty_certified` | Both filters; certified with the novelty filter applied |

**Calibrated probabilities are for reporting only.** Platt scaling is fitted on the calibration split,
with its slope forced positive, so it never reorders examples. Its quality is reported as Brier score,
log loss and calibration error on the ID test split. Under a positive-slope transform `g`,
`score ≥ t` exactly when `g(score) ≥ g(t)`. A calibrated threshold would therefore accept the same
questions and carry the same certificate, so no separate calibrated-threshold policy is run.

#### Certification

The certified population is the calibration split: supported ID questions plus refusals, one rendering
per instance. Answering a refusal counts as an error.

**The rule family** is "answer when the raw score is at least t", for t in a grid fixed in advance and
ordered strictest first. The provisional grid is 0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8,
0.6, 0.4, 0.2, 0.

**The test.** For each t in order:
- the null hypothesis is "accepted-error rate ≥ 1%";
- its p-value is P(Binomial(n_t, 0.01) ≤ e_t), with n_t accepted and e_t errors;
- testing stops at the first t whose p-value exceeds δ = 0.05, and the loosest rejected t is selected.

With a fixed order, stopping at the first non-rejection bounds by 0.05 the chance of selecting a rule
whose true accepted-error rate exceeds 1%. This holds even though the tests are dependent and risk need
not fall monotonically. This follows the multiple-testing framing of Learn then Test (Angelopoulos,
Bates, Candès, Jordan and Lei, [arXiv 2110.01052](https://arxiv.org/abs/2110.01052)).

**What the guarantee assumes.** It treats calibration questions as independent draws given the unit. One
rendering per instance supports that, and extra paraphrases are never added to raise n.

**Power.** With zero errors, the 95% bound reaches 1% only with at least 299 accepted examples (473,
628 and 773 for 1, 2 and 3 errors).
- **The first rule matters.** A first rule accepting fewer than 299 cannot pass, and the stopping rule then
  tests nothing looser.
- **Counts here.** The calibration split has about 374 supported questions per development world. A fixed
  policy with a true error rate of 0.5% is certified only about 15% of the time at that size.
- **Recording.** "No certificate" is therefore recorded separately from an observed failure.
- **Design consequence.** The pilot checks how full the strictest rule is, and the grid's start is fixed
  from that. The calibration split is not halved.

Every step is recorded (threshold, accepted, errors, p-value, outcome), together with the stopping
reason, the selected threshold, and every calibration question's ID, score and correctness.

#### Measurements

For each unit, arm, policy and category:
- coverage (answered over all questions);
- accepted errors over answers, with the 95% Clopper–Pearson bound;
- abstention;
- exact program-and-entity accuracy;
- answers that are right despite a wrong parse (coincident facts or cancelling paths);
- why each question was not answered: a learned unsupported or ambiguous output, a malformed output, a
  runtime rejection (ill-typed), an unknown entity, the threshold, the novelty filter, or unsupported
  characters;
- for wrong answers, whether the entity or the program was wrong.

**Input collisions.** These are reported per world for the word and character encodings:
- **program collisions:** the same encoded input and copied entity, with different reference programs
  or answers;
- **decision conflicts:** the same encoded input and copied entity, requiring an answer in one case and a
  refusal in another;
- **encoding-only conflicts:** the stricter version without the entity. This is the relevant one for S3
  against S4.

**Retained records.** Every test question's compact decision for every arm, and the full certification
records.

### What runtime refusals do and do not show

- **Ill-typed refusals hold only given a faithful parse.** If the interpreter reads "what is the capital of
  country_0_city_0 ?" as `country_of(country_0_city_0)`, the program is well-typed and the runtime
  answers "country_0". So R abstention measures the interpreter as well as the runtime.
- **Unknown-entity refusal has a narrower guarantee.** Such a question contains no valid world entity, and
  the entity head can only copy a question token, so the policy cannot bind a valid entity.

### Proposed predictions (one-sided; bars provisional until the pilot)

**Answers where it should:** coverage at least **0.80** and a 95% upper bound on accepted error of at most
**1%**. **Abstains where it should:** abstention at least **0.95** (an observed proportion).

**How the criteria aggregate:**
- each criterion is judged per unit, and holds for a world on at least 4 of its 5 inits;
- this is a benchmark aggregation rule, and the per-unit 95% statements do not form one simultaneous
  guarantee;
- "no certificate" is reported apart from failure.

1. **Sanity.** Each arm with `certified` answers ID where it should, with coverage at least 0.90, and
   abstains on R where it should.
2. **Taught wording was learned.** Each arm's T1 accuracy under `none` is at least 0.95. This is a
   prerequisite for reading S2.
3. **Taught wording transfers (two steps).** `char_aug` with `certified` answers S2 where it should.
4. **Inessential novelty.** `char_aug` with `certified` answers S3 where it should. `novelty` does not:
   its S3 coverage is about 0 by construction.
5. **Essential novelty.** `char_aug` with `certified` abstains on S4a and on S4b where it should.
6. **Structure.** `char_aug` with `certified` answers S1's non-cancelling chains where it should.

**Reported, not predicted:**
- the full 2 × 2 table for S3 and S4 against the grammar control. It answers three questions: did the
  augmentation help, did character input add anything beyond it, and did any learned arm reach the
  grammar's standard?
- the length-3 diagnostics;
- the collision counts;
- probability calibration.

**What I expect** (stated, not predicted):
- 1 and 2 hold.
- 3 probably holds.
- 4 and 5 together are the open question.
- 6 probably fails.

### Limits

- **Controlled language with defined synonyms.** Nothing here concerns natural language.
- **The augmentation teaches the property under test,** with other words. A success shows it can be taught,
  not that it emerges unaided.
- **The grammar control knows the grammar.** Matching it is useful; it does not establish that a learned
  model is needed.
- **Table modules.** End-to-end learned-module results are #48's.

### Compute

CPU only. Measured in a smoke run, 100 updates take about 1 s for a word arm and 2.5 s for a char arm. At
1,500 updates:
- **pilot:** 16 fits, about 10 minutes;
- **confirmation:** 60 fits, under 1 hour.

### Changes from the first draft (after review)

- **Training targets.** All training targets, refusals included, are limited to two relations, and audited.
  Taught wording stays in one-step training contexts.
- **Category structure:**
  - T1 added as a prerequisite;
  - S2 restricted to two steps, with three-step versions as diagnostics;
  - S3, S4a and S4b matched one- and two-step questions, with S4b added;
  - S1 split by family;
  - the length-3 contradiction removed.
- **Wording.** "currently" was replaced by "precisely". The "cannot be guessed" claim was narrowed.
- **Arms and controls.** `word_aug` was added (a 2 × 2 factorial) with matched update budgets, and the
  pseudo-word pool and count distribution were matched across uses. The grammar control was added.
- **Collisions.** Program collisions are separated from decision conflicts. The character claim now
  concerns input sequences under a declared contract.
- **Certification.**
  - It certifies raw-score rules, with Platt kept for reporting only (positive slope).
  - The population and sampling assumption are stated.
  - Power is computed, "no certificate" is kept apart from failure, and the calibration split is not
    halved.
- **Runtime refusal.** The claim was narrowed to "given a faithful parse"; the unknown-entity invariant is
  stated.

### Development pilot, round 1 (2026-10-18; disclosed; not a result)

**Setup.** Worlds 100–101 × inits 100–101: 4 units, 4 arms and 1,500 updates, run at `613e736`. Every audit
passed. Retraining two arms of one unit reproduced their recorded decisions and scores exactly.

**1. The certification design cannot work at this size. No arm earned a certificate in any unit.**
- **The first rule was too thin.** The strictest rule (0.9999) accepted only 77–289 of the roughly 375
  calibration questions, below the 299 needed even with zero errors, so the stopping rule never reached
  looser rules.
- **The next rule would have worked, barely.** At 0.999, the rules accepted 346–378 questions with 0–1
  errors. A single calibration error at that size already prevents a certificate (473 are needed for one
  error).
- **What it was not.** Under `none`, every arm answered all ID calibration questions with 0–9 errors. The
  failure is statistical power, not accuracy.

**2. What each arm did, pooled over 4 units.**

| Arm | T1 accuracy | ID at 0.999: coverage, errors | S3 at 0.999: coverage, errors | S4a abstention at 0.999 (type-ambiguous / type-determined) | S2 at 0.999: coverage, errors | S1 at 0.999: coverage, errors |
|---|---|---|---|---|---|---|
| `word` | 1.00 | 1.00, 0 | 0.31, 0 | 0.94 / 0.87 | 0.34, 0 | 0.23, 552 |
| `word_aug` | 1.00 | 1.00, 0 | **0.98, 0** | **1.00 / 1.00** | 0.60, 19 | 0.26, 617 |
| `char` | 1.00 | 0.97, 0 | 0.27, 6 | 0.77 / 0.86 | 0.38, 0 | 0.21, 510 |
| `char_aug` | 1.00 | 0.98, 0 | 0.43, 1 | 0.88 / **0.42** | 0.51, 4 | 0.16, 395 |
| grammar control | 1.00 | 1.00, 0 | 1.00, 0 | 1.00 / 1.00 | 1.00, 0 | 1.00, 0 |

**The 0.999 columns are diagnostic.** 0.999 is the first adequately populated rule, applied to see what a
certified policy would do; nothing was certified. Under `none`, every arm answered nearly all of S3 and S2,
with more errors.

**3. Many held-out questions can be answered from the entity's type.**
- **The leak.** `operator_of` is the only relation that applies to an airport, so "the outfit managing X"
  on an airport can be answered from the type alone.
- **The character model uses it.** When `char_aug` answered S4 (one unit), 231 of its 245 answers were the
  *intended* program, mostly `operator_of`.
- **The pattern.** Character input lets the model guess untaught wording from type and spelling; the word
  model sees only `<unk>` and refuses.

**4. Certifying on ID does not bound errors under the length shift.** At 0.999, every accepted
three-step answer in every arm was wrong.

**5. No information limit was found between S3 and S4.**
- No program collisions or decision conflicts with the copied entity, in either encoding.
- No S3/S4 conflicts.
- The word encoding has 16 and 26 conflict groups if the copied entity is ignored, all from refusals
  whose entity is unknown.

**Corrections to round 1** (made after review; the statements above are left as written):
- **Errors without a threshold.** That arms answered all ID calibration questions with 0–9 errors and no
  threshold does not establish that any arm is below the 1% target. Round 1 established only that the
  first rule failed because it accepted too few questions.
- **No certificate existed,** so item 4 must not be read as an ID certificate failing under shift. The
  observed statement is narrower: at the diagnostic raw cutoff 0.999, every accepted three-step answer was
  wrong. Separately, and in theory, a guarantee scoped to ID questions would not cover three-step
  questions.
- **Rounded scores.** The pooled 0.999 table used test scores rounded to six decimals. 8 of 86,944
  records sit exactly on the boundary, and only three could move a count:
  - two correct `word_aug` S2 answers and one correct `char` S2 answer (coverage of about 1,050);
  - one wrong `word` S1 answer.

  No conclusion changes. The calibration counts used full precision. Round 2 keeps full precision
  throughout.
- **The type-inference reading is not established.** 231 of 245 accepted `char_aug` S4 answers matching the
  intended program is consistent with type or context inference. It does not identify which cue
  decided.
- **The arms differ in dropout too.** Word arms get entity-token dropout and character arms do not, so the
  comparison is between complete configurations.

**Changes proposed for review before freezing** (round 1's proposals; applied in revision 3 as below):
1. **Make certification feasible:**
   - start the grid at 0.999, which accepted 92–100% of calibration questions in the pilot;
   - double the worlds to 96 countries, giving about 750 calibration questions so that up to two errors
     can still certify;
   - verify both with a short second pilot on development worlds.
2. **Make the primary S4 type-ambiguous.** Use only held-out phrases whose input type allows two
   relations (capital or largest city on a country; country or airport on a city). Report the
   type-determined case (`operator_of` on airports) separately.
3. **Restate the predictions per arm,** informed by this pilot and disclosed as such. The pilot suggests:
   - `word_aug` separates S3 from S4;
   - character input trades S4 refusals for type and spelling inference;
   - no arm's ID certificate protects S1;
   - S2 coverage at a certified threshold is low for every arm.

### Round 2 design (revision 3; changes applied after review of round 1)

**1. Certification, both provisional.**
- **Grid.** It starts at 0.999: 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.
- **World size.** 96 countries, giving 763 supported calibration questions per development world, plus
  184 refusals.
- **What a certificate needs.** Tolerating two errors requires at least 628 *accepted* questions, after
  the runtime, the threshold and any novelty filter. A total count does not help if too few survive.
- **Possible is not reliable.** For a fixed rule with a true accepted-error rate of 0.5%, the chance of
  certification is about 33% at 690 accepted and about 28% at 750. The allowed error count stays at two
  until 773. A nearly perfect policy has much better chances.
- **What round 2 decides.** It records accepted counts and errors at every tested threshold, and the
  power at the observed counts. Confirmation sample sizes are then chosen against a declared error
  margin and power.

**2. S4 strata over the complete typed program.**
- **The construction.** At the held-out phrase's position, every relation is tried while the rest of the
  question stays fixed. An alternative is compatible if the whole program is well typed from the entity.
- **Three strata, for S4a, S4b and S4_L3:**
  - **answer-ambiguous:** at least two compatible programs giving at least two answers. This is the
    primary test.
  - **program-ambiguous with one answer:** several compatible programs, all giving the same answer
    (for example, the capital and the largest city, then `country_of`). A diagnostic.
  - **type-determined:** only one compatible program. A diagnostic.
- **Why not count outgoing relations.** "The operator of the aerodrome near X" allows two relations from
  a city, but only `airport_of` fits under `operator_of`.
- **Counts (development worlds 100 and 101).**

  | World | Answer-ambiguous | One answer | Type-determined |
  |---|---:|---:|---:|
  | 100 | 541 | 98 | 379 |
  | 101 | 537 | 100 | 381 |

  These reproduce the review's independent counts. The full S4 populations and their results are kept.

**3. S1 allocation.**
- **Separate counts.** The primary non-cancelling family gets its own declared count of 800, and round
  trips get 300.
- **Why.** At 80% coverage, 800 allows the 1% bound with up to two errors, which needs 785. Round 1's 600
  mixed questions held only 260–268 non-cancelling ones, below the 299 needed even with zero errors and
  full coverage.
- **Diagnostics.** The length-3 diagnostics are matched prefix variants of the first 300 S1 non-cancelling
  instances.

**4. Inventory per development world** (96 countries):

| Category | Count |
|---|---|
| ID | 1,018 |
| R | 244 |
| T1 | 473 / 464 |
| S2 (two-step) | 545 / 554 |
| Q_base, S3, S4a, S4b | 1,018 each |
| S1 | 800 non-cancelling + 300 round trips |
| Length-3 diagnostics | 300 each |

**S2's power is limited.** With 545–554 questions, 80% coverage allows a 1% bound with at most one error.

**5. Records.** These are kept:
- full-precision scores;
- the proposed program, copied entity and produced answer;
- acceptance at the diagnostic cutoff, computed before any rounding;
- `bench/interpreter_robustness_tables.py`, which rebuilds every table from the run file, per unit, with
  S4a and S4b separate.

**6. Unresolved: the sampling justification.**
- **The gap.** The sampler partitions a finite population of instances without replacement, fixes
  refusal counts by kind, and deduplicates refusal strings. One rendering per instance avoids inflating
  n, but does not make the observations independent and identically distributed, which the exact
  binomial test assumes.
- **What must be decided before a confirmation-stage guarantee:**
  - either justify the test for this design (for example, as a statement about the finite instance
    population), or change the sampler to draw calibration questions independently from a declared
    question distribution;
  - which future question distribution the certificate is meant to cover. The calibration split is the
    evidence, not that population.

  Round 2 is diagnostic and does not depend on this.

#### Predictions, restated per arm (revision 3; informed by round 1, disclosed)

**`word_aug` now leads.** It is the leading prospective candidate for the novelty question. The choice
was made after round 1, on development worlds, so confirmation on fresh worlds is what tests it. All four
arms remain.

**The criteria are unchanged (provisional):**
- **answers where it should:** coverage ≥ 0.80 and a 95% error bound ≤ 1%;
- **abstains where it should:** abstention ≥ 0.95;
- each is judged per unit, holds for a world on 4 of 5 inits, and is reported per unit;
- "no certificate" is kept apart from failure.

1. **Sanity.** Each arm with `certified` answers ID where it should (coverage ≥ 0.90) and abstains on R
   where it should.
2. **Taught wording was learned.** Each arm's T1 accuracy under `none` is at least 0.95.
3. **The lead hypothesis.** `word_aug` with `certified` answers S3 where it should, *and* abstains where
   it should on the answer-ambiguous strata of both S4a and S4b.
4. **Taught-wording transfer (S2) is evaluated separately from novelty.** No arm is predicted to meet it:
   round 1's diagnostic coverage was 0.34–0.60.
5. **S1 is expected to fail for every arm** (non-cancelling chains). The evaluation can now recognize
   success if it occurs.

**Reported, not predicted:**
- the other strata;
- the full 2 × 2 factorial against the grammar control;
- the length-3 diagnostics;
- collisions and probability calibration.

**Kept for the record: revision 2's predictions, superseded.**
- `char_aug` was the primary arm for S2, S3, S4 and S1.
- My stated expectation was that `char_aug` would meet S2, with S3 and S4 uncertain.
- Round 1 did not support those, but they remain the original prospective statement.

### Changes from revision 2 (after review of round 1)

1. **Certification and world size:** the grid starts at 0.999 and worlds have 96 countries, both
   provisional.
2. **S4:** strata over the complete typed program, with the original S4 kept.
3. **S1:** the non-cancelling family gets its own count, and the diagnostics are matched to it.
4. **Records:** full precision, and a tables script.
5. **Predictions:** restated per arm, with `word_aug` leading.
6. **Round 1:** its wording corrected.
7. **Sampling:** the justification is recorded as unresolved.


### Development pilot, round 2 (2026-10-18; disclosed; not a result)

**Setup.**
- Worlds 100–101 at 96 countries, inits 100–101: 4 units, 4 arms, 1,500 updates.
- Run at `3986bba`, and every audit passed.
- Tables rebuilt with `bench/interpreter_robustness_tables.py` from full-precision records.

**Certification.**
- **Feasible and adequately populated.** The first rule (0.999) accepted 737–763 of the 763 calibration
  questions. No arm made a calibration error at any threshold, so every arm certified all the way to the
  loosest rule (0).
- **Here, `certified` is the same as `none`.** When calibration shows no errors, the certified policy
  filters nothing. The certificate then bounds in-distribution risk but does nothing about shifted
  questions.
- **Power.** At about 760 accepted with zero errors, the chance of certification is 0.96 if the true
  error is 0.1%, 0.70 at 0.25% and 0.27 at 0.5%.

**Per unit, certified policy (here the same as `none`).** In each cell, units are listed in the order
w100 i100, w100 i101, w101 i100, w101 i101. "Amb." is the answer-ambiguous stratum.

| Arm | S3 coverage, errors | S4a amb. abstention | S4b amb. abstention | S2 coverage, errors | S1 non-cancelling errors (of 800) |
|---|---|---|---|---|---|
| `word` | 0.95/119, 0.94/69, 0.82/4, 1.00/0 | 0.75, 0.57, 0.50, 0.78 | 0.93, 0.60, 0.64, 0.79 | 0.83/55, 0.75/0, 0.74/0, 1.00/14 | 144, 537, 628, 516 |
| `word_aug` | 0.99/0, 0.98/0, 1.00/0, 1.00/0 | **1.00, 1.00, 1.00, 0.94** | **1.00, 1.00, 0.97, 0.94** | 1.00/0, 1.00/0, 0.99/11, 1.00/16 | 244, 682, 409, 425 |
| `char` | 0.88/196, 0.68/226, 0.69/128, 0.91/60 | 0.39, 0.58, 0.42, 0.38 | 0.60, 0.61, 0.44, 0.56 | 0.85/12, 0.89/0, 0.60/0, 1.00/0 | 433, 740, 469, 360 |
| `char_aug` | 0.90/60, 0.95/0, 0.94/34, 0.98/0 | 0.64, 0.59, 1.00, 0.69 | 0.89, 0.87, 1.00, 0.91 | 0.69/0, 0.73/0, 0.60/1, 0.98/0 | 696, 188, 800, 686 |

**Other categories, every arm:**
- ID coverage was 1.00, with at most 1 error.
- R abstention was 1.00.
- T1 accuracy was 1.00.
- The grammar control was perfect on every category, by construction.
- The other S4 strata and S1's round trips are in the tables output.

**What round 2 shows.**
- **The lead hypothesis meets its criteria in 3 of 4 development units.** In the fourth, both S4 variants
  reached 0.94 against 0.95.
- **Augmentation is necessary but not sufficient.** It helps the word encoder a great deal. The character
  encoder with it is unstable across inits.
- **S2 is not solved.** `word_aug`'s errors in two units break the bound. The revision-3 expectation that
  coverage would be low is wrong once the certificate is loose.
- **S1 fails in every arm,** and the evaluation could now have recognized success.

**Decisions still needed before freezing:**
1. **Sampling** (unresolved item 6).
2. **S2's sample size,** which allows at most one error at 80% coverage.
3. **The 0.95 abstention bar** is an observed proportion. With about 540 answer-ambiguous questions,
   its sampling noise is about ±0.02 (95%). A true rate of 0.96 is observed below 0.95 10–14% of the time
   (exact binomial at 541 and 537 questions).
