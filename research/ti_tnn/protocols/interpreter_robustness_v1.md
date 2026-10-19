# Interpreter robustness: new structure, taught wording, unfamiliar wording (DRAFT brief, revision 2)

**Status: DRAFT, not frozen.** This revision follows the review of the first draft (dfe142e); the changes
it made are listed at the end. A bounded development pilot on worlds 100–101 runs next. Its results will
fix the threshold grid, the bars, the sample sizes and the aggregation rule before anything is frozen.
Confirmation worlds 700–702 stay untouched until then.

## Why

In [#48](../reports/typed_routing_v1.md) the interpreter failed under unseen phrasings, but that result
mixed three things:
- **information loss:** every unfamiliar word became `<unk>`;
- **a trivial control:** rejecting unfamiliar words would have refused the whole shift;
- **calibration:** it was barely needed, and its threshold had no selection-corrected guarantee.

This study separates the questions, runs simple controls first, and requires coverage for any
competence claim.

## Design

Code: `bench/interpreter_robustness.py`, `integrated/interpreters.py`, `integrated/grammar.py`, and the
certification functions in `integrated/calibration.py`.

### Worlds, units, modules

- **Worlds.** `generate_world`, 48 countries. Development: worlds 100–101, inits 100–101. Confirmation:
  worlds **700–702**, inits 0–4. Each (world, init) is a unit.
- **Modules.** Table modules, to isolate the interpreter. In #48 the module registries gave identical
  results.

### Language

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

### Items

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

### Audit (before any training; the run refuses to start if any check fails)

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

### Interpreters: a 2 × 2 factorial, and a grammar control

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

### Answering policies

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

### Certification

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

### Measurements

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

## What runtime refusals do and do not show

- **Ill-typed refusals hold only given a faithful parse.** If the interpreter reads "what is the capital of
  country_0_city_0 ?" as `country_of(country_0_city_0)`, the program is well-typed and the runtime
  answers "country_0". So R abstention measures the interpreter as well as the runtime.
- **Unknown-entity refusal has a narrower guarantee.** Such a question contains no valid world entity, and
  the entity head can only copy a question token, so the policy cannot bind a valid entity.

## Proposed predictions (one-sided; bars provisional until the pilot)

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

## Limits

- **Controlled language with defined synonyms.** Nothing here concerns natural language.
- **The augmentation teaches the property under test,** with other words. A success shows it can be taught,
  not that it emerges unaided.
- **The grammar control knows the grammar.** Matching it is useful; it does not establish that a learned
  model is needed.
- **Table modules.** End-to-end learned-module results are #48's.

## Compute

CPU only. Measured in a smoke run, 100 updates take about 1 s for a word arm and 2.5 s for a char arm. At
1,500 updates:
- **pilot:** 16 fits, about 10 minutes;
- **confirmation:** 60 fits, under 1 hour.

## Changes from the first draft (after review)

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

## Development pilot, round 1 (2026-10-18; disclosed; not a result)

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

**Changes proposed for review before freezing** (none applied yet):
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
