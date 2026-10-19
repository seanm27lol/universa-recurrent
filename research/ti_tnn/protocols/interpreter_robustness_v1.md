# Interpreter robustness: new structure, taught wording, unfamiliar wording (DRAFT brief)

**Status: DRAFT for review, not frozen.** Nothing here has run. A development pilot on worlds 100–101 may
revise the proposed bars before freezing, as the plan allows. Confirmation worlds stay untouched until
then. The decisions still open are listed at the end.

## Why

In [#48](../reports/typed_routing_v1.md), a learned interpreter turned controlled-language questions into
programs. In distribution it answered 98–100% of supported questions with 2 errors in 13,714. Under
unseen phrasings, 59 of the 100 answers it accepted were wrong. The [review](../integration_plan.md) found
three reasons that result cannot be read as a statement about interpretation:

- **Information loss.** Every unfamiliar word becomes `<unk>`, so "the nation that includes X" and "the
  airfield that serves X" are the same input. That forces 9.9–12.0% errors before any learning question
  arises.
- **A trivial control would have passed.** Every shifted question contained an unfamiliar word and no
  in-distribution one did, so rejecting unfamiliar words refuses the whole shift. The old rule (bounded
  error, or nothing answered) would count that as success.
- **Calibration was barely needed and not validly certified.** With no threshold, the error bound held in
  14 of 15 units. The threshold search also had no selection-corrected guarantee.

**This experiment separates three questions:**
1. Can the interpreter compose familiar words in structures it has not seen?
2. Can it use alternate wording whose meaning it was taught?
3. Does it abstain when essential wording is unfamiliar, while still answering when only inessential
   wording is unfamiliar?

The simple controls run first. Any competence claim requires coverage as well as bounded error.

## Design

### Worlds, units and modules

- **Worlds.** Development pilot: worlds 100–101, inits 100–101. Confirmation: `generate_world` seeds
  **700, 701, 702** (48 countries, as in #48), inits 0–4. Each (world, init) is a unit.
- **Modules.** The relation modules are the table modules. In #48 the specialist, shared and table
  registries gave identical results, so the interpreter was the only source of error. Fixing the modules
  isolates it. *(Open decision 4.)*

### Language

Entities are copied from the question as in #48. All other words come from four disjoint phrase sets:

| Set | Where it appears | Examples |
|---|---|---|
| **Base** | Training, calibration, every test category | The #48 phrasings ("the capital of X", "X 's capital", "the airport serving X") and frames ("what is ... ?", "name ... .") |
| **Taught synonyms (T)** | Training **only in one-step questions**; tested in longer questions | "the administrative centre of X" (capital), "the most populous city of X" (largest city), "the nation of X" (country), "the airfield of X" (airport), "the firm running X" (operator) |
| **Held-out essential (H)** | Never in training; tested only | "the seat of government of X" (capital), "the metropolis of X" (largest city), "the state that holds X" (country), "the aerodrome near X" (airport), "the outfit managing X" (operator); unsupported relations with unseen words ("the currency of X") |
| **Novel inessential (N)** | Never in training; tested only | Frames "could you tell me ... ?", "i would like to know ... ." and fillers "exactly", "currently", inserted where they don't change the program |

**How H and N were chosen.** H phrases share no word with T or Base, apart from "the" and "of", so their
meaning cannot be guessed from shared words. N words carry no relation meaning by construction. Every
question uses one style (all prefix or all possessive), as repaired in #48, so each has exactly one
reading. Refusal-question strings are deduplicated across splits.

### Training and test categories

Programs of length 1–2 are split by (program, entity) instance: training 30%, calibration 30%,
in-distribution test 40%. Length-3 instances appear only in category S1.

| Category | What it contains | Correct behavior | Question it answers |
|---|---|---|---|
| **ID** | Base wording, length 1–2, test instances | Answer | Sanity |
| **S1, structure** | Base wording, **length 3** (training has length ≤ 2) | Answer | 1 |
| **S2, taught synonyms** | Length 2–3 with at least one T phrase (training saw T only in one-step questions) | Answer | 2 |
| **S3, inessential novelty** | ID questions with an N frame or filler | Answer | 3 |
| **S4, essential novelty** | Length 1–3 with at least one H phrase | **Abstain** | 3 |
| **R, refusals** | Ill-typed, unsupported, unknown entity, ambiguous (base wording) | **Abstain** | Sanity |

### Interpreters

All three are trained on the same training split.

| Name | What it is |
|---|---|
| `word` | The #48 interpreter: word embeddings, every unfamiliar word `<unk>` (the control) |
| `char` | The same encoder and decoder, but each word is embedded by a character-level BiGRU, so unfamiliar spellings stay distinct |
| `char_aug` (proposed method) | `char`, plus **declared extra supervision**: novelty augmentation, described below |

**Novelty augmentation** (`char_aug`) adds two kinds of training question:
- some relation phrases are replaced by pseudo-words, labeled `<unsupported>`. This teaches abstention
  when the word that decides the relation is unknown;
- pseudo-word frames and fillers are inserted into supported questions, keeping the original program.
  This teaches that unknown words elsewhere don't matter.

The pseudo-words come from a seeded generator and are checked to be disjoint from every H and N word, so
the test words themselves are never seen. *(Open decision 3.)*

### Answering policies

Each policy is applied to each interpreter.

| Policy | Rule |
|---|---|
| `none` | Answer whenever the interpreter proposes a program that the runtime accepts and runs |
| `raw` | Answer only above a threshold on the raw score, certified as below |
| `calibrated` | The same, on Platt-calibrated probabilities |
| `novelty_reject` | `none`, plus abstain whenever a non-entity token is outside the training vocabulary |

**Certification.** Thresholds are chosen on the calibration split, which is in distribution, with
selection controlled as a multiple-testing problem, in the spirit of Learn then Test (Angelopoulos,
Bates, Candès, Jordan and Lei, [arXiv 2110.01052](https://arxiv.org/abs/2110.01052), 2021).
- A grid of thresholds is fixed in advance and tested in order, from strictest to loosest.
- For each threshold, the null hypothesis is "accepted error > 1%", with an exact binomial p-value.
- Testing stops at the first threshold that cannot be rejected at δ = 0.05, and the loosest rejected one
  is chosen. If none is rejected, the policy never answers.
- Testing a fixed sequence this way bounds the probability of choosing a threshold whose true error
  exceeds 1% by 0.05.

The test categories are then scored once. Per-unit 95% Clopper–Pearson bounds are reported for every
category. *(Open decision 6.)*

### Measurements

For each unit, interpreter, policy and category:
- coverage, as answers over questions that should be answered;
- accepted errors over answers, with the 95% bound;
- abstention on S4 and R;
- the interpreter's exact program-and-entity accuracy;
- input collisions: identical model inputs with the same copied entity and different required answers.

**Retained records.** Every shifted-question decision is kept: question id, category, parse, score,
decision and correctness. A deterministic sample of 200 per unit for ID is kept too.

## Proposed predictions (one-sided; the bars may be revised by the development pilot, then frozen)

**What counts as competence.** For a category that should be answered, an interpreter-and-policy pair
**answers where it should** if:
- its coverage is at least **0.80**;
- its 95% bound on accepted error is at most **1%**.

For a category that should be refused, it **abstains where it should** if it abstains on at least
**0.95**. Either holds for a world if it holds on at least 4 of its 5 inits.

1. **Sanity.** Every interpreter with `calibrated` answers in-distribution questions where it should,
   with coverage of at least 0.90, and abstains on R where it should.
2. **The representation removes collisions.** `char` and `char_aug` have no conflicting input
   collisions on S1–S4; `word` has some on S4. *(This holds by construction, since distinct strings give
   distinct character inputs; it is recorded, not claimed as learned.)*
3. **Taught synonyms (question 2).** `char_aug` with `calibrated` answers S2 where it should.
4. **Inessential novelty (question 3a).** `char_aug` with `calibrated` answers S3 where it should, and
   `novelty_reject` does not (its S3 coverage is near zero by construction).
5. **Essential novelty (question 3b).** `char_aug` with `calibrated` abstains on S4 where it should.
6. **The augmentation is what separates 4 from 5.** No pairing of `word` or `char` with any of the
   four policies meets both 4 and 5.
7. **Structure (question 1).** `char_aug` with `calibrated` answers S1 where it should.

**What I expect** (stated, not predicted):
- 1–4 hold.
- 5 and 6 are uncertain. They are the core question: can a model learn which unfamiliar words matter
  from their position alone?
- 7 is likely to fail. Recurrent sequence-to-sequence models are known to generalize poorly to longer
  outputs than they were trained on. If so, that is a finding about this interpreter, not a defect of the
  test.

**Holds by construction** (recorded, not counted as findings):
- 2;
- `novelty_reject`'s S4 abstention (1.0) and its S3 coverage (about 0);
- the runtime's refusal of ill-typed and unknown-entity questions.

## Limits

- **Controlled language, with synonyms we defined.** This says nothing about natural language.
- **The augmentation is extra supervision.** It teaches the property tested in S3 and S4 using different
  words, so a success shows the property can be taught. It does not show it emerges unaided.
- **Pseudo-words are random strings.** Real unfamiliar words have morphology; a character model may
  exploit it, rightly or not.
- **Table modules.** End-to-end results with learned modules are #48's.

## Compute

CPU only. Expected cost:
- **Pilot:** 2 worlds × 2 inits × 3 interpreters, estimated under 15 minutes.
- **Confirmation:** 3 worlds × 5 inits × 3 interpreters. The char encoder is roughly 3–5 times the word
  encoder's cost, about 30–60 minutes in total.

The pilot measures this before the budget is frozen.

## Open decisions for review

1. **Representation:** a character-level word encoder (proposed), bytes, or subwords. Characters keep
   unfamiliar spellings distinct with the least machinery; subwords would need a tokenizer trained on the
   training split.
2. **Taught versus held-out wording:** the T and H tables above.
3. **Novelty augmentation as the method under test.** It is declared extra supervision. Without it,
   prediction 6 asks whether confidence alone can tell essential from inessential novelty, which #48
   suggests it cannot.
4. **Table modules** to isolate the interpreter (proposed), or learned specialists as in #48.
5. **The bars:** coverage 0.80 with a 1% bound; abstention 0.95; in-distribution coverage 0.90.
6. **Certification:** fixed-sequence testing on the in-distribution calibration split (proposed), or a
   separate select-then-certify split.
