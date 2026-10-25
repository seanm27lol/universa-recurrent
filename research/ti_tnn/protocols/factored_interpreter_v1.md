# A factored interpreter: learned lexical grounding under a supplied typed grammar (draft, revision 3)

**Status: CLOSED (2026-10-25) as a development engineering result, at revision 3.** It was never frozen, and no
confirmation was run ("Closing record" below; [report](../reports/factored_interpreter_v1.md)).

**Revision 3 (2026-10-25), after the review of revision 2:**
- **The bounded engineering pilot has run** on development worlds 100–101: 16 fits of A and F, with A+K, F+B, L, G
  and W evaluated from them. The independent verifier re-derived it. Results are in "History".
- **The lexical-generalization extension** is at the specification and no-fit-control stage. No neural model has
  been fitted on it.
- **Nothing is frozen.** Worlds 1000–1002 are reserved and untouched (logged inventory check in "History").

**Where this comes from.** It is item 2 of the order recorded in the [integration plan](../integration_plan.md)
after the review of #61: the next main learning study. It responds to:
- **S2:** taught wording did not transfer into two-step questions ([robustness](../reports/interpreter_robustness_v1.md));
- **S1:** non-cancelling three-step composition failed;
- **the observed failure modes** in the [S2 diagnosis](../reports/s2_diagnosis_v1.md): premature stopping and dropped
  relations. They were measured; a causal shortcut explanation was not established.

## A concrete example

In training, "the airfield of" appeared only in one-step questions ("what is the airfield of country_29_city_0 ?").
The question below is from the robustness confirmation (world 702, init 2):

> name the company operating the airfield of country_29_city_0 .

- **The current interpreter** (`word_aug`, a sequence-to-sequence decoder) proposed `airport_of`, then stopped.
- **The supplied grammar** splits it, with no lexicon, into the segments "the airfield of" and "the company
  operating". The innermost applies first.
- **A grounding step** maps each segment to a relation, `<unsupported>` or `<ambiguous>`.
- **The program** (`airport_of`, then `operator_of`) is type-checked from the entity and executed.

## What the no-fit controls already show

**The development result.** L is the supplied grammar plus a lookup dictionary induced from the aligned training
pairs. On development worlds 100 and 101, it made **no wrong decision, no error and no accidental agreement**:
- over all 17,900 calibration and test items per world;
- under full and reduced exposure;
- with raw tokens as keys, and with the training-vocabulary representation proposed for F.

G, the reference lexicon, did the same. So did W, the word-association baseline added in revision 3. The independent
verifier re-derived every record. This reproduces the review's no-fit check, with a separate implementation.

**Why.** Every supported segment in every evaluation category occurs as an exact string in training, under both
regimes:
- the categories: ID, T1, S2, S3, S1 and the length-3 diagnostics;
- the forms: 20 relation forms, 4 per relation (two Base prefix phrasings, the taught phrasing, the possessive
  phrasing);
- the reduced regime keeps all 20, because 3 examples of a taught phrase present the same segment as 28–107.

Held-out (S4) segments never occur in training:
- with raw keys, they are unseen, and L refuses them;
- with the training vocabulary, about 82% map to an `<unk>` pattern that essential augmentation labelled
  `<unsupported>`, and the rest are unseen. Either way L refuses.

**What follows.**
- **This language requires no lexical generalization.** Exact lookup over the training segments suffices.
- **The rest of this study is an engineering comparison** of replacement systems (A, A+K, F, F+B) against L and G,
  on this language. It cannot establish a need for neural lexical generalization or for bounded alternatives.
- **L stays in every comparison.** If L solves the task, that is reported. It is not excluded to create an
  apparent need for neural grounding.
- **A lexical-generalization study would need a separately specified condition.** One is sketched below, as a
  decision for the reviewer. Nothing in it is implemented.

## The question

1. **Replication and the decoder control.** Under exact-program correctness, does A fail S2 and S1 again? A is
   also reported under the historical answer metric. Does constraining A's own recurrent decoding (A+K) recover
   what the supplied grammar recovers?
2. **The engineering comparison.** Does a neural grounding model (F) match the exact-lookup control (L) on this
   language? What do bounded alternatives (F+B) add? Each system is measured by certified coverage, accepted
   error, refusal behaviour and total cost. Construction properties are kept separate from empirical results
   (table below).
3. **Lexical generalization** is studied only under the separately specified condition, if the reviewer
   approves it.

## Systems

| System | What it is | Fitted? |
|---|---|---|
| **A** | `word_aug` as in the robustness study: the same encoder, decoder, augmentation, training and greedy decoding, with its entity pointer | yes |
| **A+K** | A grammar-admitted constrained policy sharing A's model and entity pointer: A's fit, decoded under the supplied grammar's constraints inside its recurrence (below) | no: A's fit |
| **F** | The supplied grammar plus a neural segment-grounding model, greedy | yes |
| **F+B** | F's fitted model, with the bounded-alternatives policy (below) | no: F's fit |
| **L** | The supplied grammar plus a lookup dictionary induced from the same aligned training pairs available to F | no |
| **G** | The supplied grammar plus the reference lexicon (Base and taught phrasings) | no |
| **W** | The supplied grammar plus training-derived word associations: each word's label set from the aligned pairs, intersected across a segment (below) | no |

**L and G** differ only in where the lexicon comes from. L learns it from examples; G receives it.

**Shared by every system:**
- the maximum program length (3);
- the runtime's type checks and execution on table modules (`runtime.execute`);
- the refusals (`<unsupported>`, `<ambiguous>`, unknown entity, ill-typed program).

**Entity binding** is deterministic in F, F+B, L and G: the grammar's entity token. A and A+K keep A's learned
entity pointer, so that A+K remains a control for the decoder constraints alone.

### The supplied grammar (implemented: `integrated/segment_grammar.py`)

- **Entity:** exactly one token names an entity of the world. With none, the question is refused as an unknown
  entity; with several, as malformed.
- **Prefix style:** "the" occurs before the entity, and "'s" does not occur after it. Segments run from the first
  "the" to the entity, and each "the" starts a new one. The words before the first "the" are frame.
- **Possessive style:** "'s" occurs after the entity, and "the" does not occur before it. The first token after
  the entity is "'s". Each "'s" starts a segment, which runs to the next "'s" or the final "?" or ".". The words
  before the entity are frame.
- **Malformed:** both styles, neither, words between the entity and its first "'s", or more than 3 segments.
- **Execution order:** innermost first for prefix questions, left to right for possessive ones.
- **It never consults a lexicon.** Held-out phrases are segmented like any other. Frames, novel frames and
  fillers fall outside the segments, so F, F+B, L and G never see them.

### Training-derived segment labels (F and L)

**Source:** the same source training pairs, with supplied structural alignment and declared filtering. This is
not identical effective supervision to A's:
- A sees whole questions;
- F and L see aligned segments.

**Aligned.** A program target (relations, then `<end>`) aligns its relations to the segments in execution order.
That includes ill-typed refusals: their relation phrases are meaningful, and only the whole question must be
refused, by the runtime. Their labels are kept.

**Single-segment refusals.** A refusal target (`<unsupported>`, `<ambiguous>`) labels a segment only when the
question has exactly one segment.

**Excluded,** with every exclusion exported:
- questions the grammar cannot segment: the unknown-entity refusals, which have no world entity to bind;
- refusal targets over several segments: the target does not say which segment carries the refusal.

**Sources:**
- the training-split questions of the regime: Base, taught and the quota refusals;
- the robustness study's novelty augmentation. Its inessential frames fall outside the segments; its essential
  pseudo-word phrases become `<unsupported>` segments, when single.

**Exported before any fit:**
- total and unique (segment, style, label) tuples, with multiplicities;
- conflicts;
- the contributing questions;
- the excluded refusals and augmentation, with reasons;
- the dictionaries.

F's actual segment presentations during training are exported with its fit.

**Two key representations,** both with the style:
- **raw:** the tokens;
- **vocab:** the word interpreters' training vocabulary. That is every word of the regime's training-split
  questions, augmentation excluded; any other word becomes `<unk>`.

### L, the training-derived lookup

**The dictionary:**
- a key observed with one label maps to it;
- a key observed with conflicting labels refuses;
- an unseen key refuses.

**A decision.** Segments are looked up in execution order. The first that is unseen, conflicting or refusal-labelled
decides the abstention and its reason. Otherwise the program is bound to the entity, type-checked and executed by
the runtime.

**What L may not use:** generator phrase labels or evaluation data.

**L is reported in both representations,** raw and vocab. The vocab one uses F's input representation.

### W, the training-derived word association (no-fit; added in revision 3)

**The association.** From the same aligned training pairs as L, W collects, for each (style, word), the set of
labels of the training segments containing that word.

**A segment's label** is the intersection of its words' label sets:
- a unique relation is used;
- a unique refusal label refuses;
- an unseen word, an empty intersection or several labels refuse.

**Its decisions** are made like L's: segments in execution order, then binding, type checks and execution by the
runtime.

**Why it is here.** It is the no-fit baseline for lexical generalization: it grounds a segment never seen as a
string, when its words identify it.

### F, the neural grounding model (specified before any fit)

- **Input:**
  - the segment's tokens in the vocab representation, with `<unk>` for unseen words;
  - the style;
  - never the entity or the frame.
- **Architecture:**
  - a word embedding (d = 48) over the training vocabulary plus `<pad>` and `<unk>`;
  - a one-layer bidirectional GRU (hidden 64 per direction) over the segment;
  - the final states of both directions, concatenated with a learned style embedding (d = 8);
  - one linear layer to 7 labels: the five relations, `<unsupported>` and `<ambiguous>`.
- **Optimizer:**
  - Adam at learning rate 3e-3;
  - 1,500 updates, A's budget;
  - batches of 64.
- **Sampling and weighting:** per extracted segment presentation. Each batch draws 64 presentations uniformly with
  replacement from the aligned presentations, so multiplicities from the source pairs are preserved. The loss is
  unweighted cross-entropy. A samples per question, so the two differ in weighting, and this is disclosed.
- **Initialization:** the init seed. CPU only, 4 threads.
- **Its score** is the product of its segments' chosen-label probabilities. The entity is bound deterministically,
  so its binding contributes 1.
- **Training cost is reported as realized:** the presentations actually drawn, per (style, segment, label) tuple;
  the tokens; the forward MACs from the layer shapes; and wall time. Equal update counts (1,500 × 64 for A and for
  F) are not equal compute.

### F+B, the bounded-alternatives policy (complete)

Everything uses F's **original seven-label probabilities**. Nothing is renormalized after refusal labels are set
aside or after ill-typed programs are pruned. The steps, in order:

1. **Refusal before search.** If any segment's highest probability belongs to `<unsupported>` or `<ambiguous>`,
   F+B abstains. A tie between a refusal label and a relation also abstains. This only guarantees that search
   cannot override an argmax refusal.
2. **Candidates per segment:** the relations with probability ≥ 0.05 (inclusive), ranked by probability, with ties
   broken by relation name. The top k = 2 are kept. A segment with no such relation means abstention.
3. **Programs:** every combination of the segments' candidates, at most 8. A program's score is the product of its
   relations' original probabilities.
4. **Type pruning:** programs that are ill-typed from the entity are pruned. Every survivor is executed.
5. **Selection:** the survivor with the highest score. A tie in score goes to the lexicographically smallest
   program. With no survivor, F+B abstains.
6. **Disagreement:** if another survivor scores at least 0.5 × the selected score (inclusive) and gives a different
   answer, F+B abstains.
7. **Refusal check:** every relation of the selected program must have a strictly greater original probability
   than both `<unsupported>` and `<ambiguous>` for its segment. Otherwise F+B abstains; a tie abstains too. This
   stops type pruning from promoting a weak relation over a stronger refusal prediction.

**What it guarantees.** None of this guarantees that every unknown meaning is recognized. These are declared policy
choices, and S4 measures how they behave.

**Exported per item:**
- every segment's complete distribution;
- the candidates, and each program's pruning decision;
- the executed programs and their answers;
- the selected program;
- the decision and its reason.

**A worked fixture** (it becomes a test). The question is one segment over an Airport entity, with probabilities:

| capital_of | `<unsupported>` | operator_of | largest_city_of | country_of | airport_of | `<ambiguous>` |
|---|---|---|---|---|---|---|
| 0.31 | 0.30 | 0.10 | 0.09 | 0.08 | 0.07 | 0.05 |

Step by step:
- **Step 1:** the argmax is `capital_of`, so search proceeds.
- **Step 2:** the candidates are `capital_of` and `operator_of`.
- **Step 4:** `capital_of` is ill-typed from an Airport and is pruned. `operator_of` survives.
- **Step 5:** `operator_of` is selected.
- **Step 7:** 0.10 is not greater than `<unsupported>`'s 0.30, so **F+B abstains.**

Revision 1's rule would have answered `operator_of`, though `<unsupported>` was three times as likely.

### A+K, constrained decoding inside the recurrence (complete)

**A+K is a grammar-admitted constrained policy sharing A's model and entity pointer.** It answers only questions the
grammar admits. Its grammar refusals are reported separately from its own decisions.

The decoder feeds each chosen token into its next GRU step. So the constraints are applied while decoding, not to
the logits of an unconstrained pass. The steps:

1. **The grammar comes first.** If it refuses (unknown entity, malformed), A+K abstains with its reason; it needs
   the slot count n.
2. **The entity.** A's entity pointer chooses the entity token. If that is not a world entity, A+K abstains
   (unknown entity). Otherwise its type starts the type constraint.
3. **Each step t = 0 … n:**
   - compute the logits under the prefix actually chosen so far;
   - restrict the choice to the allowed set:
     - for t < n: the relations that keep the prefix well typed from the entity, plus `<unsupported>` and
       `<ambiguous>`;
     - at t = n: `<end>`, `<unsupported>` and `<ambiguous>`;
   - choose the allowed output with the highest logit, with ties going to the earlier output index;
   - feed that output into the next step. Choosing `<unsupported>` or `<ambiguous>` ends the decoding with that
     refusal.
4. **The score** is the product of the chosen outputs' original unmasked probabilities, `<end>` included, times the
   entity pointer's probability. A forced choice keeps its original probability; nothing becomes 1 by
   renormalization.
5. **Exported per item and step:**
   - the full output distribution;
   - the allowed set;
   - the chosen output;
   - the entity pointer's choice and its probability.

## Training regimes

| Regime | Taught phrases in training | Status |
|---|---|---|
| **full** | The robustness study's training: each taught phrase in 28–107 one-step questions (development worlds) | The reference condition |
| **reduced exposure** | Each taught phrase keeps exactly 3 of its one-step questions, by a seeded draw; nothing else changes | A reduced-exposure condition |

**F discards the entity and the frame.** So 3 examples of a fixed taught phrase present exactly the distinct
grounding input that the full regime presents. Only the multiplicity changes: 15 taught presentations instead of
354–365.

**Training is not reduced further** to make the neural model uncertain. Ceiling results are reported as they are.

## The lexical-generalization condition (specification and no-fit controls only)

**What it is for.** It tests grounding of supported segment forms that never occur as exact strings in training,
but whose meaning training identifies. It is kept separate from S4, whose untaught essential wording must still be
refused. No neural model is fitted on it.

**Identifiability, defined operationally.** A form is identifiable from the training pairs of a regime when all three
hold:
- every word of the form occurs, in the same style, in aligned training segments;
- the intersection of those words' training label sets is exactly one relation, which is W's grounding;
- the form occurs nowhere in training as a segment string.

No held-out (H) word may occur in a form. Revision 2 described the rule as "every word occurs in phrasings of one
relation", which was wrong for "the", "of" and "city". The intersection is the rule.

**The complete phrase inventory** (`bench/lexical_condition.py`, `inventory`) follows one declared rule:
- take each relation's taught head (the words between "the" and the taught phrasing's last word);
- combine it with each connector (last word) of the relation's Base prefix phrasings, except "of";
- drop any string that already occurs among the declared phrasings.

That gives **4 distinct phrases.** `capital_of` has none, because its Base phrasings use only "of".

| Relation | Taught | Base phrasing with another connector | New phrase |
|---|---|---|---|
| `largest_city_of` | the most populous city of | the biggest city in | the most populous city in |
| `country_of` | the nation of | the country containing | the nation containing |
| `airport_of` | the airfield of | the airport serving | the airfield serving |
| `operator_of` | the firm running | the company operating | the firm operating |

**Sampling** (`lexical_items`). Draws are independent, with replacement, from the robustness study's held-out
population (instances never used in training), in prefix style, with a seed fixed per world:
- **LG1:** 400 draws of a one-step instance whose relation has a new phrase;
- **LG2:** 800 draws of a two-step instance, with a position j chosen uniformly among the positions whose relation
  has one.

That is 4,400 questions per world.

**Matched controls.** Every draw is one group of variants, each its own category, never pooled:

| Design | Variant | The phrasing at the position | Expected |
|---|---|---|---|
| LG1 | `LG1` | the new phrase | answer |
| LG1 | `LG1_taught` | the taught phrasing | answer |
| LG1 | `LG1_base` | a Base phrasing | answer |
| LG2 | `LG2` | the new phrase at j, Base elsewhere | answer |
| LG2 | `LG2_taught` | the taught phrasing at j | answer |
| LG2 | `LG2_base` | Base throughout | answer |
| LG2 | `LG2_held` | the held-out phrasing at j | abstain |

**Criteria, for any future neural lexical run.** A system grounds the new phrases when, in the same unit, all four
hold:
- it has a certificate;
- `LG1` and `LG2` each have coverage ≥ 0.80 with a 95% error bound ≤ 1%, under primary correctness;
- `LG2_held`, `S4a_primary` and `S4b_primary` each have abstention ≥ 0.95;
- it passes the sanity checks.

It must be reported against W, the no-fit standard of this condition. A neural run that only matches W shows no
need for neural lexical generalization.

**The no-fit results** (development worlds 100–101, both regimes; verified independently):
- **The inventory:** 4 distinct phrases in each world. They are used in 1,200 LG1 and LG2 questions per world:
  - "the nation containing" 490 and 534;
  - "the airfield serving" 317 and 273;
  - "the firm operating" 210 and 218;
  - "the most populous city in" 183 and 175.
- **Identifiability:** each of the 4 phrases is identifiable in both regimes. All its words are seen, W's
  intersection is its relation, and it is never seen as a segment.
- **W grounds every new phrase:**
  - `LG1` 400 of 400 and `LG2` 800 of 800 answered, with exact programs;
  - it refuses `LG2_held` (0 of 800 answered);
  - it keeps S4 refusal: `S4a_primary` and `S4b_primary` 2,000 of 2,000 each;
  - on the standard 17,900 items it makes no wrong decision.
- **L and G refuse the new phrases** by construction (0 of 400 and 0 of 800) and answer the taught and Base variants.

**What follows.** On this condition, a no-fit word association already solves the proposed phrases. A neural lexical
run is therefore not proposed with this inventory. A future condition would need forms that W cannot ground, yet
whose meaning training identifies by a stated rule. That rule would have to be specified, with its own no-fit
standard, before any fit.

## Populations, categories and correctness

**Populations and categories** are the robustness study's: its declared distributions, independent draws and
audit, with 96 countries.
- **The ID distribution:** calibration 1,000 draws, then the ID and R test draws.
- **T1, S2, S3.**
- **S4a and S4b:** the primary answer-ambiguous samples and the population sample, by stratum.
- **S1:** non-cancelling chains and round trips.
- **The length-3 diagnostics.**

In all, 17,900 evaluated items per world.

**Primary correctness** (every system). An answer is correct only when all four hold:
- answering is expected;
- the ordered relation program is exactly the reference's;
- the bound entity is the reference's;
- the executed answer is the reference's.

**Errors and abstentions:**
- an answer on an S4 or refusal item is an error, even when its program or answer happens to match;
- an abstention is correct exactly when abstaining is expected.

**Accidental agreement,** a right answer from a wrong program or entity, is reported separately.

**The certification code needs a new predicate.** The robustness study's certification code checks answer
correctness only, so it is not reused unchanged. Its fixed-sequence procedure receives the primary-correctness
predicate.

**Kept beside it:**
- the historical answer-based metric, retained separately for A's replication;
- every category is reported per stratum;
- an empty stratum is reported as untestable, never as perfect abstention.

## Measures

**Per system, unit, policy and category:**
- coverage, accepted errors and the 95% upper error bound;
- abstention, with its exact interval;
- exact-program accuracy, with abstentions counted as failures;
- accidental agreement;
- abstention reasons.

**By S4 stratum:** answer-ambiguous, one answer, type-determined. A system that answers a type-determined held-out
phrase by type has not shown that it grounded the phrase, and the contract requires abstention.

**Scores.** The certified policy uses the probability product, as fixed above. The minimum and the geometric mean
are reported diagnostics, by program length. No score is chosen after confirmation calibration or test results
are seen.

**Costs per question:**
- forward passes and decoding steps;
- candidates proposed, pruned and executed;
- module applications;
- wall time.

Training cost is reported separately, along with F's segment presentations.

**The admission detector.** Where refusals act as an admission check, its false admissions (S4 answered) and false
refusals (S3, ID, S2 or T1 refused) are reported.

## Certification

**Each system's complete release policy is certified on its own calibration draws.** It uses the fixed-sequence
rule:
- the grid 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2 and 0, strictest first;
- an accepted error of 1%;
- δ = 0.05;
- the primary-correctness predicate.

A certificate for A's greedy policy does not carry over to A+K, F or F+B.

**The deterministic policies** L, G and W have no score. Each gets a single calibration risk test, on the same
calibration items and the same primary-correctness endpoint as the neural policies. The null "accepted error ≥ 1%"
is rejected when P(Binomial(n, 0.01) ≤ e) ≤ 0.05, where n is the number of calibration items it answers and e its
errors. A policy that fails the test answers nothing under the certified policy.

**The feasibility check in development.**
- **The risk.** The strictest rule may accept too few calibration questions to certify, and that ends the sequence
  before looser rules are tested.
- **The numbers.** Fewer than **299** accepted items cannot certify, even with zero errors (0.99²⁹⁹ ≈ 0.0495 ≤ 0.05;
  0.99²⁹⁸ ≈ 0.0500). The largest error count that can still certify grows with the accepted count:

  | Accepted items n | Errors allowed |
  |---|---|
  | fewer than 299 | none certifies |
  | 299–472 | 0 |
  | 473–627 | 1 |
  | 628–772 | 2 |
  | 773–912 | 3 |

  Revision 2 omitted the 299 floor.
- **The check.** Development checks every system. Any change to the grid or its order is disclosed and made before
  freezing.

**What the guarantee covers:** the ID distribution of one world and fitted model. It does not cover S2, S1 or S4, and
it is not simultaneous across systems or units.

## Construction checks and empirical results

| Outcome | A | A+K | F | F+B | L | G |
|---|---|---|---|---|---|---|
| S3 wrapper invariance (wrappers outside the segments) | empirical | empirical | **construction** | **construction** | **construction** | **construction** |
| S2, S1, T1, ID exact programs | empirical | empirical | empirical, but on segment forms all seen in training: a memorization check | as F | **construction**, given the development support counts | **construction** |
| S4 refusal (unseen or `<unk>`-pattern segments) | empirical | empirical | empirical. About 82% of held-out segments map to `<unk>` patterns trained as `<unsupported>`; the rest are unseen patterns | empirical (policy) | **construction** | **construction** |
| R refusals | empirical | empirical | grammar and runtime refusals: construction; refusal labels: empirical | as F | **construction** | **construction** |
| Certified coverage and risk | empirical | empirical | empirical | empirical | observed (deterministic) | observed |
| Cost | measured | measured | measured | measured | measured | measured |

**W,** like L and G, is deterministic: its outcomes are construction checks, given the training support.

## Predictions (one-sided; bars provisional until development)

**The criteria:**
- **answers where it should:** a certificate, coverage ≥ 0.80 and a 95% error bound ≤ 1%, under primary
  correctness;
- **abstains where it should:** abstention ≥ 0.95.

**The joint rule.** A unit succeeds on a criterion set only if the same unit meets all of it. A world holds when
≥ 4 of 5 inits succeed. A prediction holds when every world holds.

| # | Prediction | Regime | Kind |
|---|---|---|---|
| P1 | **A's world-level verdicts replicate:** in every world, A's S2 verdict and its S1 (non-cancelling) verdict do not hold. That means fewer than 4 of 5 inits meet the criterion, which is not the same as at least 4 failing. It is judged under the historical answer metric, with A's historical answer-based certificate reconstructed separately (the robustness study's code, unchanged). A unit's failure counts as replicated only when that unit passes the historical sanity checks and T1, so that a broken fit cannot pass as a composition failure. Also reported under primary correctness | full | Replication |
| P2 | **Learned grounding composes under the supplied structure:** F answers where it should on S2 and on S1's non-cancelling chains | full | Empirical (a memorization check on this language) |
| P3a | **Grammar invariance:** every S3 item's segments equal its S3_base item's | both | Construction check |
| P3b | **Learned novelty handling:** F abstains where it should on S4a and S4b primary | full | Empirical |
| P4 | **Useful novelty handling, jointly:** F, and separately F+B, has a certificate, S3 coverage ≥ 0.80 with a bound ≤ 1%, and S4a and S4b primary abstention ≥ 0.95, all in the same unit. Refusing everything cannot pass | full | Empirical |
| P5 | **Learning the lexicon costs little:** in each unit, F's unthresholded exact-program accuracy on the complete S2 sample is at least G's minus 2 percentage points, with abstentions counted as failures | full | Empirical |

**Sanity checks** (every system with a score), in the same unit:
- a certificate;
- ID coverage ≥ 0.90 with a bound ≤ 1%;
- R abstention ≥ 0.95;
- T1 exact-program accuracy ≥ 0.95.

**Reported without a prediction:**
- A+K against A;
- every system under reduced exposure;
- F and F+B against L as well as G;
- F+B against F, including its cost;
- the three scores;
- every S4 stratum;
- the admission detector's false admissions and false refusals.

## Export and independent verification

**The export:**
- the training-derived labels and dictionaries, before any fit;
- per item and system: the decision, reason, program, bound entity, answer and score;
- for F and F+B: every segment's complete distribution and every step of the alternatives policy;
- for A and A+K: the step traces, including A+K's allowed sets along its selected prefix.

**Two verifiers.** `bench/factored_interpreter_verify.py` covers the no-fit stage: L, G, W, the support report and
the lexical condition. `bench/factored_pilot_verify.py` covers the pilot.

**What they import:**
- the generators: the robustness study and `lexical_condition`;
- the world;
- numeric binomial helpers and the canonical hash;
- the fixed-sequence certificate written independently in `interpreter_robustness_verify`.

They do not import the producer's grammar, labels, grounding, decoding, decisions, selection or metrics.

**What they re-implement, from this text:**
- the training-derived labels, L's dictionaries and W's associations;
- the segment boundaries and execution order;
- the ordered segment-to-program alignment audit: each segment must be a declared phrasing of the relation at its
  position, not merely equal in number;
- the support report: presentations, sources, unique tuples and keys, conflicts, exclusions, phrase multiplicities and
  evaluation coverage;
- the study and lexical-item identities, and the vocabulary size;
- the decisions of every system:
  - A's, by binding and executing its proposed program, with its float32 score product;
  - A+K's, by re-deriving every allowed set along the chosen prefix and every choice, and the score from the recorded
    original probabilities;
  - F's and F+B's, completely, from the exported distributions;
  - L's, G's and W's;
- exact-program correctness, A's historical answer correctness and accidental agreement, each separately;
- every certificate, the deterministic policies' risk test, every metric and criterion, P1's sanity-gated replication
  and P5;
- realized training cost: A's draws, by replaying the same generator calls, and F's draws against the aligned set,
  with tokens and forward MACs;
- the evaluation cost counts;
- the requested inventory, the configuration, the producer-source hashes and the run-level totals.

**Rejection tests**, all implemented and passing. Each corruption is rejected with its intended message:
- **The no-fit stage:**
  - reordered segments;
  - a wrong entity binding;
  - a wrong label;
  - an altered refusal decision;
  - a missing record;
  - changed correctness counts;
  - altered phrase exposure;
  - altered dictionary coverage;
  - an altered calibration risk test;
  - an altered study identity;
  - one regime missing;
  - both regimes of a requested world removed;
  - a source mismatch;
  - a configuration mismatch;
  - altered run-level and unit totals.
- **The pilot:**
  - an A+K distribution renormalized over its allowed set;
  - an A+K score computed from renormalized probabilities;
  - a widened A+K allowed set;
  - a missing F+B alternative;
  - an altered F+B refusal decision;
  - a changed certificate correctness label;
  - an altered historical certificate;
  - altered F presentations;
  - altered A draws;
  - an altered P5;
  - an altered evaluation cost;
  - an incomplete inventory;
  - a configuration mismatch;
  - a source mismatch.

**Sampled reproduction.** One init per world and regime is chosen by an outcome-independent hash, fixed before
confirmation. A and F are retrained; A+K and F+B are regenerated from those fits; L and G are rebuilt.

## Units, inventory and budget

- **Development:**
  - worlds 100–101 × inits 100–101 × 2 regimes × 2 fitted models (A, F) = **16 fits**;
  - A+K, F+B, L and G are not separately fitted.
- **Confirmation,** if later approved:
  - worlds 1000–1002 × inits 0–4 × 2 regimes × 2 fitted models = **60 fits**, with 96 countries;
  - sampled reproduction: **12 more fits** (20%).
- **Time.**
  - **The robustness word arms** took about 17 s per fit, plus evaluation of about 18,000 items.
  - **The development pilot took 4 minutes of CPU at 4 threads,** 27–30 s per unit. Its realized costs are in
    "History".
  - **The confirmation would take about 15 minutes,** plus about 3 minutes for reproduction.

## Interpretation limits

- **Structure is supplied.** Success under the supplied grammar is not structure learning. This language has a
  lexicon-free segment rule; natural language does not.
- **This language needs no lexical generalization** (the no-fit result). On it, F and F+B can match L, and do no
  more.
- **Controlled language, table modules, synthetic worlds.**
- **One reduced-exposure level,** not a learning curve.
- **The certificates are per world and model,** on the ID distribution only.

## Decisions (from the review of revision 1)

| # | Decision | As specified above |
|---|---|---|
| 1 | Systems | A, A+K, F, F+B and G, with L added. F and F+B are reported against L as well as G |
| 2 | Supervision | Training-derived segment labels: "the same source training pairs, with supplied structural alignment and declared filtering". Ill-typed labels are kept, and every exclusion is exported |
| 3 | Regimes | Full is the reference; reduced exposure is renamed. The no-fit controls were completed before any neural fit |
| 4 | F+B | k = 2, a 0.05 floor and a 0.5 ratio (provisional). The policy is complete: original probabilities, no renormalization, deterministic ties, the refusal check and the fixture |
| 5 | Predictions | Primary correctness is exact program, entity and answer, with a new certification predicate. P1–P5 are repaired, sanity checks are kept, and certification feasibility is checked in development |
| 6 | Sampled reproduction | One init per world and regime by hash; A and F refitted; the 16 / 60 / 12 fit inventory |
| 7 | Confirmation worlds | 1000–1002 reserved, subject to the logged inventory check ("History"). Development stays on 100–101 |

## Decisions (from the review of revision 2)

| # | Decision | As specified above |
|---|---|---|
| 1 | The pilot | The bounded 16-fit engineering pilot (worlds 100–101 × inits 100–101 × full and reduced exposure × A and F). A+K and F+B are evaluated from those fits, with L, G and W retained. The lexical extension stays at specification and no-fit controls |
| 2 | F's sampling | Per extracted segment presentation, multiplicities preserved, with realized presentations and training cost reported. Equal update counts are not described as equal compute |
| 3 | A+K | It abstains when the grammar refuses, and those refusals are reported separately. It is described as a grammar-admitted constrained policy sharing A's model and entity pointer |
| — | Before the pilot | The verifier rebuilds the support report, study identity and vocabulary size; checks configuration, sources and run-level totals; and binds units to the requested inventory. There are rejection tests, the single L/G/W risk test, A's separate historical certificate with sanity gating, and the corrected feasibility table |

## Decisions for the reviewer (revision 3)

| # | Decision | Recommendation |
|---|---|---|
| 1 | Whether this language merits a confirmation at all. F and F+B were at the ceiling and matched L, G and W exactly, and F+B never differed from F | Not as a learning result. A confirmation would replicate an engineering equivalence. The informative finding is A+K against A |
| 2 | Whether a lexical condition with forms W cannot ground should be specified | Only with a stated identifiability rule beyond word association and its own no-fit standard. Otherwise the main learning question moves on |

## History

### Revision 1 (2026-10-24, reviewed at `30b0df6`)

The review kept the direction and found a missing baseline. An exact-segment lookup, induced only from the
grammar's alignment of training pairs, was perfect on development worlds 100–101. Its decisions:
- **systems:** add L;
- **supervision:** describe it precisely, and keep ill-typed labels;
- **regimes:** rename scarce as reduced exposure, and record that it keeps the same distinct grounding inputs;
- **F+B:** make the policy complete, with the refusal check and the fixture;
- **predictions:** repair the correctness predicate and P1–P5, and check certification feasibility;
- **reproduction:** the inventory of fits;
- **worlds:** reserve 1000–1002 after a logged check;
- **implementation:** A+K must constrain the recurrence itself, and verification must cover semantics.

**Revision 1's wording** that called premature stopping a diagnosed cause is replaced by "observed failure modes".

### Development check of the segment rule (2026-10-24; no training)

The segment rule segmented every supported and held-out question of development world 100: 19,212 questions in
all, training included. The only exceptions were the 163 unknown-entity refusals, which are refused. Revision 2's
ordered audit strengthens this check.

### The no-fit controls (2026-10-25; development worlds 100–101; no training)

`bench/factored_interpreter.py --worlds 100 101` was verified by `bench/factored_interpreter_verify.py`, a separate
implementation that re-derived every label, dictionary, record, summary and audit.

**The ordered alignment audit.** Every item with a program had each segment, in execution order, as a declared
phrasing of the relation at that position: 18,905 items in world 100 and 18,892 in world 101. The only other cases
were the 191 essential-augmentation replacements per world, by design. Nothing was misaligned.

**Decisions over 17,900 evaluated items per world and regime:**

| World, regime | L (raw) | L (vocab) | G |
|---|---|---|---|
| 100 full | 0 wrong decisions, 0 errors | 0, 0 | 0, 0 |
| 100 reduced | 0, 0 | 0, 0 | 0, 0 |
| 101 full | 0, 0 | 0, 0 | 0, 0 |
| 101 reduced | 0, 0 | 0, 0 | 0, 0 |

S2's 800 and S1's 1,100 items were answered with exact programs in every case, and there was no accidental agreement.

**Support and exposure:**

| | World 100, full | World 100, reduced | World 101, full | World 101, reduced |
|---|---:|---:|---:|---:|
| Labelled segment presentations | 2,007 | 1,657 | 2,000 | 1,661 |
| Contributing questions | 1,485 | 1,135 | 1,467 | 1,128 |
| Relation forms (4 per relation) | 20 | 20 | 20 | 20 |
| Taught presentations per phrase | 28–107 | 3 | 30–105 | 3 |
| Unique keys, raw | 116 | 116 | 111 | 111 |
| Unique keys, vocab | 32 | 32 | 33 | 33 |
| Conflicting keys | 0 | 0 | 0 | 0 |
| Excluded questions | 209 | 209 | 216 | 216 |

**What the unique keys are:**
- **raw:** most are one-off pseudo-word segments from essential augmentation (87–93 per world);
- **vocab:** 20 relation forms, one `<ambiguous>` form ("the main city of") and 11–12 `<unsupported>` keys. Of
  those, 9 are `<unk>` patterns and 2–3 are real unsupported phrases ("the founder of", "the population of",
  "the mayor of").

**What was excluded:**
- the 46 unknown-entity training refusals;
- refusal targets over several segments: 98–103 essential augmentations, 41–42 unsupported refusals and 24–25
  ambiguous ones.

**Evaluation coverage, raw keys:**
- every ID, T1, S2, S3 and S1 segment was in L's dictionary;
- no held-out segment was.

**Evaluation coverage, vocab keys:** of the S4a primary abstentions in world 100, 1,631 were refused by an
`<unsupported>` `<unk>` pattern and 369 by an unseen one.

**P3a, checked as a construction property:** every S3 item's segments equalled its S3_base item's (1,000 of 1,000
groups per world), and every S3_L3 item's equalled its L3_base item's (300 of 300).

### Revision 2 (2026-10-25, reviewed at `fbef2fd`)

The review authorized the bounded engineering pilot after a set of repairs:
- **verifier coverage** of the support report, study identity, vocabulary size, configuration, sources, run-level
  totals and the requested inventory, with new rejection tests;
- **a single calibration risk test** for the deterministic policies;
- **A's historical certificate** reconstructed separately, with sanity gating for P1;
- **the corrected feasibility table;**
- **W,** and the complete lexical specification.

All are above. It kept the lexical extension at specification and no-fit controls.

### The no-fit controls with W and the lexical condition (2026-10-25; development worlds 100–101)

The run was `bench/factored_interpreter.py --worlds 100 101`, from a clean clone of `0e1a3c9`.
`bench/factored_interpreter_verify.py` accepted it in full: 4 units.
- **The standard benchmark.** L (raw and vocab), G and W made no wrong decision, no error and no accidental
  agreement on 17,900 items per world and regime. Each passed its single calibration risk test: 760 (world 100) or
  767 (world 101) answered calibration items, with 0 errors.
- **The lexical condition:** the results under "The lexical-generalization condition" above.

### The bounded engineering pilot (2026-10-25; development worlds 100–101 × inits 100–101 × both regimes)

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 python research/ti_tnn/bench/factored_pilot.py \
        --worlds 100 101 --inits 100 101 --out <dir>

**The run.** It used a clean clone of `0e1a3c9` and made 16 fits (A and F), in 27–30 s per unit, 4 minutes in all.
`bench/factored_pilot_verify.py` accepted all 8 units against the declared design and the producer-source hashes.
- **A disclosure.** That commit's no-fit test still expected three controls, because W had just been added. The
  assertion was corrected in the next commit. Tests are not among the hashed producer sources.

**Results (4 units per regime; certified policies; primary exact-program correctness):**

| System | Certified (threshold) | Sanity | S2 criterion | S1 (non-cancelling) criterion | Joint novelty | S2 exact accuracy | S1 exact accuracy |
|---|---|---|---|---|---|---|---|
| A, full | 4/4 (0) | 4/4 | 0/4 | 0/4 | 4/4 | 0.809–0.975 | 0.000–0.233 |
| A, reduced | 4/4 (0) | 4/4 | 0/4 | 0/4 | 4/4 | 0.771–0.954 | 0.006–0.471 |
| A+K, full | 4/4 (0.2) | 4/4 | 3/4 | 0/4 | 4/4 | 0.976–1.000 | 0.611–0.985 |
| A+K, reduced | 4/4 (0.2) | 4/4 | 4/4 | 0/4 | 4/4 | 0.964–1.000 | 0.470–0.823 |
| F, both | 8/8 (0) | 8/8 | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 |
| F+B, both | 8/8 (0) | 8/8 | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 |
| L (raw, vocab), G, W, both | 8/8 (risk test) | 8/8 | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 |

**Exact accuracies** are unthresholded: abstentions count as failures. Every system had ID coverage 1.000 with 0
errors, R abstention 1.000, and S4a and S4b primary abstention 1.000.

**P1, A's replication** (historical answer-based certificate, the robustness code unchanged). In all 8 units A passed
the historical sanity checks and T1, and its S2 and S1 criteria failed. So the composition failure replicates in
every development unit, gated on sanity. The world-level form of P1 needs 5 inits and is not evaluable on 2.

**P5.** F's (and F+B's) unthresholded S2 exact accuracy minus G's was 0.000 in every unit. So was L's and W's.

**F+B against F.**
- **No difference:** in all 8 units F+B made exactly F's decisions.
- **Second candidates were rare:** at most 26 records per unit had one, above the 0.05 floor.
- **Its abstentions:**
  - refusal argmax `<unsupported>`, 6,391–6,419 per unit;
  - refusal argmax `<ambiguous>`, 124–137;
  - no well-typed candidate (the ill-typed refusals), 109–135;
  - the grammar's unknown entity, 108–117.

**A+K against A.** The constraints recovered most of A's S2 failures and much of its S1 failure. They did not reach
the S1 criterion in any unit. A+K's grammar refusals (unknown entity, 108–117 per unit) are reported separately from
its own `<unsupported>` and `<ambiguous>` abstentions.

**Certification feasibility.** At the strictest rule (0.999), every scored system accepted 760–767 calibration items
with 0 errors, well above the 299 floor. A, F and F+B passed every rule down to 0. A+K stopped at 0, so it certified
0.2.

**Realized cost per unit:**

| | A | F |
|---|---|---|
| Updates × batch | 1,500 × 64 | 1,500 × 64 |
| Presentations drawn | 96,000 questions | 96,000 segments, drawn from 1,657–2,007 aligned presentations |
| Tokens processed | 742,457–768,389 | 263,769–275,158 |
| Forward MACs over training presentations | 86.3–87.4 G | 11.4–11.9 G |
| Parameters | 223,817–223,865 | 71,775–71,823 |
| Training wall time | 15.3–17.5 s | 4.2–4.7 s |

**Equal update counts are not equal compute.** F's training used about 7× fewer forward MACs and about 3.5× less wall
time.

**Evaluation over 17,900 items:**
- **A:** 1.1–1.4 s, with 17,900 encoder passes and 71,600 decoder steps;
- **A+K:** 1.2–1.6 s, with 17,783–17,792 encoder passes and 37,492–37,973 decoder steps;
- **F and F+B together:** 0.66–0.72 s, with 30,151–30,156 segment passes per unit. F+B adds 11,238–11,290
  enumerated candidates, of which 109–161 are pruned;
- **L, G and W:** 0.17–0.32 s.

The module applications for answered programs were identical across the systems that answered the same programs:
20,015 in world 100 and 19,988 in world 101.

### The logged inventory check of worlds 1000–1002 (2026-10-25 04:19 UTC)

- **The repository:**
  - no tracked file uses 1000, 1001 or 1002 as a world seed, apart from this draft;
  - no bundled evidence lists them.
- **Local run directories.** The run roots of this worktree and of 12 other local checkouts held 548 JSON files,
  and none lists those worlds.
- **Directory names:** two matched "100[012]". Both are false positives: a timestamp (`…T131002`) and a data ID
  (`data-91000`).

## Closing record (2026-10-25; nothing above was changed)

**The decision.** The review of revision 3 closed this phase as a development engineering result. The 60-fit
confirmation was not run, and worlds 1000–1002 remain untouched.

**Corrections made in closing.** The recorded pilot and controls runs stay bound to `0e1a3c9`: the verifiers check
their recorded sources against that commit's hashes, from `git show`.
- **The pilot verifier** now also compares alpha and delta in every certificate, A's historical eligible calibration
  records, and both historical category tables in full. Each has a new rejection test.
- **The accounting** separates emitted decisions from evaluated decoder row-steps, labels A's training MACs as
  unpadded or padded, and keeps F's and F+B's duration as one combined measurement, with its boundary stated.
- **A+K's selection** uses the allowed logits in corrected code, with scores still from the original probabilities,
  and has an underflow fixture. The recorded pilot used the earlier, probability-based rule. On its records this
  changes no choice: none of its recorded steps tied at the maximum allowed probability.

**The export's integrity.** All 58 original export files are complete: every gzip file decompressed in full, every
JSON document parsed, and every records file holds 17,900 items. Their sha256 are in
`experiments/results/ti_tnn/factored_interpreter/export_integrity.json`.

**The A+K diagnosis** was made from the saved records only, with no new fit. It is in the report and in `ak_diagnosis.json`.

**For a later proposal.** A lexical task is not begun merely to defeat W. A later proposal must start from a useful
missing capability, specify what makes its semantics learnable, and include the cheapest relevant baseline.
