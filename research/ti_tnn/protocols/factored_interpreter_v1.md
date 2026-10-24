# A factored interpreter: learned lexical grounding under a supplied typed grammar (draft, revision 1)

**Status: DRAFT for review, revision 1 (2026-10-24).** Nothing is implemented, frozen or run. No world
1000–1002 has been touched, and no exploratory probe was run for this draft.

**Where this comes from.** This is item 2 of the order recorded in the
[integration plan](../integration_plan.md) after the review of #61: the next main learning study. It responds to
three failures:
- **S2:** taught wording did not transfer into two-step questions
  ([robustness](../reports/interpreter_robustness_v1.md));
- **S1:** non-cancelling three-step composition failed;
- **the diagnosed cause:** premature stopping and dropped relations ([S2 diagnosis](../reports/s2_diagnosis_v1.md)).

## A concrete example

In training, "the airfield of" appeared only in one-step questions ("what is the airfield of
country_29_city_0 ?"). The question below is from the robustness confirmation (world 702, init 2):

> name the company operating the airfield of country_29_city_0 .

**The current interpreter** (`word_aug`, a sequence-to-sequence decoder) proposed `airport_of`, then stopped. It
read "the airfield of" correctly, and then answered with the airport instead of its operator.

**A factored interpreter splits the job in two:**
1. **Structure is supplied, not learned.**
   - In prefix questions, every relation phrase begins with "the" and contains no other "the". So the noun
     phrase splits into the segments "the company operating" and "the airfield of", ending at the entity.
   - The innermost segment applies first, so the program has two slots: slot 1, then slot 2. This rule uses no
     lexicon.
2. **Meaning is learned, per segment.** A small grounding model maps each segment's words to a relation,
   `<unsupported>` or `<ambiguous>`:
   - "the airfield of" → `airport_of`. Training showed this segment, in one-step questions;
   - "the company operating" → `operator_of`.

**Composition.** The program `airport_of`, then `operator_of`, is type-checked from the entity and executed.

**What can still go wrong** is grounding. "the aerodrome near" (held-out wording, never taught) must be
refused. It must not be guessed from the entity's type, even when only one relation would fit.

## The question

**When program structure is supplied by a lexicon-free typed grammar and only lexical grounding is learned,
does taught wording transfer into composition while the robustness study's novelty behaviour is kept? And what
does the learned grounding contribute, compared with knowing the lexicon?**

1. **Transfer:** S2 exact-program transfer, and S1 non-cancelling three-step composition.
2. **Preservation:** `word_aug`'s joint novelty behaviour on S3, S4a and S4b, by S4 stratum.
3. **The learned grounding's contribution.** The factored system and the grammar control share the same
   structural rule. They differ only in whether the lexicon is learned from (question, target) pairs or given.
   The gap between them is the cost of learning it.
4. **Bounded alternatives.** Do they add coverage without answering ungrounded meaning? Their total cost is
   counted, rejected alternatives included.

**What the supplied grammar settles.** With it, S2 and S1 succeed whenever every segment is grounded right;
the grammar control already shows that. **This study measures the conditions under which learned grounding is
right,** not whether composition works once grounding is.

## Systems (one fitted interpreter per arm and unit; fixed table modules; the same execution contract)

| System | What it is | Role |
|---|---|---|
| **A: current greedy** | `word_aug` as in the robustness study: the same encoder, decoder, augmentation, training and greedy decoding | The baseline, and a replication of its S2 and S1 failures |
| **A+K: constrained decoding** | A's fitted model, decoded with the supplied grammar's constraints. Exactly as many relations as segments, then `<end>`; every prefix must be well typed from the copied entity; `<unsupported>` and `<ambiguous>` stay allowed at every step | The control for "factoring helps", as opposed to "constraining helps" |
| **F: factored, greedy** | The supplied grammar, plus a learned segment-grounding model. The program is each segment's argmax label in execution order | The main system |
| **F+B: factored, bounded alternatives** | F's fitted model, with up to k labels per segment, type pruning, execution and a selection rule (below) | Does a bounded search add coverage safely? |
| **G: grammar control** | The same structural rule, with the Base and taught lexicon given | The standard, by construction; not a finding |

**Shared by every system:**
- the maximum program length (3, as `routing.MAX_STEPS`);
- the entity binding (the world's entity domains);
- the type checks and execution (`runtime.execute` on table modules);
- the refusals (`<unsupported>`, `<ambiguous>`, unknown entity, ill-typed).

**The supplied grammar**, used by A+K, F, F+B and G, is written down in full:
- **Entity binding:** the entity is the one token naming an entity of the world. With none, the question is an
  unknown entity; with several, it is malformed.
- **Prefix style:** the entity is preceded by segments, each a maximal span that begins with "the" and contains
  no other "the". The noun phrase begins at the first such "the"; words before it are the frame, and are
  ignored.
- **Possessive style:** the entity is followed by segments, each beginning with "'s" and ending before the next
  "'s" or the final punctuation. Words before the entity are the frame.
- **Malformed questions:** mixed styles, no segment, or more than 3 segments.
- **Execution order:** innermost first for prefix questions, left to right for possessive ones.
- **What it never does:** use the lexicon. Base, taught and held-out wording all fit this rule, so held-out
  phrases are segmented, and grounding must refuse them.

**The grounding model** (F and F+B):
- **Input:** a segment's words, with unseen words as `<unk>` (as in the word arms), and the style.
- **Encoder:** a small encoder over the segment only, context-free by design.
- **Output:** a softmax over the five relations, `<unsupported>` and `<ambiguous>`.
- **Supervision** (decision 2): segment labels come from the same (question, target) pairs that train A.
  - **Programs:** the grammar aligns a program's relations to its segments in execution order.
  - **Refusals:** a refusal target labels a segment only when the question has one segment.
  - **Augmentation:** the same novelty augmentation as `word_aug`. An essential pseudo-word phrase in a
    one-segment question gives `<unsupported>`; inessential frame and filler words fall outside the noun phrase.

**F+B's selection rule** (the whole rule is certified as one policy):
- **Refusal decides first.** If any segment's top label is `<unsupported>` or `<ambiguous>`, the system
  abstains. Alternatives cannot supply a meaning that was never grounded.
- **Candidate programs.** Every combination of each segment's top-k relation labels (k = 2) with probability at
  least 0.05: at most 8 programs.
- **Pruning and execution.** Programs that are ill-typed from the entity are pruned. The rest are executed, and
  every proposed, pruned and executed program is counted.
- **The answer.** The highest-scoring survivor is answered.
- **Preserved ambiguity.** If another survivor scores at least half of it and gives a different answer, the
  system abstains.
- **No survivor:** the system abstains.

## Training regimes (decision 3)

| Regime | Taught phrases in training | Why |
|---|---|---|
| **full** | As in the robustness study: the taught phrasing of every one-step training instance | Comparability. F is expected near the grammar control here |
| **scarce** | Each taught phrase in exactly 3 one-step training questions; everything else unchanged | Grounding must work from few examples. Scores spread, so risk and coverage, and the alternatives, become informative |

A non-zero threshold is not the goal in either regime. **The question is whether each system's certified
release policy lowers accepted error while keeping coverage,** against the same system answering whenever it
proposes a program.

## Populations and categories

The populations and categories are the robustness study's, with its declared distributions, independent draws
and audit:
- **the ID distribution:** calibration 1,000 draws, then the ID and R test draws;
- **T1, S2, S3;**
- **S4a and S4b:** the primary samples (answer-ambiguous) and the population sample, with all three strata;
- **S1:** non-cancelling chains and round trips;
- **the length-3 diagnostics.**

There are 96 countries per world.

**Every category reports:**
- exact ordered-program accuracy as well as answer accuracy, because different programs can agree by accident;
- right answers from wrong programs.

## Measures

- **Per system, unit, policy and category:**
  - coverage, accepted errors and the 95% upper error bound;
  - abstention, with its exact interval;
  - exact-program accuracy;
  - the reasons for abstaining.
- **By S4 stratum:**
  - **answer-ambiguous;**
  - **one answer;**
  - **type-determined.** Here a system that answers by type has not shown that it grounded the phrase, and the
    contract requires abstention.
- **Scores.** Three scores are compared as reported diagnostics, each by program length:
  - the probability product, the current score;
  - the minimum segment or step probability;
  - the geometric mean.

  For each, risk and coverage are reported on ID and on S2 and S1 separately. None is assumed better.
- **Total cost per question:**
  - interpreter forward passes;
  - programs proposed, pruned and executed;
  - module applications;
  - wall time.

  Training cost is reported separately.
- **The admission detector.** Where a system's refusals act as a distribution-admission check (unknown words,
  `<unsupported>`), they are evaluated for false admissions (S4 answered) and false refusals (S3, ID and S2
  refused).

## Certification

**Each system's complete release policy is certified on its own calibration draws.** That covers:
- greedy or constrained decoding, or grounding;
- for F+B, the alternatives, pruning, selection and ambiguity rule;
- the threshold on its own score.

It uses the robustness study's fixed-sequence rule: grid 0.999 … 0, accepted error 1%, δ = 0.05. **A certificate
for A's greedy policy does not carry over to A+K, F or F+B.**

**What the guarantee covers:** the ID distribution of one world and fitted model. It does not cover S2, S1 or
S4 (other distributions), and it is not simultaneous across systems or units.

## Units and procedure

- **Worlds:** confirmation **1000, 1001, 1002**, used by no earlier run. Development uses worlds 100–101.
- **Inits:** 0–4 in confirmation, 100–101 in development.
- **Unit:** a (world, init, regime). Every system in a unit shares that unit's study, training items and
  evaluation draws. A and A+K share A's fit; F and F+B share F's fit.

## Predictions (one-sided; bars to be set with the development pilot before freezing)

A world holds when ≥ 4 of 5 inits succeed jointly. A prediction holds when every world holds.

**Criteria:** "answers where it should" is a certificate, coverage ≥ 0.80 and a 95% error bound ≤ 1% on
exact-program-correct answers. "Abstains where it should" is abstention ≥ 0.95.

| # | Prediction | Regime | Kind |
|---|---|---|---|
| P1 | A fails S2 and S1, as in the robustness study | full | Replication |
| P2 | F answers where it should on S2 and on S1's non-cancelling chains | full | Empirical: grounding learned from one-step questions composes under the supplied structure |
| P3 | F meets the joint novelty criterion: S3 answers, S4a and S4b primary abstain | full | Empirical: preservation |
| P4 | F+B abstains ≥ 0.95 on S4a and S4b in every stratum, the type-determined one included | full and scarce | Empirical: alternatives do not answer ungrounded meaning |
| P5 | Where F and G differ, it is in grounding. On S2, F's exact-program accuracy is within 2 points of G's | full | Empirical: in the full regime, learning the lexicon costs little |

**Reported without a prediction:**
- **A+K against A,** on S2 and S1. Does constraining the decoder recover what factoring recovers?
- **The scarce regime for every system.** Here F's grounding is genuinely uncertain.
- **F+B against F:** coverage, risk and cost.
- **The three scores.**
- **Every S4 stratum.**
- **The admission detector's false admissions and refusals.**

## Export and independent verification

**The export.** Per item and system it holds:
- the decision, reason and score;
- the proposed program and the bound entity;
- the answer.

For F and F+B it also holds every segment's span and its full label distribution. So the verifier can
re-derive F's greedy program and F+B's entire selection (candidates, pruning, execution on the tables, the
ambiguity rule and costs) without the model.

**The verifier** (`bench/factored_interpreter_verify.py`) does not call the producer's model or selection code.
It:
- regenerates each study from its seed and re-audits it;
- re-implements the supplied grammar from this text and checks every segmentation;
- re-derives F+B from the exported distributions;
- re-derives correctness, every certificate, every metric and every verdict.

It enforces a frozen inventory and source binding, as in #58–#64.

**What stays the producer's record:**
- A's and A+K's decoder outputs, because they depend on the decoder state;
- F's label distributions;
- that the records came from the trained models.

**A sampled reproduction** (decision 6) would retrain a frozen, outcome-independent sample of units and require
identical records.

## Interpretation limits

- **Structure is supplied.** S2 and S1 success under a supplied grammar is not structure learning. It shows what
  learned grounding achieves when structure is given. This language was designed with a lexicon-free segment
  rule, and natural language has none.
- **Controlled language, table modules, synthetic worlds.**
- **The scarce regime is one declared exposure level,** not a learning curve.
- **The comparison.** If A+K matches F, the result says constraints suffice; if not, the comparison is between
  complete replacement systems. Either way it is not a claim that factoring is necessary.
- **The certificates are per world and model,** on the ID distribution only.

## Decisions for the reviewer (revision 1)

| # | Decision | Recommendation |
|---|---|---|
| 1 | The systems: A, A+K, F, F+B, with G as the standard | Yes. A+K is the control you asked for if the claim is that factoring helps |
| 2 | Grounding supervision: segment labels aligned from (question, target) pairs by the supplied grammar, with multi-segment refusals unused. The alternative is phrase labels from the generator's metadata, which is stronger supervision | The aligned labels, for comparability with A |
| 3 | Training regimes: full and scarce (3 one-step questions per taught phrase) | Yes. Without the scarce regime, F is expected near G by construction, and the study is a ceiling study |
| 4 | F+B's rule: k = 2, a 0.05 floor, a 0.5 ambiguity ratio, refusal before alternatives | Yes for the pilot; confirm or revise from the pilot |
| 5 | The predictions, and P5's 2-point bar | Draft bars, to be confirmed with the development pilot |
| 6 | Sampled reproduction: retrain one unit per world and regime, chosen by a hash fixed in advance | Yes |
| 7 | Confirmation worlds 1000–1002 × inits 0–4, development 100–101 × 100–101, 96 countries | Yes |

**Next steps, after review:**
1. Implement the grammar, grounding model, F+B, A+K, the export and the verifier, with corruption tests.
2. Run the development pilot. It checks segmentation coverage, the regimes' informativeness (not at ceiling or
   floor in the scarce regime), F+B's costs, and the runtime.
3. Fix the bars, then freeze.

## History

### Development check of the segment rule (2026-10-24; no training)

I applied the supplied grammar's segment rule, as written above, to every question the robustness generator
makes for development world 100 (96 countries, seed 100): 19,212 questions in all.
- **Every supported question segmented,** in every category: ID, T1, S2, S3 and S1, the length-3 diagnostics,
  and training.
- **Every held-out question segmented too:** S4a, S4b, the primaries and S4_L3. So held-out phrases reach the
  grounding model, as intended.
- **The segments matched the program:** where a program exists, the number of segments equalled its length in
  every case.
- **The only unsegmented questions** were the 163 unknown-entity refusals (CAL_R 51, R 112). Those are refused,
  as required.
- **The full regime's taught exposure** is 365 one-step training questions, about 73 per taught phrase. So the
  scarce regime's 3 per phrase is a real reduction.

No model was trained, and no question of worlds 1000–1002 was generated.
