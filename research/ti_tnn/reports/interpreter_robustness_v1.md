# Interpreter robustness: augmented word interpreter handles unfamiliar wording; taught-synonym transfer and longer composition fail — 2026-10-19

**Protocol:** [`protocols/interpreter_robustness_v1.md`](../protocols/interpreter_robustness_v1.md), revision 4,
frozen at `69b7fd7` before any run on confirmation worlds 700–702.

**Raw results:**
[`experiments/results/ti_tnn/interpreter_robustness/`](../../../experiments/results/ti_tnn/interpreter_robustness/).
The bundle holds the aggregate file, the verdict, and all 15 per-unit record files (calibration and test
decisions for all four arms and the grammar control).

**Verification:** every number below was re-derived from those records by
`bench/interpreter_robustness_verify.py`. It regenerates each study from its seed and checks it against the
stored values; see "Evidence".

## The distinction being tested

These are real questions and decisions from confirmation world 700, init 0:

| Question | Correct behavior | `word` (no augmentation) | `word_aug` |
|---|---|---|---|
| "name the airport serving the biggest city in country_93 ." | answer: country_93_city_1_airport | answers correctly | answers correctly |
| "name the aerodrome near the biggest city in country_93 ." | refuse: "aerodrome near" was never taught | answers country_93_city_1: it drops the unknown phrase and runs `largest_city_of` alone | refuses |
| "name exactly the aerodrome near the biggest city in country_93 ." | refuse | answers country_93_city_1 | refuses |
| "could you tell me the largest city of the country containing country_13_city_2 ?" | answer: the frame is new, but inessential | answers correctly | answers correctly |

The unfamiliar words in the second and third questions carry the relation, so they must be refused. In
the fourth they are only a frame, so it must be answered.

That held-out question is **answer-ambiguous:** under the surrounding type constraints, two complete
programs fit the unknown phrase's position, and they give different answers. The question's types
alone therefore cannot reveal what was meant.

## Result

**The three frozen predictions passed. The separate S2 transfer and S1 structure criteria did not.**

- **Novelty, the lead hypothesis.** Under the frozen controlled-language protocol, the augmented word
  interpreter (`word_aug`) met the joint novelty criterion in all 15 confirmation units: all 5 inits in
  each of three fresh worlds. The same fitted model met every part of the criterion:
  - it answered questions with unfamiliar inessential wording (S3);
  - it refused questions whose essential wording was unfamiliar, with and without a novel wrapper (S4a
    and S4b, answer-ambiguous primary samples of 2,000 per unit).
- **S2: taught-synonym transfer into two-step questions failed.** Every arm learned the taught phrases on
  their own: T1 accuracy was 1.000 in every unit. But no arm met the S2 criterion in 4 of 5 inits in any
  world. `word_aug` met it in 3, 1 and 0 of 5. Every S2 error, in every arm, was a wrong relation program
  with the right entity.
- **S1: three-step composition failed.** Every arm failed the competence criterion on non-cancelling
  three-relation chains in every unit (0 of 15 for each arm). Some answers were right: `word_aug` answered
  6,617 of the 12,000 chains across 15 units, 2,051 of them correctly. But no unit came close to the
  required coverage and error bound.

## Frozen predictions and per-world successes

| | `word` | `word_aug` | `char` | `char_aug` |
|---|---|---|---|---|
| **1. Sanity** (certificate; ID coverage ≥ 0.90 with a 1% bound; R abstention ≥ 0.95) | 5/5/5 | 5/5/5 | 4/5/4 | 5/5/5 |
| **2. Taught wording learned** (T1 accuracy ≥ 0.95) | 5/5/5 | 5/5/5 | 5/5/5 | 5/5/5 |
| **3. Joint novelty** (lead: `word_aug`) | 0/0/0 | **5/5/5** | 0/0/0 | 1/1/1 |
| S2 transfer (separate verdict) | 1/2/1 | 3/1/0 | 0/1/1 | 1/3/1 |
| S1 structure (separate verdict) | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |

**How to read the table.**
- **Cells.** Each cell gives successes out of 5 inits in worlds 700/701/702. A world holds when at least 4
  succeed; a prediction holds when all three worlds hold.
- **Predictions.** Prediction 1 covers every arm and holds. Prediction 2 covers every arm and holds.
  Prediction 3 (`word_aug`) holds.
- **The other rows** are recorded verdicts, not predictions: joint novelty fails for `word`, `char` and
  `char_aug`, and S2 and S1 fail for every arm.
- **The two uncertified `char` units** (world 700, init 2; world 702, init 2) stay in the denominator, as
  failures. In both, `char` answered calibration refusal questions (9 and 5 of 244), all of them errors.
  At the first rule that gave 3 and 5 accepted errors among about 750, which cannot be certified. Without
  a certificate, those units answer nothing.

## Per-unit results for `word_aug`

All under the certified policy. Every unit certified at threshold 0 with zero calibration errors, so the
threshold filtered nothing (see "Interpretation limits").

| Unit | S3 coverage | S3 errors (95% bound) | S4a primary abstention [95% CI] | S4b primary abstention [95% CI] | S2 coverage, errors |
|---|---|---|---|---|---|
| 700/0 | 0.969 | 0 (0.31%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.807, 53 |
| 700/1 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.998, 0 |
| 700/2 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 0.9970 [0.9935, 0.9989] | 0.970, 0 |
| 700/3 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.899, 0 |
| 700/4 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 0.9995 [0.9972, 1] | 0.979, 44 |
| 701/0 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.890, 42 |
| 701/1 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.988, 17 |
| 701/2 | 1.000 | 0 (0.30%) | 0.9995 [0.9972, 1] | **0.9795 [0.9723, 0.9852]** | 1.000, 0 |
| 701/3 | 0.974 | 0 (0.31%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.965, 53 |
| 701/4 | 1.000 | 1 (0.47%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.995, 4 |
| 702/0 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.959, 101 |
| 702/1 | 1.000 | 1 (0.47%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.986, 36 |
| 702/2 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.975, 84 |
| 702/3 | 1.000 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.996, 26 |
| 702/4 | 0.999 | 0 (0.30%) | 1.0000 [0.9982, 1] | 1.0000 [0.9982, 1] | 0.875, 29 |

**Sample sizes.** S3 has 1,000 questions per unit, S4 primary 2,000 per variant, and S2 800.

**The margins are not marginal crossings:**
- the worst S3 coverage was 0.969 against the 0.80 bar;
- the largest S3 error bound was 0.47% against 1%;
- the worst S4b abstention was 1,959 of 2,000, with an exact interval of [0.972, 0.985] against the 0.95
  bar.

**Other `word_aug` results:**
- ID coverage was 1.000 with no errors, and R abstention was 1.000, in every unit.
- The matched Base references for S3, the primary S4 sample and S2 were answered in full, with no
  errors.
- On the full S4 population sample (1,000 per unit, every stratum), abstention was 0.996–1.000
  (answer-ambiguous), 1.000 (one answer) and 1.000 (type-determined) for S4a. For S4b it was 0.963–1.000,
  0.973–1.000 and 0.979–1.000.

**Certification.**
- **Accepted counts.** For `word_aug`, the first rule (0.999) accepted 755–764 calibration questions in 14
  units and 551 in the fifteenth (700/4), all with zero errors.
- **Selected thresholds.** Every `word_aug` and `word` unit selected threshold 0. One `char` unit selected
  0.6 and one `char_aug` unit 0.999.
- **Power.** At about 760 accepted, a fixed rule with a true error rate of 0.1% certifies with probability
  about 0.96, 0.25% with about 0.70, and 0.5% with about 0.27.

## The four arms and the grammar control (certified policy, ranges over the 15 units)

| | S3 coverage | S3 errors per unit | S4a primary abstention | S4b primary abstention | Joint novelty |
|---|---|---|---|---|---|
| `word` | 0.47–1.00 | 0–162 | 0.562–0.892 | 0.644–1.000 | 0 of 15 |
| `word_aug` | 0.97–1.00 | 0–1 | 0.9995–1.000 | 0.980–1.000 | **15 of 15** |
| `char` | 0–0.96 | 0–203 | 0.196–1.000 | 0.455–1.000 | 0 of 15 |
| `char_aug` | 0.46–1.00 | 0–92 | 0.532–1.000 | 0.757–1.000 | 3 of 15 |
| grammar control | 1.00 | 0 | 1.000 | 1.000 | (by construction) |

**What the comparison shows.**
- **Augmentation on word input** is the clearest difference. Without it the word model answers
  held-out-wording questions often, as in the example, where it drops the unknown phrase. With it, it
  refuses them and still answers novel frames.
- **Character input does worse.** Preserving spelling information did not produce better behavior under
  this refusal contract. Both character arms answered many held-out-wording questions, and `char_aug`
  succeeded jointly in only 3 of 15 units.
- **Where the zeros come from.** The `char` ranges include the two uncertified units, which answer nothing,
  hence coverage 0 and abstention 1.
- **The novelty-rejection control.** Refusing any unfamiliar word abstains on all of S4 and answers none
  of S3, in every arm, by construction.
- **The grammar control,** given the lexicon and one structural rule, handles every category, S2 and S1
  included, by construction. It is the standard the learned interpreters are compared with, not a
  finding.

**The other categories:**
- T1 accuracy was 1.000 for every arm in every unit.
- S1 non-cancelling chains, answered and correct across 15 units:

  | Arm | Answered | Correct |
  |---|---:|---:|
  | `word` | 6,841 | 2,601 |
  | `word_aug` | 6,617 | 2,051 |
  | `char` | 6,173 | 372 |
  | `char_aug` | 8,283 | 2,113 |

- The length-3 diagnostics (Base, taught, novel frame, held-out) show the same structural failure.

## Interpretation limits

- **The confidence threshold did not produce the novelty behavior.** Every `word_aug` unit selected
  threshold 0, so the certified policy behaved exactly like answering whenever the runtime accepts. The
  refusals came from the interpreter's own `<unsupported>` outputs and the runtime. The finding concerns
  the learned configuration's decisions, not confidence filtering.
- **The phrase families were fixed during development.** The confirmation used fresh worlds and inits,
  but the held-out and novel phrase families were chosen and examined in the two development pilots.
  "Unfamiliar" means unseen in training. The result is a replication over the declared wording
  distributions in fresh synthetic worlds. It does not test a new set of paraphrase families chosen after
  development.
- **Fifteen units are three worlds with five fits each,** not fifteen worlds. Each world's study, and so
  its evaluation questions, is shared across its inits and arms. The results are reported per unit and
  per world (5/5, 5/5, 5/5) and are not pooled into one confidence interval.
- **The certificate's scope.** Its 5% statement is for one world and fitted model and the ID
  distribution. The overall verdict does not give a simultaneous 95% guarantee over every model and
  category.
- **This is an interpreter result with table modules.** The relation modules are supplied tables; the
  learned part is the interpreter, with declared novelty supervision. The augmentation teaches the
  property under test using other words. A success shows that it can be taught this way, not that it
  emerges unaided.
- **Controlled language.** Nothing here concerns natural language.

## Development history (disclosed in the protocol)

- **Round 1** (48-country worlds 100–101) could certify nothing: the strictest rule held too few
  calibration questions. It showed character input answering held-out wording from entity types and
  spelling.
- **Round 2** (96 countries) made certification feasible. `word_aug` met the joint criteria in 3 of 4
  development units; the fourth reached 0.94 on both S4 variants.
- **Choices made after the pilots.** `word_aug` was named the lead after these pilots. The superseded
  prediction, with `char_aug` as the primary arm, is kept in the protocol.
- **Revision 4** came after review, and nothing was trained for it:
  - the independent sampler;
  - larger S2 and primary S4 samples;
  - the inventory-enforcing audit;
  - the verifier;
  - the explicit joint rule.

## Evidence

- **What the bundle holds.** `experiments/results/ti_tnn/interpreter_robustness/20261018T232057/`:
  - `interpreter_robustness.json`: configuration, environment, audits, collisions, and per-unit summaries;
  - `verdict.json`;
  - `records/w{world}_i{init}.json.gz`, 15 files, with full-precision decision records (decision, reason,
    score, proposed program, copied entity, produced answer) for calibration and test items, all four
    arms and the grammar control;
  - `verification_output.txt`, the verifier's per-unit tables.

  The manifest gives every file's sha256 and size. The whole package is about 16 MB.
- **Reconstruction:**

  ```
  python research/ti_tnn/bench/interpreter_robustness_verify.py experiments/results/ti_tnn/interpreter_robustness/20261018T232057 --confirmation
  ```

  It regenerates every study from its seed, re-audits it, and re-derives from the records:
  - correctness, novelty and acceptance under every policy;
  - the certificate;
  - every category metric, probability calibration and the grammar control;
  - the joint verdict with per-world counts.

  It rejects any discrepancy. Training time, training loss and execution provenance cannot be derived
  from decision records and are not covered. Hashes identify artifacts; they do not authenticate the
  training run.
- **Versions.** Frozen training and source: `69b7fd7`. The verifier's later corrections (frozen
  configuration, certificate metadata, per-world counts, grammar and calibration reconstruction) are in
  the commit that adds this report. The original run artifacts were not modified, and the corrected
  verifier passes on them.

## Execution

- **The run.** `python research/ti_tnn/bench/interpreter_robustness.py --worlds 700 701 702 --inits 0 1 2 3
  4`, from `69b7fd7` with a clean tree:
  - 60 fits (word arms about 17 s, character arms about 41 s), CPU only (4 threads, `nice`), no failures;
  - run 2026-10-19 03:20–03:54 UTC.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.

## What this suggests next

- **S2 is the concrete open problem.** Every model learned each taught phrase on its own (T1 = 1.000), yet
  failed to use it reliably in a two-step question. Every S2 error is a wrong program with the right
  entity. The saved proposed programs and error categories are the place to start. The next step is to
  establish whether the decoder drops, reorders or substitutes the taught relation, before choosing among
  representation, decoding, supervision or search.
- **The novelty success gives the deferred evolutionary search no objective.**
