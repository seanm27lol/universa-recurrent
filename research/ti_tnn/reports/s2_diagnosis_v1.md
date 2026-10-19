# S2 diagnosis: taught wording yields one-relation programs in the augmented word interpreter; the specified wrapper augmentation did not beat repetition — 2026-10-19

**Briefs:**
- [development brief](../protocols/s2_diagnosis_brief.md), committed at `d991d12` before any intervention fit;
- [confirmation protocol](../protocols/s2_context_confirmation_v1.md), frozen at `07bb97e` before any
  confirmation fit.

**Raw results:** [`experiments/results/ti_tnn/s2_diagnosis/`](../../../experiments/results/ti_tnn/s2_diagnosis/)
(see "Evidence").

**Scope.** This diagnoses the S2 failure of the [interpreter robustness study](interpreter_robustness_v1.md).
It changes none of that study's protocol, records or verdicts. There are three parts, kept separate
throughout:
1. **Historical diagnosis:** a descriptive analysis of the existing confirmation records, worlds 700–702.
2. **Development:** instrumented fits and training-data interventions, worlds 100–101. These are not
   confirmatory.
3. **Fresh confirmation:** arms A, B and C on worlds 710–712. It ran because the frozen development rule was
   met.

**S2**, as in the robustness study, is a two-step question in which one Base phrase is replaced by a synonym
taught only in one-step questions. Its matched **S2_base** question has the same program, entity and frame, in
Base phrasing.

**Execution order.** Program position 0 is the relation applied first, to the entity. In a prefix question
that is the innermost phrase. "Taught first" means the taught relation is applied first.

## A concrete example

These are real questions from the robustness confirmation (world 702, init 2, `word_aug`). In training,
"the airfield of" appeared only in one-step questions, of the form "what is the airfield of <a city> ?".

| Question | Reference program | Proposed program | Outcome |
|---|---|---|---|
| "name the company operating the airport of country_29_city_0 ." (S2_base) | `airport_of`, then `operator_of` | `airport_of`, then `operator_of` | right: the operator |
| "name the company operating the airfield of country_29_city_0 ." (S2) | `airport_of`, then `operator_of` | `airport_of`, then stop | wrong: the airport, not its operator |

The model reads the taught phrase correctly: "airfield of" becomes `airport_of`. It then ends the program,
although "the company operating" still asks for another relation.

**Known analogue.** Lake and Baroni's SCAN "add jump" split is the closest known case
([Lake & Baroni, 2018](https://arxiv.org/abs/1711.00350)): a primitive seen only in isolation during training
is not composed by sequence-to-sequence models.

**The candidate explanation.** It is a **shortcut** in the sense of
[Geirhos et al., 2020](https://arxiv.org/abs/2004.07780): a decision rule that fits the training distribution
but fails to transfer. Here the rule would be "taught wording means a one-relation program", because that is
the only context in which training showed the wording. This diagnosis measures the behaviour that rule would
produce; it does not establish the rule as the mechanism.

## The five answers

1. **What goes wrong in S2.** For the augmented word interpreter, mostly **one relation where two are required**,
   then refusals.
   - **Historical records.** `word_aug` parsed 10,927 of 12,000 S2 questions exactly. Of the other 1,073:

     | What the program did | Count |
     |---|---:|
     | kept only the relation applied first | 434 |
     | kept only the relation applied second | 224 |
     | refused (`<unsupported>` 161, `<ambiguous>` 96) | 257 |
     | substituted one relation | 121 |
     | reordered | 34 |
     | other | 3 |

     - **Entities:** no S2 error, in any arm or part of this diagnosis, involved the entity.
     - **S2_base:** exact in every `word_aug` unit.
     - **Pairs:** in no arm was S2 right while its S2_base pair was wrong.
     - **Other arms:** the same kinds of failure occur, in different proportions (counts below). `char_aug`'s
       failures are mostly `<unsupported>` refusals (1,839 of 2,336), not one-relation programs (174).
2. **Premature stopping, where another relation is required: yes,** in arms A, B and C, in both world sets.
   - **The measurement.** With the correct first relation supplied (teacher forcing), we compare the
     probability of `<end>` on S2 and on S2_base.
   - **The result.** The mean probability of `<end>` is 0.048–0.093 on S2 and at most 3×10⁻⁵ on S2_base. It
     exceeds 0.5 for 4–9% of S2 questions, and for no S2_base question.
   - **D.** D, trained on taught two-step questions, shows none of this.
   - **Its share of failures.** Greedy decoding stops after the first relation in:
     - 322 of 447 failures (arm A, development);
     - 390–525 of 1,317–1,502 failures per arm (confirmation, fresh worlds).

     The other large failure kind in fresh worlds is skipping the relation applied first.
3. **Does longer context help beyond exposure? The specified wrapper augmentation did not.**
   - **Development.** C (taught one-step questions in longer real-word wrappers) beat B (the same questions
     repeated) in both development worlds. That met the rule for a confirmation.
   - **Fresh worlds.** Both frozen predictions failed:
     - C passed S2 and S3/S4 together in 0, 0 and 3 of 5 units per world;
     - C's mean S2 exact accuracy exceeded B's in only one of three worlds.
   - **Exposure alone** did not consistently help either. B was below A in both development worlds, and
     above A in two of three fresh worlds.
   - **Scope.** C added randomly assembled filler, made of familiar frame words, around the original one-step
     question. This experiment does not determine whether other ways of changing input length or context would
     help. The frozen design set no minimum detectable effect or equivalence margin. C minus B in mean S2 exact
     accuracy was −5.450, −2.675 and +3.725 percentage points in worlds 710, 711 and 712.
4. **Does any improvement preserve S3/S4?**
   - **No intervention broke novelty handling in fresh worlds.** A, B and C met the S3/S4 joint criterion in
     all 15 confirmation units.
   - **In development,** B lost one unit (S4b abstention 0.946).
   - **The only clear S2 improvement was D,** which trains on taught two-step questions. It was exact on S2 in
     all six development units, with S3/S4 intact. That is additional compositional supervision, not zero-shot
     transfer, and D was not part of the confirmation.
5. **What explanation is supported, and what remains open.**
   - **What the evidence supports.** The failure is specific to taught wording inside composition:
     - T1 and S2_base are essentially perfect;
     - the typical failure is a one-relation program;
     - stop probability rises after the first relation;
     - training on taught two-step questions removes it.

     This is consistent with a learned association between taught wording and one-relation programs, the
     only context in which training showed that wording.
   - **What it does not show.** It does not establish that association as the cause: the evidence is
     behavioural. The specified wrapper augmentation did not remove the failure.
   - **Open questions:**
     - the large variation between inits (S2 exact 0.58–0.99 for A in fresh worlds);
     - why the relation applied first is sometimes skipped;
     - whether anything short of taught two-step supervision gives zero-shot transfer.

     A next intervention should be judged on recovering the correct ordered program, not on making outputs
     longer. In fresh worlds, A's one-relation outputs kept the second relation (537) more often than the first
     (413), so some failures omit a required relation before stopping becomes the issue.

   S2 remains a valid transfer test under the declared training design. An association like this would
   explain the failure; it would not invalidate the measurement.

## Part 1: historical diagnosis (worlds 700–702, descriptive)

`bench/s2_diagnosis.py` covered 15 units × 4 arms × 800 S2 questions, plus the grammar control.
- **Verification.** It ran the robustness verifier first; the verifier passed, and its output matched the
  bundled `verification_output.txt` exactly.
- **Pairing and position.** It regenerated each study, paired every S2 question with its S2_base question, and
  recovered the taught relation's execution-order position by exact re-rendering.
- **Classification.** It classified every proposed program.
- **Cross-check.** Its S2 and S2_base totals were checked against the verifier's independently derived
  metrics.

**S2 questions by class.** The cell format is n (answered correctly / answered wrongly / not answered). A
complete program that the runtime refused (ill-typed) keeps its program class and counts as not answered.

| Class | `word` | `word_aug` | `char` | `char_aug` | grammar |
|---|---:|---:|---:|---:|---:|
| exact | 10,314 | 10,927 | 9,633 | 9,664 | 12,000 |
| dropped, the taught relation | 572 (0/151/421) | 336 (0/157/179) | 554 (0/340/214) | 113 (0/113/0) | 0 |
| dropped, the other relation | 314 (0/219/95) | 322 (0/277/45) | 282 (0/63/219) | 61 (0/13/48) | 0 |
| substituted, the taught relation | 188 (8/81/99) | 109 (5/52/52) | 193 (31/156/6) | 126 (25/101/0) | 0 |
| substituted, the other relation | 0 | 12 (4/3/5) | 15 (0/7/8) | 0 | 0 |
| reordered | 26 | 34 | 424 | 69 | 0 |
| other | 4 | 3 | 191 (0/27/164) | 97 | 0 |
| refused: `<ambiguous>` | 86 | 96 | 181 | 31 | 0 |
| refused: `<unsupported>` | 496 | 161 | 527 | 1,839 | 0 |

Each column totals 12,000.

**Which relation survives.** In one-relation programs, the relation kept is the one applied first, or the one
applied second:

| Arm | Kept the relation applied first | Kept the relation applied second |
|---|---:|---:|
| `word` | 370 | 516 |
| `word_aug` | 434 | 224 |
| `char` | 403 | 433 |
| `char_aug` | 126 | 48 |

**Pairs.**
- **Base right, S2 not:** 1,672 (`word`), 1,064 (`word_aug`), 2,336 (`char`), 2,311 (`char_aug`).
- **The reverse:** 0 in every arm.
- **Wrong program, right answer:** 8, 9, 31 and 25. A wrong program still landed on the right answer, for
  example when a country's capital is also its largest city.

**`word_aug` by taught position:**

| Class | Taught first (n = 6,135) | Taught second (n = 5,865) |
|---|---:|---:|
| exact | 5,618 | 5,309 |
| dropped, the taught relation | 179 | 157 |
| dropped, the other relation | 277 | 45 |
| substituted, the taught relation | 31 | 78 |
| substituted, the other relation | 7 | 5 |
| reordered | 9 | 25 |
| refused | 14 | 243 |
| other | 0 | 3 |

**Largest single cells for `word_aug`:**

| Relation pair | Taught position | Failure | Count |
|---|---|---|---:|
| `airport_of > operator_of` | taught first | dropped `operator_of`: answered with the airport | 171 |
| `country_of > capital_of` | taught first | dropped `country_of`: `capital_of` applied to a city, refused as ill-typed | 81 |
| `country_of > capital_of` | taught second | refused (`<unsupported>`) | 107 |
| `country_of > largest_city_of` | taught second | refused (`<ambiguous>`) | 95 |

The `<ambiguous>` cell is consistent with "the most populous city of the country of …" being read as close to
the trained ambiguous phrase "the main city of …". That reading is an interpretation, not tested here.

**`word_aug` S2 per unit:**

| Unit | 700/0 | 700/1 | 700/2 | 700/3 | 700/4 | 701/0 | 701/1 | 701/2 | 701/3 | 701/4 | 702/0 | 702/1 | 702/2 | 702/3 | 702/4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 exact | 0.741 | 0.998 | 0.970 | 0.899 | 0.924 | 0.838 | 0.966 | 1.000 | 0.887 | 0.990 | 0.833 | 0.941 | 0.870 | 0.964 | 0.839 |

S2_base was exact (1.000) in every unit. The failure is concentrated in some inits rather than spread evenly.

Strata by relation pair, taught relation, question length, world, init and frame are in the bundled
`s2_diagnosis.json`. So are 5 deterministic examples per class and arm: the smallest sha256 of
(arm, world, init, id).

## Part 2: instrumentation (arm A, development worlds 100–101 × inits 100–102)

**Arm A** is the robustness study's `word_aug` recipe, unchanged.
- **Results:**

| Unit | 100/100 | 100/101 | 100/102 | 101/100 | 101/101 | 101/102 |
|---|---:|---:|---:|---:|---:|---:|
| S2 exact | 0.964 | 0.975 | 0.748 | 0.969 | 0.809 | 0.978 |

  S2_base, T1 and ID were 1.000 throughout, and the S3/S4 joint criterion held in all six units.
- **Traces.** For every S2, S2_base, T1 and T1_base question, the decoder's full output distribution was
  stored at full precision:
  - under greedy decoding (the decisions the policy used); these reproduce every recorded decision bit for
    bit;
  - under teacher forcing with the reference outputs.

**The decisive comparison: P(`<end>`) at step 1.** That is the probability of stopping right after the first
required relation, when a second is still required. The correct prefix is supplied.

| Taught position | n | S2 mean | S2 median | S2 share > 0.5 | S2_base mean | S2_base share > 0.5 | S2 higher than its pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4,800 | 0.069 | 5.3×10⁻⁵ | 6.6% | 2.5×10⁻⁵ | 0 | 73% |
| taught first | 2,523 | 0.080 | 3.8×10⁻⁵ | 8.0% | 2.5×10⁻⁵ | 0 | 70% |
| taught second | 2,277 | 0.057 | 6.6×10⁻⁵ | 5.0% | 2.4×10⁻⁵ | 0 | 76% |

Most S2 questions continue normally. A minority stop with high probability, and no S2_base question does.

**Greedy failure kinds** (program only). Let t be the first step where greedy output departs from the
reference, and E the set of steps where the teacher-forced choice is wrong (t is always in E):

| Kind | Rule |
|---|---|
| premature stop | E = {t}, and the output stops where a relation is required |
| relation confusion | E = {t}, a wrong relation at t, and the rest of the output is right |
| error propagation | E = {t}, a wrong relation at t, and later departures, although the correct prefix would have continued correctly |
| broader | wrong at two or more steps even with the correct prefix |

| Taught position | Right | Premature stop | Relation confusion | Error propagation | `<unsupported>` | Broader |
|---|---:|---:|---:|---:|---:|---:|
| all | 4,353 | **322** | 41 | 26 | 8 | 50 |
| taught first | 2,304 | 202 | 1 | 16 | 0 | 0 |
| taught second | 2,049 | 120 | 40 | 10 | 8 | 50 |

In development, premature stopping was the main failure (322 of 447), in both taught positions. This is a
behavioural pattern, not an established mechanism.

## Part 3: training-data interventions (development; not confirmatory)

All four arms share the six development units, 1,500 updates of 64, the architecture, the novelty
augmentation and the vocabulary. B, C and D each add N_w examples: 365 in world 100 and 354 in world 101, the
number of taught one-step training questions.

| Arm | Addition |
|---|---|
| A | none |
| B | every taught one-step training question again, verbatim (exposure) |
| C | the same questions inside longer wrappers of Base-frame words (`what`, `is`, `name`), with lengths drawn from the training data's Base-phrase lengths. A C question is as long as one with one more Base phrase (9–13 tokens). Targets stay one relation |
| D | taught two-step questions built from training instances, with both relations as targets (additional compositional supervision) |

Every addition was audited before training:
- training instances only;
- at most two relations per target;
- no held-out or novel word;
- no evaluation question;
- matched counts.

**Per-unit results.** S2 exact accuracy, S2 certified coverage / errors, and the S3/S4 joint criterion.
S2_base and T1 were 1.000 in every unit.

| Arm | 100/100 | 100/101 | 100/102 | 101/100 | 101/101 | 101/102 | S2 criterion | S3/S4 joint |
|---|---|---|---|---|---|---|---:|---:|
| A | 0.964 · 0.981/11 | 0.975 · 1.000/20 | 0.748 · 0.909/129 | 0.969 · 0.986/14 | 0.809 · 1.000/153 | 0.978 · 0.999/17 | 0/6 | 6/6 |
| B | 0.960 · 1.000/32 | 0.899 · 0.964/52 | 0.765 · 0.946/145 | 0.994 · 1.000/5 | 0.833 · 0.979/117 | 0.863 · 1.000/110 | 0/6 | 5/6 |
| C | 0.828 · 0.906/63 | 0.904 · 0.953/39 | 0.956 · 0.956/0 | 0.953 · 1.000/38 | 0.979 · 1.000/17 | 0.953 · 1.000/38 | 1/6 | 6/6 |
| D | 1.000 · 1.000/0 | 1.000 · 1.000/0 | 1.000 · 1.000/0 | 1.000 · 1.000/0 | 1.000 · 1.000/0 | 1.000 · 1.000/0 | 6/6 | 6/6 |

B's joint failure is unit 100/100 (S4b abstention 0.946).

**The frozen decision rule was met.** C's mean S2 exact accuracy was above B's in both worlds, and C met the
joint criterion in all six units.

| World | A | B | C | D |
|---|---:|---:|---:|---:|
| 100 | 0.8954 | 0.8746 | **0.8958** | 1.0000 |
| 101 | 0.9183 | 0.8963 | **0.9613** | 1.0000 |

**Teacher-forced stopping and failures in development:**

| Arm | P(`<end>`) at step 1, S2 mean | S2 share > 0.5 | Greedy failures | of which premature stops | `<unsupported>` refusals |
|---|---:|---:|---:|---:|---:|
| A | 0.069 | 6.6% | 447 | 322 | 57 |
| B | 0.093 | 9.2% | 550 | 440 | 81 |
| C | 0.048 | 4.1% | 343 | 195 | 142 |
| D | 0.000 | 0 | 0 | 0 | 0 |

In development, C lowered stopping and raised `<unsupported>` refusals; most of the refusals came when the
taught relation was second. D removed both stopping and failures.

## Part 4: fresh confirmation (worlds 710–712 × inits 200–204; arms A, B, C)

The protocol was frozen at `07bb97e`; the run recorded that commit and an unmodified research tree.
- **Fits:** 45, all completed. Every metric was re-derived from the decision records, and every greedy trace
  reproduced its decisions.
- **Every unit:** S2_base and T1 were 1.000.
- **Certified threshold:** 0, except B 711/203 (0.9).
- **The rule:** a unit passes when it meets the S2 criterion **and** the S3/S4 joint criterion; an arm holds
  in a world at ≥ 4 of 5 units.

| Arm | Units passing S2 and S3/S4 together (710/711/712) | Holds | S2 criterion alone | S3/S4 joint alone | Mean S2 exact (710/711/712) |
|---|---|---|---|---|---|
| A | 2/2/1 | no | 2/2/1 | 5/5/5 | 0.9197 / 0.8592 / 0.8455 |
| B | 2/0/1 | no | 2/0/1 | 5/5/5 | 0.9123 / 0.9045 / 0.8540 |
| C | 0/0/3 | **no** | 0/0/3 | 5/5/5 | 0.8578 / 0.8778 / 0.8912 |

- **Primary prediction (C holds): not met.**
- **Secondary prediction (C's mean S2 exact above B's in every world): not met.** C was below B in worlds 710
  and 711, and above it in 712.

S2 exact per unit ranged from 0.583 to 0.994 for A, 0.719 to 0.993 for B and 0.726 to 1.000 for C.

**Teacher-forced stopping and failures in fresh worlds:**

| Arm | P(`<end>`) at step 1, S2 mean (S2_base) | S2 share > 0.5 | Greedy failures | Premature stop | Relation confusion | Error propagation | `<unsupported>`/`<ambiguous>` | Broader |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.055 (2.3×10⁻⁵) | 5.2% | 1,502 | 413 | 170 | 316 | 347 | 256 |
| B | 0.053 (3.0×10⁻⁵) | 4.8% | 1,317 | 525 | 396 | 82 | 253 | 61 |
| C | 0.052 (2.6×10⁻⁵) | 5.0% | 1,493 | 390 | 263 | 106 | 300 | 434 |

- **Stopping replicates; C's reduction does not.** Elevated stopping after the first relation appears in all
  three arms, but C's development reduction in stopping is gone.
- **Another failure kind in fresh worlds.** A large share of failures start with a wrong first relation:
  - mostly when the taught relation is applied first, where A has 285 error-propagation cases;
  - the model emits the relation applied second, skipping the taught one, and then stops;
  - with the correct first relation supplied, it would have continued.
- **One-relation programs:** these account for 950 of A's 1,502 failures (413 keeping the relation applied
  first, 537 keeping the relation applied second).

## Interpretation limits

- **The historical part is descriptive.** It analyses a completed experiment; nothing in it was predicted
  before the records were read.
- **Development results are not confirmation.** There were two worlds × three inits, and every init of a world
  scores the same 800 S2 draws, so inits are not independent samples of questions. The C-over-B difference
  that triggered the confirmation was within the large init-to-init variation, and it did not replicate.
- **What the padding test can show.** C leaves "taught wording → one-relation target" intact; it tests whether
  input length and surrounding context contribute. The specified wrapper augmentation did not consistently
  outperform the repetition control or satisfy the frozen competence criteria. This experiment does not
  determine whether other ways of changing input length or context would help. It is not evidence about the
  shortcut itself.
- **What D shows.** D trains on the distribution S2 tests (on other instances). Its development success shows
  that the architecture and recipe can learn taught wording inside two-step programs when shown it. It is not
  zero-shot transfer, and it was not confirmed on fresh worlds.
- **Behaviour, not mechanism.** Stop probabilities and failure kinds describe what the decoder does under
  teacher forcing and greedy decoding. Teacher forcing means feeding the reference output as the previous
  output ([Williams & Zipser, 1989](https://doi.org/10.1162/neco.1989.1.2.270)); it is used here only as a
  diagnostic, never for decisions. No representation was intervened on.
- **The setting.** A controlled language, table modules and the study's word interpreter.

## Evidence

The bundle is in [`experiments/results/ti_tnn/s2_diagnosis/`](../../../experiments/results/ti_tnn/s2_diagnosis/)
(82 files, about 26 MB). Every file is listed in `manifest.json` with its sha256.

| Path | What it holds |
|---|---|
| `historical/s2_diagnosis.json` | the classification, strata, per-unit metrics and examples, from the bundled robustness records |
| `development/20261019T082758/` | arm A's summary, its six decision-record files and its six trace files |
| `development/20261019T083141/` | the B, C and D summary and 18 decision-record files |
| `development_analysis/s2_development_analysis.json` | the development analysis |
| `confirmation/20261019T084110/` | the confirmation summary and 45 decision-record files |
| `confirmation_analysis/s2_development_analysis.json` | the confirmation analysis, with the frozen verdict |
| `local_traces.json` | sha256 of the 63 trace files that were not bundled, for size: B–D in development, all confirmation traces |
| `trace_release.json` | where those 63 files are published (a GitHub release asset per file) and how to rebuild from them |

**The trace release.** The 63 trace files not bundled (85.2 MiB) are published as assets of the release
[`ti-tnn-s2-diagnosis-traces-v1`](https://github.com/seanm27lol/universa-recurrent/releases/tag/ti-tnn-s2-diagnosis-traces-v1).
Each asset is a byte-identical copy of its run's file. Its sha256 is the one committed in `local_traces.json`
with the original results, so a downloaded set is checked against hashes recorded before the release existed.

**Rebuilding.** `python experiments/results/ti_tnn/make_tables.py` re-derives everything from the bundled
records and stops on any difference from the bundled analyses (`s2_diagnosis_tables`):
- every class count;
- every unit metric, certificate and outcome;
- the decision rule;
- the confirmation verdict;
- arm A's development trace tables.

Every trace-derived table for every arm needs the trace release:

```bash
gh release download ti-tnn-s2-diagnosis-traces-v1 --repo seanm27lol/universa-recurrent --dir s2_traces
python experiments/results/ti_tnn/make_tables.py --s2-traces s2_traces
```

That covers teacher-forced stopping and the greedy failure kinds that separate premature stopping, relation
confusion and error propagation. It first checks every downloaded file against the committed hashes. Each trace
is then checked against its decision records as it loads.

**What the verification checks** (`bench/s2_development_analysis.py`), for every fit:
- **Certificate:** the complete certificate is re-derived from the decision records: threshold, every step's
  counts and p-value, the stop reason, alpha and delta.
- **Training additions:** each world's additions are rebuilt with the frozen construction and checked against
  the recorded counts, sha256, audit and length distributions.
- **Traces:** every trace present is compared with the decision records, independently of the run's own
  mismatch list:
  - the greedy chain must give the recorded program and refusal reason;
  - its chosen probabilities times the entity probability must give the recorded score exactly.
- **Source binding:**
  - the development rule requires each development run's recorded commit (`ece0f86` or `d991d12`) and the
    frozen producer-file hashes;
  - the confirmation verdict requires `07bb97e` and the same hashes.

`tests/test_ti_tnn_s2_diagnosis.py` alters a certificate's p-value and stop reason, a trace's greedy choice and
entity probability, a recorded addition hash, and the confirmation's commit and source hash. Each change is
rejected; the version merged in #54 accepted all of them.

**Reproduction.** All runs are CPU only: `CUDA_VISIBLE_DEVICES=`, 4 threads, `nice -n 10`.

| Run | Fits | Source |
|---|---:|---|
| `runs/titnn_s2_diagnosis/development/20261019T082758` (arm A) | 6 | `ece0f86` |
| `runs/titnn_s2_diagnosis/development/20261019T083141` (B, C, D) | 18 | `d991d12` |
| `runs/titnn_s2_diagnosis/confirmation/20261019T084110` | 45 | `07bb97e` |

- **Clean starts.** Each run recorded its commit and a clean research tree, and every fit completed.
- **Fit time:** 16–19 s.
- **Regenerated outputs.** The three analyses were regenerated from the committed code and are byte-identical
  to the bundled files.
- **Smoke tests.** Two harness runs on a synthetic world checked the code; neither is a result.
  - **8 countries:** it stopped because D needed more two-step training instances than the world had.
  - **16 countries, 20 updates:** it completed.
- **Total fits.** 69 counted fits: 24 development and 45 confirmation, the budget's maximum.

## Corrections (2026-10-19, after an outside review of #54)

No number in this report changed. The review recomputed the frozen verdict from the bundled summaries and found
it exact. It also found that every bundled A trace agreed with its decision records.

| What changed | Why |
|---|---|
| **The failed intervention is scoped.** The title, answer 3, answer 5 and the padding paragraph now name the specified wrapper augmentation. One sentence was replaced: it had called the result "evidence against a length or context contribution of a size this design could detect". The replacement says the augmentation did not consistently outperform repetition or meet the frozen criteria, and that the experiment does not determine whether other ways of changing length or context would help. | The frozen design set no minimum detectable effect, power analysis or equivalence margin. C was one construction: random familiar filler words. |
| **The one-relation diagnosis is scoped to the augmented word interpreter.** The title and answer 1 change, and answer 1 now states `char_aug`'s refusal-dominated failures. | `char_aug` failed mostly by refusing (1,839 of 2,336), not with one-relation programs (174). |
| **The public evidence now covers every trace-derived table.** The trace release and `--s2-traces` were added. | Before, only arm A's development trace tables could be rebuilt from public evidence. That left out the fresh-world stopping table that supports "stopping replicates". |
| **The verification gained three checks:** complete certificates, independent trace checks and source binding (see "What the verification checks"). | The analysis compared only the certificate threshold, relied on the run's own trace-mismatch list, and did not bind the confirmation to its frozen source. No discrepancy was found in the supplied evidence; the strengthened analysis reproduces both bundled analyses byte for byte. |
| **One open question was added:** judge interventions on the correct ordered program. | In fresh worlds, A's one-relation outputs more often kept the second relation than the first. |

