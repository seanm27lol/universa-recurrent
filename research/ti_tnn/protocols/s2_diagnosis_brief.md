# S2 diagnosis: development brief for the training-data interventions

**Status: committed 2026-10-19 (UTC) before any intervention fit (arms B, C, D).**
- **What has already run.** Two things, both disclosed below:
  - the record analysis of the interpreter robustness confirmation (part 1);
  - arm A's six instrumented development fits (part 2), at `ece0f86`.
- **The code for the remaining fits** is the commit that adds this brief:
  - `bench/s2_development.py` (unchanged since `ece0f86`: arm construction, addition audit, fits, traces);
  - `bench/s2_development_analysis.py` (re-derivation, traces, decision rule);
  - `bench/s2_diagnosis.py`.
  The study's generator, audit, decisions, certification and bars are imported unchanged from
  `bench/interpreter_robustness.py` (frozen at `69b7fd7`).
- **Development only.** Nothing here revises the interpreter robustness protocol, its records or its verdicts.
  Worlds 700–702 are historical evidence for this diagnosis, not a fresh confirmation.

## The question

The confirmation ([report](../reports/interpreter_robustness_v1.md)) found that no arm met the S2 criterion.
S2 is a two-step question in which one Base phrase is replaced by a synonym taught only in one-step questions.
In the same questions with Base phrasing (S2_base), every arm parsed essentially every program exactly.

This diagnosis asks three things:
- what goes wrong in S2;
- whether the model stops after the first relation when a second is required;
- whether the training data's shape contributes: taught wording appeared only in short, one-relation questions.

## Part 1: what the records show (historical, descriptive)

`bench/s2_diagnosis.py` covers the 15 confirmation units × 4 arms × 800 S2 questions:
- **Verification.** It runs the robustness verifier first. The verifier passed, and its output is identical
  to the bundled `verification_output.txt`.
- **Pairing.** It pairs each S2 question with its S2_base question.
- **Taught position.** It recovers the taught relation's execution-order position by exact re-rendering.
- **Classification.** It classifies every proposed program.

Word_aug, over 12,000 S2 questions:

| Class | Count | How the runtime handled them |
|---|---:|---|
| exact | 10,927 | all answered correctly |
| dropped, the taught relation | 336 | 157 answered wrongly, 179 refused (ill-typed) |
| dropped, the other relation | 322 | 277 answered wrongly |
| substituted | 121 | |
| reordered | 34 | |
| refused | 257 | `<unsupported>` 161, `<ambiguous>` 96 |
| other | 3 | |

- **S2_base exact:** 1.000 in all 15 units.
- **Base right, S2 wrong:** 1,064 pairs.
- **The reverse:** never.
- **Unit dependence:** large. S2 exact runs from 0.741 (700/0) to 1.000 (701/2).

The full tables are in the report.

## Part 2: arm A, instrumented (development worlds 100–101 × inits 100–102)

Arm A is the robustness study's `word_aug` recipe, unchanged.
- **S2 exact:** 0.964, 0.975 and 0.748 (world 100); 0.969, 0.809 and 0.978 (world 101).
- **Other categories:** S2_base 1.000, T1 1.000, and the S3/S4 joint criterion met in all 6 units.
- **Teacher-forced decoding** supplies the reference prefix at every step. At step 1, after the first
  required relation, while a second is still required:

| | S2 | S2_base |
|---|---:|---:|
| mean P(`<end>`) | 0.069 | 2.5×10⁻⁵ |
| P(`<end>`) > 0.5 | 6.6% of questions | none |

  S2 is higher than its pair in 73% of pairs. The median S2 value is 5×10⁻⁵: most S2 questions continue, and
  a minority stop with high probability.
- **Greedy failures.** 447 S2 programs were wrong:
  - premature stops (the only teacher-forced error is `<end>` where a relation is required): 322, with
    202 when the taught relation comes first and 120 when it comes second;
  - errors at two or more steps: 50;
  - relation confusions: 41;
  - error propagation: 26;
  - `<unsupported>`: 8.

This is a behavioural pattern, not an established causal mechanism.

## Part 3: the interventions (to run now)

### Arms

Four arms on the same six development units (worlds 100–101 × inits 100–102, 96 countries). A's fits are
reused, so there are 18 new fits and 24 in all.
- **A (original):** the `word_aug` training set: ordinary training questions plus novelty augmentation.
- **B (exposure):** A plus every taught one-step training question again, verbatim.
- **C (context beyond exposure):** A plus the same taught one-step questions as B (same instances, relations
  and frames), each inside a longer inessential wrapper:
  - **filler words:** k words drawn uniformly, independently, from the Base frames' words (`what`, `is`,
    `name`). All are trained wrapper words; no held-out or novel S3/S4 word is used.
  - **filler length k:** drawn from the empirical token-length distribution of the Base prefix phrases in the
    world's prefix-style training questions. That is frozen as 3 words for 463 phrases and 4 words for 151
    (world 100), and 3 words for 446 and 4 words for 128 (world 101). A C question is then as long as the same
    question with one more Base phrase:
    - world 100: 9–13 tokens (124/153/63/24/1);
    - world 101: 9–13 tokens (100/161/72/20/1).
  - **filler position:** after the frame's opening words or before its closing punctuation, half each. The
    noun phrase is untouched.
  - **target:** still the one relation.

  The wrappers are ungrammatical sequences of real words. C tests whether input length and surrounding
  context contribute; it is not a paraphrase.
- **D (additional compositional supervision):** A plus N_w two-step training instances, sampled without
  replacement:
  - **rendering:** prefix style, with one uniformly chosen position's Base phrase replaced by its taught
    phrase. Taught first: 179/170 (worlds 100/101); taught second: 186/184.
  - **target:** both relations.

  D trains on exactly the kind of question S2 tests, on other instances. A gain from D is the effect of
  extra compositional supervision, **not** zero-shot S2 transfer.

### What is matched and fixed

- **Added counts:** N_w equals the number of taught one-step training questions: 365 (world 100) and 354
  (world 101), the same for B, C and D.
- **Training:**
  - the same 1,500 updates of 64 examples, so a larger training set means fewer passes over each example;
  - the same architecture, learning rate (3e-3), entity dropout (0.1) and initialisation seeds;
  - the same novelty augmentation and word vocabulary (from the ordinary training questions; every added
    word is already in it).
- **Additions:**
  - built from training instances only;
  - targets of at most two relations;
  - none is an evaluation question.
  `audit_additions` checks this before training. The additions' sha256 values are recorded in each run.
- **Evaluation and decoding:** the study's evaluation sets, greedy decoding, certification and bars.

### Measurements

Every fit reports, from its records (re-derived with the robustness verifier's code):
- **S2:**
  - exact accuracy over all 800 questions (program and entity);
  - answer accuracy;
  - certified coverage and errors;
  - the S2 criterion (certified coverage ≥ 0.80 with a 95% error bound ≤ 1%).
- **Other categories:** S2_base, T1, ID coverage and refusals (sanity).
- **The S3/S4 joint criterion:** certified S3 coverage ≥ 0.80 with an error bound ≤ 1%, and primary S4a and
  S4b abstention ≥ 0.95.
- **Diagnostics:** the classes of part 1, the teacher-forced step probabilities, and the greedy failure kinds.

### Comparisons

| Comparison | What it isolates |
|---|---|
| B against A | exposure |
| C against B | length and context beyond exposure |
| D against the others | additional compositional supervision |

### The decision rule (frozen)

**Proceed to a fresh confirmation of A, B and C only if both hold:**
1. **S2 accuracy.** In **each** development world, C's mean S2 exact accuracy over its three inits is
   strictly higher than B's.
2. **Novelty.** C meets the S3/S4 joint criterion in **all six** development units.

Otherwise:
- report these as development findings;
- do not construct further variants.

D's result is reported separately; it cannot trigger the confirmation.

### The confirmation, if the rule is met (fixed now, run only after its own freeze commit)

- **Units:** worlds **710, 711, 712**, 96 countries; inits **200–204**; arms **A, B, C**; 45 fits.
  - No TI-TNN run or protocol has used worlds 710–712; this was checked against every run configuration under
    `runs/` and every protocol.
- **Design and bars:** the robustness study's frozen IID design and bars.
- **The per-unit criterion:** a unit passes when the same unit meets **both** the S2 criterion and the S3/S4
  joint criterion.
- **Aggregation:** an arm holds in a world when at least 4 of its 5 units pass, and holds when every world
  does.
- **Prediction (one-sided):** C holds.
- **Reporting:** A and B are reported with the same rule whatever the outcome.
- **Freeze:** code, sampling, counts, hypotheses and analysis are committed before the first confirmation fit.
- **If time is short,** the protocol is committed and left unrun.

## Interpretation limits

- **What C tests.** Padding a taught one-step question leaves "taught wording → one-relation target" intact.
  C tests whether input length or surrounding context contributes to the failure. It does not establish that
  shortcut as the cause.
- **S2 remains valid.** S2 is a valid transfer test under the declared training design. A shortcut would
  explain the failure; it would not invalidate the measurement.
- **Statistical weight.** Two worlds × three inits is development evidence, not confirmation. The same 800 S2
  draws are scored by every init of a world, so inits are not independent samples of questions. Unit-to-unit
  variation in S2 is large (part 1).
- **Negative findings.** A well-supported negative finding is a successful outcome.

## Execution record (appended 2026-10-19, after the intervention fits)

- **The intervention fits:** run `20261019T083141`.
  - Ran from `d991d12` with an unmodified research tree.
  - 18 fits, all completed.
  - Every greedy trace reproduced its recorded decisions.
  - Every metric was re-derived from the decision records.
- **The decision rule was met:**

| World | C mean S2 exact | B mean S2 exact |
|---|---:|---:|
| 100 | 0.8958 | 0.8746 |
| 101 | 0.9613 | 0.8963 |

  C also met the S3/S4 joint criterion in all six units.
- **Reported separately:**
  - D: S2 exact 1.000 in all six units, with the S2 criterion and the joint criterion met in each.
  - B: lost the joint criterion in unit 100/100 (S4b abstention 0.946).
- **What followed:** the fresh confirmation was frozen at `07bb97e` and run
  ([protocol](s2_context_confirmation_v1.md), [report](../reports/s2_diagnosis_v1.md)).
- **Deviations from this brief:** none.
