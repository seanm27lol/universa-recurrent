# A factored interpreter, a development engineering result: learned grounding reaches lookup performance under supplied structure, and the language needs nothing beyond lookup — 2026-10-25

**Protocol:** [`protocols/factored_interpreter_v1.md`](../protocols/factored_interpreter_v1.md) (draft, revision 3). It was
never frozen, and the phase closed at the development stage, without a confirmation.

**Raw results:** [`experiments/results/ti_tnn/factored_interpreter/`](../../../experiments/results/ti_tnn/factored_interpreter/).
Both runs were recorded at `0e1a3c9`, with a clean tree: the bounded engineering pilot and its no-fit controls.

**Verification:**
- `bench/factored_interpreter_verify.py` (the no-fit stage) and `bench/factored_pilot_verify.py` (the pilot) each
  re-derive every record, certificate, metric, criterion and cost count of their run independently.
- The A+K diagnosis (`bench/factored_pilot_diagnose.py`) uses the saved records only.

**This is a development result.** Worlds 100–101 and inits 100–101 were also used to design the study. Nothing here
is a confirmation. Worlds 1000–1002, reserved for one, were never touched.

## The question, by example

In training, "the airfield of" appeared only in one-step questions. The robustness study's interpreter (`word_aug`, a
sequence-to-sequence decoder) failed on questions like:

> name the company operating the airfield of country_29_city_0 .

It proposed `airport_of`, then stopped.

**The factored interpreter** takes structure from a supplied grammar and learns only the lexicon:
- **the grammar** cuts the question, with no lexicon, into the segments "the airfield of" and "the company operating",
  and fixes their order;
- **a learned grounding model, F,** maps each segment alone to a relation, `<unsupported>` or `<ambiguous>`;
- **the program** is type-checked from the entity and executed.

**What it was compared with:**
- **A,** the robustness interpreter;
- **A+K,** a grammar-admitted constrained policy sharing A's model and entity pointer;
- **F+B,** F with bounded alternatives;
- **no-fit controls:**
  - **L,** an exact-segment lookup induced from the same aligned training pairs;
  - **G,** the reference lexicon;
  - **W,** a training-derived word association.

## What was found

1. **Exact lookup solves the original language.**
   - **L,** with no fitting, made no wrong decision and no error on all 17,900 calibration and test items per world.
     That held in both worlds, both exposure regimes and both key representations, and G and W did the same.
   - **Why:** every supported segment of every evaluation category occurs as an exact string in training. That is
     20 relation phrasings, all present even with 3 examples per taught phrase.
2. **F reaches lookup performance under the supplied structure.** In all 8 units, F was certified at threshold 0 with
   exact S2 and non-cancelling S1 accuracy 1.000, refused all of S4 and R, and matched G on S2 exactly (P5 = 0.000). On
   this language that is memorization of seen segment forms. It is not lexical generalization.
3. **F+B showed no decision benefit in this pilot.** It never decided differently from F. At most 26 questions per
   unit had a second candidate above its 0.05 floor.
4. **Word association solves the declared lexical recombinations.** The new phrases recombine a taught head with a Base
   connector, for example "the airfield serving".
   - **W** grounded every one: 400 of 400 one-step and 800 of 800 two-step questions, with exact programs.
   - **It still refused** every matched held-out control and all of S4.
   - **L and G** refuse the new phrases by construction.
5. **A's composition failure replicates in every development unit,** under the historical answer-based metric. It is
   counted only where A also passed its sanity checks and T1, so the failure is not a broken fit.
6. **Constraining A's own decoding (A+K) repairs two of A's errors, but not the rest.** Where only A+K was exact, A
   itself had stopped early or chosen a relation of the wrong source type. The saved-record diagnosis below shows
   what remains:
   - **Every remaining wrong program** chose between two relations with the same source type, which the
     constraints cannot separate.
   - **At all but 7 of those choices,** A itself preferred an output the constraint did not allow there: `<end>`, or a
     relation of the wrong type. The choice between the two allowed relations was made among small probabilities.
   - **On three-relation questions,** the same holds at the weakest step of most correct programs. So their scores
     are low, and the certified threshold removes most answers, correct and wrong alike.

## The engineering results (4 units per regime; certified policies; exact-program correctness)

| System | Certified (threshold) | Sanity | S2 criterion | S1 criterion | Joint novelty | S2 exact | S1 exact |
|---|---|---|---|---|---|---|---|
| A, full | 4/4 (0) | 4/4 | 0/4 | 0/4 | 4/4 | 0.809–0.975 | 0.000–0.233 |
| A, reduced | 4/4 (0) | 4/4 | 0/4 | 0/4 | 4/4 | 0.771–0.954 | 0.006–0.471 |
| A+K, full | 4/4 (0.2) | 4/4 | 3/4 | 0/4 | 4/4 | 0.976–1.000 | 0.611–0.985 |
| A+K, reduced | 4/4 (0.2) | 4/4 | 4/4 | 0/4 | 4/4 | 0.964–1.000 | 0.470–0.823 |
| F, F+B (both regimes) | 8/8 (0) | 8/8 | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 |
| L (raw, vocab), G, W | 8/8 (risk test) | 8/8 | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 |

**How to read the table.**
- **Exact accuracies** are unthresholded: abstentions count as failures.
- **S1** means the non-cancelling chains.
- **Every system** had ID coverage 1.000 with 0 errors, R abstention 1.000, and S4a and S4b primary abstention 1.000.
- **Certification was feasible for every system.** At the strictest rule each accepted 760–767 calibration items with
  0 errors. Fewer than 299 could not have certified even with zero errors.

## The A+K diagnosis (saved records only)

Each S2 and non-cancelling S1 question falls in exactly one class:
- **released** means at or above A+K's certified threshold (0.2), and **filtered** means below it;
- under the `none` policy, everything proposed is released.

No threshold was chosen from these results.

**S2 (800 questions per unit):**

| Unit | Grammar refusal | Entity failure | Decoder refusal | Wrong, released | Wrong, filtered | Correct, released | Correct, filtered | None: coverage, errors (bound) | Certified: coverage, errors (bound) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| w100 i100 full | 0 | 0 | 0 | 5 | 0 | 795 | 0 | 1.000, 5 (0.0131) | 1.000, 5 (0.0131) |
| w100 i100 reduced | 0 | 0 | 0 | 0 | 0 | 703 | 97 | 1.000, 0 (0.0037) | 0.879, 0 (0.0043) |
| w100 i101 full | 0 | 0 | 0 | 0 | 0 | 780 | 20 | 1.000, 0 (0.0037) | 0.975, 0 (0.0038) |
| w100 i101 reduced | 0 | 0 | 0 | 2 | 0 | 696 | 102 | 1.000, 2 (0.0078) | 0.873, 2 (0.0090) |
| w101 i100 full | 0 | 0 | 11 | 0 | 0 | 776 | 13 | 0.986, 0 (0.0038) | 0.970, 0 (0.0039) |
| w101 i100 reduced | 0 | 0 | 0 | 0 | 29 | 764 | 7 | 1.000, 29 (0.0491) | 0.955, 0 (0.0039) |
| w101 i101 full | 0 | 0 | 0 | 1 | 18 | 650 | 131 | 1.000, 19 (0.0347) | 0.814, 1 (0.0073) |
| w101 i101 reduced | 0 | 0 | 2 | 0 | 4 | 654 | 140 | 0.998, 4 (0.0114) | 0.818, 0 (0.0046) |

**Non-cancelling S1 (800 per unit):**

| Unit | Grammar refusal | Entity failure | Decoder refusal | Wrong, released | Wrong, filtered | Correct, released | Correct, filtered | None: coverage, errors (bound) | Certified: coverage, errors (bound) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| w100 i100 full | 0 | 0 | 0 | 0 | 12 | 181 | 607 | 1.000, 12 (0.0242) | 0.226, 0 (0.0164) |
| w100 i100 reduced | 0 | 0 | 0 | 0 | 424 | 181 | 195 | 1.000, 424 (0.5595) | 0.226, 0 (0.0164) |
| w100 i101 full | 0 | 0 | 0 | 0 | 145 | 161 | 494 | 1.000, 145 (0.2051) | 0.201, 0 (0.0184) |
| w100 i101 reduced | 0 | 0 | 160 | 0 | 0 | 408 | 232 | 0.800, 0 (0.0047) | 0.510, 0 (0.0073) |
| w101 i100 full | 0 | 0 | 0 | 14 | 297 | 0 | 489 | 1.000, 311 (0.4180) | 0.018, 14 (1.0000) |
| w101 i100 reduced | 0 | 0 | 76 | 0 | 182 | 20 | 522 | 0.905, 182 (0.2793) | 0.025, 0 (0.1391) |
| w101 i101 full | 0 | 0 | 0 | 0 | 82 | 207 | 511 | 1.000, 82 (0.1219) | 0.259, 0 (0.0144) |
| w101 i101 reduced | 0 | 0 | 0 | 0 | 142 | 254 | 404 | 1.000, 142 (0.2012) | 0.318, 0 (0.0117) |

**Where the wrong programs go wrong** (all 8 units, by first differing position):

| | First difference | Reference → selected | Count | What A itself preferred there |
|---|---|---|---:|---|
| S2 (59) | position 1 | `airport_of` → `country_of` | 30 | `<end>` 29; the selected relation 1 |
| | position 1 | `country_of` → `airport_of` | 25 | `<end>` 23; the selected relation 2 |
| | position 1 | `capital_of` → `largest_city_of` | 4 | the selected relation 4 |
| S1 (1,298) | position 2 | `airport_of` → `country_of` | 797 | `<end>` |
| | position 1 | `largest_city_of` → `capital_of` | 308 | a relation the constraint forbids |
| | position 1 | `capital_of` → `largest_city_of` | 131 | a relation the constraint forbids |
| | position 0 | `largest_city_of` → `capital_of` | 54 | a relation the constraint forbids |
| | position 1 | `airport_of` → `country_of` | 8 | `<end>` |

**Every remaining confusion is between two relations with the same source type,** which the type constraints cannot
prune:
- `airport_of` and `country_of` both start from a City;
- `capital_of` and `largest_city_of` both start from a Country.

**"What A itself preferred"** is A's own top output at that step, read from the full distribution each record keeps.
Where it is not the selected relation, the constraint did not allow it there.

**The front end never failed.** No S2 or S1 question was refused by the grammar, and A's entity pointer was always
right.

**The probabilities.**
- **Non-cancelling S1, wrong programs.** At the first difference, the selected and the reference relation both had
  median probability near 0: from 0.000 to 0.022 by unit. A's own preference there held almost all the mass:
  - **`<end>`,** for all 805 `airport_of`/`country_of` confusions. Its unit medians were 1.000 in the four units
    holding 792 of them, and 0.693 for the other 13.
  - **A relation the constraint forbids,** for all 493 `capital_of`/`largest_city_of` confusions. Those relations'
    total had unit medians of 0.956–0.997 in the three units holding 492 of them, and 0.527 for the last one.
  - **The final `<end>`,** which completes the program, kept a median probability of 0.998–1.000, so it cost the
    score little.
  - **The context:** A was trained on programs of at most two relations, and these are three-relation questions.
- **Correct S1 programs scored low in the same way.** At the weakest step of 3,633 of the 4,866 correct programs, A
  itself preferred `<end>` (2,817) or a forbidden relation (816). The weakest step is the one whose chosen output had
  the lowest probability.
- **The consequence on S1:** wrong programs scored near 0 and were filtered, but so were many correct ones
  (correct-program scores have medians of 0.000–0.845 by unit). Certified coverage was 0.018–0.510, and wrong programs
  were released in one unit only (14).
- **S2, wrong programs.**
  - **In world 100 they were genuine confusions,** all 7 released. A itself preferred the selected relation, at
    about 0.40, against 0.21–0.39 for the reference.
  - **In world 101, A preferred `<end>`** (52; unit medians 0.966–0.987). The forced second relation was chosen among
    small probabilities, the programs scored low (score medians 0.005–0.028 by unit), and all but one were filtered.
  - **Correct S2 programs** scored about 0.99.

**A against A+K, question by question.**
- **A+K never broke a question A had exact.**
- **Where only A+K was exact,** A's own program had first left the reference by stopping early or by choosing a
  relation of the wrong source type:

  | | Questions | A stopped early | A first chose a relation of the wrong source type | A produced no program |
  |---|---:|---:|---:|---:|
  | S2 | 585 | 426 | 159 | 0 |
  | S1 | 3,633 | 2,240 | 1,386 | 7 |

  - **This is judged by A's first divergence from the reference,** not by its length. A program that begins with a
    wrong-type relation and then ends early counts as a wrong-type choice.
  - **"No program"** is A's `malformed` outcome: `<end>` before any relation, or no `<end>` within the step limit.

So the constraints repair two things: premature stopping, through the slot count, and wrong-type choices, through the
types. They cannot repair a choice between two relations of the same source type, or the low scores on programs
longer than A's training.

**Selection ties.** None of the 37,492–37,973 recorded A+K steps per unit had a tie at the maximum allowed
probability, or an allowed probability of 0. The recorded pilot chose the highest allowed probability. The corrected
code chooses the highest allowed logit, and it would have made every recorded choice the same.

## Cost (per unit; corrected accounting from the saved records)

| | A | F |
|---|---|---|
| Updates × batch | 1,500 × 64 | 1,500 × 64 |
| Presentations drawn | 96,000 questions | 96,000 segments, drawn from 1,657–2,007 aligned |
| Tokens | 742,457–768,389 | 263,769–275,158 |
| Training forward MACs, unpadded (each question's own tokens) | 86.3–87.4 G | 11.4–11.9 G |
| Training forward MACs at the padded batch width | 86.8–87.9 G | (packed GRU and no attention: padding adds no counted MACs) |
| Parameters | 223,817–223,865 | 71,775–71,823 |
| Training wall time | 15.3–17.5 s | 4.2–4.7 s |

**Equal update counts are not equal compute:** F used about 7× fewer forward MACs.

**Evaluation over 17,900 items:**

| | Decoder work | Wall time |
|---|---|---|
| A | 71,600 evaluated row-steps (every step for every row); 36,112–37,129 emitted decisions | 1.1–1.4 s |
| A+K | 55,004–55,040 evaluated row-steps (a batch steps until its longest row ends); 37,492–37,973 emitted decisions; 108–117 grammar refusals | 1.2–1.6 s |
| F and F+B | 30,151–30,156 segment passes; F+B adds 11,238–11,290 enumerated candidates, 109–161 pruned | 0.66–0.72 s, **one combined measurement** |
| L, G, W | segment lookups only | 0.17–0.32 s |

**The F and F+B measurement** runs from after F's fit to the end of both policies. It covers the grounding pass, both
policies' decisions and their records. The grammar's segmentation ran once before A's fit, outside every system's
timing. L, G and W re-segment inside their own timing.

## What this establishes, and what it does not

**It establishes, on development worlds 100–101 of this controlled language:**
- **The listed results:** that lookup suffices, F's lookup-level performance, F+B's lack of benefit, W's grounding of
  the declared recombinations, and A's sanity-gated replication.
- **The diagnosis of A+K's remaining limitations.**

**It does not establish:**
- **Anything on fresh worlds.** There was no confirmation.
- **A need for neural lexical generalization or for bounded alternatives.** No-fit controls solve both the language
  and the declared lexical condition.
- **That factoring is necessary.** A+K shows how far constraints on the existing model go, and where they stop.
- **Anything about natural language.** Its segment rule is supplied, and only possible because this language was
  designed to admit it.

## Closing this phase, and what a later proposal needs

**This phase closes as a development engineering result.** A lexical task is not begun merely to defeat W. A later
proposal must do three things:
- start from a useful missing capability;
- specify what makes its semantics learnable;
- include the cheapest relevant baseline.

On this language, those baselines are exact lookup (L) and word association (W).

## Corrections made while closing

The recorded pilot stays bound to `0e1a3c9`. Its source hashes are checked against that commit's (from `git show`),
not replaced.
- **The verifier** now also compares, for every certificate:
  - alpha and delta;
  - A's historical eligible calibration records;
  - both of A's historical category tables, in full.

  It previously accepted fabricated values in these fields. New rejection tests cover each.
- **The accounting:**
  - emitted decisions are now separate from evaluated decoder row-steps;
  - A's training MACs are labelled unpadded and padded;
  - F and F+B have one combined duration, with its boundary stated.

  The recorded pilot's field `decoder_steps` for A+K counts emitted decisions.
- **A+K's selection** now uses the allowed logits, with scores still from the original probabilities, plus an
  underflow fixture. The recorded pilot used the earlier, probability-based rule. On its records that changes no
  choice, because no recorded step tied.

## Evidence

**The export is complete.** All 58 original export files passed an integrity check: every gzip file decompressed in
full and every JSON document parsed. Every per-item records file holds the 17,900 items, and each file's sha256 is in
`export_integrity.json`.

**In the bundle,** `experiments/results/ti_tnn/factored_interpreter/`:
- the no-fit controls run, whole;
- the pilot's `run.json`, per-unit `unit.json`, and A, F and F+B records;
- the current verifiers' outputs (`controls_verification.json`, `pilot_verification.json`);
- the diagnosis (`ak_diagnosis.json`).

**The eight A+K step-trace files** (27.1 MB) are bound by their sha256 in `pilot_trace_release.json`.

**Rebuilding:** `python experiments/results/ti_tnn/make_tables.py --factored-traces <dir>`. It checks each trace
against its hash, re-verifies the pilot, re-runs the diagnosis, and requires both to equal the bundled outputs.
Without the traces, the controls are re-verified from the bundle alone.
