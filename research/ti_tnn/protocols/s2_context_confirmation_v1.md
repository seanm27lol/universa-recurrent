# S2 diagnosis: fresh confirmation of arms A, B and C (frozen)

**Status: FROZEN 2026-10-19 (UTC), before any fit on worlds 710–712.**
- **The frozen code** is the commit that adds this file:
  - `bench/s2_development.py` (arms, addition audit, fits; unchanged since `ece0f86`);
  - `bench/s2_development_analysis.py` (re-derivation from records, `confirmation_verdict`);
  - `bench/s2_diagnosis.py`.
  The study's generator, audit, decisions, certification and bars come from `bench/interpreter_robustness.py`,
  unchanged since `69b7fd7`.
- **The design** was fixed in the [development brief](s2_diagnosis_brief.md) before any intervention fit.
  This file adds the secondary prediction and the exact commands.

## Why it runs

The brief's frozen decision rule was met by the development fits (worlds 100–101 × inits 100–102):
- **C above B in both worlds:**

| World | C mean S2 exact | B mean S2 exact |
|---|---:|---:|
| 100 | 0.8958 | 0.8746 |
| 101 | 0.9613 | 0.8963 |

- **C met the S3/S4 joint criterion in all six units.**

**Expectation, stated before running.** The primary prediction is unlikely to hold:
- C met the study's S2 criterion in **1 of 6** development units;
- its certified S2 error counts were 63, 39, 0, 38, 17 and 38.

The confirmation runs because the frozen rule says it should, not because a pass is expected.

## Units and arms

- **Worlds:** `generate_world` seeds **710, 711, 712**, with 96 countries.
  - No TI-TNN run or protocol has used them; this was checked against every run configuration under `runs/`
    and every protocol.
  - Worlds 700–702 are historical evidence and are not reused.
- **Inits:** **200–204**.
- **Arms:** **A, B, C** as defined in the brief:
  - **Count:** each addition has N_w examples, the number of taught one-step training questions in world w.
  - **C's filler-length distribution:** the brief's frozen procedure applied to each world's own training
    data, namely the token lengths of the Base prefix phrases in its prefix-style training questions. The
    values are recorded in the run.
  - **Fits:** 15 per arm, 45 in all.
- **Training:** the robustness study's `word_aug` recipe, unchanged: 1,500 updates of batch 64.
- **Not run:** D.
- **Sampling:** the study's frozen generator (revision 4); the additions are seeded
  `s2-development|<world identity>|<arm>`.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/s2_development.py \
    --arms A B C --worlds 710 711 712 --inits 200 201 202 203 204 --out runs/titnn_s2_diagnosis/confirmation
.venv-phase2/bin/python research/ti_tnn/bench/s2_development_analysis.py runs/titnn_s2_diagnosis/confirmation/<run> \
    --out <dir>
```

## Criteria (the robustness study's frozen bars)

- **Certification:** fixed-sequence certification on the calibration split, using:
  - the study's grid (0.999 down to 0);
  - risk 1% at level 0.05.
  An uncertified unit fails every criterion.
- **The S2 criterion:**
  - certified S2 coverage ≥ 0.80;
  - a Clopper–Pearson 95% upper bound ≤ 1% on the error rate of answered S2 questions.
- **The S3/S4 joint criterion:**
  - certified S3 coverage ≥ 0.80, with the same error bound;
  - abstention ≥ 0.95 on both S4a_primary and S4b_primary.
- **A unit passes** when the **same unit** meets both the S2 criterion and the S3/S4 joint criterion.
- **Aggregation:** an arm holds in a world when at least **4 of its 5** units pass. It holds when it does so
  in **every** world.

## Predictions (one-sided)

1. **Primary.** C holds.
2. **Secondary.** In every world, C's mean S2 exact accuracy over the five inits (program and entity, all 800
   S2 questions) is strictly higher than B's.

**Reported regardless of outcome:**
- A and B under the same rule;
- each criterion alone, by world;
- every unit's metrics;
- the S2 classes.

## Analysis

`s2_development_analysis.py` re-derives every metric, certificate and unit outcome from the decision records,
using the robustness verifier's independent code, and stops on any difference from the stored summary.
`confirmation_verdict` then checks four things before judging:
- the exact inventory of A, B, C × 710–712 × 200–204;
- the frozen configuration (96 countries, 1,500 updates);
- that the research tree was unmodified when the run started;
- the rule above.

## What each outcome would mean

| Outcome | Meaning |
|---|---|
| Primary holds | In fresh worlds, adding the taught one-step questions inside longer wrappers gave S2 transfer that meets the bar, while keeping novelty behaviour. |
| Primary fails, secondary holds | The longer context improved S2 exact accuracy beyond the exposure control, but not enough to meet the S2 criterion. |
| Both fail | The development difference between C and B did not replicate. |

In no case does the result establish "taught wording → one-relation target" as the mechanism. C changes length
and surrounding context and leaves that mapping intact (see the brief's interpretation limits).

## Time limit

The overnight budget allows the 45 fits. If they cannot complete, the run is reported as incomplete: no verdict
is drawn from a partial inventory.

## Execution record (appended 2026-10-19, after the run)

- **The run:** `20261019T084110`, from `07bb97e` with an unmodified research tree.
  - 45 fits, all completed, 12:41–12:55 UTC on CPU.
  - Every greedy trace reproduced its decisions.
- **The analysis:** run unchanged. Every metric, certificate and unit outcome was re-derived from the records
  and matched the stored summary. The verdict is complete.
- **Primary prediction: not met.** C passed S2 and S3/S4 in the same unit in 0, 0 and 3 of 5 units (worlds
  710, 711 and 712).
- **Secondary prediction: not met.** C's mean S2 exact accuracy was 0.8578, 0.8778 and 0.8912, against B's
  0.9123, 0.9045 and 0.8540, so C was above B only in world 712.
- **Reported arms:**
  - A: 2/2/1;
  - B: 2/0/1;
  - every arm met the S3/S4 joint criterion in all 15 units.
- **Deviations from this protocol:** none.
