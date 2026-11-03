# C4/T4 learning pilot: results (development; controllers only)

**Registered outcome: A3, no pilot difference by the pre-set rule.** T4 cleared L in both seeds; C4_params did not.
Under the frozen protocol that makes the finding an asymmetry: one arm rose above the no-fit control in both seeds,
the other did not. The paired results are reported in full, but descriptively. They are not an architectural result
between two arms that both learned.

"Cleared L" is descriptive only. Beating a no-fit control does not isolate the causal effect of training. This is
development evidence from 64 questions and two seeds, and confirms nothing.

Protocol: [`PROTOCOL_pilot.md`](PROTOCOL_pilot.md), frozen at 5bb852c (sha256 81427fdc…).

**Corrected 2026-11-02:** reporting, verification and resume only. The run records and the frozen outcome are
unchanged; see [Corrections](#corrections-of-2026-11-02-reporting-verification-and-resume-only).

## The counts (64 questions, every request in the denominator)

| | seed 100 | seed 101 | pooled (of 128) | cleared L in both seeds |
|---|---|---|---|---|
| T4 (typed specialists) | 9 | 9 | 18 | yes (9 > 4, 9 > 4) |
| C4_params (shared, parameter-matched) | 4 | 10 | 14 | no (4 = 4 in seed 100) |
| L (retrieval + templates, no fit; deterministic) | 4 | 4 | 8 | — |

- **Differences:**
  - T4 − C4_params is +5 in seed 100 and −1 in seed 101: +4 pooled, of 128 question-seed pairs.
  - A1 needs a lead of at least 7 and a positive difference in both seeds; seed 101 is negative, so the rule outcome
    is A3.
  - Per question, with T4 − C4_params summed over the two seeds: 51 questions at 0, 7 at +1, 1 at +2, and 5 at −1.
- **Why C4_params missed the gate:** seed 100's C4_params ties L at 4. Its seed 101 scores 10, more than T4's 9. So
  the asymmetry rests on one seed of one arm, and the seed-to-seed variation within C4_params (4 against 10) is
  larger than the pooled difference between the arms.

**Exploratory database-block sign-flip analysis** (not part of the rule):
- **The data:** the observed pooled difference is +4; 7 of the 14 database blocks are nonzero.
- **The test:** all 2⁷ = 128 sign assignments were enumerated, giving a two-sided p of 0.36 (the smallest attainable
  is 0.016).
- **What it covers:** it conditions on the four fitted models, so it covers between-database variation, not
  training-seed variation. No false-positive rate is claimed.

## Per database

| database | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 | pooled difference |
|---|---|---|---|---|---|---|---|
| computer_student | 4 | 0 | 1 | 1 | 1 | 1 | +0 |
| disney | 4 | 0 | 1 | 1 | 1 | 1 | +0 |
| food_inspection | 4 | 0 | 1 | 0 | 1 | 2 | −2 |
| hockey | 5 | 0 | 0 | 1 | 0 | 0 | +1 |
| ice_hockey_draft | 5 | 0 | 0 | 0 | 0 | 0 | +0 |
| image_and_language | 5 | 1 | 1 | 0 | 0 | 0 | +1 |
| music_platform_2 | 4 | 0 | 1 | 1 | 0 | 1 | +1 |
| professional_basketball | 5 | 0 | 0 | 0 | 0 | 0 | +0 |
| public_review_platform | 5 | 0 | 0 | 0 | 0 | 0 | +0 |
| simpson_episodes | 5 | 1 | 1 | 1 | 0 | 1 | +1 |
| soccer_2016 | 5 | 0 | 0 | 0 | 0 | 0 | +0 |
| software_company | 4 | 0 | 0 | 0 | 0 | 0 | +0 |
| talkingdata | 4 | 0 | 1 | 1 | 0 | 1 | +1 |
| world | 5 | 2 | 2 | 3 | 1 | 3 | +1 |

Five databases (ice_hockey_draft, professional_basketball, public_review_platform, soccer_2016, software_company)
have no correct answer from any arm or seed.

## Gold-program representability

58 of the 64 gold programs are representable: they convert to tirel-1 and trace through the action grammar with their
literals in the question or evidence tokens. The other 6 are not encodable for that literal reason. All 64 stay in
every denominator.

| stratum | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 |
|---|---|---|---|---|---|---|
| representable | 58 | 4 | 8 | 9 | 3 | 9 |
| not representable | 6 | 0 | 1 | 0 | 1 | 1 |

An unrepresentable gold program can still be answered correctly, because EX compares result sets and a different
program can return the same rows. The 6.5% (330 of 5,071) is the *training exclusion rate*: training-pool questions
without a supervisable gold program. It is not an evaluation ceiling.

## Training and serving

| run | updates | training loss, first → last | seconds per update (mean / median) | checkpoints (save s) |
|---|---|---|---|---|
| T4 s100 | 1,500 | 4.656 → 0.159 | 0.632 / 0.580 | 6 (0.77) |
| T4 s101 | 1,500 | 4.643 → 0.174 | 0.611 / 0.574 | 6 (0.87) |
| C4_params s100 | 1,500 | 4.720 → 0.169 | 0.588 / 0.557 | 6 (0.85) |
| C4_params s101 | 1,500 | 4.679 → 0.167 | 0.583 / 0.567 | 6 (0.67) |

At evaluation, every arm answered all 64 requests; nothing abstained.

**Time to SQL submission.** Each record's time runs from the request's start until its final SQL is submitted, at
batch size one. For the controllers, that covers the fresh forward of the frozen 4B (most of it) and the decode; for
L, retrieval and its candidate queries. The scorer's execution of the submitted query comes afterwards and is not
included.
- **Means per request:** the four controller runs averaged 1.33–1.35 s; L averaged 3.00 s.

**Generation cost per EX-correct submission:** the 64 requests' time to SQL submission, divided by the EX-correct
submissions.
- **It is the frozen protocol's cost unit, unchanged.** The protocol calls it "end-to-end seconds"; the recorded
  seconds end at submission, so the metric is relabelled here, not recomputed.

**Descriptive accounting, outside the frozen metric:** the scorer's recorded execution of each submitted query, and
generation plus that execution per EX-correct submission.
- **Timeouts dominate the execution column.** Six submitted queries hit the scorer's 30 s timeout: T4 1 and 2, and
  C4_params 0 and 3, by seed. They score 0, and each recorded 30.0–33.2 s.

| run | EX-correct | time to SQL submission, s (64 total / mean) | generation cost per EX-correct submission, s | recorded submitted-query execution, s (timeouts) | generation + execution per EX-correct, s (descriptive) |
|---|---|---|---|---|---|
| T4 s100 | 9 | 86.2 / 1.35 | 9.6 | 36.0 (1) | 13.6 |
| T4 s101 | 9 | 86.0 / 1.34 | 9.6 | 66.4 (2) | 16.9 |
| C4_params s100 | 4 | 86.2 / 1.35 | 21.5 | 1.7 (0) | 22.0 |
| C4_params s101 | 10 | 85.2 / 1.33 | 8.5 | 94.1 (3) | 17.9 |
| L | 4 | 192.3 / 3.00 | 48.1 | 4.5 (0) | 49.2 |

The execution column changes the ordering: C4_params' seed 101 has the lowest generation cost per EX-correct
submission (8.5 s), but not the lowest once its three timed-out queries are counted (17.9 s).

## Integrity

Every gate of the frozen protocol passed:
- **Preparation:** all 4,741 training examples were prepared from anchored entries with validated sha256, dtype and
  shape.
- **Final checkpoints:** each evaluated checkpoint is update 1,499 and matches its recorded digests.
- **Fresh states:** for every request in both evaluations, they equal their anchors and are identical across the two
  arms.
- **Reconstruction:** every request's reconstruction equals its timed decode.
- **Selection:** exactly one passing evaluation per seed, and the L run postdates the freeze.

**Incidents, all in the record:**
1. **The first feasibility run** (0.130 h) found the evaluation's TF32 setting leaking into the backbone's float32
   matmuls: its fresh states matched the anchors 0/8.
   - **The fix:** the backbone runs at "highest"; TF32 is used only for decoding.
   - **Review and re-run:** the fix was reviewed, and a re-run (0.042 h) gave 8/8, equal reconstructions, and a
     cross-process match with the smoke's states.
2. **A host crash on 2026-10-30, after 18:53:09 UTC,** killed the encoding of chunk 1200–2400 at 848 of 1,200
   (0.559 h charged from heartbeats).
   - **The damage:** two cache entries written in its last seconds lost their data, though their fsynced anchors
     survived.
   - **The cost:** the first resume stopped on them (0.031 h, failed).
   - **The recovery:** both were quarantined (kept), recomputed and found exactly equal to their original pre-crash
     anchors.
   - **The fix:** cache, checkpoint, log and manifest writes are now made durable before they are recorded; this was
     reviewed before training.
   - **What remained sound:** every other one of the 2,202 entries anchored at the time hashed exactly to its anchor.

## Resources

| stage | projected h (re-run feasibility) | actual h |
|---|---|---|
| feasibility (two runs, before the projection) | — | 0.171 |
| encoding: 64 dev contexts and 4,741 training contexts, with loads | 2.504 | 2.565 (the interrupted reservation's 0.559 and the failed resume's 0.031 included) |
| training: 2 seeds × 2 arms × 1,500 updates | 0.834 | 1.026 |
| evaluation and reconstruction: 2 seeds × 2 arms | 0.164 | 0.137 |
| **after the projection** | **3.502** | **3.728** |
| **pilot total** | | **3.899** |

- **Why training ran over (+0.19 h, 23%):** updates took 0.58–0.63 s against the 0.50 s measured on the
  160-context sample; prefetch waits explain only about 0.04 s of that.
- **Encoding held overall,** though its chunks varied about 4×:
  - long contexts cluster in release order (the resumed chunk's 346 new contexts averaged 13,951 tokens), at a steady
    0.37–0.40 s per 1,000 tokens;
  - the 846 sound entries of the crashed reservation were reused, so the crash cost only the failed resume, two
    recomputes and a reload.
- **The whole study:** 6.878 of the 8 authorized GPU-hours, every reservation included; the full table is in the
  bundle (`python -m scaling.pilot_report`).
- **Unfinished:** nothing within this authorization.
- **Deferred:** B4/Q27 on the 64, the reserved benchmark, and any confirmation.

## Independent verification

**The verdict reproduced exactly, twice:** A3, with T4 clearing L in both seeds and C4_params not.
- **First, by the independent reviewer**, against an earlier snapshot of the bundle.
- **Then, on the final bundle,** by the corrected verifier.

The second run's public arithmetic reconstruction and its local checks are reported separately, because they
establish different things.

### The reviewer's verification (2026-11-02, preserved)

The independent reviewer reconstructed the verdict from the records with `independent/verify_pilot.py` as committed
in 0c420ab, which reuses `verify_baselines.py`'s helpers.
- **Its own code** covers the EX rule, the JSON and ledger parsing, the tensor and optimizer digests, the training
  stream (`torch.randperm`) and the sign-flip enumeration. The pins and contracts are read at each run's commit.
- **Producer code** was used only where flagged: the representability trace, and an informational CPU re-decode.
- **Outputs:** `pilot_verdict.json`, `pilot_verdict_redecode.json`, `pilot_verdict_summary.json` (sha256
  78714dfc…) and `verify_L64.json`, all in `evidence/pilot_v1/independent/`. They are kept byte for byte.

**They verified an earlier, 84-file snapshot of the bundle,** taken before the quarantine record and the outputs
themselves were added. Their `bundle` sections describe that snapshot, which is labelled in
`evidence/pilot_v1/independent/README.md`.

**What it checked, all equal to the analysis:**
- **The 64:** reconstructed from the raw release.
- **Counts and analysis:** every final SQL was re-executed against gold, giving T4 9/9, C4_params 4/10 and L 4. The
  gates, the per-seed and pooled differences, the per-question histogram, the per-database table and the sign-flip
  (p = 0.359375 over 7 nonzero blocks) all match.
- **Representability:** 58/64, with the same strata. The 6 unrepresentable ids are 01167, 01686, 02636, 02670,
  03130 and 06228, all "literal not groundable".
- **Integrity:** 133 checks and the L run's 1,348 (`verify_baselines`), with 0 failures:
  - each final checkpoint re-hashes to its name, and its model and optimizer digests are recomputed;
  - each embedded config equals `config.json`;
  - each train log holds updates 0–1,499 once, and every update's examples equal the seeded stream over
    `pilot_train_v1`, identical across the arms of a seed;
  - within a seed, the arms' configs differ only in arm, `c4_params_hidden` and parameters;
  - the fresh states equal the anchors and are identical across the arms;
  - exactly one passing evaluation per seed;
  - the L run's commit contains the freeze;
  - the bundle (that snapshot) is byte-equal to the runs.
- **CPU re-decode** (informational: float32 on the CPU against TF32 on the GPU): 256 of 256 requests gave identical
  actions and SQL.
- **The ledger:** 6.878 h in total, 3.899 h for the pilot.

**Two limits of that version**, corrected below; neither affects its verdict:
- it took an evaluation's eligibility from the producer's `passed` flag, without deriving it from the complete
  reconstruction inventory (both inventories are entirely true);
- it compared `analysis.json`'s inputs by absolute path (they were equal).

**Discrepancies it reported, none affecting the verdict:**
- the quarantine record was missing from the bundle (now included);
- training ran 23% over its projection;
- per-chunk encoding costs varied about 4×;
- C4_params misses the gate by a single tie;
- 3 question-runs were answered correctly although their gold program is unrepresentable, which EX allows.

### Verification of the final bundle (2026-11-02 correction pass)

The producer-side session ran the corrected `independent/verify_pilot.py` at the user's request; this is not a
rerun by the reviewer. It verified the bundle at commit 4f2a495, as listed by the `SHA256SUMS` with sha256
3f6a1337…. Its output is `evidence/pilot_v1/independent/final_bundle_verification.json`, added afterwards.

**Public arithmetic reconstruction:** 87 checks, 0 failures.
- **What it reads:** only the bundle, plus the checkout's pins, protocol and git history.
- **Eligibility:** each evaluation's comes from its complete reconstruction inventory; both runs are eligible, and
  the producer's `passed` flags agree.
- **Totals:** each run's totals and cost ratios are recomputed from its records and equal the summaries.
- **The decision:** the counts, gates, differences, rule, histogram, database table and sign-flip, recomputed from
  the *recorded* EX scores, equal `analysis.json`. Runs are matched by their path relative to the runs root.
- **What it establishes:** that the decision follows arithmetically from the bundled records. It does not
  establish that the recorded scores are right.

**Local checks:** 343 checks, 0 failures. They needed the runs root: the raw release, the SQLite databases,
the checkpoint files and the cached states. They establish what the public part cannot:
- the 64 are rebuilt from the raw release;
- all 256 controller SQL submissions and L's 64 are re-executed with the reviewer's EX rule, and every recorded
  score is re-derived;
- the four final checkpoints re-hash to their names and records, with their model and optimizer digests recomputed;
- 85 bundled run copies equal their run originals (the 84 of the earlier snapshot and the quarantine record);
- representability recomputes to 58/64 (producer grammar, flagged);
- the CPU re-decode gives identical actions and SQL on 256 of 256 requests (producer decoder, flagged,
  informational).

**From a relocated clean checkout.** A fresh clone of the pushed branch at 5366639, in another directory, had no
runs root and `TI_TNN_SCALING_RUNS` unset. In it:
- `python -m scaling.pilot_report` printed tables identical to those printed in place (98 lines);
- the public reconstruction gave 87 checks with 0 failures, and an output identical to the in-place run's, section
  by section, over the 91-file bundle (`SHA256SUMS` a551fbda…);
- the focused rejection tests passed: 20 of 20 (bundle consistency and relocation, the verifier's rejections, run references, both resume cases and a moved runs root, inventory eligibility, and the protocol refusals).

## Corrections of 2026-11-02 (reporting, verification and resume only)

After review, one CPU-only correction pass. No run record, cache entry or checkpoint was changed, and the frozen
outcome stands. No fit, GPU reservation or new evaluation was made.

1. **The public bundle command** (`python -m scaling.pilot_report`) now recomputes the decision from the decision
   records before printing anything:
   - **what it recomputes:** the counts, L statuses, paired differences, database totals, histogram, rule and
     block statistic, each run's totals, and each evaluation's eligibility from its reconstruction inventory;
   - **what it compares them with:** `analysis.json` and the summaries;
   - **on any disagreement** it stops, naming it.
   - **Tests:** tampered bundle copies, with `SHA256SUMS` regenerated so the checksums pass, are refused.
2. **The independent verifier** now derives each evaluation's eligibility from its complete reconstruction inventory:
   - a single false entry rejects the run even when the summary's `passed` stays true, and that disagreement is
     itself a failed check;
   - it recomputes each run's reported totals and cost ratios against the producer's summaries.
3. **Run identity.** Runs are compared by their path relative to the runs root (`identity.run_ref`), not by absolute
   path. That covers:
   - the analysis inputs, the bundle command and the verifier;
   - the dev verifier's ledger match;
   - resume, which now finds each recorded checkpoint by name in the run's own directory.

   The frozen `analysis.json` keeps its absolute paths; both forms are read. Reconstruction was tested after
   relocating the evidence: a moved bundle copy in the tests, and a fresh clone (see below).
4. **The serving metric is relabelled:** time to SQL submission, and generation cost per EX-correct submission.
   - **The frozen metric is unchanged.** The earlier text said every arm took 1.33–1.35 s end to end; that range
     describes the controller runs' means only, and L averaged 3.00 s.
   - **The scorer's recorded execution** of the submitted queries is now reported separately, as descriptive
     accounting.
5. **Resume:** a run with a complete training log but a missing or damaged final checkpoint was refused as already
   complete, with no way to recover. It now resumes from its newest sound checkpoint.
   - **Refused only when complete:** a run whose log is complete *and* whose final checkpoint passes the
     final-checkpoint gate is still refused.
   - **Regression tests:** both cases, and a moved runs root.
6. **The reviewer's original outputs** are preserved byte for byte and labelled as verifying an earlier 84-file
   bundle snapshot (`evidence/pilot_v1/independent/README.md`). A verification of the final bundle is appended,
   with the public arithmetic reconstruction and the local checks reported separately.

## What this does not establish

- That T4's specialization helps. The rule outcome is A3.
- That either arm learned the task in a causal sense. "Cleared L" is descriptive, and C4_params' seeds disagree (4
  against 10).
- Held-out or reserved-benchmark performance, or anything about B4 or Q27.

Development gold informed the grammar's design.

## Evidence and reproduction

**The bundle.** `research/ti_tnn/scaling/evidence/pilot_v1/` (weight-free, with `SHA256SUMS`) holds:
- every encoding, feasibility, training, evaluation and analysis run, failed, interrupted and superseded ones
  included;
- the L run, the quarantine record and the ledger;
- the verification outputs.

**Rebuilding from the bundle alone,** from `research/ti_tnn` in any checkout:

```bash
python -m scaling.pilot_report                                     # recompute from the records, compare, print the tables
python scaling/independent/verify_pilot.py                         # the public arithmetic reconstruction
python scaling/independent/verify_pilot.py --local --representability --redecode   # with the runs root (local checks)
```

`pilot_report` stops, naming the disagreement, if `analysis.json` or a summary differs from what the decision
records give.

**Reproduction.** From the run worktree at 120412d (encodings: 4484ee6, then 41dd184):

```bash
python -m scaling.cli pilot-feasibility --projected-hours 0.3
python -m scaling.cli pilot-encode --what dev --projected-hours 0.1
for r in 0:1200 1200:2400 2400:3600 3600:4741; do python -m scaling.cli pilot-encode --what train --range $r --projected-hours 0.9; done
python -m scaling.cli pilot-train --seed 100 --projected-hours 0.3 && python -m scaling.cli pilot-train --seed 101 --projected-hours 0.3
python -m scaling.cli pilot-eval --seed 100 --projected-hours 0.15 && python -m scaling.cli pilot-eval --seed 101 --projected-hours 0.15
python -m scaling.cli baseline --arm L --subset 64 && python -m scaling.cli pilot-analyze
```
