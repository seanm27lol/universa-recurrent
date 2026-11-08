# Does database evidence improve ordinary revision? Results of the frozen D0 / D1 / V1 run

**Result.** V1 did not meet the prespecified continuation threshold on this population, and this version stops.
There was no measured net accuracy improvement over ordinary revision, at a higher cost.
- **V1 against D1:** 80 of 128 each, a difference of 0 (-1 in seed 200, +1 in seed 201).
- **Interval:** the 95% question-level bootstrap interval is [-5, +5] correct answers out of 128, about -3.9 to +3.9
  percentage points. It does not account for clustering by database.
- **Discordant answers:** 3 recoveries and 3 regressions.
- **Cost:** V1's end-to-end cost was 3.4% higher than D1's.

This measures one frozen lookup on one fresh population. It is not proof that database evidence has no effect, and
not proof that its effect is small.

**Status:** one GPU run from the frozen protocol (`docs/PROTOCOL_database_evidence_revision.md`, pins at 8e1ee1b,
`frozen_at` 644c090). The run took 0.461 GPU-reservation hours of the 1-hour allowance, and the ledger stands at 23.951
of 32 h. No training, no reserved-benchmark read, no rerun.

## A concrete example

Take a question whose direct proposal fails because of how the database stores a value or names a column.
- **Ordinary revision (D1)** shows the model its query's result and asks it to check the query.
- **Database-evidence revision (V1)** shows the same, plus facts looked up in the database: nearest stored values,
  near-miss columns and tables, declared keys and value formats.

The lookup is reference-blind and deterministic. V1's message is D1's with that evidence inserted, and nothing else.

## What ran

- **Population.** Inventory positions 256-319: 64 questions over 14 databases, seeds 200 and 201, so 128
  question-seed pairs per policy. The classified prior-use scan found only the disclosed, allowed uses: inventory
  membership and the declared mechanical screening (`pins/dbe_prior_use_scan.json`).
- **Policies.** All three share one initial proposal per question and seed:
  - **D0:** the initial direct Qwen3.5-4B proposal;
  - **D1:** one ordinary revision;
  - **V1:** the same revision with the db-evidence-v1.4 payload.
- **Held equal across D1 and V1:**
  - the initial reply, the raw observation, the revision seed and the prefix;
  - 512 tokens;
  - the selection rule;
  - the model-revision allowance. The lookup ran before the shared remaining time was taken, so the policies had
    equal revision opportunity, not identical end-to-end time.
- **Lookup cost.** The lookup is charged to V1's cost: 6.2 s in total, median 0.008 s, maximum 0.55 s.
- **Lookup outcomes:**
  - none failed;
  - no instruction-budget stops;
  - 12 hit a scan limit;
  - none was truncated;
  - 30 recorded unresolved columns;
  - the payload median was 786 characters, with a maximum of 1,718.

## Results

| policy | seed 200 | seed 201 | pooled (of 128) |
|---|---|---|---|
| D0 | 38 | 32 | 70 |
| D1 | 41 | 39 | 80 |
| V1 | 40 | 40 | 80 |

**By evidence signal.** The groups are computed from the shown payload without any reference, and are descriptive
only.
- **"Value not stored"** is a value item with no verbatim row and candidates shown.
- **"Other signal"** is a value found elsewhere, a flagged join, or a flagged value representation.
- **"No signal"** means no flagged signal. It does not mean the payload was empty: those payloads still showed declared
  keys, column notes, near-miss columns and tables matching the question.

| signal group (descriptive) | requests | D0 | D1 | V1 | V1 right, D1 wrong | D1 right, V1 wrong | D0 right, V1 wrong |
|---|---|---|---|---|---|---|---|
| no flagged signal | 85 | 49 | 54 | 56 | 3 | 1 | 0 |
| other signal | 42 | 21 | 26 | 24 | 0 | 2 | 1 |
| value not stored | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

**All three V1-only recoveries occurred in the no-flagged-signal group.** The one V1 regression of an initially correct
answer occurred in the other-signal group.

| comparison | pooled | per seed | recoveries | regressions | 95% interval (correct answers of 128) | leave-one-database-out min | all conditions | sign-flip p (exploratory) |
|---|---|---|---|---|---|---|---|---|
| V1-D1 | +0 | -1, +1 | 3 | 3 | [-5, +5] | -1 | False | 1.000 |
| D1-D0 | +10 | +3, +7 | 10 | 0 | [+4, +18] | +8 | True | 0.031 |
| V1-D0 | +10 | +2, +8 | 11 | 1 | [+3, +18] | +7 | True | 0.031 |

**Decision:** V1 did not meet the prespecified continuation threshold on this population: this version stops.

| transition | count |
|---|---|
| V1 right where D1 is wrong | 3 |
| D1 right where V1 is wrong | 3 |
| initially correct (D0) | 70 |
| initially correct, broken by D1 / V1 | 0 / 1 |
| initially wrong, fixed by D1 / V1 | 10 / 11 |

| policy | generation s | lookup s | tool s | submitted execution s | total s | per request s | per EX-correct s | prompt tokens | generated tokens | peak memory GB |
|---|---|---|---|---|---|---|---|---|---|---|
| D0 | 517.1 | 0.0 | 0.0 | 3.3 | 520.4 | 4.07 | 7.4 | 332,006 | 7,092 | 11.06 |
| D1 | 1039.5 | 0.0 | 8.6 | 3.2 | 1051.3 | 8.21 | 13.1 | 684,238 | 14,154 | 11.13 |
| V1 | 1063.1 | 6.2 | 11.7 | 6.1 | 1087.1 | 8.49 | 13.6 | 719,542 | 14,308 | 11.23 |

Lookups: 0 failed; 0 instruction-budget stops; 12 with scan limits; 0 truncated; 30 with unresolved columns; 0 wall-clock safety stops; total 6.2 s, max 0.55 s; largest payload 1718 characters.

GPU reservations: dbe run 0.461 h (ok). Ledger: 23.951 of 32.0 h.

| database | questions | D0 | D1 | V1 |
|---|---|---|---|---|
| computer_student | 4 | 4 | 4 | 4 |
| disney | 5 | 4 | 6 | 6 |
| food_inspection | 4 | 6 | 6 | 6 |
| hockey | 4 | 4 | 4 | 4 |
| ice_hockey_draft | 5 | 4 | 4 | 5 |
| image_and_language | 4 | 3 | 5 | 6 |
| music_platform_2 | 5 | 8 | 8 | 8 |
| professional_basketball | 5 | 1 | 3 | 2 |
| public_review_platform | 4 | 4 | 6 | 6 |
| simpson_episodes | 5 | 6 | 7 | 6 |
| soccer_2016 | 4 | 4 | 5 | 5 |
| software_company | 5 | 7 | 7 | 7 |
| talkingdata | 5 | 5 | 5 | 5 |
| world | 5 | 10 | 10 | 10 |

**Reading:**
- **The primary comparison is a tie.** V1 and D1 end on the same final SQL in 91 of 128 requests. V1 changed the
  initial SQL more often than D1 (48 revisions changed it, against 33), but the changes cancel out: 3 answers V1 fixed
  that D1 missed, and 3 the other way. The interval [-5, +5] covers effects of either sign up to about 5 of 128.
- **Leaving out one database.** V1 - D1 is -1, 0 or +1 for every database left out. No database carries an effect.
- **Harm to initially correct answers.** V1 broke 1 of the 70 initially correct answers (bird23tf-00441, seed 200),
  and D1 broke none. That request showed relationship and table evidence. One case is too few to call a distractor
  effect, so it is reported, not interpreted.
- **The specific signal barely fired.** "Value not stored, candidates shown" occurred in 1 of 128 requests, and that
  request stayed wrong under all three policies. In the coverage study, it was the signal confined to the targeted
  grounding cases. Here most V1 requests carried only less specific evidence (42 "other signal") or none of it
  (85). This population gave the lookup's strongest signal little to act on, and a different population could
  differ.
  - The signal groups are descriptive and computed without references. No subgroup is selected or proposed as a
    policy.
- **Secondary: ordinary revision helps on this population.**
  - D1 - D0 is +10, with 10 recoveries and no regressions. Its interval is [+4, +18] correct answers of 128, about
    +3.1 to +14.1 percentage points.
  - It meets the registered rule: at least +8 pooled, positive in both seeds (+3 and +7), and positive after leaving
    out any one database (minimum +8).
  - V1 - D0 is also +10 and meets the same rule: +2 and +8 by seed, minimum +7 leaving out one database, interval
    [+3, +18], about +2.3 to +14.1 points.
  - In PR #71's population, D1 - D0 was +2. The difference between populations is not explained here.
  - These are secondary comparisons, and they do not change the primary decision.

## The six requests where D1 and V1 disagree on correctness

These are described from the saved records only: the initial SQL, each revision, the selection and the shown payload.
They are not a policy or a causal account.

| request | D0 / D1 / V1 | what the saved records show |
|---|---|---|
| bird23tf-01144, seed 200 (professional_basketball) | 0 / 1 / 0 | The initial SQL failed: `T3.award` does not exist. D1 dropped that condition and executed. V1 kept it, and changed `draftFrom = 'California'` to `LIKE 'California%'` after the payload listed 'California ...' values. Its revision failed, so V1 abstained. The payload did not mention the missing column. |
| bird23tf-00441, seed 200 (talkingdata) | 1 / 1 / 0 | The initial SQL was correct. D1 added `DISTINCT`. V1 replaced `events` with `events_relevant`, the first table the payload listed as matching the question, which returned 0 rows. This is the one initially correct answer V1 broke. |
| bird23tf-03981, seed 200 (talkingdata) | 0 / 0 / 1 | The initial SQL failed (`T1.device_id` does not exist). D1 resubmitted it unchanged and abstained. V1 joined `app_events` to `gender_age` through `events_relevant`, which appeared in the payload only in its declared keys (`events_relevant.device_id -> gender_age.device_id`), and executed. |
| bird23tf-01645, seed 201 (simpson_episodes) | 0 / 1 / 0 | The initial SQL failed (`T1.name` does not exist). D1 used `Award.person` and executed. V1 kept `T1.name`, so its revision failed and V1 abstained. The payload did not mention the missing column. |
| bird23tf-02677, seed 201 (ice_hockey_draft) | 0 / 0 / 1 | The initial SQL failed (`height_in_cm` is not in `PlayerInfo`). D1 averaged `PlayerInfo.height`, which is an id. V1 joined `height_info`, which the payload listed with the declared key `PlayerInfo.height -> height_info.height_id`. |
| bird23tf-04913, seed 201 (image_and_language) | 0 / 0 / 1 | The initial SQL failed (`OBJ_CLASS` is not in `IMG_OBJ`). D1 resubmitted it unchanged and abstained. V1 joined `OBJ_CLASSES` on the declared key the payload listed. |

**Patterns in the saved records, stated without interpretation:**
- **The three V1-only recoveries.** In each, V1's revision used a table or key that the payload listed under its
  tables or declared keys.
- **The two V1 abstentions D1 avoided.** The initial query named a column that does not exist, and the payload said
  nothing about that column. The lookup records such columns as unresolved but does not render them.
- **The broken correct answer.** V1 swapped in a listed table.

The sample is six requests, so no rate or effect is estimated from them.

## Uncertainty, as pinned

- **Bootstrap.** One shared resampling of the 64 questions, each with both seeds and all policies: 10,000 replicates
  with `random.Random(20261107)`. The 95% percentile intervals are in correct answers of the 128 question-seed pairs;
  one answer is about 0.78 percentage points. The interval treats questions as independent and does not account for
  clustering by database. Leave-one-database-out is a sensitivity check, not a correction for that.
- **Sign-flip.** The exact sign-flip over per-database differences is exploratory: p = 1.000 for V1 - D1 and 0.031
  for each D0 comparison.

## Cost

V1's end-to-end cost was 1,087 s, against D1's 1,051 s (+3.4%). The difference comes from longer prompts (719,542
tokens against 684,238), the lookup (6.2 s) and tool and execution time. Peak memory was 11.2 GB.

The run took 0.461 GPU-reservation hours, against 0.65 projected and the 1-hour allowance. The host load stayed at
or below 3.0 throughout (co-tenancy log). An idle 533 MiB process from another project was on the GPU.

## Independent reconstruction

The independent reviewer wrote its own verifier, `independent/verify_dbe.py` (sha256 d71704da…). It ran 165 checks
with 0 failures. Its outputs and limitations are in `evidence/dbe_v1/independent/`.

**What it checked:**
- **Scores:** every one of the 384 submitted SQL re-executed against its gold on the pinned snapshots, with its own
  EX rule.
- **Lineage:** the shared initial candidate and B4 extraction, D1's and V1's messages, and the prompt prefixes by
  token count. The remaining time equals 180 s minus the initial call, execution and lookup. The revision order
  follows the seed rule, and every selection, fallback and abstention re-derives.
- **The lookup:** a fresh db-evidence-v1.4 lookup equals every recorded package, and no gold is reachable.
- **Statistics:** every paired statistic, including its own bootstrap and sign-flip.
- **Costs:** the cost totals, the single reservation within the 1-hour allowance, and the co-tenancy log.
- **Integrity:** one request per question and seed, nothing dropped, no rerun.
- **This report:** its numbers and statements. One wording point was corrected: in bird23tf-03981, `events_relevant`
  appears only in the payload's declared keys.

**Its stated limitations:**
- model outputs are taken as recorded, not regenerated;
- seed equality rests on the frozen code and the matching prefixes, since seeds are not recorded per call;
- the fresh lookup and the prompt inputs ran from the producer's code in a separate, flagged process;
- timings are checked for consistency only;
- the foreign GPU process's activity between the reservation's start and end is not logged;
- partial overlaps between a payload and a gold result cannot be told apart from legitimate database facts.

## Reporting correction: the public reconstruction (after the review above)

At the owner's request, the public reconstruction (`dbe_report.py`) was completed. Nothing the run used, nothing it
recorded, and no displayed number changed; the output of `python -m scaling.dbe_report` is byte-identical (sha256
b81cbe67…).

**It now checks, before building anything:**
- the frozen pins;
- the population binding;
- a stable run identity:
  - the bundled run, its analysis and both manifests name each other;
  - both manifests record the same clean commit;
  - the ledger's start entry for the run's one reservation names the run directory;
- every raw value, with #73's protections:
  - integer seeds;
  - integer 0/1 correctness;
  - nonnegative integer counts;
  - genuine Booleans;
  - finite nonnegative times.

It requires the analysis's exact key set. It recomputes the GPU reservation and the ledger totals from the bundled
ledger instead of trusting file hashes.

**Verification.** The independent reviewer verified the correction in two labelled rounds, stored in
`evidence/dbe_v1/independent/reporting_correction/`.
- **Round 1:** 195 corruption probes on copies with regenerated checksums. One case was wrongly accepted: a
  consistently renamed run directory. It was fixed by binding the directory to the ledger's start entry.
- **Round 2:** 198 probes, none wrongly accepted, and every earlier refusal kept. The pin probes were refused,
  legitimate round-off at 1e-12 passed, the reconstruction from a relocated clean clone was byte-identical, and 138
  focused tests passed.
- **Remaining judgement calls,** none of which changes a displayed value:
  - a relabel that also rewrites the bundled ledger;
  - the same other commit in both manifests;
  - informational ledger fields;
  - the summary's reservation hours, which differ from the ledger by about 5e-8 h by construction;
  - two undisplayed times.

## What this does and does not show

- **The rule was not met.** The frozen db-evidence-v1.4 payload, added to ordinary revision, did not meet the
  prespecified continuation threshold against ordinary revision on this population. There was no measured net
  accuracy improvement, at a higher cost. db-evidence-v1.4 is closed as a revision aid:
  - the lookup is not tuned against these results;
  - this population is not rerun;
  - no further revision-prompt study is started from this result.
- **The practical baselines stay D0 and D1.** D1 is the better of the two on this population.
- **Not shown:**
  - that database evidence cannot help;
  - that its effect is small in general;
  - anything about the learned TI-TNN specialist architecture. A positive result would have been evidence for
    database-assisted Qwen revision, not for that architecture.
- **Limits.** One population, one model, one lookup version, and two seeds that share questions. The 128 pairs are
  not 128 independent questions.

## Files and reproduction

- `evidence/dbe_v1/`, under SHA256SUMS: the run (requests with every evidence package, candidate and selection, and
  the scores), the analysis, the logs with co-tenancy, the ledger, and the independent reconstruction.
- To recompute from the bundle: `python -m scaling.dbe_report`. It checks the sums, validates the records,
  recomputes with the frozen `dbe_run.analyze` and refuses any disagreement.
