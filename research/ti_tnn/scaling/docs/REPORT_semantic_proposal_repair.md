# Semantic proposal and typed repair: results (development)

This study ran the frozen protocol [`PROTOCOL_semantic_proposal_repair.md`](PROTOCOL_semantic_proposal_repair.md):
sha256 `e5cd55db…`, frozen at 6e1003f, run from 2f76bb7.

The pinned Qwen3.5-4B, with the working direct-SQL prompt, proposed SQL once per question and sampling seed, and that
proposal was shared by three policies:

| Policy | Behavior |
|---|---|
| D0 | submits the proposal as it is |
| D1 | revises it once with the raw execution observation |
| TI1 | revises it once with the same observation plus a deterministic typed analysis: the SQL parsed into the project's typed IR against the exact database snapshot, definite errors apart from observational warnings |

The run covered 64 new development questions (inventory positions 192-255) at sampling seeds 200 and 201, so 128
requests per policy.

This is a development comparison in the same database collection as the earlier studies, not a confirmation. The
reserved BIRD Mini-Dev 500 was not read, nothing was merged, and no claim against Qwen3.8 is made: it was not run on
this population. The branch is built on PR #70, which is not yet merged, and this PR is kept separate.

## The decision

**Neither revision helps by the rule: retain direct Qwen and stop this repair direction.**

EX-correct of 64 per seed, every request in the denominator:

| policy | seed 200 | seed 201 | pooled (of 128) |
|---|---|---|---|
| D0 | 37 | 37 | 74 |
| D1 | 38 | 38 | 76 |
| TI1 | 38 | 40 | 78 |

| comparison | per seed | pooled | leaving one database out | recoveries | regressions |
|---|---|---|---|---|---|
| **TI1 - D1 (primary)** | 0, +2 | **+2** | 0 .. +3 | 4 | 2 |
| D1 - D0 | +1, +1 | +2 | -2 .. +4 | 6 | 4 |
| TI1 - D0 | +1, +3 | +4 | +1 .. +4 | 4 | 0 |

**The rule.** A useful lead needs at least +8 of 128, positive in both seeds and still positive leaving any one
database out. TI1 - D1 is +2: positive in one seed, zero in the other. Neither revision meets the rule against D0
either.

**The narrower engineering effect, reported as such:**
- 18 of the 128 initial proposals failed to execute; 16 of those named identifiers the database does not have.
- Both revisions turned most of those into executable queries. The final queries' statuses:
  - D0: 18 errors;
  - D1: 8 abstentions, 120 executed;
  - TI1: 7 abstentions, 121 executed.
- That recovered EX for only 2 of the 18 under either policy.
- TI1 regressed no answer that D0 had right; D1 regressed 4 and recovered 6. On these counts, TI1 looks the more
  conservative revision.

None of this is a capability gain by the rule. A non-erroring query is not a correct one.

## What was kept apart

| question | what answers it | here |
|---|---|---|
| syntax and type validity | the typed analysis | the initial proposals: 109 of 128 typed; 18 with definite errors, 13 with warnings |
| execution success | read-only execution | above |
| agreement with the benchmark answer | EX, the pinned scorer | the decision above |
| proven semantic equivalence | nothing here | never claimed |
| faithfulness of the execution record | the recorded runs, independently re-executed | the independent verification below |

## The preservation test (before any new model call)

The typed path took the 64 saved direct-Qwen outputs of the grounding-interface study, with no model call:
- **Supported:** 47 parse into the IR. All 47 keep their EX outcome when compiled and executed (32 correct either way).
  46 give identical result sets and ordered rows; one times out both ways.
- **Unexplained changes:** zero.
- **Unsupported:** 17 are reported apart (16 identifier errors in the model's SQL, 1 `YEAR` function).

This shows that the typed path preserves answers. It cannot improve semantics by itself.

## Lineage and costs

Recomputed from the bundled records and equal to analysis.json (seeds [200, 201]).

| policy | seed 200 (of 64) | seed 201 (of 64) | pooled (of 128) |
|---|---|---|---|
| D0 | 37 | 37 | 74 |
| D1 | 38 | 38 | 76 |
| TI1 | 38 | 40 | 78 |

| comparison | seed 200 | seed 201 | pooled | leaving one database out (min .. max) | recoveries | regressions |
|---|---|---|---|---|---|---|
| TI1-D1 | +0 | +2 | +2 | +0 .. +3 | 4 | 2 |
| D1-D0 | +1 | +1 | +2 | -2 .. +4 | 6 | 4 |
| TI1-D0 | +1 | +3 | +4 | +1 .. +4 | 4 | 0 |

Decision (the protocol's rule): neither revision helps by the rule: retain direct Qwen and stop this repair direction.

| database | questions | D0, both seeds | D1, both seeds | TI1, both seeds |
|---|---|---|---|---|
| computer_student | 5 | 5 | 5 | 5 |
| disney | 5 | 4 | 4 | 4 |
| food_inspection | 5 | 7 | 6 | 7 |
| hockey | 5 | 4 | 4 | 4 |
| ice_hockey_draft | 4 | 4 | 5 | 5 |
| image_and_language | 5 | 5 | 5 | 5 |
| music_platform_2 | 4 | 8 | 8 | 8 |
| professional_basketball | 4 | 5 | 5 | 5 |
| public_review_platform | 5 | 4 | 5 | 4 |
| simpson_episodes | 4 | 4 | 8 | 7 |
| soccer_2016 | 5 | 7 | 6 | 7 |
| software_company | 4 | 6 | 6 | 6 |
| talkingdata | 5 | 7 | 5 | 7 |
| world | 4 | 4 | 4 | 4 |

Initial proposals (shared): 128 requests; no SQL 0; truncated 0; typed 109; untyped reasons {'ambiguous column': 2, 'row value': 1, 'unknown column': 11, 'unresolved column': 5}; execution {'error': 18, 'ok': 110}; with typed errors 18; with warnings 13.

| repair policy | final: revised / original / abstain | revision rejected | revision unchanged SQL | revision truncated | revised typed |
|---|---|---|---|---|---|
| D1 | 116 / 4 / 8 | 12 | 94 | 0 | 117 |
| TI1 | 120 / 1 / 7 | 8 | 98 | 0 | 121 |

| policy | final execution status | generation s | validation s | tool s | submitted execution s | total s | per request s | per EX-correct s | prompt tokens | generated tokens | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D0 | {'error': 18, 'ok': 110} | 516 | 0.0 | 0.0 | 8.7 | 525 | 4.10 | 7.1 | 335,824 | 6,836 | 11.1 |
| D1 | {'abstain': 8, 'ok': 120} | 1039 | 0.0 | 49.8 | 8.7 | 1098 | 8.58 | 14.4 | 690,144 | 13,665 | 11.1 |
| TI1 | {'abstain': 7, 'ok': 121} | 1048 | 0.1 | 56.2 | 15.0 | 1119 | 8.75 | 14.4 | 708,372 | 13,680 | 11.2 |

| GPU reservation (this work order) | hours | status |
|---|---|---|
| spr dev (20261106T150737-7404232e) | 0.117 | ok |
| spr dev (20261106T151520-4ba102ec) | 0.070 | ok |
| spr run (20261106T204616-df38ecaa) | 0.480 | ok |
| total | 0.667 | |

Ledger at the analysis: 23.490 of 32.0 h.

**Timing boundary:**
- Generation is every model call.
- Validation is TI1's typed analysis of the initial proposal; it is the only extra work TI1 does.
- Tool time is the read-only execution of the initial and revised proposals.
- Submitted execution is the scorer's execution of the final query.

The frozen run's reservation was quiet: no CPU-heavy job overlapped it (see the co-tenancy log in the bundle). The
two dev debugging runs overlapped the grounding-interface reviewer's CPU work, so their times are development only.

## Development (before the freeze)

Two runs on 16 of the 128 earlier dev questions, at debugging seed 900, chose the revision instruction on truncation:
- the first instruction let 4 of 32 revisions argue past the 512-token ceiling;
- the second asks for the query alone.

Their EX was not read.

## Software correctness

- **Tests:** the scaling suite at 1289d7e: 390 passed, 8 skipped (they need the downloaded BIRD data).
- **The study's tests** (`tests/test_ti_tnn_scaling_spr.py`) cover:
  - the selection rule;
  - the shared revision seed;
  - the bounded raw observation;
  - the typed analysis: a one-source ON is a warning; a real self-join gets none; an unknown column is an error; an
    empty result and an absent literal are warnings; a construct outside the IR is an untyped warning, never a
    rejection.

## Independent verification

⟨REVIEW⟩

## What this does not establish

- **Not a statement about structured feedback in general.** One feedback format, one model, one revision, two seeds.
- **Not semantic correctness.** A query that parses, validates and executes is not called correct.
- **Not a comparison with Qwen3.8** or with the thinking-mode baselines.

## Recommendation

Retain the direct Qwen3.5-4B proposal as the baseline, and stop this repair direction, as the protocol's rule says.
The one observation worth keeping is narrow: TI1 produced no regressions against D0 on 128 requests. It is not a lead
by the rule and should not by itself justify a larger study.

## Evidence and reproduction

- **The bundle:** `evidence/spr_v1/` (weight-free, `SHA256SUMS`), built by `python -m scaling.spr_report bundle`.
- **The tables:** `python -m scaling.spr_report` rebuilds the tables and refuses any disagreement with
  `analysis.json`.
- **The commands:** `python -m scaling.spr_run <command>`.
- **GPU:** this work order used 0.667 GPU-reservation hours of the 6 allotted (two dev runs, 0.187; the frozen run,
  0.480), and the cumulative total is 23.490 of 32.
