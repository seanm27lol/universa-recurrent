# TI-TNN scaling against Qwen3.8 — development protocol (NOT frozen)

**Status:** development harness and feasibility only. Nothing here is frozen, preregistered or confirmatory. Sixteen
development questions and a tiny paired smoke run establish whether the pipeline works and what it costs. They do not
establish superiority over Qwen, an architectural advantage, calibration guarantees or confirmation.

## The question, by example

A request from the development split reads, in effect:

> *Question:* "Who is the director of the movie Sex, Drink and Bloodshed?"
> *Evidence:* "Sex, Drink and Bloodshed refers to movie title = 'Sex, Drink and Bloodshed';"
> *Database:* `movie_platform` (read-only SQLite), with its schema descriptions and three retrieved training examples.

A correct system returns a query whose full result set equals the gold query's
(`SELECT director_name FROM movies WHERE movie_title = 'Sex, Drink and Bloodshed'`), as judged by BIRD's pinned EX
rule. Other correct programs also count.

**The question this work order prepares to answer:** can TI-TNN's typed recurrent controller and specialist
organisation (T4) improve real semantic program construction over an otherwise identical shared controller (C4)? And
could the resulting system eventually compete with Qwen3.8-27B (Q27) on task quality and total serving cost?

Three claims are kept separate:
1. a useful TI-TNN system built on a pretrained language backbone;
2. an improvement attributable to TI-TNN's specialist architecture (C4 against T4);
3. a better independently trained language-model architecture (not addressed; the from-scratch scaling ladder is not
   authorized).

## Pinned sources (see `contracts.py`)

| Source | Identity |
|---|---|
| Primary competitor | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (BF16, 27.78 B parameters, hybrid linear/full attention) |
| Shared small backbone | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (BF16, 4.66 B parameters) |
| Learning data | `birdsql/bird23-train-filtered` @ `4068469807b255fcfc0816bdd520946fe460d256` (6,601 questions, 69 databases) and the official BIRD `train.zip` databases |
| Reserved public benchmark (pinned only) | `birdsql/bird_mini_dev` @ `f65faf4ae3b638c1fa6df1d3370c8d92c8366301`, the original 500 SELECT-only SQLite questions, and `minidev.zip` database hashes |
| Official scorer | `bird-bench/mini_dev` @ `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`, `evaluation/evaluation_ex.py` (EX: set equality of full result sets) |

**The reserved 500 questions are never evaluated in this work order.** Their gold SQL is never used to design the
representation or the training targets. Only their ids, database ids and file hashes are recorded.

## Data and split (fixed before any model inference)

- **Learning** uses only the official training release.
- **About 20% of its databases** are held out for development, by a deterministic hash rule fixed before inference:
  order databases by `sha256("ti-tnn-scaling-dev-split-v1:" + db_id)`; the first `round(0.2 · 69) = 14` are
  development. The split is by database, never by question.
- **The development inventory** holds up to 512 questions:
  - questions are ordered per database by a salted hash;
  - databases take one question each per round;
  - questions whose gold SQL does not execute within the 30 s timeout are excluded.
- **Nested subsets** of 16 and 64 questions are prefixes of the inventory's ordering.
- **The exact identifiers and hashes** are in `pins/`.

**The split as computed** (`pins/split_v1.json`):

- **Development** (14 databases, 1,530 questions): `public_review_platform` (256), `soccer_2016` (190), `hockey` (156), `image_and_language` (95), `simpson_episodes` (164), `world` (73), `ice_hockey_draft` (69), `professional_basketball` (113), `music_platform_2` (40), `software_company` (56), `disney` (82), `talkingdata` (118), `computer_student` (53), `food_inspection` (65).
- **Training pool** (55 databases, 5,071 questions).

**The inventory** (`pins/inventory_v1.json`):
- **Gold executability:** all 1,530 development questions were checked (read-only executor, 30 s timeout, SQLite 3.45.1); 1 excluded: `bird23tf-03541` (timeout: exceeded 30 s).
- **The 16-question subset** covers all 14 development databases (two each from `public_review_platform` and `soccer_2016`).
- **The 64-question subset** has 4–5 per database; the 512 have 36–37 per database.

## Arms

| Arm | System | Adaptation |
|---|---|---|
| **L** | Retrieval plus deterministic query-template filling, from permitted training material only | none (no fit) |
| **Q27** | Qwen3.8-27B with the shared tools and at most one repair | none |
| **B4** | Qwen3.5-4B with the same tools and repair budget | none (initially) |
| **C4** | Frozen 4B backbone plus one recurrent semantic controller (shared semantic MLP), the relational IR, masks and executor | supervised controller only |
| **T4** | Identical to C4, with the shared semantic module replaced by a typed tree of four specialists, one executed per step | supervised controller only |

**The shared inputs.** Every arm receives the same question, schema descriptions, benchmark evidence, retrieved training
examples (k = 3) and read-only database access.

**What no arm receives:** gold SQL, reference answers or evaluator-only metadata.

**The principal architectural comparison is C4 against T4.** B4 has no supervised adaptation. So T4 beating B4 would not,
by itself, separate adaptation from structured execution.

## Development limits (written into every run configuration before inference)

| Limit | Value |
|---|---|
| Context tokens per request | 32,768 |
| New tokens per assistant turn | 4,096 thinking / 1,024 non-thinking |
| Generated tokens per request (all turns) | 16,384 thinking / 4,096 non-thinking |
| Tool calls per request | 8 (`describe_schema`, `lookup_values`, `execute_select`) |
| Repairs | at most 1 for Q27 and B4, none for C4 and T4 |
| Request timeout | 600 s (4B), 1,800 s (27B) |
| SQL timeout (tools and scorer) | 30 s, the official `meta_time_out` |
| Rows shown to a model | 20; never the rows that are scored |
| Retrieved training examples | 3 |

**These are development choices, not a final comparison protocol.** The model cards recommend far larger reasoning
budgets (up to 262,144 reasoning tokens for the 27B). Those cannot be met within this work order's GPU allowance, so
any truncation is recorded and reported.

## Generation settings (from the official model cards)

| Model, mode | temperature | top_p | top_k | min_p | presence penalty |
|---|---|---|---|---|---|
| Q27 thinking (`reasoning_effort` xhigh, medium or low) | 1.0 | 0.95 | 20 | 0 | 0.0 |
| Q27 non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |
| B4 thinking (precise coding) | 0.6 | 0.95 | 20 | 0 | 0.0 |
| B4 non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |

Repetition penalty is 1.0 throughout, and the precision is BF16. Each request is seeded deterministically, and the seed
is recorded.

## Outcomes and costs

**Primary quality** is correctly completed requests divided by all requested tasks. Unsupported operations, invalid
plans, abstentions and timeouts stay in the denominator. Outcome correctness is primary; exact SQL or plan matching is a
diagnostic.

**The primary serving-cost unit**, declared here before any run, is:
- the total end-to-end elapsed seconds across all requests;
- divided by the correctly completed requests;
- at batch size one, on this host, under a fixed warm-cache policy: the model loaded, one warm-up request excluded,
  each pinned database read once before timing, and the retrieval index built.

CPU-side request work is included. Cold start, throughput, memory, preprocessing and training are reported separately.
This measure will not be switched after seeing which metric improves.

**Every request preserves:**
- the input and configuration identity;
- the proposed plan or SQL, every tool call and observation, every repair and failed attempt;
- the final output or abstention, and the scorer's result;
- token counts, including reasoning;
- the controller and specialist work;
- elapsed time, peak memory and timeout status.

## GPU budget and run order

**Budget.** At most 8 aggregate GPU-reservation hours, across all agents and attempts. Model loading, profiling, failed
runs, baseline inference and smoke training all count. The ledger (`ledger.py`) enforces one GPU owner and the cap.

**Run order:**
1. CPU-side data, representation and scorer checks.
2. Short 4B and Q27 inference profiles, run serially.
3. L, B4 and Q27 on the fixed 16 development questions.
4. Independent reconstruction of those outputs.
5. One optional paired C4/T4 smoke, only if both arms fit the remaining budget:
   - seed 100;
   - at most 128 deterministically chosen training questions;
   - at most 100 optimizer updates per arm;
   - identical examples and recipe;
   - the same 16 development questions;
   - save, reload and reconstruct.

**The planned allocation** (written before the first reservation; the ledger records what was actually spent):

| Step | Planned GPU hours |
|---|---|
| 4B profile (load, prefill and decode at 1k/4k/12k-token prompts, memory) | ≤ 0.4 |
| 27B profile (same) | ≤ 1.0 |
| Mode probes: thinking against non-thinking (and the 27B's reasoning effort) on 4 development questions **outside** the 16 (inventory positions 17–20), so the 16 do not choose the configuration | ≤ 1.0 |
| B4 and Q27 on the 16 development questions, in the configurations the probes select | ≤ 2.5 |
| Paired C4/T4 smoke (encoding, 2 × ≤ 100 updates, evaluation, save/reload) | ≤ 1.0 |
| Margin for failures and reloads | ≥ 2.1 |

**Choosing each model's mode from the probes** (written 2026-10-29, after the B4 probes and before any Q27 probe
result was read):
1. The mode must fit: its projected GPU hours for the 16 questions, from the probe's measured per-request time,
   must leave room for the remaining planned steps under the 8-hour cap.
2. Among the modes that fit, take the one with more correct probe answers.
3. On a tie, take the one with the lower total elapsed seconds over the probe requests (the cost unit's numerator).

Four questions cannot separate the modes statistically. This rule only makes the choice mechanical and stops the 16
from choosing it. For Q27, thinking is probed at `reasoning_effort` low. At the measured 3.7–4.0 decode tokens per
second, one full 4,096-token thinking turn takes about 1,050 s, so the default xhigh would exceed the 1,800 s request
timeout within two turns. It is not probed. B4 has no effort option.

**As run** (2026-10-26 to 2026-10-29; results in `REPORT_dev.md`):
- **Modes chosen by the rule:** B4 thinking (a tie at 2 of 4 correct; 198.9 s against 229.0 s) and Q27 thinking at
  effort low (4 of 4 against 3 of 4).
- **Harness changes made during the probes, before any run on the 16:**
  - one `finalize` turn after the tool budget, then a stop;
  - a reply without a final query uses the one repair;
  - harness refusals are recorded as `not_run`;
  - the untimed warm-up request stops after 256 generated tokens.
- **The smoke** paired T4 with C4_params, the parameter-matched shared controller. Each final checkpoint was
  reloaded from disk and evaluated with a fresh backbone forward per request, then decoded again from a second reload
  (the reconstruction).
- **GPU:** 2.861 of 8 hours, every reservation in the ledger, including one stopped by the ledger.

**Before each run:**
- commit the producer code, and run from a clean worktree at that commit;
- bind source, configuration, model, data and snapshot identities in the manifest;
- the independent reviewer has checked the relevant code;
- producer files stay unchanged until the run finishes.

**Not authorized in this work order:**
- merging;
- paid APIs or cloud rentals;
- the full two-seed learning pilot;
- any evaluation on the reserved public benchmark;
- confirmation runs;
- larger training allocations or from-scratch pretraining.
