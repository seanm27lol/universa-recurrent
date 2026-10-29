# TI-TNN scaling against Qwen3.8: development report (feasibility; not frozen)

**Recommendation: repair, then run the two-seed pilot.** The harness, the baselines and the controller machinery all
work, and they are verified. Two things should change before the pilot's GPU hours are spent:
- the controllers' training throughput;
- a frozen pilot protocol.

The pilot also needs the user's decision on budget (below). Nothing here argues for stopping, and nothing here is
evidence for or against TI-TNN's specialist architecture. Sixteen development questions and a 100-update smoke
establish feasibility only.

## Results in brief

On the first 16 development questions of the BIRD training release (Li et al., "Can LLM Already Serve as A Database
Interface?", NeurIPS 2023), from 14 databases, at batch size one on one DGX Spark:

| Arm | Configuration | Correct | 95% interval (exact) | Elapsed s, all 16 | Seconds per correct request (the cost unit) |
|---|---|---|---|---|---|
| L | retrieval + templates, no fit | 1/16 | 0.002–0.302 | 0.6 | 0.6 |
| B4 | Qwen3.5-4B, thinking, tools, ≤ 1 repair | 11/16 | 0.413–0.890 | 1,617.3 | 147.0 |
| Q27 | Qwen3.8-27B, thinking (effort low), tools, ≤ 1 repair | 13/16 | 0.544–0.960 | 2,314.3 | 178.0 |
| T4 (smoke) | typed specialists, 100 updates on 128 questions | 0/16 | 0–0.206 | 20.2 | undefined (no correct request) |
| C4_params (smoke) | shared controller, parameter-matched, same 100 updates | 0/16 | 0–0.206 | 20.0 | undefined |

- **Q27 against B4.** Q27's 13 include all of B4's 11. The two arms differ on 2 questions, both in Q27's favour, so an
  exact McNemar test gives p = 0.5. The 16 cannot separate them.
- **B4's 4 failures.** All are abstentions: three spent the tool budget without a final query; one was a 4,096-token
  thinking turn truncated at the limit. Q27 abstained on none.
- **The smoke controllers** write executable SQL with plausible tables and joins: 31 of 32 queries ran, and one hit
  the 30 s timeout. All are wrong. That is the expected result of 100 updates on 128 examples, during which the
  training loss fell from 4.64 to 0.23 (memorization). It says nothing about whether either arm can learn the task.
- **Serving cost.** A controller request costs 1.25 s: 1.19 s for the frozen 4B's forward pass over a 3,174-token
  context, and about 0.07 s for decoding. That is roughly 80 times cheaper per request than B4 (101 s mean) and 115
  times cheaper than Q27 (145 s mean) on this harness. It would matter only if the controllers became accurate.
- **GPU use.** 2.861 of the 8 authorized hours, including one ledger-stopped probe and one probe under a harness
  defect that was later fixed.

## What was built (`research/ti_tnn/scaling/`)

| Part | Modules |
|---|---|
| Pins, split, inventory, schema text, read-only tools, the scorer, retrieval | `contracts`, `data`, `schema`, `tools`, `scorer`, `retrieval` (`docs/BENCHMARK.md`) |
| L | `baseline_l` |
| B4 and Q27 | `runtime` (pinned BF16 loading, per-turn generation with time and token accounting), `llm_agent` (tools, the budgets, one repair) |
| The relational representation | `ir` (tirel-1: validator and compiler), `sql_to_ir`, `actions` (the typed action grammar and its masks) |
| C4 and T4 | `encode` (the frozen 4B backbone), `controller`, `train`, `infer_controller` (`docs/ARCHITECTURE.md`) |
| Accounting and evidence | `identity` (manifests), `ledger` (the GPU-hours cap, enforced), `records`, `verify`, `evidence`, `cli` |

There are 239 CPU tests in `tests/test_ti_tnn_scaling_*.py`. The development protocol is `docs/PROTOCOL_dev.md`, and it
is not frozen.

## Identities

| What | Identity |
|---|---|
| Q27 | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, 27,356,728,560 parameters as loaded (`Qwen3_5ForConditionalGeneration`) |
| B4 and the frozen backbone | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16, 4,539,265,536 parameters as loaded |
| Model files | every file checked against `pins/models_v1.json` (sizes and sha256; LFS ids checked against the Hugging Face API and by re-hashing) at each load |
| Learning data | `birdsql/bird23-train-filtered` @ `4068469807b255fcfc0816bdd520946fe460d256` and `train.zip` (8,919,543,554 bytes, sha256 in `pins/snapshots_v1.json`) |
| Reserved benchmark | `birdsql/bird_mini_dev` @ `f65faf4ae3b638c1fa6df1d3370c8d92c8366301`: ids, database ids and file hashes pinned only; never read by any arm |
| Scorer | `bird-bench/mini_dev` @ `abd11b6db92a1c9f809b32f7564c7c71b34d67f0`, `evaluation/evaluation_ex.py` rule, reimplemented read-only (`scorer.py`); agreed with an independent implementation on 260 of 260 cases |
| Environment | Python 3.12.3, torch 2.13.0+cu130, transformers 5.20.2, tokenizers 0.23.3, sqlglot 30.24.0, numpy 2.5.4 (`.venv-scaling`, `requirements-scaling.txt`); SQLite 3.45.1 |
| Linear-attention kernels | the transformers reference functions (`TI_TNN_SCALING_LINEAR_KERNELS=reference`); flash-linear-attention 0.6.0 is installed but not used (see Hardware) |
| Run identity | every run's manifest binds the commit (clean research tree), the sha256 of every `scaling/*.py`, the configuration hash, the model identity, the inventory and snapshot pins |

## Hardware and throughput

**Host.** NVIDIA DGX Spark (GB10, aarch64), CUDA 13.0, driver 580.173.02, 121 GB unified memory (about 100 GB free at
start). One foreign GPU process was present throughout (an unrelated `ui_server.py`, 558 MiB) and was left alone.

**Profiles** (synthetic prompts, 256 decoded tokens, batch size one, BF16, reference kernels):

| Model | Load s | Prompt tokens | Prefill tok/s | Time to first token s | Decode tok/s | Peak allocated GB | Peak reserved GB |
|---|---|---|---|---|---|---|---|
| B4 | 52 | 1,040 | 2,694 | 0.39 | 17.3 | 9.6 | 10.2 |
| | | 4,120 | 2,634 | 1.56 | 17.0 | 10.4 | 10.8 |
| | | 12,320 | 2,605 | 4.73 | 16.0 | 12.7 | 13.7 |
| | | 28,680 | 2,501 | 11.47 | 14.3 | 17.2 | 21.7 |
| Q27 | 292 (cold) | 1,040 | 664 | 1.57 | 4.03 | 55.4 | 56.1 |
| | | 4,120 | 680 | 6.06 | 3.98 | 56.7 | 57.4 |
| | | 12,320 | 602 | 20.46 | 3.87 | 60.3 | 63.1 |
| | | 28,680 | 615 | 46.64 | 3.66 | 67.4 | 75.7 |

The 27B fits in BF16 with room to spare; no quantization was needed, and none was used.

**Kernels.** With flash-linear-attention, the 4B's prefill was erratic: 6.7 s at 4k tokens and 13.5 s at 28.7k, against
1.6 s and 11.5 s with the reference functions, apparently compiling Triton kernels per new shape. Decode was only about
4% faster. Both paths give the same top-20 next tokens on a fixed prompt (total variation 0.014). The reference path is
used for every run.

**What these numbers are not.** They are one host, batch size one, eager transformers `generate` with no prefix caching
across turns: each assistant turn re-reads the whole conversation. A serving stack with prefix caching and batching
would be much faster for both LLM arms. The cost comparison here is between arms on the same harness, not against a
production deployment.

## Operator coverage (tirel-1 and the action grammar)

| Split | Questions | Converted to tirel-1 | Same result as gold after compiling | Encodable as actions | Not encodable |
|---|---|---|---|---|---|
| training pool | 5,071 | 5,064 | 5,062 | 4,741 (93.5% of questions) | 323, all for a literal not found in the question or evidence |
| development | 1,530 | 1,525 | 1,524 | 1,447 (94.6%) | 78, the same reason |

Unsupported constructs: RIGHT JOIN (4), window functions (2), GROUP_CONCAT (2), a reference into a derived table with
`*` (2), `IS` with a non-NULL operand (1), an unknown column (1).

**Two caveats.** First, the grammar's coverage was measured on development gold during design, so development
results are development evidence, not held-out. Second, 11 of the first 139 smoke candidates (8%) could not be
supervised for the literal reason above; they were skipped by the pinned rule, and such questions count as failures in
every evaluation.

## What the probes changed in the harness

The probes ran on inventory positions 16–19 (zero-based), outside the 16, so the 16 never chose a configuration.

1. **A loop after the tool budget.** The first B4 non-thinking probe (2026-10-26) kept calling tools after the
   8-call budget: up to 124 turns per request, until the generated-token limit, for 1 correct out of 4 in 1,582 s.
   Now the harness says once that the budget is spent; the next turn is labelled `finalize`. A tool call in a turn
   that starts with the budget spent ends the request. Records reject more than tool calls + repairs + 1 turns.
   Rerun: 2 of 4 correct in 229 s. The old run is kept and is rejected by both verifiers for exactly this.
2. **A prose answer.** In thinking mode B4 once answered in prose with no query. A reply without a final query now
   uses the one repair, like a failing query.
3. **Tool refusals miscounted.** The reviewer found that a statement the read-only check refuses (for example
   `DELETE`) counted against the budget but not in the record checks. That would have failed a whole run's summary.
   Harness refusals are now `not_run`; a tool's own rejection is a call. No probe had hit it.
4. **The warm-up.** The untimed warm-up request is capped at 256 generated tokens; a full thinking-mode warm-up could
   spend 10 minutes of budget.

**Mode choice**, by the rule written before any Q27 probe result was read (more correct; on a tie, less elapsed time):

| Probe (positions 16–19) | Correct | Elapsed s | Tool calls | Reasoning tokens | Chosen |
|---|---|---|---|---|---|
| B4 non-thinking | 2/4 | 229.0 | 32 | 0 | |
| B4 thinking | 2/4 | 198.9 | 8 | 2,388 | yes (tie, less time) |
| Q27 non-thinking | 3/4 | 414.3 | 11 | 0 | |
| Q27 thinking, effort low | 4/4 | 289.3 | 4 | 426 | yes |

Four questions cannot separate these modes. The rule only makes the choice mechanical.

## Baselines on the 16

Every request is in the denominator. Timing runs from retrieval to the final answer, at batch size one, with a warm
cache: the model loaded, one excluded warm-up request served, the databases re-hashed and read, the retrieval index
built. Scoring is outside the timing.

| Question | Database | L | B4: outcome, s | Q27: outcome, s, reasoning tokens |
|---|---|---|---|---|
| bird23tf-04392 | public_review_platform | 0 | wrong, 68.1 | wrong, 158.6, 262 |
| bird23tf-00788 | soccer_2016 | 0 | abstained (tool budget), 148.6 | correct, 133.8, 242 |
| bird23tf-05697 | hockey | 0 | correct, 45.0 | correct, 59.8, 65 |
| bird23tf-04901 | image_and_language | **1** | correct, 33.7 | correct, 45.8, 53 |
| bird23tf-01627 | simpson_episodes | 0 | correct, 77.8 | correct, 80.8, 124 |
| bird23tf-03130 | world | 0 | correct, 79.8 | correct, 119.3, 148 |
| bird23tf-02636 | ice_hockey_draft | 0 | correct, 169.1 | correct, 164.6, 447 |
| bird23tf-06046 | professional_basketball | 0 | abstained (tool budget), 124.6 | wrong, 291.2, 479 |
| bird23tf-04982 | music_platform_2 | 0 | correct, 34.3 | correct, 64.1, 84 |
| bird23tf-03420 | software_company | 0 | abstained (tool budget), 342.9 | correct, 357.8, 937 |
| bird23tf-01859 | disney | 0 | abstained (thinking turn truncated at 4,096 tokens), 245.4 | wrong, 364.2, 956 |
| bird23tf-00469 | talkingdata | 0 | correct, 56.4 | correct, 51.9, 98 |
| bird23tf-03943 | computer_student | 0 | correct, 60.4 | correct, 152.3, 300 |
| bird23tf-03526 | food_inspection | 0 | correct, 47.1 | correct, 90.9, 130 |
| bird23tf-04345 | public_review_platform | 0 | correct, 26.7 | correct, 99.7, 189 |
| bird23tf-04097 | soccer_2016 | 0 | correct, 57.2 | correct, 79.5, 132 |

| Arm | Tool calls | Repairs | Prefill tokens | Generated | Reasoning | Median / slowest request s | Peak memory GB |
|---|---|---|---|---|---|---|---|
| L | 37 | 0 | — | — | — | 0.02 / 0.2 | — |
| B4 | 47 | 5 | 328,345 | 24,981 | 20,428 | 64.3 / 342.9 | 11.8 |
| Q27 | 28 | 0 | 158,038 | 8,258 | 4,646 | 109.5 / 364.2 | 58.5 |

**Truncation.** One B4 request was truncated: a thinking turn reached the 4,096-token turn limit. No request reached
the per-request token limit, the context limit or the request timeout.

**What the costs are.** These are eager transformers `generate` costs with no prefix cache across turns, so each turn
re-reads the conversation (B4's 16 requests prefilled 328k tokens). They are comparable between arms on this harness
only. B4 is cheaper per correct request than Q27 here (147 s against 178 s) because it is about 4 times faster per
token, not because it is better.

## The paired smoke

Both arms used:
- seed 100;
- the 128 pinned training-pool questions;
- 100 updates of 32 examples (AdamW, lr 3e-4, weight decay 0.01, clip 1.0);
- the same examples in the same order at every update;
- trunks identical at initialization.

| | T4 (typed specialists) | C4_params (shared, parameter-matched) |
|---|---|---|
| Trainable parameters | 17,016,548 | 17,016,548 |
| Training loss, update 0 → 99 | 4.639 → 0.231 | 4.703 → 0.257 |
| Training time (100 updates) | 178.8 s (1.78 s per update) | 178.0 s |
| Dev: correct / answered | 0/16 / 16 | 0/16 / 16 |
| Dev: decode steps per request (mean) | 17.6 | 18.9 |
| Heads executed (dev, all steps) | projection 102, schema 72, predicate 66, aggregate 26 | shared 286 |
| Dev: seconds per request (encode + decode) | 1.26 | 1.25 |
| Checkpoint reloaded from disk, weights digest equal | yes | yes |
| Reconstruction: a second reload, cached states, same actions and SQL on all 16 | yes | yes |

Encoding the 128 training contexts took 250.6 s of backbone forward (1.96 s per context). The training updates are
bound by host-side batching, not by the GPU: the same update took 2.2–2.4 s on the CPU.

## GPU accounting

Every reservation, including the failures, from `gpu_ledger.jsonl` (the watchdog holds each run to
min(8 h, used + 1.5 × projected)):

| Reservation | Command | Status | Projected h | Charged h |
|---|---|---|---|---|
| 20261026T192033 | profile B4 (flash-linear-attention kernels) | ok | 0.3 | 0.045 |
| 20261026T192331 | profile B4 (reference kernels) | ok | 0.3 | 0.038 |
| 20261026T192650 | profile Q27 | ok | 0.5 | 0.177 |
| 20261026T193855 | probe B4 non-thinking | **stopped by the ledger** (projection too low; no request finished) | 0.1 | 0.150 |
| 20261026T194920 | probe B4 non-thinking (the tool-budget loop) | ok | 0.4 | 0.471 |
| 20261029T194301 | probe B4 non-thinking (fixed) | ok | 0.25 | 0.099 |
| 20261029T200113 | probe B4 thinking | ok | 0.7 | 0.075 |
| 20261029T200709 | probe Q27 non-thinking | ok | 0.5 | 0.202 |
| 20261029T202024 | probe Q27 thinking, low | ok | 0.6 | 0.174 |
| 20261029T203121 | B4 thinking on the 16 | ok | 0.5 | 0.470 |
| 20261029T205935 | Q27 thinking, low, on the 16 | ok | 0.7 | 0.747 |
| 20261029T214509 | the paired smoke | ok | 0.4 | 0.213 |
| **Total** | | | | **2.861 of 8** |

Against the plan: profiles 0.26 h (planned ≤ 1.4); probes, including the two harness failures, 1.17 h (≤ 1.0);
the 16, 1.22 h (≤ 2.5); smoke 0.21 h (≤ 1.0). No paid API or cloud resource was used, and no quantization.

## Verification

- **In-package verifier** (`scaling.verify`, `python -m scaling.cli verify`). For each run it checks:
  - the identities: the commit, a clean tree, and each source file at that commit;
  - that the requests match the pinned inventory;
  - the snapshot pins, with the files re-hashed;
  - every outcome, re-derived by re-executing the final SQL;
  - the record rules (hidden retries, turn and tool budgets, token sums, finite numbers);
  - the summary, recomputed;
  - the ledger;
  - for the smoke: the checkpoint hashes, the weight and optimizer digests, the training logs and the
    reconstruction.

  It accepts 9 of the 10 finished runs. It rejects the old-harness probe `20261026T194920Z`, for exactly the defect
  described above (`evidence/dev_v1/verification.json`).
- **Independent verifiers** (`independent/`, the reviewer's). They were written without the producer modules and use
  the raw release, the pins at each run's commit, and their own EX rule. Output is in
  `evidence/dev_v1/independent_verification.txt`.
  - Every baseline run passes: the 16 for L (356 checks), B4 (501) and Q27 (497), and the five current-harness probes
    (110–141 each).
  - The old-harness probe fails, for its 233 turn-bound violations only.
  - The smoke passes 244 checks:
    - the training ids reconstruct from the pins, all from the training pool;
    - the checkpoints re-hash, and their tensor digests are recomputed with the reviewer's own code;
    - the sampling stream is recomputed and is identical across arms;
    - all 32 outputs are re-scored, including the one scorer timeout.
  - **CPU re-decode** from the final checkpoints and the cached GPU states (the one step that uses producer decoding
    code): 16/16 requests per arm identical in every action and the final SQL.
- **Tamper tests.** The reviewer's verifier caught each of 14 tampered copies of a run with the expected check:
  - a flipped score, and a swapped final SQL;
  - a dropped or duplicated record, and reordered ids;
  - a dirty tree, and an edited configuration;
  - a NaN, an edited token sum, and an edited cost;
  - a hidden tool call, and a relabelled finalize turn;
  - a wrong snapshot hash;
  - an error on an answered request.

The evidence bundle `research/ti_tnn/scaling/evidence/dev_v1/` holds, weight-free:
- every run's manifest, summary and records;
- the smoke's configs and training logs;
- the ledger;
- the verification output;
- `SHA256SUMS`.

`python -m scaling.evidence tables` rebuilds this report's tables from it and refuses mismatched bytes.

## Reviews before GPU use

An independent reviewer checked the code before each class of run. Every finding was fixed before the runs it could
affect:
- **The first review (18 findings):**
  - per-worktree ledgers;
  - an unenforced cap;
  - final-SQL fences;
  - weak record checks;
  - missing kernels;
  - deadlines not checked inside tools;
  - lost attempts on exceptions;
  - a missing stop token;
  - unsynchronized time to first token;
  - projection errors;
  - and more.
- **A re-check:** a swallowed stop signal.
- **The review of 641ff3c**, before the smoke:
  - grammar dead ends that crashed decoding;
  - masks that allowed plans SQLite rejects (2,326 of 3,553 random plans, now 1 of 3,571);
  - an encoding cache key without the kernel path;
  - non-deterministic checkpoint bytes;
  - a snapshot-cache bypass.
- **The re-check of 41804b0:** tool refusals miscounted by the record checks.

The scorer (260 of 260 cases against an independent implementation), the tools, the split, the inventory, the leakage
controls and L were checked and found correct.

## What this does not establish

- Anything about the reserved public benchmark, which no arm read.
- Q27 against B4, or either against L, beyond feasibility. The intervals overlap, and n = 16.
- Whether C4 or T4 can learn the task, or whether T4's specialists help. The smoke tests the machinery only.
- Held-out performance. The action grammar's coverage was measured on development gold during design, so all
  development results are development evidence.
- Production serving cost (see Hardware).

## The next pilot: budget (not authorized here)

The pilot this work order leaves for later: two seeds (100, 101) × {T4, C4_params}, trained on all 4,741
supervisable training-pool questions, evaluated on the 64-question development subset. B4 and Q27 run on the same 64
in the selected modes, and their 16 are reused, since the 16 are the 64's prefix. The costs below are the measured
unit costs from this report, not advertised FLOPs.

| Item | From | As measured now | After the throughput repair |
|---|---|---|---|
| Encode the training pool (once, cached) | 4,741 × 1.96 s | 2.6 h | 2.6 h |
| Train 4 runs × 1,500 updates (about 10 epochs) | 6,000 × 1.78 s | 3.0 h | ≤ 0.5 h (target ≤ 0.3 s per update) |
| Evaluate 4 runs on the 64 | 256 × 1.25 s | 0.1 h | 0.1 h |
| B4 thinking on the 48 new questions | 48 × 101 s + load | 1.4 h | 1.4 h |
| Q27 thinking (low) on the 48 new questions | 48 × 145 s + load | 2.0 h | 2.0 h |
| Margin for failures and reloads (25%) | | 2.3 h | 1.7 h |
| **Total** | | **11.4 h** | **8.3 h** |

The encoded states take about 24 MB per context on disk (the mean of the 146 cached here), about 113 GB for the
pool. A third arm, C4_active (the active-compute control), would add about 0.25 h of training after the repair.

## Recommendation: repair

**Why not stop.** Every part of the pipeline works end to end and is verified:
- the pinned data and models;
- the leakage controls;
- the LLM harness with its turn and tool bounds;
- the scorer;
- the controllers' training, checkpointing, reload, reconstruction and serving.

The baselines are strong enough to be meaningful competitors (Q27 13/16, B4 11/16 on development questions). The
controllers' per-request cost is low enough (about 1.25 s) that the cost question is worth asking.

**Why not proceed as is.** Two problems should be fixed first.

1. **Training throughput.** Training is bound by host-side batching at 1.78 s per update. That would make the pilot's
   training cost 3 GPU-hours for work the GPU does in a fraction of that, and it pushes the pilot past 8 hours.
2. **No frozen protocol.** The pilot needs one, written before its first GPU hour:
   - the 64, the seeds and the updates;
   - the limits and the selected modes;
   - the budget;
   - the predictions.

**Two decisions for the user**, not repairs:

1. **The literal ceiling.** About 6% of questions need a literal that is not in the question or evidence. C4 and T4
   cannot produce those literals, while the LLM arms can look values up. The options:
   - give the controllers a value-lookup pointer, which is a design change;
   - or accept the ceiling, with those questions kept in every denominator.
2. **The pilot budget**: about 8.3 GPU-hours after the repair, or 11.4 h without it.

## Reproduction

From a clean checkout of this branch, on an aarch64 or x86-64 host with a CUDA GPU of at least 80 GB (the 27B's peak
reserved memory was 75.7 GB at a 28.7k-token prompt):

```bash
python3.12 -m venv .venv-scaling && .venv-scaling/bin/pip install -r requirements-scaling.txt   # no Conda
export PYTHONPATH=research/ti_tnn PY=.venv-scaling/bin/python
export TI_TNN_SCALING_RUNS=$PWD/runs/ti_tnn_scaling          # one shared, absolute location for the ledger
# place the pinned downloads as docs/BENCHMARK.md lists them (train.zip, the question release, the scorer), then
$PY -m scaling.data all              # extract, hash, split, gold-check, inventory: must reproduce pins/ byte for byte
$PY -m scaling.cli preflight --verify-models
$PY -m scaling.cli inventory         # the split and the development inventory, rebuilt and checked against pins/
$PY -m pytest -q tests/test_ti_tnn_scaling_*.py

# profiles, then the mode probes on positions 16-19, then the 16, then the smoke (each a ledger reservation)
$PY -m scaling.cli profile --model B4 --projected-hours 0.3
$PY -m scaling.cli profile --model Q27 --projected-hours 0.5
$PY -m scaling.cli baseline --arm B4  --positions 16:20 --mode nonthinking --projected-hours 0.25
$PY -m scaling.cli baseline --arm B4  --positions 16:20 --mode thinking --projected-hours 0.7
$PY -m scaling.cli baseline --arm Q27 --positions 16:20 --mode nonthinking --projected-hours 0.5
$PY -m scaling.cli baseline --arm Q27 --positions 16:20 --mode thinking --reasoning-effort low --projected-hours 0.6
$PY -m scaling.cli baseline --arm L   --subset 16
$PY -m scaling.cli baseline --arm B4  --subset 16 --mode thinking --projected-hours 0.5
$PY -m scaling.cli baseline --arm Q27 --subset 16 --mode thinking --reasoning-effort low --projected-hours 0.7
$PY -m scaling.cli smoke --projected-hours 0.5

$PY -m scaling.cli verify runs/ti_tnn_scaling/baselines/* runs/ti_tnn_scaling/smoke/*
$PY -m scaling.evidence tables       # every table in this report, from the committed bundle alone
```

Sampling is seeded per request, but GPU generation in BF16 is not bit-reproducible across hosts or library versions,
so a rerun's texts and scores can differ. The verifiers re-derive every outcome from the recorded SQL, which is
deterministic.
