# C4/T4 learning pilot: DRAFT protocol (not frozen; nothing here runs before your approval)

**Status.** This is a draft for review. It becomes frozen only after you decide the items under "Decisions" and
approve, and after the independent reviewer has checked it and the pilot code. A frozen copy is then committed with
its sha256, before any pilot GPU hour. Until then nothing in it is binding, and no pilot run is authorized.

## Decisions for you

| # | Decision | Options | Recommended |
|---|---|---|---|
| D1 | The literal ceiling: 6.5% of training-pool questions (330 of 5,071) can't be supervised, mostly because a literal is not in the question or evidence | (a) accept it; those questions stay in every denominator. (b) Add a value-lookup pointer to C4/T4 first: a design change, needing another development round | **(a)**. It affects T4 and C4 equally, so the principal comparison is unaffected. It matters only for competing with the LLMs |
| D2 | Scope | (a) **Core**: C4 and T4 only, about 4.7 GPU-hours. (b) **Full**: also B4 and Q27 on the 48 new questions of the 64, about 9.0 GPU-hours | (a) first. The LLM numbers on the 64 answer the cost question, which matters only if the controllers learn; they can be added later on the same pinned questions |
| D3 | The GPU cap | The ledger's cap is 8 h in total, and 2.979 h are used. The core scope (about 4.7 h, below) fits under it, ending near 7.7 h with little room; the full scope needs a raised cap | core: raise it to used + 4.7 h, which is about the same as the 8 h cap but states the pilot's own limit. Full: used + 9.0 h |
| D4 | The decision rules below (L1, L2, A1–A3) | as written, or changed now | as written |

## The question, by example

Take a development question from a database neither arm trained on: `bird23tf-04097` (`soccer_2016`), *"In which
country do the majority of the players are from?"*, with the evidence *"country refers to Country_Name; the majority of
the players refers to max(count(Country_Name))"*. The frozen Qwen3.5-4B encodes it. The controller then builds a tirel-1
plan, action by action. T4 does so with four typed specialists (schema, predicate, aggregate, projection),
executing only the specialist for each step's family. C4_params does so with one shared module, matched to T4's
17,016,548 trainable parameters. The compiled SQL is scored by BIRD's EX rule (Li et al., NeurIPS 2023).

**The pilot asks two things:**
1. **Learning.** Does either controller learn the task from the full supervisable training pool?
2. **Architecture.** If both learn, is T4's typed specialization ahead of the parameter-matched shared controller at
   this size?

The pilot estimates; it does not confirm. Development gold informed the grammar's design, so pilot results are
development evidence.

## Fixed in advance

| Item | Value |
|---|---|
| Arms | T4 and C4_params (`controller.py`); same trunk, recipe, examples and order; only the semantic module differs |
| Seeds | 100 and 101 (development seeds); each seed gives both arms identical trunk initialization and example order |
| Training set | `pins/pilot_train_v1.json`: all 4,741 supervisable training-pool questions (of 5,071) under `tirel-1/actions-v2` and the pinned tokenizer. The 128 smoke questions are among them. Each run's manifest binds the pin's sha256 and its exact id order. Both arms of a seed train from one prepared list |
| Encodings | `encode-pool` writes each context's states once, recording their sha256. The pilot prepares its examples on the CPU, before any reservation, from those entries only: `Backbone.cache_reader` with `require_hit`, so a miss stops the pilot and is never silently re-encoded. Each file's sha256 is checked once per run at preparation, and each later read against a CRC32. The manifest binds `encode-pool`'s backbone identity |
| Recipe | AdamW, lr 3 × 10⁻⁴, weight decay 0.01, clip 1.0, effective batch 32 in micro-batches of 8 grouped by length, 1,500 updates (about 10 epochs), loss weights 1/1/1, `matmul_precision` "high" (TF32; `REPAIR_training_throughput.md`), states read from the encoding cache (`lazy=True`). The work order fixed the optimizer settings. The update count was chosen from the budget, before any controller result except the smoke's 0/16, which no setting was tuned on |
| Checkpoints | every 250 updates, for resumption only. **Only the final checkpoint (update 1,499) is evaluated, and the pilot CLI refuses any other.** No early stopping and no selection on development data |
| Evaluation | the first 64 questions of `pins/inventory_v1.json` (14 databases; the 16 are its prefix). Greedy legal decoding, at most 256 steps, batch size one, at the training precision ("high"), with a fresh backbone forward per request (timed). Then a reconstruction pass from a second reload. Pairing by question requires both arms of a seed to decode from identical states, so each request's `states_sha256` is compared across the arms, and any mismatch is reported and stops the comparison |
| L, the no-fit control | run on the same 64 (CPU, no GPU hours); L1 is judged against it. L has not been run on positions 16–63, so its result could not inform these rules |
| Scoring | the pinned EX scorer, every request in the denominator (abstentions, dead ends and invalid plans count as wrong) |
| Cost unit | as in `PROTOCOL_dev.md`: total end-to-end seconds over all requests / correct requests, at batch size one, warm cache |
| Full scope only | B4 (thinking) and Q27 (thinking, effort low) on inventory positions 16–63, with the harness of the 16 (`llm_agent` as at 92f01cd; d8e31b6 changed only its torch import). **The join** is fixed now. Each part is verified on its own (two runs, two summaries). The 64-question figures are the sums of the two parts' correct counts and elapsed seconds; the cost unit is the summed elapsed over the summed correct. The parts' configurations may differ only in their positions, which the join checks. Positions 16–19 chose the modes, so results are also reported without them |

## Outcomes and how they will be read

**Primary measure.** For each arm and seed, the correct answers out of 64, written c(arm, seed).

**The comparison test.** The two seeds answer the same questions, so the comparison pairs by question. For question
q, let d_q = (T4 correct, summed over seeds) − (C4_params correct, summed over seeds), which is in {−2, …, 2}. The
test is an exact two-sided sign test on the questions with d_q ≠ 0. The counts are always reported, whatever the
p-value.

**Rules, fixed before the run:**
- **L1, learning.** An arm *learned* if c(arm, seed) > c(L), L's correct count on the same 64, in both seeds. (L is
  deterministic; it scored 1/16 on the prefix.)
- **L2, the learning outcomes:**
  - neither arm learned: the pilot says nothing about architecture; the recommendation is to revise the training or
    the representation, not to scale;
  - exactly one arm learned: reported as that arm learning where the other did not. That is evidence about
    learnability, not about quality at matched learning; A1–A3 don't apply;
  - both learned: A1–A3 apply.
- **A1, T4 ahead.** Both learned, and all of the following hold:
  - c(T4, seed) − c(C4_params, seed) > 0 in both seeds;
  - the pooled difference is at least +6;
  - the sign test gives p < 0.05.

  This motivates a frozen confirmation on data the design never touched (that needs a held-out decision, see below).
- **A2, C4_params ahead.** The mirror image of A1.
- **A3, otherwise:** no difference detected at this size. With no true difference, a two-sided exact test at 0.05 makes
  A1 or A2 fire at most 5% of the time, and the seed-consistency requirement lowers that further. A difference of a
  few questions is expected from seed noise alone (development lesson: init-to-init variance has been large in this
  project).

**Reported, not judged:**
- training losses;
- abstention reasons;
- heads executed;
- per-database results;
- the cost unit;
- in full scope, B4 and Q27 quality and cost on the 64.

**What it can't establish:** confirmation, held-out generalization, superiority over Qwen, or any reserved-benchmark
result.

## Run order and budget

Every GPU step runs inside a ledger reservation with an explicit projection. Each run's manifest binds the frozen
protocol's hash, the code commit and the pins. The unit costs are this host's measured ones: `REPORT_dev.md` and
`REPAIR_training_throughput.md`.

| Step | Basis | GPU h |
|---|---|---|
| 1. Encode the 4,741 training contexts (chunked; resumable through the content-keyed cache) | 4,741 × 2.03 s (the smoke's preparation, CPU work included) | 2.67 |
| 2. Encode the 64 development contexts (the untimed cache pass) | 64 × 1.2 s | 0.02 |
| 3. Train 2 seeds × 2 arms × 1,500 updates | 6,000 × 0.52 s | 0.87 |
| 4. Evaluate 4 final checkpoints on the 64, plus reconstruction | 256 × 1.25 s, plus decoding | 0.10 |
| 5. The 4B load in each reservation | about 8 reservations × 1 min | 0.13 |
| (Preparing the 4,741 examples, about 60 s of tokenizing and tracing plus reading the states, runs on the CPU before any reservation) | | 0 |
| **Core subtotal** | | **3.79** |
| 6. Full scope only: B4 on positions 16–63 | 48 × 101 s + load | 1.38 |
| 7. Full scope only: Q27 on positions 16–63 | 48 × 145 s + load | 2.01 |
| **Full subtotal** | | **7.18** |
| Margin for failures, reloads and stragglers (25%) | | core 0.95 / full 1.80 |
| **Total** | | **core 4.7 / full 9.0** |

The update time (0.52 s) was measured on 128 page-cached examples with T4. A cold-read, pilot-sized run and c4_params
after the repair were not measured (in the smoke, c4_params' updates took as long as T4's).

**Disk.** The training states take about 126 GB in the encoding cache; 1.7 TB is free.

**Not yet built** (after approval, reviewed before use):
- `cli encode-pool`: chunked encoding into the cache, recording each entry's sha256;
- `cli pilot`:
  - CPU preparation from the cache with `require_hit`;
  - per seed and arm, training with lazy states and TF32;
  - evaluating the final checkpoint only, and reconstructing it;
  - the cross-arm `states_sha256` check;
- L on the 64;
- the verifier's checks for the pilot's runs, including the rules L1, L2 and A1–A3 recomputed from the records.

**Stop rules.**
- **The reservation cap.** Any reservation that reaches its limit stops. Its partial results are reported, never
  discarded.
- **A defect.** A defect found in a pilot run stops the pilot until it is fixed and reviewed. Runs already done stay
  in the ledger and the report.

## Later, not part of this pilot

A confirmation needs evaluation data the design never touched. The 14 development databases informed the grammar.
The options are:
- authorize the reserved BIRD Mini-Dev 500 for that confirmation only;
- or hold out new training-pool databases before any further design work, which reduces the training set.
