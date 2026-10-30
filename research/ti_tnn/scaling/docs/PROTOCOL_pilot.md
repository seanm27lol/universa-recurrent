# C4/T4 learning pilot: protocol

**Status.** This file is frozen when its sha256 is committed in `pins/pilot_protocol_v1.json`, after the independent
review of the protocol and the pilot code. Every pilot command refuses to run if the file's hash differs from the
pin. Its draft is `PROTOCOL_pilot_draft.md`. The user's decisions of 2026-10-30 are below. Everything here is fixed
before any pilot GPU hour.

## Decisions (user, 2026-10-30)

| # | Decision |
|---|---|
| D1 | **Accept the literal limitation** of the action grammar for this pilot. Every evaluation question stays in every denominator. Gold-program representability is reported separately. The 6.5% (330 of 5,071) is a *training exclusion rate*: questions without a supervisable gold program are left out of training. It is not an evaluation accuracy ceiling |
| D2 | **T4 against C4_params only.** The B4 and Q27 evaluations on the 64 are deferred |
| D3 | **The absolute 8-hour cumulative GPU cap stays**, including the 2.979 h already used and every further check |
| D4 | The decision rules are revised as below |

## The question, by example

Take a development question from a database neither arm trained on: `bird23tf-04097` (`soccer_2016`), *"In which
country do the majority of the players are from?"*, with the evidence *"country refers to Country_Name; the majority of
the players refers to max(count(Country_Name))"*. The frozen Qwen3.5-4B encodes it. The controller then builds a tirel-1
plan, action by action. T4 does so with four typed specialists (schema, predicate, aggregate, projection),
executing only the specialist for each step's family. C4_params does so with one shared module, matched to T4's
17,016,548 trainable parameters. The compiled SQL is scored by BIRD's EX rule (Li et al., NeurIPS 2023).

**The pilot asks two things:**
1. Does each controller beat the no-fit control L after training on the full supervisable training pool?
2. How do the two controllers compare, paired question by question?

It estimates; it does not confirm. Development gold informed the grammar's design, so its results are development
evidence.

## Fixed in advance

| Item | Value |
|---|---|
| Arms | T4 and C4_params (`controller.py`): same trunk, recipe, examples and order; only the semantic module differs |
| Seeds | 100 and 101. Within a seed both arms share the trunk initialization and the example order |
| Training set | `pins/pilot_train_v1.json`: all 4,741 supervisable training-pool questions of 5,071, in release order, under `tirel-1/actions-v2` and the pinned tokenizer. Each training run's manifest binds the pin's sha256 and the ordered ids' hash. Both arms of a seed train in one process from one prepared list |
| Encodings | written fresh into the pilot cache (`encode.PILOT_CACHE`); no legacy entry is used. Each entry records its states' sha256 in the file, and the encoding run writes the same digest into its `entries.jsonl` (the anchor, fsynced line by line). Training is encoded in four reservations, the chunks 0:1200, 1200:2400, 2400:3600 and 3600:4741 of the pin, plus one reservation for the 64. Anchors are accepted only from encoding runs made on a clean tree under this frozen protocol. An entry that exists without an accepted anchor (a crash between the write and the append) is recomputed with a fresh forward and anchored only if equal. A half-written last anchor line is ignored and reported, and its entry is recomputed by the next encoding run |
| Preparation | on the CPU, before any reservation, from anchored entries only (`Backbone.cache_reader`, `require_hit`, `require_verified`, `anchors`). Each entry's bytes must hash to the recorded sha256, which must equal the anchor, and the dtype (BF16) and full shape must match. Every later read is checked against a CRC32 taken at preparation, and again for dtype and shape. A miss, a missing hash or a mismatch stops the pilot (`CacheMiss`, `CacheUnverified`, `CacheCorrupt`) |
| Recipe | AdamW, lr 3 × 10⁻⁴, weight decay 0.01, clip 1.0, effective batch 32 in micro-batches of 8 grouped by length, 1,500 updates (about 10 epochs), loss weights 1/1/1, `matmul_precision` "high" (TF32), states read from the cache per update. The work order fixed the optimizer settings; the update count was chosen from the budget. No setting was tuned on any controller development result (the only one is the smoke's 0/16) |
| Checkpoints | every 250 updates, for resumption. A resumed run must match the checkpoint's recorded configuration field for field (arm, seed, ordered ids, backbone identity, precision, controller, source-file hashes) or it is refused (`CheckpointMismatch`). **Only the final checkpoint (update 1,499) is evaluated**, and `pilot-eval` refuses any other: the training log must hold updates 0–1,499 exactly once, and the checkpoint must match its recorded digests |
| Evaluation | the first 64 questions of `pins/inventory_v1.json` (14 databases; the 16 are its prefix). Greedy legal decoding, at most 256 steps, at batch size one, at the training precision, with a fresh backbone forward per request (timed). Then a reconstruction from a second reload of the checkpoint, on the anchored cache entries |
| Integrity at evaluation | per request: the fresh states equal the anchored states (sha256), and they are identical across the two arms of the seed (the pairing). The reconstruction gives the same actions and final output. The records validate and summarize. Any failure stops the pilot, and the report says so |
| Cross-process determinism, checked before training | the evaluation gate needs the 4B's BF16 forward to be bit-identical across processes. `pilot-encode --what dev` therefore recomputes the feasibility run's 8 development contexts in its own new process, before any training, and compares them with their anchors. A difference stops the pilot then; the gate is redesigned and reviewed, and no training hour is spent. The feasibility run also reports its fresh states against the smoke's recorded ones from another process (2026-10-29) |
| Which evaluations count | exactly one evaluation per seed: the one that passed every integrity check above. An evaluation is rerun only after a stop has been fixed and reviewed. Every evaluation run, passing or not, is listed in the analysis. `pilot-analyze` refuses a seed with no passing evaluation or with more than one, and refuses anything but exactly one L run on the 64. That run must be made from a commit that contains the freeze, and must pass `verify` |
| L | `baseline --arm L --subset 64`, on the CPU, on the same 64. It has not been run on positions 16–63 before this protocol |
| Scoring | the pinned EX scorer. Every request is in the denominator: abstentions, dead ends and invalid plans count as wrong |
| Reported alongside | gold-program representability of the 64 (whether each gold program converts and traces with its literals in the question/evidence tokens, or why not); the arms' correct answers on unrepresentable questions; the training exclusion rate; the cost unit (total end-to-end seconds / correct requests at batch size one, warm cache); training losses; abstention reasons; heads executed |

## Outcomes and how they are read

Write c(arm, s) for the correct answers out of 64 for an arm and seed s, and c(L) for L's (L is deterministic).

**Gate G, per arm.** An arm *cleared L in both seeds* if c(arm, s) > c(L) for s = 100 and s = 101. Clearing L shows
that training lifted the arm above the no-fit control on these questions. It is not a comparison of initializations.

**The paired architectural results are always reported**, whatever G's outcome, including when only one arm or
neither cleared L:
- the per-seed differences d_s = c(T4, s) − c(C4_params, s), and the pooled difference D = d_100 + d_101 (out of 128
  question-seed pairs);
- the per-question histogram of (T4 correct − C4_params correct), summed over seeds;
- the per-database table, with each database's questions, the four counts, L's count and the pooled difference.

**The pre-set rule**, applied to every outcome and reported with each arm's gate status:
- **A1, T4 ahead in this pilot:** D ≥ +7 and d_s > 0 in both seeds. A lead of 7 of 128 keeps the original 5-point
  target (7/128 = 5.5%).
- **A2, C4_params ahead in this pilot:** the mirror image, D ≤ −7 and d_s < 0 in both seeds.
- **A3, no pilot difference by the pre-set rule:** otherwise.

**How the combinations are read:**
- neither arm cleared L: the training or the representation needs revision before any scaling; the rule outcome is
  reported but carries no architectural conclusion;
- exactly one arm cleared L: the finding is that asymmetry, that one arm cleared the no-fit control and the other
  did not. The paired results and the rule outcome are reported in full, descriptively; they are not read as an
  architectural result between two arms that both learned;
- both cleared L: the rule outcome is the pilot's architectural result.

An A1 or A2 outcome motivates a frozen confirmation on data the design never touched. It does not establish one.

**Exploratory database-block sign-flip analysis** of D. It is reported beside the rule, never as a gate.
- **The blocks.** Each of the 14 development databases is one block. Its value is the pooled difference over its
  questions and both seeds, so both seeds and all of a database's questions stay together.
- **The statistic.** Every assignment of signs to the blocks with a nonzero value is enumerated (at most 2¹⁴). The
  two-sided p-value is the share of assignments whose signed sum is at least |D| in absolute value. The smallest
  attainable p-value is reported with it.
- **The assumptions.**
  - The databases are independent units.
  - The null hypothesis concerns these four fitted models, not the architecture: their per-database pooled
    differences are sign-exchangeable. Swapping the arm labels for all of a block's questions and both seeds
    together leaves the joint distribution of the block values unchanged.
- **What it leaves out.**
  - It conditions on the four trained models. Training randomness is shared by every database, and there are only
    two seeds, so it captures between-database variation, not training-seed variation, and it can understate the
    uncertainty.
  - It is exploratory. No false-positive rate is claimed for it or for the rule.

## The resource-feasibility check, and when the pilot stops

**The check.** Before the remaining GPU budget is committed, `pilot-feasibility` runs on the final, frozen code. It
is charged to the ledger, and every entry it encodes is anchored and reused. It measures:
- encoding cost per context, CPU work included, on 160 training contexts at evenly spaced quantiles of context
  length, plus 8 of the evaluation questions;
- preparation from the anchored entries;
- both arms' updates (40 each, measurement seed 999) with file-backed, integrity-checked reads, the first epoch read
  from the disk (the files' cached pages are dropped first);
- checkpoint saves (every 10 updates), and each fit's first update and fixed cost;
- timed evaluation and reconstruction per request (8 requests per arm, unscored).

The measurements are written to disk before the projection is computed.

The check never scores a development question, and its models are discarded.

**The projection.** From those measurements, `pilot-feasibility` projects the complete pilot:
- the remaining encodings and their backbone loads;
- four fits of 1,500 updates, with their checkpoint saves;
- the evaluation and reconstruction of four final checkpoints.

It compares that with the hours left under the 8-hour cap.

**The stop rule.** If the projection does not fit, the pilot stops and reports the measured shortfall. It never
shortens an arm, selects an earlier checkpoint, or raises the cap. "Fits" means the point projection is at most the
hours left. No margin is required beyond that; the slack is reported. A later overrun (a straggler, a resume) is
bounded by the cap and stops the pilot under the stop rules below.

**Other stops:**
- **A reservation's limit.** A reservation that reaches its limit, min(8 h, used + 1.5 × projected), stops. Its
  partial results are reported, never discarded. A training run is resumed from its last checkpoint only under the
  binding above, and the ledger charges every attempt.
- **An integrity failure** of any kind stops the pilot, and the report says so.
- **A defect** found in a pilot command stops the pilot until it is fixed and reviewed. Runs already done stay in the
  ledger and the report.

## Run order

1. `pilot-feasibility`, then the projection decides.
2. `pilot-encode --what dev` (with the cross-process recheck), then `pilot-encode --what train --range a:b` for the
   four chunks.
3. `pilot-train --seed 100`, then `--seed 101`: one reservation per arm.
4. `pilot-eval --seed 100`, then `--seed 101`.
5. `baseline --arm L --subset 64` (CPU).
6. `pilot-analyze`.
7. `verify`, then the independent reconstruction of every count, difference, rule outcome and the sign-flip
   analysis from the records.

**The estimate before the check:** encoding about 2.7 h, training about 0.9 h, evaluation about 0.1 h, loads about
0.1 h, against 5.02 h left. The feasibility projection, not this estimate, decides.

## What it can't establish

Confirmation; held-out generalization; superiority over Qwen, which is deferred; any reserved-benchmark result; and
any guarantee from the exploratory analysis.

## Later, not part of this pilot

A confirmation needs evaluation data the design never touched. The 14 development databases informed the grammar.
The options are:
- authorize the reserved BIRD Mini-Dev 500 for that confirmation only;
- or hold out new training-pool databases before any further design work, which reduces the training set.
