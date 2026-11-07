# Grounding interface: does explicit, task-conditioned grounding help the saved controllers? (development protocol)

**Status.** Frozen when its sha256 is committed in `pins/gi_protocol.json`. Every command on the new population
refuses to run if this file or `pins/gi_new_eval.json` differs from the pin. Everything below is fixed before any
policy runs on the new population. The development population (the performance study's 128 evaluated questions) was
used to build and select it, as the work order allows.

**Dependency.** This study's branch (`ti-tnn-grounding-interface-dev`) is built on the corrected performance-study
branch (PR #69, 4c73eec), before #69 is merged. It is a separate PR. The historical producers and their evidence are
read, never written.

## The question, with an example

Does explicit, task-conditioned grounding improve execution accuracy beyond what the existing controller extracts
from its frozen-backbone features?

Take "Identify by their id all the apps that belong to the game-stress reliever category." A grounding step can say,
before the controller decodes, that the relevant tables are `label_categories` and `app_labels`, and that the
relevant columns are `category`, `app_id` and `label_id`. The controller's own pointers then receive a small fixed
push toward those items. Whether the push helps is the question.

The interpretability work motivates this hypothesis but does not establish it. Nothing here claims that toy-program
findings transfer to Qwen3.5-4B on SQL, that failed probes show absent state, or that a grounding record explains the
pretrained model's computation.

## Fixed in advance

| Item | Value |
|---|---|
| Controllers | the six saved v2 controllers of the performance study (T4-v2 and C4-v2, seeds 100-102, final checkpoints u002999), not retrained |
| Policies | G0: the saved controller, unchanged. Glex: `gi_lexical` (names, descriptions, question and evidence overlap; deterministic; no labels). G4: one Qwen3.5-4B grounding call (`gi_llm`) |
| G4 call | the pinned Qwen3.5-4B (`contracts.MODELS["B4"]`), non-thinking, the card's non-thinking sampling, seed `int(sha256("G4:" + task_id)[:8], 16)`, at most 512 generated tokens, no tools, no repair. The B4 harness's user message (question, evidence, schema text at the 24,000-character budget, the retrieved training examples from other databases with their queries) and the system prompt `gi_llm.SYSTEM_GROUND` (prompt `gi-g4-prompt-v2`, selected on the development questions: v1 left out "status" in 7 of 128 replies and quoted names in 3) |
| Record | `gi_record` v1: ranked tables, ranked columns (`{table, column}`), literal bindings (`{span, value, table, column}`), a status (`ok` / `unresolved`); exactly these keys |
| Parser and fallback | the last fenced block or the whole reply, parsed as JSON; if that fails, again with invalid backslash escapes removed; then the outermost braces alone. One layer of SQL quoting around a name is removed before it is matched (both repairs selected on the development questions; neither changes meaning). Any parse failure, wrong key or type, unknown table or column, or truncated output falls back to G0, and the call is charged in full. A literal binding whose span or value does not occur in the question or evidence, or whose column is unknown, is dropped and counted, and never causes the fallback |
| Consumption | a candidate at rank r gains ln 2 / r on the table-pointer (tables) or column-pointer (columns) logits at every step. Unlisted items gain 0. At most 8 tables and 16 columns are read. Legality masks, operators and families are unchanged. Literals and status are recorded and scored, not consumed. Glex and G4 are consumed by the same rule |
| Device | the new population is decoded on the GPU at the performance study's controller precision ("high"); every request runs its own fresh backbone forward; each decode is reconstructed from the anchored cache with the same prior and must be equal |
| New population | `pins/gi_new_eval.json`: inventory positions 128-191 (64 questions). They were checked unused by any model, controller, baseline or diagnosis run before this freeze (`gi_run pin-new` scans the runs root, the evidence bundles and the pins). Only three files name them: the inventory pin itself, the data-preparation check that their gold SQL executes, and the grammar's gold-program coverage analysis, which informed the grammar's design |
| Scoring | the pinned EX scorer; every requested question in the denominator, fallback, malformed grounding, abstention and timeout included |
| Direct control | the same Qwen3.5-4B call configuration asked for SQL directly (`gi_llm.SYSTEM_DIRECT`, seed `"B4-direct:" + task_id`), extracted as the B4 harness does, no repair; scored the same way. A short-budget control, not the performance study's thinking-mode B4 |
| Budget | at most 8 GPU-reservation hours within authorization v2's 32 cumulative hours: no reservation may let the cumulative total plus 1.5 x its projection pass 29.416 h |

## The interface gate (development, before the new population)

**The oracle diagnostic.** On the 128 development questions, the oracle record holds the tables and columns the gold
program uses, all at rank 1 (unordered: the gold first FROM table is not singled out). It is consumed by the same
rule. This is an intervention with evaluator information; it never enters ordinary results, training data or
deployment inputs.

**The rule.** The interface "meaningfully influences decoding" if the oracle, Glex or G4 changes the final SQL on at
least 10% of the 768 development decodes (six controllers x 128 questions). If none does, the study stops before the
new population and reports the failed interface.

**The gate's result (dev decode `decode_dev/20261103T193217Z`, the G4 records of `ground/20261103T191242Z_dev`).**
- **Influence:** the share of the 768 decodes whose final SQL the policy changes:
  - Glex: 80 of 768 (10.4%);
  - G4: 78 (10.2%);
  - the oracle: 68 (8.9%).

  **The gate passes, narrowly, on the ordinary policies.**
- **EX-correct of 384 (128 questions x 3 seeds):**

  | | G0 | Glex | G4 | oracle |
  |---|---|---|---|---|
  | C4-v2 | 68 | 69 | 71 | 74 |
  | T4-v2 | 70 | 71 | 69 | 69 |

  Even correct grounding through this interface (the oracle) moves few decisions and adds 5 answers pooled. So the
  interface has not shown that it can help much. That is a finding about this fixed prior, not about grounding.
- **G4 records:** 124 of 128 are valid. They recall 0.94 of the gold tables at 0.85 precision, list a gold table
  first in 90% of requests, and cover 0.50 of the gold columns. Their literal values are right 76% of the time.
- **Glex records:** table recall 0.99 at precision 0.42; column coverage 0.85.
- **The swap check:** another question's G4 record changes 36 of 768 decodes.
- **Reproduction:** G0 decoded on the CPU equals the saved GPU decode for 766 of 768.

These results were seen before this freeze and chose nothing but the v2 prompt and the parser repairs (both decided
on the first G4 run's validity, not on EX). The ranking prior, the candidate limits and the fallback are the work
order's.

## The budget for the new population (measured before the freeze)

| step | measured unit | projection |
|---|---|---|
| encode-new (8 rechecks and 64 contexts) | 1.2 s per context, about 60 s to load | 0.05 h (reserved at 0.15) |
| ground new (64 G4 calls) | 8.7 s per call on the dev questions, 52 s to load | 0.17 h (reserved at 0.3) |
| eval-new (6 controllers x 3 policies x 64, timed, reconstructed) | 1.51 s per request in the performance study's timed evaluation | 0.48 h (reserved at 0.6) |
| direct-new (64 calls) | taken as the G4 call | 0.17 h (reserved at 0.3) |

The projections total about 0.9 h. Each reservation's watchdog allows 1.5 times its projection, so the worst case is
2.0 h. The ledger held 22.042 h at this freeze, against the work order's limit of 29.416 h. The comparison runs in
full, both architectures by seed, or not at all.

## Outcomes on the new population

**Primary: EX-correct requests out of every requested question**, for G0, Glex and G4, for both architectures and all
three seeds.

Write c(p, a, s) for policy p, architecture a and seed s on the 64 new questions. For each architecture, the paired
differences are summed over seeds, so each is a count over 192 question-seed pairs:
- G4 - G0;
- Glex - G0;
- G4 - Glex.

Each difference is read with the performance study's three conditions:
- at least +10 of 192;
- the same sign in every seed;
- the same sign leaving any one database out.

A difference that meets all three is a **consistent difference**; anything else is "no consistent difference".

**Reading, in this order:**
- **G4 beats both G0 and Glex consistently**, for both architectures: the task-conditioned grounding interface merits
  a larger paired study.
- **G4 beats G0 but not Glex:** explicit grounding helps; neural grounding has not earned its cost.
- **Both architectures improve similarly:** an interface improvement, not evidence for specialization. "Similarly"
  means the architecture interaction, (T4: G4 - G0) - (C4: G4 - G0), is not a consistent difference.
- **T4 improves more** (the interaction is a consistent positive difference): a development interaction worth
  investigating, not confirmation.
- **The direct short-budget B4** answers more questions than the best controller policy in every seed: retain that
  baseline and reconsider whether the controller adds value.
- **Nothing improves:** do not spend the remainder on more seeds or larger controllers.

## Diagnostics (reported, not gated)

- **The records:**
  - table-set recall and precision against the gold program's tables;
  - whether the first listed table belongs to the gold table set, separately from agreeing with the gold first FROM
    table;
  - column coverage;
  - literal-binding accuracy (value and column against the gold program's literals) and the unresolved rate;
  - malformed, invalid and truncated output, and fallback rates.

  Reference sets are annotations, not proof that every alternative is invalid.
- **The decodes:**
  - changed decisions against G0;
  - correct-to-wrong regressions and wrong-to-correct recoveries;
  - EX-correct programs with another table set than gold (alternative structures).
- **The interface check:** each question decoded with another question's G4 record from the same database (the next
  in population order), apart from the ordinary evaluation. A change shows influence, not correctness or
  faithfulness to the backbone.
- **Costs:**
  - marginal serving per request and per EX-correct answer: controller time (fresh forward and decode), plus the
    lexical step or the G4 call (timed in its own run), plus the submitted query's execution;
  - the GPU-reservation hours of every step.

## Limits

A development comparison in the same database collection as the earlier studies, with three saved seeds of each
architecture. Not a confirmation. The 500-question reserved BIRD Mini-Dev benchmark is not read.
