# Phase Seven v1: separate question placement from question wording

**Design specification v1, prepared before any Phase Seven model forward.**
The design becomes launch-frozen when the root reviewer records this file's
SHA-256, the completed runner/evaluator, source, stimuli, tests and model
identities in a new `launch_freeze.json`. The stimulus-preparation script
performs no model forward. Changes after that freeze require a new protocol
version and run directory; preserve incomplete and negative outcomes.

## Example and question

Consider this unchanged program:

```text
x = 9
y = 8
y = 3
y = y + 3
x = x + 1
```

Its final values are x = 10 and y = 6. Phase Six changed both the position
and wording of the question. This experiment crosses those two changes so
their behavioral effects can be distinguished. It measures answers only;
**there is no new activation capture, probe fitting, or activation editing.**
The design and behavioral analysis do not depend on the saved-feature probe
follow-up's results.

## Exact conditions

Let `P` be the original program bytes without the original question or a
trailing newline, and let `v` be `x` or `y`. The two question strings are:

```text
Q_original(v) = What is {v}? Reply with only the integer.
Q_expanded(v) = What is {v} at the end of this program? Reply with only the integer.
```

Every `\n` below denotes exactly one newline. There are no extra headings,
code fences, blank lines, or trailing whitespace in the user message.

| Condition | Exact user-message composition | Role |
|---|---|---|
| `original_before` | `Q_original(v) + "\n" + P` | Primary factorial |
| `original_after` | `P + "\n" + Q_original(v)` | Primary factorial |
| `expanded_before` | `Q_expanded(v) + "\n" + P` | Primary factorial |
| `expanded_after` | `P + "\n" + Q_expanded(v)` | Primary factorial |
| `original_repeat` | `Q_original(v) + "\n" + P + "\n" + Q_original(v)` | Separate diagnostic |

The repeated-question diagnostic uses **original wording only**. It is not
a fifth factorial cell or evidence by itself of a particular internal
mechanism: it also changes length and adds a recent reminder. Within each
wording's primary before/after comparison, the question string is identical.

## Models and fixed inputs

- Gemma-3-12B-it is primary; Qwen2.5-7B-Instruct is a separately reported
  secondary family. Reuse their existing immutable model locks and tokenizer
  files. Load only the target model, one family at a time; no AV/AR models.
- Reuse exactly the already-opened Phase Two calibration and pilot programs:
  256 calibration groups plus 128 pilot groups, with both A/B sides and both
  x/y questions. This is 384 groups, 768 programs, and 1,536 base prompts per
  family. Five formats produce **7,680 generations per family**, 15,360 total.
- The exact source run names and manifest SHA-256 values are pinned in
  `prompt_factorial.SOURCE_INPUTS`. The CPU builder rejects other run names
  before reading their manifests, verifies hashes and reference answers, and
  never reads `validation_a` or `validation_b`.
- These programs have informed earlier work. This is a controlled behavioral
  follow-up on reused inputs, not an untouched test of generalization.
  Report calibration and pilot subsets separately as descriptive breakdowns.
- Keep all five formats for a base prompt together. Process base IDs in sorted
  order and permute the five-format order independently per base prompt using
  Python `random.Random(2026100507)`. Preserve the resulting order in the
  stimuli manifest. Both families must contain identical base IDs, program
  bytes, questions, labels and ordering before launch.
- Use each family's locked `apply_chat_template` with one user message,
  `tokenize=True`, `add_generation_prompt=True`. No manually added system
  prompt. Token IDs are prepared and saved before the run.

## Decoding and records

- Frozen inference settings: BF16, eager attention, `eval()`, gradients off,
  no KV cache, one prompt at a time, no sampling, argmax greedy decoding,
  `max_new_tokens = 8`, `TARGET_BUCKET = 128`, TF32 disabled. Do not apply
  sampling settings or repetition penalties from a generation config.
- The source fix in `Target.greedy` stops at any declared
  `model.generation_config.eos_token_id` (scalar or list), falling back to the
  tokenizer's EOS only when the model declaration is absent. Store the resolved
  stop set. This fixes future stopping; it does not rewrite Phase Six outputs.
- Right-pad every forward to 128 using the established mask and read next-token
  logits at the last real token. Preparation must verify that every prompt plus
  eight generated tokens fits. **An overflow fails preparation**; do not trim
  prompts, enlarge the bucket, or silently omit rows in v1.
- Preserve the family's frozen text comparison: Gemma `rstrip`, Qwen `raw`,
  both through `answer_text_matches`. Primary correctness uses this same text
  rule; retain termination separately. Also report the descriptive fraction
  that is both text-correct and terminated. No post hoc whitespace repair.
- For every generation save family, condition, base/group IDs, side, asked
  variable, source split, prompt token IDs, generated token IDs, decoded text,
  resolved EOS set, whether/which stop token occurred, token count, termination,
  reference answer and correctness. Preserve truncated outputs and failures.
- Recompute **all five formats**, including the two previously evaluated
  combinations, under this single version. Historical Phase Six answers are
  contextual comparisons only, because stopping code has changed.

## Fixed analysis

Let `A(w,p)` be mean correctness for wording `w` and position `p`, pooled
over the 1,536 base prompts within one model. Every model is analyzed separately.
Five planned primary contrasts per model are:

1. Original-wording position effect: `A(original,before) - A(original,after)`.
2. Expanded-wording position effect: `A(expanded,before) - A(expanded,after)`.
3. Wording effect before: `A(expanded,before) - A(original,before)`.
4. Wording effect after: `A(expanded,after) - A(original,after)`.
5. Interaction: contrast 2 minus contrast 1.

All comparisons pair the same base prompts. Resample **whole program groups**,
retaining both A/B sides, both asked variables and all formats together:
10,000 NumPy `default_rng` bootstrap resamples, seed 701100 for Gemma and 701101
for Qwen. Each group has four base prompts, so averaging group contrasts equals
the prompt-level contrast. Report point estimates and percentile **99.5% CIs**
(quantiles 0.0025 and 0.9975), giving a Bonferroni family across the ten declared
model-by-contrast comparisons. These are bootstrap uncertainty summaries,
not exact finite-sample guarantees. Keep all outcomes even when intervals span
zero. A five-percentage-point absolute effect is the predeclared practical
reference; distinguish a resolved sign from an interval supporting an effect
of at least that size.

Report each cell's numerator, denominator and accuracy, then the contrasts.
In addition report the full cells and before-minus-after comparisons by:

- model and asked variable (`x`, `y`);
- whether the program's final line assigns the asked variable;
- asked variable's last update (`assign`, `add`, `subtract`, `copy`), also
  grouped as literal/arithmetic/copy;
- final program operation, source calibration/pilot split and termination.

Include denominators and empty categories. These subgroup summaries are
descriptive; do not turn nominal subgroup intervals into additional success
claims. Report the joint asked-variable × final-writer table to expose the
last-line pattern without selecting a favorable variable.

For `original_repeat`, separately report paired differences against
`original_before` and `original_after`, by model, with nominal 95% group
bootstrap intervals. Label these diagnostics; exclude them from the primary
factorial family and from any claimed isolation of a tracking mechanism.

There is no activation-readability gate. A weak before-position result, with
wording held fixed, concerns this prompting format's answer accuracy. A
repeat-question improvement would show that a reminder helps this task; it
would not establish where or how state is represented. Do not infer absence
of internal state from behavioral errors.

## Preparation, freeze and execution boundary

The implemented CPU entry point is:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv-phase2/bin/python -m open_weight_lingua.prompt_factorial \
  --family gemma3-12b \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --calibration-run research/open_weight_lingua/runs/calibration-20260923T042912Z-d0e9499f \
  --pilot-run research/open_weight_lingua/runs/pilot-20260923T043613Z-f7e71d7b \
  --cache research/open_weight_lingua/model-cache \
  --output research/open_weight_lingua/runs/phase-seven-preparation-v1/gemma_stimuli.json
```

For Qwen the concrete preparation command from this checkout is:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv-phase2/bin/python -m open_weight_lingua.prompt_factorial \
  --family qwen2.5-7b \
  --lock research/open_weight_lingua/configs/model-lock.json \
  --calibration-run /home/seanjazm27/projects/universa-recurrent-m2/research/open_weight_lingua/runs/calibration-20260921T235017Z-fd111b21 \
  --pilot-run /home/seanjazm27/projects/universa-recurrent-m2/research/open_weight_lingua/runs/pilot-20260921T235825Z-6164d210 \
  --cache /home/seanjazm27/projects/universa-recurrent-git/research/open_weight_lingua/model-cache \
  --output research/open_weight_lingua/runs/phase-seven-preparation-v1/qwen_stimuli.json
```

The builder
loads only tokenizer/metadata, checks all non-weight target artifacts against
the lock, and refuses overwrites. It does not run or load a target model.
Use a fresh preparation directory under ignored `runs/`; the example name is
not an implicit resume destination.

Before any GPU forward, complete and CPU-test the behavior runner and grouped
evaluator against this specification. Freeze their exact source and tests,
this protocol, both prepared stimuli manifests, all four source manifests, model
locks, target tokenizer/chat-template/generation-config files, dependency lock,
and every imported source file by SHA-256 in a new `launch_freeze.json` with
timestamp. The reviewer verifies that all primary analysis choices above are
implemented. Record Git HEAD and dirty-source hashes: HEAD alone is insufficient.
Hash-check the complete target checkpoint shards against each model lock at
launch; record identities again at completion. Abort on any mismatch.

The launch-freeze schema is
`format = "open_weight_lingua.prompt_factorial_launch.v1"`,
`wall_limit_seconds = 10800`, `family_stimuli` mapping `gemma3-12b` and
`qwen2.5-7b` to their absolute prepared-JSON paths, and `files_sha256` mapping
absolute dependency paths to SHA-256 strings. Include `created_at` and Git
identity for review. The runner verifies both families' input/source identities
before loading either model. Model shards are bound by the frozen model locks
and verified against them at launch and completion.

Run the two model stages sequentially, in separate processes sharing exactly
the same campaign directory. For a newly frozen preparation at the example
path, the concrete commands are:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-phase2/bin/python -m open_weight_lingua.prompt_factorial_run run \
  --family gemma3-12b --device cuda \
  --freeze research/open_weight_lingua/runs/phase-seven-preparation-v1/launch_freeze.json \
  --campaign research/open_weight_lingua/runs/phase-seven-campaign-v1 \
  --output research/open_weight_lingua/runs/phase-seven-campaign-v1/gemma
PYTHONDONTWRITEBYTECODE=1 .venv-phase2/bin/python -m open_weight_lingua.prompt_factorial_run run \
  --family qwen2.5-7b --device cuda \
  --freeze research/open_weight_lingua/runs/phase-seven-preparation-v1/launch_freeze.json \
  --campaign research/open_weight_lingua/runs/phase-seven-campaign-v1 \
  --output research/open_weight_lingua/runs/phase-seven-campaign-v1/qwen
```

`campaign.json` starts its shared clock before verification and loading in the
first process. The second stage inherits that deadline, including the gap
between processes. A family can be claimed only once and concurrent family
runs are rejected. The runner flushes every completed observation to
`generations.jsonl`; timeouts and failures retain those records and write a
completion status. Partial records cannot enter the primary evaluator.

After both stages, run the CPU evaluator once per complete family directory:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv-phase2/bin/python -m open_weight_lingua.prompt_factorial_run evaluate \
  --freeze research/open_weight_lingua/runs/phase-seven-preparation-v1/launch_freeze.json \
  --run research/open_weight_lingua/runs/phase-seven-campaign-v1/gemma \
  --output research/open_weight_lingua/runs/phase-seven-campaign-v1/gemma_analysis
```

For Qwen replace the last two directory names with `qwen` and `qwen_analysis`.
The evaluator checks generated tokens against their decoded text and frozen
scoring rules, then writes cell/subgroup counts, paired contrasts and the
group-bootstrap draws. Termination-stratified cells condition on each format's
observed termination, so their differences are descriptive conditional means;
they are not additional paired treatment-effect estimates.

Every execution gets a new directory. No writes to old runs, no implicit
resume and no source edits during the run. Set a **three-hour wall cap** for
the combined model stages; record partial status and completed rows if it
expires. No condition-dependent retries or parameter changes. Record setup,
hashing/loading, generation, serialization, analysis and total wall time.

`validation_a` and `validation_b` remain closed throughout this v1 follow-up.
Any later confirmatory study must freeze its separate analysis before opening
those data; do not substitute them into this stage to improve a result.

## Spark estimate, not a measurement of this follow-up

The recorded Phase Six `completion.json` files give these actual seconds:

| Family | Two-format answers | Capture | Probe fitting | Run wall | Separate setup |
|---|---:|---:|---:|---:|---:|
| Gemma-3-12B | 1250.347 | 248.014 | 35.296 | 1699.199 | 443 |
| Qwen2.5-7B | 1027.383 | 254.971 | 34.429 | 1429.976 | 562 |

Source runs are `p6-question-first-gemma3-12b-20261004T231457Z-96041d06` and
`p6-question-first-qwen2.5-7b-20261004T235340Z-4f429ce9` under `runs/`.

Scaling answer time from two formats to five yields approximately **5694 s
(94.9 min)**. Four primary formats alone account for 75.9 min, and the separate
repeat diagnostic adds 19.0 min. Retaining the old non-capture/non-probe wall
residual adds about 279 s. Repeating both old setup costs adds 1005 s, giving
**about 116 min nominal** before any extra reporting variation. Plan for
**105–135 minutes total**, with the three-hour hard stop above.

Assumptions: same DGX Spark, model revisions, BF16/eager/no-cache/128-token
inference, one load per family, similar generated lengths and no competing GPU
work. New wording or repetition may change EOS timing; multi-EOS stopping may
shorten some trajectories. Setup could be cheaper if environments and tests
are already ready. **Capture and probe-fit time are not included**, because
this stage performs neither. Model loading and analysis are not demonstrated
by the CPU stimulus preparation; the estimate is not a benchmark claim.

## Evidence and limits

The interpreter certifies these short programs' answers. Factorial contrasts
separate the declared wording/placement manipulations on the reused programs;
they do not establish a general instruction-order law or a neural mechanism.
The existing [Phase Six brief](phase_six_brief.md) records the motivating
question-first manipulation and primary entity-tracking precedents; this v1
adds its missing wording control. Existing scalar/affine probe follow-ups
remain a separate CPU analysis of saved states. If that analysis motivates a
new activation experiment, give it its own design, instrumentation checks and
freeze after this behavior study is specified.
