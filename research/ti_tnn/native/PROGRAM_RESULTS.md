# Varied-program study results

**Status:** all three arms completed the frozen study at source commit
`2eb2e527be9c4f4db60902d78073fd05c3d23f32`, with no intermediate-checkpoint selection.

Run: `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/runs/20261110T210743-921461de/`

Reservation: `20261110T210743-bd8d1d5b`. The run directory contains `run_status.json` with `complete=true` and
`failure=null`. Historical protocol text saying `proposed_not_authorized_or_launched`, and the evaluator template's
reference to a future trained checkpoint, remain unchanged in their original artifacts; this report describes the
completed run rather than altering those records.

This is one bounded development experiment, not a confirmation study, architecture ranking, 124M campaign or
weight-space-programming result. The three architectures have different parameter counts and are not claimed to be
simultaneously parameter- and compute-matched.

## 1. Frozen endpoint and loss

Every arm consumed exactly **3,392 updates and 5,570,640 loss tokens**, with 16 epochs of 212 updates and 348,165
loss tokens per epoch. The sampler produced the same batch order across arms. No arm was truncated.

| Arm | First → last train LM loss | Final dev LM loss | Training seconds | Actual loss tokens/s | Checkpoint seconds | Arm pipeline wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 10.770858 → 0.512624 | 0.478083 | 217.509 | 25,611.1 | 6.705 | 555.415 |
| Untyped MoE | 10.950928 → 0.534946 | 0.495394 | 207.595 | 26,834.2 | 5.888 | 640.312 |
| Typed MoE | 10.957602 → 0.532055 | 0.493684 | 210.433 | 26,472.3 | 5.856 | 652.994 |

Final checkpoint paths and hashes:

| Arm | Path relative to run directory | SHA256 |
|---|---|---|
| Dense | `arms/dense/train/checkpoints/final-update-00003392.pt` | `be4d3405e278ce6a6afe6c7dc4c002cd0f86f9ac07942a6a158b7e0299596418` |
| Untyped MoE | `arms/untyped/train/checkpoints/final-update-00003392.pt` | `1608739f43fb83a88e851b3e8cc9aeee4ec0b1c42ca7323594d46b66963a2199` |
| Typed MoE | `arms/typed/train/checkpoints/final-update-00003392.pt` | `cf7d6bb49a4d67a1ad8c188b4a03252b4f586521181e34cc5f99e5d8b75b187f` |

Peak allocated / reserved CUDA memory was 1.77 / 8.90 GiB dense, 1.77 / 9.00 GiB untyped and 1.78 / 8.99 GiB typed.
The checkpoint files and manifests also pass their internal model, optimizer, scheduler, sampler, RNG and state-hash
checks. All 62 data-artifact hashes and the protocol's 12 source-file hashes remain valid.

## 2. Exact answers, answer-token CE, format and register error

Each row below has its own **128-example denominator**. Exact answers require the complete canonical register text,
not merely a valid prefix or correct individual register. Answer-token cross-entropy (CE) is teacher-forced and
includes EOS. Value MAE is mean absolute register error **only over format-valid completions**; it is not CE.
`Valid` is the number of canonical format-valid outputs. `Trunc.` is the number of generations that reached the
48-token generation bound without EOS. `Changed` is completion/generated-suffix changes in the separate 32-example
intervention set, not in the primary 128-example denominator.

| Arm | Population | Exact /128 | Answer CE | Valid | EOS / trunc. | Value MAE | Changed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | Train | 13 | 0.386559 | 128 | 128 / 0 | 4.542969 | 2 / 2 |
| Dense | Dev | 6 | 0.477592 | 127 | 128 / 0 | 5.133858 | 0 / 0 |
| Dense | Initial values | 0 | 0.931426 | 126 | 128 / 0 | 11.587302 | 0 / 0 |
| Dense | Unseen composition | 3 | 0.592629 | 125 | 128 / 0 | 6.608000 | 1 / 1 |
| Dense | Longer length | 0 | 3.164358 | 4 | 60 / 68 | 5.062500 | 0 / 0 |
| Untyped | Train | 4 | 0.498006 | 128 | 128 / 0 | 5.582031 | 0 / 0 |
| Untyped | Dev | 1 | 0.564119 | 128 | 128 / 0 | 5.978516 | 0 / 0 |
| Untyped | Initial values | 0 | 0.841986 | 127 | 128 / 0 | 10.169291 | 0 / 0 |
| Untyped | Unseen composition | 0 | 0.639380 | 128 | 128 / 0 | 7.628906 | 0 / 0 |
| Untyped | Longer length | 0 | 3.281691 | 2 | 12 / 116 | 8.250000 | 0 / 0 |
| Typed | Train | 5 | 0.485189 | 128 | 128 / 0 | 5.800781 | 0 / 0 |
| Typed | Dev | 2 | 0.554353 | 128 | 128 / 0 | 6.023438 | 0 / 0 |
| Typed | Initial values | 0 | 0.919669 | 126 | 128 / 0 | 10.847222 | 0 / 0 |
| Typed | Unseen composition | 1 | 0.639090 | 127 | 128 / 0 | 7.688976 | 1 / 1 |
| Typed | Longer length | 0 | 3.401804 | 1 | 18 / 110 | 6.750000 | 0 / 0 |

The exact-answer counts are therefore:

| Arm | Train | Dev | Initial values | Unseen composition | Longer length |
|---|---:|---:|---:|---:|---:|
| Dense | 13 | 6 | 0 | 3 | 0 |
| Untyped | 4 | 1 | 0 | 0 | 0 |
| Typed | 5 | 2 | 0 | 1 | 0 |

The supplied-program executor scores 128/128 in every split. Exact lookup scores 128/128 on the selected training
prompts and zero elsewhere. These are separate baselines, not the neural model or a symbolic replacement for it.

### Per-register accuracy

Values are `r0 / r1 / r2 / r3`, with each register measured over the valid-output denominator for that population.
The longer-length denominators are only 4, 2 and 1 respectively; these are not 128-example estimates.

| Arm | Population | Per-register accuracy |
|---|---|---|
| Dense | Train | 0.570 / 0.563 / 0.547 / 0.641 |
| Dense | Dev | 0.480 / 0.520 / 0.425 / 0.472 |
| Dense | Initial values | 0.135 / 0.079 / 0.119 / 0.167 |
| Dense | Unseen composition | 0.408 / 0.392 / 0.344 / 0.400 |
| Dense | Longer length | 0.750 / 0.250 / 0.250 / 0.250 |
| Untyped | Train | 0.469 / 0.445 / 0.477 / 0.438 |
| Untyped | Dev | 0.484 / 0.422 / 0.320 / 0.336 |
| Untyped | Initial values | 0.118 / 0.094 / 0.157 / 0.134 |
| Untyped | Unseen composition | 0.313 / 0.344 / 0.289 / 0.273 |
| Untyped | Longer length | 0.000 / 0.000 / 0.000 / 0.000 |
| Typed | Train | 0.422 / 0.523 / 0.477 / 0.422 |
| Typed | Dev | 0.461 / 0.422 / 0.336 / 0.320 |
| Typed | Initial values | 0.119 / 0.159 / 0.143 / 0.127 |
| Typed | Unseen composition | 0.299 / 0.362 / 0.244 / 0.299 |
| Typed | Longer length | 0.000 / 0.000 / 0.000 / 1.000 |

Full per-register absolute-error arrays, answer CE per example, format failures, first divergences and generated
text are preserved in each arm's `evaluation.json` and in the weight-free bundle.

### Final-operation results

Counts are exact answers **per final operation**, with denominator 32 for every cell. Format validity by operation
is preserved separately in `generation.by_final_operation` in the full evaluation and independent diagnostics.

| Arm | Population | Set | Add | Multiply | Copy |
|---|---|---:|---:|---:|---:|
| Dense | Train | 4 | 0 | 1 | 8 |
| Dense | Dev | 2 | 0 | 1 | 3 |
| Dense | Initial values | 0 | 0 | 0 | 0 |
| Dense | Unseen composition | 1 | 1 | 0 | 1 |
| Dense | Longer length | 0 | 0 | 0 | 0 |
| Untyped | Train | 1 | 1 | 0 | 2 |
| Untyped | Dev | 0 | 0 | 0 | 1 |
| Untyped | Initial values | 0 | 0 | 0 | 0 |
| Untyped | Unseen composition | 0 | 0 | 0 | 0 |
| Untyped | Longer length | 0 | 0 | 0 | 0 |
| Typed | Train | 2 | 0 | 0 | 3 |
| Typed | Dev | 0 | 0 | 0 | 2 |
| Typed | Initial values | 0 | 0 | 0 | 0 |
| Typed | Unseen composition | 0 | 1 | 0 | 0 |
| Typed | Longer length | 0 | 0 | 0 | 0 |

For longer length, format-valid counts are `set/add/multiply/copy = 2/0/2/0` dense, `0/0/2/0` untyped and
`1/0/0/0` typed. All other operation exact counts are zero.

## 3. What the holdouts mean

**Initial values.** All 512 scalar initial values in this split are in `[24,32)`, whereas training initial scalars
are in `[0,24)`. The 128 instruction sequences are reused training sequences. This is an initial-value-role holdout,
**not** a claim that numbers 24–31 are absent everywhere in training. In fact, 974 training examples contain high
`set` immediates, 1,508 have high final-state values and 1,945 have high noninitial execution-state values.

**Unseen composition.** The reserved adjacent pattern is `multiply` immediately followed by `add`. It is absent
from train, dev, initial-value and longer-length splits. There are 133 such adjacent pairs across 128 programs in
the unseen-composition split. Only 64/128 examples have the direct local dependency where the add reads the multiply
destination through either add input; 14/32 intervention examples have it. Later overwrites are a separate issue.
Thus this is a structural holdout, not a guarantee of direct data dependency in every example.

**Longer length.** Programs have 8–12 instructions. No prompt was truncated. Output truncation is a measured result,
not an incomplete evaluator: 68/128 dense, 116/128 untyped and 110/128 typed outputs reached the generation bound.
Intervention-set truncations are 12/32, 28/32 and 25/32 respectively. Format and per-register coverage must be read
with those small denominators in mind.

## 4. Targeted writer interventions

Each intervention suppresses all four slot-writer output vectors at the explicitly identified final-instruction
anchor in **pass two**, the first supervised after-state. The incoming state is retained; later passes and generated
suffixes receive no direct suppression. Every intervention targeted exactly once and suppressed four writer updates.
The historical `interventions.operation` string says "zero-based pass two"; the actual implementation targets
`PASS_INDEX=1`, meaning one-based pass two (`pass_1`), the first after-state. The readout table below also uses
`pass_1`, not the later pass with the same target values.

`Baseline exact` and `Intervened exact` use the **separate 32-example intervention denominator**, with eight examples
per final operation. They are not the 128-example primary scores. Changed completions and generated suffixes are
reported independently, not merged into a new accuracy denominator.

| Arm | Population | Baseline exact → intervened exact | Changed completions / suffixes |
|---|---|---:|---:|
| Dense | Train | 1 → 0 | 2 / 2 |
| Dense | Dev | 0 → 0 | 0 / 0 |
| Dense | Initial values | 0 → 0 | 0 / 0 |
| Dense | Unseen composition | 1 → 1 | 1 / 1 |
| Dense | Longer length | 0 → 0 | 0 / 0 |
| Untyped | Train | 1 → 1 | 0 / 0 |
| Untyped | Dev | 0 → 0 | 0 / 0 |
| Untyped | Initial values | 0 → 0 | 0 / 0 |
| Untyped | Unseen composition | 0 → 0 | 0 / 0 |
| Untyped | Longer length | 0 → 0 | 0 / 0 |
| Typed | Train | 1 → 1 | 0 / 0 |
| Typed | Dev | 0 → 0 | 0 / 0 |
| Typed | Initial values | 0 → 0 | 0 / 0 |
| Typed | Unseen composition | 0 → 0 | 1 / 1 |
| Typed | Longer length | 0 → 0 | 0 / 0 |

The four changed cases are:

- Dense train `structured-train-e86283c67b38fa786c2b`, final operation `multiply`: `r0=20 r1=11 r2=9 r3=11` became
  `r0=0 r1=11 r2=9 r3=11`, a correct answer becoming incorrect.
- Dense train `structured-train-897b0313be428bd6cea3`, final operation `multiply`: `r0=0 r1=7 r2=7 r3=7` became
  `r0=0 r1=0 r2=7 r3=7`, an incorrect answer becoming another incorrect answer.
- Dense unseen `structured-unseen_composition-905171aa0409307ec5bb`, final operation `set`: `r2=29` became `r2=13`.
- Typed unseen `structured-unseen_composition-407e585e64580e5638ec`, final operation `set`: `r1=31` became `r1=9`.

The last two are also incorrect-to-incorrect changes; there are four changed examples total. Dense has
3/160 changed interventions (two train `multiply`, one unseen `set`), untyped has 0/160 and typed has 1/160 (one
unseen `set`). These are sensitivities to the local intervention, **not semantic faithfulness**. No change does
**not** prove state absence.

Pass-two after-state readout is reported separately from these functional interventions:

| Arm | Train | Dev | Initial values | Unseen composition | Longer length |
|---|---:|---:|---:|---:|---:|
| Dense | 0.375399 | 0.336420 | 0.077569 | 0.312705 | 0.229633 |
| Untyped | 0.263323 | 0.221193 | 0.069664 | 0.225000 | 0.167732 |
| Typed | 0.310273 | 0.271605 | 0.070158 | 0.249590 | 0.183307 |

Readout accuracy measures prediction of the supervised state target over packed rows; it does not prove that
writing or generating an answer uses that state semantically. All state-path evaluations cover every packed row:
1690 train, 51 dev, 54 initial-value, 63 composition and 74 longer-length rows. Answer CE instead covers the
separate 128-example selection, with 2,176 answer tokens including EOS per arm and split.

## 5. Reconstruction and numerical limits

All three saved reports pass the declared CPU FP32 **scale-based reconstruction policy**: relative RMS error <= 0.05,
with exact integer routes and exact fixed-replay generated tokens. That policy is not bitwise equality.

| Arm | Overall agreement | Strict `allclose` at `atol=rtol=3e-6` | Reconstruction report SHA256 |
|---|---|---|---|
| Dense | Pass | Pass | `82d1271209b51b29af54eb27738b4a04d1aae55ea31d03912bdc65f510098e24` |
| Untyped MoE | Pass | Pass | `6cd33ea400e713cebc4a6cf2790500360c859393f6617595d1c58900b9c1a612` |
| Typed MoE | Pass under scale policy | **Fail for `route_logits`** | `e70bc3c131a0efd406fae0cbcc3369406217aedae3d8baf3a91a4a121278f8e6` |

Typed `route_logits` has its first tolerance violation at `[0,1,3,1]`, absolute difference `4.291534423828125e-6`,
and maximum absolute difference `6.198883056640625e-6`. Its relative RMS error is only `3.5788078335711026e-7`, so
its scale-based agreement passes. An independent CPU replay reproduced the same violation. This is not a bitwise
replay or BF16 serving-equivalence claim; the earlier BF16 route failure history remains preserved.

## 6. Resource accounting

The canonical elapsed-time ledger reports:

- Study reservation: `20261110T210743-bd8d1d5b`, closed with status `ok`.
- Charged time: **0.5136816186119617 hours** (1,849.254 seconds), below the authorized 2.5 hours.
- Native cumulative usage: **4.2996365206090665 / 12 hours**.
- Native remaining: **7.7003634793909335 hours**.
- No open native authorization-3 reservation remains after completion. A pre-existing authorization-1 reservation
  remains open; that does not represent a conflicting active lock or a new native study process.

The raw end record's redundant `hours` field differs from canonical elapsed-derived accounting by about
`2.62e-10` hours because the base ledger samples elapsed twice. It is not a charge or cap discrepancy. The actual
run took less time than the 1.539988-hour projection; no endpoint or evaluation was reduced to achieve it.

## 7. Weight-free evidence and reproduction

Use the **new** `evidence-final` directory:

```text
/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/runs/20261110T210743-921461de/evidence-final/
```

It contains 87 files, with 85 entries in `SHA256SUMS`, totaling 69,758,769 hashed payload bytes. The manifest records
that every file except `SHA256SUMS` and `bundle_manifest.json` is covered. It contains no model weights, packed NPY
arrays or raw external corpus. Checkpoint manifests and hashes identify local weights but do not include them.

- `summary.json`: `0907eee25b85cbc27c2c0cc0be3c23ec10a8477b863e05850443b31e52ef5954`.
- `bundle_manifest.json`: `201d08ebd12222801c230cd9cff6a5762f508a521f91d97078cb9c6870c0af55`.
- `generation_diagnostics.json`: `ab16331e1858b85e2452718ebf97b65cdaf66427baffcfaaf1b26fe28f30979f`.
- `generation_diagnostics.reproduced.json`: `f0fbc830c14dff745b128165449dd5cb410e8afed06556ffd7e80f730ff95362`.

Each arm independently scored 640 primary outputs, 30 supplemental train baselines and 160 intervened outputs:
830 per arm and 2,490 total, with **zero score changes**. The whole-study scorer was rerun using only the bundle;
its output matches the preserved diagnostics except for the `study` path.

The exact repeatable reproduction interface is in `evidence-final/REPRODUCE.md`. In outline:

```bash
export BUNDLE=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/runs/20261110T210743-921461de/evidence-final
WORK="$(mktemp -d)"
PYTHONDONTWRITEBYTECODE=1 python "$BUNDLE/code/rescore_generations.py" \
  --study "$BUNDLE" --data "$BUNDLE/data" --output "$WORK/generation_diagnostics.json"
```

The file also contains the Python assertion that compares the fresh output to the preserved diagnostics while
excluding only the `study` path. `SHA256SUMS` and the bundle manifest bind all included evidence.

## 8. Interpretation boundaries

The varied-program run demonstrates learning behavior and limited, measurable generalization; it does not establish
broad composition learning or architectural superiority. Dense has higher exact counts in this single seed, but the
arms differ in parameter count and executed computation. Small intervention sets, truncation-heavy longer-length
outputs and limited initial-value results do not support a claim of robust learned state usage.

No answer-focused training, shaping penalty, extra seed, adaptive stopping, larger model or WSP was added. The
report preserves the low exact scores, truncation, routing data and numerical qualification alongside successful
checks. `PROGRAM_REVIEW.md` records the independent review and its limits.
