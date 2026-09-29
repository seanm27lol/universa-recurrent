# Gemma-3-27B confirmation addendum — Phase Two open_weight_lingua

**Status: FROZEN 2026-09-29 (UTC), before any model forward of the Gemma-3-27B
confirmation pilot.** For `x = 3; y = 8; x = x + 2`, the reference answer for
`x` is 5, and the Gemma target writes it as `5` + `"\n"` + `<end_of_turn>`. The
closed 27B pilot `pilot-20260926T185550Z-d085d53c` failed the usability gate
(P0 0.2539) because the frozen `raw` metric rejects that trailing newline. This
addendum declares one confirmation pilot for 27B under the existing amendment.
It adds no rule of its own. **The closed pilot keeps its frozen outcome and
its STOP decision. Nothing here relabels it.**

## What this run applies, unchanged

| Item | Value |
|---|---|
| Answer instrument | [gemma_answer_convention.md](gemma_answer_convention.md), unchanged: `--answer-convention rstrip`, exactly correct iff the generation terminated and `exact_integer(text.rstrip(), expected_answer)`; the teacher-forced scoring channel (digits + `<end_of_turn>`) is untouched |
| Model lock | `configs/model-lock-gemma3-27b.json`, sha256 `ca078e528a9daaeb93681bc95fc88e505b8c37cc5d9121e194939cf9e0686d49` (the closed pilot's lock); model artifacts re-verified against it on 2026-09-29 (UTC) before this addendum was committed |
| AV serving cast | The lock-declared BF16 serving cast of the float32-native AV, as in the closed pilot. It is still unauditable locally, because no fp32 A/B comparison fits on this machine. It attaches to every AV-dependent number |
| Split plan and groups | Plan hash `ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f`, split seed 203001, the same 128 pilot groups (512 prompt variants) as the closed pilot |
| Calibration fit | `runs/calibration-20260926T183645Z-c3374c27`, reused as-is, not refit: fit identity `784594e89f6c315ba7c77eed8d2cf00a6aa5c0bfbe8204013c9dad85e83ec886`, tensors sha256 `7aaa32f1…`, frozen median norm 41,629.35. Calibration is a target-only extraction, so the answer convention cannot affect it |
| Frozen thresholds | P0 accuracy ≥ 0.80; donor/perturbation sensitivity present; one-sided 95% upper P2 accuracy loss ≤ 5 pp; P2 − P3 correct-answer log-probability lower > 0; edit-eligible groups ≥ 32 of 128 (edit rule v1.0.0, sha256 `e1560d7e…`); stage budget 28,800 s |
| Bootstrap | Whole-group, 3,000 resamples, seed 203100 |
| Code | The `gemma3-port` runner used by the Gemma-12B confirmation, source-identical (`runner.py` `7ad60ee5…`, `audit.py` `a91fc8a9…`, `metrics.py` `aedc88e7…`) |

The launch, from the repository root, with the calibration fit passed as a
directory and the default model cache (as in the closed pilot):

```
bash research/open_weight_lingua/scripts/run_pilot.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-27b.json \
  --calibration-fit research/open_weight_lingua/runs/calibration-20260926T183645Z-c3374c27 \
  --answer-convention rstrip
```

## The prediction under test (not a threshold)

The post-hoc lens ([reports/post_hoc_answer_lens.md](../reports/post_hoc_answer_lens.md))
re-read the closed pilot's saved 27B generations with trailing whitespace
stripped. It predicts:

| Quantity | Predicted |
|---|---:|
| P0 exact-answer accuracy | 0.9648 (494/512) |
| P2 exact-answer accuracy | 0.9531 (488/512) |
| P2 accuracy loss, point | 2.34 pp |
| P2 accuracy loss, one-sided 95% upper | 6.25 pp |
| Implied decision | STOP: preservation fails (6.25 > 5 pp); edit coverage is text-independent and expected to stay 0/128 |

These numbers are the prediction this run tests. They are **not** thresholds
and they change no rule. The decision comes only from the frozen thresholds
above. If the run reproduces the closed pilot's generations bitwise, as the
12B confirmation did (512/512), the lens numbers should reproduce exactly. Any
departure is reported as a departure, with its size.

## One run

- Exactly one confirmation pilot. No threshold, bootstrap, split plan, group
  set, calibration fit, lock or serving cast changes.
- Its outcome is recorded whatever it is, including a result that contradicts
  the prediction. There is no rerun to get a different answer.
- A failed or false-start run is preserved with its error record and reported
  as a failure. It is not silently replaced.
- Model weights and run directories stay local; reports cite run IDs and
  hashes.

## Limits

This is a confirmation of an instrument fix on a second model size. It is not
a new claim channel. Whatever it shows, it does not vindicate the language
route beyond this family, task and site. It does not rank model sizes: the
27B AV's BF16 cast and the different sites (block 41 of 62, a full-attention
block, against the 12B's block 32 of 48, a sliding-window block) forbid that
comparison. The edit hypothesis remains untested unless edit coverage changes,
which the answer metric cannot cause.
