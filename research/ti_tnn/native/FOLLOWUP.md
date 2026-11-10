# Native TI-TNN follow-up: intact data and consumed-state results

**Scope:** bounded follow-up to merged PR #76. Original checkpoints, data, protocols, summaries and the original
113-check review are unchanged. No 124M campaign or WSP edit was launched. This follow-up is a negative learning
readiness result, not an architectural comparison or a scaling recommendation.

## 1. Structured packing audit

The saved packer treats every row as an independent model input. When a structured document continues into another
row, that row cannot see the initial register values or earlier instructions; its first label is also `-100`.
Program source, including `Initial:` and the ordered program instructions, is present in the original prompt.

The complete saved-array audit found:

| Split | Multi-row examples | Continuation rows / masked labels | Masked state targets | Masked route targets | Rows missing prior instructions | Answer continuations |
|---|---:|---:|---:|---:|---:|---:|
| Train | 8,504 | 8,596 | 5,184 | 1,296 | 6,814 | 1,782 |
| Dev | 399 | 406 | 208 | 52 | 316 | 90 |

Every continuation row lacks its initial values in the row-local input. The first continuation label is masked in
all 8,596 train and 406 dev cases. These are defects in the old training path; the saved examples themselves contain
complete prerequisites.

`fixtures/structured_boundary_counterexample.json` and its test construct two documents whose four-token continuation
rows are identical while their required after-states differ. The intact builder keeps the full, different inputs
visible instead.

The new `packing="intact"` path:
- keeps every fitting structured example in one model-visible row;
- explicitly raises on over-context examples by default, or audits them under `structured_overflow="exclude"`;
- computes storage after row-padding, avoiding partial arrays and silent truncation;
- leaves `packing="legacy"` behavior available for reproduction of the old path.

The bounded study data has 1,999,998 train tokens / 1,975,083 loss targets and 99,942 dev tokens / 98,695 loss
targets. It contains 24,915 train and 1,247 dev structured documents with **zero cross-row documents**. No bounded-
prefix example required exclusion. Whole-example budget caps now permit documented underfill rather than failing
when the next complete example would cross the cap; the manifest records the underfill amount and reason.

## 2. Consumed-state repair

The earlier pass-three placement can affect current-token logits, but it has no direct path through the saved
attention cache to later answer tokens: slot tensors are local to each token and are not carried in that cache.
The corrected coherent schedule makes the after-state available early enough for hidden-state attention to affect
later positions:

```text
pass 1: before    pass 2: after    pass 3: after    pass 4: after
```

All four modes receive the same target placement; no new loss or model initialization was introduced. The causal
test now identifies an instruction-token after-state intervention during the second pass and measures its effect on
a later answer token through the saved attention cache. This establishes a later-position causal path, not learned
semantic use.

## 3. Saved-generation diagnostics

`independent/rescore_generations.py` preserves exact-match scoring and adds:
- canonical format validity;
- per-register accuracy and mean absolute register-value error;
- answer-value loss over format-valid completions;
- EOS and truncation rates;
- first format/register/text error;
- complete versus truncated/incomplete prerequisites.

The original structured prompts contain `Initial:` and all ordered program instructions. All 512 structured examples
per arm have complete prerequisites. All 64 code prompts have truncated prerequisites. Answer-value loss is not a
cross-entropy score; it is the mean absolute register-value error over format-valid completions.

The independent scorer also parses and executes every saved program itself, validating canonical syntax, register
ranges, target text and example identity. All **71,680 stored targets passed; zero inconsistent targets and zero
score changes were discovered**. Original score files are preserved beside the new executed-rescoring files.

Answer-token cross-entropy is computed separately from mean absolute register error. On ordinary dev it is
0.801797 dense, 0.812047 shared, 0.812917 untyped and 0.814691 typed; on the paired intact run it is 0.821640
untyped and 0.823888 typed. These are model losses, not register-value errors.

The independent rescore is reproducible in the weight-free bundle. Its `study` path field is the only difference
between source and bundle outputs; all result content is identical.

## 4. Evidence scope and bundle

The original 113-check output remains byte-identical. It is best described as **artifact consistency verification**,
not independent semantic validation. New independent rescoring uses only the Python standard library and does not
import native producer modules.

`runs/ti_tnn_native_followup/evidence/` contains selected prompts, all saved completions, diagnostics, curve endpoints,
checkpoint manifests, protocol/configuration identities, numerical history, and the original review. It contains no
weights, packed arrays or raw corpus. `SHA256SUMS` binds its files. The original full boundary inventory remains
outside the compact bundle.

Numerical acceptance history is additive: the original FP32 strict failure and untyped BF16 route flip are preserved.
The paired follow-up reconstruction uses the existing FP32 serving rule; BF16 serving equivalence is not established.

## 5. Paired intact learning check

Protocol: `paired_protocol.json`, using the existing engineering configuration and identical data, state/route labels,
initialization, sampler and AdamW schedule for untyped and typed arms.

- **1,975,083 loss tokens and 1,057 updates per arm**.
- Untyped checkpoint: `arms/untyped/train/checkpoints/final-update-00001057.pt`,
  SHA256 `318c21de51b61a6bbc100510d55e8690d705d550a9f1473b114713ada42d1f51`.
- Typed checkpoint: `arms/typed/train/checkpoints/final-update-00001057.pt`,
  SHA256 `0f0175707f6570dcb7a0d6dc2493c966714e5154e7db0393f6482765f1c81aaa`.

| Metric | Untyped | Typed |
|---|---:|---:|
| Initial → final train LM loss | 10.7602 → 0.5484 | 10.7714 → 0.5509 |
| Final structured dev LM loss | 0.5489 | 0.5501 |
| Ordinary dev exact programs | 0/128 | 0/128 |
| Composition-only exact programs | 0/85 | 0/85 |
| Longer-length exact programs | 0/128 | 0/128 |
| Pass-three after-state readout | 4.83% | 3.86% |
| FP32 full/cached replay | pass | pass |

Both models learned next-token prediction but **did not demonstrate complete-program learning or a usable learned
state path**. Register accuracy and answer-value diagnostics remain poor; format validity alone is not capability.
The paired result does not establish architectural superiority or readiness for the proposed 124M campaign.

## Resource accounting

The follow-up consumed **0.169192370 native GPU-reservation hours**, including one failed evaluation wrapper. The
successful untyped and typed training reservations consumed 0.024110 and 0.024768 hours; their corrected evaluation
reservations consumed 0.058049 and 0.058418 hours. Total native usage across the original authorization is now
3.303721975/12 hours, leaving 8.696278025 hours. No old checkpoint or historical ledger entry was modified.

## Reproduction

```bash
export PYTHONPATH=research/ti_tnn
export PYTHONDONTWRITEBYTECODE=1
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python

# Native construction and causal-path checks.
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q tests/test_ti_tnn_native_followup.py

# Independent rescore of the paired outputs.
$PY research/ti_tnn/native/independent/rescore_generations.py \
  --study /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_followup \
  --data /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native/data \
  --output /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_followup/paired_generation_diagnostics.json

# Rebuild the compact weight-free bundle.
$PY research/ti_tnn/native/independent/export_followup_bundle.py
```

The full native suite passed **223 tests** after the data/state changes. No WSP phase has begun.
