# Milestone 1 source and compatibility audit — 2026-09-21

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. The implementation
asks whether a released NLA description can reconstruct one activation direction
well enough to preserve this measured behavior. No released-model result is
reported here.

The initial checkout was clean at
`ecc8d79873b6f76b4e896986092c7ce8909cdf75`. Work is isolated on
`phase-two-milestone-one`. Phase One source, package/dependency declarations,
protocols, weights and historical results are unchanged. The root changes are
an ignore entry and documentation/claim-ledger links.

## Immutable sources actually read

| Component | Revision | Pinned artifact bytes |
|---|---|---:|
| [Target](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) | `a09a35458c702b33eeacc393d103063234e8bc28` | 15,242,805,751 |
| [AV](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av/tree/b88469162777ae6553bc14208eb0cb579336f8f4) | `b88469162777ae6553bc14208eb0cb579336f8f4` | 15,247,195,731 |
| [AR](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar/tree/e2c9e57eac213d37a31612087f645ab6332c1bb6) | `e2c9e57eac213d37a31612087f645ab6332c1bb6` | 10,920,058,064 |
| [Inference recipe](https://github.com/kitft/nla-inference/blob/38b802a33d1d317f21b6825a9116f388c2141f86/nla_inference.py) | `38b802a33d1d317f21b6825a9116f388c2141f86` | Source hashes in lock |
| [Training extractor](https://github.com/kitft/natural_language_autoencoders/blob/0577769b55ad4fdd96d159e983361b97fa4e7331/nla/datagen/extractors.py) | `0577769b55ad4fdd96d159e983361b97fa4e7331` | Source hashes in lock |

The model inventory excludes `.gitattributes`; all runtime configs, tokenizers,
metadata, licenses and safetensors are locked individually with SHA-256 and size.
Weight hashes are read from the Hub's LFS inventory, **not verified against
downloaded weights yet**. Small files were downloaded and hashed locally. All
three model licenses are Apache-2.0. Attribution and copied metadata licensing
are in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Interface findings and implemented checks

| Concern | Actual source finding and implementation |
|---|---|
| Layer indexing | Training extractor hooks `model.model.layers[K]`; output is post-residual block K. For interior blocks it matches `hidden_states[K+1]`. We test that equality, not a guessed `hidden_states[20]`. |
| Metadata schema | Both sidecars have schema version 2, top-level `extraction_layer_index: 20`, and `d_model: 3584`. AR also has `critic.extraction_layer_index: 20`; it does **not** use the README's illustrative `ar.num_hidden_layers` key. |
| Target and AV | 28 Qwen2 blocks, width 3,584, native BF16. Qwen embeddings have scale one. |
| AV prompt | Loaded verbatim from sidecar and passed through the checkpoint's chat template once. Real tokenizer: marker U+320E maps to 149705, neighbors 29/522, absolute injection position 111 in 125 prompt tokens. |
| AV magnitude | Float32 direction rescaled to sidecar injection norm 150, then cast once to BF16 at model input. It is separate from receiver norm restoration. |
| Published AV interface | `NLAClient(checkpoint_dir, sglang_url, injection_scale_override, device)` and `.generate(activation, prompt=None, extract_explanation=True, **sampling)`. Upstream uses embedding-only SGLang requests and requires disabled radix cache. |
| Local AV alternative | Eager Transformers, no server, no token/prefix/KV/result cache; fixed prompt only. Every generation pass rebuilds the suffix after the same injected prefix. Greedy decoding and a 200-token ceiling are our engineering choices. No SGLang numerical-equivalence claim. |
| AR depth | Config has 21 blocks, matching K+1. The checkpoint index contains neither `lm_head.weight` nor `model.norm.weight`; missing versions of those two discarded tensors are permitted, all other loading mismatches fail. |
| AR head | Mandatory `value_head.safetensors`: one 3,584 × 3,584 weight, bias-free linear map. Missing/wrong shape/dtype/nonfinite tensors fail. Both `lm_head` and final norm are replaced by `Identity`; reconstruction calls the inner model and then this separate head. |
| AR text interface | Only description and fixed template. `add_special_tokens=True`; the actual tokenizer has no BOS token. Terminal IDs are `[1318, 29, 366, 1708, 29]`, checked on every description; reconstruct from the final token. |
| Direction score | Sidecar `mse_scale = sqrt(3584) = 59.86651818838306`. Unit-direction squared L2 is `2(1-cosine)`. Raw elementwise MSE is a different quantity. |

Original norm retention is deliberately outside the AR API. Nothing in these
interface checks establishes that an explanation's stated semantics are true.

## Local compatibility evidence

An independent `.venv-phase2` installed successfully on Linux aarch64 / Python
3.11.14: PyTorch `2.10.0+cu130`, Transformers `4.57.6`, safetensors `0.7.0`,
NumPy `2.2.6`, PyYAML `6.0.3`, huggingface-hub `0.36.0`. The source config records
Transformers 4.57.1; the pinned 4.57.6 Qwen implementation is exercised by the
random CPU fixtures. The root environment still uses its pre-existing PyTorch
`2.14.0+cu130`; it was not modified.

GPU discovery found NVIDIA GB10, capability 12.1. The CUDA 13 wheel reports
`sm_80`, `sm_90`, `sm_100`, `sm_110`, `sm_120`, `compute_120` and emits a warning
that its maximum listed capability is 12.0. An 8 × 8 BF16 matrix multiplication
executed correctly. **This is not a full Qwen/AV/AR compatibility result.** Any
unsupported operation during the real smoke remains a blocker to fix before a
pilot, with unchanged scientific tolerances.

The actual tokenizers/configs/sidecars passed the metadata-only preflight. Downloaded
small-file bytes verified: target 11,533,863; AV 15,923,867; AR 15,916,888. No model
weight files were downloaded. The brief makes weight fetch an explicit user step;
the supplied `--fetch-models` command performs it using the lock. There is no
credential blocker for these currently public ungated repositories.

DGX Spark's [documented 128 GB unified memory](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
is a shared CPU/GPU pool. It does not guarantee this implementation's memory use
or throughput. Suggested planning headroom and unmeasured call bounds are in
the [run instructions](../README.md#run-the-eight-group-engineering-smoke).

## Remaining gates

Released-weight hash verification, BF16 target identity, real AV explanations,
real AR directions, cross-backend equivalence and real behavioral preservation
are **NOT RUN**. No scientific pilot or validation was executed. The fixed norm
diagnostic currently uses a smoke median; a scientific calibration norm and PCA
require the separate calibration split in Milestone 2. Confidence intervals and
edit coverage are not fabricated from this engineering work.
