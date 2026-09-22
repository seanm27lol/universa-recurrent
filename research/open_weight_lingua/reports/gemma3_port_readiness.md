# Gemma-3 second-family port readiness — 2026-09-22

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. The Qwen2.5-7B
pipeline asked whether a released NLA description can reconstruct one activation
direction well enough to preserve that measured behavior; this page records the
port of the same engineering gate to a second family — google/gemma-3-12b-it with
the released kitft Gemma-3 NLA pair — and exactly what is and is not established.
**No released-Gemma weight was downloaded and no real Gemma forward has run: the
target repo is license-gated. Everything below is fixture/API/metadata evidence,
not a released-model result.**

## What is ported and CPU-tested

`src/open_weight_lingua/architectures.py` is a small audited registry
(`ArchSpec`): per family it pins the repository trio, the config `model_type`
markers, the attribute path from the loaded model to the decoder block list, the
audited text-stack width/depth, the released extraction block, and the
input-embedding scaling convention. `target.py`, `nla_adapter.py` and
`preflight.py` now resolve those facts from the registry instead of asserting
`qwen2`/`(3584, 28)`/`L20` literally; anything outside the registry fails closed.
The Qwen2 entries resolve to the identical values as before, and the full
pre-existing suite (128 tests) passes unchanged alongside the 21 new
Gemma-3 tests (149 total; see `tests/test_gemma3_arch.py`, random tiny
`Gemma3ForCausalLM`/`Gemma3ForConditionalGeneration` fixtures on CPU float32 —
software checks, not measurements on released weights):

- capture/patch/greedy/`score_answer` equivalence on the real Gemma-3 target
  hook path `model.language_model.layers.N` (the conditional-generation
  wrapper), including the hidden-state index check, the last-block rejection,
  the same-length dummy-suffix bitwise causality gate, and
  padded-bucket == unpadded logits/argmax with sliding-window layers present
  (fixture window 64 < bucket 128, so the sliding path is genuinely exercised);
- the embedding-scale convention: `Gemma3TextScaledWordEmbedding` multiplies
  rows by `sqrt(hidden)` inside the lookup, and `build_embeddings` overwrites
  one post-scale slot at exactly the sidecar `injection_scale`;
- metadata validation against the real fetched sidecars (byte-identical
  fixtures, hash-tied to the lock), the AR depth rule (33 = 32 + 1), the live
  Gemma BOS check in `ar_prompt`, and the AR truncation convention loaded
  through `preflight.load_model` (a checkpoint without `model.norm.weight` /
  `lm_head.weight` loads for role `ar` and is rejected for role `av`);
- lock validation: the shipped pending lock fails `read_lock` closed with an
  actionable message; a completed copy passes and resolves the family; the
  Qwen lock resolves the Qwen family.

`tasks.py` needed no changes (model-agnostic); its chat-template path is
exercised with a Gemma-shaped stub tokenizer in the new tests.

## Architecture deltas verified (2026-09-22, from the fetched artifacts)

| Fact | Qwen2.5-7B family | Gemma-3-12B family |
|---|---|---|
| Target class (transformers 4.57.6) | `Qwen2ForCausalLM` | `Gemma3ForConditionalGeneration` (`AutoModelForCausalLM` maps `gemma3` → this class; verified in the installed 4.57.6 auto-mapping) |
| Target block-list path | `model.model.layers` (28 blocks) | `model.model.language_model.layers` (48 blocks; verified by instantiation) |
| NLA AV/AR class | `Qwen2ForCausalLM`, `model_type qwen2` | `Gemma3ForCausalLM`, `model_type gemma3_text` (from the fetched configs; text-only — the vision tower is not part of the NLA pair) |
| Text width / depth | 3584 / 28 | 3840 / 48 (AV config; AR truncated to 33 = 32+1, matching the released extraction layer) |
| Extraction block | 20 (`hidden_states[21]`) | 32 (`hidden_states[33]`) |
| Embedding scale | none (scale 1) | `sqrt(3840) = 61.96773353931867` inside the embedding lookup; matches sidecar `mse_scale` |
| AV injection scale | 150.0 | 80000.0 — upstream `nla_inference.py` (pinned 38b802a) documents: "Qwen7B: 150. Gemma-3-12B: 80000 (√d embed scaling inflates residual norms)" |
| Injection marker | ㈎ id 149705, neighbors 29/522 | ㈜ id 246566, neighbors 236813/954 (sidecar `tokens` block) |
| AR terminal suffix | `[1318, 29, 366, 1708, 29]` | `[1005, 236813, 655, 6011, 236813]` (decodes to the `</text> <summary>` pieces) |
| AR value head | 3584×3584, 25,690,200 B | 3840×3840, 29,491,288 B (size consistent with BF16 square head) |
| Tokenizer | BPE (`tokenizer.json`) | `GemmaTokenizerFast` from `tokenizer.json`; `tokenizer.model` (SentencePiece) ships in AV only and is not needed at runtime |
| BOS/EOS/PAD | no BOS; EOS 151645 | BOS 2 live (`add_bos_token: true`); EOS 1 (`<eos>`); PAD 0 |

Quoted from the real fetched `kitft/nla-gemma3-12b-L32-av/raw/main/nla_meta.yaml`:
`schema_version: 2`, `role: av`, `d_model: 3840`, `extraction_layer_index: 32`,
`extraction: {injection_scale: 80000.0, mse_scale: 61.96773353931867}`,
`tokens: {injection_char: ㈜, injection_token_id: 246566,
injection_left_neighbor_id: 236813, injection_right_neighbor_id: 954,
critic_suffix_ids: null}`. The AR sidecar matches it pairwise and adds
`critic_suffix_ids: [1005, 236813, 655, 6011, 236813]` and
`critic.extraction_layer_index: 32`. Both fixtures are byte-identical copies.

## Interleaved attention and the L32 site

Gemma-3 interleaves local (sliding-window) and global (full) attention in a 5:1
pattern. Verified from the fetched AV config's `layer_types` (48 entries):
`full_attention` at indices 5, 11, 17, 23, 29, 35, 41, 47 — every 6th block —
and `sliding_attention` elsewhere, with `sliding_window: 1024`. The AR keeps
blocks 0–32 of the same pattern (full at 5, 11, 17, 23, 29).

**Block 32 is a LOCAL (sliding-window) block** (`layer_types[32] ==
"sliding_attention"`; asserted by a test against the fetched configs). For our
machinery this changes nothing, for a stated reason: every target forward is
right-padded to the fixed `TARGET_BUCKET = 128`, and 128 < 1024, so the window
never truncates — inside a pinned forward, a sliding block is exactly causal
over the full padded length. `create_sliding_window_causal_mask` masks pads
with the same finite-minimum additive bias as the full-attention mask, so
masked pads still contribute exactly zero, the same-length dummy-suffix
causality gate stays bitwise, and kernel-shape pinning is unaffected. The
tiny-fixture equivalence test runs with a 64-token window against the 128-token
bucket to keep this honest. Had the bucket exceeded the window, late positions
in local blocks could no longer see early prefix tokens; the runner's
`check_target_bucket_fit` fails loudly long before that regime.

## The new lock

`configs/model-lock-gemma3-12b.json` (schema_version 1, same shape as
`configs/model-lock.json`, which is untouched):

| Role | Repository | Revision | Locked bytes |
|---|---|---|---:|
| target | `google/gemma-3-12b-it` (**GATED — manual license**) | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | 24,414,161,479 |
| av | `kitft/nla-gemma3-12b-L32-av` (public) | `7aec22599e8a9cd533564868999b443fcc963cf4` | 23,571,432,021 |
| ar | `kitft/nla-gemma3-12b-L32-ar` (public) | `3d6901d8243182d642af9dea452ca91549a94615` | 16,871,717,079 |

Total 64,857,310,579 bytes (60.4 GiB). All three repos carry `license: gemma`
in their card data — the Gemma license, not Apache-2.0; the NLA pair
additionally ships Google's NOTICE.

Hash provenance, stated plainly: every `.safetensors` shard and both LFS
tokenizer files carry the Hub API's LFS sha256 (2026-09-22; not yet verified
against downloaded weights, same caveat as the Qwen lock). Every non-LFS AV/AR
file was downloaded and hashed locally. The AV `tokenizer.json` /
`tokenizer.model` downloads were hashed locally and agree with the LFS
metadata; all three repos reference the byte-identical tokenizer blobs. The
pinned GitHub source files were re-downloaded and re-hashed; all five match the
Qwen lock's `sources` block, which is carried over unchanged.

**Pending hashes:** the gated target's ten small non-LFS files (`config.json`,
`tokenizer_config.json`, `chat_template.json`, `generation_config.json`,
`special_tokens_map.json`, `added_tokens.json`, `preprocessor_config.json`,
`processor_config.json`, `model.safetensors.index.json`, `README.md`) cannot be
content-hashed without license acceptance; unauthenticated fetches return
HTTP 401 (verified 2026-09-22), and public mirrors demonstrably diverge from
the gated bytes in exactly these files (unsloth's `config.json` is 1,660 bytes
vs Google's 916), so no hash is asserted for them. Their entries carry
`"sha256": null` and exact pinned sizes; `read_lock` refuses the pending lock
(fail closed — nothing runs on unverified sources) and names the completion
command. `scripts/complete_gemma3_lock.py` performs the one-time completion:
it downloads exactly the pending files at the pinned revision with the user's
token, cross-checks sizes, fills only the null hashes, and rewrites the lock
only after the completed version passes `read_lock`. It refuses to touch a
lock that is already complete.

## Real-tokenizer validation already done (public artifacts, 2026-09-22)

Using the pinned AV/AR tokenizer files (ungated) through the pipeline's own
`av_prompt` / `ar_prompt` / `answer_tokens` / `tokenize_groups`: AV prompt is
108 tokens with the marker at position 93 flanked by 236813/954, BOS-first
(Gemma's chat template includes `<bos>`); the AR prompt starts with BOS 2 and
ends with the sidecar suffix; `answer_tokens` round-trips "0"/"5"/"19" with
EOS 1; four real smoke groups tokenize to 40–53-token prompts with the shared
boundary token 107 (`\n` after `<start_of_turn>model`), passing
`check_target_bucket_fit`. The gated target's own `chat_template.json` (1,615
bytes) differs in bytes from the AV/AR `chat_template.jinja` (1,532 bytes) and
is verified only after unlock; `tokenize_groups` fails loudly if its
source/donor boundary tokens ever disagree.

**Watch item for the first real smoke (decide, don't silently fix):** the
pipeline's termination convention is the tokenizer's single `eos_token_id`
(1, `<eos>`). The released AV `generation_config.json` lists
`eos_token_id: [1, 106]` — Gemma-it models also terminate turns with
`<end_of_turn>` (106), and the gated target is expected to share that
convention. Greedy generations that end at 106 instead of 1 count as
unterminated (intention-to-test incorrect), which could depress measured P0
accuracy and verbalizer stop behavior on Gemma relative to Qwen. The
convention is deliberately unchanged from the audited pipeline; the smoke
stage's P0 accuracy and AV status counts are the measurement that decides
whether a protocol amendment is needed.

## Exact commands once the token exists

Prerequisite blocker: a human must accept the terms at
https://huggingface.co/google/gemma-3-12b-it and export `HF_TOKEN` for an
account with access. Then, from the worktree root
(`/home/seanjazm27/projects/universa-recurrent-gemma`):

```bash
# 0. one-time: fill the ten pending gated hashes (uses HF_TOKEN), proving the
#    completed lock passes read_lock before it replaces the file
.venv-phase2/bin/python research/open_weight_lingua/scripts/complete_gemma3_lock.py

# 1. smoke (creates/syncs .venv-phase2, runs the CPU suite, fetches + verifies
#    the pinned artifacts, runs 8 groups / 32 variants)
bash research/open_weight_lingua/scripts/run_smoke.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json --fetch-models

# 2. calibration (256 groups; fits the P4 baseline, freezes the median norm)
bash research/open_weight_lingua/scripts/run_calibration.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json

# 3. pilot (128 groups; the locked-validation stage remains unimplemented here)
bash research/open_weight_lingua/scripts/run_pilot.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --calibration-fit research/open_weight_lingua/runs/<calibration-run>/baseline_fit.safetensors
```

(If `.venv-phase2` does not exist yet in this worktree, step 0 can use any
Python with `huggingface_hub`; step 1 creates it via `uv sync --locked`.)

## Memory and storage planning (planning numbers, not measurements)

Locked artifacts total 60.4 GiB; `fetch_models` additionally requires pinned
bytes plus 5 GiB of free storage before downloading, so plan ≥ 75 GiB free.
The runner loads one model per stage and releases between stages, so the
largest active BF16 weight set is the 22.7 GiB target (loaded twice, in
separate stages); AV is 21.95 GiB and AR 15.71 GiB. Against the documented
128 GB unified CPU/GPU pool this is comfortable but unmeasured: suggest
keeping ≥ 80 GiB of the pool free for load transients, fp32 copies and the
262,208-token logit buffers until the smoke records actual peaks
(`compatibility.json` carries max RSS and CUDA peak allocations). Forward-call
counts are unchanged from the Qwen smoke design (at most 6,400 AV, 32 AR, 4,032
target calls for eight groups); per-call cost at 12B/48-blocks is higher than
at 7B/28-blocks and is unmeasured — do not authorize the pilot from these
numbers.

## Limits

This is an engineering port with fixture-level and metadata-level evidence.
The Qwen2.5-7B results and lock are unchanged. The pending-hash completion,
the real identity gate, AV explanations, AR directions, behavioral
preservation, the termination-convention watch item above, and any scientific
read of a Gemma pilot are all NOT RUN. The frozen pilot thresholds were locked
for the Qwen phase; applying them to Gemma is a new pilot decision, and the
locked validation stage remains unimplemented in this runner for both
families.
