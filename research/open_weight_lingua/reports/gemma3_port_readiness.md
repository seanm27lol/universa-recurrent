# Gemma-3 second-family port readiness — 2026-09-22

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. The Qwen2.5-7B
pipeline asked whether a released NLA description can reconstruct one activation
direction well enough to preserve that measured behavior; this page records the
port of the same engineering gate to a second family — Gemma-3-12B-it with the
released kitft Gemma-3 NLA pair — and exactly what is and is not established.
**Update 2026-09-22 (second revision): the target now pins the public unsloth
mirror, all artifacts fetch and hash-verify through the pipeline's own path, and
fixture/tokenizer/config checks are green. Real-weight GPU stages remain
NOT RUN — they are handed off with the exact commands below.**

## What is ported and CPU-tested

`src/open_weight_lingua/architectures.py` is a small audited registry
(`ArchSpec`): per family it pins the repository trio, the config `model_type`
markers, the attribute path from the loaded model to the decoder block list, the
audited text-stack width/depth, the released extraction block, and the
input-embedding scaling convention. `target.py`, `nla_adapter.py` and
`preflight.py` resolve those facts from the registry instead of asserting
`qwen2`/`(3584, 28)`/`L20` literally; anything outside the registry fails
closed. The Qwen2 entries resolve to the identical values as before, and the
full pre-existing suite (128 tests) passes unchanged alongside the 21 Gemma-3
tests (149 total; `tests/test_gemma3_arch.py`, random tiny
`Gemma3ForCausalLM`/`Gemma3ForConditionalGeneration` fixtures on CPU float32 —
software checks, not measurements on released weights):

- capture/patch/greedy/`score_answer` equivalence on the real Gemma-3 target
  hook path `model.language_model.layers.N`, including the hidden-state index
  check, the last-block rejection, the same-length dummy-suffix bitwise
  causality gate, and padded-bucket == unpadded logits/argmax with
  sliding-window layers present (fixture window 64 < bucket 128);
- the embedding-scale convention: `Gemma3TextScaledWordEmbedding` multiplies
  rows by `sqrt(hidden)` inside the lookup, and `build_embeddings` overwrites
  one post-scale slot at exactly the sidecar `injection_scale`;
- metadata validation against the real fetched sidecars (byte-identical
  fixtures, hash-tied to the lock), the AR depth rule (33 = 32 + 1), the live
  Gemma BOS check in `ar_prompt`, and the AR truncation convention loaded
  through `preflight.load_model`;
- lock validation: the mirror lock passes `read_lock` and resolves the family,
  the official-revision metadata rides alongside, the weight shards/tokenizer
  blobs are asserted byte-identical between the mirror pin and the official API
  record, and a null hash anywhere still fails closed.

`tasks.py` needed no changes (model-agnostic); its chat-template path is
exercised with a Gemma-shaped stub tokenizer in the tests and with the real
mirror tokenizer below.

## Architecture deltas verified (2026-09-22, from the fetched artifacts)

| Fact | Qwen2.5-7B family | Gemma-3-12B family |
|---|---|---|
| Target class (transformers 4.57.6) | `Qwen2ForCausalLM` | `Gemma3ForConditionalGeneration` (`AutoModelForCausalLM` maps `gemma3` → this class; meta-device `from_config` on the pinned mirror config constructs the wrapper with 48 text layers at `model.language_model.layers` and a 27-layer SigLIP vision tower) |
| Target block-list path | `model.model.layers` (28 blocks) | `model.model.language_model.layers` (48 blocks; verified by instantiation) |
| NLA AV/AR class | `Qwen2ForCausalLM`, `model_type qwen2` | `Gemma3ForCausalLM`, `model_type gemma3_text` (text-only; no vision tower) |
| Text width / depth | 3584 / 28 | 3840 / 48 (target text_config and AV config; AR truncated to 33 = 32+1) |
| Extraction block | 20 (`hidden_states[21]`) | 32 (`hidden_states[33]`) |
| Embedding scale | none (scale 1) | `sqrt(3840) = 61.96773353931867` inside the embedding lookup; matches sidecar `mse_scale` |
| AV injection scale | 150.0 | 80000.0 — upstream `nla_inference.py` (pinned 38b802a): "Qwen7B: 150. Gemma-3-12B: 80000 (√d embed scaling inflates residual norms)" |
| Injection marker | ㈎ id 149705, neighbors 29/522 | ㈜ id 246566, neighbors 236813/954 |
| AR terminal suffix | `[1318, 29, 366, 1708, 29]` | `[1005, 236813, 655, 6011, 236813]` (the `</text> <summary>` pieces) |
| AR value head | 3584×3584, 25,690,200 B | 3840×3840, 29,491,288 B (size consistent with BF16 square head) |
| Tokenizer | BPE (`tokenizer.json`) | `GemmaTokenizerFast`; the same `tokenizer.json`/`tokenizer.model` blobs (LFS-identical) in all three repos |
| BOS/EOS/PAD | no BOS; EOS 151645 | BOS 2 live (`add_bos_token: true`); EOS **differs by role — see below**; PAD 0 |
| Checkpoint key naming | `model.layers.N…` | target: `language_model.model.layers.N…` + vision tower keys (renamed on load by the 4.57.6 `_checkpoint_conversion_mapping`); NLA pair: `model.layers.N…` |

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

## The lock: mirror-sourced target, official metadata alongside

`configs/model-lock-gemma3-12b.json` (schema_version 1, same shape as
`configs/model-lock.json`, which is untouched):

| Role | Repository | Revision | Locked bytes |
|---|---|---|---:|
| target | `unsloth/gemma-3-12b-it` (public mirror) | `9478e665381f42974aa06177b019352fb6291876` | 24,414,165,251 |
| av | `kitft/nla-gemma3-12b-L32-av` (public) | `7aec22599e8a9cd533564868999b443fcc963cf4` | 23,571,432,021 |
| ar | `kitft/nla-gemma3-12b-L32-ar` (public) | `3d6901d8243182d642af9dea452ca91549a94615` | 16,871,717,079 |

Total 64,857,314,351 bytes (60.4 GiB). All three repos carry `license: gemma`.
**Gemma Terms of Use apply to the user regardless of download source; the HF
gate is an access mechanism, not the license itself.**

Mirror provenance, stated plainly (the same text rides in the lock's
`provenance` field): the official google/gemma-3-12b-it @
`96b6f1eccf38110c56df3a15bffe176da04bfd80` is gated-manual; anonymous fetches
returned HTTP 401 on 2026-09-22, so its small non-LFS files could not be
byte-verified. The five weight shards and both tokenizer blobs carry identical
LFS sha256 in both repos' own API records, so those 24.37 GB are byte-identical
between mirror and official and are hash-verified on download. The mirror's
`tokenizer.model` was additionally byte-compared against the already-fetched
AV blob (`cmp`, identical); its `chat_template.jinja` is byte-identical to the
AV/AR template file, and its `chat_template.json` and embedded
`tokenizer_config.chat_template` carry the same template text. The target
entry records the official revision, gating and API-listed file inventory
under `official_source` for later reconciliation; if gated access is later
obtained, re-verify against the official revision and reconcile.

Divergence is confined to four small files (mirror vs official API sizes):
`config.json` 1,660 vs 916 B, `generation_config.json` 210 vs 215 B,
`special_tokens_map.json` 670 vs 662 B, `tokenizer_config.json` 1,158,492 vs
1,156,999 B. Field-level inspection of the mirror's versions:

- `config.json`: standard `Gemma3ForConditionalGeneration` schema; text stack
  3840/48, sliding window 1024 pattern 6, SigLIP 1152/27 vision tower — all
  matching the audited registry entry — plus `"unsloth_fixed": true` and
  top-level `eos_token_id: 106`. No quantization fields; `torch_dtype:
  bfloat16`. The official 916 B file is terser (relies on library defaults);
  nothing in the mirror's additions alters the text-stack fields the pipeline
  audits, and the runner never reads generation defaults.
- `generation_config.json`: `eos_token_id: [1, 106]`, sampling defaults —
  unused by this pipeline (greedy argmax, explicit tokenizer EOS).
- `special_tokens_map.json` / `tokenizer_config.json`: the mirror declares
  `eos_token: <end_of_turn>`; the AV/AR pair declares `eos_token: <eos>`.
  Consequence: per role, `tokenizer.eos_token_id` is 106 for the target and 1
  for the NLA pair. The pipeline reads the per-role tokenizer, so target
  greedy/scoring terminates on `<end_of_turn>` — the native gemma-it turn
  terminator — while the AV/AR paths follow the released NLA convention. This
  is self-consistent within each role and resolves the previously flagged
  target-side termination concern in favor of the -it convention. **Remaining
  watch item (unchanged, AV side only):** our local AV reader stops at
  `eos_token_id` 1 (`<eos>`), while the AV's own `generation_config.json`
  allows [1, 106]; if the released AV prefers 106 at turn end, descriptions
  would read as truncated. The first smoke's AV status counts measure this;
  do not silently patch decoding.

Hash provenance otherwise unchanged from the Qwen lock's policy: weight shards
carry the Hub API's LFS sha256 (verified against the downloaded bytes by
`verify_models` on fetch); every non-LFS file of all three roles was downloaded
and hashed locally this time (the mirror is public, so no hash is metadata-only
except the five shards and `tokenizer.json`, whose LFS hashes are corroborated
by the AV download's local hash). The pinned GitHub source files were
re-downloaded and re-hashed; all five match the Qwen lock's `sources` block,
which is carried over unchanged.

## Sanity checks on the mirror weights without the official bytes

- Index/shard consistency: `model.safetensors.index.json` lists 1,065 tensors
  across exactly the five pinned shards; 48 text layers, complete SigLIP tower
  and projector; payload total 24,374,650,080 B consistent with the shard file
  sizes; tensor names follow the wrapper layout the pinned transformers renames
  on load.
- Total bytes: mirror target 24,414,165,251 vs official API listing
  24,414,161,479 — delta 3,772 B, exactly the four divergent small files; the
  weight shards and tokenizer blobs are LFS-identical.
- Dtype: `config.json` declares `torch_dtype: bfloat16`; per-tensor dtype is
  re-checked from the downloaded shard headers after fetch (see below).
- Load far enough to report its config: `AutoConfig` parse plus a meta-device
  `AutoModelForCausalLM.from_config` on the pinned files yields
  `Gemma3ForConditionalGeneration` with 48 text layers at the registry path —
  no weights materialized.

## Tokenizer/template checks with the real (mirror) files

`GemmaTokenizerFast` assembled from the pinned mirror files: BOS 2 first on
chat prompts, boundary token 107 (`\n` after `<start_of_turn>model`) shared
across all variants of four real smoke groups (lengths 40–53, bucket-fit PASS),
`answer_tokens("19")` → `[236770, 236819, 106]` = "1", "9", `<end_of_turn>`,
round-tripping. The AV/AR checks from the first revision (marker/neighbor
positions, AR suffix, live BOS) are unchanged and still pass.

## Exact commands for the real stages (hand-off; not run here)

From the worktree root (`/home/seanjazm27/projects/universa-recurrent-gemma`),
with the hash-verified model cache already populated (see below):

```bash
# smoke (8 groups / 32 variants; runs the CPU suite first)
bash research/open_weight_lingua/scripts/run_smoke.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --cache research/open_weight_lingua/model-cache

# calibration (256 groups; fits the P4 baseline, freezes the median norm)
bash research/open_weight_lingua/scripts/run_calibration.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --cache research/open_weight_lingua/model-cache

# pilot (128 groups; locked validation remains unimplemented)
bash research/open_weight_lingua/scripts/run_pilot.sh \
  --lock research/open_weight_lingua/configs/model-lock-gemma3-12b.json \
  --cache research/open_weight_lingua/model-cache \
  --calibration-fit research/open_weight_lingua/runs/<calibration-run>/baseline_fit.safetensors
```

`--cache` is the runner's default (`research/open_weight_lingua/model-cache`);
it is spelled out here for clarity. The completion script
(`scripts/complete_gemma3_lock.py`) is no longer on the run path — it refuses
the now-complete mirror lock and remains only as the reconciliation tool for a
future official-sourced lock.

## Fetch and verification record

Fetch ran through the pipeline's own path
(`python -m open_weight_lingua.preflight --lock configs/model-lock-gemma3-12b.json
--cache model-cache --fetch-models --device cpu`) on 2026-09-22:
`fetch_models` downloaded only the pinned files at the pinned revisions
(64,857,314,351 new bytes in 2,124 s), `verify_models` re-hashed every byte
against the lock (PASS), and `inspect_metadata` passed on the real artifacts
(width 3840, extraction block 32, injection scale 80000.0, AR 33 layers, AR
suffix `[1005, 236813, 655, 6011, 236813]`, AV prompt 108 tokens with the
marker at position 93).

| Role | Verified bytes | Result |
|---|---:|---|
| target | 24,414,165,251 | PASS (sha256 + size, every file) |
| av | 23,571,432,021 | PASS (sha256 + size, every file) |
| ar | 16,871,717,079 | PASS (sha256 + size, every file) |

Post-fetch shard inspection (headers via `safe_open`, no weight
materialization): every tensor in all 14 shards is BF16; per-role tensor
payloads match the index `total_size` exactly (target 1,065 tensors / 5 shards
including the SigLIP tower, 12.19 B parameters total; AV 626 / 5; AR 430 / 4);
block 32 contributes its 13 tensors in each role's index.

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

This is an engineering port with fixture-level, metadata-level and
download-verification evidence. The Qwen2.5-7B results and lock are unchanged.
The mirror's weight bytes are LFS-identical to the official repo's listed
hashes; the four divergent small files are the mirror's own and are pinned as
such — a future official-access reconciliation may swap them. Real
released-Gemma forwards, the identity gate, AV explanations, AR directions,
behavioral preservation, and any scientific read of a Gemma pilot are all
NOT RUN. The frozen pilot thresholds were locked for the Qwen phase; applying
them to Gemma is a new pilot decision, and the locked validation stage remains
unimplemented in this runner for both families.
