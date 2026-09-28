# NLA ecosystem reconnaissance notes — 2026-09-22

Reconnaissance for the post-pilot phase decision. The pinned pilot on the released
Qwen2.5-7B L20 NLA pair ended STOP (P2 accuracy loss 34.4 pp vs the 5 pp bound;
0/128 edit-eligible groups; P4 PCA baseline at 0.803 vs P0 0.811 — see
[ milestone_two.md ](milestone_two.md) addendum). This note surveys what else
exists in the NLA ecosystem and what each follow-up option would cost.

**Honesty register.** Every claim below is tagged:

- **[inspected 2026-09-22]** — fetched and read today from the cited URL
  (HF Hub API/raw files, GitHub API/raw files, the paper HTML).
- **[inferred]** — my estimate or deduction; not a measured or published fact.

Local fixture check: `tests/fixtures/upstream/av-nla_meta.yaml` is **byte-identical**
(`diff` empty) to `kitft/nla-qwen2.5-7b-L20-av/raw/main/nla_meta.yaml` fetched today
[inspected 2026-09-22].

## 1. Released artifacts (HF Hub, author `kitft`)

`https://huggingface.co/api/models?author=kitft` returns **exactly 8 models, no
datasets** [inspected 2026-09-22]. They form four AV/AR pairs, one layer each,
all at roughly two-thirds depth of the base model. **No other layers of
Qwen2.5-7B exist** (no L10/L14/L24), and no other Qwen sizes. The same eight are
the members of the collection
[`kitft/nla-models`](https://huggingface.co/collections/kitft/nla-models)
[inspected 2026-09-22]. A HF tag search (`?tags=nla`, `?tags=activation-decoding`)
surfaced no third-party NLA pairs; substring noise only [inspected 2026-09-22].

| repo id | base model | layer | total size | license (tag) | created | lastModified |
|---|---|---|---|---|---|---|
| `kitft/nla-qwen2.5-7b-L20-av` | Qwen/Qwen2.5-7B-Instruct | 20/28 | 15.25 GB (18 files) | apache-2.0 | 2026-03-16 | 2026-05-07 |
| `kitft/nla-qwen2.5-7b-L20-ar` | " | 20/28 | 10.92 GB (18 files) | apache-2.0 | 2026-03-16 | 2026-05-07 |
| `kitft/nla-gemma3-12b-L32-av` | google/gemma-3-12b-it | 32/48 | 23.57 GB (18 files) | gemma | 2026-03-30 | 2026-05-07 |
| `kitft/nla-gemma3-12b-L32-ar` | " | 32/48 | 16.87 GB (16 files) | gemma | 2026-03-30 | 2026-05-07 |
| `kitft/nla-gemma3-27b-L41-av` | google/gemma-3-27b-it | 41/62 | 108.08 GB (38 files) | gemma | 2026-04-25 | 2026-05-07 |
| `kitft/nla-gemma3-27b-L41-ar` | " | 41/62 | 37.60 GB (22 files) | gemma | 2026-04-25 | 2026-05-07 |
| `kitft/Llama-3.3-70B-NLA-L53-av` | meta-llama/Llama-3.3-70B-Instruct | 53/80 | 141.12 GB (40 files) | llama3.3 | 2026-04-24 | 2026-05-07 |
| `kitft/Llama-3.3-70B-NLA-L53-ar` | " | 53/80 | 94.80 GB (32 files) | llama3.3 | 2026-04-25 | 2026-05-07 |

Sizes summed from the Hub API blob inventory (`?blobs=true`) [inspected
2026-09-22]. Every repo carries an `nla_meta.yaml`; AR repos additionally carry
`value_head.safetensors` (the d×d reconstruction head). Gemma repos are
**gated** (need `HF_TOKEN` + license acceptance; per model cards and
docs/setup.md [inspected 2026-09-22]); the Gemma AVs are the multimodal
`Gemma3ForConditionalGeneration` wrapper (`config.text_config`,
`model.language_model`), per the nla-inference README [inspected 2026-09-22].

### `nla_meta.yaml` schema comparison vs our fixture

All eight sidecars fetched and compared today [inspected 2026-09-22]:

- **Qwen L20 AV**: byte-identical to our fixture (kind, schema_version 2, role,
  stage, d_model 3584, extraction{injection_scale 150.0, mse_scale √d}, tokens{
  injection_char ㈎ id 149705, neighbors 29/522, critic_suffix_ids},
  prompt_templates{av, ar}, created_at, created_by `nla.train_actor.NLAFSDPActor`,
  training{rollout_id, lr, loss_type, global_batch_size}, extraction_layer_index 20).
- **Qwen L20 AR / both Gemma pairs**: same schema_version-2 schema; ARs add
  `critic.extraction_layer_index` and non-null `critic_suffix_ids` (fixture already
  documents this AR/AV asymmetry). Differences are values only: d_model
  3840/5376, injection_scale 80000.0 (Gemma-12B) / 60000.0 (Gemma-27B),
  injection_char ㈜ id 246566, neighbors 236813/954, layer 32/41.
- **Llama-3.3-70B pair**: schema_version 2 with **additive fields not in the
  fixture**: `layer: 53`, `role_aliases: {verbalizer: actor, recon: critic}`,
  `training.num_layers` (80 AV / 54 AR), `created_by:
  nla.megatron.train_actor.NLAMegatronActor`, and **no `training.rollout_id`**.
  injection_scale 30.0, injection_char ㎡ id 105565, neighbors 29/524.

Model cards report in-distribution `fve_nrm`: Gemma-3-12B **0.768**, Gemma-3-27B
**0.763**, Llama-3.3-70B **0.80** (training set, 50/50 WildChat + Ultra-FineWeb)
[inspected 2026-09-22, per-repo README.md]. Training-data attribution in every
card: WildChat-1M (ODC-BY) + Ultra-FineWeb (Apache-2.0, from FineWeb ODC-BY).

Other distribution site: the paper's Neuronpedia collaboration hosts an
interactive sampler at `https://www.neuronpedia.org/nla` (HTTP 200 today); it is
a frontend over the same four pairs, not additional weights [inspected
2026-09-22].

## 2. The paper: what was trained/released, at what cost

Source: [Natural Language Autoencoders Produce Unsupervised Explanations of LLM
Activations](https://transformer-circuits.pub/2026/nla/index.html), full HTML
fetched and text-extracted today [inspected 2026-09-22].

- **Released**: "training code and trained NLAs for four open models:
  Qwen-2.5-7B, Gemma-3-12B, Gemma-3-27B, and Llama-3.3-70B" — exactly the four
  pairs above. Claude NLAs (Haiku 3.5, Haiku 4.5, Opus 4.6; trained at a layer
  two-thirds through each model) are **not** released.
- **Layer sensitivity is documented in the paper itself**: an NLA trained at
  two-thirds depth on a reward-sycophancy model organism showed no
  reward-model signal, while one trained at roughly half depth did; the
  late-layer NLA applied to *earlier* layers' activations also surfaced the
  terms. So "try another layer" is a paper-acknowledged axis — but no
  alternate-layer checkpoints are released; using one means training.
- **Training data scale (open models)**: 100,000 UltraFineWeb-mix documents × 5
  activation vectors/document (max context 4,096) = 500,000 (context, summary)
  pairs, split evenly by document between AV and AR warm-start; summaries
  generated by **Claude Sonnet 4.6**, "250,000 API calls for each of the AV and
  AR datasets" (Gemma-3 models), averaging ~2,000 input tokens/call, outputs
  reused across tokenizer-compatible models. RL drew a further 500,000
  UltraFineWeb + 500,000 WildChat activations.
- **Training compute (open models)**: GRPO, group G=8; RL ran 3,000 steps on
  **two 8×H100 nodes**; "approximately 1.5 days to reach 70% FVE and one week to
  reach 75% FVE (approximately $1500–$5000 of compute)" — stated for Gemma-3-27B
  ("NLA RL on Gemma-3-27B required 1.5 days on two 8xH100 nodes to increase from
  0.38 to 0.71 FVE"). SFT: batch 256, Adam, lr 1e-5; RL: batch 128 prompts, lr
  1e-5, 150-token rollout cap, several thousand steps. The authors state they
  did not sweep batch/LR/group size and "do not regard our configurations as
  near-optimal."
- **Training code released**: yes — github.com/kitft/natural_language_autoencoders
  (Apache-2.0), full datagen → SFT → RL → conversion pipeline (§3).

## 3. Training repo: requirements and DGX Spark feasibility

Repo [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders),
**Apache-2.0** [inspected 2026-09-22 via GitHub API: license field, repo tree,
and raw files]. Created 2026-05-05, last push 2026-08-02 (13 commits; latest
trim KL comments / pass `--kl-loss-type k2`). Open items today: PR #3
"Normalize apply_chat_template return type for transformers>=5" (open),
issue #2 "Constrained-vocabulary NLA experiment with NSM semantic primes"
(community result sharing) [inspected 2026-09-22].

Pipeline ([README](https://github.com/kitft/natural_language_autoencoders/blob/main/README.md),
[docs/setup.md](https://github.com/kitft/natural_language_autoencoders/blob/main/docs/setup.md),
[configs/TRAINING_NOTES.md](https://github.com/kitft/natural_language_autoencoders/blob/main/configs/TRAINING_NOTES.md),
all inspected 2026-09-22):

1. **Datagen** (`nla/datagen/`): 4-stage pipeline — extract activations at one
   layer (GPU), split 25/25/50 (AV-SFT/AR-SFT/RL), generate summaries via the
   **Anthropic API** (`ANTHROPIC_API_KEY` required; shipped
   `configs/datagen/qwen7b_fineweb_1M.yaml` uses `claude-haiku-4-5-20251001`,
   concurrency 400 — note the paper says Sonnet 4.6 for the released data; the
   shipped config differs), build parquets. Scale: the 1M config extracts ~1M
   vectors (100k docs × 10 positions); TRAINING_NOTES describes the released
   recipe as 100k docs × 5 vectors = 500k pairs.
2. **AR SFT then AV SFT** via Miles (Ray + FSDP2) on pinned
   `radixark/miles@051cd15` + NLA integration patches. Qwen7B reference measured
   on **2×H100-80GB**: actor SFT step 4.97s at micro-batch 16 with FA2 and **no
   gradient checkpointing**, **peak 67–80GB per GPU**; actor OOMs at micro-batch
   ≥32 without checkpointing. Critic SFT ~3s/step, peak ~67GB.
3. **RL**: simultaneous AV GRPO + AR supervised; production Qwen7B run on
   **2×8×H100**; defaults 8 actor + 4 critic + 4 rollout GPUs; single-node
   fallback documented as ACTOR_GPUS=4 CRITIC_GPUS=2 ROLLOUT_GPUS=2 (i.e. one
   8-GPU node minimum in the documented configurations). Rollout served by
   **SGLang from a source checkout** with the repo's patches; stock
   `sglang[all]>=0.5.6` suffices for inference only.

**Could this run on one DGX Spark (GB10, 128 GB unified, ARM64)?** — tagged
assessment:

- [inspected 2026-09-22] The authors' own floor is 2×H100-80GB (160GB) for 7B
  SFT and 8–16 H100s for RL; peak per-GPU memory 67–80GB *with FSDP sharding
  across 2 GPUs*.
- [inferred] Full-parameter Adam SFT of a 7B on a single device needs roughly
  14GB (bf16 weights) + 28GB (fp32 master) + 56GB (Adam m/v) + ~14GB gradients ≈
  112GB static, plus activations and the ~4.8GB lm_head logits — over or at the
  edge of 128GB unified memory (which also houses the OS and, in RL, the rollout
  server and the AR). Feasible only with changes the repo does not ship:
  8-bit/paged optimizers or gradient checkpointing (documented as 36% slower and
  deadlock-prone in their RL path), much smaller micro-batches.
- [inferred] RL as architected (actor trainer + AR trainer + SGLang rollout,
  three concurrent 7B-class residents, 20GB+ each just for weights, plus
  optimizer states for two of them) does not fit 128GB without deep rework;
  sequentializing stages violates their "keep training on-policy / AR learns
  alongside" design.
- [inspected 2026-09-22] Platform support is partial: `sgl-kernel` 0.3.21 on
  PyPI ships `manylinux2014_aarch64` wheels, but `flash-attn` 2.8.3.post1 is
  **source-only** (their SFT config wants FA2; SDPA fallback exists and was
  their slowest measured config), and GB10 is sm_121 — our own stack already
  warns PyTorch cu130 lists max capability 12.0 (milestone reports). Nothing
  about Miles/SGLang on sm_121 is verified by anyone cited here.
- [inspected 2026-09-22] API cost floor: ≥500k summary calls (250k AV + 250k
  AR) at ~2,000 input tokens each; [inferred] at Sonnet-class API prices this
  alone is in the low thousands of USD, before any GPU time.

Verdict for a 7B-class target on one Spark: datagen stage 0 (extraction) is
realistic (it is inference-scale, and we already run 7B inference here);
summary generation is an API-budget question; SFT is marginal-to-infeasible as
configured and would need memory surgery; the RL stage is **not realistic as
documented** on a single 128GB node [inferred].

## 4. nla-inference status

Repo [`kitft/nla-inference`](https://github.com/kitft/nla-inference),
Apache-2.0: **a single commit (38b802a3, 2026-05-07), no updates since**
[inspected 2026-09-22]. We pin exactly that commit in
`reports/source_compatibility.md`. Pins: `sglang[all]>=0.5.6`,
`torch<2.11` or cu124 wheel index (CUDA/torch pin note in docs/setup.md
[inspected 2026-09-22]).

SGLang embedding-injection constraint (per the nla-inference README +
GitHub PR API, all [inspected 2026-09-22]):

- Stock sglang≥0.5.6 `/generate` accepts `input_embeds`; `--disable-radix-cache`
  mandatory. This constraint is why **our runner does not use SGLang at all**
  (local eager-Transformers adapter, no server — see
  `reports/source_compatibility.md` and `THIRD_PARTY.md`).
- Correctness fixes **merged upstream**: PR #20376 (slice input_embeds on
  chunk-overflow, merged 2026-03-15), PR #14110 (retract-path output_ids reset,
  merged 2026-03-11).
- Throughput PRs #20205/#20206/#20207 (numpy IPC, skip-validation, bytes
  transport) are **closed unmerged**; relevant only past ~10 req/s.
- Gemma-3 AVs need the multimodal bypass patch
  (`patches/nla_gemma3_mm_input_embeds.patch`) under SGLang; under a local
  Transformers path the equivalent work is the `text_config` nesting + √d
  `embed_scale` handling.
- No notes naming checkpoints beyond the four pairs.

## 5. Adjacent alternatives

- **Cycle-Consistent Activation Oracles (Chalnev)** — concurrent independent
  work cited by the paper: a verbalizer–reconstructor pair with supervised
  warm-start trained by RL under a KL penalty. Released artifacts today: only an
  output-examples repo
  ([`slavachalnev/cco_examples`](https://github.com/slavachalnev/cco_examples),
  HTML demo pages, no license file, no weights, no training code) [inspected
  2026-09-22]. Not a usable checkpoint source.
- **Activation oracles / LatentQA line** (Pan et al., Karvonen et al., Costarelli
  et al., Choi et al. — cited in the paper's related work): supervised models
  that *answer questions* about activations rather than autoencode them. The
  paper itself reports NLA-initialized AVs are good AO initializations but a
  PastLens-style baseline is competitive and cheaper. No released AO checkpoint
  matching our site/model was found today [inspected 2026-09-22, via the paper's
  related-work section and HF search]; this is a training-your-own path too.
- **BABEL codec for GPT-2** ([wpferrell/babel-codec-gpt2](https://github.com/wpferrell/babel-codec-gpt2),
  found via web search 2026-09-22): a third-party certified two-way
  English↔activation dictionary for GPT-2 with a different correctness claim —
  interesting as a contrast object, wrong model family and scale for us
  [inspected 2026-09-22, repo landing page only].
- **Reconstructed-difference steering**: the paper's recipe (edit the AV
  explanation, take `AR(edit) − AR(orig)`, steer at the NLA's layer and token
  with norm-scaled direction; main text reports ~50% success and sometimes
  incoherent completions) is a **paper-described method, not a released code
  artifact** — neither repo ships steering code (repo trees inspected
  2026-09-22). But it needs only artifacts we already pin: the target and the
  released AR. The same AR-as-direction-encoder move is what our pilot's P2/P3
  conditions already exercise, so the marginal build is the edit→difference
  protocol, not new models.

## 6. Per-option viability table

"Runner+lock compatibility" refers to `research/open_weight_lingua`: lock format
`configs/model-lock.json` (per-repo `repo_id`/revision/`download_bytes` +
per-file sha256/bytes — format itself is model-agnostic) and the runner, which
**hard-asserts `model_type == "qwen2"`** in `src/open_weight_lingua/target.py:136`,
`nla_adapter.py:59,206,297`, and preflight pins `(3584, 28, "qwen2")`
(`preflight.py:18-19,187-188`) [inspected 2026-09-22]. Costs marked
[inspected] are quoted from the cited sources; everything else is [inferred].

| Option | What exists today | What we would need to build/fetch | Rough cost | Runner+lock compatibility | Sources |
|---|---|---|---|---|---|
| **Same pipeline, other released layer/model** | 3 other pairs, one layer each: Gemma-3-12B L32 (23.6+16.9 GB, gemma license, gated), Gemma-3-27B L41 (108+37.6 GB), Llama-3.3-70B L53 (141+94.8 GB, llama3.3 license). **No other Qwen2.5-7B layer exists.** Same schema_version-2 sidecars (Llama adds additive fields). [inspected 2026-09-22] | Gemma-3-12B: fetch ~40 GB + gated base (~24 GB), accept Gemma terms + HF_TOKEN; port runner off the qwen2 asserts (Gemma3 `text_config` nesting, √d embed_scale, tokenizer.model); new lock entries (format fits as-is). 27B and 70B pairs **do not fit 128 GB unified memory** (70B AV alone 141 GB bf16) [inferred from inspected sizes] | Fetch: hours of bandwidth; port: days of engineering + re-validation of all gates; zero training cost | Lock: compatible as new entries. Runner: incompatible today (qwen2-only asserts; Gemma multimodal wrapper); needs a real port and re-run of calibration+pilot | [HF API](https://huggingface.co/api/models?author=kitft), [collection](https://huggingface.co/collections/kitft/nla-models), [nla-inference README](https://github.com/kitft/nla-inference), local files cited above |
| **Train new pair, this layer (Qwen2.5-7B L20)** | Full Apache-2.0 training repo; pinned Miles fork; documented Qwen7B recipe (the most-profiled model); datagen configs for exactly this model/layer [inspected 2026-09-22] | Anthropic API budget for ~500k summary calls; GPU time: SFT documented on 2×H100-80GB, RL on 2×8×H100; on our single Spark: memory surgery (8-bit optimizer/checkpointing), aarch64 flash-attn build or slow SDPA path, RL re-architecture [inferred] | Authors' reference: RL 1.5 days–1 week on 2×8×H100 (~$1500–$5000) + SFT data at ~250k API calls per dataset [inspected]. On one Spark: **not realistic as documented**; weeks of rework, unverified aarch64 stack [inferred] | New checkpoints would ship our known schema; but a self-trained pair invalidates nothing in the lock format — and gains us a re-roll of the same dice at the same site the pilot already measured | [training repo](https://github.com/kitft/natural_language_autoencoders), [TRAINING_NOTES.md](https://github.com/kitft/natural_language_autoencoders/blob/main/configs/TRAINING_NOTES.md), [paper](https://transformer-circuits.pub/2026/nla/index.html) |
| **Train new pair, other layer/model** | Same training repo supports arbitrary `layer_index` in datagen configs (per-model YAMLs exist for 12B/27B/70B); paper documents layer sensitivity as a real axis [inspected 2026-09-22] | Everything in the row above, plus: new-layer datagen (stage-0 extraction is inference-scale — realistic on the Spark [inferred]), α (injection_scale) sweep per the appendix heuristic (75th-percentile activation norm), and a target-side port if we also change model family | Same order as above plus per-model tuning the authors skipped; the paper's own evidence says layer choice matters, so this is the scientifically interesting but most expensive option [inferred] | Same as previous row; a new layer on Qwen2.5-7B at least keeps the qwen2 runner path | [paper appendix](https://transformer-circuits.pub/2026/nla/index.html), [datagen configs](https://github.com/kitft/natural_language_autoencoders/tree/main/configs/datagen) |
| **Steering-only follow-up (reconstructed-difference)** | Recipe published in paper (Planning in Poetry; ~50% success, messy completions); **no released code artifact**; required artifacts (target + released AR) already pinned and cached here [inspected 2026-09-22] | Edit→`AR(edit)−AR(orig)`→norm-scaled patch protocol on top of existing AR machinery; frozen edit rule and coverage accounting already exist from Milestone 2 | GPU-cheap: reuses 41.4 GB verified cache; engineering-days, no training, no API budget [inferred] | Fully compatible: same repos, same lock, runner already does AR reconstruction + site patching (P2/P3 conditions) | [paper](https://transformer-circuits.pub/2026/nla/index.html), [nla-inference](https://github.com/kitft/nla-inference), local `milestone_two.md` |
| **Stop here** | Pilot decision STOP with all frozen criteria recorded; negative result documented per bounded stopping rule [inspected: local pilot decision record] | Nothing | Nothing | n/a | local `protocols/pilot_decision.md`, `reports/milestone_two.md` |

## Source list (all fetched 2026-09-22 unless noted)

- https://huggingface.co/api/models?author=kitft — model enumeration (8 models, 0 datasets)
- https://huggingface.co/api/models/{repo}?blobs=true — per-repo file inventories and sizes
- https://huggingface.co/kitft/<repo>/raw/main/nla_meta.yaml — all 8 sidecars
- https://huggingface.co/kitft/<repo>/raw/main/README.md — model cards (fve_nrm, licenses, data attribution)
- https://huggingface.co/api/collections/kitft/nla-models — collection membership
- https://transformer-circuits.pub/2026/nla/index.html — paper (models, layers, data/compute scale, steering recipe, layer sensitivity)
- https://github.com/kitft/natural_language_autoencoders (+ /raw/ README.md, configs/TRAINING_NOTES.md, configs/rl.sh, configs/datagen/qwen7b_fineweb_1M.yaml, docs/setup.md, docs/inference.md, pyproject.toml, nla/miles_patches/UPSTREAM_PIN; API: repo meta, commit list, issues)
- https://github.com/kitft/nla-inference (+ /raw/ README.md; API: single-commit history)
- https://api.github.com/repos/sgl-project/sglang/pulls/{20205,20206,20207,20376,14110} — SGLang input_embeds PR statuses
- https://pypi.org/pypi/sgl-kernel/json — aarch64 wheel availability (0.3.21)
- https://pypi.org/pypi/flash-attn/json — source-only distribution (2.8.3.post1)
- https://github.com/slavachalnev/cco_examples — CCO released artifacts (examples only)
- https://github.com/wpferrell/babel-codec-gpt2 — adjacent third-party artifact (GPT-2)
- https://www.neuronpedia.org/nla — hosted sampler (HTTP 200)
- Local: `tests/fixtures/upstream/av-nla_meta.yaml`, `configs/model-lock.json`,
  `src/open_weight_lingua/{target,nla_adapter,preflight}.py`,
  `reports/{milestone_two,source_compatibility}.md`, `THIRD_PARTY.md`
