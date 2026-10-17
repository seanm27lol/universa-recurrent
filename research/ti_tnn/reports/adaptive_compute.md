# TI-TNN adaptive compute: not close in quality at any large saving — 2026-10-17

**Protocol:** [`protocols/adaptive_compute.md`](../protocols/adaptive_compute.md), frozen at `64ecc86`
before any run. Evaluation fixes from two outside reviews (`43239ae`, `d9a8c1f`) were made before any
part B result or latency existed; the protocol's execution record lists them.

## The question

Can TI-TNN be close in quality to a standard model while using drastically less compute?

TI-TNN adaptive gives each token, at each layer, three options: skip the block, run it with small
typed experts, or run it with full ones. A cap holds mean compute to 50% or 30% of a full block. It is
compared on a quality-against-compute frontier with dense models of several sizes, top-1 and top-2
mixtures of experts, and plain layer skipping (`mod`, Mixture-of-Depths-style).

The pass bar was fixed in advance:
- quality within 2% of the dense model, at no more than half the compute of the cheapest alternative
  that is as good;
- and better quality than every alternative that uses no more compute.

## Answer

**The efficiency claim is not validated.** By the frozen rule it needs predictions 5 and 6 to hold;
both failed.

- **Not close in quality.**
  - At 30% compute, TI-TNN adaptive reaches perplexity 49.1, 23% worse than dense (39.85).
  - At 50%, it reaches 45.3, 14% worse.
  - The bar was 2% (40.65).
- **A smaller dense model does better at 50%.** The 4-layer dense model reaches 43.7 with slightly
  less compute (13.6M against 14.1M decoder multiply-accumulates per token). Halving the depth beat
  learning to skip.
- **At 30%, TI-TNN beats everything that is as cheap:** plain skipping (52.9) and the 256-wide dense
  model (55.3).
- **Against plain skipping at the same cap,** TI-TNN is better at both caps (49.1 against 52.9; 45.3
  against 46.7). But it has four times the parameters (202M against 51M), and its experts start from a
  different initialization (see the execution record). Neither is controlled here.
- **Real speed is untestable under the rule fixed in advance.** Cached decoding of the 30%-cap model
  did not match the full pass closely enough: logits differed by up to 7.5% of their scale where
  routing agreed, against a 5% tolerance. Read only as raw numbers, it decoded at 0.87 times the dense
  model's time per token (1.92 against 2.21 ms). That is far from the 0.6 the prediction asked for.
  The 4-layer dense model was faster still (1.59 ms).
- **Part A (the lookup task built to favor adaptive compute) is a negative of a different kind.** The
  adaptive models mostly never learned the task, and neither did the full-compute top-1 mixture of
  experts. The 8-layer dense model learned it perfectly.

## Part B: WikiText-103 (3 seeds; validation perplexity, lower is better)

| Variant | Perplexity | Decoder MACs per token | Total MACs per token (with output layer) | Realized block compute | Parameters |
|---|---:|---:|---:|---:|---:|
| `dense_d256_l8` | 55.27 ± 1.26 | 7.34M | 20.21M | 1.00 | 19M |
| `mod_c30` | 52.95 ± 0.49 | 7.86M | 33.59M | 0.297 | 51M |
| `ti_tnn_adaptive_c30` | 49.09 ± 0.37 | 8.33M | 34.06M | 0.300 | 202M |
| `mod_c50` | 46.72 ± 0.68 | 13.35M | 39.08M | 0.500 | 51M |
| `dense_d512_l4` | 43.69 ± 0.73 | 13.63M | 39.37M | 1.00 | 39M |
| `ti_tnn_adaptive_c50` | 45.26 ± 2.44 | 14.07M | 39.80M | 0.500 | 202M |
| `dense_d384_l8` | 44.56 ± 0.02 | 15.73M | 35.03M | 1.00 | 34M |
| `dense` (reference) | 39.85 ± 0.10 | 27.27M | 53.00M | 1.00 | 51M |
| `moe_top1` | 39.57 ± 0.46 | 27.30M | 53.03M | 1.00 | 169M |
| `moe_top2` | 37.92 ± 0.75 | 44.08M | 69.81M | — | 169M |

- **Per-seed perplexities.**

  | Variant | Seed 0 | Seed 1 | Seed 2 |
  |---|---:|---:|---:|
  | `ti_tnn_adaptive_c30` | 49.21 | 49.39 | 48.67 |
  | `ti_tnn_adaptive_c50` | 44.35 | 48.03 | 43.41 |
  | `mod_c30` | 53.50 | 52.77 | 52.57 |
  | `mod_c50` | 47.46 | 46.11 | 46.59 |

- **Test perplexities** follow the same order.
- **Reused runs.** `dense`, `moe_top1` and `moe_top2` are reused from the
  [real benchmark](real_benchmarks.md); their code path is bitwise unchanged.

**How the adaptive models spend compute.**
- Both kinds run the first layer for almost every token (99–100% on every seed).
- Which later layers they skip varies by seed. The third layer, for example, runs for 2% to 97% of
  tokens depending on the seed.
- At the 50% cap, TI-TNN sends 45–58% of block evaluations to the small experts.

**Compute.** "Half the compute" here means half the decoder. The output layer (25.7M multiply-accumulates
per token at width 512) is the same for every 512-wide model. So the 30%-cap models still use 64% of
the dense model's total arithmetic.

## Latency (batch-1 decoding, median ms per token; raw, not a verdict)

| Model | ms per token | Execution check |
|---|---:|---|
| `dense` (random initialization) | 2.21 | valid |
| `dense_d512_l4` | 1.59 | valid |
| `dense_d256_l8` | 1.41 | valid |
| `mod_c30` (trained) | 1.57 | logits up to 14.7% of scale apart |
| `ti_tnn_adaptive_c30` (trained) | 1.92 | logits up to 7.5% of scale apart; top-1 agreement 0.980 on one window |
| `ti_tnn_adaptive_c50` (trained) | 2.39 | logits up to 9.9% of scale apart |
| `moe_top1` (random initialization) | 2.42 | logits up to 110% of scale apart |

- **Routing decisions agreed** on at least 99.4% of (token, layer) pairs for every adaptive model. So
  the logit differences come from elsewhere.
- **Likely sources** are the expert choice inside a branch (not covered by the agreement check) and
  bf16 arithmetic, which differs between the masked full pass and the cached decoder.
- **The random-initialization mixtures** have near-uniform expert routing, so tiny numeric differences
  flip their expert choices. Their timings are not clean records of the same execution.
- **These numbers measure something narrow.** Eager PyTorch on one GB10 at batch 1, dominated by
  kernel launches and the synchronizations that data-dependent routing needs. They show that skipped
  layers save some real time, and that expert dispatch costs some back.

## Part A: variable-hop lookups (3 seeds; hard-query accuracy, chance about 0.14)

| Variant | Accuracy | 1 hop | 2 hops | 3 hops | Decoder MACs per token |
|---|---:|---:|---:|---:|---:|
| `dense` (8 layers) | **1.000** | 1.00 | 1.00 | 1.00 | 1.70M |
| `dense_d128_l6` | 0.648 ± 0.234 | 1.00 | 0.60 | 0.34 | 1.28M |
| `dense_d128_l4` | 0.848 ± 0.228 | 1.00 | 0.87 | 0.68 | 0.85M |
| `dense_d128_l3` | 0.684 ± 0.230 | 1.00 | 0.69 | 0.36 | 0.64M |
| `dense_d128_l2` | 0.455 ± 0.021 | 1.00 | 0.19 | 0.17 | 0.43M |
| `moe_top1` | 0.143 ± 0.003 | 0.14 | 0.14 | 0.14 | 1.71M |
| `mod_c50` / `c30` / `c15` | 0.212 / 0.202 / 0.142 | 0.41 / 0.32 / 0.14 | about 0.11–0.14 | about 0.11–0.14 | 0.75 / 0.47 / 0.24M |
| `ti_tnn_adaptive_c50` / `c30` / `c15` | 0.238 / 0.141 / 0.143 | 0.42 / 0.14 / 0.15 | about 0.14 | about 0.14 | 0.75 / 0.46 / 0.25M |

- **Easy tokens** (the count) were learned perfectly by every variant.
- **The validity condition holds:** the reference scored 1.000.
- **The hop queries were not learned.** Apart from one seed each of `mod_c50`, `mod_c30` and
  `ti_tnn_adaptive_c50`, which learned 1-hop queries only, every adaptive model stayed at chance on
  them.
- **The mixture of experts failed too.** `moe_top1` stayed at chance on every hop count and seed,
  although it runs at full compute with the dense model's attention. So the failure is not only about
  saving compute.
- **Compute did not follow difficulty.** At the 50% cap, TI-TNN gave the easy tokens more compute
  (0.50 of a full block) than the hard queries (0.18–0.32).
- **Learnability is fragile even for dense models.** At 3–6 layers, results vary strongly by seed;
  the 6-layer model scored below the 4-layer one.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| A-validity | 8-layer dense ≥ 0.90 on part A | **Holds** (1.000); part A is readable |
| 1 | Part A: some candidate within 0.02 of dense at ≤ half the cheapest-as-good compute | **Failed** (no candidate above 0.24) |
| 2 | Part A: every candidate beats every baseline using no more compute | **Failed** |
| 3 | Part A: TI-TNN beats `mod` at each cap | **Failed** (c30 and c15 at chance alongside `mod`) |
| 4 | Part A: compute follows difficulty | **Failed** (easy tokens got more compute at c50; flat across hop counts at c30 and c15) |
| 5 | Part B: some candidate within 2% of dense at ≤ half the cheapest-as-good compute | **Failed** (best 45.26 against a 40.65 bar) |
| 6 | Part B: every candidate beats every baseline using no more compute | **Failed** for c50 (the 4-layer dense model beats it); held for c30 |
| 7 | Part B: TI-TNN beats `mod` at each cap | **Held** (49.09 < 52.95; 45.26 < 46.72), with the parameter and initialization confounds above |
| 8 | Part B: `ti_tnn_adaptive_c30` decodes in ≤ 0.6× dense's time | **Untestable**: execution checks failed (raw ratio 0.87) |

**Decision rule:** validated only if 5 and 6 hold. **Not validated.**

## What this shows, and what it does not

- **Shows.** At this scale and with this training recipe, per-token adaptive compute buys less quality
  per unit of compute than simply using a smaller dense model. TI-TNN's typed experts help compared
  with plain skipping, but not enough to reach the dense frontier.
- **Does not show that adaptive compute cannot work.** Part A suggests the recipe may be a cause:
  - the compute cap applies from the first step, before the model has learned anything;
  - the expert initialization differs from the dense model's;
  - even a full-compute mixture of experts failed to learn the lookup task.

  This report does not test those causes.
- **Scale.** A 51M-parameter dense reference trained on 100M tokens. Results for large models do not
  follow either way.

## Execution

- **Runs.** All 57 completed with no failures (36 part A, 21 new part B), from 2026-10-16 20:49 to
  2026-10-17 16:05 UTC on the GB10, each behind the idle-GPU gate.
- **Co-tenancy.** A GPU co-tenancy log ran from the start. My own test runs opened a CUDA context
  between queued runs three times, which made the gate wait. No 30-second sample shows another
  process during a training run.
- **Results.** `runs/titnn_efficiency/frontier.json` sha256 `314b63ca…`, which includes the run manifest
  with every result file's hash. `latency.json` sha256 `5e54a859…`. Both were built by
  `bench/frontier.py` and `bench/latency.py` at `d9a8c1f`.

Code: [`bench/`](../bench/): `ti_tnn_models.py`, `lm.py`, `hops.py`, `decode.py`, `latency.py`,
`frontier.py`, `run_efficiency.sh`.
