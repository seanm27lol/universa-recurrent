# TI-TNN: close in quality at drastically less compute? (frozen brief)

**Status: FROZEN 2026-10-16 (UTC), before any run of the adaptive variants that reports quality.**
Branch `ti-tnn-adaptive-compute`, based on `main` at `fd25199`.

**Pilots before freezing** (none of their quality numbers for `mod` or `ti_tnn_adaptive` was looked at):
- *Timing and compute cap.* 200 steps of part A for `dense`, `moe_top1`, `mod_c30` and
  `ti_tnn_adaptive_c30`, and 300 steps of part B for the three new dense sizes, `mod_c30` and
  `ti_tnn_adaptive_c30`. Only throughput and the compute and price fields of the training log were read.
  The cap holds: realized compute settles at 0.21–0.33 against a cap of 0.30.
- *Part A task design, dense baseline only.* The first task setting, with hops 1–4 equally likely,
  could not be learned. The 8-layer dense model stayed at chance on every hop count: hard-query accuracy
  0.14 after 6,000 or 20,000 steps, or with twice the queries, or with learning rate 3e-3. Hops 1–4 with
  half the queries one-hop were learned on seed 0 (1.000). Seeds 1 and 2 learned hops 1–2 but not 4
  (0.20 and 0.30). Training 2.5 times longer fixed seed 2 (1.000) but made seed 1 worse (0.39, with 2-hop failing too). Capping
  the task at 3 hops, with half the queries one-hop, was learned on all three seeds (1.000 each in 6,000
  steps). That is the frozen setting.
  So those dense-baseline numbers were seen before freezing; no adaptive model was trained on the
  final task before freezing.

## The question

The goal for TI-TNN is efficiency: quality close to a standard model's at drastically less compute. The
earlier TI-TNN layer could not show that. It always runs at least one full expert, the same as a top-1
mixture of experts, and the bridge only adds compute
([real-benchmark report](../reports/real_benchmarks.md)). A mixture of experts already avoids touching
most of its parameters. What it does not do is spend less on easy tokens than on hard ones; that is
what this tests.

So the test is a frontier: quality against compute per token, for TI-TNN with adaptive compute and for
the alternatives at several sizes. The pass bar is fixed here, before any run.

## Architectures (`research/ti_tnn/bench/ti_tnn_models.py`)

The same GPT-style decoder as the real benchmark. Two new block kinds let each token, at each layer,
choose how much of the block to run:

| Variant | Options per token per layer | Cost (fraction of a full block) |
|---|---|---|
| `mod` | skip the block; or run it with the dense MLP | 0; 1 |
| `ti_tnn_adaptive` | skip; or run it with a small typed-expert layer (8 experts, D → D → D, top-1); or with a full one (8 experts, D → 4D → D, top-1) | 0; 0.5; 1 |

- **Skipping.** A skipped token leaves the block unchanged and is not a key or value in its attention,
  as in Mixture-of-Depths (Raposo et al. 2024). `mod` is Mixture-of-Depths-style with *token choice*
  rather than expert choice, so every decision is causal and the model can decode one token at a time.
- **Cost.** A block's cost counts its projections (4D²) and MLP (2mD²): a full block is 12D², and a small
  one 6D².
- **Training.** The choice is a hard argmax forward and straight-through backward. Every option is
  computed for every token in training, so the router sees each option's first-order value and no
  option is absorbing. Training therefore costs more than the dense model: the claim is about inference.
- **The compute cap.** Mean block compute must stay at or below a cap. A Lagrange multiplier enforces it,
  updated by dual ascent with step 0.01. The price of compute rises while the cap is exceeded and falls
  to zero when it is not binding. Caps: 0.5, 0.3 and 0.15 in part A; 0.5 and 0.3 in part B.
- **What TI-TNN keeps and drops.** It keeps typed experts behind a router. The choice of compute
  replaces the Noul gate, and the bridge is dropped because it did not earn its compute.
- **A deviation from the plan.** The plan said the cheap path would be chosen by the Jev router's
  *calibrated confidence*. Calibration needs a label saying when the cheap path was good enough, and
  text has none. The design instead prices compute: the router runs the expensive option when its
  first-order value exceeds the current price. No Brier term is used.

**Baselines.** Dense models at several sizes, `moe_top1`, `moe_top2` and `mod`. The dense models trace
the ordinary quality-against-compute curve; `mod` isolates what TI-TNN's typed and small branches add to
plain skipping.

**Compute** is non-embedding forward multiply-accumulates per token: the decoder blocks, attention
scores included, counted from the decisions each model actually made on the evaluation data. Routers
are counted; layer norms are not. The tied output layer (V × D per token) is the same for every variant
at one width and is reported separately.

## Part A: variable-hop lookups (`bench/hops.py`) — where it should win first

- **Task.** Each sequence states 8 facts "a f(a)" (f is a random 8-cycle on 8 of 32 symbols), then a
  stream of mostly easy tokens: a count, where each target follows from the current token alone. 12 hard
  queries "H_k x" are mixed in, answered at x by f^k(x): k = 1 for half of them, and 2 or 3 (equally
  likely) for the rest.
  T = 128 tokens, 87 of them easy and 12 scored queries.
- **Why it favors adaptive compute.** A k-hop answer needs about k + 1 layers, an easy token about one.
  If TI-TNN cannot win here, it will not win on text. The facts are given in context, not stored in
  weights.
- **Model.** Width 128, 4 heads, 8 experts, context 128. 6,000 steps of 64 fresh sequences,
  AdamW (0.9, 0.95), weight decay 0.1, learning rate 1e-3, 2% warmup, cosine to 10%, bf16 autocast.
- **Variants.** `dense` at 2, 3, 4, 6 and 8 layers, `moe_top1`, and `mod` and `ti_tnn_adaptive` at caps
  0.5, 0.3 and 0.15. All but the shallower dense models have 8 layers.
- **Score.** Hard accuracy = mean exact accuracy over hop counts, on 2,048 fixed evaluation sequences
  (the same for every run). Easy accuracy is reported. Seeds 0, 1, 2.

## Part B: WikiText-103 (`bench/lm.py`) — the real test

- **Same setup as the real benchmark.** Data, tokenizer, 100M-token budget, optimizer, schedule, data
  order and scoring as in [`real_benchmarks.md`](real_benchmarks.md). Validation perplexity is the
  quality measure.
- **Reused runs.** The `dense`, `moe_top1` and `moe_top2` runs (8 layers, width 512) are reused from
  that benchmark. Their code path is unchanged: outputs and losses are bitwise identical
  under the new code (checked before freezing).
- **New runs.** Dense at 4 layers × 512, 8 × 384 and 8 × 256; `mod` and `ti_tnn_adaptive` at caps 0.5
  and 0.3. Seeds 0, 1, 2.
- **Latency** (`bench/latency.py`). Median milliseconds per token, decoding one token at a time (batch 1)
  with key/value caches, so skipped blocks really are skipped. Every variant decodes the same two
  validation windows in three interleaved rounds after all training. Adaptive runs use their trained
  weights; the others are timed at a random initialization, since their latency does not depend on the
  weights. Token-by-token decoding gives the same logits as the full forward pass (tested).

## Predictions (recorded before any run; all one-sided; means over 3 seeds)

"Candidate" means a `ti_tnn_adaptive` variant. "Baseline" means every other variant in the same part.
"The cheapest baseline as good" is the baseline with the least compute among those whose quality is at
least the candidate's. If there is none, it is the best baseline.

**Part A** (quality = hard accuracy; reference = 8-layer dense):
- **A-validity** (a condition for reading part A, not a prediction): the reference's hard accuracy is
  at least 0.90.
1. **Drastic efficiency.** Some candidate is within 0.02 of the reference's hard accuracy, using at most
   half the compute of the cheapest baseline as good.
2. **Equal compute.** Every candidate is more accurate than every baseline that uses no more compute.
3. **Against plain skipping.** At each cap, `ti_tnn_adaptive` is more accurate than `mod`.
4. **Compute follows difficulty.** In every candidate, 3-hop queries get more compute than 1-hop ones,
   and easy tokens get less than queries of any hop count.

**Part B** (quality = validation perplexity; reference = dense, 8 × 512):
5. **Drastic efficiency.** Some candidate's perplexity is at most 1.02 × the reference's, using at most
   half the compute of the cheapest baseline as good.
6. **Equal compute.** Every candidate has lower perplexity than every baseline that uses no more compute.
7. **Against plain skipping.** At each cap, `ti_tnn_adaptive` has lower perplexity than `mod`.
8. **Real speed.** `ti_tnn_adaptive_c30` decodes in at most 0.6 × the reference's median milliseconds
   per token.

**The decision rule.** TI-TNN's efficiency claim is validated only if 5 and 6 hold. Predictions 1–4
holding alone is a necessary condition met, not validation. Prediction 2 or 6 counts as untestable, not
held, for a candidate cheaper than every baseline.

`bench/frontier.py` applies these rules exactly as written. Every variant and seed is reported whatever
happens.

## Limits

- **Scale.** A 51M-parameter dense reference trained on 100M tokens. Efficiency claims about large
  models do not follow from this either way.
- **Parameters.** `ti_tnn_adaptive` has 202M parameters, more than `moe_top1` (169M) or dense (51M).
  A win could come partly from parameters; `mod` (51M) and `moe_top1` bound that.
- **Latency.** Batch-1 decoding in eager PyTorch on one GB10 is dominated by kernel launches. It shows
  whether skipped layers save real time, not what an optimized server would see.
- **Training cost.** The adaptive variants train at more than full compute.
- **Not tested.** Composition (see the [functorial composition report](../reports/functorial_composition.md)).

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_efficiency.sh`, then
`python research/ti_tnn/bench/frontier.py --hops runs/titnn_efficiency/hops --lm runs/titnn_bench/lm
runs/titnn_efficiency/lm --latency runs/titnn_efficiency/latency.json`.
