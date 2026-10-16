# TI-TNN on real benchmarks: WikiText-103 and SCAN (frozen protocol)

**Status: FROZEN 2026-10-15 (UTC), before any full run.** Branch
`ti-tnn-real-benchmarks`.
- A 150-step throughput pilot per variant ran beforehand, to size the
  budget. Its losses and perplexities are not results.
- A 300-step SCAN timing check ran for two variants.

Nothing below was tuned after seeing any result.

## The question

How does a TI-TNN feed-forward layer compare with standard architectures on
established benchmarks? The candidate is top-1 experts plus a Noul-gated
sheaf bridge to the runner-up expert. The baselines:
- a dense MLP;
- Switch-style top-1 mixture of experts;
- top-2 mixture of experts.

Real text has no question-type labels, so no variant gets routing labels.
This follows the unsupervised-routing result
([brief](unsupervised_routing.md)).

## Architectures (`research/ti_tnn/bench/ti_tnn_models.py`)

A GPT-style decoder: pre-LayerNorm, learned positions, tied embeddings. Only
the feed-forward block differs, and every expert has the dense block's shape
(D → 4D → D).

| Variant | Feed-forward block | Expert runs per token (inference) | Training compute |
|---|---|---|---|
| `dense` | one MLP | 1 (dense) | 1× |
| `moe_top1` | 8 experts, top-1, scaled by router probability | 1 | 1× |
| `moe_top2` | 8 experts, top-2, renormalized weights | 2 | 2× |
| `ti_tnn_charged` | `moe_top1` + Noul gate; where it fires, runner-up expert + sheaf bridge W_e tanh(F_u o_u − F_v o_v); each firing charged 0.01 in the loss | 1 + firing rate | 2× (runner-up and bridge run for every token so the gate learns) |
| `ti_tnn_free` | the same, with no firing charge | 1 + firing rate | 2× |

Every mixture variant uses the Switch load-balancing loss (coefficient 0.01).

## Benchmark 1: WikiText-103 language modeling (`bench/lm.py`)

- **Data.** WikiText-103 raw (`Salesforce/wikitext`, wikitext-103-raw-v1),
  tokenized with GPT-2 byte-pair encoding: 117,920,140 training, 247,289
  validation and 283,287 test tokens. Hashes are in
  `runs/titnn_data/wikitext103/meta.json`; preparation is in
  `bench/prep_wikitext.py`.
- **Model.** 8 layers, width 512, 8 heads, context 512. The dense model has
  51M parameters; the mixture variants about 168M, with the same per-token
  expert compute as dense at top-1.
- **Training.** 100M tokens (6,103 steps of 32 × 512), AdamW (0.9, 0.95),
  weight decay 0.1, peak learning rate 6e-4, 2% warmup, cosine to 10%, bf16
  autocast, gradient clipping 1.0. Every variant sees the same data order for
  a given seed.
- **Score.** Perplexity = exp(mean cross-entropy) over the whole validation
  and test splits, in non-overlapping 512-token windows. It is comparable
  across these runs, not with published numbers.
- **Seeds.** 0, 1, 2.

## Benchmark 2: SCAN (`bench/scan.py`)

- **Data.** SCAN (Lake and Baroni 2018) splits `simple`, `length`,
  `addprim_jump` and `addprim_turn_left`, from
  github.com/brendenlake/SCAN. The file hashes are recorded in each
  result.
- **Model and training.** The same decoder at 4 layers, width 256, 4 heads,
  context 72 (8 experts, edge width 64). It reads `command <sep> actions <eos>`
  and is trained only on action tokens. 5,000 steps of 128 examples, AdamW
  (0.9, 0.98), weight decay 0.1, learning rate 5e-4, 5% warmup, cosine decay.
- **Score.** Test exact match with greedy decoding: the whole action sequence
  must be correct, and then stop. Accuracy on a 2,000-example training sample
  is reported as a fit check.
- **Seeds.** 0, 1, 2.

## Predictions (recorded before any full run; all one-sided)

**WikiText-103** (validation perplexity, every seed unless stated):
1. `moe_top1` is lower than `dense`: more parameters at equal per-token
   compute, as in the Switch Transformer.
2. `moe_top2` is lower than `moe_top1`.
3. `ti_tnn_charged` fires on under 5% of validation tokens, and its perplexity
   is within 2% of `moe_top1`'s. With the charge, it reduces to top-1, as the
   pilot already suggested.
4. `ti_tnn_free` fires on over 10% of validation tokens. Its mean perplexity
   is lower than `moe_top1`'s but not lower than `moe_top2`'s. **This is the
   open question**, and the prediction is a guess.
5. Training throughput (tokens/s): `dense` > `moe_top1` > `moe_top2` ≥ the
   TI-TNN variants.

**SCAN** (test exact match, mean over seeds):
1. Every variant scores at least 0.95 on `simple`.
2. Every variant scores at most 0.20 on `length`. Learned absolute positions
   never see the longer outputs in training.
3. Every variant scores at most 0.20 on `addprim_jump`. This is a known failure
   of standard sequence models, and nothing in unsupervised routing targets it.
4. No TI-TNN variant beats every baseline on `addprim_jump` by more than 0.10.
   The toy benchmark's composition result relied on isolation enforced by
   construction, which real data does not provide.

All results are reported whatever they show, with per-seed values.

## Limits

- **Small models and short training.** These are comparisons among
  architectures under one fixed budget and one fixed set of hyperparameters,
  shared by every variant and tuned for none. They are not state-of-the-art
  numbers.
- **Three seeds and no significance tests.** Claims use "on every seed" or
  means with spread.
- **Compute.** Counted feed-forward MACs plus measured training throughput.
  The mixture code loops over experts and is not optimized; the same code
  serves every mixture variant.

## Execution

`research/ti_tnn/bench/run_all.sh` runs the queue, each run only on an idle
GPU (gated):
1. WikiText-103 seed 0, all five variants;
2. SCAN seed 0, all splits and variants;
3. seeds 1 and 2 of each.

Results go to the git-ignored `runs/titnn_bench/`; `bench/summarize.py`
builds the tables. About 21 GPU-hours.

**Execution record.** The first launch (2026-10-15 19:51 UTC) used `lm.py`'s
200M-token default instead of the protocol's 100M, because the runner did not
pass `--tokens`. I stopped it after 100 of 12,207 steps of the first dense run.
It produced no result, and its partial output was deleted. The runner and the
default now both use 100M tokens. Nothing else changed.
