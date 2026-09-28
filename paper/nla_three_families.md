<!--
Number-trace sources. tests/test_paper_numbers.py checks that every
nontrivial number in this file appears verbatim in at least one of these
committed files. Add a source here before citing a new number.

sources:
  docs/phase_two_results.md
  docs/phase_two_technical_report.md
  docs/claims.md
  research/open_weight_lingua/README.md
  research/open_weight_lingua/protocols/phase_two_brief.md
  research/open_weight_lingua/protocols/pilot_decision.md
  research/open_weight_lingua/protocols/gemma_answer_convention.md
  research/open_weight_lingua/reports/gemma3_pilot.md
  research/open_weight_lingua/reports/gemma3_27b_pilot.md
  research/open_weight_lingua/reports/post_hoc_answer_lens.md
  research/open_weight_lingua/reports/steering_assay.md
  research/open_weight_lingua/reports/nla_ecosystem_notes.md
-->

# Do natural-language descriptions of activations preserve behavior?

**A frozen-protocol evaluation of released NLA pairs on three open-weight models**

seanm27lol · [universa-recurrent](https://github.com/seanm27lol/universa-recurrent)

*Draft, not peer reviewed. Every number below comes from a committed report and
is checked against it by `tests/test_paper_numbers.py`.*

## Abstract

Natural language autoencoders (NLAs) pair an *activation verbalizer* (AV), which
describes one internal activation of a language model in English, with an
*activation reconstructor* (AR), which turns that English back into a vector
(Fraser-Taliente, Kantamneni, Ong et al., 2026). If the English carries what the
model is computing, swapping a vector for its reconstruction should leave the
model's behavior intact, and editing the text should change behavior in a
targeted way. We tested both properties on the three released NLA pairs that fit
our hardware, for Qwen2.5-7B-Instruct, Gemma-3-12B-it and Gemma-3-27B-it, using a
small arithmetic task with paired counterfactuals. Every threshold, control and
stopping rule was frozen before outcomes were read. On the only family where the
task was usable under the frozen metric (Qwen2.5-7B), the description route
failed the preservation criterion decisively: the one-sided 95% upper bound on
accuracy loss was 41.4 percentage points against a 5-point limit. A generic PCA
reconstruction held to the same byte budget stayed close to the unmodified model.
The descriptions were clearly on-task (they beat a shuffled-description control
on every family), but the frozen edit rule found no editable current-value
statement in any group on any family (0/128 each), so the edit hypothesis is
untested. Both Gemma families initially failed the usability gate because of an
instrument artifact (a trailing newline); a post-hoc re-read with that newline
stripped shows all three families exceeding the preservation limit. A separate
steering assay and a serving-backend equivalence gate were also negative. We
report these as bounded negative results for one task, one site per model and
the released checkpoints, not as a verdict on NLAs in general.

## 1. Introduction

Give Qwen2.5-7B-Instruct this prompt:

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The answer is 5. Before the model answers, capture one internal vector `h`. Ask
the released AV to describe it in English, ask the paired AR to rebuild a
direction from that English, and put the rebuilt vector back in place of `h`.
Does the model still answer 5? And if the description says something like
"x is currently 5", does editing that text to "6" make the model answer 6?

These two questions, *preservation* and *editability*, are what a behavioral user
of an explanation method needs. A description can read well and reconstruct the
vector with high cosine similarity while still dropping the part of the
computation that matters for the next answer. The NLA paper itself reports
confabulations and layer sensitivity, which makes behavior the right place to
test (Fraser-Taliente et al., 2026).

**Contributions.**

1. **A frozen behavioral assay** for description round trips: a paired
   counterfactual arithmetic task, an adapter-correctness control, a
   wrong-description control, a byte-budgeted numerical baseline, a random
   direction and a raw donor, with decision gates and stopping rules fixed
   before any outcome was read (§3).
2. **Results on three model families** (§4). Where the task was usable, the
   description route failed preservation decisively while a generic PCA baseline
   did not. The descriptions carried task-relevant information on every family.
   The edit interface never appeared.
3. **Two follow-up measurements**, both negative: steering by description
   differences, and equivalence of a faster serving backend (§4.5–4.6).
4. **An instrument-failure case study** (§4.3): the Gemma usability failures
   were a trailing-newline artifact in the answer metric. It was caught post hoc
   and handled by freezing a metric amendment for a confirmation run, not by
   relabeling closed runs.

## 2. Background

**Natural language autoencoders.** An NLA's AV and AR are first fine-tuned on
English summaries of activations, then optimized with reinforcement learning so
that the AR can reconstruct `h` from the AV's text (reported as fraction of
variance explained). The authors released
training code and four open-model pairs, each trained at a single layer
(Fraser-Taliente et al., 2026; `kitft/natural_language_autoencoders`). We used
the three that fit a single NVIDIA DGX Spark (GB10, unified memory); the
Llama-3.3-70B pair was excluded because its AV alone (141.12 GB) exceeds the
machine's physical memory.

| Target model | Released NLA pair | Capture site (block / depth) |
|---|---|---|
| Qwen2.5-7B-Instruct | `kitft/nla-qwen2.5-7b-L20-av` / `-ar` | 20 / 28 |
| Gemma-3-12B-it | `kitft/nla-gemma3-12b-L32-av` / `-ar` | 32 / 48 |
| Gemma-3-27B-it | `kitft/nla-gemma3-27b-L41-av` / `-ar` | 41 / 62 |

**Activation patching.** Replacing one activation and measuring the downstream
change is standard practice; its results depend on the metric and the corruption
used, and a patching effect is not by itself a complete causal account (Zhang and
Nanda, 2023). We therefore report several behavioral channels (exact-answer
accuracy, agreement with the unmodified generation, next-token KL, and
correct-answer log probability) alongside explicit controls.

## 3. Method

### 3.1 Task

Short straight-line programs over two variables use initialization,
assignment/copy and addition or subtraction by constants, with 3–6 statements and
reference values in 0–19, evaluated by a small explicit interpreter. Each *group*
pairs a source program with a counterfactual that changes the first assignment
(for example `x = 3` becomes `x = 4`, moving the answer for `x` from 5 to 6 while
the answer for `y` stays 8). Each group contributes four prompt variants
(source/counterfactual × query x/y). Groups, not variants, are the unit of
statistical resampling, because variants within a group are not independent.

### 3.2 Intervention

At the capture site, `h` is the residual-stream output of the stated block at the
last non-padding token, including the assistant-generation prefix. The AV
produces a description `c`; the AR maps `c` to a direction `r`; the installed
replacement is

```text
h' = n · r / ||r||₂,   where n = ||h||₂
```

The norm `n` is a retained four-byte side channel, **not** something recovered
from text; a diagnostic that replaces it with the calibration median norm is
reported alongside. The remaining computation then runs with the original prompt
and all other token vectors untouched.

### 3.3 Conditions

| ID | Intervention | Purpose |
|---|---|---|
| P0 | Unmodified target | Behavioral reference |
| P1 | Reinsert the original activation | Adapter correctness gate |
| P2 | Own description → AR direction + original norm | Main condition |
| P3 | Another group's description → AR direction + receiver norm | Does the *specific* description matter? |
| P4 | Calibration-fitted PCA reconstruction | Generic numerical baseline under a byte budget |
| P5 | Norm-matched random direction | Coarse sensitivity diagnostic |
| donor | Raw activation from the paired counterfactual | Sensitivity of the intended measurement |

P4 fits PCA on unit directions from a separate calibration split, with no task
labels; its rank is capped so that float16 coefficients use no more bytes than
the description's UTF-8 text. When the fitted rank caps storage below that budget,
the comparison is *no-more-than-budget*, not an optimal compression baseline.

### 3.4 Frozen gates

All thresholds below were fixed before any pilot outcome was read.

| Gate | Frozen rule |
|---|---|
| 1. Task usable | P0 exact-answer accuracy at least 80%, and donor/perturbation controls show the site can affect the measurement |
| 2. Limited behavioral preservation | One-sided 95% upper estimate of P2 accuracy loss versus P0 at most 5 percentage points, **and** P2 beats P3 on correct-answer log probability with a positive lower confidence estimate |
| 3. Edit feasibility | At least 32 of the 128 pilot groups contain one unambiguous statement of a variable's current value that the frozen editor can change; otherwise close with preservation-only results |
| Uncertainty | Whole-group bootstrap, 3,000 resamples, fixed seed |

The frozen answer metric counts a generation correct only if it is exactly a
canonical integer: no stripping, so extra whitespace, commentary and leading zeros
are failures.

### 3.5 Stages and stopping rule

Each family ran an 8-group engineering smoke, a 256-group calibration (which fits
P4 and freezes the median norm), and one 128-group pilot. A locked 512-group
validation opens only if the pilot meets the gates and its projected cost fits an
eight-hour budget (28,800 s). One pilot per family; no threshold re-tuning; a
failed pilot closes the family. Engineering gates ran on every run: bitwise
identity checks for P1, kernel-shape pinning so BF16 kernel reselection cannot
masquerade as an effect, a frozen suffix-drift bound, and an independent auditor
that recomputes every decision and statistic from saved results.

## 4. Results

All pilots completed 128/128 groups with auditor PASS. The decision on every
family was **STOP**, and no validation stage was opened.

### 4.1 Outcomes against the frozen gates

| | Qwen2.5-7B, block 20/28 | Gemma-3-12B, block 32/48 | Gemma-3-27B, block 41/62 |
|---|---:|---:|---:|
| P0 accuracy (frozen metric) | 0.811 | 0.520 | 0.254 |
| Task usable (P0 ≥ 0.80) | yes | no | no |
| P2 accuracy-loss upper (pp; limit 5) | 41.4 | 3.91 | 2.34 |
| P2 − P3 log-prob lower (must be > 0) | +3.80 | +2.61 | +1.84 |
| P2 agreement with P0 | 0.615 | 0.854 | 0.816 |
| Edit-eligible groups (need 32) | 0/128 | 0/128 | 0/128 |
| AV round-trip cosine (median) | 0.841 | 0.990 | 0.982 |
| Projected validation cost (s; budget 28,800) | 49,253 | 75,348 | 181,295 |
| Decision | STOP | STOP | STOP |

Read naively, the Gemma rows look like the language route *works* on Gemma
(preservation met) and fails on Qwen. Section 4.3 shows why that reading is wrong.

### 4.2 Per-condition behavior

Exact-answer accuracy / agreement with P0 / mean next-token KL, over 512 variants
per family, under the frozen metric:

| Condition | Qwen2.5-7B | Gemma-3-12B | Gemma-3-27B |
|---|---|---|---|
| P0 unmodified | 0.8105 / 1.000 / 0.0 | 0.5195 / 1.000 / 0.0 | 0.2539 / 1.000 / 0.0 |
| P1 original reinserted | 0.8105 / 1.000 / 0.0 | 0.5195 / 1.000 / 0.0 | 0.2539 / 1.000 / 0.0 |
| P2 own description | 0.5645 / 0.615 / 2.71 | 0.5527 / 0.854 / 0.309 | 0.2266 / 0.816 / 0.216 |
| P3 shuffled description | 0.2676 / 0.299 / 7.38 | 0.3223 / 0.361 / 3.574 | 0.2227 / 0.574 / 2.809 |
| P4 PCA baseline | 0.8027 / 0.967 / 0.0157 | 0.5273 / 0.957 / 0.0018 | 0.2617 / 0.939 / 0.0042 |
| P5 random direction | 0.0059 / 0.008 / 11.75 | 0.0000 / 0.000 / 24.672 | 0.0039 / 0.004 / 13.408 |
| donor | 0.7852 / 0.924 / 0.173 | 0.5312 / 0.924 / 0.085 | 0.2656 / 0.924 / 0.059 |
| calibration-median norm | 0.5605 / 0.611 / 2.72 | 0.5664 / 0.859 / 0.309 | 0.2305 / 0.805 / 0.213 |

Three patterns hold on every family:

- **The site matters.** A norm-matched random direction (P5) destroys the
  behavior, and the adapter reinserts the original activation exactly (P1 = P0).
- **The descriptions carry task-relevant information.** P2 beats the
  shuffled-description control P3 on agreement, KL and correct-answer log
  probability everywhere.
- **Generic reconstruction does better under the same budget.** P4 stays close to
  P0 (agreement 0.939–0.967, KL at most 0.0157), while P2 does not.

On Qwen, P2's residual accuracy does not come from the retained norm: replacing
it with the calibration median norm gives 0.5605 against 0.5645.

### 4.3 An instrument failure, and what the corrected re-read shows

Gemma-3-12B's P0 accuracy of 0.5195 was suspicious for a task this simple. A
failure taxonomy of the saved generations found that the mirror-sourced target
answers with digits, then a newline, then `<end_of_turn>`. The frozen metric
rejected 228 of 512 variants for trailing whitespace, and 221 of those had the
correct value once the newline was removed. The Qwen pilot has zero
whitespace-affected rows.

The closed pilots were not relabeled: thresholds and instruments are frozen per
phase, so the STOP decisions stand. Instead, a metric amendment (`rstrip`: strip
trailing ASCII whitespace once, then apply the same exact-integer rule) was frozen
*before* a planned Gemma-12B confirmation pilot. Separately, a post-hoc
descriptive re-read of the saved generations under that lens gives:

| Family | P0 accuracy (lens) | P2 accuracy (lens) | P2 loss point / one-sided upper (pp) | Limit |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 0.8105 | 0.5645 | 34.38 / 41.41 | 5 |
| Gemma-3-12B | 0.9512 | 0.9336 | 5.47 / 10.16 | 5 |
| Gemma-3-27B | 0.9648 | 0.9531 | 2.34 / 6.25 | 5 |

![Upper bound on P2 accuracy loss per family, frozen metric versus post-hoc lens, against the 5-point limit](figures/fig1_preservation_bounds.png)

**Figure 1.** One-sided 95% upper bound on P2 accuracy loss versus P0. Under the
frozen metric (decisions of record), the Gemma bounds sit under the limit only
because the metric rejected most correct answers. Under the post-hoc lens, both
Gemma tasks become usable (P0 0.951 and 0.965) and all three families exceed the
5-point limit. The lens values are descriptive re-reads of saved generations,
not new frozen outcomes.

So the apparent Gemma preservation passes were floor effects of the format
rejections. With the instrument corrected, the three families agree: the
description route loses more than the frozen tolerance. The size of the loss
differs a lot, from 41.41 points on Qwen to 6.25 on Gemma-3-27B, but three
families cannot separate model size from site depth, family or the NLA pair's
training, and the 27B AV was served with a BF16 cast that could not be audited
locally.

**Status of the confirmation run.** The amendment names a fresh Gemma-12B
confirmation pilot under the `rstrip` convention, whose numbers would supersede
the lens preview. Its results are not recorded in this repository at the time of
writing.

### 4.4 The edit interface never appeared

On every family, 0 of 128 groups contained an unambiguous statement of the
affected variable's current value. On Qwen, 127 of 128 receiver parses for `x`
and all 128 for `y` were *absent*; the single eligible parse stated 10 against a
reference of 6. Zero coverage means the descriptions at these sites rarely state
explicit current values in a form the frozen rule accepts. It does not show that
the descriptions lack editable content of other kinds. The edit hypothesis is
**untested**, not refuted.

### 4.5 Steering by description differences

A separate, single frozen run on Qwen reused the pilot split to test the NLA
paper's reconstructed-difference steering recipe,
`h' = h + α·||h||·Δ/||Δ||` with `Δ = AR(d_steered) − AR(d_orig)`, at the same
block-20 site, for α in {−1, 0.5, 1, 2}. In arm 1 the two descriptions are the
pilot's own saved AV descriptions of the counterfactual (B) and source (A)
activations; in arm 2 they are fixed oracle templates (two styles) asserting the
B and A values, passed through the live AR. Frozen success criteria at α = 1 were a
flip to the counterfactual answer in at least 0.30 of groups, the unaffected
answer intact in at least 0.90, and each arm's matched control (a same-value
AV difference for arm 1, a wrong-variable oracle difference for arm 2) moving the
target answer in less than 0.10.

| Arm (α = 1) | Flip to target | Unaffected answer intact | Control moved target | Success |
|---|---:|---:|---:|:---:|
| AV difference | 0.0781 | 0.5703 | 0.4453 | no |
| Oracle, terse template | 0.0313 | 0.4219 | 0.4844 | no |
| Oracle, structured template | 0.0234 | 0.3594 | 0.5625 | no |

Intended deltas flipped the target in 2–8% of groups while disturbing the
unaffected answer in 43–64%, and matched controls moved the target as much as the
intended deltas did. Higher α bought disruption, not targeting. This covers one
checkpoint, one task family and one site. The NLA authors' steering demonstration
used a different, stronger model, site and task; this result does not refute it.

### 4.6 A faster backend is a different instrument

An opt-in vLLM path for the AV and AR stages was held to a measured equivalence
gate against the default eager Transformers path, replaying a pinned smoke
bundle. It failed: 0/32 AV greedy continuations were token-identical (median
first divergence at token 10.5, at a sampled near-tie with a top-2 logit margin
of 0.125), and AR direction cosine ranged from a minimum of 0.8097 to a maximum of
0.99979. The backend therefore stays non-default, and its outputs are never mixed
into eager-path evidence. It was also not faster on this workload (559.5 s versus
522.4 s eager), because reproducing the eager recompute-per-pass decode rules out
decode caching.

## 5. Discussion

**What the evidence supports.** At the released sites, on this task, the English
descriptions carry information the model uses (P2 beats P3 everywhere), but not
enough to reproduce the model's behavior within a 5-point tolerance: decisively
so on the usable Qwen task under the frozen metric, and on all three families
under the post-hoc lens. A generic numerical reconstruction, given
no more bytes than the text, stays close to the unmodified model. The descriptions
did not expose current variable values in an editable form, and description
differences did not steer targeted behavior.

**What it does not support.** None of this says language is useless for
interpretability: P4 is a no-more-than-budget baseline for one task, not an
optimal compressor, and the brief explicitly forbids that upgrade. It also does
not rank model sizes, and it does not speak to layers other than the one each pair
was released for. The NLA paper documents layer sensitivity, and no
alternate-layer checkpoints are released, so testing another site means
training a new NLA.

**Why the process matters as much as the numbers.** Three things in this study
could easily have produced a misleading headline. A BF16 kernel reselection could
have looked like an intervention effect; kernel-shape pinning and a bitwise
identity gate ruled that out. A faster backend could have silently changed the
measurements; an equivalence gate caught it. A metric artifact made two families
look like they preserved behavior; freezing thresholds per phase and freezing the
fix as an amendment for a new run, rather than relabeling closed runs, kept the
record honest. The negative results are only credible because each of those
failure modes had a written rule before the outcome was seen.

## 6. Limitations

- One task family (two-variable arithmetic), one capture site per model, one
  capture token, and 128 pilot groups per family. No locked validation ran.
- Released checkpoints only; the Gemma targets are sourced from a public mirror
  whose weight and tokenizer files are content-identical to the gated originals,
  with four small configuration files differing.
- The Gemma-3-27B AV (float32-native) was served with a BF16 cast to fit memory.
  That deviation is declared in the lock and could not be audited locally.
- The lens results in §4.3 are post-hoc and descriptive. The planned confirmation
  run under the frozen amendment is not recorded here.
- Behavioral measurements are engineering instruments, not semantic evidence. No
  claim is made about what the descriptions mean.

## 7. Reproducibility

Everything needed to audit or re-run the study is public in
[`research/open_weight_lingua`](../research/open_weight_lingua/README.md):
hash-pinned model locks, the frozen protocols, the runner, the independent
auditor, per-run manifests and the reports quoted here. Its fixture test suite
runs on CPU in CI, on tiny random models with no downloads. Real runs need one
GPU host; measured pilot wall-clock times were 11,980.8 s (Qwen2.5-7B), 18,472.7 s
(Gemma-3-12B) and 44,681.4 s (Gemma-3-27B).

| Record | Where |
|---|---|
| Governing brief, conditions and gates | [protocols/phase_two_brief.md](../research/open_weight_lingua/protocols/phase_two_brief.md) |
| Qwen pilot decision, gate by gate | [protocols/pilot_decision.md](../research/open_weight_lingua/protocols/pilot_decision.md) |
| Gemma-3-12B and 27B pilots | [reports/gemma3_pilot.md](../research/open_weight_lingua/reports/gemma3_pilot.md), [reports/gemma3_27b_pilot.md](../research/open_weight_lingua/reports/gemma3_27b_pilot.md) |
| Answer-metric amendment and lens | [protocols/gemma_answer_convention.md](../research/open_weight_lingua/protocols/gemma_answer_convention.md), [reports/post_hoc_answer_lens.md](../research/open_weight_lingua/reports/post_hoc_answer_lens.md) |
| Steering assay | [reports/steering_assay.md](../research/open_weight_lingua/reports/steering_assay.md) |
| Every claim and its status | [docs/claims.md](../docs/claims.md) |
| Full technical report (Qwen) | [docs/phase_two_technical_report.md](../docs/phase_two_technical_report.md) |

Figure 1 is regenerated by `python paper/make_figures.py`.

## References

- Fraser-Taliente, Kantamneni, Ong et al. (2026). *Natural Language Autoencoders
  Produce Unsupervised Explanations of LLM Activations.*
  <https://transformer-circuits.pub/2026/nla/>. Training code:
  <https://github.com/kitft/natural_language_autoencoders>; inference recipe:
  <https://github.com/kitft/nla-inference>.
- Zhang and Nanda (2023). *Towards Best Practices of Activation Patching in
  Language Models: Metrics and Methods.* <https://arxiv.org/abs/2309.16042>.
- Qwen team. *Qwen2.5-7B-Instruct* model card.
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>.
- Google. *Gemma 3* model cards, <https://huggingface.co/google/gemma-3-12b-it>
  and <https://huggingface.co/google/gemma-3-27b-it>. Gemma Terms of Use apply.
