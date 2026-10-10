# Phase Eight: when does the answer stop needing the program? (attention knockout, exploratory)

**Status: FROZEN 2026-10-06 (UTC), before any Phase Eight model forward.
Branch `phase-eight-knockout`, based on `main` at `5a71cba`. This is an
exploratory study on reused calibration and pilot prompts. `validation_b`
is not touched here. A confirmatory test for it will be frozen separately,
and only if these results name a specific claim.**

## The question

```text
x = 7
y = 16
y = y + 1
y = y - 3
What is x? Reply with only the integer.      → 7
```

Phases Three to Seven measured what probes can read and how answers change
with the prompt. None of them tested where the answer comes from when the
model produces it.
- **Retrieval at answer time.** The positions after the program (the
  question, the chat template, the answer slot) look back at the program
  tokens in late layers.
- **Earlier extraction.** The needed value was moved out of the program
  tokens earlier, and later layers no longer need to see the program.

**Attention knockout** separates these (Geva et al. 2023, *Dissecting Recall
of Factual Associations in Auto-Regressive Language Models*, EMNLP). From
layer k onward, block attention from chosen later positions to the program
tokens. If the answer survives, whatever the answer needs has left the
program tokens before layer k. The smallest k at which blocking stops
hurting marks when.

Unlike the probes, this is an intervention: it shows what the model's
answer *uses*, not just what is decodable.

## Design

- **Models.** Gemma-3-12B-it and Qwen2.5-7B-Instruct, locks hash-verified.
  Both are analyzed separately; neither gates the other.
- **Prompts.** Phase Two's calibration and pilot prompts in the original
  format (question after the program), from their saved token IDs: 1,536
  per family.
- **Program span.** The tokens from the first program character through the
  token that completes the program's last line, located by decoding. All
  3,072 prompts were checked with the real tokenizers before this freeze:
  every span decodes exactly to the program (13–39 tokens), followed by
  exactly 16 tokens of question and chat template.
- **Scoring by teacher forcing.** Append the correct answer's digit tokens
  to the prompt, run one forward pass, and read the logits at the positions
  that predict each digit.
  - **Correct** means every answer digit is the argmax at its position.
  - **Log-prob** is the sum over digits of the log-probability of the
    correct digit.
  - This digit-accuracy is close to, but not the same as, the frozen
    greedy-text metric. The baseline is reported under it.
  - Forwards are right-padded to the pinned 128-token bucket, with no KV
    cache, BF16, eager attention and TF32 off.
- **Knockout.** In every layer from k to the last, the additive attention
  mask gets the dtype's minimum value at (query ∈ Q, key ∈ program span),
  for every head. Two query sets:
  - **A, `answer`:** Q is the positions that predict answer digits (the last
    prompt token and each appended digit except the last). Other positions
    may still read the program and pass information on.
  - **B, `after_program`:** Q is every position after the program: the
    question, the template and the answer positions. From layer k on,
    nothing after the program can read it.
- **Layer grid.** k = floor(j·L/8), j = 0…7.
  - Gemma (L = 48): 0, 6, 12, 18, 24, 30, 36, 42.
  - Qwen (L = 28): 0, 3, 7, 10, 14, 17, 21, 24.
- **Controls.**
  - **No-op hook.** Hooks are installed on every layer with an empty key set.
    The logits must equal the baseline bitwise, or the run fails.
  - **Floor.** B from layer 0 removes all access to the program after it
    ends. Accuracy there is the "no program" floor.

Each prompt takes 2 + 16 forward passes per family.

## Summary statistics (descriptive; no gate)

- Accuracy and mean log-prob for the baseline and every (variant, k), on
  calibration and on pilot separately.
- **Release layer**, per model and variant: the smallest grid k such that
  knockout from k, and from every larger grid k, keeps calibration accuracy
  within 5 points of baseline.
  - It is chosen on calibration only. The pilot reports whether it
    replicates.
  - If no grid k qualifies, the release layer is "none" (the program is
    needed through layer 42 / 24).
- Accuracy by the asked variable's last update (literal / arithmetic /
  copy) and by whether the last line assigns the asked variable.

## Predictions (recorded before any forward)

1. The no-op control reproduces the baseline exactly.
2. B from layer 0 drops accuracy to at most 0.30 on both models.
3. Both variants' release layers lie in the second half of the network
   (k ≥ L/2): the answer still reads the program late. This fits Phase
   Three's late, question-dependent readout of the asked value.
4. B's release layer is at or after A's. B blocks a superset of A's edges,
   including relays through the question tokens.

## What follows

- If a release layer is clear on calibration and replicates on pilot
  (within one grid step), a confirmatory protocol for `validation_b` can
  test two things:
  - knockout from that layer keeps accuracy within 5 points of baseline
    (non-inferiority);
  - knockout from the grid layer below it costs more than 5 points.
- If the curves are flat, non-monotone or model-specific in ways that do
  not name such a layer, `validation_b` stays reserved.

## Budget, honesty, limits

- **Budget:** at most 2.5 GPU-hours. Expected about 50 minutes for Gemma and
  25 for Qwen (about 27,600 forwards each at 0.06–0.11 s). Each run starts
  only on an idle GPU. Foreign GPU processes are logged, never acted on.
- **Data:** reused calibration and pilot. Exploratory, as Phases Three to
  Seven were. `validation_b` stays unopened.
- **Limits.**
  - Knockout pushes the model off its usual input distribution, and can
    overstate dependence (Wang et al. 2023 discuss such ablation
    artefacts).
  - It removes direct attention edges only. Under A, information can still
    flow through other later positions; B closes that route.
  - One prompt format (question after), one task family.
  - "Survives knockout from layer k" shows the answer no longer *needs*
    direct access to the program after k. It does not show where the value
    sits in the meantime.

**Code:** `src/open_weight_lingua/answer_knockout.py`, launched with
`scripts/run_answer_knockout.sh`.
