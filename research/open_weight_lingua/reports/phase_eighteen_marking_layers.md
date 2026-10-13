# Phase Eighteen: with the question first, the program reads the question in the middle layers, before the late read — exploratory — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

Phases Sixteen and Seventeen found that, with the question first, the
program tokens attend to the question to mark which statements belong to
the asked variable. Block that attention at every layer, and the late read
takes the other variable's value. Phase Eighteen
([brief](../protocols/phase_eighteen_marking_layers_brief.md)) applied the
same block to only some layers, to locate the marking in depth.

**All predictions held. The marking happens in the middle layers and ends
just before the late read begins.**
- **Gemma: about layers 6–23 of 48.**
  - Blocking from layer 24 onward is harmless.
  - Blocking from 18 onward still costs, and so does blocking only layers
    0–11.
  - The read starts at 24 (Phases Eight and Fourteen).
- **Qwen: about layers 7–16 of 28.**
  - Blocking from 17 onward is harmless.
  - Blocking from 14 onward still costs. Blocking only layers 0–2 does
    nothing, and blocking layers 0–6 nearly nothing.
  - The read runs in layers 21–23.

These are exploratory results on opened programs.

## What ran

- **Freeze.** The brief, module, tests and launcher were committed and
  pushed at `783b9b0` before any forward. Both manifests record that commit
  and the brief's hash `db8c59bae2da…`. All 130 source hashes equal the
  files at `783b9b0`.
- **Prompts.** Phase Eight's calibration and pilot prompts in the
  question-first order, kept where the mark decides the answer: programs
  ending on the other variable, with an answer that does not depend on it,
  and a distinct other value. That is 718 per family (482 calibration, 236
  pilot).
- **Conditions.**
  - The block (every program token → every question token) in all layers;
  - from layer k on (`from_k`), or below layer k (`until_k`), for k on
    Phase Eight's grid.
- **Scoring.** Teacher-forced, for the answer and for the other variable's
  final value. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p18-layers-gemma3-12b-20261013T073926Z-66a828f5` | `06e7cd9e…` / `349b1ffa…` / `4ec633e2…` | 3,959 s / 4,104 s |
| Qwen2.5-7B | `p18-layers-qwen2.5-7b-20261013T085801Z-61a6e8b7` | `acd9d2b4…` / `722f8101…` / `31a22d38…` | 2,247 s / 2,336 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB).

## Results

**Rise in other-value answers over baseline**, all 718 prompts. Baseline
rates: Gemma 0.146, Qwen 0.241.

| Gemma: k | 6 | 12 | 18 | 24 | 30 | 36 | 42 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blocked from layer k on | +0.535 | +0.341 | +0.092 | **+0.014** | 0.000 | 0.000 | 0.000 |
| Blocked below layer k | **+0.038** | +0.269 | +0.550 | +0.585 | +0.586 | +0.586 | +0.585 |

Blocked in every layer: +0.585.

| Qwen: k | 3 | 7 | 10 | 14 | 17 | 21 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blocked from layer k on | +0.272 | +0.214 | +0.124 | +0.100 | **−0.011** | −0.001 | −0.006 |
| Blocked below layer k | **−0.010** | +0.006 | +0.118 | +0.135 | +0.187 | +0.175 | +0.169 |

Blocked in every layer: +0.169.

**The marking window** (tolerance +0.05; calibration / pilot):

| Family | End: smallest k with `from_k` harmless from k on | Onset: largest k with `until_k` harmless up to k |
|---|---|---|
| Gemma | 24 / 24 | 6 / none (`until_6` +0.051 on pilot) |
| Qwen | 17 / 17 | 7 / 7 |

**Predictions.**
1. The no-op control is exact. *Held.*
2. The all-layer block reproduces Phase Sixteen. Other-value answers rise
   by at least 30 points (Gemma) and 8 points (Qwen). *Held:* +58.5 and
   +16.9, the same cells as Phase Sixteen's exploratory run.
3. **Marking precedes the read.** The window ends at or before layer 24
   (Gemma) and 17 (Qwen), on both splits. *Held,* exactly at the bound: 24
   and 17.
4. **The first layers do not mark.** Blocking the first grid interval
   raises other-value answers by less than 5 points. *Held:* +0.038 for
   Gemma (`until_6`) and −0.010 for Qwen (`until_3`).
   - On Gemma's pilot split alone, `until_6` reaches +0.051, just over the
     tolerance. So Gemma's onset is "by about layer 6", not later.

**A detail.** In Qwen, blocking from layer 3 on raises other-value answers
more than blocking every layer does (+0.272 against +0.169). Also blocking
layers 0–2 seems to disrupt the read more broadly, so fewer of the errors
are the other variable's value. This fits the broader disruption Phase
Sixteen's confirmatory test found for Qwen.

## What this establishes (exploratory)

- **With the question first, the model works in two stages.**
  - **Middle layers: marking.** The program tokens read the question and
    mark both variables' statements: Gemma in about layers 6–23, Qwen in
    about 7–16.
  - **Late layers: reading.** The post-program positions read the marked
    program: Gemma from about layer 24, Qwen in 21–23.
  - The marking ends where the reading begins in Gemma (layer 24), and a
    few layers before it in Qwen.
- **Most of the marking is mid-network.** Blocking below layer 12 already
  gives Gemma about half the full effect (+0.269 of +0.585). Blocking from
  layer 12 on gives a little more than half (+0.341). The marking is
  spread over roughly a third of the network.

## What this does not establish

- **Confirmation.** Opened programs only.
- **Single-layer resolution.** The grid steps are 6 layers (Gemma) and 3–4
  (Qwen).
- **The form of the marks.**
- **Scope.** Off-distribution blocks; one wording.

Code:
[marking_layers_knockout.py](../src/open_weight_lingua/marking_layers_knockout.py),
launched with `scripts/run_marking_layers_knockout.sh`.
