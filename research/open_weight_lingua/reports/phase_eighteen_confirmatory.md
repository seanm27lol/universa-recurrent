# Phase Eighteen confirmatory: the layers where the program reads the question, on fresh programs — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Eighteen
([report](phase_eighteen_marking_layers.md)) found that, with the question
first, the program tokens read the question in the middle layers, before
the late read. Six claims about the edges of that window were frozen
([protocol](../protocols/phase_eighteen_confirmatory_fresh.md)) before the
test programs existed. They were tested once on **768 newly generated
groups**: seed 213001, excluding every Phase Two program and the seven
earlier fresh splits.

**Four of six were supported.**
- **The window ends where predicted.** Blocking the program from the
  question from layer 24 on (Gemma, of 48) or 17 on (Qwen, of 28) is
  harmless (G1, Q1).
- **It has begun early.** Blocking below layer 12 (Gemma) or 10 (Qwen)
  raises other-value answers (G3, Q3).
- **Not supported: G2 and Q2.** The claim was that marking continuing past
  layer 18 (Gemma) and 14 (Qwen) carries more than 5 points. It came out at
  +0.056 [+0.036, +0.077] and +0.058 [+0.034, +0.083]: resolved above zero,
  but not above the threshold.

So on unseen programs, most of the marking is done by layer 18 (Gemma) and
14 (Qwen). A small remainder continues to 24 and 17.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `18b6598`. The tests used test-only seeds.
  - Both manifests record that commit and the protocol hash
    `ce2d6700746c…`. All 133 source hashes equal the files at `18b6598`.
- **Fresh programs.**
  - 768 groups under `fresh_2026_10h`: 1,722 attempts. 223 candidates were
    rejected for repeating an earlier program, and 731 were invalid pairs.
  - The seven earlier fresh splits regenerated to their recorded
    inventories before being excluded.
  - Both families received the same programs (inventory `fb17767b3304…`).
  - **1,412 prompts in 758 groups** where the mark decides the answer
    (programs ending on the other variable, with an independent answer and
    a distinct other value).
- **Settings.** As frozen. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p18-confirm-gemma3-12b-20261013T100005Z-7c3d752d` | `aa2e9f66…` / `9ca61e5b…` | 2,274 s / 2,425 s |
| Qwen2.5-7B | `p18-confirm-qwen2.5-7b-20261013T105047Z-809ab6f4` | `c6b8b9c8…` / `670a4426…` | 1,314 s / 1,408 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB). One CPU evaluation wrote
`p18-confirm-evaluation-20261013.json` (`d042c4ee…`).

## Results

Accuracy and the rate of saying the other variable's final value (1,412
prompts):

| Gemma | baseline | blocked layers 24–47 | blocked layers 18–47 | blocked layers 0–11 |
|---|---|---|---|---|
| Accuracy | 0.816 | 0.801 | 0.740 | 0.625 |
| Says the other value | 0.129 | 0.140 | 0.185 | 0.317 |

| Qwen | baseline | blocked layers 17–27 | blocked layers 14–27 | blocked layers 0–9 |
|---|---|---|---|---|
| Accuracy | 0.405 | 0.410 | 0.309 | 0.283 |
| Says the other value | 0.184 | 0.177 | 0.242 | 0.282 |

**The six confirmatory hypotheses** (rise in other-value answers). These are
99.17% whole-group bootstrap intervals (10,000 resamples; seed 810100
Gemma, 810101 Qwen), Bonferroni across the six tests.

| ID | Blocked layers | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| G1 | Gemma 24–47 | +0.011 | [+0.004, +0.020] | upper < 0.05 | **Supported** |
| G2 | Gemma 18–47 | +0.056 | [+0.036, +0.077] | lower > 0.05 | **Not supported** |
| G3 | Gemma 0–11 | +0.188 | [+0.153, +0.226] | lower > 0.05 | **Supported** |
| Q1 | Qwen 17–27 | −0.007 | [−0.016, +0.001] | upper < 0.05 | **Supported** |
| Q2 | Qwen 14–27 | +0.058 | [+0.034, +0.083] | lower > 0.05 | **Not supported** |
| Q3 | Qwen 0–9 | +0.098 | [+0.064, +0.133] | lower > 0.05 | **Supported** |

**Predictions.**
- All six were predicted to be supported. G2 was flagged as the closest
  call; it failed, and so did Q2.
- Both came out at about 60% of their exploratory values: G2 +0.056 against
  +0.092, Q2 +0.058 against +0.100.
- G3 and Q3 also shrank, to +0.188 against +0.269 and +0.098 against
  +0.118, but stayed well above their threshold.

## What this establishes

- **On unseen programs, with the question first, the program reads the
  question in the middle layers, before the late read.**
  - **Gemma.** The marking has begun below layer 12 and is finished by
    layer 24, where the late read begins. Most of it is done by layer 18;
    the rest is resolved above zero but below 5 points.
  - **Qwen.** The marking has begun below layer 10 and is finished by layer
    17, ahead of its late read in layers 21–23. Most of it is done by layer
    14.
- **The two-stage account, with the question first:**
  1. In the middle layers, the program tokens read the question and mark
     both variables' statements (Phases Sixteen to Eighteen).
  2. In the late layers, the post-program positions read the marked
     program and take the asked variable's most recent intact assignment.
  3. Without the marks, the read falls back on recency (Gemma).

## What this does not establish

- **Single-layer edges.** The window is confirmed at the tested layers
  only.
- **The form of the marks.**
- **Scope.** Off-distribution blocks; one wording; two models.

Code: [layers_confirm.py](../src/open_weight_lingua/layers_confirm.py),
launched with `scripts/run_layers_confirm.sh`.
