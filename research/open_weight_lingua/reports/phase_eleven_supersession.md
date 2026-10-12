# Phase Eleven: a disrupted final assignment makes the model fall back to the previous one — exploratory — 2026-10-12

```text
x = 4
y = 3
x = 7              ← the asked variable's final statement
y = 5
What is x? Reply with only the integer.      → 7   (superseded value: 4)
```

Phase Ten ([report](phase_ten_dependency_knockout.md)) found that blocking
the asked variable's final statement from earlier assignments of that
variable damages the answer, even for a literal. Phase Eleven
([brief](../protocols/phase_eleven_supersession_brief.md)) compared that
block with a control block, and also scored the superseded value.

**Disruption, not a variable-specific mark, but with a telling failure
mode.**
- Blocking the final statement from the *other* variable's earlier lines,
  which are as long on average, hurts too: Gemma literals 0.968 → 0.629;
  Qwen literals 0.896 → 0.457.
- Under either block, most wrong answers are the **superseded value**, the
  asked variable's previous assignment. That holds for 64–92% of wrong
  answers across models, kinds and blocks.

So when the latest assignment is damaged, the late read retrieves the
previous one.

These are exploratory results on reused calibration and pilot prompts.

## What ran

- **Brief.** Frozen and pushed at `884fba8` before any Phase Eleven forward.
  Both manifests record that commit, the brief's hash `ae2b29f83562…`, and
  source hashes identical to the frozen code.
- **Prompts.** 954 per family: final asked statement a literal with earlier
  assignments (280; 260 with a superseded value different from the answer)
  or arithmetic (674).
- **Blocks.** From layer 0, the final statement's tokens are blocked from
  either the asked variable's earlier lines (`same_variable`) or the other
  variable's earlier lines (`other_variable`). The two key sets average
  about 9 tokens each.
- **Scoring.** Teacher-forced digit accuracy for the answer and for the
  superseded value. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p11-supersession-gemma3-12b-20261012T044749Z-74fe4a40` | `e3c4af13…` / `3537ad57…` / `3d3d53fb…` | 1,228 s / 1,375 s |
| Qwen2.5-7B | `p11-supersession-qwen2.5-7b-20261012T051959Z-6c54dd2b` | `5c7fb5c1…` / `a94f8247…` / `9e3c082d…` | 697 s / 784 s |

Both runs were admitted to an idle GPU, with no foreign processes.

## Results (all prompts; calibration and pilot agree in direction)

| | Baseline | Same-variable block | Other-variable block (control) |
|---|---|---|---|
| **Gemma, literal** (280): accuracy | 0.968 | 0.479 | 0.629 |
| superseded value given (rate) | 0.019 | 0.454 | 0.412 |
| share of wrong answers that are the superseded value | 0.56 (of 9) | 0.74 | 0.92 |
| **Gemma, arithmetic** (674): accuracy | 0.895 | 0.585 | 0.728 |
| share of wrong answers superseded | 0.10 | 0.64 | 0.69 |
| **Qwen, literal** (280): accuracy | 0.896 | 0.579 | 0.457 |
| superseded value given (rate) | 0.073 | 0.400 | 0.542 |
| share of wrong answers superseded | 0.52 | 0.79 | 0.82 |
| **Qwen, arithmetic** (674): accuracy | 0.709 | 0.322 | 0.399 |
| share of wrong answers superseded | 0.22 | 0.55 | 0.68 |

**Preference** for the superseded value, log P(old) − log P(answer), mean
over prompts:
- Gemma literals: −14.1 at baseline, −0.6 under the same-variable block,
  −3.2 under the control.
- Qwen literals: −11.4, −1.6 and **+4.8**.

**Predictions.**
1. The no-op control is exact. *Held.*
2. The same-variable block reproduces Phase Ten's literal damage. *Held:*
   49 (Gemma) and 32 (Qwen) points.
3. The control costs under 10 points. **Wrong** for both models: 34 and 44
   points for literals.
4. Under the same-variable block, the superseded preference rises by more
   than 2 nats and the superseded rate exceeds 25% of literal prompts.
   *Held* (+13.5 and +9.7 nats; 45% and 40%). But the control does nearly
   the same, so this is **not specific** to the same-variable block.

## What this establishes (exploratory)

- **Phase Ten's effect is not a same-variable mark.** Blocking the final
  statement from either variable's earlier lines degrades it. In Qwen the
  control block is even more damaging.
- **The failure mode is specific: fallback to the previous assignment.**
  Once the final assignment is degraded, the answer is mostly the asked
  variable's previous value. This holds across both blocks, both kinds of
  final statement and both models.
- **Combined with Phases Eight and Nine:** the late read retrieves the
  asked variable's statements. When the latest one is corrupted, the
  previous assignment is retrieved instead. The late read appears to select
  among the asked variable's assignments, with the most recent intact one
  winning.

## What this does not establish

- **Why any context block degrades the final statement.** For example, it
  may lose position or structure cues. Only layer-0 blocks were tested.
- **"Most recent intact" is an interpretation.** It fits the fallback
  pattern but was not tested directly, for example by corrupting the
  previous assignment too.
- **Knockout limits.** An off-distribution intervention; teacher-forced
  scoring; reused data; exploratory.

## Next

A confirmatory test on fresh programs (a new seed, excluding every earlier
program) of three claims, for literal final statements in both models:
- the same-variable block damages the answer;
- the control block also damages it;
- under the same-variable block, the superseded value's rate rises by more
  than 10 points.

Code: [supersession_knockout.py](../src/open_weight_lingua/supersession_knockout.py),
launched with `scripts/run_supersession_knockout.sh`.
