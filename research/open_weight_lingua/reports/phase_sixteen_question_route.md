# Phase Sixteen: with the question first, the asked variable reaches the late read through the program — exploratory — 2026-10-12

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

With the question first, the late read happens at the chat-template
positions after the program, and it must know which variable was asked
(Phases Fourteen and Fifteen). Phase Sixteen
([brief](../protocols/phase_sixteen_question_route_brief.md)) cut the two
possible routes by blocking attention at every layer:
- **reader → question:** the post-program positions cannot see the
  question;
- **program → question:** the program tokens cannot see it.

**The prediction was wrong.** The brief predicted a direct route. Instead,
the asked variable reaches the read mainly **through the program tokens**.
- **Program blind to the question.** On programs that end on the other
  variable, Gemma's accuracy falls from 0.813 to 0.194, and 73% of its
  answers become the other variable's value.
  - On programs that end on the asked variable, accuracy barely moves
    (0.900 → 0.849).
  - So the read falls back to the most recent assignment of *either*
    variable.
  - Qwen shifts the same way, more weakly: other-value answers rise from
    24% to 41%.
- **Reader blind to the question.** Accuracy drops but other-value answers
  *fall*, and Qwen's accuracy goes to exactly 0. That fits losing the
  answer format ("Reply with only the integer"), not the variable. So the
  direct route is not ruled out; this design cannot test it.

Of five predictions, only the no-op control held cleanly. These are
exploratory results on opened programs.

## What ran

- **Freeze.** The brief, module, tests and launcher were committed and
  pushed at `81ea9c8` before any forward. Both manifests record that commit
  and the brief's hash `ff800ad8a702…`. All 118 source hashes equal the
  files at `81ea9c8`.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, question first only.
- **Spans.** The question (10 tokens in both tokenizers), its variable-name
  token and the program span were located by decoding.
- **Conditions.** From layer 0. The bitwise no-op control held on every
  prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p16-route-gemma3-12b-20261012T214843Z-c98b4f59` | `835891a9…` / `37974a66…` / `0aa0df1c…` | 3,963 s / 4,109 s |
| Qwen2.5-7B | `p16-route-qwen2.5-7b-20261012T230654Z-b049abb5` | `41f2604a…` / `cd8bbf11…` / `9b287b6e…` | 2,246 s / 2,332 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB).

## Results

Accuracy, with the rate of saying the other variable's final value in
parentheses. The main stratum is programs ending on the other variable
whose answer does not depend on it (758). "Asked last" is programs ending
on the asked variable (768).

| Condition | Gemma, main | Gemma, asked last | Qwen, main | Qwen, asked last |
|---|---|---|---|---|
| Baseline | 0.813 (14.6%) | 0.900 (3.0%) | 0.393 (24.1%) | 0.663 (5.4%) |
| **Program → question blocked** | **0.194 (73.1%)** | **0.849** (6.7%) | **0.158 (40.9%)** | **0.512** (10.0%) |
| Reader → question blocked | 0.591 (4.6%) | 0.484 (3.4%) | 0.000 (0.0%) | 0.000 (0.0%) |
| Both blocked | 0.008 (12.4%) | 0.115 (0.3%) | 0.000 (0.0%) | 0.000 (0.0%) |
| Program → name token blocked | 0.702 (26.2%) | 0.905 (3.0%) | 0.347 (28.8%) | 0.656 (5.5%) |
| Reader → name token blocked | 0.781 (16.6%) | 0.898 (3.0%) | 0.383 (23.3%) | 0.665 (5.2%) |
| Both → name token blocked | 0.685 (27.9%) | 0.904 (3.0%) | 0.335 (29.1%) | 0.647 (5.9%) |

**Predictions.**
1. The no-op control is exact. *Held.*
2. **Direct route:** blinding the reader costs at least 30 points on the
   main stratum and raises other-value answers by at least 20. *Failed.*
   - Gemma lost 22 points and Qwen 39, but other-value answers *fell* in
     both (by 10 and 24 points).
3. **No marking through the program:** blinding the program costs at most
   5 points overall. *Failed:* it cost 33 points (Gemma) and 19 (Qwen).
4. `both_blind` is at most 5 points below `reader_blind`. *Failed for
   Gemma* (0.061 against 0.535). *Held for Qwen,* trivially, at 0 against
   0.
5. **Recency default under `reader_blind`:** asked-last accuracy exceeds the
   main stratum's by at least 30 points. *Failed:* Gemma −11; Qwen 0.

**Post hoc.** The recency default predicted for `reader_blind` appears
under `program_blind` instead. Each program is asked about both variables,
so it contributes one prompt to each stratum. Within programs, the main
stratum's loss exceeds the asked-last loss by:
- **Gemma: +0.567**, 99.17% bootstrap interval at 512 groups about
  [+0.50, +0.64];
- **Qwen: +0.080**, about [+0.01, +0.15].

## What this establishes (exploratory)

- **With the question first, the program tokens carry the asked variable
  to the late read.**
  - When they cannot attend to the question, the read cannot tell the
    variables apart. It takes the most recent assignment of either one.
  - The program stays readable: programs ending on the asked variable are
    still answered (Gemma 0.849).
  - In Gemma this route does most of the work. In Qwen it does part.
- **The name token alone is not the carrier.** Blocking only it from the
  program costs 11 points (Gemma) and 5 (Qwen) on the main stratum. The
  later question tokens evidently also carry the name.
- **Together with Phase Fifteen.** With the question first, the asked
  variable is marked on the program as it is read. The marking is
  imperfect, and the read's competing preference for the most recent
  assignment then wins some of the time: the other-variable errors.
- **Contrast with the question after.** There, the program tokens cannot
  see the question, so the read must take the variable from the question
  tokens. In Gemma, the question tokens do the reading themselves (Phase
  Eight).

## What this does not establish

- **The direct route.** Blinding the reader to the whole question also
  hides the answer format, so its result is uninterpretable for variable
  choice. A test that blocks only `What is x?` would separate them.
- **What "marking" is.** The program tokens might tag the asked variable's
  statements, or carry the variable forward to the end of the program.
  This design does not say which.
- **Confirmation.** Opened programs only.
- **Scope.** Off-distribution blocks; one wording.

Code:
[question_route_knockout.py](../src/open_weight_lingua/question_route_knockout.py),
launched with `scripts/run_question_route_knockout.sh`.
