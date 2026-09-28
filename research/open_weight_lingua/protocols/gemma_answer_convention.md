# Gemma-family answer convention amendment — Phase Two open_weight_lingua

**Status: FROZEN 2026-09-27, before the Gemma-12B confirmation pilot.** This
amendment changes the answer *comparison* instrument for Gemma-family runs
only. It does not apply retroactively: the closed Qwen2.5-7B and the two
Gemma pilots keep their frozen outcomes and decisions untouched
(`pilot-20260921T235825Z-6164d210`, `pilot-20260923T043613Z-f7e71d7b`,
`pilot-20260926T185550Z-d085d53c`), because thresholds and instruments are
frozen per phase. The evidence motivating the fix is the failure taxonomy in
[reports/post_hoc_answer_lens.md](../reports/post_hoc_answer_lens.md): 228 of
512 Gemma-12B pilot variants were rejected for a trailing `"\n"` (221 with the
correct value), while the Qwen pilot had zero such rows.

## The amended rule (exact text)

For Gemma-family runs declared with `--answer-convention rstrip`, an answer
counts as exactly correct iff the generation terminated and

```
exact_integer(text.rstrip(), expected_answer)
```

— trailing ASCII whitespace (space, `\t`, `\n`, `\r`, `\f`, `\v`) is stripped
once, at comparison time, before the same canonical-integer fullmatch used by
the frozen convention. Leading whitespace, internal whitespace, commentary,
and leading zeros remain failures, exactly as before. The default convention
is `raw` (`exact_integer(text, expected)`), which is byte-identical to the
closed pilots' instrument and remains the convention for Qwen runs. The
convention in force is recorded in the run manifest as `answer_convention`.

## What does not change

- The teacher-forced scoring channel is untouched. `answer_tokens` still
  appends the canonical answer token IDs plus the tokenizer's EOS: for the
  mirror Gemma target that is digits + `<end_of_turn>` (id 106) — verified on
  the real tokenizer (e.g. "19" → `[236770, 236819, 106]`). The scored suffix
  carries no newline; the model's preferred emitted suffix is digits +
  `"\n"` + `<end_of_turn>`. The amendment does not touch this channel; the
  divergence is noted, not silently reconciled.
- The task generator, the split plan, the 128 pilot groups, the calibration
  artifacts, the thresholds, the bootstrap (3,000 resamples, seed 203100), and
  the stopping rule are unchanged. The calibration fit is reused as-is:
  calibration is a target-only extraction with no answer-metric involvement,
  so the convention cannot affect it.
- The same 128 pilot groups are kept for direct comparability with the closed
  Gemma-12B pilot: only the metric lens changes, so every difference in
  outcomes is attributable to the instrument, not the sample.

## Scope and limits

This is an instrument fix proven by a saved-evidence taxonomy, not a new
scientific claim channel. A confirmation run under the amended instrument
produces its own frozen-convention record; it does not relabel any closed
pilot, and it does not vindicate the language route beyond this family, task
and site. The edit hypothesis remains untested regardless of the outcome
(AV-side coverage is unaffected by the answer metric).
