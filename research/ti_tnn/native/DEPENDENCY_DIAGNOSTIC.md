# What the saved answers depend on

Consider `add r0 r1; set r0 7; copy r2 r0`. The final `r0` comes from a
set, and final `r2` comes from a copy of that set. The earlier addition
cannot affect either answer. `r1` and `r3` are untouched. Calling the whole
answer arithmetic would hide these distinctions.

This CPU-only analysis classifies the **reference task**, then scores saved #79
completions. It does not identify the model's algorithm. No weights were loaded,
outputs regenerated, or parameters fitted. The frozen decision remains
`retain_full_sequence` for every parent.

## Definitions and assumptions

Programs are straight-line operations on four registers, with values in
`{0,...,31}` and add/multiply modulo 32. Instruction indexes start at zero.
Every read refers to the most recent preceding version of that register.

| Field | Precise meaning |
|---|---|
| Final effective write | Last assignment to this target register: set/copy/add/multiply; untouched if none. An identity write such as `copy r0 r0` still counts as copy. This is not the program's last operation. |
| Relevant instructions | Transitive read ancestors of that final version, including its own write. Set has no register inputs; copy reads its source; add/multiply read both the old destination and source. |
| Dependency depth | Longest number of instruction nodes along a path to the final version. Initial leaves have depth 0, set has depth 1, and every other write has depth one plus the maximum input depth. It is structural depth, including algebraically redundant writes. |
| Reads initial registers | Initial leaves in that read graph; a structural upper bound on functional dependence. |
| Depends on initial values | Exact global functional dependence over all initial assignments, rather than only changes around the observed initial vector. Modular cancellations and zero annihilation are accounted for. |
| Prior target writes | All writes to this target before its last assignment. Mere reassignment does not imply loss: an add can read the preceding write. |
| Discarded prior target writes | Prior writes to this target that are absent from this target's final read graph. A value retained through another register and later copied back is retained. A discarded write can still matter to another final register. |

Formally, let `P` be a fixed instruction sequence, `x` its four initial values,
and `f_j(x)` the final value of register `j`. Register `j` depends on initial
coordinate `i` if there are `x` and `x'` differing only in coordinate `i` with
`f_j(x) != f_j(x')`. The implementation enumerates all 32 values of every
structurally reachable initial coordinate, at most `32^4` assignments per target.
It tests each coordinate against its zero slice while all other coordinates vary.
This is exact for this finite reference language, not an approximation by gradients
or polynomial coefficient inspection. For example, five self-adds give `32*x=0`
modulo 32: the read graph remains nonempty, but functional support is empty.

For a graph node `v`, depth is `d(v)=0` for an initial value, otherwise
`d(v)=1+max(d(u): u is an input of v)`, with the maximum of no inputs defined as
zero. `W_j` denotes all writes to target `j`, and `R_j` its relevant instruction
set. Discarded prior writes are `(W_j minus the final write) minus R_j`.
These definitions separate overwritten writes from arithmetic that happens
elsewhere, and structural depth from exact functional support.

## Runnable implementation

Use the pinned native environment and a **new** output directory:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/ti_tnn \
  "$PY" -m native.independent.dependency_diagnostic \
  --bundle "$BUNDLE" --output "$NEW_OUTPUT"
```

[`dependency_diagnostic.py`](independent/dependency_diagnostic.py) imports no
producer executor, model, training or decomposition code. It uses the existing
independent scorer for canonical parsing and target validation, then independently
builds the versioned read graph and exhaustive support table. Its tests cover
self-copy, indirect retention, overwritten arithmetic, modular cancellation,
four-coordinate support, malformed outputs, and denominator/target mismatches.

The output directory contains aggregate `summary.json` and private
`register_details.jsonl`. The latter binds each target's classification to its
saved example and child; it is intentionally kept local. Historical artifacts
are read-only and an existing output directory is refused.

## Observed findings

All **3,840 primary outputs** were scored: six children × five populations × 128.
All 30 exact-answer and format-valid counts match the historical evaluations.
The 180 supplemental and 960 intervened outputs are excluded here; their separate
historical rescore remains unchanged. Selected intervention baselines reuse saved
outputs and add no primary observations.

- On high initial values, untouched targets score **0–1/171** per child. This is
  not only a failure to perform arithmetic. Among all initial-dependent targets,
  scores are only **3–7/391**, while initial-independent targets score **30–72/121**.
- Ordinary dev has **120/512 targets with depth >=2**. Correct counts range from
  **6 to 13/120**. Untouched targets instead score **135–159/181**. These are
  different task populations, not a causal estimate of what depth does to a model.
- Dense answer-focused training reaches **102/128** complete training answers and
  **99/113** training registers at depth >=2, but only **8/120** such dev registers.
  Training fit does not establish reusable transitions.
- Longer programs remain dominated by malformed/truncated completions: only
  **0–3/128** outputs are format-valid per child. Their zeros cannot be assigned
  solely to dependency depth or value computation.

These observations motivate separate tests of copying/preserving initial values,
one-step transitions, and composition. Multiple initial states per exact program
are worth testing, but training only on low initial values might still fail the
high-value role shift. The proposed study therefore separates in-range value
transfer from the high-initial-value diagnostic.

### Complete answers and individual registers

A = full sequence, B = answer + EOS. Every complete-answer denominator is 128.
Each `rj` cell is correct/**128**, including malformed completions as failures.
For comparison with the historical valid-only metrics, divide the same numerator
by the row's `Valid` count instead. A zero valid denominator is undefined, not 0%.
A malformed four-register string fails all four targets; correct-looking prefixes
are not salvaged. EOS and truncation preserve the saved evaluator flags.

| Child | Population | Exact /128 | Valid /128 | EOS / trunc. | r0 /128 | r1 /128 | r2 /128 | r3 /128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense A | train | 59 | 128 | 128 / 0 | 101 | 108 | 105 | 109 |
| dense A | dev | 7 | 126 | 128 / 0 | 63 | 63 | 51 | 60 |
| dense A | initial_values | 0 | 117 | 128 / 0 | 9 | 15 | 12 | 17 |
| dense A | unseen_composition | 1 | 125 | 128 / 0 | 44 | 46 | 36 | 46 |
| dense A | longer_length | 0 | 1 | 35 / 93 | 1 | 0 | 0 | 1 |
| dense B | train | 102 | 128 | 128 / 0 | 122 | 121 | 117 | 124 |
| dense B | dev | 9 | 127 | 128 / 0 | 69 | 65 | 57 | 66 |
| dense B | initial_values | 0 | 127 | 128 / 0 | 23 | 16 | 21 | 18 |
| dense B | unseen_composition | 2 | 127 | 128 / 0 | 51 | 44 | 44 | 47 |
| dense B | longer_length | 0 | 2 | 26 / 102 | 1 | 0 | 0 | 0 |
| untyped A | train | 12 | 128 | 128 / 0 | 72 | 67 | 75 | 71 |
| untyped A | dev | 3 | 128 | 128 / 0 | 53 | 53 | 45 | 40 |
| untyped A | initial_values | 0 | 123 | 128 / 0 | 18 | 13 | 18 | 17 |
| untyped A | unseen_composition | 0 | 126 | 128 / 0 | 39 | 35 | 35 | 27 |
| untyped A | longer_length | 0 | 3 | 18 / 110 | 1 | 1 | 0 | 0 |
| untyped B | train | 25 | 127 | 128 / 0 | 80 | 80 | 90 | 81 |
| untyped B | dev | 2 | 125 | 128 / 0 | 55 | 59 | 53 | 42 |
| untyped B | initial_values | 0 | 128 | 128 / 0 | 20 | 12 | 24 | 21 |
| untyped B | unseen_composition | 1 | 126 | 128 / 0 | 44 | 40 | 38 | 28 |
| untyped B | longer_length | 0 | 3 | 49 / 79 | 0 | 1 | 1 | 0 |
| typed A | train | 16 | 126 | 128 / 0 | 77 | 75 | 73 | 70 |
| typed A | dev | 2 | 128 | 128 / 0 | 59 | 62 | 48 | 40 |
| typed A | initial_values | 0 | 110 | 128 / 0 | 11 | 9 | 9 | 7 |
| typed A | unseen_composition | 1 | 125 | 128 / 0 | 44 | 40 | 36 | 31 |
| typed A | longer_length | 0 | 2 | 18 / 110 | 1 | 1 | 0 | 0 |
| typed B | train | 22 | 126 | 128 / 0 | 81 | 87 | 81 | 87 |
| typed B | dev | 4 | 128 | 128 / 0 | 52 | 59 | 48 | 55 |
| typed B | initial_values | 0 | 119 | 128 / 0 | 9 | 8 | 13 | 14 |
| typed B | unseen_composition | 1 | 127 | 128 / 0 | 33 | 31 | 36 | 40 |
| typed B | longer_length | 0 | 0 | 39 / 89 | 0 | 0 | 0 | 0 |

### Final effective write, independently for each target

Each cell is correct/all target positions of that type, including malformed
outputs. The five denominators sum to 512 for each row. The classification does
not say whether that final write reads an earlier arithmetic result; dependency
properties are separate below and in the local joint-category aggregate.

| Child | Population | Untouched | Set | Copy | Add | Multiply |
|---|---|---:|---:|---:|---:|---:|
| dense A | train | 187/190 | 81/86 | 73/86 | 35/74 | 47/76 |
| dense A | dev | 159/181 | 45/98 | 20/86 | 6/71 | 7/76 |
| dense A | initial_values | 0/171 | 43/99 | 4/85 | 2/76 | 4/81 |
| dense A | unseen_composition | 113/142 | 26/60 | 11/66 | 11/121 | 11/123 |
| dense A | longer_length | 0/37 | 0/128 | 2/129 | 0/114 | 0/104 |
| dense B | train | 190/190 | 85/86 | 85/86 | 61/74 | 63/76 |
| dense B | dev | 159/181 | 57/98 | 25/86 | 9/71 | 7/76 |
| dense B | initial_values | 1/171 | 61/99 | 6/85 | 8/76 | 2/81 |
| dense B | unseen_composition | 114/142 | 35/60 | 12/66 | 14/121 | 11/123 |
| dense B | longer_length | 0/37 | 0/128 | 1/129 | 0/114 | 0/104 |
| untyped A | train | 170/190 | 58/86 | 29/86 | 10/74 | 18/76 |
| untyped A | dev | 135/181 | 33/98 | 14/86 | 0/71 | 9/76 |
| untyped A | initial_values | 1/171 | 55/99 | 2/85 | 4/76 | 4/81 |
| untyped A | unseen_composition | 82/142 | 21/60 | 11/66 | 12/121 | 10/123 |
| untyped A | longer_length | 1/37 | 1/128 | 0/129 | 0/114 | 0/104 |
| untyped B | train | 174/190 | 65/86 | 43/86 | 14/74 | 35/76 |
| untyped B | dev | 145/181 | 35/98 | 15/86 | 2/71 | 12/76 |
| untyped B | initial_values | 0/171 | 64/99 | 4/85 | 4/76 | 5/81 |
| untyped B | unseen_composition | 93/142 | 18/60 | 9/66 | 15/121 | 15/123 |
| untyped B | longer_length | 0/37 | 2/128 | 0/129 | 0/114 | 0/104 |
| typed A | train | 174/190 | 55/86 | 42/86 | 8/74 | 16/76 |
| typed A | dev | 150/181 | 35/98 | 17/86 | 2/71 | 5/76 |
| typed A | initial_values | 0/171 | 26/99 | 3/85 | 4/76 | 3/81 |
| typed A | unseen_composition | 93/142 | 22/60 | 9/66 | 8/121 | 19/123 |
| typed A | longer_length | 1/37 | 1/128 | 0/129 | 0/114 | 0/104 |
| typed B | train | 179/190 | 67/86 | 49/86 | 14/74 | 27/76 |
| typed B | dev | 150/181 | 33/98 | 18/86 | 4/71 | 9/76 |
| typed B | initial_values | 0/171 | 34/99 | 3/85 | 4/76 | 3/81 |
| typed B | unseen_composition | 86/142 | 24/60 | 8/66 | 8/121 | 14/123 |
| typed B | longer_length | 0/37 | 0/128 | 0/129 | 0/114 | 0/104 |

### Initial dependence and discarded writes

The two initial-dependence columns partition 512 targets; the two discarded-write
columns separately partition the same 512. Their overlap is intentional: they
answer different questions. Counts are unadjusted descriptive associations.

| Child | Population | Depends on initial | Independent of initial | Has discarded prior target write | No discarded prior target write |
|---|---|---:|---:|---:|---:|
| dense A | train | 328/407 | 95/105 | 60/74 | 363/438 |
| dense A | dev | 189/403 | 48/109 | 16/58 | 221/454 |
| dense A | initial_values | 3/391 | 50/121 | 19/68 | 34/444 |
| dense A | unseen_composition | 143/439 | 29/73 | 17/59 | 155/453 |
| dense A | longer_length | 0/313 | 2/199 | 2/252 | 0/260 |
| dense B | train | 382/407 | 102/105 | 69/74 | 415/438 |
| dense B | dev | 198/403 | 59/109 | 22/58 | 235/454 |
| dense B | initial_values | 7/391 | 71/121 | 23/68 | 55/444 |
| dense B | unseen_composition | 148/439 | 38/73 | 16/59 | 170/453 |
| dense B | longer_length | 0/313 | 1/199 | 1/252 | 0/260 |
| untyped A | train | 216/407 | 69/105 | 34/74 | 251/438 |
| untyped A | dev | 156/403 | 35/109 | 13/58 | 178/454 |
| untyped A | initial_values | 6/391 | 60/121 | 18/68 | 48/444 |
| untyped A | unseen_composition | 112/439 | 24/73 | 14/59 | 122/453 |
| untyped A | longer_length | 1/313 | 1/199 | 1/252 | 1/260 |
| untyped B | train | 251/407 | 80/105 | 44/74 | 287/438 |
| untyped B | dev | 172/403 | 37/109 | 13/58 | 196/454 |
| untyped B | initial_values | 5/391 | 72/121 | 25/68 | 52/444 |
| untyped B | unseen_composition | 129/439 | 21/73 | 13/59 | 137/453 |
| untyped B | longer_length | 0/313 | 2/199 | 2/252 | 0/260 |
| typed A | train | 230/407 | 65/105 | 33/74 | 262/438 |
| typed A | dev | 173/403 | 36/109 | 14/58 | 195/454 |
| typed A | initial_values | 6/391 | 30/121 | 7/68 | 29/444 |
| typed A | unseen_composition | 125/439 | 26/73 | 16/59 | 135/453 |
| typed A | longer_length | 1/313 | 1/199 | 1/252 | 1/260 |
| typed B | train | 256/407 | 80/105 | 41/74 | 295/438 |
| typed B | dev | 180/403 | 34/109 | 14/58 | 200/454 |
| typed B | initial_values | 5/391 | 39/121 | 12/68 | 32/444 |
| typed B | unseen_composition | 114/439 | 26/73 | 12/59 | 128/453 |
| typed B | longer_length | 0/313 | 0/199 | 0/252 | 0/260 |

### Exact structural depths

Each row partitions the same 512 targets. A dash means no target in that category;
it is not a failed prediction or a dropped denominator.

| Child | Population | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense A | train | 187/190 | 165/209 | 55/85 | 14/24 | 2/4 | — | — | — |
| dense A | dev | 159/181 | 70/211 | 7/87 | 1/27 | 0/5 | 0/1 | — | — |
| dense A | initial_values | 0/171 | 44/227 | 8/82 | 1/24 | 0/8 | — | — | — |
| dense A | unseen_composition | 113/142 | 42/171 | 7/113 | 9/64 | 1/19 | 0/2 | 0/1 | — |
| dense A | longer_length | 0/37 | 0/175 | 1/130 | 1/101 | 0/50 | 0/14 | 0/4 | 0/1 |
| dense B | train | 190/190 | 195/209 | 75/85 | 21/24 | 3/4 | — | — | — |
| dense B | dev | 159/181 | 90/211 | 6/87 | 2/27 | 0/5 | 0/1 | — | — |
| dense B | initial_values | 1/171 | 65/227 | 9/82 | 1/24 | 2/8 | — | — | — |
| dense B | unseen_composition | 114/142 | 54/171 | 10/113 | 8/64 | 0/19 | 0/2 | 0/1 | — |
| dense B | longer_length | 0/37 | 0/175 | 0/130 | 1/101 | 0/50 | 0/14 | 0/4 | 0/1 |
| untyped A | train | 170/190 | 83/209 | 23/85 | 8/24 | 1/4 | — | — | — |
| untyped A | dev | 135/181 | 49/211 | 7/87 | 0/27 | 0/5 | 0/1 | — | — |
| untyped A | initial_values | 1/171 | 58/227 | 6/82 | 0/24 | 1/8 | — | — | — |
| untyped A | unseen_composition | 82/142 | 34/171 | 14/113 | 5/64 | 1/19 | 0/2 | 0/1 | — |
| untyped A | longer_length | 1/37 | 1/175 | 0/130 | 0/101 | 0/50 | 0/14 | 0/4 | 0/1 |
| untyped B | train | 174/190 | 109/209 | 35/85 | 12/24 | 1/4 | — | — | — |
| untyped B | dev | 145/181 | 53/211 | 11/87 | 0/27 | 0/5 | 0/1 | — | — |
| untyped B | initial_values | 0/171 | 66/227 | 7/82 | 3/24 | 1/8 | — | — | — |
| untyped B | unseen_composition | 93/142 | 34/171 | 14/113 | 8/64 | 1/19 | 0/2 | 0/1 | — |
| untyped B | longer_length | 0/37 | 2/175 | 0/130 | 0/101 | 0/50 | 0/14 | 0/4 | 0/1 |
| typed A | train | 174/190 | 92/209 | 24/85 | 5/24 | 0/4 | — | — | — |
| typed A | dev | 150/181 | 53/211 | 6/87 | 0/27 | 0/5 | 0/1 | — | — |
| typed A | initial_values | 0/171 | 29/227 | 4/82 | 2/24 | 1/8 | — | — | — |
| typed A | unseen_composition | 93/142 | 34/171 | 17/113 | 7/64 | 0/19 | 0/2 | 0/1 | — |
| typed A | longer_length | 1/37 | 1/175 | 0/130 | 0/101 | 0/50 | 0/14 | 0/4 | 0/1 |
| typed B | train | 179/190 | 113/209 | 33/85 | 10/24 | 1/4 | — | — | — |
| typed B | dev | 150/181 | 51/211 | 12/87 | 1/27 | 0/5 | 0/1 | — | — |
| typed B | initial_values | 0/171 | 36/227 | 7/82 | 0/24 | 1/8 | — | — | — |
| typed B | unseen_composition | 86/142 | 37/171 | 11/113 | 5/64 | 1/19 | 0/2 | 0/1 | — |
| typed B | longer_length | 0/37 | 0/175 | 0/130 | 0/101 | 0/50 | 0/14 | 0/4 | 0/1 |

## Evidence identity and limits

Aggregate JSON SHA256: `e049236ea2f7df0d257bcaeb596f8b64b45b2e5f77659364bb72d2fda908c092`.

The aggregate records SHA256 values for its five source JSONL files and six
saved evaluation files. The input bundle's 260 payload checks and the documented
4,980-output historical rescore also passed during takeover. Those checks identify
local artifacts; they do not authenticate a remote execution or independently
replicate learning. No teacher-forced CE, state readout or numerical checkpoint
replay was recomputed for this diagnostic. The earlier strict-allclose failures
and scale-based CPU FP32 qualification remain in force.

All populations are reused development evidence, one trained seed, with correlated
registers within programs and paired outputs across objectives. Final-write,
depth, initial support and overwrite strata are not randomized treatments. No
architectural ranking, internal-mechanism explanation or statistical superiority
follows. In particular, exact symbolic dependence of the **reference answer** is
not evidence that the neural model represents that dependence.

The reference read slicing follows established program-analysis ideas
([Weiser, 1984, *Program Slicing*](https://doi.org/10.1109/TSE.1984.5010248)).
Separating familiar primitives from new combinations is an established testing
principle ([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html)).
This is a diagnostic application of those ideas, not a new discovery about
slicing or compositional generalization. See the [next-study draft](INITIAL_STATE_STUDY.md).

## Changed versus unchanged reference values (v2, 2026-11-13)

For an initial vector `x`, a program `P`, and target register `j`, call the target
**changed** exactly when `F_P(x)[j] != x[j]`; otherwise call it **unchanged**.
This is an example-specific value comparison, independent of final-write type,
functional initial support and depth. An arithmetic write may preserve a value;
a constant set may change it. Neither category identifies the model's algorithm.

All 512 register positions remain counted per child/population. A malformed
completion fails all four positions in their reference categories. The JSON
also reports valid-only denominators and each register's changed/unchanged split;
an empty group has undefined accuracy. The two groups' correct counts sum to
the existing all-register count. Every preexisting v1 aggregate field and all
historical complete-answer scores were checked equal after rescoring saved text.
The original aggregate and evidence remain unchanged.

A = full sequence, B = answer + EOS. Cells show A → B correct/total positions.

| Architecture | Population | Changed | Unchanged |
|---|---|---:|---:|
| dense | train | 203 → 260 / 288 | 220 → 224 / 224 |
| dense | dev | 61 → 82 / 297 | 176 → 175 / 215 |
| dense | initial_values | 51 → 75 / 313 | 2 → 3 / 199 |
| dense | unseen_composition | 48 → 57 / 342 | 124 → 129 / 170 |
| dense | longer_length | 2 → 1 / 457 | 0 → 0 / 55 |
| untyped | train | 87 → 130 / 288 | 198 → 201 / 224 |
| untyped | dev | 36 → 43 / 297 | 155 → 166 / 215 |
| untyped | initial_values | 64 → 76 / 313 | 2 → 1 / 199 |
| untyped | unseen_composition | 32 → 39 / 342 | 104 → 111 / 170 |
| untyped | longer_length | 1 → 2 / 457 | 1 → 0 / 55 |
| typed | train | 91 → 128 / 288 | 204 → 208 / 224 |
| typed | dev | 40 → 44 / 297 | 169 → 170 / 215 |
| typed | initial_values | 36 → 44 / 313 | 0 → 0 / 199 |
| typed | unseen_composition | 40 → 38 / 342 | 111 → 102 / 170 |
| typed | longer_length | 1 → 0 / 457 | 1 → 0 / 55 |

Dense's ordinary-development gain is concentrated in changed values (+21 correct),
while unchanged values decrease by one. Untyped's gain is larger for preservation
(+11) than changed values (+7); typed gains four and one respectively. This sharpens
what to measure in the proposed multi-state study without changing #79's frozen
`retain_full_sequence` decision or the new complete-answer development screen.
The high-initial-value population still preserves only 0–3/199 values per child,
and longer outputs remain dominated by malformed strings. These remain weak
reused development results, not evidence for reusable neural transitions.

V2 aggregate SHA256: `c6e167ab29be14effe6becca4da704d041e8a61a84dc6a62feffa9089f23a36d`.
Rescoring made zero model calls. The main study will report these two value
categories alongside its unchanged complete-answer endpoint and screen.
