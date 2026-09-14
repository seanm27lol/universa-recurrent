# Can a small mathematical description preserve the next computation?

**Implemented experiment, not yet a DGX result. No trained natural-language
interpreter. Existing weights, update rules, thresholds and package version
0.5.2 are unchanged.**

## Start with a familiar example

Pause a calculation halfway through. You could save every digit of the working
numbers, round them, or forget them. On resuming, does the answer change? Keeping
the original problem on the desk can let you recover from a poor note, so test
that alternative explanation too.

```text
                     frozen four-step or eight-step solver
                                      |
                               pause at one cut
                                      |
                      coordinates + current route evidence
                                      |
                 numerical description with a fixed byte budget
                                      |
                              reconstruct that state
                                      |
               resume once, with the original remaining updates
                                      |
                compare estimates, route choices and claim masks
```

The original problem stays on the solver's desk. It does NOT go to the decoder.
The initial experiment covers one cut, not a whole window of states in one note.
A later temporal-description experiment must first pass this resumption boundary.

## What is actually being saved?

At step `t`, candidate `k` has cycle coordinates `a_(t,k)`. Its ambient flow is
`z_(t,k) = Q_k a_(t,k)`, where `Q_k` is its known cycle basis. Current route logits
`ell_t` determine weights through `softmax(ell_t)`.

The dynamic state is the concatenation `v_t = [all a_(t,k), ell_t]`. For the current
two-candidate, two-coordinate task, that is SIX float32 numbers. Leaving out the
route evidence would not be a full restoration.

The solver also retains observations, masks, encoder context, and prior route
logits. Their per-example tensor bytes are explicitly reported. Fixed weights
and the candidate library are additional common context, not encoded in the note.
The decoder only sees its payload and calibrated codec manifest. It never sees
input observations, cached encoder features, ground truth or later activations.

**A smaller dynamic-state payload is not a smaller complete model or memory
footprint.** In particular the toy dynamic state is only 24 bytes. A record's
32-byte codec hash alone is larger than that; common metadata and unchanged
context must not disappear from the accounting.

## The mathematical description and control

The named codec encodes each coordinate/evidence field using symmetric uniform
fixed-point rounding at 4, 8, 12 or 16 bits. Each field's range is 1.05 times its
maximum absolute value on the codec-calibration split, floored at 1e-8. Test values
outside that range are clipped AND counted. Ranges are never refitted on test data.
For range `s` and `qmax = 2^(bits-1)-1`:

```math
q = round(clip(v/s, -1, 1) * qmax)
v_reconstructed = q * s / qmax
```

An equally sized numeric-payload control first rotates the six fields by one
fixed orthogonal matrix, quantizes, and rotates back. Its per-state bytes and
bit width match the named codec. Its shared matrix/scale metadata can cost more;
that difference is reported, so this is not a claim of equal total wire cost.
Both codecs ultimately restore structured coordinates. This tests coordinate
choice for compression, NOT whether mathematical structure beats an unconstrained
architecture or whether the named coordinates capture discovered semantics.

The generated readable text lists known cycle coordinates and evidence values.
It is a display of the decoded mathematical fields, NOT a secret extra input to
the decoder and NOT an independently learned interpretation of hidden neurons.
Its UTF-8 byte size is recorded separately from the binary numerical record.

## Required controls and gates

| Case | Purpose |
|---|---|
| Unmodified full rollout | Reference answer, with original weights and updates |
| Raw float32 encode/decode/resume | Mandatory full-state gate; failures stop the study |
| Named versus rotated, 4/8/12/16 bits | Match numerical payload budget and vary coordinate representation |
| Zero all dynamic state | Can unchanged context rebuild the answer anyway? |
| State from another input, circularly paired | Does the particular input's state matter? Diagnostic only, not a deployable codec |
| Erase only coordinates / only logits | Check which named state component matters |
| Stop at the original cut | Distinguish a good early answer from useful continuation |

Cuts are fixed as `{max(1, depth//4), max(1, depth//2), depth-1}`: 1/2/3 for four
steps and 2/4/7 for eight steps. All reconstructed states at a cut receive the SAME
remaining budget. The stop-at-cut diagnostic deliberately receives zero additional
updates and is labeled separately. No injection of a later reconstruction occurs.

The raw control requires all output fields within existing numerical tolerances,
with identical routes and claim decisions. Lossy controls need NOT pass that test:
all deviations, per-example errors, clipping and discrete changes are results,
not reasons to hide or abort an otherwise valid experiment.

## Evidence and costs

Primary measurements: final estimate MSE versus uninterrupted continuation, maximum
absolute estimate change, task MSE, route/claim changes, clipping, and separate
coordinate/logit reconstruction errors. Per-example arrays are saved as numeric
NPZ files (read with `allow_pickle=False`). Codec manifests and the first two
readable samples per named condition are retained for inspection without selecting
especially successful examples. No model weights are included in the reports ZIP.

One-pass encode/decode/continuation wall times are recorded for scale only. They
are NOT a randomized repeated latency benchmark. No compression winner or bit
width is selected on these test outcomes automatically. Common metadata, record
hashes, retained context and actual value bytes have separate counts.

The model and stored structural-claim cutoffs stay frozen. A separate 1,024-input
codec-calibration cohort (seed 61000) sets ranges. A shared 1,024-input test cohort
(seed 62000) evaluates all five checkpoints. The script checks overlap with the
checkpoint training and claim-calibration seeds. Whole inputs, not adjacent states,
are split. This is one common cohort, not five times as many independent inputs.

## What success would and would not mean

Good continuation supports numerical sufficiency under the retained context and
remaining budget. If zero/shuffled states recover equally well, the compact note
has not earned credit for the result. If small state error changes a claim, that
sensitivity must be reported even when task MSE barely moves.

This is not a trained NLA, semantic explanation, state-discovery mechanism,
verified execution trace, or general causal-abstraction result. The next step
would test temporal summaries or explicit semantic edits only after this baseline.
Do not silently promote numerical compression into language interpretability.

## Grounding in existing work

[Natural language autoencoders](https://transformer-circuits.pub/2026/nla/)
use a verbalizer and reconstructor with an actual language bottleneck. Their
reconstruction objective is not by itself a guarantee of explanation reliability.
Our first controlled experiment is much narrower: known mathematical fields and
an explicit numerical codec, without either trained language module.

[Causal abstraction](https://www.jmlr.org/papers/v26/23-0058.html) motivates asking
whether interventions preserve the relevant behavior, not merely whether vectors
look similar. This test is not a full abstraction validation. PyTorch
[reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
motivates recording execution settings and comparing identical within-run baselines.

## Run on the existing DGX checkout

```bash
git pull --ff-only origin main
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_release.py
.venv/bin/python -m pytest -q tests/test_continuation_codec.py tests/test_state_continuation.py
bash scripts/run_state_continuation_study.sh /absolute/path/to/replication/results cuda
```

No retraining or new GPU is required. New reports go into a new local `runs/`
folder and a reports-only ZIP in Downloads. Existing artifacts are never overwritten.
The runner preserves failing controls; only invalid inputs, reference changes,
source changes and broken raw-state restoration stop it.
