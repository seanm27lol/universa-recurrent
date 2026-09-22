# universa-recurrent

**Estimate a signal. Keep alternatives alive. State exactly what can be checked.**

Permanent rule: start with a familiar example, explain the intuition, define the
math, show runnable code, ground it in known fields, and state the limits.

## Main findings: phase one complete

**We can record, approximately preserve, and deliberately edit known mathematical
state fields. Predicting what the model will do next is much harder.** The final
locked validation supports limited local prediction, not a general interpreter
for opaque models. No further GPU run is required for this phase.

| Finding | Measured result | What it means |
|---|---|---|
| Recurrence is useful, not an automatic winner | Shared eight-step mean error was 5.90% below one-pass in the five-fit study; four steps nearly matched eight | Keep accuracy, depth and runtime tradeoffs separate |
| Recording and checking can be cheaper | Shared endpoint pipeline: **27.13 -> 14.15 ms** for 256 inputs with batched creation/checking | A measured pipeline improvement, not faster neural reasoning |
| Numerical state descriptions preserve most behavior | Frozen 16-bit codec: **7 changed claim/abstention decisions** in 184,320 repeated comparisons on **2,048 underlying problems** | Approximate preservation, not an exact-decision guarantee |
| Named edits carry the intended intervention | Correct decoded edits: **1.72e-11 effect-disagreement MSE** versus direct edits; wrong-candidate control: **0.00513** | An editable interface to specified fields, not discovered neuron meanings |
| Combined edits interact | Effects of two cycle edits can fail to add up after recurrence | Preserving an intervention is different from predicting its consequences |
| The final predictor has limited usefulness | **69.1% relative RMS error**, versus 100% for predicting no change, on **512 new inputs** | **30.9% less RMS prediction error**; not task accuracy. Limited-usefulness criterion met; close-prediction criterion not met |

Each row has its own experiment and denominator. Repeated fits, cuts and edits
are not independent input problems. Pipeline times are warmed local requests;
models and verifier references are already loaded.

**[Read the important findings and limitations](docs/phase_one_results.md)** ·
[Technical closeout report](docs/phase_one_technical_report.md) ·
[Machine-readable result summary and provenance](experiments/results/phase_one_20260916.json)

```text
Check a recorded calculation     Preserve a state     Apply a named edit
             |                         |                     |
             +-------------------------+---------------------+
                                       |
                    These do NOT automatically explain
                    or predict the remaining computation.
```

The final local-response evaluator still used **73 probes and 1,720 bytes of
coefficients per input**, compared with a 24-byte dynamic state. The positive
prediction result is not a speedup or total-memory saving.

## Research history and reproducibility

Phase Two asked whether a language description of one Qwen2.5-7B-Instruct
activation, reconstructed through the released NLA pair, preserves behavior and
supports targeted text edits. The pinned 128-group pilot completed with decision
**STOP**: description-route preservation failed the frozen 5-point loss bound,
edit coverage was 0/128 so the edit hypothesis is untested, and the projected
validation cost exceeded the eight-hour budget. The phase is closed; the locked
validation was not run. This does not reopen or extend the completed Phase-One
claims. **[Phase-two findings](docs/phase_two_results.md)** ·
[technical closeout report](docs/phase_two_technical_report.md) ·
[implementation README](research/open_weight_lingua/README.md).

The existing experiment scripts remain available to reproduce results; they are
not a new required run list. The [roadmap](docs/roadmap.md) marks this experimental
sequence closed, and the [claim ledger](docs/claims.md) separates measured results
from unimplemented or unestablished capabilities.

| Topic | Evidence / explanation |
|---|---|
| Five independent training/data seeds, one shared evaluation cohort | [Replication audit](docs/replication_20260909.md) |
| Numerical estimate versus optional structural claim | [Dual outputs](docs/dual_outputs.md) |
| Prepared references and full-request measurements | [Setup audit](docs/verifier_setup_20260910.md), [pipeline results](docs/verified_pipeline_20260911.md) |
| Claim reliability and abstention tradeoffs | [Completed 20,000-input study](docs/claim_reliability_20260911.md) |
| Final-state-only inference and receipt costs | [Retention study](docs/final_state_inference.md), [batched checks](docs/batched_receipts.md), [bounded conversion](docs/bounded_receipts.md) |
| Frozen state descriptions and new-input checks | [Continuation](docs/state_continuation.md), [overflow fallback](docs/overflow_continuation.md), [generalization](docs/codec_generalization.md) |
| Known-field edits and predictive limits | [Typed edits](docs/mathematical_edits.md), [composition](docs/edit_composition.md), [local response](docs/local_response.md) |
| Final locked validation | [Protocol](experiments/phase_closeout_v1.json), [completed results](docs/phase_one_technical_report.md) |

[Temporal Lingua](docs/temporal_lingua.md) remains a broader research proposal.
No temporal natural-language decoder has been trained or implemented.

## Earlier in 0.5.0: separate the estimate from the claim

```text
                    candidate estimates + learned weights
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
      weighted numerical estimate          optional structural proposal
      returned on every example             carries its own candidate
               │                                     │
      weighted-sum arithmetic check          candidate-constraint check
```

Changing the proposal threshold does NOT change the weighted estimate or its
computation depth. A mixture across different subspaces generally belongs to
neither, so it never inherits a single candidate's certificate.

The dual-output release introduced a **matched-output experiment**, not a
newly proven superior architecture. It reuses v2 checkpoints and calibrates every structured control
with the same empirical coverage rule on the same calibration data.

## Reproduce the dual-output comparison

In your existing checkout and Python environment:

```bash
git pull --ff-only
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_release.py
.venv/bin/python -m pytest -q

bash scripts/run_dual_study.sh /absolute/path/to/neural_v2.pt
```

No retraining is required. This leaves weights untouched and creates a fresh
`runs/` directory with calibration, evaluation, benchmark and Lingua reports.
The script defaults to CUDA; pass `cpu` as its second argument for CPU operation.

| Report | Question |
|---|---|
| `eval.json` | Which estimator has lower error? How many structural proposals are issued or wrong? |
| `benchmark.json` | What does that EXACT output mode cost on the same examples? |
| `lingua.json` and `verification.json` | Do the weighted sum and separately scoped candidate checks agree? |

[Start with the worked dual-output explanation](docs/dual_outputs.md).
For independent training repetitions and historical v1 comparisons, that page
also documents `python -m universa_recurrent.neural.dual_cli`.

## Install from scratch

```bash
git clone https://github.com/seanm27lol/universa-recurrent.git
cd universa-recurrent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,neural]'
python scripts/check_release.py
python -m pytest -q
```

NumPy is enough for the classical solver. Neural studies require PyTorch >=2.10.
Use only checkpoints you created or otherwise trust. No external account, GPU
service, or private application is required by the public source.

## Earlier paths remain available

```bash
python -m universa_recurrent.cli demo --trace full --output runs/classical.json
python -m universa_recurrent.cli verify runs/classical.json
python -m universa_recurrent.cli --help
```

| Piece | Contribution | Boundary |
|---|---|---|
| HOMYMOLY | Motivation for appropriate structural constraints | Not a universal topology benefit |
| Universa | Explicit spaces and optional pinned adapter | General discovery/transport not integrated |
| Recurrence | Shared learned updates and comparison models | Exploratory accuracy advantages are task-specific, not a universal or matched-compute win |
| Lingua | Checkable records, approximate state descriptions and specified edits | No discovered hidden-neuron semantics, neural replay, or execution authentication |
| Applied CMCM | Motivation to ask which records are worth retaining/checking | Measured receipt engineering is not validation of a general CMCM theory |

Historical source and results stay in their original projects:
[HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY),
[Universa](https://github.com/seanm27lol/Universa),
[Applied CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM).

## Reading map

| Question | Document |
|---|---|
| What did the experiments establish? | [Phase-one findings](docs/phase_one_results.md) |
| What did the Phase Two pilot establish? | [Phase-two findings](docs/phase_two_results.md) |
| Where do I start? | [Beginner walkthrough](docs/start_here.md) |
| Why separate estimates from claims? | [Dual outputs](docs/dual_outputs.md) |
| What does v2 do? | [Multiple hypotheses](docs/neural_v2.md) |
| What was checked in this release? | [0.5.0 audit](docs/release_audit_0.5.0.md) |
| What experiment is proposed? | [Draft protocol](experiments/DUAL_OUTPUT_PROTOCOL_DRAFT.md) |
| What are the earlier mathematical guarantees? | [Mathematics](docs/mathematics.md) and [Lingua](docs/lingua.md) |

## Honest status

This is a small synthetic research prototype with a completed, bounded phase of
experiments. Tests establish software properties; the linked measurements support
only their stated claims. Numerical-output fidelity, structural-claim reliability,
record-checking guarantees and predictive explanations are different results.
Receipt-pipeline improvements are not general neural speedups. Similar parameter
counts do not equalize objectives, computation or inductive bias. General reasoning,
universal interpretability and autonomous discovery/transport between spaces are
not established.

## State-description continuation: implemented and evaluated

Pause the frozen model, encode known cycle coordinates and route evidence,
reconstruct them, and resume the original remaining computation. Raw-state,
equal-payload rotated, erasure, shuffled-state and stop-at-cut controls keep
compression separate from semantic claims. No training or hidden-neuron explanation
is claimed. [Protocol and run command](docs/state_continuation.md);
[measured preservation and remaining failures](docs/phase_one_results.md#3-approximate-state-preservation-is-not-exact-decision-preservation).
