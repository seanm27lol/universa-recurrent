# universa-recurrent

**Estimate a signal. Keep alternatives alive. State exactly what can be checked.**

This is a small research prototype built around one question: if you already know
a rule your answer has to obey, can you estimate the answer *inside* that rule, and
hand back a record that someone else can check without trusting you?

It starts with a five-pipe example that runs in about a minute on a laptop. From
there it goes to learned recurrent models (phase one) and to descriptions of one
real language model's internal activations (phase two). Both phases are closed,
and their negative results are kept on purpose.

## A concrete example: five pipes, three sensors

```text
          e0
      0 ─────→ 1
      ↑ ╲      │
   e3 │  ╲ e4  │ e1
      │   ↘    ↓
      3 ←───── 2
          e2
```

Flow circulates around a square with one diagonal pipe. At every junction, what
flows in must flow out. You have noisy sensors on three pipes (`e0`, `e2`, `e4`)
and two more readings (`e1`, `e3`) held back for choosing between rules. What is
the flow in all five?

Two things make this more than curve fitting:

- **The rule shrinks the problem.** Conservation rules out most combinations of
  five numbers. Every balanced flow on this graph is described by just **two**
  cycle coordinates, so the solver works with two numbers instead of five.
- **You might not know which rule is right.** The demo gets two candidate rules:
  real conservation, and a deliberately wrong one with the diagonal reversed. It
  has to pick using the held-back readings. The hidden true flow is used only for
  the final error report, never for choosing or solving.

This is conservation alone, not a pipe simulator: no pressure, pumps or capacity.
[The analogy and where it stops](docs/real_world_connections.md).

## Quickstart

NumPy is the only runtime dependency for this part. No GPU or account is needed.

```bash
git clone https://github.com/seanm27lol/universa-recurrent.git
cd universa-recurrent
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[test]'
python -m universa_recurrent.cli demo --trace full --output runs/classical.json
```

```text
A small circulation: estimate flow without creating or destroying it at junctions.
Synthetic measurements; classical baseline; no speedup claim.

Candidate balanced_flow                validation MSE 0.001612
Candidate wrong_diagonal_constraint    validation MSE 0.138974

Structure: balanced_flow
Operation: repeated measurement fitting constrained to balanced states.
Iterations: 35; stop: stationarity_tolerance.
Trace: full; final numerical witness retained.
This explains explicit operations, not why a learned model would choose them.
Coordinates: 5 edge values -> 2 cycle coordinates
Reconstruction: [0.96953 0.96953 1.34293 1.34293 0.3734 ]
Distance from direct-solve baseline: 2.778e-09
Distance from hidden synthetic truth (evaluation only): 9.529e-02
Witness: PASS (full numerical update chain)
Saved runs/classical.json
```

| Output line | What it tells you |
|---|---|
| `validation MSE 0.001612` vs `0.138974` | The held-back readings favor real conservation by a wide margin, so it is chosen. |
| `5 edge values -> 2 cycle coordinates` | The working state is two numbers, computed from the rule, not learned. |
| `Iterations: 35; stop: stationarity_tolerance` | One small update was repeated 35 times and stopped because a declared test passed, not because the budget ran out. |
| `Distance from direct-solve baseline: 2.778e-09` | It agrees with solving the same small equation directly. The iterative path is not claimed to be faster. |
| `Distance from hidden synthetic truth ...: 9.529e-02` | The estimate is not the truth: the sensors are noisy and a mild penalty is added. |
| `Witness: PASS (full numerical update chain)` | A separate checker confirmed every recorded step with matrix-vector products, without rerunning the solver. |

Now check the saved record yourself, the way a stranger would:

```bash
python -m universa_recurrent.cli verify runs/classical.json
python -m pytest -q    # optional: the test suite
```

A PASS means the recorded numbers satisfy specific equations. It does not mean the
sensors were honest, that conservation was the right model of the world, or that
some other machine actually ran this computation.
[The same example, one idea at a time](docs/start_here.md).

## The idea in plain language

Three ingredients, each an established idea:

1. **Structure.** Work in coordinates that can only express valid answers. Any
   pair of cycle coordinates you write down is automatically a balanced flow.
2. **Recurrence.** Start with a guess, nudge it toward the measurements, keep it
   valid, and repeat the same update. Stop when a declared test passes or the
   step budget runs out. Here that is ordinary projected gradient descent.
3. **Receipts** (called *Lingua* in this repo). Record what was done, plus a small
   extra vector (a *witness*) that makes checking much cheaper than solving. A
   route through a maze works the same way: you can check it without searching.

The neural versions keep the same visible graph and candidate rules but learn the
update. Version 2 keeps one working state per candidate rule, revises how much it
trusts each one, and then either commits to a rule or abstains.
[How v2 works](docs/neural_v2.md) · [why the estimate and the claim are separate outputs](docs/dual_outputs.md).

## The mathematics on one screen

| Symbol | Meaning | In the example |
|---|---|---|
| z | Flow on each pipe | 5 numbers |
| B | Junction rule: entry i of Bz is the net flow into junction i | 4 × 5 |
| A | Which pipes are measured, with each sensor's gain | 3 × 5 |
| y | Sensor readings | 3 numbers |
| ρ | Small positive penalty weight (ridge) | 0.01 |
| Q | Orthonormal basis for all balanced flows (the null space of B) | 5 × 2 |
| a | Cycle coordinates, with z = Qa | 2 numbers |

The solver minimizes a fit-plus-penalty loss over balanced flows only:

$$
\min_{Bz=0}\ f(z),\qquad f(z)=\tfrac12\|Az-y\|_2^2+\tfrac\rho2\|z\|_2^2 .
$$

Writing z = Qa removes the constraint. With H = (AQ)ᵀ(AQ) + ρI and b = (AQ)ᵀy,
the recurrent update is

$$
a \leftarrow a-\eta\,(Ha-b),\qquad \eta = 1/\lambda_{\max}(H),
$$

repeated until ‖Ha − b‖ ≤ 10⁻⁹ or 256 steps. The receipt stores z and a
multiplier λ. The checker confirms Bz ≈ 0 and ∇f(z) + Bᵀλ ≈ 0: feasibility and
stationarity, the standard optimality conditions for this convex problem.

**Assumptions:** B is known and correct, the circulation is closed (no sources or
sinks), and ρ > 0. Checks use floating-point tolerances; they are not exact proofs.
[Full derivation, symbols and proofs](docs/mathematics.md).

| To read the code for... | Open |
|---|---|
| Building Q from B once, with no repeated SVD | [structures/core.py](src/universa_recurrent/structures/core.py) |
| The recurrent solver and the direct baseline | [recurrence/solver.py](src/universa_recurrent/recurrence/solver.py) |
| The independent checker (NumPy only; never calls the solver) | [verification/checks.py](src/universa_recurrent/verification/checks.py) |
| Choosing among candidate rules | [routing.py](src/universa_recurrent/routing.py) |
| The example itself | [examples.py](src/universa_recurrent/examples.py) |

## Where these ideas come from

| Idea here | Known field | Primary source |
|---|---|---|
| Conservation at junctions | Kirchhoff's current law | [OpenStax, *University Physics 2*, §10.3](https://openstax.org/books/university-physics-volume-2/pages/10-3-kirchhoffs-rules) |
| Repeated correction; optimality certificate | Convex optimization | [Boyd and Vandenberghe (2004)](https://web.stanford.edu/~boyd/cvxbook/) |
| Learned iterations | Deep unfolding | [Hershey, Le Roux and Weninger (2014)](https://arxiv.org/abs/1409.2574) |
| Commit or abstain | Classification with a reject option | [Chow (1970)](https://doi.org/10.1109/TIT.1970.1054406) |
| Several hypotheses at once | Bayesian model averaging | [Hoeting et al. (1999)](https://doi.org/10.1214/ss/1009212519) |

Nothing here claims novelty in conservation laws, null spaces, projected gradient
descent or optimality conditions.
[Full table, including where each analogy stops](docs/real_world_connections.md).

This repository also brings together questions from earlier projects without
importing their conclusions wholesale:

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

## What the experiments found

### Phase one: learned recurrence on synthetic flow problems (complete, 2026-09-16)

The neural versions were trained and tested on synthetic versions of the example
above: each flow obeys one of the same two candidate rules, and the model sees only
a noisy, partial view of it. The studies asked four separate questions:

```text
RECORD                   PRESERVE                 EDIT                   PREDICT
Do the listed            Can we restore           Can we change          Can we forecast
calculations check?      useful working values?   the intended field?    the consequences?
       |                        |                        |                       |
       +------------------------+------------------------+-----------------------+
                                None alone proves
                      that the model's answer is correct.
```

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
models and verifier references are already loaded. The final local-response
evaluator still used **73 probes and 1,720 bytes of coefficients per input**,
compared with a 24-byte dynamic state. The positive prediction result is not a
speedup or total-memory saving.

**[Phase-one findings and limitations](docs/phase_one_results.md)** ·
[technical closeout report](docs/phase_one_technical_report.md) ·
[machine-readable summary and provenance](experiments/results/phase_one_20260916.json)

### Phase two: describing a real model's activation in words (closed)

Give Qwen2.5-7B-Instruct a tiny program (`x = 3`, `y = 8`, `x = x + 2`, "What is
x?"). Capture one internal vector before it answers, describe it in English with a
released activation verbalizer (AV), rebuild the vector's direction from that
English with the paired reconstructor (AR), and swap the rebuilt vector back in. Does the model
still behave the same, and can editing the text make a targeted change?

Phase Two asked whether a language description of one Qwen2.5-7B-Instruct
activation, reconstructed through the released NLA pair, preserves behavior and
supports targeted text edits. The pinned 128-group pilot completed with decision
**STOP**: description-route preservation failed the frozen 5-point loss bound,
edit coverage was 0/128 so the edit hypothesis is untested, and the projected
validation cost exceeded the eight-hour budget. The phase is closed; the locked
validation was not run. This does not reopen or extend the completed Phase-One
claims.

Two follow-up measurements were each run once and closed, both negative:

- **Steering assay (2026-09-22).** Can the *difference* between two descriptions
  steer the model toward a targeted answer? In one frozen run, all three arms
  failed every frozen pre-run criterion. Intended deltas flipped the target answer
  in 2–8% of groups while disturbing the unaffected answer in 43–64%, no better
  than wrong-variable controls. This covers one checkpoint, one task family and
  one site; it does not refute NLA steering in general.
  [Ledger entry](docs/claims.md#post-phase-two-steering-assay-2026-09-22).
- **vLLM backend (2026-09-22).** An opt-in vLLM path for the AV and AR stages
  failed its measured equivalence gate against the default eager path: 0/32 AV
  greedy continuations were token-identical, and AR direction cosine fell as low
  as 0.8097. It stays non-default, and its outputs are not mixed into eager-path
  evidence. [Ledger entry](docs/claims.md#phase-two-vllm-backend-2026-09-22).

**[Phase-two findings](docs/phase_two_results.md)** ·
[technical closeout report](docs/phase_two_technical_report.md) ·
[implementation README](research/open_weight_lingua/README.md)

## Limits

This is a small research prototype with two completed, bounded phases of
experiments. Tests establish software properties; the linked measurements support
only their stated claims. Numerical-output fidelity, structural-claim reliability,
record-checking guarantees and predictive explanations are different results.
Receipt-pipeline improvements are not general neural speedups. Similar parameter
counts do not equalize objectives, computation or inductive bias. General reasoning,
universal interpretability and autonomous discovery/transport between spaces are
not established.

A trace schema, a property certificate, execution provenance, causal faithfulness
and real-world task accuracy answer different questions, and credit does not
transfer between them.
Compact records do not check discarded intermediate history. Hashes identify
artifacts but do not authenticate a remote execution.
[Temporal Lingua](docs/temporal_lingua.md) remains a broader research proposal; no
temporal natural-language decoder has been trained or implemented. The
[claim ledger](docs/claims.md) lists every claim with its status, including the
ones that failed.

## Reading map

| I want to... | Read |
|---|---|
| Walk through the example slowly | [Beginner walkthrough](docs/start_here.md) |
| See the full mathematics and guarantees | [Mathematics](docs/mathematics.md), [Lingua records](docs/lingua.md) |
| Understand the neural versions | [v1 recurrence](docs/neural_recurrence.md), [v2 multiple hypotheses](docs/neural_v2.md), [dual outputs](docs/dual_outputs.md), [architecture](docs/architecture.md) |
| Know what phase one established | [Findings](docs/phase_one_results.md), [technical report](docs/phase_one_technical_report.md) |
| Know what phase two established | [Findings](docs/phase_two_results.md), [technical report](docs/phase_two_technical_report.md), [implementation](research/open_weight_lingua/README.md) |
| See every claim and its status | [Claim ledger](docs/claims.md) |
| See what is closed and what is not planned | [Roadmap](docs/roadmap.md) |
| See what a release checked | [0.5.0 audit](docs/release_audit_0.5.0.md) |
| See a proposed experiment | [Dual-output draft protocol](experiments/DUAL_OUTPUT_PROTOCOL_DRAFT.md) |

**Phase-one evidence by topic**

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

## Reproducing the studies

The experiment scripts remain available to reproduce results; they are not a new
required run list. Many studies record hashes of the scripts that produced them,
so those files are kept as they were run.

The neural studies need PyTorch >=2.10. Use only checkpoints you created or
otherwise trust.

```bash
python -m pip install -e '.[test,neural]'
python scripts/check_release.py
python -m pytest -q
```

The dual-output comparison reuses existing v2 checkpoints with no retraining. It is
a **matched-output experiment**, not a newly proven superior architecture: every
structured control is calibrated with the same empirical coverage rule on the same
calibration data.

```bash
bash scripts/run_dual_study.sh /absolute/path/to/neural_v2.pt   # add `cpu` as a second argument for CPU
```

It writes a fresh `runs/` directory and leaves the weights untouched.

| Report | Question |
|---|---|
| `eval.json` | Which estimator has lower error? How many structural proposals are issued or wrong? |
| `benchmark.json` | What does that EXACT output mode cost on the same examples? |
| `lingua.json` and `verification.json` | Do the weighted sum and separately scoped candidate checks agree? |

Details, independent training repetitions and historical v1 comparisons are in
[dual outputs](docs/dual_outputs.md). Phase two has its own pinned environment and
needs a GPU; see its [implementation README](research/open_weight_lingua/README.md).

## Contributing

Start with [AGENTS.md](AGENTS.md). The project's permanent rule is to start with a
familiar example, explain the intuition, define the math, show runnable code,
ground it in known fields, and state the limits. See also
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

Licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
