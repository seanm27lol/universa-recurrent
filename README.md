# universa-recurrent

**GitHub-native release 0.4.1.** Use this checkout, not the retired overlay installers.
See [the release audit](docs/release_audit_0.4.1.md) for the fixes and tested scope.

**Keep several mathematical explanations alive, refine them, then commit only when the evidence is strong enough.**

> Permanent rule: begin with a familiar example, define the mathematics, show runnable code, connect it to established fields, and state the limits.

**Status:** transparent classical baseline, historical neural v1, and exploratory neural v2. Nothing here is yet a sealed scientific result.

## The idea in one picture

```text
partial noisy measurements
          │
          ▼
  candidate structures Q₁, Q₂, ...
          │
     ┌────┴────┐
     ▼         ▼
 state in Q₁  state in Q₂       keep every plausible hypothesis
     │         │
     └────┬────┘
          ▼
 shared recurrent update
          │
 revise route evidence after every step
          │
   ┌──────┼────────┐
   ▼      ▼        ▼
continue  commit   abstain
                    │
                    └─ return a provisional mixture, not a single-structure claim
          │
          ▼
 estimate + Lingua record + independent checker
```

The first example is a five-edge circulation. Some measurements are hidden or noisy, and balanced flow must not appear or disappear at a junction. This resembles Kirchhoff's current law, but it is not a complete circuit or hydraulic model.

## What changed in neural v2

Neural v1 selected one structure before recurrence. Its first DGX experiment showed that wrong early routes dominated error, while adaptive stopping mainly limited damage after those mistakes. Neural v2 therefore:

1. carries one recurrent state per candidate structure;
2. revises route probabilities after each update;
3. makes an explicit **continue / commit / abstain** decision;
4. calibrates that decision on data separate from the final test set;
5. records candidate trajectories and rejected alternatives in Lingua;
6. compares against direct, untied-depth, ambient recurrent, dedicated fixed-depth, transparent, and privileged reference methods.

This is not yet Universa's full structure-switching vision: v2 does not transport one persistent state between different structures or invoke structure discovery.

## Install

```bash
git clone https://github.com/seanm27lol/universa-recurrent.git
cd universa-recurrent
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e ".[test,neural]"
python -m pytest -q
```

`--system-site-packages` is convenient on an NVIDIA DGX Spark that already has a working CUDA PyTorch installation. A normal isolated environment also works.

## Run the readable classical example

```bash
python -m universa_recurrent.cli demo \
  --trace full \
  --output runs/classical.json

python -m universa_recurrent.cli verify runs/classical.json
```

## Train neural v2

```bash
python -m universa_recurrent.cli neural-v2-train \
  --device cuda \
  --train-size 20000 \
  --calibration-size 4000 \
  --epochs 20 \
  --batch-size 512 \
  --hidden-dim 64 \
  --candidate-embedding-dim 8 \
  --steps 8 \
  --output checkpoints/neural_v2.pt
```

The default run trains the main model and the declared controls. Use `--no-controls` only for a plumbing smoke test, not a comparative experiment.

## Evaluate, benchmark, and inspect Lingua

```bash
python -m universa_recurrent.cli neural-v2-eval \
  --device cuda \
  --checkpoint checkpoints/neural_v2.pt \
  --n 4000 \
  --output runs/neural_v2_eval.json

python -m universa_recurrent.cli neural-v2-benchmark \
  --device cuda \
  --checkpoint checkpoints/neural_v2.pt \
  --n 65536 \
  --batch-size 4096 \
  --output runs/neural_v2_benchmark.json

python -m universa_recurrent.cli neural-v2-demo \
  --device cuda \
  --checkpoint checkpoints/neural_v2.pt \
  --output runs/neural_v2_lingua.json

python -m universa_recurrent.cli neural-v2-verify \
  runs/neural_v2_lingua.json \
  --checkpoint checkpoints/neural_v2.pt
```

## Read results correctly

```text
fewer logical steps
      ≠ fewer candidate updates actually executed
      ≠ lower wall-clock latency
      ≠ better reconstruction
```

| Layer | Implemented now | Not established |
|---|---|---|
| **Structures** | Explicit candidate subspaces and cached bases | Correct structure for every real task |
| **Recurrence** | Shared update over every candidate state | A universal benefit from recurrence |
| **Routing** | Evidence revision at every step | Perfectly calibrated probabilities |
| **Selective decision** | Calibrated commit or abstain rule | One policy optimal for all costs |
| **Lingua** | Candidate states, probabilities, residuals, decisions, rejected alternatives | Human meanings for arbitrary hidden features |
| **Checker** | Arithmetic, structural, policy, and checkpoint-binding checks | Network replay or remote execution attestation |
| **Controls** | Direct, untied, ambient, dedicated fixed depth, transparent, Gaussian reference | A confirmatory causal conclusion from one exploratory run |

## Where the earlier projects fit

| Project | Real connection | Boundary |
|---|---|---|
| [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY) | Motivates testing computation restricted to an appropriate structure. | Its lifting result does not prove this neural architecture helps. |
| [Universa](https://github.com/seanm27lol/Universa) | Supplies the broader route, transport, project, and discover program. | This repo currently keeps parallel subspace hypotheses; it does not yet perform general typed transport or discovery. |
| [Applied CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM) | Motivates measuring which computational records are worth retaining. | More witness information is not assumed to improve learning or checking. |

No private RF-MoE code, weights, data, or results are included.

## Read next

| Question | Document |
|---|---|
| How does v2 work? | [Neural v2](docs/neural_v2.md) |
| What does Lingua actually verify? | [Lingua](docs/lingua.md) |
| Which claims are permitted? | [Claims ledger](docs/claims.md) |
| How do the pieces connect? | [Architecture](docs/architecture.md) |
| What is still missing? | [Roadmap](docs/roadmap.md) |
| What was the classical foundation? | [Start here](docs/start_here.md) |

**Founding question:** Can a structure-aware recurrent solver reach a target quality with less total work while retaining enough evidence for specified checks?

Efficiency, accuracy, calibrated refusal, and interpretability are separate hypotheses. Any of them may fail.
