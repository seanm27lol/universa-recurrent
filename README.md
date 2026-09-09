# universa-recurrent

**Choose a mathematical space. Reuse a latent state. Keep a record that says exactly what can be checked.**

> Permanent design rule: begin with a familiar example, define the mathematics,
> show runnable code, connect it to established fields, and state the limits.

**Status: transparent classical baseline + exploratory learned recurrence.**
Nothing here is yet a sealed scientific result.

## The idea in one picture

```text
partial noisy measurements
          │
          ▼
  learned structure router
          │
          ▼
 choose coordinates z = Q a
          │
          ▼
 shared update Fθ ───────────────┐
          │                      │
          ├─ measure progress    │
          └─ stop or continue ───┘
          │
          ▼
 estimate + Lingua record
          │
          ▼
 independent property checker
```

The first example is a circulation on a small graph. Incoming and outgoing flow
must balance at each junction. This resembles Kirchhoff's current law, but it is
not a complete circuit or hydraulic model.

## What each layer does

| Layer | Implemented now | Not yet established |
|---|---|---|
| **HOMYMOLY connection** | Motivates testing exact structural restrictions | A universal benefit from topology |
| **Universa connection** | Candidate subspaces, cached coordinates, optional pinned adapter | Learned discovery or recurrent switching between arbitrary structures |
| **Recurrence** | One learned update reused over latent-coordinate steps | A general reasoning system |
| **Adaptive execution** | Dense and active-sample-compacted inference paths | A speedup merely because mean logical steps fall |
| **Lingua** | Typed records of routes, measured dynamics, and structural checks | Semantic decoding of hidden features or execution attestation |
| **Applied CMCM connection** | Full versus compact records and property-specific verification | A theorem that more witness information always helps |

The earlier repositories remain separate and keep their own evidence histories:
[HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY),
[Universa](https://github.com/seanm27lol/Universa), and
[Applied Experiments of CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM).

## Install

```bash
git clone https://github.com/seanm27lol/universa-recurrent.git
cd universa-recurrent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,neural]"
python -m pytest -q
```

NumPy is enough for the classical path. The learned path requires PyTorch 2.10
or newer; `--device auto` selects CUDA when available. Load only checkpoints you
created or otherwise trust.

## 1. Transparent classical example

```bash
python -m universa_recurrent.cli demo \
  --trace full \
  --output runs/classical.json

python -m universa_recurrent.cli verify runs/classical.json
```

The direct solver, repeated update, Lingua record, and independent checker are all
small enough to inspect line by line.

## 2. Train learned recurrence

```bash
python -m universa_recurrent.cli neural-train \
  --device cuda \
  --train-size 20000 \
  --val-size 4000 \
  --epochs 20 \
  --batch-size 512 \
  --hidden-dim 64 \
  --steps 8 \
  --output checkpoints/neural_v1.pt
```

Neural v1 learns:

1. a two-way structure route;
2. a shared coordinate update reused at every step;
3. a readiness signal trained from a synthetic future-regret teacher.

The teacher can see synthetic truth during training. Inference cannot.

## 3. Evaluate against the references we almost missed

```bash
python -m universa_recurrent.cli neural-eval \
  --device cuda \
  --checkpoint checkpoints/neural_v1.pt \
  --n 4000 \
  --output runs/neural_v1_eval.json
```

The report includes:

```text
adaptive thresholds
fixed depths 1 / 2 / 4 / 8
non-learned fit-each-structure baseline
generator-aware Gaussian Bayes reference
route-correct versus route-wrong reconstruction error
logical updates versus examples actually sent through the update network
```

The Gaussian reference uses privileged knowledge of the toy generator. Its soft
posterior mixture is the squared-error Bayes reference under those assumptions;
its hard MAP route is only a diagnostic. Neither is a deployable method.
The two candidate spaces share one direction, and partial noisy measurements can
be ambiguous, so perfect route recovery is not assumed. Recurrence itself also
needs tied/untied and direct feed-forward ablations before receiving causal
credit for any gain.

## 4. Measure real execution, not just “mean steps”

```bash
python -m universa_recurrent.cli neural-benchmark \
  --device cuda \
  --checkpoint checkpoints/neural_v1.pt \
  --n 65536 \
  --batch-size 4096 \
  --halt-threshold 0.50 \
  --output runs/neural_v1_benchmark.json
```

```text
logical halting  ≠  skipped GPU work  ≠  lower wall-clock latency
```

The dense adaptive path freezes halted states but still evaluates the whole batch.
The compact path gathers only active samples. Dynamic indexing may still cost more
than it saves, so the benchmark reports time and quality together. It also times
the transparent fit-each-structure solver; the neural model does not receive a
free pass merely because it is the focus of the repository.

## 5. Emit and check a neural Lingua record

```bash
python -m universa_recurrent.cli neural-demo \
  --device cuda \
  --checkpoint checkpoints/neural_v1.pt \
  --seed 9001 \
  --output runs/neural_v1_lingua.json

python -m universa_recurrent.cli neural-verify \
  runs/neural_v1_lingua.json \
  --checkpoint checkpoints/neural_v1.pt
```

The checker can independently recompute the final constraint residual, final
observed-coordinate residual, and event consistency. With `--checkpoint`, it also
binds the embedded structure and configuration to those exact checkpoint bytes.
It **does not replay the network**, explain the router causally, or prove that a
remote machine executed the recorded path.

## Where to read next

| Question | Document |
|---|---|
| What should I understand first? | [Start here](docs/start_here.md) |
| How does the neural loop work? | [Neural recurrence](docs/neural_recurrence.md) |
| What did the code audit find? | [Development audit](docs/development_audit.md) |
| What does Lingua establish? | [Lingua](docs/lingua.md) |
| Which claims are allowed? | [Claims ledger](docs/claims.md) |
| How do the projects connect? | [Architecture](docs/architecture.md) |
| What real fields resemble these ideas? | [Known examples and sources](docs/real_world_connections.md) |

## Repository map

```text
src/universa_recurrent/
  structures/       explicit constraint spaces and cached bases
  recurrence/       transparent classical recurrence and direct baseline
  neural/            data, model, baselines, training, benchmarking, Lingua
  lingua/           classical record schema
  verification/     classical independent checks
experiments/        exploratory protocols and benchmark runners
tests/              success, failure, tampering, and compatibility tests
docs/               explanations, claim boundaries, and known precedents
```

**Founding question:** Can a structure-aware recurrent solver reach a target
quality with less total work while retaining enough evidence for specified
checks?

Efficiency and checkability are separate hypotheses. Either may fail.
