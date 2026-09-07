# universa-recurrent

**Compute in a suitable mathematical space. Reuse the state. Keep a checkable record.**

> Start with a familiar example, then show the mathematics, then the code.
> Anyone curious should be able to find a way in.

**Status: runnable research foundation, not a trained model.** The first release
contains a classical recurrent solver, a non-learned structure selector, typed
mathematical records, and independent numerical witness checks. Learned routing,
neural recurrence, general structure switching, and causal interpretability are
research plans, not completed features. No inference speedup is claimed.

## See the idea before the terminology

Imagine sensors on a closed flow network. Some readings are missing; others are
noisy. At every junction, incoming flow should equal outgoing flow. Can we repair
the estimate while keeping a short record of the checks we performed?

```text
Noisy measurements
        ↓
Choose a candidate conservation rule
        ↓
Fit the measurements → enforce the rule → check progress
        ↑                                      │
        └────────── another step, if needed ────┘
                                               ↓
                                  Estimate + numerical witness
```

This conservation example is related to Kirchhoff's current law in circuits.
It is **not** a full electrical or hydraulic simulator: we do not model voltage,
resistance, pressure, capacity, or time-dependent storage. Sources and sinks need
an affine constraint such as Bz=b rather than the homogeneous Bz=0 used here.
See [real examples and sources](docs/real_world_connections.md).

## Start in the place that suits you

| You want to… | Open… |
|---|---|
| Understand the idea without prerequisites | [Start here](docs/start_here.md) |
| Try a working example | The quickstart below |
| See how the research pieces connect | [Architecture](docs/architecture.md) |
| Understand what a witness actually establishes | [Lingua and witnesses](docs/lingua.md) |
| Read equations and their assumptions | [The mathematics](docs/mathematics.md) |
| Know what is demonstrated versus planned | [Claims and limits](docs/claims.md) |
| Extend the system without adding mystery | [Contributing](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) |

## Quickstart

Python 3.11+ and NumPy are sufficient; no GPU, model download, or credentials.
Run these commands **inside the repository folder**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m universa_recurrent.cli demo --trace full --output runs/first.json
python -m universa_recurrent.cli verify runs/first.json
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate` instead. Existing output files
are not overwritten. Choose a new filename for each run.

The demo reduces five edge values to two independent cycle coordinates, selects
between two candidate constraints using validation measurements, and compares the
iterative answer to a direct solve. It does **not** prove the iterative path is
faster. [Read a retained example run](docs/example_run.txt), then inspect its
[full mathematical record](examples/traces/flow_full.json).

## What exists today?

| Piece | Implemented | Not claimed |
|---|---|---|
| Structural operators | Cached nullspace coordinates; optional pinned Universa adapter | Every data domain is equivalent |
| Selection | Solve all candidates, score validation fit, refuse ties | Cheap learned routing or discovery |
| Recurrence | State-reusing projected gradient descent; fixed budget or tolerance stop | Trained neural recurrence or faster inference |
| Lingua v0 | Machine-readable names, operations, assumptions, and witnesses | A translation of arbitrary hidden thoughts |
| Verification | Projection and final optimality checks; full-trace update checks | Exact floating-point proof or execution attestation |
| Witness comparison | Full versus compact records, with different declared guarantees | Compression preserving every aspect of a computation |

## How the earlier projects fit

| Project | Connection | Boundary |
|---|---|---|
| [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY) | Motivation: appropriate structural information helped a tested lifting task. | Its lifting is not automatically a Universa chain map; its experiments are not this project's results. |
| [Universa](https://github.com/seanm27lol/Universa) | Structural foundation; optional adapter pinned to one public commit. | The demo selector is not the upstream learned router. |
| [Applied CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM) | Study which records are worth retaining and what information they preserve. | More trace information does not universally improve learning or checking. |

Existing repositories and evidence remain separate. No private application code,
weights, datasets, or results are dependencies of this public project.

## Repository map

```text
src/universa_recurrent/
  structures/       valid spaces and a pinned upstream adapter
  recurrence/       classical updates and a direct-solve baseline
  lingua/           the trace format and plain-language rendering
  verification/     independent matrix-vector checks
  routing.py        transparent candidate-selection baseline
examples/           small runnable entry points
experiments/        benchmark runner and an explicitly unsealed protocol draft
tests/              correctness, malformed inputs, and tampered witnesses
docs/               explanations, sources, assumptions, and roadmap
```

**Founding question:** Can a structure-aware solver reach a target quality with
less total work while retaining enough evidence for specified checks?
Efficiency and checkability are separate hypotheses. [Build roadmap](docs/roadmap.md).

Apache-2.0 for this original scaffold. See [NOTICE](NOTICE) for attribution and
[dependency provenance](docs/upstream.md) for the optional upstream project.

## Publishing the prepared archive

If you received this as an archive, extract it and run `bash scripts/publish.sh`
from this folder after authenticating GitHub CLI. [Publishing guide](docs/publishing.md).
This instruction is not a claim that a remote already exists.
