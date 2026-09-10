# universa-recurrent

**Estimate a signal. Keep alternatives alive. State exactly what can be checked.**

Permanent rule: start with a familiar example, explain the intuition, define the
math, show runnable code, ground it in known fields, and state the limits.

## Research update: replication and temporal Lingua

The [five-training-seed audit](docs/replication_20260909.md) reports the accuracy,
timing and limitations of the supplied DGX experiments. Recompute the report
checks with `python scripts/audit_replication.py reports.zip`.

[Temporal Lingua](docs/temporal_lingua.md) proposes describing short windows of
states, with a [separate draft protocol](experiments/TEMPORAL_LINGUA_PROTOCOL_DRAFT.md).
It distinguishes NLA reconstruction, J-space readouts and causal evidence.
That research-plan update left the model and training recipe unchanged.
No temporal language decoder has been trained or implemented.

## New in 0.5.2: final-state-only inference and measured audit costs

Keep the current state for ordinary inference; retain the numerical trajectory
only when requested. Same frozen weights and depth, with paired output checks.
The experiment separately measures inference, memory, record generation and checks.
[Read the scope and run instructions](docs/final_state_inference.md).

```bash
git pull --ff-only
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_release.py
.venv/bin/python -m pytest -q
bash scripts/run_retention_study.sh /path/to/replication/results cuda
```

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

This release adds a **matched-output experiment**, not a newly proven superior
architecture. It reuses v2 checkpoints and calibrates every structured control
with the same empirical coverage rule on the same calibration data.

## Run the new comparison

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
| Recurrence | Shared learned updates and comparison models | Not established as better than direct or untied models |
| Lingua | Typed arithmetic and constraint records | No hidden-neuron semantics, neural replay, or execution authentication |
| Applied CMCM | Ask which records are worth retaining/checking | A faster or more useful witness is still an experiment |

Historical source and results stay in their original projects:
[HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY),
[Universa](https://github.com/seanm27lol/Universa),
[Applied CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM).

## Reading map

| Question | Document |
|---|---|
| Where do I start? | [Beginner walkthrough](docs/start_here.md) |
| Why separate estimates from claims? | [Dual outputs](docs/dual_outputs.md) |
| What does v2 do? | [Multiple hypotheses](docs/neural_v2.md) |
| What was checked in this release? | [0.5.0 audit](docs/release_audit_0.5.0.md) |
| What experiment is proposed? | [Draft protocol](experiments/DUAL_OUTPUT_PROTOCOL_DRAFT.md) |
| What are the earlier mathematical guarantees? | [Mathematics](docs/mathematics.md) and [Lingua](docs/lingua.md) |

## Honest status

This is a small synthetic research scaffold. Tests establish software properties,
not scientific success. The new study deliberately keeps inference depth fixed
to isolate output policy; adaptive stopping requires separate evidence. Similar
parameter counts do not equalize objectives, computation, or inductive bias.
No GPU speedup, general reasoning ability, or universal interpretability is claimed.
