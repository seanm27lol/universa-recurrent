# Neural v2: keep alternatives alive before committing

## Familiar example

A mechanic hears a vibration that could come from either the wheel or the engine. Committing immediately to one diagnosis can make every later observation look like evidence for that choice. A better process keeps both hypotheses alive, updates each against new evidence, and either commits or says the evidence is still insufficient.

The toy flow task uses the same idea with two mathematical constraint spaces.

```text
measurements
   ├─ hypothesis 1: balanced-flow state
   └─ hypothesis 2: alternate constrained state
                 │
        update both candidates
                 │
        revise their probabilities
                 │
        continue / commit / abstain
```

## Mathematics

Let `Q_k ∈ R^(N×D)` be an orthonormal basis for candidate structure `k`.
At recurrent step `t`, the model carries coordinates `a_(t,k) ∈ R^D` and decodes

```math
z_(t,k) = Q_k a_(t,k).
```

Thus every candidate state lies in its declared subspace, up to floating-point error. For an observation `x` and binary mask `m`, the observed residual is

```math
r_(t,k) = m ⊙ (z_(t,k) - x).
```

One shared learned map updates every candidate:

```math
a_(t+1,k) = a_(t,k) + F_θ(c, k, a_(t,k), Q_k^T r_(t,k), ||r_(t,k)||, p_(t,k)),
```

where `c` is the encoded observation and `p_(t,k)` is the current route probability. A second shared map scores evidence from the updated state and residual, producing the next probability distribution across candidates.

The structural membership of each `z_(t,k)` follows from its parameterization. The probability that candidate `k` is correct does not.

## Decision policy

After each step, calculate the largest probability `p_max` and the margin between the two largest probabilities.

```text
before the final step:
  commit when p_max and the margin exceed calibrated thresholds
  otherwise continue

at the final step:
  commit under weaker calibrated thresholds
  otherwise abstain
```

An abstained output is the probability-weighted mixture of the candidate states. It is useful as a provisional estimate, but it does not carry a claim that one structure is correct.

The thresholds are selected on a calibration split using a declared objective containing reconstruction error, step cost, abstention cost, and wrong-commit cost. The final evaluation uses a different seed.

## Why this is grounded in known fields

- **Algorithm unrolling:** repeated algorithmic updates are represented by trainable layers. Gregor and LeCun's LISTA is a canonical example: <https://icml.cc/Conferences/2010/papers/449.pdf>.
- **Adaptive computation:** different examples may receive different recurrent depth. Graves introduced Adaptive Computation Time for recurrent networks: <https://arxiv.org/abs/1603.08983>.
- **Classification with rejection:** a classifier may decline a decision when error is more costly than rejection. Chow's classical error-reject analysis is <https://doi.org/10.1109/TIT.1970.1054406>.
- **Model uncertainty:** averaging predictions across plausible models rather than pretending one selected model is certain is a standard statistical idea; see Hoeting et al., *Bayesian Model Averaging: A Tutorial*: <https://doi.org/10.1214/ss/1009212519>.

These are precedents, not evidence that this implementation succeeds.

## Controls included

| Control | Question |
|---|---|
| Direct structured network | Does recurrence add anything beyond a one-pass predictor? |
| Untied-depth network | Does sharing one update across depth matter? |
| Ambient recurrent network | Does explicit structural parameterization help? |
| Dedicated depth 1/2/4 | Does a model trained for a short depth beat truncating an 8-step model? |
| Fit-each-structure solver | Does a transparent non-learned rule already solve routing? |
| Gaussian hard/mixture reference | What is achievable with privileged knowledge of the toy generator? |

Parameter counts are reported, but similar parameter counts do not equalize sequential compute or inductive bias.

## What v2 still is not

```text
parallel hypotheses     yes
route revision          yes
commit or abstain       yes

transport one state between structures   no
learn a new structure                    no
arbitrary-size candidate library          no
real-world validation                     no
semantic decoding of hidden neurons       no
```

Those missing pieces are not renamed as future successes. They remain separate experiments.
