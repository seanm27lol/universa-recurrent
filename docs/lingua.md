# Lingua: say what ran, and what can be checked

Lingua is not a story about what a model “must have thought.” It is a typed record of explicit operations, measured dynamics, assumptions, and property-specific evidence.

## Record families

| Record | The checker establishes | It does not establish |
|---|---|---|
| Classical full trace | Every retained numerical transition and final constrained optimum, within tolerance | Remote execution attestation |
| Classical compact trace | Final constrained optimum, within tolerance | Discarded intermediate history |
| Neural v1 + checkpoint | Final structure, summaries, and binding to checkpoint/library | Learned update replay or correct routing |
| Neural v2 + checkpoint | Candidate state feasibility, residual/probability arithmetic, route revisions, policy decision, selected-or-mixture output, and checkpoint/library binding | Probability calibration, hidden semantics, optimality, or remote execution |

## V2 example

```text
STEP 1
  balanced_flow       p=0.58  observed residual=0.31
  alternate_structure p=0.42  observed residual=0.36
  decision: continue

STEP 4
  balanced_flow       p=0.89  observed residual=0.07
  alternate_structure p=0.11  observed residual=0.19
  decision: commit to balanced_flow

CHECKED
  each candidate obeyed its own declared boundary constraint
  probabilities summed to one
  the threshold rule permitted commitment at step 4
  the final state equals the selected candidate state

NOT CLAIMED
  a hidden feature means “conservation”
  the selected structure is true
  the network executed on a particular remote device
```

If confidence remains below the final rule, the record says `abstain` and the output is a mixture. It must not attach a single-structure certificate to that mixture.

## Five claims that remain different

```text
constraint satisfaction
        ≠ correct structural choice
        ≠ optimization optimality
        ≠ execution provenance
        ≠ faithful semantic interpretation
```

## Applied CMCM connection

Applied CMCM asks when the record of a computation matters. This repo turns that into engineering questions:

- which fields are retained;
- which properties remain independently checkable;
- how many bytes and how much time the record costs;
- which questions become impossible after compression.

Every new field needs an explicit use, cost, and test. More record information is not presumed better.
