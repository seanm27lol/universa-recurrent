# Claim ledger

| Statement | Status | Evidence or required test |
|---|---|---|
| Classical solver reuses reduced coordinates and avoids repeated SVD | Implemented | Classical solver tests |
| Neural v1 routes once, then recurs inside one selected space | Implemented; historical exploratory path | v1 source and tests |
| Neural v2 maintains one state per candidate structure | Implemented | Shape, gradient, and inference tests |
| V2 revises candidate probabilities after recurrent updates | Implemented | Retained trajectories and Lingua events |
| V2 can continue, commit, or abstain | Implemented | Dense/compact and policy tests |
| Abstention returns a provisional probability-weighted mixture | Implemented | Arithmetic and tamper tests |
| Policy thresholds use a disjoint calibration split | Implemented | Checkpoint seed metadata and tests |
| Candidate states satisfy their declared linear constraints | Implemented up to floating-point tolerance | `z = Q a` plus independent checker |
| Dense and compact v2 make matching decisions | Tested on listed cases | Equivalence tests within declared tolerances |
| Compact v2 skips candidate update examples | Implemented | Execution counters |
| Compact v2 is faster | **Not established** | DGX wall-clock benchmark at matched quality and batch size |
| V2 improves on v1 | **Not established** | New DGX evaluation on untouched test seeds |
| V2 beats the transparent solver | **Not established** | Held-out evaluation and timing |
| Recurrence itself causes a gain | **Not established** | Direct, dedicated-depth, and untied controls |
| Structural coordinates cause a gain | **Not established** | Ambient recurrent control |
| Route probabilities are calibrated | **Not established** | Reliability/calibration diagnostics on untouched data |
| The calibrated costs are universally appropriate | FALSE | They are declared experimental choices |
| Lingua verifies candidate trajectories and policy arithmetic | Implemented for retained one-example records | Independent checker and tamper tests |
| Lingua explains hidden-feature semantics | **Not established** | Predictive and causal intervention evidence |
| V2 transports one persistent state between structures | **Not implemented** | Future typed-transport experiment |
| V2 discovers new structures | **Not implemented** | Future Universa integration |
| A passing test is a scientific result | FALSE | A confirmatory protocol must be sealed before its test block is opened |

Keep negative and ambiguous outcomes visible. Do not turn logical step reduction into a latency claim or a valid structural state into a correct modeling assumption.
