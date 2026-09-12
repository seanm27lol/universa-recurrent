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
| Lingua checks candidate geometry, recorded diagnostics and policy arithmetic | Implemented for retained one-example records | Independent checker and tamper tests |
| Lingua explains hidden-feature semantics | **Not established** | Predictive and causal intervention evidence |
| V2 transports one persistent state between structures | **Not implemented** | Future typed-transport experiment |
| V2 discovers new structures | **Not implemented** | Future Universa integration |
| A passing test is a scientific result | FALSE | A confirmatory protocol must be sealed before its test block is opened |
| Execution benchmarks can avoid inheriting training settings | Implemented in 0.5.1 | Fresh workers, explicit common controls, and stored backend settings |
| Determinism explains the earlier GPU timing shift | **Not established** | Rebenchmark existing checkpoints; historical settings were not fully recorded |
| Switching an execution profile leaves every output unchanged | Must be checked per run | Numeric tolerances and exact discrete-claim comparison; mismatches suppress matched-output ratios |

Keep negative and ambiguous outcomes visible. Do not turn logical step reduction into a latency claim or a valid structural state into a correct modeling assumption.

For the no-retraining profile comparison and its limits, see [Execution audit](execution_audit.md).

## Final-state retention study (0.5.2)

The opt-in final-only adapter retains no history stack and reuses the existing
fixed-depth updates. Tests and the GPU runner compare final tensors and discrete
claims to rollout. This is not adaptive stopping or an established speedup.
Retained-history checks cover candidate geometry and diagnostic arithmetic, not
learned transitions. Sampled Lingua timings and full-cohort inference have distinct
denominators. See [the retention study](final_state_inference.md).

## Prepared verifier and request pipeline

| Statement | Status | Evidence or required test |
|---|---|---|
| Preparing a pinned reference amortizes repeated verification setup | Reported CPU timings, independently recomputed | [September 10 setup audit](verifier_setup_20260910.md); 97.95% less time for 64 shared endpoint receipts, preparation included |
| The setup ZIP establishes new neural accuracy or independently replayed correctness | FALSE | Four saved receipts are repeated; receipt/reference bytes are absent |
| Prepared checking still runs each record's arithmetic and geometry checks | Source-reviewed and regression-tested | Unchanged per-record checker plus exact snapshot matching; no cached verdict |
| Preparing a new reference helps a one-record batch | Not supported in this run | About 2.2% slower with preparation included |
| Prepared references reduce measured full request costs | Reported DGX timings; locally audited reports and saved receipts | [September 11 pipeline results](verified_pipeline_20260911.md): 80.61x paired speedup for 256 shared endpoint records, preparation included |
| Every saved pipeline receipt passes the existing property checker | Independently replayed locally | 6,400 receipts, 256 distinct inputs; no checkpoint binding or neural execution replay |
| A passing property check establishes a correct structural choice | FALSE | Shared claims choose the wrong synthetic label about 13.4% of the time in the small post-hoc cohort |
| The fresh pipeline cohort establishes confirmatory model superiority | FALSE | Post-hoc 256-input comparison; shared/four-step differences are small and mixed |

## Prospective claim-reliability study

The [fixed next protocol](claim_reliability.md) evaluates the existing five fits
on 20,000 fresh common test inputs after every coverage threshold is frozen on
5,000 separate calibration inputs. It specifies a primary shared/four-step
comparison at 75% target coverage and input-paired uncertainty conditional on the
fits and calibration. Software implementation and local smoke tests do not
establish its scientific result. Actual held-out coverage, wrong-claim rates and
numerical-estimate error must be interpreted separately; the protocol promises no
model winner, risk guarantee, equivalence result or new receipt-verification claim.
