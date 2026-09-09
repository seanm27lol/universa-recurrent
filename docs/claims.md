# Claim ledger

| Statement | Status | Evidence or next test |
|---|---|---|
| Classical flow demo reuses reduced state and avoids repeated SVD | Implemented | Solver and factorization-blocking tests |
| Classical full records support update-chain checks | Implemented numerically | Independent checker and tamper tests |
| Learned two-way structure router | Implemented on one synthetic family | Held-out accuracy, classical baseline, generator-aware reference |
| Learned state-reusing recurrence | Implemented | Shared update network over explicit latent coordinates |
| Decoded neural state lies in the selected subspace | Implemented up to floating-point residual | Parameterization `z = Q a` and independent record check |
| Learned halting reduces logical updates | Measurable, not guaranteed | Threshold table on held-out data |
| Learned halting reduces executed update examples | Available only in compact mode | `update_examples_per_sample` |
| Learned halting makes inference faster | **Not established** | GPU wall-clock benchmark against fixed depths at comparable quality |
| Learned router beats a simple structural router | **Not established** | `fit_each_structure` baseline can match or beat it |
| Explicit subspace coordinates improve over an unstructured recurrent model | **Not tested** | Requires a matched ambient/reduced random-subspace control |
| Neural recurrence beats the strongest fixed depth | **Not established in general** | Compare all reported depths; do not use only the eight-step control |
| Weight sharing or recurrence itself causes any observed gain | **Not isolated** | Requires tied-vs-untied and recurrent-vs-direct feed-forward ablations |
| The structure label is perfectly recoverable in the toy generator | FALSE | The candidate spaces overlap and partial noisy observations can be ambiguous; use the privileged reference for context |
| Generator-aware Bayes reference is deployable | FALSE | It knows the synthetic prior and noise level |
| Neural Lingua verifies the learned update path | FALSE in v1 | It checks final structural claims and summaries, not network replay |
| Standalone neural records are authenticated | FALSE | Use optional local-checkpoint binding; neither mode is remote execution attestation |
| `weights_only=True` makes arbitrary checkpoints safe | FALSE | PyTorch is pinned to 2.10+ for known fixes, but checkpoints must still come from a trusted source |
| Lingua explains hidden-feature semantics | **Not established** | Requires predictive and causal intervention evidence |
| Recurrence switches between structures | **Not implemented** | Current inference routes once, then remains in that space |
| General structure discovery | **Not integrated** | Future Universa-based experiment |
| Compact records preserve every historical property | FALSE by design | Endpoint/path counterexamples |
| Topology or category theory generally improves models | Not claimed | Narrow task-specific evidence only |

Engineering tests are not preregistered scientific results. Exploratory thresholds,
architectural revisions, and negative findings must remain visible rather than
being retroactively described as confirmed hypotheses.
