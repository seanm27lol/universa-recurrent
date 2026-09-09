# Familiar examples and primary sources

These are conceptual anchors, not claims that the toy program is deployed in those
fields. Each source supports only the connection stated in its row.

| Idea here | Known setting | Connection | Where the analogy stops |
|---|---|---|---|
| Conservation subspace | Electrical junctions | Kirchhoff's current law expresses charge balance. [1] | Our graph lacks voltage/resistance laws; conservation alone is not a complete circuit model. |
| Repeated correction | Convex optimization | Gradient methods and equality-constrained optimality are established numerical tools. [2] | Our classical baseline does not establish an ML advantage. |
| Trainable iterations, planned | Speech enhancement and learned inference | Deep unfolding builds networks from iterative inference procedures. [3] | Some unfolded networks untie weights; shared-weight recurrence is a separate design choice. |
| A certificate checked independently | Constrained optimization | Feasibility plus stationarity certifies a convex solution under the stated assumptions. [2] | Numerical residuals do not establish that the real-world model is right. |
| Faithful neural explanations, planned | Mechanistic interpretability | Causal abstraction relates neural computations to interpretable algorithms. [4] | A readable operation log is not itself a causal abstraction of a neural network. |
| Keeping selected records | Applied CMCM experiments | Compare endpoints with explicit and partial computational records. [7] | Information availability did not universally predict learned benefit. |
| Commit or abstain | Statistical classification with rejection | A decision system may reject an uncertain case under explicit error/rejection costs. [8] | A calibrated toy threshold is not automatically appropriate in deployment. |
| Preserve several hypotheses | Bayesian model averaging | Predictions can account for model uncertainty instead of pretending one model is certain. [9] | V2 probabilities are learned and are not claimed to be Bayesian posteriors. |

## Primary references

1. OpenStax, *University Physics Volume 2*, section 10.3,
   [Kirchhoff's Rules](https://openstax.org/books/university-physics-volume-2/pages/10-3-kirchhoffs-rules).
2. Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (2004),
   [author-hosted book page](https://web.stanford.edu/~boyd/cvxbook/).
   The self-contained special-case proof used here is in [mathematics.md](mathematics.md).
3. John R. Hershey, Jonathan Le Roux, Felix Weninger (2014),
   [Deep Unfolding: Model-Based Inspiration of Novel Deep Architectures](https://arxiv.org/abs/1409.2574).
4. Atticus Geiger et al. (2025),
   [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://jmlr.org/papers/v26/23-0058.html).
5. [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY): revised lifting result and its explicit scope limitations.
6. [Universa](https://github.com/seanm27lol/Universa/tree/132fdf8d5ebe87deb3d6cb597c8ab77407176918): structural prototype at the pinned integration commit.
7. [Applied Experiments of CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM): use the current `paper/` sources, not its superseded root PDF.
8. C. K. Chow (1970), [On optimum recognition error and reject tradeoff](https://doi.org/10.1109/TIT.1970.1054406).
9. Hoeting, Madigan, Raftery, and Volinsky (1999), [Bayesian Model Averaging: A Tutorial](https://doi.org/10.1214/ss/1009212519).

No claimed novelty in conservation laws, nullspaces, projected gradient descent,
optimality conditions, or the existence of causal abstraction. Proposed novelty
must be established by new integrations and careful comparative experiments.
