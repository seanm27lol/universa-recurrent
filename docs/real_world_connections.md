# Familiar examples and primary sources

These are conceptual anchors, not claims that the toy program is deployed in
those fields. Each source supports only the connection stated in its row.

| Idea here | Known setting | Connection | Where the analogy stops |
|---|---|---|---|
| Conservation subspace | Electrical junctions | Kirchhoff's current law expresses charge balance. [1] | Conservation alone is not a complete circuit model |
| Repeated correction | Convex optimization | Gradient methods and equality-constrained optimality are standard numerical tools. [2] | A known optimizer does not establish an ML advantage |
| Trainable repeated update | Sparse coding and deep unfolding | LISTA and deep unfolding replace hand-designed iterations with trainable operations. [3,4] | Weight sharing, structure routing, and this task are separate design choices |
| Input-dependent computation | Adaptive computation time | A learned mechanism can allocate different step counts to different inputs. [5] | A lower logical step count is not automatically lower hardware latency |
| Constraint certificate | Constrained optimization | Feasibility plus stationarity can certify a convex solution under stated assumptions. [2] | A valid certificate does not prove the real-world model is appropriate |
| Mathematical account of neural mechanisms | Causal abstraction | Interventions test whether neural variables realize a proposed high-level algorithm. [6] | A readable log alone is not a causal explanation |
| Keeping selected records | Applied CMCM experiments | Compare endpoints with explicit and partial computational records. [9] | Information availability did not universally predict learned benefit |

## Primary references

1. OpenStax, *University Physics Volume 2*, section 10.3,
   [Kirchhoff's Rules](https://openstax.org/books/university-physics-volume-2/pages/10-3-kirchhoffs-rules).
2. Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (2004),
   [author-hosted book](https://web.stanford.edu/~boyd/cvxbook/).
3. Karol Gregor and Yann LeCun (2010),
   [Learning Fast Approximations of Sparse Coding](https://proceedings.mlr.press/v9/gregor10a.html).
4. John R. Hershey, Jonathan Le Roux, and Felix Weninger (2014),
   [Deep Unfolding: Model-Based Inspiration of Novel Deep Architectures](https://arxiv.org/abs/1409.2574).
5. Alex Graves (2016),
   [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983).
6. Atticus Geiger et al. (2025),
   [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://jmlr.org/papers/v26/23-0058.html).
7. [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY): revised lifting result and explicit scope limitations.
8. [Universa](https://github.com/seanm27lol/Universa/tree/132fdf8d5ebe87deb3d6cb597c8ab77407176918): structural prototype at the pinned integration commit.
9. [Applied Experiments of CMCM](https://github.com/seanm27lol/Applied-Experiments-of-CMCM): use the current `paper/` sources, not the superseded root PDF.

No novelty is claimed for conservation laws, nullspaces, projected gradient
descent, KKT conditions, deep unfolding, adaptive computation, or causal
abstraction. Proposed novelty must come from measured integrations and explicit
claim boundaries.
