# The mathematics, from the flow example

## Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| z | Edge-flow estimate | n numbers |
| B | Junction constraints | m by n |
| A | Which weighted measurements we observe | p by n |
| y | Measured values | p numbers |
| Q | Orthonormal basis of the allowed subspace | n by d |
| a | Reduced working coordinates, z=Qa | d numbers |
| ρ | Positive ridge weight | one number |

The model is a **closed circulation**: Bz=0. With sources, sinks, or prescribed
injections, Bz=b is the relevant extension. We do not apply a homogeneous cycle
constraint to arbitrary data or to an arbitrary neural hidden state.

## The reconstruction problem

$$
\min_{Bz=0} f(z),\qquad
f(z)=\tfrac12\|Az-y\|_2^2+\tfrac\rho2\|z\|_2^2,\quad \rho>0.
$$

The first term fits measurements. The second mildly discourages large estimates.
That regularization introduces bias; the optimizer need not equal the hidden
synthetic truth. We compare both, without calling them the same target.

Choose Q with orthonormal columns spanning the numerical kernel of B. Let
H=(AQ)^T(AQ)+ρI and b=(AQ)^Ty. In coordinates, the known update is

$$
a_{k+1}=a_k-\eta(Ha_k-b),\qquad \eta=1/\lambda_{\max}(H).
$$

For exact arithmetic and d>0, H is positive definite, so this step size gives
convergence to the unique constrained minimizer. The direct baseline solves
Ha=b. If d=0, the only allowed vector is zero; no iterative inference is needed.
The code checks stationarity and imposes a maximum number of steps.

The same update in ambient coordinates is z_{k+1}=P(z_k-η∇f(z_k)), provided z_k
is feasible and P=QQ^T. The implementation retains reduced coordinates; its trace
uses the equivalent projected-gradient operation, not a fictitious neural step.

## Projection witness: checking without re-solving

Claim: z is the orthogonal projection of v onto ker(B).
Witness: a vector λ such that

$$
Bz=0,\qquad v-z=B^T\lambda.
$$

For any feasible u, B(u-z)=0, so
$\langle v-z,u-z\rangle=\langle\lambda,B(u-z)\rangle=0$.
Pythagoras then gives
$\|v-u\|^2=\|v-z\|^2+\|z-u\|^2\geq\|v-z\|^2$.
This is a proof about the specified projection, not about whether B is appropriate.

## Final optimization witness

Check feasibility Bz=0 and stationarity ∇f(z)+B^Tλ=0. For any feasible u,

$$
f(u)-f(z)
=\nabla f(z)^T(u-z)+\tfrac12\|A(u-z)\|^2+\tfrac\rho2\|u-z\|^2
\geq0.
$$

The first term vanishes under the witness equations. This establishes the unique
minimizer in exact arithmetic because ρ>0. This is standard constrained convex
optimization, not a new theorem; see [Boyd and Vandenberghe](real_world_connections.md).

## Numerical and complexity boundaries

The producer caches one dense SVD per CompiledConstraint and a multiplier map.
The checker uses matrix-vector products and norms, never SVD or a solve. This
implementation uses **dense NumPy arrays**; sparse O(nnz) checking is future work,
not a measured property of the shipped code. The checker can still be more costly
than a tiny cached direct solve; benchmark the whole path.

Checks use absolute and relative tolerances. Small feasibility/optimality residuals
are numerical evidence, not exact proof, and an error bound needs conditioning
assumptions. Rank is numerical and controlled by a declared relative cutoff.
Ill-conditioned cases can make upstream/local bases disagree; the adapter refuses
rather than silently reinterpret the constraint.

No tolerance check establishes correct structure, causality of a learned decision,
physical fidelity, or cryptographic provenance. See [Lingua](lingua.md).
