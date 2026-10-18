# How the categorical pieces connect

This page maps the projects that inform the [CDL and TI-TNN integration plan](../research/ti_tnn/integration_plan.md)
to their actual interfaces, their assumptions, and what is missing between them. Every connection below
is a **documentation connection only**: no sibling repository is imported into the neural runtime.

**CDL** means *categorical deep learning*: using category theory to state what an architecture's parts
are (objects, operations) and which composition rules they obey. It is a specification language, not a
guarantee. A model satisfying a categorical law can still give wrong answers. Primary source: Gavranović
et al., [Categorical Deep Learning](https://proceedings.mlr.press/v235/gavranovic24a.html), ICML 2024.

## The projects

| Project (pinned) | What it actually provides | Assumptions it states | Missing for this integration |
|---|---|---|---|
| [Liquid-TQFT-lean-CMCM @ `25775b3`](https://github.com/seanm27lol/Liquid-TQFT-lean-CMCM/tree/25775b35d60cfadfec85e3d827b15dafe560ac0d) | Lean scaffolding: `AbstractTQFT`, a braided monoidal functor between arbitrary braided monoidal categories, and `AbstractTQFT.transfer`, which carries that data along a supplied braided monoidal functor ([source](https://github.com/seanm27lol/Liquid-TQFT-lean-CMCM/blob/25775b35d60cfadfec85e3d827b15dafe560ac0d/RequestProject/LiquidTQFT.lean)) | It says it "does not yet formalize a geometric Atiyah-Segal TQFT" and that `AbstractTQFT` "should not be identified with a complete TQFT formalization" | Any statement about a PyTorch model, learned semantics, calibration or deployment authority. A possible later use: formalizing a small typed program language |
| [HOMYMOLY @ `5946bf2`](https://github.com/seanm27lol/HOMYMOLY/tree/5946bf2100665d5e7a3b68beb0779321efd921e2) | Exact finite chain maps: degree maps parameterized in the nullspace of the chain-map equations, so d_D F = F d_C holds up to roundoff ([source](https://github.com/seanm27lol/HOMYMOLY/blob/5946bf2100665d5e7a3b68beb0779321efd921e2/src/homymoly/models/chain_map.py)) | "Task or paired signal losses are still required because the zero chain map is always legal" | Relevant only where a module really has degree-0 and degree-1 signals and boundary matrices. Its positive result is a specific scarce-data lifting problem |
| [Universa @ `132fdf8`](https://github.com/seanm27lol/Universa/tree/132fdf8d5ebe87deb3d6cb597c8ab77407176918), via this repo's [optional adapter](../src/universa_recurrent/structures/universa_adapter.py) | Compiled linear constraints. The adapter refuses when the upstream and local numerical nullspaces disagree | A supplied boundary operator | Numerical constraints identify geometry, not semantic concepts |
| [Neural v2](../src/universa_recurrent/neural/v2.py) (this repo) | Flow-specific recurrence with explicit continue, commit and abstain decisions; public `rollout` and `infer` | Fixed candidate bases, flow observations and masks | It is not a generic recurrent backend. Its private `_step` is not an interface |
| [TI-TNN and Jev](../research/ti_tnn/ti_tnn_train.py) (this repo) | Typed specialist branches, a router (Jev) trained with a Brier score against route and gate labels, conditional execution, and optional categorical branches ([`cdl.py`](../research/ti_tnn/cdl.py)) | Supplied route labels (or none, in the unsupervised variant) | Jev reads the input before any specialist runs, so its probabilities cannot detect a changed specialist. Query routing, token-level expert selection and recurrent halting are different decisions |
| [Lingua records](../src/universa_recurrent/neural/lingua.py) (this repo) | Versioned trace schemas (`universa-recurrent.neural-trace.v1`, `universa-recurrent.neural-v2-trace.v1`) and an [independent v2 checker](../src/universa_recurrent/neural/v2_verification.py) that validates a record's mathematical claims without running the network | Each schema's own scope | No universal semantic explanation or proof of execution. New typed-execution traces will get a new schema, not new fields in these |

## What the integration takes from each

- **Liquid-TQFT:** the habit of stating categorical assumptions and missing constructions explicitly.
- **HOMYMOLY:** the experimental pattern of comparing an exact structural constraint with unconstrained
  and properly matched soft alternatives. The TI-TNN composition controls already showed why the
  matching matters.
- **Universa:** compile constraints once per module version, count the setup cost, and refuse on
  disagreement.
- **Neural v2:** a worked example of explicit decision states. A later `FlowV2Adapter` would wrap its
  public API.
- **TI-TNN:** specialists, routing and conditional execution, plus a set of negative results about which
  structure actually helped.
- **Lingua:** scoped records, rejection, and checkers that never call the producer.

## Guarantees that do not transfer

A passing check in one column says nothing about another:
- type validity, algebraic conformance, calibration, semantic correctness, trace faithfulness and
  authorization are separate (see the [integration plan](../research/ti_tnn/integration_plan.md));
- [`tests/test_ti_tnn_guarantee_boundaries.py`](../tests/test_ti_tnn_guarantee_boundaries.py) keeps two
  counterexamples in the test suite.
