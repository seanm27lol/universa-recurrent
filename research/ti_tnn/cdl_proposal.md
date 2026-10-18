# Proposal: categorical structure in TI-TNN (functorial branches and a flat sheaf bridge)

**Status: proposal plus an experimental implementation, tested for its structural
guarantees only.** No training result exists yet. The experiment in the last
section must be frozen, with predictions, before it runs.

## Why, in the project's own terms

The original notes for the tree-informed network name the problem directly:
- "The sheer inefficient computation makes interp hard, and the combination of
  opaque recurrence makes it even worse."
- Categorical deep learning is listed among the remedies.

The design sketch decomposes "What is the capital of France?" into typed parts
(noun, location with importance) with their own representations (countries,
capitals), then reconstructs the answer. The aims were cheaper compute and
interpretability.

TI-TNN already has typed branches, a router and a bridge. What it lacks is a
guarantee that the pieces mean something stable. Today a branch is an
unconstrained MLP and the bridge compares two learned projections. Nothing
says that translating a country representation into a capital representation
and back returns the same thing, or that two translations compose.

This proposal adds that structure, and adds it **exactly where possible**:
- **The HOMYMOLY lesson.** Its sealed replication found that an exact
  structural constraint beat a soft penalty encoding the same information, and
  that a soft penalty's zero set can be trivial ("W = 0 also has zero defect").
- **This design.** Laws that can be built into the parameterization are built
  in. A soft, router-weighted loss is used only for the one law that cannot
  be, and the trivial collapse to zero is made unreachable.

## The mathematics

**The query category.** A finitely presented category Q:
- objects are entity kinds (country, capital, region, size);
- generating morphisms are relations (`capital_of: country → capital`,
  `size_of: capital → size`, …);
- inverse pairs are declared (`country_of = capital_of⁻¹`).

`routing_world_category()` is the routing benchmark's relations in this form.

**A functorial branch (`CDLSpecialistBranch`).** It keeps the original residual
encoder (objects may be any vectors) and adds a functor F_k: Q → ℝ^d:
- every free generator g gets an orthogonal matrix M_g on the branch's stalk;
- the partner of an inverse pair gets M_gᵀ;
- a path g₁, …, gₙ acts by M_gₙ ⋯ M_g₁.

Identity, composition and inverse laws therefore hold **exactly, for any
parameter values**. A router-chosen morphism head selects which generator a
query applies. It is hard (one-hot) in both training and inference, so the
applied map is always an exact functor image.

**A flat sheaf bridge (`FunctorialSheafBridge`).** The existing bridge has one
restriction map per *edge role*. This one has one orthogonal restriction map
Q_k per *branch*, into a shared edge frame: a flat O(d) connection sheaf on the
branch graph (Hansen and Ghrist; Bodnar et al. use orthogonal restriction maps
in neural sheaf diffusion).
- **Transport across the bridge** is T(u→v) = Q_vᵀ Q_u. So, exactly and for any
  parameters:
  - T(u→u) = I;
  - T(v→w) T(u→v) = T(u→w);
  - T(v→u) = T(u→v)ᵀ = T(u→v)⁻¹;
  - holonomy around every cycle is trivial.
- **The coboundary** Q_u s_u − Q_v s_v vanishes exactly on transported sections,
  so the bridge's correction W_e tanh(δ) is "no news" when two branches agree up
  to transport.
- **An analogy, not a construction.** In a TQFT, gluing cobordisms maps to
  composing linear maps. Here, chaining bridge translations maps to composing
  transports.

**The one soft law: naturality.** The bridge's transports should be natural
transformations between branch functors:

```text
T(u→v) · M^u_g  =  M^v_g · T(u→v)      for every generator g
```

That is, translating and then applying a relation equals applying the relation
and then translating. This cannot be built in independently of each branch's
morphisms, so `naturality_loss` measures the defect:

```text
L_nat = Σ_{u<v} w_uv · mean_g ‖T(u→v) M^u_g − M^v_g T(u→v)‖²_F / d
w_uv  = floor + E_batch[p_u p_v + p_v p_u]      (router probabilities, detached)
```

- **The weight** is the router's expected use of the branch pair. The constraint
  is strongest where the bridge actually pairs two branches.
- **The floor** (default 0.05) keeps idle pairs from drifting into unconstrained
  maps.
- **Calibration.** The probabilities are calibrated by the Brier loss only when
  routing labels exist. In unsupervised routing they are just the router's
  softmax.
- **Collapse.** Orthogonality makes the trivial zero unreachable (M = 0 is not
  orthogonal). The trivial *identity* functor (all M = I) still satisfies
  naturality, and only the task loss keeps morphisms informative. Watch for it.

**Isolation is preserved.** The naturality loss couples branches only through
global matrices, never through per-entity representations. The routed-training
isolation property (Branch 1's state never shapes Branch 0's update) is tested
with both options on.

## Implementation

| Piece | Where |
|---|---|
| `QueryCategory`, `routing_world_category`, `CDLSpecialistBranch`, `FunctorialSheafBridge`, `pair_weights`, `naturality_loss` | `research/ti_tnn/cdl.py` |
| Opt-in wiring: `TypeInformedTreeNN(branch_kind="cdl", bridge_kind="functorial", category=..., functorial_weight=0.1, functorial_floor=0.05)` | `research/ti_tnn/ti_tnn_train.py` |
| Tests | `tests/test_ti_tnn.py` (15 new) |

**Defaults are unchanged.** With default arguments the model is identical to
`main`'s: the same parameters, initial values, and training and evaluation
outputs for the same seed (checked against `main`'s script). Every recorded
TI-TNN result stays reproducible.

**What the tests verify:**
- **Exact laws under heavy perturbation.** Every parameter gets noise of scale
  10, and the modules are built in float64 so the orthogonal base matrix is
  float64 too. Then:
  - the generator isomorphisms and the inverse pair hold;
  - step-by-step application equals the path's image;
  - associativity holds;
  - bridge transports satisfy identity, inverse and composition for every
    triple, with trivial cycle holonomy;
  - the coboundary vanishes on transported sections and is nonzero on
    inconsistent ones.

  Errors are about 10⁻¹⁴.
- **The soft law.** The naturality loss is zero on a natural family and
  positive when one branch is perturbed. It ignores pairs the router never
  uses unless the floor is on. Zeroed parameters still give orthogonal maps.
- **The model.** A CDL model trains (gradients reach the morphisms and
  restriction maps), and its conditional inference equals running every
  branch. The routed-training isolation property survives.

## Proposed experiment (not run; freeze before running)

**Functorial composition on the routing world.**
- **Train** on single-hop queries only: `capital_of <country>`,
  `country_of <capital>`, `size_of <capital>`, `region_of <country>`.
- **Test** on two-hop queries never seen in training: `size_of capital_of
  <country>` and `region_of country_of <capital>`. The model applies two
  generator steps.
- **Compare:**
  - CDL branches with the functorial bridge;
  - CDL branches with the original bridge;
  - MLP branches given the same two-step interface (each step an unconstrained
    map);
  - a dense model.
- **Measure:**
  - two-hop accuracy (the claim: composition by construction);
  - one-hop accuracy (no loss of fit);
  - the naturality defect at the end of training;
  - whether morphisms stayed informative rather than collapsing to the
    identity.
- **Interpretability check:** read each branch's morphisms directly. Does
  M_country_of M_capital_of = I as learned behaviour, not only by construction
  (it is), and do transports map a country's stalk onto its capital's?

**What would count against the idea:**
- two-hop accuracy no better than the unconstrained two-step model;
- or naturality reaching zero only through the identity functor.

## Limits

- Real text has no given category. On WikiText or SCAN there is no
  `capital_of` to be functorial over. A learned or discovered category is a
  separate, harder question.
- Orthogonal morphisms are invertible by design. Many-to-one relations
  (`region_of`) are forced to be invertible on the stalk; a
  linear-relation (span) parameterization would lift that. That's future work.
- The exact laws are guaranteed structure, not evidence of better performance or
  interpretability. Only the experiment above can show either.

## Status map (appended 2026-10-18; the proposal above is unchanged)

| Part | Status | Where |
|---|---|---|
| Structural laws (composition, identity, inverse pairs; flat transports) | Implemented and tested, exact by parameterization | `cdl.py`, `tests/test_ti_tnn.py` |
| Starting every orthogonal map in SO(d), so naturality is reachable | Implemented and tested | `cdl.orthogonal_square`, `tests/test_ti_tnn.py` |
| Standalone transport experiments: composition of relations learned one at a time | Run (5 seeds each), on a separate embedding model, not the routed system | [first run](reports/functorial_composition.md), [controls](reports/functorial_composition_controls.md) |
| The integrated experiment proposed above (categorical branches inside the routed model, with naturality) | Not run | Superseded by the [integration plan](integration_plan.md), which keeps semantic relations and frame changes separate |

The controls found that the standalone result is explained by a shared embedding space with
geometry-preserving maps; the category laws showed no measured advantage there. They do not count as
evidence for the integrated system.

