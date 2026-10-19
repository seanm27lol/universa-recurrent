# CDL and TI-TNN integration plan

**Status: proposed research and engineering plan, adopted 2026-10-18.** Nothing here is an established
architecture, experiment result, safety guarantee or performance result. Progress is tracked in the
[status table](#status-by-pull-request) below; [`integrated/`](integrated/) holds the code as it lands.
The plan was written against `main` at `09403bd` and is reproduced from its author's draft with
repository links made relative. Review amendments are in [section 11](#11-review-amendments-2026-10-18).

**Date:** October 18, 2026. **Repository reviewed:** [universa-recurrent at 09403bde48bdf59d111c4e971b8b6292d2afc642](https://github.com/seanm27lol/universa-recurrent/tree/09403bde48bdf59d111c4e971b8b6292d2afc642). The plan incorporates the original handwritten sketches, the reviews of PRs 38–40, and the source audit below.

## 1. Goal and first milestone

Build a small system that interprets a question, selects meaningful operations, executes them through typed interfaces, records the intermediate values it actually uses, and returns an answer or abstains. Test whether this organization improves compositional accuracy, inspection, inference cost, or the locality of later updates.

Begin with a fictional example:

> Which company operates the airport serving the largest city of Ardel?

The intended computation is `largest_city_of`, followed by `airport_of`, followed by `operator_of`. Each intermediate value has an explicit type: Country, City, Airport, or Operator. Include a separate `capital_of: Country -> City` relation whose answers frequently differ from `largest_city_of`. This prevents an executor from treating all relations between the same types as equivalent.

The first milestone is a complete, small semantic executor and a controlled experiment. Start with a correct supplied program to test the operators; then use a learned interpreter over controlled language to test question-to-answer integration. Both conditions must be reported separately. Another large language-model run should follow evidence that these components work and that their added complexity improves a specified property.

The integration should preserve five distinctions throughout:

1. **Type validity:** the chosen interfaces compose.
2. **Algebraic conformance:** specified equations hold under stated assumptions and tolerances.
3. **Calibration:** a probability predicts a defined outcome on a defined distribution.
4. **Semantic behavior:** the system answers correctly and its intermediate values have the claimed effects.
5. **Authorization:** the runtime is permitted to activate an update or execute an operation.

These are separate claims. A passing result for one does not establish the others.

## 2. Correct the source interpretation before integration

### Categorical deep learning

CDL means **categorical deep learning**. The ICML position paper uses category theory to connect architectural specifications and implementations, including recurrent networks. It is not a requirement to abandon continuous vectors, and it does not automatically give every architecture strict laws, human-readable semantics, or safe updates. Each chosen category, representation and equation must be specified. [CDL primary paper](https://proceedings.mlr.press/v235/gavranovic24a.html)

In the current TI-TNN implementation, orthogonal maps, declared inverse pairs and some composition identities are built into the parameterization. Naturality is a separate soft loss. The existing categorical proposal remains useful, but its structural tests and the standalone composition experiments must not be described as validation of the entire proposed routed system. [Current proposal](cdl_proposal.md)

### Cammarata and Goodfire

Cammarata’s essay is a working hypothesis. It reports a preliminary roughly 90% one-job neuron-pole estimate and describes small models’ features as broader. The pasted “folding unrelated things together” passage presents an explanation the author rejects. The essay supplies no TI-TNN validation or 32B cleanliness threshold. Use it to motivate native-neuron and circuit measurements. [Essay](https://already-clean.goodfire.com/)

The separate Goodfire monitoring result concerns a Kimi K3 monitoring stack: approximately 93% harmful-session detection with 5.5% of benign sessions flagged. It is not the Qwen neuron audit or a universal safety rate. It need not appear in the TI-TNN overview. [Goodfire’s report](https://www.goodfire.com/blog/goodfire-baseten-partnership)

### Jev and Brier calibration

Jev is the project’s name for its toy query router. Brier training compares probabilities with specified route or gate labels. A Brier objective neither authenticates weights nor detects every downstream change. The current Jev reads the input before the specialist executes; a specialist can change while those probabilities remain identical. A correctness-confidence estimator would require its own outcome labels and evaluation for each relevant model version. [Router and loss implementation](ti_tnn_train.py), [calibration reference](https://proceedings.mlr.press/v70/guo17a.html)

### Isolation, capacity and updates

The XOR bridge-off result is a useful causal control in a specified synthetic setup. It is not a universal capability lock. The current README already discloses an indirect cross-step channel through shared parameters and describes the chance-level cap as expected and observed rather than proved. Preserve that correction. [Isolation scope](README.md)

A narrow specialist can still lack capacity. Sparse active computation can coexist with a large total parameter footprint. The historical toy MAC counts describe particular arithmetic, not measured energy, total edge-system cost, or clean representations. Local parameter writes can affect many downstream uses of a module; a fixed root does not guarantee fixed behavior.

## 3. A permanent counterexample against overclaiming

Add a regression fixture where categorical laws hold and Brier is unchanged, yet the answer changes.

For example, use a category presented by an inverse pair and no further relations. Represent the pair by two-dimensional identity matrices, then replace both by minus the identity. Both versions are determinant-positive orthogonal matrices and satisfy the declared inverse laws; path maps compose functorially in both. With fixed input `(1, 0)` and a decoder distinguishing `(1, 0)` from `(-1, 0)`, the answer flips. Keep route probabilities `(0.9, 0.1)` and route target `(1, 0)` unchanged: the multiclass Brier score is 0.02 in both versions.

A second fixture should follow the current CDL implementation directly: keep every morphism and transport fixed, and change only a selected branch’s output projection. Its output can change while the upstream Jev probabilities and every categorical defect remain unchanged.

The correct checker behavior is:

- structural checker: the relevant laws pass;
- router-calibration measurement: unchanged;
- behavioral checker: the undeclared semantic change fails;
- activation policy: rejects the candidate for the failed behavioral contract.

The tests should preserve this distinction, rather than incorrectly requiring the algebraic or Brier checks to detect every semantic change.

## 4. Give each existing project a bounded role

| Project or component | Contribution to this integration | Boundary |
|---|---|---|
| Liquid-TQFT | Examples of explicit categorical assumptions and composition proofs; optional later formalization of a small typed program language | Existing Lean statements do not certify a PyTorch model, learned semantics, calibration, or deployment authority |
| HOMYMOLY | Motivation for comparing a correct known structural constraint with unconstrained and properly matched soft alternatives | Its current positive result is a specific scarce-data lifting problem; its original defect-routing thesis did not survive as stated |
| Universa | Optional pinned adapters for declared linear geometry, transport and projection | Numerical constraint discovery is not discovery of semantic concepts |
| universa-recurrent classical core | Independent verification, compile-once geometry, budgets, and a transparent flow reference | The solver handles a supplied constrained quadratic |
| universa-recurrent neural v2 | A flow-specific example of continue, commit and abstain with explicit candidate states | It is not an arbitrary language-model hidden-state backend |
| TI-TNN and Jev | Specialist modules, routing objectives, conditional execution and existing controls | Query routing, token-level expert selection and recurrent halting are different decisions |
| Lingua work | Patterns for scoped records, rejection, state boundaries and independent checking | Existing record formats have distinct guarantees; none is a universal semantic explanation or proof of execution |

### Verified upstream pins

- [HOMYMOLY at 5946bf2100665d5e7a3b68beb0779321efd921e2](https://github.com/seanm27lol/HOMYMOLY/tree/5946bf2100665d5e7a3b68beb0779321efd921e2).
- [Universa at 132fdf8d5ebe87deb3d6cb597c8ab77407176918](https://github.com/seanm27lol/Universa/tree/132fdf8d5ebe87deb3d6cb597c8ab77407176918).
- [Liquid-TQFT-lean-CMCM at 25775b35d60cfadfec85e3d827b15dafe560ac0d](https://github.com/seanm27lol/Liquid-TQFT-lean-CMCM/tree/25775b35d60cfadfec85e3d827b15dafe560ac0d).

Liquid-TQFT’s `AbstractTQFT.transfer` composes supplied braided monoidal data with a supplied compatible functor. Its source explicitly states the additional assumptions and missing constructions. Begin with a documentation connection; add no Lean dependency to the neural runtime. [Formal scope](https://github.com/seanm27lol/Liquid-TQFT-lean-CMCM/blob/25775b35d60cfadfec85e3d827b15dafe560ac0d/RequestProject/LiquidTQFT.lean#L1-L87)

HOMYMOLY’s exact chain-map implementation is relevant only where a module actually has declared degree-zero and degree-one signals and boundary matrices. The zero map satisfies a chain-map equation, so behavioral usefulness must be checked separately. Its lifting result and this module are distinct pieces. [HOMYMOLY thesis](https://github.com/seanm27lol/HOMYMOLY/blob/5946bf2100665d5e7a3b68beb0779321efd921e2/README.md), [chain-map implementation](https://github.com/seanm27lol/HOMYMOLY/blob/5946bf2100665d5e7a3b68beb0779321efd921e2/src/homymoly/models/chain_map.py)

The current optional Universa adapter already checks agreement between upstream and local projectors. Reuse that pattern; compile immutable geometry once per module version, include setup costs, and avoid importing entire sibling repositories. [Existing adapter](../../src/universa_recurrent/structures/universa_adapter.py)

## 5. Proposed architecture

Place the new research code under **`research/ti_tnn/integrated/`**, a proposed directory. Keep completed experiments and core public APIs stable. Start with adapters and new protocols; promote a reusable component into `src/` only after its semantics and compatibility requirements are established.

### 5.1 Typed values and relation identities

Define a small contract with these records:

| Record | Minimum content |
|---|---|
| `TypeSpec` | Stable type ID, version, value schema, dimensions where applicable, semantic domain |
| `RelationSpec` | Unique relation ID, source and destination types, arity, supported behavior, declared laws, dependency description |
| `ModuleVersion` | Immutable code/config/parameter/buffer identities, implementation of a relation, precision, calibration reference |
| `TypedValue` | Type/version, canonical value or explicit distribution, uncertainty and provenance |
| `Program` | Ordered typed operations or bounded control-flow graph, source binding and expected output type |
| `Decision` | Execute, continue, commit or abstain; chosen relation/module, probabilities and calibration version |
| `Budget` | Remaining steps and explicit resource limits; accounting policy version |
| `RunState` | Typed working values, program position, module-version snapshot, budget and declared continuation state |

Distinguish semantic types from learned tensor shapes. Country and City may have the same vector dimension while remaining different types. Give `capital_of` and `largest_city_of` distinct identifiers and operators even though their endpoints match.

Use a hierarchy if it helps the router choose a domain. Use a relation graph for the actual semantics. Do not impose a strict tree on shared concepts or repeated relations merely to match the original sketch’s visual layout.

### 5.2 An explicit interface first

For the first experiment, modules communicate through canonical finite typed values. Each operation receives only its declared value inputs and relation/module identity. The original question, tracing metadata, uncertainty fields and provenance are unavailable to that operation unless explicitly declared as additional inputs. The next module must consume the value recorded in the trace. This creates a measurable bottleneck and makes semantic interventions unambiguous.

Distribution-valued interfaces are a separate experimental condition: an arbitrary floating-point probability vector can carry additional information. Record and intervene on the entire consumed distribution, not just its displayed most-likely label. The initial types, relation identities and entity vocabulary are supplied by the benchmark; success does not show that the model discovered this structure unaided.

If a later variant passes a hidden vector as well, label that an additional information channel and test it separately. A decoded label displayed alongside an unrestricted latent vector is a readout hypothesis; it does not show that the label fully explains the computation.

Use controlled-language questions and a short answer renderer initially. All applicable neural baselines receive the same interpreter, data and supervision. Report an answer-only baseline separately to quantify the contribution of the structured interface itself.

### 5.3 Select categorical constraints deliberately

Specify the category’s objects, generators and relations before training. Path composition in a free category is automatic when a path is implemented by composing its generators; that alone is not a semantic generalization theorem.

Declare inverse laws only where the synthetic world really supports them. A country can have one capital while many cities belong to that country. In that setting, one round trip may be valid on a restricted domain while the opposite round trip is not. Many-to-one relations require suitable noninvertible operators or restricted types.

Separate **semantic relation operators** from **changes of representation frame**. The existing flat transport `Q_dst^T Q_src` depends only on endpoints and cannot be the sole definition of two different relations with identical endpoints. Optional frame alignment can surround relation-specific operators; it must not erase their identities.

Maintain an explicit conformance mode for each equation:

- enforced by parameterization in real arithmetic;
- checked numerically at a registered tolerance;
- encouraged by a soft penalty;
- not applicable.

For equal-dimensional orthogonal branch representations, strict naturality can be parameterized as `M_g^k = Q_k^T A_g Q_k` and `T_uv = Q_v^T Q_u`. This ties branches to a shared representation in different frames. Compare it with a shared-space baseline, because the constraint reduces independent freedom. It does not identify the correct meaning of `A_g`.

Report maximum unweighted defects by relation and branch pair. Router-weighted averages can hide failures in rarely used components. Never substitute orthogonality for the correct semantic treatment of a many-to-one relation.

### 5.4 Jev as a decision policy

Initially Jev predicts the next relation or a short typed program. The runtime checks composability before execution. The policy can continue, return an answer, or abstain. Avoid making the first classification irrevocable when later evidence can reveal an error.

Define each confidence event explicitly. Examples are “the selected next relation matches the reference program” and “the proposed final answer is correct.” These need separate labels and diagnostics. A route-label Brier score is not automatically an answer-correctness estimate.

Fit any calibration transform and abstention threshold on a disjoint calibration split. Calibrate the actual decision rule, including type masks, normalization, fallback and halting. Bind the calibration artifact to the interpreter, modules and policy version that generated its data. After a material module update, evaluate and, where necessary, refit calibration using the registered procedure.

Report reliability diagrams, Brier score, log loss, accepted-answer error, coverage and per-relation/path-length results. Treat shifted distributions separately. A model can appear accurate by answering almost nothing; coverage is part of the acceptance gate.

### 5.5 Recurrence with an honest state boundary

First add bounded repetition to the typed executor: current entity, program position or selected relation, visited nodes, remaining budget and continuation state are explicit. Test cycles, dead ends, early stopping and budget exhaustion.

The existing flow model can be wrapped as a separate `FlowV2Adapter` using public `rollout` and `infer` APIs. Its observations, masks, fixed candidate bases and latent coordinates remain flow-specific. Do not treat its private `_step` method as a generic language-model interface. [Current flow recurrence](../../src/universa_recurrent/neural/v2.py)

An opaque recurrent neural backend is later work. It must declare everything needed to continue: hidden states, caches, positions, context, random-generator state when relevant and module versions. Test uninterrupted versus pause/resume execution. A checksum or a summary of one tensor does not establish complete continuation state.

A maximum-step budget bounds controller iterations, assuming each operation terminates. Each operation, fallback and checker also needs an enforced resource limit or a verified finite bound. Learned halting quality and semantic correctness remain empirical. If a fallback’s internals are outside the typed trace, mark that scope in its result and count the fallback’s full cost.

### 5.6 Records and independent verification

Introduce a new schema, for example `ti_tnn_execution_v1`. Do not add fields silently to historical Lingua schemas. Existing checkers have their own guarantees and may reject unknown fields.

Record at least the input identity, world/version, program, module versions, actual typed values, route distributions, precision, operation counts, continuation decisions, abstention reason and checker version. Trace values should be generated by execution, not by a second model writing an explanation afterward.

Provide an independent finite-world checker that reads facts and the recorded program directly. It must not call the neural producer’s solver, decoder or learned decomposition. Distinguish:

- schema and type validation;
- reference semantic evaluation;
- replay of the declared computation;
- measured causal faithfulness under interventions;
- artifact integrity;
- authenticated execution provenance, only if separately supplied.

For floating-point states, preserve precision and the relevant tolerances. A trace can faithfully document an incorrect answer. Task correctness and trace faithfulness must remain separate fields.

## 6. Experiments and acceptance gates

### Experiment A: distinguish and compose relations

Create a finite world with Country, City, Airport and Operator entities. Include `capital_of`, `largest_city_of`, `airport_of` and `operator_of`, plus any additional relations needed for controlled ambiguity and cycles. Some relations should be many-to-one.

Train on atomic facts; test new compositions of facts that the model was actually given. Distinguish this from predicting new facts absent from both training and context. Use independently generated worlds, and separate training initializations from world seeds, query seeds and language templates.

Run a component condition with the correct program supplied. Follow it with an end-to-end condition using the predicted program. Include interpretation failures in the final denominator. Include invalid, unsupported and ambiguous questions, with specified expected handling.

### Experiment B: what does the structure contribute?

Compare the following as relevant to the stage:

| Control | Question it resolves |
|---|---|
| Independent symbolic lookup and execution | What does the supplied structure already make easy? |
| Dense neural operators with the same typed interface | Is sparsity needed once the interface is supplied? |
| Ordinary flat top-1 MoE | Does ordinary specialization explain the gain? |
| Shared-only half-width module | Does the common path explain the rescued MoE recipe? |
| Shared-only module matched to the full active width/work | Does adding a routed module help beyond spending that computation on the common path? |
| Shared-plus-routed modules | Do the routed parts add useful computation? |
| Relation-specific unconstrained maps | Do the declared equations help? |
| Soft penalties targeting exactly the same equations | Does exact enforcement help beyond the same information supplied softly? |
| Shared-space and untied-orthogonal variants where appropriate | Do simpler geometric explanations account for the result? |
| Fixed and shuffled routing | Is input-dependent selection useful? |
| Dense skip/small/full control | Does TI’s small-compute option explain its advantage over skip/full alone? |

Run separate capacity-matched and active-compute-matched comparisons. Record total storage, active work, initialization, output scaling, training examples and tuning allowance. Match what can be matched and name the remaining differences.

Changing the expert count also changes the unnormalized top-1 probability multiplier in the existing implementation. A fragmentation experiment must control or explicitly measure this scaling difference. An ablation that zeroes routed outputs in a trained model is diagnostic; a separately trained shared-only model is the fairer capacity control.

### Experiment C: semantic and causal trace tests

Measure intermediate type and value accuracy. Replace a consumed intermediate value with another valid value, recompute the counterfactual answer independently, and test agreement. Count only interventions whose reference answer actually changes in the primary changed-answer metric. Call this counterfactual semantic accuracy: an inaccurate downstream module can faithfully consume the replacement and still answer incorrectly.

Evaluate interface faithfulness separately. Verify that the recorded replacement is the value actually supplied to the next module, that no undeclared context bypasses the interface, and that replay under the declared module versions reproduces the execution at the registered precision. Faithful execution and correct counterfactual semantics are separate result fields.

Swap `capital_of` and `largest_city_of` while preserving endpoint types. Disable an unexecuted module’s computation while holding the program, registry identities and policy outputs fixed. Removing a module from the router’s option set can change normalization or decisions, so evaluate removal followed by rerouting as a separate system-level intervention. Remove a required module. Compare actual execution against replay. Test whether hidden side channels, shared context or unrecorded state can bypass the declared interface.

For a later continuous-state version, distinguish a semantic readout from a semantic bottleneck. Measure whether interventions on the claimed concept produce the predicted downstream effect, including no-op, wrong-type and shuffled controls. Route purity and attractive neuron labels alone are insufficient.

### Experiment D: calibrated selection and recurrence

Evaluate accepted-answer error jointly with coverage. Include path length, ambiguity and unfamiliar forms as separate strata. Repeated computation should be tested on genuine longer paths, not cancellation identities.

Compare bounded recurrent execution with fixed-depth and unrolled alternatives at comparable work. Include cycle handling, dead ends, correct early stopping, complete pause/resume state and explicit budget exhaustion. Keep a fallback’s accuracy, coverage and cost visible.

### Experiment E: optional native-feature measurements

After task competence is established, compare dense and routed variants over several widths at matched accuracy, with both active-work and total-capacity comparisons. Predefine semantic factors and their coarser groupings in the finite world. Measure held-out neuron selectivity, stability across runs, and the downstream effects of ablation or activation replacement. Analyze positive and negative activation directions explicitly where appropriate.

Condition comparisons on the inputs each module actually receives. A specialist exposed to a narrower distribution should not earn an apparent cleanliness advantage solely because its evaluation omitted the other concepts. Include shuffled-routing and random-feature controls. Report uncertainty and counterexamples; descriptive feature labels alone are not a causal explanation. This optional experiment concerns internal representations, whereas execution traces concern communication between modules.

### Suggested new confirmation bars

The numbers below are **proposed starting criteria**, not historical frozen thresholds. A bounded development pilot may revise them before a new protocol is frozen. After confirmation starts, use the registered values unchanged.

| Claim | Proposed initial bar |
|---|---|
| Structural runtime correctness | Every accepted program is well typed; invalid programs are rejected before execution; independent reference implementations agree exactly on canonical finite inputs |
| Atomic competence | At least 99% accuracy per relation in each confirmation world |
| Compositional competence | At least 95% accuracy in every registered path family, including same-endpoint alternatives and noncancelling three-step paths |
| Finite-run reliability | At least 4 of 5 fresh initializations meet the competence bars in each of at least 3 independently generated worlds |
| Intermediate semantic accuracy | At least 95% correct typed intermediate values, reported separately from final accuracy |
| Interface and trace faithfulness | Recorded finite values are the actual consumed values; replay reproduces the recorded execution with exact canonical bindings and registered numerical tolerances; disabling an unexecuted computation with program/policy fixed has no effect |
| Counterfactual semantic accuracy | At least 95% agreement with the independent reference on eligible changed-answer interventions, reported separately from interface faithfulness |
| End-to-end integration | The registered competence bars also hold with predicted programs, including interpreter failures |
| Selective answers | A proposed accepted-error upper bound of 1% with at least 90% coverage on supported inputs; specify a one-sided confidence method and sampling assumptions in advance |
| Useful inference efficiency | At most 1 percentage point accuracy loss and at least 20% lower measured total inference cost than the cheapest qualifying neural baseline, with comparable coverage and a stated tail-latency requirement |
| Local update | Intended changes succeed; no undeclared changes occur in the registered unaffected set; all mandatory structural and behavioral checks pass |

The first confirmation should cover structural execution, atomic competence, composition and traces. Do not put calibration, recurrence, evolution and a large LM into one experiment. A failed combined experiment would be difficult to diagnose.

Use the world and trained model as explicit units when comparing reliability. Do not count thousands of tokens from a handful of runs as thousands of independent training replications. For an accepted-error confidence bound, specify whether queries are independent conditional on a fixed model/world or clustered; choose the interval accordingly. Report finite-benchmark scope rather than a population safety guarantee.

### Efficiency measurement

Count interpretation, type checking, routing, shared and routed modules, transfers, state conversion, logging, fallback and rendering. Report analytical MACs alongside measured latency, peak memory and, when hardware supports it, energy. Keep training and tuning expense separate from inference expense.

Before a speed comparison, establish semantic equivalence between the scored and timed execution paths in the measured precision. Include complete route and expert decisions where they affect execution. Choose tolerances before measuring confirmation outcomes. A failure makes the speed claim untestable.

Compare with the cheapest baseline reaching the registered quality and coverage. For this new protocol, if no baseline qualifies, report the quality result and leave the proposed efficiency ratio unjudged. Preserve the different comparison rules in older frozen protocols.

## 7. Controlled module updates and evolutionary proposals

The research question is: **can a candidate update improve specified behavior while preserving specified behavior outside its declared dependencies?**

Begin with an offline copy and a single fact update or module replacement. The first mutation boundary is data-only: fact-table edits, tensor replacements or declarative assemblies of approved primitives under fixed executable code. Keep the active model immutable during each inference episode. An evolutionary controller may later propose replacements, additions, splits or pruning within a bounded search budget. It does not modify the evaluator, acceptance criteria, confirmation set or activation mechanism.

Arbitrary candidate code is outside this first experiment. If introduced later, its execution must be isolated from the checker, activation mechanism and protected evaluation state. A matching hash or an external-checker label does not itself create that isolation.

### Update record

Bind a candidate to its base revision, new code/config/parameter/buffer identities, allowed write scope, affected relations and paths, intended semantic changes, checker version, calibration version, evaluation identities and resource limits. Include optimizer state if training will continue. Account for aliases and shared storage.

Distinguish three levels of locality:

1. **Write locality:** only declared persistent state changed.
2. **Dependency locality:** only behavior reachable through that state may change.
3. **Behavioral noninterference:** registered unrelated inputs actually retain the required behavior.

The dependency closure must include shared embeddings, normalization, readouts, caches and possible later routing changes. It cannot be derived solely from the route observed on one example. A router with frozen weights can still make different decisions when its inputs change.

### Independent gates

| Gate | Required evidence |
|---|---|
| Identity and scope | Correct base revision; permitted changes only; valid shapes/dtypes; finite values; no unexpected code or shared-state changes |
| Types and laws | Valid paths and registered algebraic checks, with maximum per-relation defects |
| Intended behavior | The specified changed answers and all registered dependent compositions match the reference |
| Unaffected behavior | Exhaustive bounded checks in the finite world; separately scoped held-out tests in larger domains |
| Trace and continuation | Faithful values, replay, interventions, and compatible complete continuation state |
| Calibration and resources | Version-bound calibration, coverage, cost and memory remain within the registered limits |
| Activation | External policy accepts the candidate; stale candidates are rejected; activation is atomic; rollback restores the complete prior snapshot |

Use pass, fail and insufficient-evidence outcomes. Failure of a mandatory gate cannot be offset by a better aggregate accuracy score.

The first suite should include an ill-typed path, broken inverse, naturality violation, lawful answer-flip mutation, out-of-scope state edit, stale revision, valid local edit and failed rollback attempt. A mock effect broker can demonstrate that high confidence does not override denied authority. Pure inference over synthetic data is enough for the first experiment.

Do not treat a neural data type as an operating-system capability. If external effects are introduced later, a separately controlled execution boundary must enforce actual authority. Hashes identify artifacts; trusted activation policy determines whether they may be used.

All candidate searches consume a declared development allowance. Choose the final candidate using development data, then use an untouched confirmation set once. Repeatedly selecting candidates against the same hidden acceptance set turns it into training data.

## 8. Ordered pull request plan

The paths below are proposed unless explicitly described as existing. Each PR should include a concrete example, a runnable small case, relevant verification and a claim-ledger update.

| Order and proposed PR title | Main files | Acceptance and dependencies |
|---|---|---|
| **1. Repair TI-TNN diagnostics and accounting** | Existing `bench/hops.py`, `bench/decode.py`, `bench/ti_tnn_models.py`, `tests/test_ti_tnn_bench.py`, relevant reports and `docs/claims.md` | Stable curriculum IDs; actual K/V-only skipped projection; shared-width parameter accounting; correct diagnostic metrics and scope; preserve old run records |
| **2. Specify CDL integration and guarantee boundaries** | New `research/ti_tnn/integration_plan.md`, `integrated/README.md`, `docs/cdl_connections.md`; small existing README/index/ledger edits | Names, hypotheses, source audit, supplied-versus-learned structure and planned claims are explicit; no performance or safety promotion |
| **3. Add typed contracts and reference execution** | New `integrated/contracts.py`, `relations.py`, `registry.py`, `runtime.py`, `reference.py`; `tests/test_ti_tnn_integrated_contracts.py` | Unique relation identities; invalid-path rejection; canonical typed values; immutable module versions; budgets and exact finite reference execution |
| **4. Add learned relation operators and composition controls** | New `integrated/modules.py`, `bench/typed_world.py`, `bench/typed_composition.py`, `protocols/typed_composition_v1.md`; focused tests | Component experiment with same-endpoint relations, many-to-one maps, fresh worlds, matched controls and frozen competence gates |
| **5. Add actual execution traces and semantic interventions** | New `integrated/records.py`, `verification.py`, `interventions.py`, `bench/semantic_interventions.py`; new schema tests | Independent checker; trace replay; actual consumed values; counterfactual interventions; lawful-change counterexample correctly classified |
| **6. Add Jev planning and calibrated abstention** | New `integrated/routing.py`, `calibration.py`, `bench/typed_questions.py`, `protocols/typed_routing_v1.md` | Common interpreter supervision across baselines; predicted-program end-to-end evaluation; versioned calibration; joint risk/coverage gates |
| **7. Add bounded recurrence and complete cost measurement** | New `integrated/recurrence.py`, `costs.py`, optional `adapters/flow_v2.py`, `bench/integrated_latency.py`, `protocols/typed_recurrence_v1.md` | Real continuation-state contract; pause/resume and cycle tests; fixed/unrolled controls; timed/scored equivalence; complete cost accounting |
| **8. Add offline module replacement** | New `integrated/updates.py`, `update_checks.py`, `bench/local_updates.py`, `protocols/local_updates_v1.md` | Explicit intended and unaffected behavior; external fixed checker; immutable active revision; lawful-but-wrong candidate rejected; exact rollback |
| **9. Evaluate bounded evolutionary proposals** | New `integrated/evolution.py`, a frozen proposal-budget protocol, experiment reports | Only after the earlier gates pass; compare ordinary fine-tuning and local replacement; fixed executable primitives; untouched final confirmation |

Native-feature measurements can follow PR 5 in a separate optional PR with `bench/neuron_audit.py` and its own frozen protocol. Optional formal adapters should also be separate changes when a concrete domain supplies their required mathematical structures. Neither is a dependency of the first semantic executor.

PR 1 should include the previously identified reporting corrections: one-expert hard means are not three-hop success rates; balanced-loss runs lacked a successful matched short-budget baseline; curriculum evidence is confounded by shifting vocabulary; TI-c30’s failed seeds have variable two-hop accuracy. Retain corrected interpretations alongside historical evidence.

The current keep-K/V MAC formula assumes a skipped token projects only K/V, whereas the cached path computes and discards Q. Preserve the intended budget by implementing the K/V slice and rechecking equivalence. Derive parameter counts from instantiated shared/routed widths, including supported top-k variants.

The initial new-runtime PRs should not rewrite the classical solver, neural v2, completed continuation experiments or historical Lingua schemas. Existing public functions can be wrapped. If a new public per-step core API becomes necessary, propose it separately with a domain-specific signature and compatibility checks.

## 9. Documentation and evidence integration

Keep the root README short and follow the repository’s order: concrete example, intuition, mathematics, code, evidence and limits. Link the detailed plan rather than placing the entire mathematical stack in the opening.

### Proposed root README paragraph

> Categorical deep learning motivates an experimental extension that represents computations as typed operations with declared composition rules. The planned system combines relation-specific specialists, calibrated routing, explicit intermediate values and bounded recurrence. Separate tests will evaluate algebraic conformance, semantic behavior, causal trace faithfulness and total inference cost. Controlled module replacement is a later research goal. Existing TI-TNN benchmarks establish limited component results; they do not establish general interpretability, capability isolation or safe autonomous self-modification.

This paragraph is proposed wording, not a statement that the new runtime exists. Link it to `research/ti_tnn/integration_plan.md` and `docs/claims.md` when integrated.

### Required documentation destinations

- `research/ti_tnn/README.md`: define CDL, distinguish original typed routing from adaptive token/layer routing, link the new integration branch and retain current limits.
- `research/ti_tnn/cdl_proposal.md`: preserve the historical proposal; append a status map distinguishing structural tests, standalone transport experiments and the unrun integrated experiment.
- `docs/cdl_connections.md`: map Liquid-TQFT, HOMYMOLY, Universa, recurrence, Jev and Lingua to actual interfaces, assumptions and missing connections.
- `docs/claims.md`: one row per claim and scope, with a source/evidence link and status.
- `research/ti_tnn/protocols/`: new frozen protocols for new experiments; no retrospective rewriting of old decision rules.
- `research/ti_tnn/reports/`: include failures, incomplete inventories, deviations and untestable outcomes alongside successful results.

Suggested new ledger rows are: typed runtime implemented; relation identity preserved; categorical equations enforced/measured; semantic traces causally tested; route confidence calibrated on a specified split; total efficiency established at an operating point; local-update behavior checked in a finite domain; integrated natural-language performance; autonomous update safety. Begin unrun experimental rows as **Proposed / Not established**, with the planned falsifier. Keep universal safety and guaranteed expert cleanliness out of positive claims.

Use the existing `experiments/results/ti_tnn/` evidence mechanism. New result schemas should include protocol and code identity, configuration, all seeds/worlds/splits, deviations, precision, per-family scores, parameter and operation counts, latency validity, trace-check results and candidate-update outcomes. Extend `manifest.json`, `make_tables.py` and evidence tests so the report is regenerable. Preserve raw historical result files; corrections are new derived records or explicit errata. Do not include weights, private application data or credentials in public evidence.

## 10. Resources, decisions and stopping rules

Start with CPU reference tests and small development models. Set a bounded pilot allowance first, measure runtime, and use that measurement to freeze the confirmation budget. No GPU-hour estimate is justified before choosing the actual fixture and model. Record tuning expense and failed runs.

For the first confirmation, the suggested 3 worlds by 5 initializations means 15 runs per selected model family. Keep the primary comparison small: a dense executor, a shared-only control, ordinary MoE and the proposed typed model. Add exact-versus-soft and recurrence comparisons as separate experiments when their premises are met.

Decision rules:

- If atomic learning fails, debug competence before interpreting composition.
- If same-endpoint relations collapse, repair relation identity before adding a more elaborate router.
- If the shared-only model matches the routed recipe, retain the simpler explanation unless another required property improves.
- If interventions fail, withdraw the semantic-faithfulness claim even if final accuracy is high.
- If calibration fails, retain abstention and report the measured risk/coverage frontier.
- If the timed path differs from the scored path, leave speed untestable.
- If a local update changes undeclared behavior, reject that candidate under the locality contract.
- If exact constraints match soft or unconstrained controls, report that result without attributing the gain to categorical laws.

The first useful outcome is a reproducible finite-domain demonstration of typed execution, distinct relation identities and faithful intermediate interventions. A later positive efficiency or update result earns a separate, bounded claim. Progress should come from identifying which component improves which property.

## Primary references and repository anchors

1. Gavranović et al., [Categorical Deep Learning](https://proceedings.mlr.press/v235/gavranovic24a.html), ICML 2024.
2. Cammarata, [Neural Networks Are Already Clean](https://already-clean.goodfire.com/), working hypothesis, accessed October 18, 2026.
3. Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017.
4. Goodfire, [Goodfire and Baseten partnership](https://www.goodfire.com/blog/goodfire-baseten-partnership), October 9, 2026; separate monitoring evaluation.
5. [Repository scientific and engineering requirements](../../AGENTS.md).
6. [TI-TNN current source and status](.).
7. [Composition control results](reports/functorial_composition_controls.md).
8. [Adaptive part A v2 results](reports/adaptive_compute_part_a_v2.md).
9. [Current claim ledger](../../docs/claims.md).

## Status by pull request

| Plan PR | Status |
|---|---|
| 1. Repair TI-TNN diagnostics and accounting | Open: [#41](https://github.com/seanm27lol/universa-recurrent/pull/41) |
| 2. Specify CDL integration and guarantee boundaries | This document, [`integrated/README.md`](integrated/README.md), [`docs/cdl_connections.md`](../../docs/cdl_connections.md), the boundary tests in `tests/test_ti_tnn_guarantee_boundaries.py` and the ledger rows |
| 3. Typed contracts and reference execution | Implemented and tested: [`integrated/`](integrated/) (`contracts.py`, `relations.py`, `registry.py`, `runtime.py`, `reference.py`, `demo.py`), `tests/test_ti_tnn_integrated_contracts.py` |
| 4. Learned relation operators and composition controls | Confirmed against its frozen gates: [`integrated/modules.py`](integrated/modules.py), [`bench/typed_composition.py`](bench/typed_composition.py), [protocol](protocols/typed_composition_v1.md), [report](reports/typed_composition_v1.md), `tests/test_ti_tnn_typed_composition.py`. All six predictions hold; only atomic competence and same-endpoint discrimination are findings, the rest hold by construction or design |
| 5. Execution traces and semantic interventions | Confirmed against its frozen gates: [`integrated/records.py`](integrated/records.py), [`verification.py`](integrated/verification.py), [`interventions.py`](integrated/interventions.py), [`faults.py`](integrated/faults.py), [`bench/semantic_interventions.py`](bench/semantic_interventions.py), [protocol](protocols/semantic_interventions_v1.md), [report](reports/semantic_interventions_v1.md), `tests/test_ti_tnn_integrated_traces.py`. All seven predictions hold; only the competence replication is a finding, the rest validate the instruments |
| 6. Jev planning and calibrated abstention | Run against its frozen gates: [`integrated/routing.py`](integrated/routing.py), [`calibration.py`](integrated/calibration.py), [`bench/typed_questions.py`](bench/typed_questions.py), [protocol](protocols/typed_routing_v1.md), [report](reports/typed_routing_v1.md), `tests/test_ti_tnn_routing.py`. In distribution the plan's selective-answer bar holds (95% bound ≤ 0.0073 at coverage ≥ 0.982). Under a phrasing shift, accepted error is not bounded (59 of 100 accepted answers wrong), so that prediction failed. By section 10's rule ("if calibration fails, retain abstention and report the measured risk/coverage frontier"), abstention is retained and the shift result reported |
| 7. Bounded recurrence and complete cost measurement | Confirmed against its frozen gates: [`integrated/recurrence.py`](integrated/recurrence.py), [`costs.py`](integrated/costs.py), `route_world` and `reference_until`, [`bench/typed_recurrence.py`](bench/typed_recurrence.py), [protocol](protocols/typed_recurrence_v1.md), [report](reports/typed_recurrence_v1.md), `tests/test_ti_tnn_recurrence.py`. All seven predictions hold. Every query and stopping reason was correct, and pause/resume was exact at 53,970 pause points. Early stopping cost 22–31% of unrolling's wall time. Bookkeeping was about 20% of wall time, and about 97% with table modules |
| 8. Offline module replacement | Confirmed against its frozen gates: [`integrated/updates.py`](integrated/updates.py), [`update_checks.py`](integrated/update_checks.py), `modules.finetune`, [`bench/local_updates.py`](bench/local_updates.py), [protocol](protocols/local_updates_v1.md), [report](reports/local_updates_v1.md), `tests/test_ti_tnn_updates.py`. All six predictions hold. All 150 specialist updates passed every gate and rolled back exactly. No shared-network update was write-local. Without rehearsal, shared updates to `capital_of` corrupted `largest_city_of` in 73 of 75 cases. All 30 faulty candidates were caught as designed |
| 9. Evaluate bounded evolutionary proposals | **Deferred.** The current task has no demonstrated optimization gap: an exploratory development probe found that PR 8's default update recipe, a full supervised refit of one small relation table, learned batch changes of up to half a relation's facts perfectly. Reconsider only after the section 12 repairs, and only with a measurable objective |

## 11. Review amendments (2026-10-18)

From the review that adopted this plan:

1. **PR 1 needs its own pull request.** #39 and #40 merged before their review fixes were pushed, so the
   curriculum, K/V and parameter-count fixes reached `main` only through #41. #41 also adds `active_max`,
   which counts the top-k experts (and the bridge's two) a token can actually run.
2. **Experiment A's supplied-program condition is partly true by construction.** With canonical finite
   values passed between relation-specific modules:
   - relation identity cannot collapse, because each relation has its own module;
   - every chain error is an atomic error propagated, so composition accuracy follows from atomic
     accuracy;
   - trace faithfulness holds if the runtime passes exactly the recorded value.

   That condition therefore mainly measures atomic learning. The informative comparisons are:
   - the predicted-program condition;
   - the answer-only baseline, which quantifies what the typed interface itself contributes;
   - controls that share parameters across relations (the only ones in which same-endpoint
     relations can collapse).

   The PR 4 protocol must state which outcomes are guaranteed by construction, so that they are not
   reported as findings, as happened with the original "functorial composition" result.
3. **The lookup-task fixes may not transfer to text.** In the WikiText-103 run with them (frozen on the
   `ti-tnn-wikitext-fixed` branch and reported separately), the first seed reached validation perplexity
   57.99, against 46.94 with dense-matched initialization alone. That supports this plan's order: no further language-model run
   until the typed executor earns one.
4. **The Kimi K3 monitoring figure is not used in this repository.** It does not appear in any document
   here, and section 2 already recommends leaving it out.

## 12. Second review amendments (2026-10-18, after PRs 41–50)

An outside review of PRs 41–50 reproduced several gaps between what the implementation guarantees and what
its contracts state. Every gap below is repaired, with a regression test that fails on the earlier code
(`tests/test_ti_tnn_contract_repairs.py`):

1. **Gate results were not bound to what they checked.** One candidate's passing results could activate
   another candidate with a matching record. Records could also name the wrong world, checker or
   calibration and still pass.

   **Repair:** a `Checker` publishes private copies and checks those. It seals a result bound to the
   record, both snapshot digests, both worlds, its version, the calibration and the resource limit. A
   deployment activates only the snapshot inside a result its own checker sealed. A new mandatory gate,
   `record_context`, rejects a mismatched context, and treats a required calibration that is absent as
   insufficient evidence.
2. **Snapshots could share mutable state.** Registry identity hashed declared versions, not state. Rollback
   kept references to live objects.

   **Repair:** `snapshots.py` publishes deep, frozen copies with a digest over all behavior-relevant state,
   verified before activation and rollback. Closures, whose captured state cannot be isolated, are refused.
3. **Evaluators accepted incomplete inventories.** **Repair:** all five frozen evaluators now require
   exactly the registered cases: none missing, none repeated, none extra. Every bundled verdict still
   rebuilds unchanged.
4. **A `str` subclass inside a typed value could carry hidden data past PR 5's fix.** **Repair:** the
   runtime rebuilds every passed value from the declared domain's own plain strings.
5. **The fixed-depth control could answer where recurrence abstains** on a missing module answer.
   **Repair:** both follow the same failure semantics.
6. **"Follows from competence" overstated a construction argument.** A 99% atomic-competence threshold
   does not imply correct chains or recurrence. On route world 500, one wrong operator fact (176/177 atomic
   accuracy) leaves recurrence anywhere from 86.6% to 100% correct, depending on which fact it is. The
   argument needs exact atomic correctness. Every reported run had exactly 1.000, and recurrence and
   chain accuracy stay separately measured requirements.

**Order of work from here:**
1. These repairs.
2. **An interpreter robustness experiment.** It should separate three questions:
   - composing familiar words in new structures;
   - alternate wording whose meaning was taught;
   - abstaining when essential wording is unfamiliar.

   It should run simple controls first:
   - no threshold;
   - a raw-score threshold with valid certification;
   - calibrated probabilities with the same certification;
   - explicit lexical-novelty rejection.

   For a competence claim it should require coverage, not only bounded risk.
3. **One complete integration test:** question, interpreted bounded query, recurrence, pause, a checked
   module update, then resume under the right version policy.
4. **Evolutionary proposals (PR 9) only for a measurable objective,** such as repeated updates with a bounded
   replay allowance. They would be compared with ordinary fine-tuning, early stopping and matched random
   search, counting every proposed and rejected candidate and every check.

**Status of section 12's order (2026-10-19):**
1. **The repairs:** merged (#52).
2. **The interpreter robustness experiment:** run against its frozen protocol
   ([report](reports/interpreter_robustness_v1.md)).
   - The augmented word interpreter met the joint novelty criterion in all 15 confirmation units.
   - Taught-synonym transfer into two-step questions and three-step composition did not meet their
     criteria.
   - **The S2 diagnosis** ([report](reports/s2_diagnosis_v1.md)): S2 failures are mostly one-relation programs,
     with stopping after the first relation far more likely than in matched Base questions.
     - Wrapping the taught training questions in longer contexts did not help beyond extra exposure in a fresh,
       frozen confirmation.
     - Training on taught two-step questions removed the failure in development. That is extra supervision, not
       transfer.
3. **The full integration test:** not started.
4. **Evolutionary proposals (PR 9):** still deferred. The novelty result gives them no objective.

