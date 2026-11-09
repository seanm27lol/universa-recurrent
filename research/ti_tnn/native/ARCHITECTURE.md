# Native causal LM: architecture differences and design contract

**Status: implemented and locally construction-tested.** The model and training/evaluation paths are in
[`model.py`](model.py), [`train.py`](train.py), and [`evaluate.py`](evaluate.py). The last full native
run passed **208 tests**; see the [reproduction interface](README.md) and
[`tests/test_ti_tnn_native_model.py`](../../../tests/test_ti_tnn_native_model.py). The full repository
suite is still pending. These are construction and execution checks, not learning, quality, efficiency,
or WSP evidence; no WSP phase has begun.

This is a new randomly initialized native causal language model, not the project's first trained weights.
Earlier adaptive TI-TNN models already trained **202M-parameter** variants and a later **129M-parameter** variant. They did not meet the frozen
quality/efficiency bars: the 202M models remained outside the 2% quality band, with a smaller dense model
winning at the 50% cap, and the 129M model reached 57.21 validation perplexity after the lookup fixes
([adaptive report](../reports/adaptive_compute.md),
[initialization control](../reports/adaptive_compute_init_control.md),
[WikiText follow-up](../reports/adaptive_compute_wikitext_fixed.md)). Those failures are evidence about
those adaptive recipes, not results for this design.

The frozen-Qwen C4/T4 controllers train small SQL controllers over cached Qwen states; the semantic-repair
and database-evidence revision studies use Qwen proposals and feedback. They are separate systems and
supply no weights, caches, targets, or results to this model
([controller architecture](../scaling/docs/ARCHITECTURE.md),
[semantic revision](../scaling/docs/PROTOCOL_semantic_proposal_repair.md),
[database-evidence revision](../scaling/docs/PROTOCOL_database_evidence_revision.md)).

## Example and intuition

For a prompt such as `The capital of Ardel is`, the model learns embeddings and predicts the next token
with its own output head. At every token it performs exactly four depth updates. In the recurrent arms,
the same Transformer cell is revisited rather than stacking four different cells. Four explicit slot
vectors carry information between those updates. Ordinary causal attention, not the slots, carries
information between token positions.

For example, information from an earlier mention of `Ardel` can reach the current token directly through
attention; it does not have to be copied through every earlier token's slots. In C/D, one of four
specialists runs at each depth update. D additionally tells the interface and router which slot is which;
C uses a generic, untyped interface. There is no early exit: an easy token and a hard token both make four
passes.

The reference points are standard causal Transformers (Vaswani et al., 2017,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)), depth-weight-shared Universal Transformers
(Dehghani et al., 2018, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)), sparsely gated experts
(Shazeer et al., 2017, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)), and straight-through
estimators (Bengio, Léonard and Courville, 2013,
[arXiv:1308.3432](https://arxiv.org/abs/1308.3432)). This note specifies their implemented combination;
it does not claim that the combination learns or helps.

## The four modes

All modes use the same tokenizer, embedding dimension `d`, slot width `d_s`, four slots, state-probe
shapes, causal mask, four updates, and untied output-head shape. Token/position embeddings, slot
initializers, and other corresponding weights start from matched random draws; each mode then trains its
own copy. Except for the differences below, the attention shape, transform input/output shapes, residual
connections, initialization rule, data, and training recipe must match.

| Mode | Core at passes `r = 1,2,3,4` | Transform | Slot interface and routing | Comparison |
|---|---|---|---|---|
| **A: dense** | Four unique core cells `Core_1,...,Core_4` | One dense transform per cell | Shared untyped interface; no router | A/B holds four state updates fixed and isolates recurrent weight sharing |
| **B: shared** | One shared `Core`, called four times | One shared dense transform | Same as A | B/C isolates generic top-1 specialist organization |
| **C: generic** | B's shared attention and interface core, four passes | Four generic experts; exactly one selected per token and pass | Shared untyped projections; h-only router | C/D isolates type-conditioned interfaces and routing |
| **D: typed** | Same shared-core arrangement as C | Same four-expert shapes and selection rule as C | Type-specific slot readers/writers; slot-conditioned router | Direct test of the added typing |

Thus A is not a one-pass or shallower control, and B/C/D do not get a fifth hidden pass. A/B tests
weight-tied depth recurrence, not the mere presence of four nonlinear updates. B/C changes the dense
transform into a top-1 expert bank and its necessary router. C/D keeps the expert bank, optional route
labels, and active expert shape fixed while changing the slot interface and router input. Parameter and
compute matching are handled separately below; no single comparison silently controls both.

## State and computation contract

Let `x_t` be the token id at position `t`, `E` the randomly initialized token-embedding table, and `p_t`
its position embedding. Define

```text
u_t       = E[x_t] + p_t                         in R^d
h_t^0     = u_t                                  in R^d
s_{t,k}^0 = I_k u_t                              in R^{d_s},  k = 1,...,4.
```

`h_t^r` is the main residual-stream state after pass `r`; `s_{t,k}^r` is slot `k` after that pass. The
initializers `I_k` and all other randomly initialized weights are trained jointly, including `E`, the
core cells, experts, routers, interface projections, probes, and output head. Each `s_{t,k}^0` depends
only on the current token and position. Slots never recurrently read another token's slots.

### Direct h paths and causal attention

At pass `r`, with `LN` denoting the configured pre-normalization,

```text
a_t^r = CausalAttn_r(LN(h_{<=t}^{r-1}))_t
q_t^r = h_t^{r-1} + a_t^r.
```

Attention has its usual Q/K/V and output projections and reads only positions `<= t`. In A, each
`CausalAttn_r` belongs to its unique cell. In B/C/D, the same attention weights are used at every pass.
This is the **h attention bypass**: cross-time information can move through `h` without passing through a
slot or a selected expert.

The residual update below likewise lets `h` bypass the transform. After pass four,

```text
logits_t = U LN(h_t^4),
```

where `U` is a randomly initialized vocabulary output matrix separate from `E`; `U` and `E` must not be
tied. This is the **direct h-to-output path**. Probes and execution-derived targets are never needed to
produce logits.

### Untyped and typed readers/writers

The h reader is `H q_t^r`. The slot read `z_t^r` and transform input `v_t^r` are

```text
A/B/C:  z_t^r = (1/4) sum_k S s_{t,k}^{r-1}
D:      z_t^r = (1/4) sum_k (S_k s_{t,k}^{r-1} + c_k)
        v_t^r = [H q_t^r ; z_t^r].
```

`[.;.]` means concatenation. A/B/C use one projection `S` for every slot and no slot-type code in the
expert interface. D uses a learned projection `S_k` and code `c_k` for each slot. `H`, `S`, `S_k`, and
`c_k` are shared across depth; only D's interface is type-specific.

The selected transform output is `g_t^r`:

```text
A:      g_t^r = G_r(v_t^r)                     four unique dense transforms
B:      g_t^r = G(v_t^r)                       one shared dense transform
C/D:    g_t^r = F_{e*_{t,r}}(v_t^r)            one selected expert, defined below.
```

`G_r`, `G`, and every `F_e` have the same input/output dimensions and matched dense-transform hidden
shape. The h writer and residual update are

```text
h_t^r = q_t^r + O_h g_t^r.
```

Every slot is read and written on every pass; routing selects an expert, not a slot and not whether to
update. The slot writer is

```text
A/B/C:  s_{t,k}^r = s_{t,k}^{r-1} + O_s   [g_t^r ; S s_{t,k}^{r-1}]
D:      s_{t,k}^r = s_{t,k}^{r-1} + O_{s,k}[g_t^r ; S_k s_{t,k}^{r-1} + c_k].
```

`O_h` is the h writer; `O_s` is the shared untyped slot writer; `O_{s,k}` are D's type-specific slot
writers. These projections are part of the cell's charged work. State probes `Q_k` have identical shapes
in every mode and read `s_{t,k}^r` without changing it. They are auxiliary outputs, not inference inputs
or shortcuts to the LM head.

### Why cached decoding is feasible

The only cross-time path is causal attention. Slots and their initialization are local to a token;
recurrence is over `r`, not over text positions. A cached decoder therefore stores the K/V tensors for
each of the four passes for each prior token. For a new token, it locally constructs `h_t^0` and all four
`s_{t,k}^0`, then runs passes one through four. At pass `r`, it attends to the cached pass-`r` K/V tensors
and appends the new token's pass-`r` K/V tensors. Prior tokens do not resume or change their slots.

The implementation follows this dependency. Local construction tests exercise full-versus-cached logits
and routes, and [`reconstruct.py`](reconstruct.py) replays hidden/slot traces at the restored checkpoint
dtype. CUDA profiling also requires cache correctness before its timing can be marked valid. These checks
establish execution equivalence within stated tolerances, not a measured speedup or learning evidence.

## Top-1 routing and training losses

C/D have four experts indexed by `e = 1,...,4`. C's router reads only `H q_t^r`; D's router reads
`[H q_t^r ; z_t^r]`, including the type-coded slot summary. Routers are shared across depth. For router
logits `ell_{t,r}`, compute

```text
p_{t,r} = softmax(ell_{t,r})                  in float32
e*      = argmax_e p_{t,r,e}                  fixed lowest-index tie break
p_st    = p + detach(one_hot(e*) - p)
g_t^r   = p_st[e*] F_{e*}(v_t^r).
```

The softmax and its backward computation remain float32 even under mixed-precision autocast. In the
forward pass, `p_st` is exactly one-hot and the selected expert has coefficient one. In backward, the
selected path supplies the straight-through derivative of `p`; the float32 softmax couples that gradient
to all router logits. Only `F_{e*}` is invoked. Implementations must not evaluate all four experts and
mask their outputs afterward. Unselected experts receive no task gradient from this routing decision.

For `N` routing decisions in a batch, let `f_e` be the detached fraction selecting expert `e`, and
`P_e = (1/N) sum p_{t,r,e}`. With four experts, the load-balancing term is

```text
L_balance = 4 sum_e f_e P_e.
```

It is active for C/D even when no route labels exist, including all ordinary text. It encourages use of
the expert bank; it is not a semantic target and does not show that experts acquire different meanings.

The total training loss is

```text
L = L_next_token
    + lambda_balance L_balance               (C/D only)
    + lambda_route   L_route                 (C/D, labelled structured decisions only)
    + lambda_state   L_state                 (all modes, labelled structured states only).
```

`L_next_token` is ordinary next-token cross-entropy. The architecture introduces no new optimizer: one
fixed existing LM optimizer recipe, update schedule, clipping rule, and batch policy must apply across
arms, with any loss weights fixed before learning comparisons.

### Structured supervision boundary

Only structured examples may carry auxiliary targets. A supplied reference program is run by the typed
executor; the resulting execution trace provides the specialist-family route for designated steps and
the discrete values in four execution-state fields. A fixed alignment maps those trace events to
`(token position t, depth pass r, slot k)` tuples. Nothing is inferred from a Qwen generation.

For C/D, `L_route` is cross-entropy between `p_{t,r}` and the execution-derived route label on the
labelled mask. C and D must receive byte-identical optional route-label and mask tensors; when labels
are absent, both use the same unlabeled rule. A/B have no route loss.

For every mode, linear probe `Q_k` predicts the execution-derived state value for its designated field,
and `L_state` is the corresponding masked cross-entropy over `Q_k s_{t,k}^r`. Probe architecture,
targets, masks, and loss weight are the same across modes. These are auxiliary training readouts, not a
claim that the raw slot equals a symbolic value.

At inference, the model receives token ids and positions plus its own attention cache. Route labels,
state targets, reference programs, execution traces, probe targets, and Qwen weights/caches/targets are
never model inputs, cache entries, attention biases, or decoding controls. Post-forward scoring against
held-out targets is evaluation, not an input channel. Ordinary text never receives execution-derived
route/state labels.

There is deliberately **no adaptive stopping, skip/small/full choice or compute-cap objective, sheaf
bridge, Qwen initialization or distillation, or new optimizer** in any mode.

## Names, probes, and interventions

Slot indices (`1,...,4`) are addresses. Calling a slot `entity`, `relation`, `plan`, or `memory` is only
a naming convenience and does not assign that meaning. A state target encourages a probe-visible
regularity on labelled structured examples; it does not establish that the model uses that value to
compute its answer, that the same code is used on text, or that a specialist has a unique function.

Probe accuracy measures readout under its stated distribution, not causal use. Zeroing, swapping, or
patching a slot can establish sensitivity to that intervention under tested conditions. It cannot by
itself prove a faithful or complete explanation: information may be redundant, distributed across `h`
and other slots, or changed by an off-distribution intervention. Route traces, state traces, probe
accuracy, intervention effects, and task accuracy are separate evidence and must be reported separately.

## Parameters and compute are separate contracts

**Total parameters** count unique stored trainable tensors. A shared B/C/D cell is counted once even
though it runs four times; all four experts count toward C/D total parameters whether or not selected.
`E` and `U` count separately. For C/D, a token's actual active set is the union of experts selected over
its four passes. The configuration's `active` value is only the backward-compatible nominal one-expert
count, not a route prediction; [`cost.py`](cost.py) computes route-union bounds and observed values. See
the [corrected count tables](README.md#corrected-parameter-and-mac-counts) and
[`configs/engineering.json`](configs/engineering.json).

**Forward compute** counts work actually performed. For a decoded token, report at least

```text
MACs_m = sum_{r=1}^4 [
             MACs_attention(m,r) + MACs_transform(m,r)
             + MACs_router(m,r) + MACs_h_reader/writer(m,r)
             + MACs_slot_readers/writers(m,r)
         ] + MACs_output_head.
```

A/B run their dense transform four times. B's shared weights do not make later calls free. C/D charge
one selected expert per pass, not the whole bank, but charge every router and all four passes of
attention and interface work. The untied output head is charged once per scored or decoded token.
Attention includes projections and QK/AV work at the actual context length; cached decoding still pays
new-token projections and attention over the retrieved cache. Probe and auxiliary-loss work is reported
when run, not hidden inside inference. Non-MAC dispatch, top-1 selection, synchronization, cache traffic,
and kernel-launch costs belong in latency measurements.

Parameter matching and compute matching are different experiments. Matching total parameters generally
changes widths or expert counts and therefore changes active compute; matching active compute does not
remove C/D's larger stored expert bank. Any controlled comparison must say which quantity is matched and
still report the other. MACs are not wall time; an efficiency claim requires end-to-end timing with valid
cached-versus-full execution, including routing, attention, head, and interface overhead.

## Construction checks, not learning evidence

The implemented local suite covers this construction checklist:

1. **Shape and call contract:** `h` and all four slots have the declared shapes; exactly four passes run
   for every token; A has four distinct core parameter sets, B/C/D share theirs; `E` and `U` are untied.
2. **Local state and bypasses:** changing future tokens cannot change earlier states or logits; slot
   initialization reads only the current token/position; attention and the LM head remain connected to
   `h` when slot outputs are blocked in a wiring test.
3. **Selected-only routing:** instrumented calls show exactly one expert per `(t,r)` in C/D; forward
   coefficients are one-hot; an isolated decision gives no gradient to unselected experts; router
   gradients are finite and router softmax runs in float32 under autocast.
4. **Cache equivalence:** full-sequence and cached incremental decoding agree on routes, logits, `h`, and
   all slot trajectories under a fixed dtype and tolerance, with the first mismatch localized rather than
   averaging it away.
5. **Supervision and accounting:** text has no execution targets and C/D still apply load balancing;
   structured route/mask tensors are identical for C/D; inference cannot access auxiliary targets; shared
   parameters are counted once while their four passes and every router/attention/head/interface cost are
   counted.

These checks can establish that the intended computation is wired and executed. They are not learning
evidence. Learning requires separately reported language quality, structured-task performance, multiple
training seeds, and the relevant parameter-, compute-, and latency-matched controls. None of those
results is claimed here.
