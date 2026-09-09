# Temporal Lingua: describe changes, not just snapshots

**Status: proposed experiment, not an implemented NLA, J-lens, or semantic decoder.**
The existing model and its checkable dual outputs remain unchanged.

## Familiar example

A mechanic compares several readings before deciding what caused a vibration.
One note saying what stayed plausible and what changed can be more useful than
four disconnected descriptions. But a plausible note is not proof that it
faithfully represents how the mechanic reached the decision.

The analogous research question is: can one short, interpretable description
preserve useful information about several successive latent states?

```text
short window of observed states
               |
        shared describer
               |
    bounded, readable description
               |
  reconstructor queried by site
        /      |      \
     state 1 state 2 state 3
               |
  reconstruction and intervention tests
```

## Grounding in existing work

Natural language autoencoders (NLAs) use an activation verbalizer to describe an
activation, and an activation reconstructor to recover an activation from that
text. The published method uses reconstruction as its training signal but
separately evaluates explanation reliability. Its authors explicitly identify
single-layer inputs, confabulation and inference cost as limitations; they
already propose multi-layer inputs. Thus "train across layers" alone is not a
novel contribution. See [the primary NLA paper](https://transformer-circuits.pub/2026/nla/index.html).

The [Jacobian-lens/J-space paper](https://arxiv.org/abs/2607.15495) is a different
method: averaged downstream sensitivities yield vocabulary-related readouts.
It does not equate every hidden feature with readable content. Our numerical
flow network has no language vocabulary/unembedding, so its two cycle
coordinates are not automatically J-space. A Jacobian of its numerical outputs
could be a sensitivity diagnostic, not a semantic dictionary.

## Specify which axis "multiple vectors" means

| Axis | Meaning | Relevant distinction |
|---|---|---|
| Layer `l` | Different blocks in a network | Layer changes can change the representation basis |
| Recurrent step `t` | Repeated use of a shared block | Same weights do not guarantee the same state distribution |
| Token/position `i` | Different sequence positions | The current flow task has no language tokens |
| Candidate `k` | Competing structural hypotheses | Candidates are alternatives, not successive time steps |

A reconstructor should receive explicit site identifiers, rather than silently
pooling all four axes. The first toy experiment should vary recurrent step and
candidate only. Cross-layer language models are a later, distinct test.

## Mathematical interface

Let `s_(t,k)` be the state of candidate `k` after update `t`. For a short window
`W`, the describer `D` produces a bounded description `z = D(W)`. The reconstructor
`R(z,t,k)` returns a reconstruction of the state at that site. For a language
model, the query would additionally specify layer and token position.

For an online readout at time `t`, use only a prefix ending at `t`. A retrospective
window may include later states, but must be labeled post-hoc and cannot be used
as evidence that an earlier explanation predicted the future.

The reconstructor gets the description and public site tags, not the original
activation, input prompt or an unreported skip connection. If it also receives
context, label that a conditional reconstruction and compare with a context-only
baseline. Continuing the model may retain original context; that is not license
for the reconstructor to silently solve the task from it.

The hypothesis is better description efficiency at a matched total budget, not
that a larger reconstructor is inherently interpretable. Compare the same
covered sites and total description tokens/bits. Report describer, reconstructor,
collection, storage and checking costs separately. A shared description may
save repeated text generation while still making reconstruction more expensive.

## Two levels of Lingua, never silently combined

1. **Instrumented mathematical trace.** In the current flow task we know the
   definitions of coordinates, candidate estimates, weights and constraints.
   A typed record can describe those operations by construction. This is not
   discovery of the meanings of arbitrary neurons.
2. **Learned activation explanation.** A proposed description of less transparent
   internal features requires reconstruction, generalization and causal evidence.
   Good reconstruction alone can reward a hidden code shared by the describer
   and reconstructor rather than human-understandable meaning.

A multiple-vector output is useful as a test target, not a faithfulness guarantee.
Do not call a numerical bottleneck an NLA unless an actual language bottleneck is
implemented and evaluated.

## Testing faithfulness without inventing a story

First validate the boundary where a recurrent computation can be resumed. The
complete state can include candidate coordinates, route probabilities, fixed
input context, masks and control state. Reconstructing a pooled output alone is
not a complete-state replacement.

Replace a reconstructed state at **one cut point**, keep the other required
variables explicit, and continue the frozen model for the original remaining
budget. Compare downstream numerical estimates and structural decisions with
the original continuation. Do not repeatedly inject future reconstructed states:
that would supply the continuation rather than test it.

This tests whether the representation is behaviorally sufficient at that cut.
To claim semantics, additionally edit one proposed mathematical field and test
its predicted effect with matched perturbations and alternative explanations.
Neither result authenticates remote execution. See the established
[causal-abstraction framework](https://www.jmlr.org/papers/v26/23-0058.html).

## Practical first study

Keep the current solver frozen. Compare per-state descriptions, one window
summary, and a keyframe-plus-change description. Add no-content, shuffled-pair,
raw-coordinate and equal-budget numerical compression controls. Split by whole
input problem/trajectory, not individual adjacent states. Fit normalizers and
choose budgets only on training/calibration splits.

Start with known mathematical fields. In a later learned-language phase,
measure changes under meaning-preserving paraphrases, description swaps and
controlled semantic edits. Hold the final response and future states out of
online describer inputs. Measure failures as well as successful examples.

The detailed [draft protocol](../experiments/TEMPORAL_LINGUA_PROTOCOL_DRAFT.md)
separates reconstruction, sufficiency, meaning and cost. No new GPU training is
required just to read or audit this proposal.
