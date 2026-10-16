# TI-TNN: does functorial structure let learned relations compose? (frozen brief)

**Status: FROZEN 2026-10-16 (UTC), before any run of this experiment.** Branch
`ti-tnn-functorial-composition`, based on `main` at `fafd54d`. A timing check
(50 steps per variant) ran beforehand and produced no reported number.

## The question

The [categorical proposal](../cdl_proposal.md) predicts one concrete benefit:
- with relation maps that obey composition laws exactly, a model trained only
  on single relations ("capital of France", "size of Paris") should answer
  chains it never saw ("size of the capital of France");
- an unconstrained model with the *same* step-by-step interface should do this
  less reliably.

The comparison isolates the constraint, not the interface. Every step model
chains relation maps in the same way; only what the maps are allowed to be
differs.

## Design (`research/ti_tnn/composition.py`)

**World.** 256 countries, each with its own capital (a random bijection), 8
regions (per country) and 6 sizes (per capital). This is a category with four
objects (entity kinds) and four generating relations:
- `capital_of: country → capital`;
- `country_of = capital_of⁻¹`;
- `region_of: country → region`;
- `size_of: capital → size`.

**Training.** Single-relation facts only: all 1,024. Full batch, Adam,
learning rate 3e-3, 3,000 steps, width d = 128, 5 seeds (0–4), on the GB10.

**Variants.** The step models share typed embeddings (one table per entity
kind, the typed branches), one map per relation applied in order, and tied
dot-product scoring against the target kind's embeddings:

| Variant | Relation maps |
|---|---|
| `functorial` | transports of a flat sheaf on the four branches (`cdl.FunctorialSheafBridge`): orthogonal, with an exact inverse law and path independence |
| `soft` | unconstrained d×d maps plus a penalty (weight 1.0) pulling `capital_of` and `country_of` toward being inverses |
| `linear` | unconstrained d×d maps |
| `mlp` | unconstrained d → 2d → d networks |
| `transformer` | a 2-layer transformer reading the relation tokens and the entity, with the same tied scoring; it never sees two relation tokens in training |

**Scoring.** Exact accuracy on every query of each family (256 per path,
except where the source kind is smaller):

| Family | Paths | Chance |
|---|---|---|
| one-hop (fit) | the 4 trained relations | 1/256, 1/8 or 1/6 |
| **two-hop** (the test) | size of the capital of a country; region of the country of a capital | 1/6 and 1/8 |
| three-hop | size_of ∘ capital_of ∘ country_of (capital → size); region_of ∘ country_of ∘ capital_of (country → region) | 1/6 and 1/8 |
| round trip | country_of ∘ capital_of; capital_of ∘ country_of | 1/256 |

For `functorial` a round trip is the identity map by construction. It is
correct whenever an entity's embedding scores highest against itself, so
round trips are reported but are not evidence.

## Predictions (recorded before any run; all one-sided; means over 5 seeds)

1. Every variant fits single relations: one-hop accuracy of at least 0.95.
2. **Composition.** `functorial` scores at least 0.80 on two-hop chains.
3. **The constraint matters.** `functorial` exceeds each unconstrained step
   model (`soft`, `linear`, `mlp`) on two-hop chains by at least 0.10.
4. **Longer chains.** `functorial` scores at least 0.70 on three-hop chains.
5. **Generic baseline.** `transformer` scores at most 0.50 on two-hop chains.
6. **Exact against soft.** On round trips, `functorial` scores at least as high
   as `soft`, and `soft` at least as high as `linear`.

If 2 or 3 fails, exact functorial structure does not buy composition here,
beyond what the shared interface already gives. Every family, variant and seed
is reported whatever happens.

## Limits

- One synthetic world, with a single relation per pair of entity kinds.
  Real text supplies no category.
- Orthogonal maps make many-to-one relations (`region_of`, `size_of`)
  norm-preserving; dot-product scoring absorbs that.
- One width, one schedule; nothing tuned after seeing results.
- This tests compositional generalization, not compute efficiency.

**Run:** `python research/ti_tnn/composition.py --device cuda`.
