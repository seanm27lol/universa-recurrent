"""The supplied, lexicon-free typed grammar of the factored interpreter (protocols/factored_interpreter_v1.md).

**Example.** "name the company operating the airfield of country_29_city_0 ."
- **The entity** is the one token that names an entity of the world: `country_29_city_0`.
- **The style** is prefix: the words before the entity contain "the".
- **The segments** are the maximal spans that begin with "the" and contain no other "the", read from the
  first "the": "the company operating", then "the airfield of".
- **The execution order** is innermost first, so the segments come out as ("the airfield of",
  "the company operating"). The program's first relation applies to the entity.
- **The frame** is the words before the first "the" ("name") and the final "." It is discarded.

**What it does not do.** It never consults a lexicon. Which relation a segment means, or whether it means
one at all, is left to the grounding step: a learned model, a training-derived lookup, or the reference
lexicon. Held-out phrases are segmented like any other.

**The rules:**
- **Entity.** Exactly one token must name an entity of the world (`entities`). With none, the question is
  refused as an unknown entity; with more than one, as malformed.
- **Prefix style:** "the" occurs before the entity, and "'s" does not occur after it. The segments begin at
  the first "the" and run to the entity; each "the" starts a new one. Words after the entity are frame.
- **Possessive style:** "'s" occurs after the entity, and "the" does not occur before it. The first token
  after the entity must be "'s". Each "'s" starts a segment, which runs up to the next "'s" or a final "?"
  or ".". Words before the entity are frame.
- **Malformed:** both styles at once, neither, words between the entity and its first "'s", or more than
  `MAX_SEGMENTS` segments.
- **Execution order:** innermost first for prefix questions (the reverse of reading order), left to right
  for possessive ones.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_SEGMENTS = 3  # the maximum program length (routing.MAX_STEPS)
PREFIX, POSSESSIVE = "prefix", "possessive"
TERMINAL = ("?", ".")


@dataclass(frozen=True)
class Segmentation:
    """The grammar's reading of one question. `status` is "ok" or the refusal reason."""

    status: str
    style: str | None = None
    entity: str | None = None
    entity_position: int | None = None
    segments: tuple[tuple[str, ...], ...] = ()  # in execution order: the first applies to the entity
    spans: tuple[tuple[int, int], ...] = ()     # token [start, end) of each segment, in execution order

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def segment(text: str, entities) -> Segmentation:
    """Segment a whitespace-tokenized question. `entities` is the set of entity names of the world."""
    tokens = text.split()
    positions = [i for i, t in enumerate(tokens) if t in entities]
    if not positions:
        return Segmentation("unknown entity")
    if len(positions) > 1:
        return Segmentation("malformed: several entities")
    e = positions[0]
    before, after = tokens[:e], tokens[e + 1:]
    prefix, possessive = "the" in before, "'s" in after
    if prefix and possessive:
        return Segmentation("malformed: mixed styles", entity=tokens[e], entity_position=e)
    if not (prefix or possessive):
        return Segmentation("malformed: no segment", entity=tokens[e], entity_position=e)
    spans = []
    if prefix:
        start = before.index("the")
        bounds = [i for i in range(start, e) if tokens[i] == "the"] + [e]
        spans = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
        spans.reverse()  # innermost (nearest the entity) first
        style = PREFIX
    else:
        if after[0] != "'s":
            return Segmentation("malformed: words between the entity and its first 's", entity=tokens[e],
                                entity_position=e)
        i, end = e + 1, len(tokens)
        while i < end and tokens[i] not in TERMINAL:
            j = i + 1
            while j < end and tokens[j] != "'s" and tokens[j] not in TERMINAL:
                j += 1
            spans.append((i, j))
            i = j
        style = POSSESSIVE
    if len(spans) > MAX_SEGMENTS:
        return Segmentation(f"malformed: more than {MAX_SEGMENTS} segments", style=style, entity=tokens[e],
                            entity_position=e)
    return Segmentation("ok", style, tokens[e], e, tuple(tuple(tokens[a:b]) for a, b in spans), tuple(spans))
