"""A conservative grammar-based interpreter: a control for the interpreter robustness study
(protocols/interpreter_robustness_v1.md). It is given an explicit grammar prior, which it declares.

It knows:
- the relation lexicon: the Base and taught-synonym phrases, in prefix ("the capital of X") and possessive
  ("X 's capital") form;
- one structural rule of the controlled language. A prefix noun phrase starts at the first "the" and is a
  sequence of lexicon phrases followed by the entity. A possessive noun phrase is the entity followed by
  "'s" phrases.

It does not know the held-out or novel word lists, or any test category.

**How it parses:**
- Anything outside the noun phrase (frames, fillers, punctuation) is wrapper material, and is allowed to
  be unfamiliar.
- If the noun phrase cannot be parsed, or parses more than one way, the question is refused
  (`unsupported`).
- An unparsed relation phrase, such as an unknown relation word, therefore causes a refusal, while an
  unknown frame does not.
- A frame that itself contained "the" would be misread. The controlled language has none.

Its parse goes through the same policy and runtime as the learned interpreters (`routing.propose`).
"""
from __future__ import annotations

from . import routing as J


class GrammarInterpreter:
    def __init__(self, prefix_phrases: dict, possessive_phrases: dict):
        """`prefix_phrases`: relation -> templates like "the capital of {x}"; `possessive_phrases`: relation ->
        templates like "{x} 's capital"."""
        self.prefix = [(tuple(t.split("{x}")[0].split()), r) for r, ts in prefix_phrases.items() for t in ts]
        self.possessive = [(tuple(t.split("{x}")[1].split()), r) for r, ts in possessive_phrases.items() for t in ts]

    def _prefix_parses(self, toks, i):
        """Every way to read toks[i:] as phrases then an entity (one token); returns (relations outer-first, entity
        position) for each reading."""
        out = []
        if i < len(toks):
            out.append(([], i))  # the entity itself
        for words, rel in self.prefix:
            if tuple(toks[i:i + len(words)]) == words:
                for rels, pos in self._prefix_parses(toks, i + len(words)):
                    out.append(([rel] + rels, pos))
        return out

    def parse(self, text: str) -> J.Parse:
        toks = text.split()
        refuse = lambda: J.Parse("unsupported", (), toks[0] if toks else "", 1.0, (), 1.0)
        if "'s" in toks:  # possessive: entity, then "'s" phrases
            e = toks.index("'s") - 1
            if e < 0:
                return refuse()
            rels, i = [], e + 1
            while i < len(toks) and toks[i] == "'s":
                matches = [(w, r) for w, r in self.possessive if tuple(toks[i:i + len(w)]) == w]
                longest = max((len(w) for w, _ in matches), default=0)
                matches = [(w, r) for w, r in matches if len(w) == longest]
                if len(matches) != 1:
                    return refuse()
                rels.append(matches[0][1])
                i += longest
            return J.Parse("program", tuple(rels), toks[e], 1.0, (), 1.0) if rels else refuse()
        if "the" not in toks:
            return refuse()
        start = toks.index("the")
        readings = [(rels, pos) for rels, pos in self._prefix_parses(toks, start) if rels]
        # a reading must stop at the entity: the entity is the last noun-phrase token; what follows is wrapper
        readings = [(rels, pos) for rels, pos in readings if toks[pos] not in ("the", "of", "'s")]
        longest = max((len(r) for r, _ in readings), default=0)
        readings = [(r, p) for r, p in readings if len(r) == longest]
        if len(readings) != 1:
            return refuse()
        rels, pos = readings[0]  # phrases are consumed contiguously from "the", so nothing in between is skipped
        return J.Parse("program", tuple(reversed(rels)), toks[pos], 1.0, (), 1.0)
