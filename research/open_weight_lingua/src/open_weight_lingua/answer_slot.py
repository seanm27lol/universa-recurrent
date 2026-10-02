"""The AV's own answer slot: the quoted candidates in its "Final token" paragraph.

Example (a real Gemma-3-27B description of `x = 9; y = 8; y = 3; y = y + 3;
x = x + 1`, asked for x, whose answer is 10):

    Final token "\\n" ends a final answer label ("the result is..."),
    immediately expecting a numeric answer like "10" or "11" to specify ...

The slot starts at the first line that begins with ``Final token`` and runs to
the next blank line. Its candidates are the quoted canonical integers 0..19
that follow the first ``expect…``/``likely``/``like`` in that slot; above they
are 10 and 11. A slot with one distinct candidate value is ``eligible``, with
several it is ``ambiguous``, and with none it is ``absent``.

This parser only reads text. A candidate is the AV's guess about the next
token; it is never evidence that the value is true of an activation, and it is
a different object from the frozen Phase Two rule's variable-state statements
(``text_edits``), which this module does not change.
"""

from dataclasses import dataclass
import hashlib
import json
import re

RULE_VERSION = "AS-1.0.0"
STATUSES = ("eligible", "absent", "ambiguous")

_SLOT_START = re.compile(r"(?m)^[ \t]*Final token")
_SLOT_END = re.compile(r"\n[ \t]*\n")
_CUE = re.compile(r"(?<![A-Za-z])(?:expect[A-Za-z]*|likely|like)(?![A-Za-z])", re.I)
_CANDIDATE = re.compile(r'"(?P<value>0|1[0-9]|[1-9])(?![0-9])')
_STANDALONE = r"(?<![0-9A-Za-z_]){value}(?![0-9])"


@dataclass(frozen=True)
class Slot:
    status: str
    candidates: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]

    @property
    def lead(self) -> int | None:
        return self.candidates[0] if self.candidates else None


def parse(description: str) -> Slot:
    """Locate the answer slot and its quoted candidates, with absolute spans."""
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    start = _SLOT_START.search(description)
    if start is None:
        return Slot("absent", (), ())
    end = _SLOT_END.search(description, start.start())
    stop = end.start() if end else len(description)
    cue = _CUE.search(description, start.start(), stop)
    if cue is None:
        return Slot("absent", (), ())
    hits = [
        (int(match.group("value")), match.span("value"))
        for match in _CANDIDATE.finditer(description, cue.start(), stop)
    ]
    if not hits:
        return Slot("absent", (), ())
    values = tuple(value for value, _ in hits)
    status = "eligible" if len(set(values)) == 1 else "ambiguous"
    return Slot(status, values, tuple(span for _, span in hits))


def _check_value(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 19:
        raise ValueError("value must be a canonical integer in 0..19")


def edit_slot(description: str, new_value: int) -> str | None:
    """Rewrite every candidate in an eligible slot; None when not eligible."""
    _check_value(new_value)
    slot = parse(description)
    if slot.status != "eligible":
        return None
    out = description
    for start, stop in sorted(slot.spans, reverse=True):
        out = out[:start] + str(new_value) + out[stop:]
    return out


def edit_everywhere(description: str, new_value: int) -> str | None:
    """Rewrite the slot's value at every standalone occurrence in the text.

    Secondary edit: it also changes the value where the description repeats
    it outside the slot (e.g. "The answer is 10"), and so can also change an
    unrelated number that happens to be equal.
    """
    _check_value(new_value)
    slot = parse(description)
    if slot.status != "eligible":
        return None
    pattern = re.compile(_STANDALONE.format(value=slot.lead))
    return pattern.sub(str(new_value), description)


def rule_definition() -> dict:
    """Canonical serialization of the rule, for manifest locking."""
    return {
        "version": RULE_VERSION,
        "slot_start": _SLOT_START.pattern,
        "slot_end": _SLOT_END.pattern,
        "cue": _CUE.pattern,
        "cue_flags": "IGNORECASE",
        "candidate": _CANDIDATE.pattern,
        "standalone": _STANDALONE,
        "value_range": [0, 19],
        "eligible": "exactly one distinct candidate value",
    }


def rule_sha256() -> str:
    canonical = json.dumps(rule_definition(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


RULE_SHA256 = rule_sha256()
