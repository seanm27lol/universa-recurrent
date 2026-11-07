"""The constraint-only diagnostic of the grounding-interface study (post hoc; CPU): the saved v1 controllers decoded
with the original v1 policy, and with only the same-reference ON restriction of grounding v2 added.

**Example.** A v1 controller decoding `... INNER JOIN label_categories AS T2 ON T2.label_id = ...` may choose
`T2.label_id` again as the second operand: `ON T2.label_id = T2.label_id` relates no two sources. Under v2 that choice
is not legal (`actions._NodeFrame.push_scalar` sets the second operand slot's `exclude` when the comparison is in an
ON clause and its first operand is a column). Here that one rule is applied to a v1 machine. Nothing else of v2 is
present: no schema encoder, no item features, no pointer relations. The weights, the backbone states and every
other legality rule are the v1 policy's.

**What it does not do.** It does not reproduce or replace the frozen v2 results; it is a post-hoc diagnostic of the
restriction's effect on v1 decoding. `RestrictedV1Machine` re-implements the v2 condition beside the frozen code,
which is not edited; a test checks that the two agree.
"""
from __future__ import annotations

import contextlib

from . import actions as A
from . import ir


class RestrictedV1Machine(A.Machine):
    """A v1 machine plus the v2 same-reference ON restriction, and nothing else of v2."""

    def push(self, frame):
        if isinstance(frame, A._ExprSlot) and frame.exclude is None and self.stack:
            parent = self.stack[-1]
            if isinstance(parent, A._NodeFrame) and parent.op in A.COMPARE_ACTIONS and frame.parent_op == parent.op \
                    and frame.child_index == 1 and len(parent.children) == 1 \
                    and isinstance(parent.children[0], ir.Column) and parent.ctx.clause == "ON":
                frame.exclude = parent.children[0]   # the same condition as grounding v2's push_scalar
        super().push(frame)


@contextlib.contextmanager
def machine_class(cls):
    """Decode with `cls` in place of `actions.Machine` (the decoder constructs `A.Machine` by name)."""
    original = A.Machine
    A.Machine = cls
    try:
        yield
    finally:
        A.Machine = original


def self_joins(plan) -> int:
    """SELECTs in which one base table appears as two or more FROM sources (a self-join)."""
    if plan is None:
        return 0
    n = 0
    for node in ir.walk(plan):
        if isinstance(node, ir.Select):
            names = [f.source.table.lower() for f in node.from_ if isinstance(f.source, ir.TableSource)]
            n += len(names) != len(set(names))
    return n
