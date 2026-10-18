"""Run the README example: which company operates the airport serving the largest city of Ardel?

    python research/ti_tnn/integrated/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import contracts as C  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402


def main() -> None:
    world = W.ardel_world()
    registry = G.table_registry(world)
    for steps in (("largest_city_of", "airport_of", "operator_of"), ("capital_of", "airport_of", "operator_of"),
                  ("capital_of", "operator_of")):
        program = C.Program(steps, world.types["Country"].value("Ardel"), "Operator")
        out = X.execute(program, registry, world.types, C.Budget(8))
        print(" -> ".join(steps))
        for s in out.trace:
            print(f"  {s.relation_id:16s} {s.consumed.value} -> {s.produced.value}   ({s.module_id} {s.module_identity[:12]})")
        expected = R.reference_answer(world, steps, "Ardel")
        print(f"  status {out.status}; answer {out.answer.value if out.answer else None}; reference {expected}"
              + (f"; problems: {list(out.problems)}" if out.problems else "") + "\n")


if __name__ == "__main__":
    main()
