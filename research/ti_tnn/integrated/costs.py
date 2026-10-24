"""Cost accounting for the typed executor (integration plan section 6, "Efficiency measurement"; PR 7).

Per query, it counts:
- **Module applications,** and their multiply-accumulates (MACs), worked out from layer shapes. A learned
  specialist costs the MACs of its linear layers. A table module costs none: it is a lookup.
- **Type checks:** one per module output.
- **Continuation state:** its size as JSON, and how many times it was copied.

It also measures wall time, split into time spent inside modules and everything else ("bookkeeping": type
checks, state copies and records). The split matters: for tiny modules, bookkeeping can cost more than
the computation it records.

Training and tuning are not counted here; they are reported separately. Timing wraps the module objects
themselves, so the timed path calls exactly the code the scored path calls. `same_result` checks that the
two produce the same final state.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Iterator


def linear_macs(net) -> int:
    import torch.nn as nn
    return sum(m.in_features * m.out_features for m in net.modules() if isinstance(m, nn.Linear))


def module_macs(module) -> int | None:
    """MACs of one application, or None when the module's structure cannot be read off."""
    from .registry import TableModule
    if isinstance(module, TableModule):
        return 0
    forward = getattr(module, "forward", None)
    if forward is not None and hasattr(forward, "modules"):
        return linear_macs(forward)
    net = getattr(forward, "net", None)  # one relation's view of a shared network (`modules.SharedForward`)
    if net is not None and hasattr(net, "modules"):
        if any(type(m).__name__ == "TopOneMoE" for m in net.modules()):
            return None  # which expert runs depends on the input
        return linear_macs(net)
    return None


def application_costs(state: dict, registry) -> dict:
    """Counts for one finished run from its recorded applications."""
    apps = state["applications"]
    macs = [module_macs(registry.module(a["relation_id"])) for a in apps]
    return {"module_applications": len(apps), "type_checks": len(apps),
            "macs": None if any(m is None for m in macs) else sum(macs),
            "state_bytes": len(json.dumps(state, sort_keys=True))}


@contextmanager
def timed_modules(registry) -> Iterator[dict]:
    """Accumulate the wall time spent inside every module object's apply()."""
    clock = {"module_ns": 0, "calls": 0}
    seen, wrapped = set(), []
    for m in registry.modules.values():
        if id(m) in seen:
            continue
        seen.add(id(m))
        original = type(m).apply.__get__(m)

        def timed(value, original=original):
            t0 = time.perf_counter_ns()
            try:
                return original(value)
            finally:
                clock["module_ns"] += time.perf_counter_ns() - t0
                clock["calls"] += 1
        object.__setattr__(m, "apply", timed)
        wrapped.append(m)
    try:
        yield clock
    finally:
        for m in wrapped:
            object.__delattr__(m, "apply")


def timed(fn, registry) -> tuple[object, dict]:
    """Run fn() once with module timing; return its result and {total_ns, module_ns, bookkeeping_ns, calls}."""
    with timed_modules(registry) as clock:
        t0 = time.perf_counter_ns()
        result = fn()
        total = time.perf_counter_ns() - t0
    return result, {"total_ns": total, "module_ns": clock["module_ns"], "bookkeeping_ns": total - clock["module_ns"],
                    "calls": clock["calls"]}
