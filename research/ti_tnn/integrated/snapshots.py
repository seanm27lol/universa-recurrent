"""Published inference snapshots: private, content-addressed copies of a registry (integration plan section 7).

A `Registry` freezes its mapping from relations to modules, but the modules themselves can hold mutable
state: network parameters, arrays, tables. Its identity hashes the declared version strings, not that
state. So a registry object is not a snapshot. Training code that keeps a reference to a network, or two
registries that share a module, can change what a "checked" registry does without changing its identity.

`publish(registry)` makes a `Snapshot`, in three steps:
- **Copy.** It deep-copies every module, so the snapshot shares no mutable state with anything outside
  it. Shared structure inside the registry stays shared: two relations reading one network still read
  one copy.
- **Freeze.** It puts copied networks in inference mode, with gradients off.
- **Digest.** It records a digest of all behavior-relevant state (`state_digest`): parameter and buffer
  values, arrays, tables, object fields, and the code and captured variables of functions.

`Snapshot.verify()` recomputes the digest. Activation and rollback refuse a snapshot that no longer
matches, so a change made through a retained reference is detected rather than silently used.

Within a torch module, the state is its registered parameters and buffers plus its plain attributes;
unregistered tensor attributes are treated as per-call diagnostics.

**Limits.** This is an in-process integrity check, not a security boundary: code in the same process can
still reach into a snapshot, and verification only detects the change afterwards. The digest covers the
state reachable from the modules' fields. Behavior that depends on anything outside them, such as global
variables or the outside world, is not captured; modules are required not to depend on such state.
"""
from __future__ import annotations

import copy
import hashlib
import types as pytypes
from dataclasses import dataclass
from typing import Mapping

from .registry import Registry

_ATOMIC = (type(None), bool, int, float, complex, str, bytes)


def state_digest(obj) -> str:
    """A SHA-256 over the behavior-relevant state reachable from `obj` (see the module docstring)."""
    memo: dict[int, tuple[object, str]] = {}  # id -> (object kept alive, digest)

    def walk(x) -> str:
        if type(x) in _ATOMIC:
            return hashlib.sha256(f"{type(x).__name__}:{x!r}".encode()).hexdigest()
        key = id(x)
        if key in memo:
            return memo[key][1]
        memo[key] = (x, "<cycle>")
        d = hashlib.sha256(_describe(x, walk).encode()).hexdigest()
        memo[key] = (x, d)
        return d

    return walk(obj)


def _describe(x, walk) -> str:
    cls = type(x)
    name = f"{cls.__module__}.{cls.__qualname__}"
    if isinstance(x, (str, int, float, bytes)):  # a subclass of a built-in: its value and anything attached
        base = next(b for b in (str, int, float, bytes) if isinstance(x, b))
        return f"{name}|{base(x)!r}|{walk(getattr(x, '__dict__', {}))}"
    try:
        import torch
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            data = t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes()
            return f"tensor|{t.dtype}|{tuple(t.shape)}|{hashlib.sha256(data).hexdigest()}"
        if isinstance(x, torch.nn.Module):
            # registered parameters and buffers are the state; unregistered tensor attributes (such as a mixture's
            # last balance loss) are per-call diagnostics, rewritten on every forward pass, and are left out
            extra = {k: v for k, v in vars(x).items()
                     if not k.startswith("_") and k != "training" and not isinstance(v, torch.Tensor)}
            return f"{name}|{x!r}|training={x.training}|{walk(dict(x.state_dict()))}|{walk(extra)}"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return f"ndarray|{x.dtype}|{x.shape}|{hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}"
        if isinstance(x, np.generic):
            return f"npscalar|{x.dtype}|{x.tobytes().hex()}"
    except ImportError:
        pass
    if isinstance(x, (list, tuple)):
        return f"{name}|" + ",".join(walk(e) for e in x)
    if isinstance(x, (set, frozenset)):
        return f"{name}|" + ",".join(sorted(walk(e) for e in x))
    if isinstance(x, Mapping):
        return f"{name}|" + ",".join(sorted(f"{walk(k)}={walk(v)}" for k, v in x.items()))
    if isinstance(x, pytypes.FunctionType):
        code = x.__code__
        cells = [c.cell_contents if _cell_full(c) else "<empty>" for c in (x.__closure__ or ())]
        return (f"function|{x.__module__}.{x.__qualname__}|{code.co_code.hex()}|{walk(code.co_consts)}|"
                f"{walk(code.co_names)}|{walk(x.__defaults__)}|{walk(x.__kwdefaults__)}|{walk(cells)}")
    if isinstance(x, pytypes.MethodType):
        return f"method|{walk(x.__func__)}|{walk(x.__self__)}"
    if isinstance(x, pytypes.CodeType):
        return f"code|{x.co_code.hex()}|{walk(x.co_consts)}"
    if isinstance(x, (type, pytypes.ModuleType, pytypes.BuiltinFunctionType)):
        return f"ref|{getattr(x, '__module__', '')}.{getattr(x, '__qualname__', getattr(x, '__name__', ''))}"
    if hasattr(x, "identity") and type(x).__name__ == "World":
        return f"world|{x.identity}"  # immutable; its identity is a content hash of types, relations and facts
    fields = dict(getattr(x, "__dict__", {}))
    for slot in getattr(cls, "__slots__", ()):
        if hasattr(x, slot):
            fields[slot] = getattr(x, slot)
    return f"{name}|{walk(fields)}"


def _cell_full(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def _networks(obj, seen=None):
    """Every torch module reachable through plain object fields (for freezing)."""
    try:
        import torch
    except ImportError:
        return []
    seen = set() if seen is None else seen
    if id(obj) in seen or type(obj) in _ATOMIC:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return [obj]
    children = obj.values() if isinstance(obj, Mapping) else obj if isinstance(obj, (list, tuple)) else \
        vars(obj).values() if hasattr(obj, "__dict__") and not isinstance(obj, (type, pytypes.ModuleType)) else []
    return [n for c in children for n in _networks(c, seen)]


def _closures(obj, seen=None) -> list[str]:
    """Functions with captured variables reachable from `obj`; copying cannot isolate what they capture."""
    seen = set() if seen is None else seen
    if id(obj) in seen or type(obj) in _ATOMIC:
        return []
    seen.add(id(obj))
    if isinstance(obj, pytypes.FunctionType):
        return [obj.__qualname__] if obj.__closure__ else []
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return []
    except ImportError:
        pass
    children = obj.values() if isinstance(obj, Mapping) else obj if isinstance(obj, (list, tuple)) else \
        vars(obj).values() if hasattr(obj, "__dict__") and not isinstance(obj, (type, pytypes.ModuleType)) else []
    return [f for c in children for f in _closures(c, seen)]


@dataclass(frozen=True)
class Snapshot:
    """A private copy of a registry and the digest of its state when it was published."""

    registry: Registry
    digest: str

    def verify(self) -> bool:
        return state_digest(self.registry) == self.digest


def publish(registry: Registry) -> Snapshot:
    """Copy, freeze and digest a registry. Refuses modules whose state hides in closures."""
    hidden = [f for m in registry.modules.values() for f in _closures(m)]
    if hidden:
        raise ValueError(f"module state captured in closures cannot be isolated: {sorted(set(hidden))}")
    memo: dict = {}
    modules = {rid: copy.deepcopy(m, memo) for rid, m in registry.modules.items()}
    for net in {id(n): n for m in modules.values() for n in _networks(m)}.values():
        net.eval()
        net.requires_grad_(False)
    frozen = Registry(registry.relations, modules)
    return Snapshot(frozen, state_digest(frozen))


def digest_of(registry: Registry) -> str:
    """The digest the registry would be published under."""
    return publish(registry).digest
