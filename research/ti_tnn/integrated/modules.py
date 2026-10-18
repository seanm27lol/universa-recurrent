"""Learned relation operators for the typed runtime (integration plan PR 4).

Example: a learned `largest_city_of` module receives the typed value Country:Ardel and returns City:Port Vey,
the argmax of its logits over the City domain. Because modules exchange canonical typed values, any learned
module can replace a table module in the registry (`registry.py`) and the runtime runs it unchanged.

Five families, trained only on atomic facts (one relation applied to one entity):
- `specialists`: one small network per relation (the typed-specialist design).
- `shared_dense`: one network over (entity, relation identity) for every relation. It can confuse two
  relations with the same endpoints if it ignores the relation identity.
- `moe`: `shared_dense` with a flat top-1 mixture of experts in place of its hidden layer.
- `endpoint`: one network per (source type, target type) pair, so `capital_of` and `largest_city_of` share
  one operator. A negative control: it must confuse them, which shows the test can detect collapse.
- `answer_only`: reads the relation sequence and the start entity and predicts the answer directly, with no
  typed intermediate values. Trained on single steps, it has never seen a chain.

Outputs are restricted to the target type's domain (the type system is supplied, not learned).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import ModuleVersion, TypedValue, content_identity

FAMILIES = ("specialists", "shared_dense", "moe", "endpoint", "answer_only")
MODULE_FAMILIES = ("specialists", "shared_dense", "moe", "endpoint")


class Vocabulary:
    """Global indices for every entity of every type, and per-type index ranges."""

    def __init__(self, world):
        self.world = world
        self.types = sorted(world.types)
        self.entities, self.offset = [], {}
        for t in self.types:
            self.offset[t] = len(self.entities)
            self.entities += list(world.types[t].values)
        self.relations = sorted(world.relations)
        self.local = {t: {v: i for i, v in enumerate(world.types[t].values)} for t in self.types}

    def global_index(self, type_id: str, value: str) -> int:
        return self.offset[type_id] + self.local[type_id][value]

    def type_range(self, type_id: str) -> range:
        return range(self.offset[type_id], self.offset[type_id] + len(self.world.types[type_id].values))


def mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


class TopOneMoE(nn.Module):
    """A flat top-1 mixture of experts (Switch-style: the expert's output is scaled by its probability)."""

    def __init__(self, d_in: int, hidden: int, d_out: int, n_experts: int):
        super().__init__()
        self.router = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList([mlp(d_in, hidden, d_out) for _ in range(n_experts)])
        self.last_balance = torch.zeros(())

    def forward(self, x):
        probs = F.softmax(self.router(x), -1)
        p, e = probs.max(-1)
        out = torch.stack([expert(x) for expert in self.experts], 1)[torch.arange(len(x)), e]
        frac = torch.bincount(e, minlength=len(self.experts)).float() / len(e)
        self.last_balance = len(self.experts) * (frac * probs.mean(0)).sum()
        return out * p.unsqueeze(-1)


class SharedNet(nn.Module):
    """One network for every relation: entity embedding and relation embedding in, global logits out."""

    def __init__(self, vocab: Vocabulary, d: int, hidden: int, moe_experts: int = 0):
        super().__init__()
        self.entity = nn.Embedding(len(vocab.entities), d)
        self.relation = nn.Embedding(len(vocab.relations), d)
        self.body = (TopOneMoE(2 * d, hidden, len(vocab.entities), moe_experts) if moe_experts
                     else mlp(2 * d, hidden, len(vocab.entities)))

    def forward(self, entity_idx, relation_idx):
        return self.body(torch.cat([self.entity(entity_idx), self.relation(relation_idx)], -1))


class AnswerOnly(nn.Module):
    """Reads up to `max_len` relation tokens (padded) and the start entity; predicts the final entity."""

    def __init__(self, vocab: Vocabulary, d: int, hidden: int, max_len: int = 3):
        super().__init__()
        self.max_len = max_len
        self.entity = nn.Embedding(len(vocab.entities), d)
        self.relation = nn.Embedding(len(vocab.relations) + 1, d)  # the last index is padding
        self.body = mlp((max_len + 1) * d, hidden, len(vocab.entities))

    def forward(self, entity_idx, relation_seq):
        x = torch.cat([self.entity(entity_idx)] + [self.relation(relation_seq[:, i]) for i in range(self.max_len)], -1)
        return self.body(x)


@dataclass(frozen=True)
class LearnedModule:
    """A registry-compatible module: typed value in, argmax over the target type's domain out."""

    version: ModuleVersion
    forward: object  # callable: global source index -> global logits, or local logits for specialists
    vocab: Vocabulary
    source: str
    target: str
    local_logits: bool = False

    def apply(self, value: TypedValue) -> TypedValue | None:
        if value.type_id != self.source:
            return None
        with torch.no_grad():
            idx = self.vocab.local[self.source][value.value] if self.local_logits else \
                self.vocab.global_index(self.source, value.value)
            logits = self.forward(torch.tensor([idx]))[0]
        if not self.local_logits:
            r = self.vocab.type_range(self.target)
            logits = logits[r.start:r.stop]
        return TypedValue(self.target, self.vocab.world.types[self.target].values[int(logits.argmax())])


def module_identity(family: str, relation_id: str, net: nn.Module) -> str:
    """A content hash over the family, relation and parameters (rounded to float32 bytes' hex)."""
    params = {k: v.detach().cpu().contiguous().numpy().tobytes().hex() for k, v in sorted(net.state_dict().items())}
    return content_identity({"family": family, "relation": relation_id, "params": params})


def facts_tensors(world, vocab: Vocabulary, relation_id: str, local: bool):
    """(source indices, target indices) for one relation's atomic facts."""
    spec = world.relations[relation_id]
    items = sorted(world.facts[relation_id].items())
    if local:
        src = [vocab.local[spec.source][s] for s, _ in items]
        tgt = [vocab.local[spec.target][t] for _, t in items]
    else:
        src = [vocab.global_index(spec.source, s) for s, _ in items]
        tgt = [vocab.global_index(spec.target, t) for _, t in items]
    return torch.tensor(src), torch.tensor(tgt)


def masked_ce(logits, target, mask):
    return F.cross_entropy(logits.masked_fill(~mask, float("-inf")), target)


def type_mask(vocab: Vocabulary, target_types: list[str]) -> torch.Tensor:
    mask = torch.zeros(len(target_types), len(vocab.entities), dtype=torch.bool)
    for i, t in enumerate(target_types):
        r = vocab.type_range(t)
        mask[i, r.start:r.stop] = True
    return mask


def train_family(world, family: str, seed: int, d: int = 32, hidden: int = 64, steps: int = 1500,
                 lr: float = 1e-2, moe_experts: int = 4) -> dict:
    """Train one family on the world's atomic facts. Returns {"modules": {relation: LearnedModule}} for module
    families, or {"answer_only": (net, vocab)}; plus the final training loss."""
    torch.manual_seed(seed)
    vocab = Vocabulary(world)
    rels = vocab.relations
    if family in ("specialists", "endpoint"):
        keys = rels if family == "specialists" else sorted({(world.relations[r].source, world.relations[r].target) for r in rels})
        nets, data = {}, {}
        for key in keys:
            members = [key] if family == "specialists" else [r for r in rels if (world.relations[r].source, world.relations[r].target) == key]
            spec = world.relations[members[0]]
            nets[key] = nn.Sequential(nn.Embedding(len(world.types[spec.source].values), d),
                                      mlp(d, hidden, len(world.types[spec.target].values)))
            xs, ys = zip(*(facts_tensors(world, vocab, r, local=True) for r in members))
            data[key] = (torch.cat(xs), torch.cat(ys))
        params = [p for n in nets.values() for p in n.parameters()]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            loss = sum(F.cross_entropy(nets[k](x), y) for k, (x, y) in data.items()) / len(data)
            opt.zero_grad()
            loss.backward()
            opt.step()
        modules = {}
        for r in rels:
            key = r if family == "specialists" else (world.relations[r].source, world.relations[r].target)
            net = nets[key].eval()
            version = ModuleVersion(f"{r}_{family}", r, family, module_identity(family, r, net), "float32")
            modules[r] = LearnedModule(version, net, vocab, world.relations[r].source, world.relations[r].target, True)
        return {"modules": modules, "loss": loss.item()}
    xs, rs, ys, tts = [], [], [], []
    for ri, r in enumerate(rels):
        x, y = facts_tensors(world, vocab, r, local=False)
        xs.append(x); ys.append(y); rs.append(torch.full_like(x, ri)); tts += [world.relations[r].target] * len(x)
    x, rel, y = torch.cat(xs), torch.cat(rs), torch.cat(ys)
    mask = type_mask(vocab, tts)
    if family == "answer_only":
        net = AnswerOnly(vocab, d, hidden)
        seq = torch.full((len(x), net.max_len), len(rels))
        seq[:, 0] = rel
        forward = lambda: net(x, seq)
    else:
        net = SharedNet(vocab, d, hidden, moe_experts if family == "moe" else 0)
        forward = lambda: net(x, rel)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        loss = masked_ce(forward(), y, mask)
        if family == "moe":
            loss = loss + 0.01 * net.body.last_balance
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    if family == "answer_only":
        return {"answer_only": (net, vocab), "loss": loss.item()}
    modules = {}
    for ri, r in enumerate(rels):
        fwd = (lambda idx, ri=ri: net(idx, torch.full_like(idx, ri)))
        version = ModuleVersion(f"{r}_{family}", r, family, module_identity(family, r, net), "float32")
        modules[r] = LearnedModule(version, fwd, vocab, world.relations[r].source, world.relations[r].target, False)
    return {"modules": modules, "loss": loss.item(), "net": net}


def finetune(world, family: str, base: dict, seed: int, relations: tuple[str, ...] = (), steps: int = 300,
             lr: float = 3e-3, rehearse: bool = True) -> dict:
    """Plan PR 8: continue training a trained family on `world`, an updated copy of the world it learned.

    - `specialists`: only the networks of `relations` are copied and trained, on all of their relation's
      facts; every other module object is reused as it is.
    - `shared_dense`: the one shared network is copied and trained, so every module changes. With
      `rehearse`, it trains on every relation's facts; without, only on the facts of `relations`, the same
      data the specialists get.

    Returns the same shape as `train_family`."""
    import copy
    torch.manual_seed(seed)
    vocab = Vocabulary(world)
    if family == "specialists":
        modules = dict(base["modules"])
        for r in relations:
            net = copy.deepcopy(modules[r].forward).train()
            x, y = facts_tensors(world, vocab, r, local=True)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for _ in range(steps):
                loss = F.cross_entropy(net(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            version = ModuleVersion(f"{r}_{family}", r, family, module_identity(family, r, net), "float32")
            modules[r] = LearnedModule(version, net, vocab, world.relations[r].source, world.relations[r].target, True)
        return {"modules": modules, "loss": None}
    if family != "shared_dense":
        raise ValueError(f"no fine-tuning for {family}")
    net = copy.deepcopy(base["net"]).train()
    xs, rs, ys, tts = [], [], [], []
    for ri, r in enumerate(vocab.relations):
        if not rehearse and r not in relations:
            continue
        x, y = facts_tensors(world, vocab, r, local=False)
        xs.append(x); ys.append(y); rs.append(torch.full_like(x, ri)); tts += [world.relations[r].target] * len(x)
    x, rel, y = torch.cat(xs), torch.cat(rs), torch.cat(ys)
    mask = type_mask(vocab, tts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        loss = masked_ce(net(x, rel), y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    modules = {}
    for ri, r in enumerate(vocab.relations):
        fwd = (lambda idx, ri=ri: net(idx, torch.full_like(idx, ri)))
        version = ModuleVersion(f"{r}_{family}", r, family, module_identity(family, r, net), "float32")
        modules[r] = LearnedModule(version, fwd, vocab, world.relations[r].source, world.relations[r].target, False)
    return {"modules": modules, "loss": loss.item(), "net": net}


def answer_only_predict(net: AnswerOnly, vocab: Vocabulary, world, path: tuple[str, ...], start: str) -> str:
    """The answer-only model's prediction for a path from a start entity, restricted to the path's target type."""
    rels = vocab.relations
    seq = torch.full((1, net.max_len), len(rels))
    for i, r in enumerate(path):
        seq[0, i] = rels.index(r)
    source, target = world.relations[path[0]].source, world.relations[path[-1]].target
    with torch.no_grad():
        logits = net(torch.tensor([vocab.global_index(source, start)]), seq)[0]
    r = vocab.type_range(target)
    return world.types[target].values[int(logits[r.start:r.stop].argmax())]
