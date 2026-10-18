"""Functorial composition: train on single relations, test on chains never seen in training.

protocols/functorial_composition.md is the frozen design. The world has four entity kinds (the
objects of a small category) and four relations (its generating morphisms):

    capital_of: country -> capital     (a bijection)
    country_of: capital -> country     (its inverse)
    region_of:  country -> region      (many to one)
    size_of:    capital -> size        (many to one)

Every variant is trained only on single-relation facts ("capital_of France", "size_of Paris")
and tested on chains ("size_of capital_of France", three-step chains, round trips).

Every variant except the transformer has the same interface: one embedding table per entity kind
(the typed branches), one map per relation applied step by step, and answers scored by dot product
with the target kind's embeddings (tied). They differ only in the relation maps:

    functorial   transports of a flat sheaf on the four branches (cdl.FunctorialSheafBridge):
                 relation src->dst is T(src->dst) = Q_dst^T Q_src, orthogonal, so the inverse law
                 holds exactly and chains are path-independent by construction
    soft         unconstrained d x d maps plus a penalty pulling capital_of and country_of toward
                 being inverses
    linear       unconstrained d x d maps
    mlp          unconstrained two-layer networks d -> 2d -> d
    transformer  a 2-layer transformer reading the relation tokens and the entity, scoring with the
                 same tied embeddings; it never sees two relation tokens in training

Controls added after the first run (protocols/functorial_composition_controls.md). The functorial
model's scores are E_s Q_s^T Q_t E_t^T = Z_s Z_t^T with Z_k = E_k Q_k^T, so it has the capacity of
typed embeddings in one shared space; and the relations here form a tree, so flatness adds nothing
beyond tying the inverse pair. These variants separate the pieces:

    shared        identity maps: typed embeddings in one shared space, scored by dot product
    ortho_tied    an orthogonal map per relation, with country_of = capital_of^T
    ortho_untied  an independent orthogonal map per relation (the inverse pair not tied)
    soft_full     unconstrained maps, softly penalized toward every equation the exact model obeys:
                  each map orthogonal, and country_of the inverse of capital_of

    python research/ti_tnn/composition.py --device cuda    # 5 seeds, every variant
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cdl():
    """research/ti_tnn/cdl.py, loaded by path."""
    if "ti_tnn_cdl" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ti_tnn_cdl", Path(__file__).with_name("cdl.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["ti_tnn_cdl"] = module
        spec.loader.exec_module(module)
    return sys.modules["ti_tnn_cdl"]


KINDS = ("country", "capital", "region", "size")
RELATIONS = {"capital_of": ("country", "capital"), "country_of": ("capital", "country"),
             "region_of": ("country", "region"), "size_of": ("capital", "size")}
# Query families scored after training. Paths are in application order.
FAMILIES = {
    "one_hop": [["capital_of"], ["country_of"], ["region_of"], ["size_of"]],
    "two_hop": [["capital_of", "size_of"], ["country_of", "region_of"]],
    "three_hop": [["country_of", "capital_of", "size_of"], ["capital_of", "country_of", "region_of"]],
    "round_trip": [["capital_of", "country_of"], ["country_of", "capital_of"]],
}
VARIANTS = ("functorial", "soft", "linear", "mlp", "transformer", "shared", "ortho_tied", "ortho_untied", "soft_full")
MATRIX_KINDS = ("functorial", "soft", "linear", "shared", "ortho_tied", "ortho_untied", "soft_full")


class World:
    """n countries, each with its own capital (a random bijection), a region and its capital's size."""

    def __init__(self, n_countries=256, n_regions=8, n_sizes=6, facts_seed=1234):
        g = torch.Generator().manual_seed(facts_seed)
        self.sizes = {"country": n_countries, "capital": n_countries, "region": n_regions, "size": n_sizes}
        self.capital_of = torch.randperm(n_countries, generator=g)          # country -> capital
        self.country_of = torch.argsort(self.capital_of)                     # capital -> country
        self.region_of = torch.randint(n_regions, (n_countries,), generator=g)  # country -> region
        self.size_of = torch.randint(n_sizes, (n_countries,), generator=g)      # capital -> size

    def apply(self, relation: str, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, relation)[x]

    def answer(self, path: Sequence[str], x: torch.Tensor) -> torch.Tensor:
        for r in path:
            x = self.apply(r, x)
        return x

    def queries(self, path: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every source entity of the path's first relation, with its answer."""
        src = torch.arange(self.sizes[RELATIONS[path[0]][0]])
        return src, self.answer(path, src)

    def training_facts(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [(r, *self.queries([r])) for r in RELATIONS]


def check_path(path: Sequence[str]) -> Tuple[str, str]:
    for f, g in zip(path, path[1:]):
        if RELATIONS[f][1] != RELATIONS[g][0]:
            raise ValueError(f"{g} cannot follow {f}")
    return RELATIONS[path[0]][0], RELATIONS[path[-1]][1]


class StepModel(nn.Module):
    """Typed embeddings + one map per relation, applied step by step; tied dot-product scoring."""

    def __init__(self, world: World, d: int, kind: str):
        super().__init__()
        self.kind = kind
        self.embed = nn.ParameterDict({k: nn.Parameter(torch.randn(n, d) / d ** 0.5) for k, n in world.sizes.items()})
        if kind == "functorial":
            self.bridge = _cdl().FunctorialSheafBridge(len(KINDS), d, d)  # only its transports are used
        elif kind in ("soft", "linear", "soft_full"):
            self.maps = nn.ParameterDict({r: nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d)) for r in RELATIONS})
        elif kind in ("ortho_tied", "ortho_untied"):
            free = [r for r in RELATIONS if not (kind == "ortho_tied" and r == "country_of")]
            self.ortho = nn.ModuleDict({r: _cdl().orthogonal_square(d) for r in free})
        elif kind == "shared":
            self.register_buffer("eye", torch.eye(d))
        elif kind == "mlp":
            self.nets = nn.ModuleDict({r: nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for r in RELATIONS})
        else:
            raise ValueError(kind)

    def relation_matrix(self, r: str) -> torch.Tensor:
        """The d x d map of relation r (not defined for mlp)."""
        if self.kind == "functorial":
            src, dst = RELATIONS[r]
            return self.bridge.transport(KINDS.index(src), KINDS.index(dst))
        if self.kind == "shared":
            return self.eye
        if self.kind == "ortho_tied" and r == "country_of":
            return self.ortho["capital_of"].weight.T
        if self.kind in ("ortho_tied", "ortho_untied"):
            return self.ortho[r].weight
        return self.maps[r]

    def step(self, r: str, h: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.nets[r](h)
        return h @ self.relation_matrix(r).T

    def forward(self, path: Sequence[str], x: torch.Tensor) -> torch.Tensor:
        src, dst = check_path(path)
        h = self.embed[src][x]
        for r in path:
            h = self.step(r, h)
        return h @ self.embed[dst].T  # logits over the target kind's entities

    def penalty(self) -> torch.Tensor:
        if self.kind not in ("soft", "soft_full"):
            return torch.zeros((), device=next(self.parameters()).device)
        d = self.maps["capital_of"].shape[0]
        a, b = self.maps["capital_of"], self.maps["country_of"]
        eye = torch.eye(d, device=a.device, dtype=a.dtype)
        inverse = ((b @ a - eye).pow(2).sum() + (a @ b - eye).pow(2).sum()) / d
        if self.kind == "soft":
            return inverse
        return inverse + sum((m.T @ m - eye).pow(2).sum() for m in self.maps.values()) / d


class TransformerModel(nn.Module):
    """Reads [relation tokens..., entity]; scores the last position against the target kind's embeddings."""

    def __init__(self, world: World, d: int, layers: int = 2, heads: int = 4, max_len: int = 4):
        super().__init__()
        self.kind = "transformer"
        self.embed = nn.ParameterDict({k: nn.Parameter(torch.randn(n, d) / d ** 0.5) for k, n in world.sizes.items()})
        self.rel = nn.Embedding(len(RELATIONS), d)
        self.pos = nn.Embedding(max_len + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.LayerNorm(d)

    def forward(self, path: Sequence[str], x: torch.Tensor) -> torch.Tensor:
        src, dst = check_path(path)
        dev = x.device
        rel = self.rel(torch.tensor([list(RELATIONS).index(r) for r in path], device=dev)).unsqueeze(0).expand(len(x), -1, -1)
        seq = torch.cat([rel, self.embed[src][x].unsqueeze(1)], dim=1)
        seq = seq + self.pos(torch.arange(seq.shape[1], device=dev))
        h = self.out(self.encoder(seq)[:, -1])
        return h @ self.embed[dst].T

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.rel.weight.device)


def build(world: World, variant: str, d: int) -> nn.Module:
    return TransformerModel(world, d) if variant == "transformer" else StepModel(world, d, variant)


def train(model: nn.Module, world: World, args) -> List[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    dev = next(model.parameters()).device
    facts = [(r, x.to(dev), y.to(dev)) for r, x, y in world.training_facts()]
    log = []
    for step in range(args.steps):
        loss = sum(F.cross_entropy(model([r], x), y) for r, x, y in facts) / len(facts)
        total = loss + args.soft_weight * model.penalty()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": loss.item(), "penalty": model.penalty().item()})
    return log


@torch.no_grad()
def evaluate(model: nn.Module, world: World) -> Dict:
    out = {}
    dev = next(model.parameters()).device
    for family, paths in FAMILIES.items():
        per_path = {}
        for path in paths:
            x, y = (t.to(dev) for t in world.queries(path))
            pred = model(path, x).argmax(-1)
            per_path["/".join(path)] = {"accuracy": (pred == y).float().mean().item(), "n": len(x),
                                        "chance": 1.0 / world.sizes[RELATIONS[path[-1]][1]]}
        out[family] = {"accuracy": statistics.fmean(p["accuracy"] for p in per_path.values()), "paths": per_path}
    if getattr(model, "kind", None) in MATRIX_KINDS:
        a, b = model.relation_matrix("capital_of"), model.relation_matrix("country_of")
        eye = torch.eye(a.shape[0], device=a.device)
        out["inverse_defect"] = ((b @ a - eye).norm() / a.shape[0] ** 0.5).item()
        out["orthogonality_defect"] = max(((m.T @ m - eye).norm() / a.shape[0] ** 0.5).item()
                                          for m in (model.relation_matrix(r) for r in RELATIONS))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--countries", type=int, default=256)
    p.add_argument("--regions", type=int, default=8)
    p.add_argument("--sizes", type=int, default=6)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--soft-weight", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_composition"))
    return p.parse_args(argv)


def run(args) -> Dict:
    world = World(args.countries, args.regions, args.sizes)
    runs = []
    for seed in args.seeds:
        for variant in args.variants:
            torch.manual_seed(seed)
            model = build(world, variant, args.d).to(args.device)
            t0 = time.time()
            log = train(model, world, args)
            res = evaluate(model, world)
            runs.append({"seed": seed, "variant": variant, "train_seconds": round(time.time() - t0, 1),
                         "final_loss": log[-1]["loss"], "eval": res, "log": log})
            print(f"  seed {seed} {variant:11s} one-hop {res['one_hop']['accuracy']:.3f}  two-hop {res['two_hop']['accuracy']:.3f}  "
                  f"three-hop {res['three_hop']['accuracy']:.3f}  round-trip {res['round_trip']['accuracy']:.3f}", flush=True)
    return {"world": {"sizes": world.sizes}, "families": FAMILIES, "runs": runs}


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    result = run(args)
    result["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    result["environment"] = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "composition.json").write_text(json.dumps(result, indent=2))
    print("\nmean over seeds (accuracy)")
    for v in args.variants:
        rows = [r["eval"] for r in result["runs"] if r["variant"] == v]
        print(f"  {v:11s} " + "  ".join(f"{fam} {statistics.fmean(r[fam]['accuracy'] for r in rows):.3f}" for fam in FAMILIES))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
