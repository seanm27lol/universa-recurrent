"""Categorical structure for TI-TNN: functorial specialist branches and a flat sheaf bridge (experimental).

Specification: research/ti_tnn/cdl_proposal.md. In short:

* A QueryCategory is a finitely presented category: objects (entity kinds), generating morphisms
  (relations such as capital_of: country -> capital) and inverse pairs (country_of = capital_of^-1).
* A CDLSpecialistBranch carries a functor from that category into R^d: every generator g gets an
  orthogonal matrix M_g on the branch's stalk, the partner of an inverse pair gets M_g^T, and a path
  of generators acts by the product of their matrices. With one-hot step choices, composition,
  identity and inverse laws hold exactly by construction, for any parameter values.
* A FunctorialSheafBridge gives every branch k one orthogonal restriction map Q_k into a shared edge
  frame (a flat O(d) connection sheaf on the branch graph). Transport across the bridge is
  T(u->v) = Q_v^T Q_u, so T(u->u) = I, T(v->w) T(u->v) = T(u->w) and T(v->u) = T(u->v)^T hold
  exactly, for any parameter values, and the coboundary Q_u s_u - Q_v s_v vanishes exactly on
  transported sections.
* One law cannot be built in: the bridge's transports should be natural transformations between the
  branch functors, T(u->v) M^u_g = M^v_g T(u->v). naturality_loss measures the defect, weighted per
  branch pair by the router's probabilities, so the constraint is strongest where the router uses
  the pair.

Orthogonal parametrization (torch.nn.utils.parametrizations.orthogonal) also removes the trivial
zero set a soft penalty would otherwise have (M = 0 is not orthogonal); the trivial functor M = I is
still a zero of the naturality defect, and only the task loss keeps morphisms informative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrizations


@dataclass
class QueryCategory:
    """A finitely presented category: objects, generators name -> (source, target), inverse pairs."""

    objects: Tuple[str, ...]
    generators: Dict[str, Tuple[str, str]]
    inverse_pairs: Tuple[Tuple[str, str], ...] = ()
    names: List[str] = field(init=False)

    def __post_init__(self):
        self.names = list(self.generators)
        for name, (src, dst) in self.generators.items():
            if src not in self.objects or dst not in self.objects:
                raise ValueError(f"generator {name} has an unknown object")
        seen = set()
        for f, g in self.inverse_pairs:
            if self.generators[f] != self.generators[g][::-1]:
                raise ValueError(f"{f} and {g} are not opposite morphisms")
            if {f, g} & seen:
                raise ValueError("a generator may belong to one inverse pair only")
            seen |= {f, g}

    def index(self, name: str) -> int:
        return self.names.index(name)

    def check_path(self, path: Sequence[str]) -> Tuple[str, str]:
        """A path is a list of generators in application order; returns (source, target)."""
        if not path:
            raise ValueError("empty path; use the identity")
        for f, g in zip(path, path[1:]):
            if self.generators[f][1] != self.generators[g][0]:
                raise ValueError(f"{g} cannot follow {f}")
        return self.generators[path[0]][0], self.generators[path[-1]][1]


def routing_world_category() -> QueryCategory:
    """The routing benchmark's relations as a category (ToyWorld in ti_tnn_train.py)."""
    return QueryCategory(
        objects=("country", "capital", "region", "size"),
        generators={"region_of": ("country", "region"), "size_of": ("capital", "size"),
                    "capital_of": ("country", "capital"), "country_of": ("capital", "country")},
        inverse_pairs=(("capital_of", "country_of"),),
    )


def orthogonal_square(d: int) -> nn.Module:
    """A bias-free d x d linear map constrained to the orthogonal group (exact up to float error)."""
    return parametrizations.orthogonal(nn.Linear(d, d, bias=False))


class CDLSpecialistBranch(nn.Module):
    """Drop-in for TypedSpecialistBranch whose stalk carries a functor image of a QueryCategory.

    encoder: the original residual block (objects may be any vectors); morphisms: one orthogonal
    matrix per free generator, transposes for inverse partners. forward(x, steps) applies, in order,
    s <- sum_g w_g M_g s for each [B, G] weight tensor in steps; one-hot weights apply a functor
    image exactly, soft weights a convex mixture (no longer exactly functorial). Returns
    (up_proj(s), s) like TypedSpecialistBranch.
    """

    def __init__(self, d_model: int, d_stalk: int, flop_cost: float, category: QueryCategory):
        super().__init__()
        self.flop_cost, self.category = flop_cost, category
        self.down_proj = nn.Linear(d_model, d_stalk)
        self.specialist_block = nn.Sequential(
            nn.LayerNorm(d_stalk), nn.Linear(d_stalk, d_stalk * 2), nn.GELU(), nn.Linear(d_stalk * 2, d_stalk)
        )
        self.up_proj = nn.Linear(d_stalk, d_model)
        partners = {g: f for f, g in category.inverse_pairs}
        self._transposed = partners  # generator -> the generator whose matrix it transposes
        self.morphisms = nn.ModuleDict({g: orthogonal_square(d_stalk) for g in category.names if g not in partners})

    def matrix(self, name: str) -> torch.Tensor:
        if name in self._transposed:
            return self.morphisms[self._transposed[name]].weight.T
        return self.morphisms[name].weight

    def matrices(self) -> torch.Tensor:
        """[G, d, d] in category.names order."""
        return torch.stack([self.matrix(n) for n in self.category.names])

    def path_matrix(self, path: Sequence[str]) -> torch.Tensor:
        """F(path): the product of generator matrices, last generator leftmost (exact functoriality)."""
        self.category.check_path(path)
        out = self.matrix(path[0])
        for name in path[1:]:
            out = self.matrix(name) @ out
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s = self.down_proj(x)
        return s + self.specialist_block(s)

    def forward(self, x: torch.Tensor, steps: Optional[Sequence[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.encode(x)
        if steps:
            M = self.matrices()
            for w in steps:  # w: [B, G]; s: [B, L, d]
                s = torch.einsum("bg,gij,blj->bli", w.to(s.dtype), M, s)
        return self.up_proj(s), s

    def macs_per_token(self, n_steps: int = 1) -> int:
        dm, ds = self.down_proj.in_features, self.down_proj.out_features
        return dm * ds + ds * 2 * ds + 2 * ds * ds + ds * dm + n_steps * ds * ds


class FunctorialSheafBridge(nn.Module):
    """A flat O(d) connection sheaf on the branch graph: per-branch orthogonal restriction maps Q_k
    into one shared edge frame. Same output form as SheafRestrictionBridge, W_e tanh(coboundary)."""

    def __init__(self, num_branches: int, d_stalk: int, d_model: int):
        super().__init__()
        self.restrict = nn.ModuleList([orthogonal_square(d_stalk) for _ in range(num_branches)])
        self.edge_to_model = nn.Linear(d_stalk, d_model)

    def maps(self) -> torch.Tensor:
        return torch.stack([r.weight for r in self.restrict])  # [K, d, d]

    def transport(self, u: int, v: int) -> torch.Tensor:
        Q = self.maps()
        return Q[v].T @ Q[u]

    def coboundary(self, s_u: torch.Tensor, s_v: torch.Tensor, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        Q = self.maps()
        return (torch.einsum("bij,blj->bli", Q[u_idx], s_u)
                - torch.einsum("bij,blj->bli", Q[v_idx], s_v))

    def forward(self, s_u, s_v, u_idx, v_idx) -> torch.Tensor:
        return self.edge_to_model(torch.tanh(self.coboundary(s_u, s_v, u_idx, v_idx)))

    def macs_per_token(self) -> int:
        d = self.edge_to_model.in_features
        return 2 * d * d + d * self.edge_to_model.out_features


def pair_weights(route_probs: torch.Tensor, floor: float) -> torch.Tensor:
    """[K, K] symmetric weights floor + E[p_u p_v + p_v p_u] from the router's probabilities (detached),
    the router's expected use of each branch pair; the floor keeps idle pairs from drifting freely."""
    p = route_probs.detach()
    joint = (p.unsqueeze(2) * p.unsqueeze(1)).mean(0)
    return floor + joint + joint.T


def naturality_loss(branches: Sequence[CDLSpecialistBranch], bridge: FunctorialSheafBridge,
                    route_probs: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    """Router-weighted naturality defect of the bridge's transports between branch functors:
    sum_{u<v} w_uv * mean_g ||T(u->v) M^u_g - M^v_g T(u->v)||_F^2 / d."""
    K = len(branches)
    w = pair_weights(route_probs, floor)
    Q = bridge.maps()
    Ms = [b.matrices() for b in branches]
    d = Q.shape[-1]
    total = Q.new_zeros(())
    for u in range(K):
        for v in range(u + 1, K):
            T = Q[v].T @ Q[u]
            defect = T.unsqueeze(0) @ Ms[u] - Ms[v] @ T.unsqueeze(0)
            total = total + w[u, v] * defect.pow(2).sum((-2, -1)).mean() / d
    return total
