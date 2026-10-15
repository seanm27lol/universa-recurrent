"""A small GPT-style decoder whose feed-forward block can be dense, a mixture of experts, or TI-TNN.

Feed-forward variants (every expert has the dense block's shape, D -> 4D -> D):

    dense     one MLP per layer
    moe_top1  Switch-style mixture of experts: each token runs its top-1 expert, scaled by its
              router probability (Fedus, Zoph and Shazeer 2022)
    moe_top2  each token runs its top-2 experts, weighted by their renormalized probabilities
              (Lepikhin et al. 2021)
    ti_tnn    top-1 expert as in moe_top1, plus a Noul gate: where it fires, the runner-up expert
              also runs and a one-edge sheaf bridge adds W_e tanh(F_u o_u - F_v o_v) from the two
              experts' outputs (lower expert index first). The gate is hard with a straight-through
              gradient, and each firing is charged in the auxiliary loss. Training runs the runner-up
              and bridge for every token so the gate can learn where firing helps; inference runs
              them only where it fires. Its training compute therefore equals moe_top2's.

All mixture variants use the Switch load-balancing loss. No variant gets routing labels: on real
text there are none.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

FFN_KINDS = ("dense", "moe_top1", "moe_top2", "ti_tnn")


def variant_name(ffn: str, bridge_cost: float) -> str:
    """ti_tnn runs are named by their bridge charge ("ti_tnn_charged" / "ti_tnn_free"); others by ffn."""
    if ffn != "ti_tnn":
        return ffn
    return "ti_tnn_free" if bridge_cost == 0 else ("ti_tnn_charged" if bridge_cost == 0.01 else f"ti_tnn_cost{bridge_cost:g}")


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn: str = "dense"
    n_experts: int = 8
    d_edge: int = 128
    balance_coef: float = 0.01
    bridge_cost: float = 0.01      # auxiliary-loss charge per unit of expected bridge firing
    bridge_threshold: float = 0.5
    dropout: float = 0.0

    def __post_init__(self):
        if self.ffn not in FFN_KINDS:
            raise ValueError(f"ffn must be one of {FFN_KINDS}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class DenseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.down(F.gelu(self.up(x))), {}


class Experts(nn.Module):
    """E MLPs of the dense block's shape, run only on the tokens assigned to each."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H, E = cfg.d_model, 4 * cfg.d_model, cfg.n_experts
        self.w_up = nn.Parameter(torch.empty(E, D, H))
        self.w_down = nn.Parameter(torch.empty(E, H, D))
        nn.init.normal_(self.w_up, std=D ** -0.5)
        nn.init.normal_(self.w_down, std=H ** -0.5)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        """x [N, D]; (tokens[i], experts[i]) pairs to run. Returns one output row per pair."""
        out = None
        for e in experts.unique().tolist():
            sel = (experts == e).nonzero(as_tuple=True)[0]
            res = F.gelu(x[tokens[sel]] @ self.w_up[e]) @ self.w_down[e]
            if out is None:  # under autocast the matmul dtype differs from x's
                out = res.new_empty(len(tokens), x.shape[-1])
            out[sel] = res
        return out if out is not None else x.new_empty(0, x.shape[-1])


def balance_loss(probs: torch.Tensor, top1: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch load-balancing loss E * sum_e f_e P_e; equals 1 under perfectly uniform routing."""
    f = torch.bincount(top1, minlength=n_experts).to(probs.dtype) / len(top1)
    return n_experts * (f * probs.mean(0)).sum()


class MoEFFN(nn.Module):
    def __init__(self, cfg: ModelConfig, top_k: int):
        super().__init__()
        self.cfg, self.top_k = cfg, top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = Experts(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        N = len(x)
        probs = F.softmax(self.router(x).float(), dim=-1)
        top_p, top_e = probs.topk(self.top_k, dim=-1)
        if self.top_k > 1:
            top_p = top_p / top_p.sum(-1, keepdim=True)
        tokens = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        outs = self.experts(x, tokens, top_e.reshape(-1))
        y = torch.zeros_like(x, dtype=outs.dtype).index_add_(0, tokens, outs * top_p.reshape(-1, 1).to(outs.dtype))
        stats = {"aux_loss": self.cfg.balance_coef * balance_loss(probs, top_e[:, 0], self.cfg.n_experts),
                 "expert_runs": torch.tensor(float(self.top_k)), "top1": top_e[:, 0]}
        return y.reshape(shape), stats


class SheafBridge(nn.Module):
    """One-edge cellular-sheaf coboundary between two expert outputs, with learned restriction maps."""

    def __init__(self, d_model: int, d_edge: int):
        super().__init__()
        self.restrict_u = nn.Linear(d_model, d_edge, bias=False)
        self.restrict_v = nn.Linear(d_model, d_edge, bias=False)
        self.edge_to_model = nn.Linear(d_edge, d_model, bias=False)

    def forward(self, o_u: torch.Tensor, o_v: torch.Tensor) -> torch.Tensor:
        return self.edge_to_model(torch.tanh(self.restrict_u(o_u) - self.restrict_v(o_v)))


class TITNNFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.noul = nn.Linear(cfg.d_model, 1)
        self.experts = Experts(cfg)
        self.bridge = SheafBridge(cfg.d_model, cfg.d_edge)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        N = len(x)
        probs = F.softmax(self.router(x).float(), dim=-1)
        top2_p, top2_e = probs.topk(2, dim=-1)
        e1, e2 = top2_e[:, 0], top2_e[:, 1]
        noul = torch.sigmoid(self.noul(x).float()).squeeze(-1)
        fire = noul >= self.cfg.bridge_threshold
        gate = fire.float() + noul - noul.detach() if self.training else fire.float()

        rows = torch.arange(N, device=x.device)
        # Training runs the runner-up and the bridge for every token so the gate learns, through its
        # straight-through gradient, where firing would help (as in the toy). Inference runs them only
        # where the gate fires.
        bridged = rows if self.training else rows[fire]
        outs = self.experts(x, torch.cat([rows, bridged]), torch.cat([e1, e2[bridged]]))
        o1 = outs[:N]
        y = o1 * top2_p[:, 0:1].to(o1.dtype)
        if len(bridged):
            o2, a = outs[N:], o1[bridged]
            first = (e1[bridged] < e2[bridged]).unsqueeze(-1)  # fixed edge orientation: lower expert index first
            o_u, o_v = torch.where(first, a, o2), torch.where(first, o2, a)
            correction = self.bridge(o_u, o_v) * gate[bridged].unsqueeze(-1).to(o1.dtype)
            y = y.index_add(0, bridged, correction)
        aux = (self.cfg.balance_coef * balance_loss(probs, e1, self.cfg.n_experts)
               + self.cfg.bridge_cost * noul.mean())
        # expert_runs counts what inference would run: the top-1 expert, plus the runner-up where fired.
        stats = {"aux_loss": aux, "expert_runs": 1.0 + fire.float().mean().detach(), "top1": e1,
                 "fire_rate": fire.float().mean().detach()}
        return y.reshape(shape), stats


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = {"dense": lambda: DenseFFN(cfg), "moe_top1": lambda: MoEFFN(cfg, 1),
                    "moe_top2": lambda: MoEFFN(cfg, 2), "ti_tnn": lambda: TITNNFFN(cfg)}[cfg.ffn]()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h, stats = self.ffn(self.ln2(x))
        return x + h, stats


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init for residual projections (GPT-2)
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        stats = []
        for block in self.blocks:
            x, s = block(x)
            stats.append(s)
        return self.head(self.ln_f(x)), stats


def parameter_counts(model: LanguageModel) -> Dict[str, int]:
    """Total parameters, and parameters a single token uses (embeddings counted once; experts at top-1)."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    per_expert = 8 * cfg.d_model ** 2
    unused = 0
    if cfg.ffn != "dense":
        unused = cfg.n_layers * (cfg.n_experts - 1) * per_expert
    return {"total": total, "active_top1": total - unused, "expert_or_mlp_params_per_layer": per_expert}


def ffn_macs_per_token(cfg: ModelConfig, expert_runs: float) -> float:
    """Feed-forward multiply-accumulates per token per layer, from layer shapes (bridge included)."""
    mlp = 8 * cfg.d_model ** 2
    if cfg.ffn == "dense":
        return mlp
    router = cfg.d_model * cfg.n_experts
    macs = expert_runs * mlp + router
    if cfg.ffn == "ti_tnn":
        macs += cfg.d_model + (expert_runs - 1) * 3 * cfg.d_model * cfg.d_edge  # gate + bridge where fired
    return macs
