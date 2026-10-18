"""A small GPT-style decoder whose feed-forward block can be dense, a mixture of experts, or TI-TNN.

Feed-forward variants (every full expert has the dense block's shape, D -> mD -> D, m = ffn_mult = 4):

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

Adaptive-compute variants (`ffn` names the whole block for these): each token, at each layer, chooses
how much of the block to run. Skipped tokens leave the block unchanged and are not keys or values in its
attention (as in Mixture-of-Depths, Raposo et al. 2024).

    mod              skip the block, or run it with the dense MLP. Mixture-of-Depths-style, but token
                     choice rather than expert choice, so every decision is causal.
    ti_tnn_adaptive  skip the block, or run it with a small or a full feed-forward layer; each is a
                     top-1 mixture of typed experts (small experts are D -> D -> D, a quarter of the
                     MLP). The choice of compute replaces the Noul gate and the bridge, which did not
                     earn their compute on WikiText-103.

The compute choice is a hard argmax in the forward pass and straight-through in the backward pass:
every option is evaluated for every token in training, so the router sees each option's first-order
value and no option is absorbing. A cap on mean compute (`target_compute`, a fraction of a full block's
projection and MLP multiply-accumulates) is enforced by a Lagrange multiplier updated by dual ascent:
the price of compute rises while the cap is exceeded and falls to zero when it is not binding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

FFN_KINDS = ("dense", "moe_top1", "moe_top2", "ti_tnn", "mod", "ti_tnn_adaptive")
EXPERT_INITS = ("legacy", "dense")
ADAPTIVE_KINDS = ("mod", "ti_tnn_adaptive")


def size_tag(d_model: int, n_layers: int, ffn_mult: int, reference: tuple = (512, 8, 4)) -> str:
    """"" for the reference size, else e.g. "d384_l8" or "d512_l8_m1"."""
    if (d_model, n_layers, ffn_mult) == tuple(reference):
        return ""
    return f"d{d_model}_l{n_layers}" + (f"_m{ffn_mult}" if ffn_mult != 4 else "")


def variant_name(ffn: str, bridge_cost: float, target_compute: float = 1.0, size: str = "") -> str:
    """ti_tnn runs are named by their bridge charge ("ti_tnn_charged" / "ti_tnn_free"), adaptive runs by
    their compute cap ("mod_c50"), others by ffn; a size tag is appended for non-reference sizes."""
    if ffn == "ti_tnn":
        name = "ti_tnn_free" if bridge_cost == 0 else ("ti_tnn_charged" if bridge_cost == 0.01 else f"ti_tnn_cost{bridge_cost:g}")
    elif ffn in ADAPTIVE_KINDS:
        name = f"{ffn}_c{round(100 * target_compute)}"
    else:
        name = ffn
    return f"{name}_{size}" if size else name


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
    ffn_mult: int = 4              # hidden width of the dense MLP and of every full expert, in units of d_model
    small_mult: int = 1            # ti_tnn_adaptive: hidden width of the small experts
    target_compute: float = 1.0    # adaptive variants: cap on mean block compute, as a fraction of a full block
    dual_lr: float = 0.01          # adaptive variants: dual-ascent step for the price of compute
    expert_init: str = "legacy"    # "legacy": std D^-1/2 up, H^-1/2 down; "dense": the dense MLP's rule
    router_full_bias: float = 0.0  # adaptive variants: initial compute-router bias toward the full option
    skip_keeps_kv: bool = False    # adaptive variants: a skipped token still writes its key and value (costs 2D^2)
    shared_expert_frac: float = 0.0  # mixtures: this share of each layer's hidden width is an always-on shared MLP

    def __post_init__(self):
        if self.ffn not in FFN_KINDS:
            raise ValueError(f"ffn must be one of {FFN_KINDS}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 < self.target_compute <= 1:
            raise ValueError("target_compute must be in (0, 1]")
        if self.expert_init not in EXPERT_INITS:
            raise ValueError(f"expert_init must be one of {EXPERT_INITS}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """key_mask [B, T] (bool): only these tokens are keys and values. Every query also sees itself, so a
        token that skips the block still gets the output it would have had if it ran (for the router's
        straight-through gradient)."""
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        if key_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            eye = torch.eye(T, dtype=torch.bool, device=x.device)
            allowed = eye.logical_or(key_mask[:, None, :]).tril()
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], dropout_p=drop)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class DenseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden: int | None = None):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model if hidden is None else hidden
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.down(F.gelu(self.up(x))), {}


class Experts(nn.Module):
    """E MLPs (the dense block's shape unless hidden_mult says otherwise), run only on the tokens assigned to each."""

    def __init__(self, cfg: ModelConfig, hidden_mult: int | None = None, hidden: int | None = None):
        super().__init__()
        D, E = cfg.d_model, cfg.n_experts
        H = hidden if hidden is not None else (cfg.ffn_mult if hidden_mult is None else hidden_mult) * D
        self.w_up = nn.Parameter(torch.empty(E, D, H))
        self.w_down = nn.Parameter(torch.empty(E, H, D))
        # "dense" matches the dense MLP: std 0.02 up, and the GPT-2 residual scaling 0.02 / sqrt(2L) down.
        # Only the standard deviation changes at the same sampling sites, so the same random numbers are
        # drawn and every other parameter is identical.
        up, down = (D ** -0.5, H ** -0.5) if cfg.expert_init == "legacy" else (0.02, 0.02 / math.sqrt(2 * cfg.n_layers))
        nn.init.normal_(self.w_up, std=up)
        nn.init.normal_(self.w_down, std=down)

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
    def __init__(self, cfg: ModelConfig, top_k: int, hidden_mult: int | None = None):
        super().__init__()
        self.cfg, self.top_k = cfg, top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.shared = None
        if cfg.shared_expert_frac > 0:  # split the width: an always-on shared MLP plus narrower routed experts
            total = (cfg.ffn_mult if hidden_mult is None else hidden_mult) * cfg.d_model
            shared = round(cfg.shared_expert_frac * total)
            self.experts = Experts(cfg, hidden=total - shared)
            self.shared = DenseFFN(cfg, hidden=shared)
        else:
            self.experts = Experts(cfg, hidden_mult)

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
        if self.shared is not None:
            y = y + self.shared(x)[0]
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


def branch_mults(cfg: ModelConfig) -> List[int]:
    """Hidden widths (in units of d_model) of an adaptive block's options after skip."""
    return [cfg.ffn_mult] if cfg.ffn == "mod" else [cfg.small_mult, cfg.ffn_mult]


def option_costs(cfg: ModelConfig) -> List[float]:
    """Compute of each option (skip first) as a fraction of a full block: projections 4D^2 plus MLP 2mD^2.
    With skip_keeps_kv, skipping still pays the key and value projections (2D^2)."""
    full = 4 + 2 * cfg.ffn_mult
    return [2 / full if cfg.skip_keeps_kv else 0.0] + [(4 + 2 * m) / full for m in branch_mults(cfg)]


class AdaptiveBlock(nn.Module):
    """A decoder block in which each token chooses its compute: skip, or run with one of the branches."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        if cfg.ffn == "mod":
            self.branches = nn.ModuleList([DenseFFN(cfg)])
        else:
            self.branches = nn.ModuleList([MoEFFN(cfg, 1, m) for m in branch_mults(cfg)])
        self.router = nn.Linear(cfg.d_model, 1 + len(self.branches))
        self.register_buffer("costs", torch.tensor(option_costs(cfg)), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        probs = F.softmax(self.router(x).float(), dim=-1)
        choice = probs.argmax(-1)
        weight = F.one_hot(choice, probs.shape[-1]).to(probs.dtype)
        if self.training:  # hard choice forward; every option's value reaches the router backward
            weight = weight + probs - probs.detach()
        run = choice > 0
        # Without skip_keeps_kv a skipped token is not a key or value here. That eligibility is a hard mask with
        # no gradient, so the router never learns what a token's visibility is worth to later tokens.
        # skip_keeps_kv keeps every token visible, so skipping saves only the token's own update.
        h = x + self.attn(self.ln1(x), key_mask=None if self.cfg.skip_keeps_kv else run)
        hn = self.ln2(h)
        y = x * weight[..., :1]
        aux = torch.zeros((), device=x.device)
        for i, branch in enumerate(self.branches):
            out, s = branch(hn)
            y = y + (h + out) * weight[..., i + 1:i + 2]
            if "aux_loss" in s:
                aux = aux + s["aux_loss"]
        stats = {"choice": choice, "aux_loss": aux, "expected_cost": (probs * self.costs).sum(-1).mean(),
                 "realized_cost": self.costs[choice].mean().detach()}
        return y.to(x.dtype), stats


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        block = AdaptiveBlock if cfg.ffn in ADAPTIVE_KINDS else Block
        self.blocks = nn.ModuleList([block(cfg) for _ in range(cfg.n_layers)])
        if cfg.ffn in ADAPTIVE_KINDS:
            self.register_buffer("cost_price", torch.zeros(()))  # the Lagrange multiplier on mean compute
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled init for residual projections (GPT-2)
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.ffn in ADAPTIVE_KINDS and cfg.router_full_bias:  # start close to the full model (no draws)
            with torch.no_grad():
                for block in self.blocks:
                    block.router.bias[-1] += cfg.router_full_bias

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
            if "expected_cost" in s:  # the priced compute, averaged over layers
                s["aux_loss"] = s["aux_loss"] + self.cost_price * s["expected_cost"] / len(self.blocks)
            stats.append(s)
        return self.head(self.ln_f(x)), stats

    @torch.no_grad()
    def dual_update(self, stats: List[Dict[str, torch.Tensor]], target: float | None = None) -> None:
        """Raise the price of compute while mean realized compute exceeds the cap (cfg.target_compute, or a
        scheduled `target`); never below zero."""
        if self.cfg.ffn not in ADAPTIVE_KINDS:
            return
        cap = self.cfg.target_compute if target is None else target
        realized = torch.stack([s["realized_cost"] for s in stats]).mean()
        self.cost_price.add_(self.cfg.dual_lr * (realized - cap)).clamp_(min=0.0)


def parameter_counts(model: LanguageModel) -> Dict[str, int]:
    """Total parameters, and parameters a single token uses at most (embeddings counted once; one expert
    per routed bank; in an adaptive block, only its widest branch). Counted from the instantiated modules,
    so shared experts and narrowed expert banks are handled (before 2026-10-18 the count assumed
    full-width banks and went negative for shared-expert models)."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    per_expert = 2 * cfg.ffn_mult * cfg.d_model ** 2
    unused, skip = 0, set()
    for block in model.blocks:
        if isinstance(block, AdaptiveBlock) and len(block.branches) > 1:
            for small in block.branches[:-1]:  # a token runs at most one branch: count only the widest
                unused += sum(p.numel() for p in small.parameters())
                skip |= {id(m) for m in small.modules()}
    for m in model.modules():
        if isinstance(m, Experts) and id(m) not in skip:
            E = m.w_up.shape[0]
            unused += (m.w_up.numel() + m.w_down.numel()) * (E - 1) // E
    return {"total": total, "active_top1": total - unused, "expert_or_mlp_params_per_layer": per_expert}


def ffn_macs_per_token(cfg: ModelConfig, expert_runs: float) -> float:
    """Feed-forward multiply-accumulates per token per layer, from layer shapes (bridge included)."""
    mlp = 2 * cfg.ffn_mult * cfg.d_model ** 2
    if cfg.ffn == "dense":
        return mlp
    router = cfg.d_model * cfg.n_experts
    macs = expert_runs * mlp + router
    if cfg.ffn == "ti_tnn":
        macs += cfg.d_model + (expert_runs - 1) * 3 * cfg.d_model * cfg.d_edge  # gate + bridge where fired
    return macs


def attention_macs(cfg: ModelConfig, run: torch.Tensor) -> torch.Tensor:
    """Attention multiply-accumulates summed over the running tokens of a [B, T] mask: q, k, v and output
    projections (4D^2) plus scores and values (2D per key; keys are the running tokens up to and including
    the query)."""
    D = cfg.d_model
    keys = run.long().cumsum(-1)
    return (run.long() * (4 * D * D + 2 * D * keys)).sum()


def macs_per_token(cfg: ModelConfig, stats: List[Dict[str, torch.Tensor]], shape: tuple) -> Dict[str, float]:
    """Forward multiply-accumulates per token, from the routing decisions actually made.

    "non_embedding" covers the decoder blocks (attention scores included); "head" is the tied output
    projection (V x D per token), which no variant reduces. Routers are counted; layer norms are not.
    """
    B, T = shape
    D, N = cfg.d_model, B * T
    total = 0.0
    for s in stats:
        if "choice" in s:
            choice = s["choice"]
            run = choice > 0
            mults = torch.tensor([0] + branch_mults(cfg), device=choice.device)
            if cfg.skip_keeps_kv:  # running tokens attend to every earlier token; skipped ones write k and v
                pos = torch.arange(1, choice.shape[-1] + 1, device=choice.device)
                total += float((run.long() * (4 * D * D + 2 * D * pos)).sum()) + float((~run).sum()) * 2 * D * D
            else:
                total += float(attention_macs(cfg, run))
            total += float((2 * D * D * mults[choice]).sum())
            total += D * (1 + len(branch_mults(cfg))) * N  # the compute router, for every token
            if cfg.ffn == "ti_tnn_adaptive":
                total += D * cfg.n_experts * float(run.sum())  # the chosen branch's expert router
        else:
            run = torch.ones(B, T, dtype=torch.bool)
            runs = float(s["expert_runs"]) if "expert_runs" in s else 1.0
            total += float(attention_macs(cfg, run)) + ffn_macs_per_token(cfg, runs) * N
    return {"non_embedding": total / N, "head": float(cfg.vocab_size * D)}


def full_block_macs_per_token(cfg: ModelConfig, T: int) -> float:
    """Non-embedding MACs per token of the same-size dense model over a T-token window (the reference)."""
    D = cfg.d_model
    return cfg.n_layers * (4 * D * D + D * (T + 1) + 2 * cfg.ffn_mult * D * D)


@torch.no_grad()
def init_diagnostics(model: LanguageModel, idx: torch.Tensor) -> Dict[str, list]:
    """The residual stream's RMS after each block, and each layer's routing probabilities, for one batch.

    Matching raw weight scales does not match activation scales: a top-1 mixture also multiplies its
    expert's output by the router probability. This records what the initialization actually produced.
    """
    was_training = model.training
    model.eval()
    rms, routing, hooks = [], [], []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(lambda m, i, o: rms.append(float(o[0].float().pow(2).mean().sqrt()))))
        router = getattr(block, "router", None) or getattr(getattr(block, "ffn", None), "router", None)
        if router is not None:
            hooks.append(router.register_forward_hook(
                lambda m, i, o: routing.append(F.softmax(o.float(), -1).reshape(-1, o.shape[-1]))))
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    out = {"residual_rms_by_layer": rms}
    if routing:
        out["mean_top_probability_by_layer"] = [float(p.max(-1).values.mean()) for p in routing]
        out["mean_router_probabilities_by_layer"] = [p.mean(0).tolist() for p in routing]
    return out


def cap_schedule(target: float, step: int, steps: int, warmup_frac: float) -> float:
    """The compute cap at a training step: full compute at step 0, falling linearly to `target` over the
    first warmup_frac of training (constant `target` when warmup_frac is 0)."""
    if warmup_frac <= 0:
        return target
    progress = min(1.0, step / max(1, warmup_frac * steps))
    return 1.0 - (1.0 - target) * progress
