"""Token-by-token decoding with per-layer key/value caches, for measuring latency (batch size 1).

Each step does only the work its routing decisions call for: a token that skips a block computes
nothing there and adds no key or value to that layer's cache, and a mixture layer runs only the chosen
expert. The logits equal the full forward pass's in eval mode (tests/test_ti_tnn_bench.py checks this
for every variant), which also shows that no routing decision depends on a later token.
"""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn.functional as F

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script


class Cache:
    """Keys and values of the tokens that ran a layer, preallocated for `length` tokens."""

    def __init__(self, n_heads: int, length: int, d_head: int, device, dtype):
        self.k = torch.empty(1, n_heads, length, d_head, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.n = 0


def attention_step(attn: models.CausalSelfAttention, h: torch.Tensor, cache: Cache) -> torch.Tensor:
    """h [1, D]: append this token's key and value, attend over the cache (itself included)."""
    D = h.shape[-1]
    H = attn.n_heads
    q, k, v = attn.qkv(h).split(D, dim=-1)
    q, k, v = (t.view(1, H, 1, D // H) for t in (q, k, v))
    cache.k[:, :, cache.n] = k[:, :, 0].to(cache.k.dtype)
    cache.v[:, :, cache.n] = v[:, :, 0].to(cache.v.dtype)
    cache.n += 1
    y = F.scaled_dot_product_attention(q, cache.k[:, :, :cache.n].to(q.dtype), cache.v[:, :, :cache.n].to(q.dtype))
    return attn.proj(y.reshape(1, D))


def kv_append(attn: models.CausalSelfAttention, h: torch.Tensor, cache: Cache) -> None:
    """Write this token's key and value to the cache without attending (a skipped but visible token)."""
    D = h.shape[-1]
    H = attn.n_heads
    _, k, v = attn.qkv(h).split(D, dim=-1)
    cache.k[:, :, cache.n] = k.view(1, H, D // H).to(cache.k.dtype)
    cache.v[:, :, cache.n] = v.view(1, H, D // H).to(cache.v.dtype)
    cache.n += 1


def expert(experts: models.Experts, h: torch.Tensor, e: int) -> torch.Tensor:
    return F.gelu(h @ experts.w_up[e]) @ experts.w_down[e]


def moe_step(ffn: models.MoEFFN, h: torch.Tensor) -> torch.Tensor:
    """The same arithmetic as MoEFFN.forward: each weight is rounded to the expert output's dtype (bf16
    under autocast) before the multiply, and the weighted outputs are summed in that dtype."""
    probs = F.softmax(ffn.router(h).float(), dim=-1)[0]
    p, e = probs.topk(ffn.top_k)
    if ffn.top_k > 1:
        p = p / p.sum()
    out = None
    for j, ej in enumerate(e.tolist()):
        o = expert(ffn.experts, h, ej)
        o = o * p[j].to(o.dtype)
        out = o if out is None else out + o
    return out


def ffn_step(ffn, h: torch.Tensor) -> torch.Tensor:
    if isinstance(ffn, models.DenseFFN):
        return ffn.down(F.gelu(ffn.up(h)))
    if isinstance(ffn, models.MoEFFN):
        return moe_step(ffn, h)
    return ffn(h)[0]  # ti_tnn: its eval path already runs the runner-up only where the gate fires


def block_step(block, x: torch.Tensor, cache: Cache) -> tuple[torch.Tensor, int]:
    """One token through one block; returns the new residual and the option taken (skip = 0)."""
    if isinstance(block, models.AdaptiveBlock):
        choice = int(block.router(x).argmax(-1))
        if choice == 0:
            if block.cfg.skip_keeps_kv:  # still a key and value for later tokens
                kv_append(block.attn, block.ln1(x), cache)
            return x, 0
        h = x + attention_step(block.attn, block.ln1(x), cache)
        return h + ffn_step(block.branches[choice - 1], block.ln2(h)), choice
    h = x + attention_step(block.attn, block.ln1(x), cache)
    return h + ffn_step(block.ffn, block.ln2(h)), 1


@torch.no_grad()
def decode(model: models.LanguageModel, idx: torch.Tensor, timed: bool = False, autocast: bool = False):
    """idx [1, T]. Returns logits [T, V], per-token seconds (if timed) and the options taken [T, n_layers]."""
    model.eval()
    cfg = model.cfg
    T = idx.shape[1]
    device = idx.device
    dtype = torch.bfloat16 if autocast else model.tok.weight.dtype
    caches = [Cache(cfg.n_heads, T, cfg.d_model // cfg.n_heads, device, dtype) for _ in model.blocks]
    logits, seconds, taken = [], [], []
    for t in range(T):
        if timed:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            x = model.tok(idx[:, t]) + model.pos.weight[t]
            row: List[int] = []
            for block, cache in zip(model.blocks, caches):
                x, c = block_step(block, x, cache)
                row.append(c)
            out = model.head(model.ln_f(x))
        if timed:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            seconds.append(time.perf_counter() - t0)
        logits.append(out.float())
        taken.append(row)
    return torch.cat(logits), seconds, torch.tensor(taken)
