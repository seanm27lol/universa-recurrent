#!/usr/bin/env python3
"""TI-TNN: train a typed-branch router with a sheaf bridge, and test it honestly.

Two synthetic benchmarks share one model (see README.md in this directory):

1. ROUTING. Queries such as "capital_of <country>" or "region_of <country>". Branches
   are token-wise, so only the router can tell the model which function to apply to
   the entity token: routing is load-bearing. Variants: routed + bridge, routed with no
   bridge, dense (every branch on every query).

2. COMPOSITIONAL ISOLATION. Every country has two independent binary traits.
   "traitA_of <c>" is supervised only through Branch 0 and "traitB_of <c>" only
   through Branch 1. "combined_status_of <c>" asks for XOR(A, B), and its queries for
   held-out countries never appear in training. With the isolation below, any single
   branch holds at most one trait of a held-out country, and XOR(A, B) is independent
   of A alone and of B alone, so one branch is capped at chance (50%). The bridge
   reads Branch 0's and Branch 1's stalks together and can in principle beat it.

   Isolation that makes the cap hold (each closes a real leak):
     * private branch embeddings: each branch reads its own embedding subspace, so a
       country's A and B never meet in one vector (a shared table would leak);
     * gradient masking: a query's gradient reaches only its designated branch, so
       Gumbel exploration cannot send Trait-B supervision into Branch 0;
     * the router reads the query context but not the entity token, so routing cannot
       carry country-specific trait information;
     * combined accuracy is a 2-way S0-vs-S1 comparison, so chance is 50%.
   The "no bridge, shared embedding" variant removes the first item on purpose.

Model changes from the original sketch (both benchmarks): a real training loop; the
bridge pairs the executed branch with the runner-up and actually runs the runner-up
at inference when the Noul gate fires; a hard gate with a straight-through gradient
in training and inference alike; the bridge edge has a fixed orientation (lower
branch index first), as an edge of a cellular sheaf does; per-query conditional
compute; multiply-accumulates counted from layer shapes. The depth/score head is
kept but nothing trains it.

Run:   python ti_tnn_train.py                          (GPU if available; both benchmarks; 3 seeds)
       python ti_tnn_train.py --benchmark compositional --device cpu --steps 2000 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCH_NAMES = [
    "Countries_Rep (Noun)",
    "Capitals_Rep (Noun)",
    "Location_With_Importance (Relational)",
    "General_Syntax_Recurrent (Fallback)",
]
K = len(BRANCH_NAMES)


# =========================================================================== model
class JevCalibratedRouter(nn.Module):
    """Layer-1 router: Choice over K branches, Noul bridge gate, depth Score (untrained here)."""

    def __init__(self, d_model: int, num_branches: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.choice_head = nn.Linear(d_model, num_branches)
        self.noul_bridge_head = nn.Linear(d_model, 1)
        self.score_depth_head = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, tau: float = 1.0, hard_prune: bool = False) -> Dict[str, torch.Tensor]:
        pooled = self.norm(h.mean(dim=1))
        choice_logits = self.choice_head(pooled)
        choice_probs = F.softmax(choice_logits, dim=-1)
        if self.training:
            # Straight-through Gumbel-Softmax: one-hot forward, soft gradient.
            routing_weights = F.gumbel_softmax(choice_logits, tau=tau, hard=True, dim=-1)
        elif hard_prune:
            top_idx = torch.argmax(choice_probs, dim=-1, keepdim=True)
            routing_weights = torch.zeros_like(choice_probs).scatter_(-1, top_idx, 1.0)
        else:
            routing_weights = choice_probs
        return {
            "choice_logits": choice_logits,
            "choice_probs": choice_probs,
            "routing_weights": routing_weights,
            "noul_prob": torch.sigmoid(self.noul_bridge_head(pooled)),
            "depth_score": torch.sigmoid(self.score_depth_head(pooled)),
        }


class TypedSpecialistBranch(nn.Module):
    """One token-wise specialist: d_model -> stalk (residual MLP) -> d_model."""

    def __init__(self, d_model: int, d_stalk: int, flop_cost: float):
        super().__init__()
        self.flop_cost = flop_cost
        self.down_proj = nn.Linear(d_model, d_stalk)
        self.specialist_block = nn.Sequential(
            nn.LayerNorm(d_stalk), nn.Linear(d_stalk, d_stalk * 2), nn.GELU(), nn.Linear(d_stalk * 2, d_stalk)
        )
        self.up_proj = nn.Linear(d_stalk, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stalk_state = self.down_proj(x)
        stalk_state = stalk_state + self.specialist_block(stalk_state)
        return self.up_proj(stalk_state), stalk_state

    def macs_per_token(self) -> int:
        dm, ds = self.down_proj.in_features, self.down_proj.out_features
        return dm * ds + ds * 2 * ds + 2 * ds * ds + ds * dm


class SheafRestrictionBridge(nn.Module):
    """One-edge cellular-sheaf coboundary between two branch stalks, with learned restriction maps."""

    def __init__(self, d_stalk: int, d_edge: int, d_model: int):
        super().__init__()
        self.restriction_u = nn.Linear(d_stalk, d_edge, bias=False)
        self.restriction_v = nn.Linear(d_stalk, d_edge, bias=False)
        self.edge_to_model = nn.Linear(d_edge, d_model)

    def forward(self, stalk_u: torch.Tensor, stalk_v: torch.Tensor) -> torch.Tensor:
        return self.edge_to_model(torch.tanh(self.restriction_u(stalk_u) - self.restriction_v(stalk_v)))

    def macs_per_token(self) -> int:
        ds, de = self.restriction_u.in_features, self.restriction_u.out_features
        return 2 * ds * de + de * self.edge_to_model.out_features


class TypeInformedTreeNN(nn.Module):
    """Vectorize -> route -> typed branches (+ gated sheaf bridge) -> reconstruct -> detokenize.

    dense=True: soft mixture of every branch on every query, bridge always computed and
    softly gated. use_bridge=False: no bridge at all. branch_embeddings="private": each
    branch reads its own embedding subspace instead of the router's shared table.
    router_context="context_only": the router pools every position except the last
    (entity) token.
    """

    BRANCH_NAMES = BRANCH_NAMES

    def __init__(self, vocab_size: int, d_model=128, d_stalk=64, d_edge=32,
                 branch_costs=(1.0, 1.0, 1.5, 4.0), bridge_threshold=0.5, dense=False, use_bridge=True,
                 branch_embeddings="shared", router_context="all"):
        super().__init__()
        if branch_embeddings not in ("shared", "private") or router_context not in ("all", "context_only"):
            raise ValueError("unknown branch_embeddings or router_context")
        self.num_branches = len(self.BRANCH_NAMES)
        self.dense, self.use_bridge = dense, use_bridge
        self.branch_embeddings, self.router_context = branch_embeddings, router_context
        self.bridge_threshold = bridge_threshold
        self.d_model, self.d_stalk = d_model, d_stalk
        self.vectorizer = nn.Embedding(vocab_size, d_model)
        if branch_embeddings == "private":
            self.branch_vectorizer = nn.Embedding(vocab_size, d_model * self.num_branches)
        self.router = JevCalibratedRouter(d_model, self.num_branches)
        self.branches = nn.ModuleList([TypedSpecialistBranch(d_model, d_stalk, c) for c in branch_costs])
        self.register_buffer("branch_flop_costs", torch.tensor(branch_costs))
        self.sheaf_bridge = SheafRestrictionBridge(d_stalk, d_edge, d_model)
        self.reconstruct_norm = nn.LayerNorm(d_model)
        self.detokenizer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, tau: float = 0.5, force_route: Optional[torch.Tensor] = None,
                bridge: str = "auto", reference: bool = False, grad_allow: Optional[torch.Tensor] = None,
                partner_perm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """bridge: "auto" (Noul gate) or "off". reference: run every branch and the bridge for
        every query (only to check that conditional execution is exact). grad_allow [B, K]:
        False blocks that branch's gradient for that query. partner_perm [B]: the bridge's
        second stalk comes from another query (shuffled-partner control)."""
        B, L = input_ids.shape
        K = self.num_branches
        rows_all = torch.arange(B, device=input_ids.device)
        h0 = self.vectorizer(input_ids)
        route = self.router(h0 if self.router_context == "all" else h0[:, :-1], tau=tau, hard_prune=not self.dense)
        probs = route["choice_probs"]
        branch_in = ([h0] * K if self.branch_embeddings == "shared"
                     else list(self.branch_vectorizer(input_ids).split(self.d_model, dim=-1)))

        if self.dense:
            weights = probs
        elif force_route is not None:
            weights = F.one_hot(force_route, K).to(probs.dtype)
        else:
            weights = route["routing_weights"]

        # Executed branch, and the runner-up it can be bridged to (always a different branch).
        top1 = weights.argmax(-1)
        top2 = probs.masked_fill(F.one_hot(top1, K).bool(), float("-inf")).argmax(-1)

        noul = route["noul_prob"].squeeze(-1)
        if not self.use_bridge:
            bridge = "off"
        if bridge == "off":
            fire = torch.zeros(B, dtype=torch.bool, device=h0.device)
            gate = torch.zeros_like(noul)
        elif self.dense:
            fire = torch.ones(B, dtype=torch.bool, device=h0.device)
            gate = noul
        else:
            fire = noul >= self.bridge_threshold
            hard = fire.to(noul.dtype)
            gate = hard + noul - noul.detach() if self.training else hard  # same forward value both modes

        need = torch.zeros(B, K, dtype=torch.bool, device=h0.device)
        need[rows_all, top1] = True
        need[rows_all[fire], top2[fire]] = True
        # Training needs every branch output for the straight-through routing gradient.
        execute = torch.ones_like(need) if (self.training or self.dense or reference) else need

        reconstructed = torch.zeros_like(h0)
        stalks = h0.new_zeros(B, K, L, self.d_stalk)
        for k, branch in enumerate(self.branches):
            rows = execute[:, k].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            out_k, stalk_k = branch(branch_in[k][rows])
            if grad_allow is not None:
                keep = grad_allow[rows, k].view(-1, 1, 1)
                out_k = torch.where(keep, out_k, out_k.detach())
                stalk_k = torch.where(keep, stalk_k, stalk_k.detach())
            reconstructed = reconstructed.index_add(0, rows, weights[rows, k].view(-1, 1, 1) * out_k)
            stalks[rows, k] = stalk_k

        bridge_rows = rows_all[:0]
        if bridge != "off":
            bridge_rows = rows_all if (self.training or self.dense or reference) else rows_all[fire]
            if bridge_rows.numel():
                u_idx = torch.minimum(top1, top2)[bridge_rows]  # fixed edge orientation: lower index first
                v_idx = torch.maximum(top1, top2)[bridge_rows]
                v_rows = bridge_rows if partner_perm is None else partner_perm[bridge_rows]
                correction = self.sheaf_bridge(stalks[bridge_rows, u_idx], stalks[v_rows, v_idx])
                reconstructed = reconstructed.index_add(0, bridge_rows, gate[bridge_rows].view(-1, 1, 1) * correction)

        return {
            "logits": self.detokenizer(self.reconstruct_norm(reconstructed)),
            "route_info": route,
            "top1": top1,
            "top2": top2,
            "bridge_fired": fire,
            "branch_execs": int(execute.sum()),
            "bridge_execs": int(bridge_rows.numel()),
        }

    def compute_calibrated_loss(self, out, answer, target_route, target_bridge,
                                alpha_brier=0.5, lambda_flops=0.05) -> Dict[str, torch.Tensor]:
        """Task cross-entropy at the answer position + Brier calibration + expected cost penalty.
        target_route: branch indices [B] or a target distribution over branches [B, K]."""
        route = out["route_info"]
        loss_task = F.cross_entropy(out["logits"][:, -1], answer)
        dist = F.one_hot(target_route, self.num_branches).float() if target_route.dim() == 1 else target_route
        brier_choice = ((route["choice_probs"] - dist) ** 2).sum(-1).mean()
        brier_noul = F.mse_loss(route["noul_prob"].squeeze(-1), target_bridge.float())
        expected_cost = (route["choice_probs"] * self.branch_flop_costs).sum(-1).mean()
        total = loss_task + alpha_brier * (brier_choice + brier_noul) + lambda_flops * expected_cost
        return {"total_loss": total, "task_loss": loss_task,
                "brier_calibration_loss": brier_choice + brier_noul, "expected_cost": expected_cost}

    def macs_per_query(self, length: int, branch_execs: float, bridge_execs: float) -> float:
        """Branch + bridge multiply-accumulates per query (shared embedding/router/detokenizer excluded)."""
        return length * (branch_execs * self.branches[0].macs_per_token()
                         + bridge_execs * self.sheaf_bridge.macs_per_token())


# =========================================================================== shared helpers
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return float(ece)


def summarize(values) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    return f"{mean:.4f} ± {statistics.stdev(values):.4f}" if len(values) > 1 else f"{mean:.4f}"


def column(runs, variant, *path) -> str:
    vals = []
    for r in runs:
        if r["variant"] != variant:
            continue
        v = r
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
        vals.append(v)
    return summarize(vals)


def optimize(model, args, sample_batch, make_kwargs, label):
    """Shared training loop: AdamW, cosine LR, Gumbel temperature annealing, gradient clipping."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(args.steps):
        tau = args.tau_start * (args.tau_end / args.tau_start) ** (step / max(1, args.steps - 1))
        batch = sample_batch()
        model.train()
        out = model(batch["ids"], tau=tau, **make_kwargs(batch))
        losses = model.compute_calibrated_loss(out, batch["answer"], batch["route"], batch["bridge"],
                                               args.alpha_brier, args.lambda_flops)
        opt.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            acc = (out["logits"][:, -1].argmax(-1) == batch["answer"]).float().mean().item()
            print(f"  [{label}] step {step:5d}  loss {losses['total_loss'].item():.4f}  "
                  f"task {losses['task_loss'].item():.4f}  train-acc {acc:.3f}  tau {tau:.2f}", flush=True)
    return model


# =========================================================================== benchmark 1: routing
class ToyWorld:
    """Fixed facts (regions, sizes, capitals) plus a query sampler."""

    RELATIONS = ["region_of", "size_of", "capital_of", "country_of", "echo"]
    TYPES = [  # (relation, entity kind, target branch, needs bridge)
        ("region_of", "country", 0, 0),
        ("size_of", "capital", 1, 0),
        ("capital_of", "country", 2, 1),
        ("country_of", "capital", 2, 1),
        ("echo", "filler", 3, 0),
    ]

    def __init__(self, n_countries=64, n_regions=6, n_sizes=4, n_fillers=48, facts_seed=1234):
        self.n = n_countries
        start = 0

        def block(size):
            nonlocal start
            ids = torch.arange(start, start + size)
            start += size
            return ids

        self.rel_tok = block(len(self.RELATIONS))
        self.country_tok = block(n_countries)
        self.capital_tok = block(n_countries)
        self.region_tok = block(n_regions)
        self.size_tok = block(n_sizes)
        self.filler_tok = block(n_fillers)
        self.vocab_size = start
        g = torch.Generator().manual_seed(facts_seed)
        self.region_of_country = torch.randint(n_regions, (n_countries,), generator=g)
        self.size_of_capital = torch.randint(n_sizes, (n_countries,), generator=g)

    def sample(self, batch: int, length: int, generator: torch.Generator, device) -> Dict[str, torch.Tensor]:
        qtype = torch.randint(len(self.TYPES), (batch,), generator=generator)
        ids = self.filler_tok[torch.randint(len(self.filler_tok), (batch, length), generator=generator)]
        entity = torch.randint(self.n, (batch,), generator=generator)
        filler = torch.randint(len(self.filler_tok), (batch,), generator=generator)
        rel_pos = torch.randint(length - 1, (batch,), generator=generator)
        answer = torch.empty(batch, dtype=torch.long)
        branch = torch.empty(batch, dtype=torch.long)
        bridge = torch.empty(batch, dtype=torch.long)
        for t, (rel, kind, target_branch, needs_bridge) in enumerate(self.TYPES):
            rows = (qtype == t).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            e = entity[rows]
            ids[rows, rel_pos[rows]] = self.rel_tok[self.RELATIONS.index(rel)]
            ids[rows, length - 1] = {"country": lambda: self.country_tok[e], "capital": lambda: self.capital_tok[e],
                                     "filler": lambda: self.filler_tok[filler[rows]]}[kind]()
            answer[rows] = {
                "region_of": lambda: self.region_tok[self.region_of_country[e]],
                "size_of": lambda: self.size_tok[self.size_of_capital[e]],
                "capital_of": lambda: self.capital_tok[e],
                "country_of": lambda: self.country_tok[e],
                "echo": lambda: self.filler_tok[filler[rows]],
            }[rel]()
            branch[rows] = target_branch
            bridge[rows] = needs_bridge
        out = {"ids": ids, "answer": answer, "route": branch, "bridge": bridge, "qtype": qtype}
        return {k: v.to(device) for k, v in out.items()}


ROUTING_VARIANTS = {  # name: (dense, use_bridge)
    "sparse": (False, True),            # routed, hard-pruned, gated bridge (the proposal)
    "sparse_nobridge": (False, False),  # same, trained with no bridge
    "dense": (True, True),              # every branch on every query, soft mixture
}


@torch.no_grad()
def evaluate_routing(model: TypeInformedTreeNN, world: ToyWorld, args, device, length: int, controls: bool) -> Dict:
    model.eval()
    b = world.sample(args.eval_size, length, torch.Generator().manual_seed(args.eval_seed), device)
    ids, ans, br, brt, qt = b["ids"], b["answer"], b["route"], b["bridge"], b["qtype"]
    names = [t[0] for t in ToyWorld.TYPES]

    def scores(out):
        ok = out["logits"][:, -1].argmax(-1) == ans
        return {"task_acc": ok.float().mean().item(),
                "task_acc_by_type": {n: ok[qt == i].float().mean().item() for i, n in enumerate(names)}}

    out = model(ids)
    probs = out["route_info"]["choice_probs"]
    noul = out["route_info"]["noul_prob"].squeeze(-1)
    res = scores(out)
    res.update({
        "route_acc": (probs.argmax(-1) == br).float().mean().item(),
        "route_ece": expected_calibration_error(probs, br),
        "route_brier": ((probs - F.one_hot(br, model.num_branches).float()) ** 2).sum(-1).mean().item(),
        "noul_acc": ((noul >= model.bridge_threshold) == brt.bool()).float().mean().item(),
        "branch_execs_per_query": out["branch_execs"] / args.eval_size,
        "bridge_execs_per_query": out["bridge_execs"] / args.eval_size,
    })
    res["branch_bridge_macs_per_query"] = model.macs_per_query(
        length, res["branch_execs_per_query"], res["bridge_execs_per_query"])
    if not model.dense:
        res["bridge_fire_rate_by_type"] = {n: out["bridge_fired"][qt == i].float().mean().item()
                                           for i, n in enumerate(names)}
    if controls and not model.dense:
        rand = torch.randint(model.num_branches, (args.eval_size,),
                             generator=torch.Generator().manual_seed(args.eval_seed + 1)).to(device)
        res["controls"] = {
            "oracle_routing": scores(model(ids, force_route=br)),
            # Bridge off: otherwise the bridge pulls in the router's runner-up, usually the right branch.
            "random_routing_bridge_off": scores(model(ids, force_route=rand, bridge="off")),
            "bridge_off": scores(model(ids, bridge="off")),
        }
        ref = model(ids, reference=True)
        res["exactness"] = {
            "max_abs_logit_diff_vs_run_everything": (out["logits"] - ref["logits"]).abs().max().item(),
            "answer_agreement_vs_run_everything":
                (out["logits"][:, -1].argmax(-1) == ref["logits"][:, -1].argmax(-1)).float().mean().item(),
        }
    return res


def run_routing(args, device) -> Dict:
    world = ToyWorld(n_countries=args.n_countries, n_fillers=args.n_fillers)
    runs = []
    for seed in args.seeds:
        for variant, (dense, use_bridge) in ROUTING_VARIANTS.items():
            torch.manual_seed(seed)
            model = TypeInformedTreeNN(world.vocab_size, args.d_model, args.d_stalk, args.d_edge,
                                       bridge_threshold=args.bridge_threshold, dense=dense,
                                       use_bridge=use_bridge).to(device)
            gen = torch.Generator().manual_seed(10_000 + seed)  # same training stream for every variant
            t0 = time.time()
            optimize(model, args, lambda: world.sample(args.batch, args.length, gen, device),
                     lambda b: {}, f"routing seed {seed} {variant:15s}")
            runs.append({"seed": seed, "variant": variant, "train_seconds": round(time.time() - t0, 1),
                         "eval": evaluate_routing(model, world, args, device, args.length, controls=True),
                         "stress_eval": evaluate_routing(model, world, args, device, args.stress_length,
                                                         controls=False)})

    print(f"\n=== ROUTING benchmark (mean ± sd over seeds {args.seeds}) ===")
    print(f"{'':44s}" + "".join(f"{v:>20s}" for v in ROUTING_VARIANTS))
    for name, path in [
        ("Task accuracy (held-out queries)", ("eval", "task_acc")),
        ("Routing accuracy", ("eval", "route_acc")),
        ("Routing ECE (lower = better calibrated)", ("eval", "route_ece")),
        ("Noul gate accuracy", ("eval", "noul_acc")),
        ("Branch executions per query", ("eval", "branch_execs_per_query")),
        ("Bridge executions per query", ("eval", "bridge_execs_per_query")),
        ("Branch+bridge MACs per query", ("eval", "branch_bridge_macs_per_query")),
        (f"Task accuracy, length-{args.stress_length} stress", ("stress_eval", "task_acc")),
    ]:
        print(f"{name:44s}" + "".join(f"{column(runs, v, *path):>20s}" for v in ROUTING_VARIANTS))
    names = [t[0] for t in ToyWorld.TYPES]
    print("Controls on the routed + bridge model (task accuracy):")
    for key, label in (("oracle_routing", "oracle routing"),
                       ("random_routing_bridge_off", "random routing, bridge off"),
                       ("bridge_off", "learned routing, bridge off")):
        per_type = ", ".join(f"{n} {column(runs, 'sparse', 'eval', 'controls', key, 'task_acc_by_type', n)}"
                             for n in names)
        print(f"  {label:28s} {column(runs, 'sparse', 'eval', 'controls', key, 'task_acc')}   [{per_type}]")
    print(f"Exactness vs running every branch: max |logit diff| "
          f"{column(runs, 'sparse', 'eval', 'exactness', 'max_abs_logit_diff_vs_run_everything')}, answer agreement "
          f"{column(runs, 'sparse', 'eval', 'exactness', 'answer_agreement_vs_run_everything')}")
    return {"world": {"vocab_size": world.vocab_size, "types": ToyWorld.TYPES},
            "variants": {k: {"dense": d, "use_bridge": u} for k, (d, u) in ROUTING_VARIANTS.items()},
            "runs": runs}


# =========================================================================== benchmark 2: compositional isolation
class CompositionalWorld:
    """Countries with two independent, balanced binary traits and a held-out XOR query.

    Countries are split train / val / test, stratified by (A, B), so every split is
    balanced in A, B and XOR(A, B). Single-trait queries use every country; combined
    queries use only train countries during training.
    """

    RELATIONS = ["traitA_of", "traitB_of", "region_of", "echo", "combined_status_of"]
    TYPES = [  # (relation, entity kind, routing target over the 4 branches, needs bridge)
        ("traitA_of", "country", (1.0, 0.0, 0.0, 0.0), 0),
        ("traitB_of", "country", (0.0, 1.0, 0.0, 0.0), 0),
        ("region_of", "country", (0.0, 0.0, 1.0, 0.0), 0),
        ("echo", "filler", (0.0, 0.0, 0.0, 1.0), 0),
        # Branch 0 first, Branch 1 second: the runner-up the bridge wakes is the Trait-B branch.
        ("combined_status_of", "country", (0.6, 0.4, 0.0, 0.0), 1),
    ]
    COMBINED = 4
    SPLITS = ("train", "val", "test")

    def __init__(self, n_countries=512, n_regions=6, n_fillers=48, facts_seed=4321, val_frac=0.125, test_frac=0.25):
        if n_countries % 4:
            raise ValueError("n_countries must be divisible by 4 for balanced traits")
        self.n = n_countries
        start = 0

        def block(size):
            nonlocal start
            ids = torch.arange(start, start + size)
            start += size
            return ids

        self.rel_tok = block(len(self.RELATIONS))
        self.country_tok = block(n_countries)
        self.trait_a_tok, self.trait_b_tok, self.status_tok = block(2), block(2), block(2)
        self.region_tok = block(n_regions)
        self.filler_tok = block(n_fillers)
        self.vocab_size = start
        g = torch.Generator().manual_seed(facts_seed)
        combo = torch.arange(4).repeat_interleave(n_countries // 4)[torch.randperm(n_countries, generator=g)]
        self.trait_a, self.trait_b = combo // 2, combo % 2
        self.status = self.trait_a ^ self.trait_b
        self.region = torch.randint(n_regions, (n_countries,), generator=g)
        split = torch.zeros(n_countries, dtype=torch.long)
        n_test, n_val = round(n_countries * test_frac / 4), round(n_countries * val_frac / 4)
        for c in range(4):
            members = (combo == c).nonzero(as_tuple=True)[0]
            members = members[torch.randperm(len(members), generator=g)]
            split[members[:n_test]] = 2
            split[members[n_test:n_test + n_val]] = 1
        self.split = split
        self.countries_in = {name: (split == i).nonzero(as_tuple=True)[0] for i, name in enumerate(self.SPLITS)}
        self.candidates = [self.trait_a_tok, self.trait_b_tok, self.region_tok, self.filler_tok, self.status_tok]
        self.route_table = torch.tensor([t[2] for t in self.TYPES])

    def sample(self, batch, length, generator, device, qtype: Optional[int] = None,
               countries: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Training (qtype None): uniform query types; combined queries draw only train countries.
        Evaluation: a single qtype and an explicit list of countries (one query each)."""
        if countries is not None:
            batch = len(countries)
        qt = (torch.randint(len(self.TYPES), (batch,), generator=generator) if qtype is None
              else torch.full((batch,), qtype, dtype=torch.long))
        ids = self.filler_tok[torch.randint(len(self.filler_tok), (batch, length), generator=generator)]
        rel_pos = torch.randint(length - 1, (batch,), generator=generator)
        filler = torch.randint(len(self.filler_tok), (batch,), generator=generator)
        if countries is None:
            country = torch.randint(self.n, (batch,), generator=generator)
            combined = (qt == self.COMBINED).nonzero(as_tuple=True)[0]
            pool = self.countries_in["train"]
            country[combined] = pool[torch.randint(len(pool), (len(combined),), generator=generator)]
        else:
            country = countries.clone()
        answer = torch.empty(batch, dtype=torch.long)
        for t, (rel, kind, _, _) in enumerate(self.TYPES):
            rows = (qt == t).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            c = country[rows]
            ids[rows, rel_pos[rows]] = self.rel_tok[t]
            ids[rows, length - 1] = self.country_tok[c] if kind == "country" else self.filler_tok[filler[rows]]
            answer[rows] = {
                "traitA_of": lambda: self.trait_a_tok[self.trait_a[c]],
                "traitB_of": lambda: self.trait_b_tok[self.trait_b[c]],
                "region_of": lambda: self.region_tok[self.region[c]],
                "echo": lambda: self.filler_tok[filler[rows]],
                "combined_status_of": lambda: self.status_tok[self.status[c]],
            }[rel]()
        out = {"ids": ids, "answer": answer, "route": self.route_table[qt],
               "bridge": torch.tensor([t[3] for t in self.TYPES])[qt], "qtype": qt, "country": country}
        return {k: v.to(device) for k, v in out.items()}


COMP_VARIANTS = {  # name: (dense, use_bridge, branch embeddings, combined-query gradients reach Branches 0/1)
    "routed_bridge": (False, True, "private", False),       # isolation as specified: only the bridge learns XOR
    "routed_nobridge": (False, False, "private", True),     # given its best shot: may train its branch on XOR
    "dense": (True, True, "private", False),                # every branch + bridge, same isolation
    "nobridge_shared_emb": (False, False, "shared", True),  # leak demonstration: one shared embedding table
}


def grad_allow_mask(batch, combined_trains_branches: bool) -> torch.Tensor:
    """A query's gradient reaches only its designated branch(es)."""
    allow = batch["route"] > 0
    if not combined_trains_branches:
        allow[batch["qtype"] == CompositionalWorld.COMBINED] = False
    return allow


@torch.no_grad()
def evaluate_compositional(model: TypeInformedTreeNN, world: CompositionalWorld, args, device) -> Dict:
    model.eval()
    R = args.queries_per_country

    def run(qtype: int, countries: torch.Tensor, seed: int, **kwargs) -> Dict:
        reps = countries.repeat_interleave(R)
        b = world.sample(0, args.length, torch.Generator().manual_seed(seed), device, qtype=qtype, countries=reps)
        if kwargs.pop("shuffle_partner", False):
            kwargs["partner_perm"] = torch.roll(torch.arange(len(reps), device=device), shifts=R)  # next country
        out = model(b["ids"], **kwargs)
        cand = world.candidates[qtype].to(device)
        pred = cand[out["logits"][:, -1, cand].argmax(-1)]  # restricted to this query's answer set
        per_country = (pred == b["answer"]).float().view(-1, R).mean(1)
        boot = torch.Generator().manual_seed(seed + 7)
        idx = torch.randint(len(per_country), (1000, len(per_country)), generator=boot).to(device)
        means = per_country[idx].mean(1)
        return {"acc": per_country.mean().item(),
                "ci95": [means.quantile(0.025).item(), means.quantile(0.975).item()],
                "n_countries": len(countries), "chance": 1.0 / len(cand),
                "branch_execs_per_query": out["branch_execs"] / len(reps),
                "bridge_execs_per_query": out["bridge_execs"] / len(reps)}

    C = world.COMBINED
    res = {f"combined_{s}": run(C, world.countries_in[s], args.eval_seed + i) for i, s in enumerate(world.SPLITS)}
    test = world.countries_in["test"]
    res["traitA_test_countries"] = run(0, test, args.eval_seed + 10)
    res["traitB_test_countries"] = run(1, test, args.eval_seed + 11)
    res["region_all"] = run(2, torch.arange(world.n), args.eval_seed + 12)
    res["echo"] = run(3, torch.arange(len(world.filler_tok)), args.eval_seed + 13)
    t = res["combined_test"]
    res["combined_test"]["branch_bridge_macs_per_query"] = model.macs_per_query(
        args.length, t["branch_execs_per_query"], t["bridge_execs_per_query"])
    if model.use_bridge and not model.dense:
        res["controls"] = {
            "combined_test_bridge_off": run(C, test, args.eval_seed + 2, bridge="off"),
            "combined_test_shuffled_partner": run(C, test, args.eval_seed + 2, shuffle_partner=True),
        }
        reps = test.repeat_interleave(R)
        b = world.sample(0, args.length, torch.Generator().manual_seed(args.eval_seed + 2), device, qtype=C,
                         countries=reps)
        out, ref = model(b["ids"]), model(b["ids"], reference=True)
        res["exactness"] = {"max_abs_logit_diff_vs_run_everything": (out["logits"] - ref["logits"]).abs().max().item()}
        routed = out["top1"], out["top2"]
        res["combined_routing_pair_0_then_1"] = ((routed[0] == 0) & (routed[1] == 1)).float().mean().item()
        res["combined_bridge_fire_rate"] = out["bridge_fired"].float().mean().item()
    return res


def run_compositional(args, device) -> Dict:
    world = CompositionalWorld(n_countries=args.comp_countries, n_fillers=args.n_fillers)
    runs = []
    for seed in args.seeds:
        for variant, (dense, use_bridge, emb, trains_branches) in COMP_VARIANTS.items():
            torch.manual_seed(seed)
            model = TypeInformedTreeNN(world.vocab_size, args.d_model, args.d_stalk, args.d_edge,
                                       bridge_threshold=args.bridge_threshold, dense=dense, use_bridge=use_bridge,
                                       branch_embeddings=emb, router_context="context_only").to(device)
            gen = torch.Generator().manual_seed(20_000 + seed)  # same training stream for every variant
            t0 = time.time()
            optimize(model, args, lambda: world.sample(args.batch, args.length, gen, device),
                     lambda b, tb=trains_branches: {"grad_allow": grad_allow_mask(b, tb)},
                     f"compositional seed {seed} {variant:19s}")
            runs.append({"seed": seed, "variant": variant, "train_seconds": round(time.time() - t0, 1),
                         "eval": evaluate_compositional(model, world, args, device)})

    print(f"\n=== COMPOSITIONAL ISOLATION benchmark (mean ± sd over seeds {args.seeds}) ===")
    print(f"{len(world.countries_in['test'])} held-out test countries, {len(world.countries_in['val'])} validation, "
          f"{len(world.countries_in['train'])} train; balanced A, B and XOR in every split; chance = 0.5000")
    print(f"{'':46s}" + "".join(f"{v:>21s}" for v in COMP_VARIANTS))
    for name, path in [
        ("XOR on HELD-OUT test countries", ("eval", "combined_test", "acc")),
        ("XOR on held-out validation countries", ("eval", "combined_val", "acc")),
        ("XOR on train countries (seen)", ("eval", "combined_train", "acc")),
        ("Trait A on test countries", ("eval", "traitA_test_countries", "acc")),
        ("Trait B on test countries", ("eval", "traitB_test_countries", "acc")),
        ("Region (all countries)", ("eval", "region_all", "acc")),
        ("Echo", ("eval", "echo", "acc")),
        ("Branch runs per combined query", ("eval", "combined_test", "branch_execs_per_query")),
        ("Bridge runs per combined query", ("eval", "combined_test", "bridge_execs_per_query")),
        ("Branch+bridge MACs per combined query", ("eval", "combined_test", "branch_bridge_macs_per_query")),
    ]:
        print(f"{name:46s}" + "".join(f"{column(runs, v, *path):>21s}" for v in COMP_VARIANTS))
    print("Held-out XOR accuracy, 95% bootstrap interval over countries, per seed:")
    for r in runs:
        ct = r["eval"]["combined_test"]
        print(f"  seed {r['seed']} {r['variant']:21s} {ct['acc']:.4f}  [{ct['ci95'][0]:.4f}, {ct['ci95'][1]:.4f}]")
    print("Controls on routed_bridge (held-out XOR):")
    print(f"  bridge off at evaluation      {column(runs, 'routed_bridge', 'eval', 'controls', 'combined_test_bridge_off', 'acc')}")
    print(f"  bridge gets another country's Trait-B stalk  "
          f"{column(runs, 'routed_bridge', 'eval', 'controls', 'combined_test_shuffled_partner', 'acc')}")
    print(f"  routed to Branch 0 then 1     {column(runs, 'routed_bridge', 'eval', 'combined_routing_pair_0_then_1')}"
          f"   bridge fire rate {column(runs, 'routed_bridge', 'eval', 'combined_bridge_fire_rate')}")
    print(f"  exactness vs running every branch: max |logit diff| "
          f"{column(runs, 'routed_bridge', 'eval', 'exactness', 'max_abs_logit_diff_vs_run_everything')}")
    sparse = [r["eval"]["combined_test"]["branch_bridge_macs_per_query"] for r in runs if r["variant"] == "routed_bridge"]
    dense = [r["eval"]["combined_test"]["branch_bridge_macs_per_query"] for r in runs if r["variant"] == "dense"]
    if sparse and dense:
        print(f"Combined-query compute, routed_bridge / dense: {statistics.fmean(sparse) / statistics.fmean(dense):.3f}")
    return {"world": {"vocab_size": world.vocab_size, "types": CompositionalWorld.TYPES,
                      "split_sizes": {s: len(world.countries_in[s]) for s in world.SPLITS}},
            "variants": {k: dict(zip(("dense", "use_bridge", "branch_embeddings", "combined_trains_branches"), v))
                         for k, v in COMP_VARIANTS.items()},
            "runs": runs}


# =========================================================================== main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="both", choices=["routing", "compositional", "both"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--stress-length", type=int, default=10, help="routing benchmark: longer test queries")
    p.add_argument("--n-countries", type=int, default=64, help="routing benchmark world size")
    p.add_argument("--comp-countries", type=int, default=512, help="compositional benchmark world size")
    p.add_argument("--n-fillers", type=int, default=48)
    p.add_argument("--queries-per-country", type=int, default=16, help="compositional evaluation")
    p.add_argument("--eval-size", type=int, default=8192, help="routing evaluation")
    p.add_argument("--eval-seed", type=int, default=999_001)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-stalk", type=int, default=64)
    p.add_argument("--d-edge", type=int, default=32)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.3)
    p.add_argument("--alpha-brier", type=float, default=0.5)
    p.add_argument("--lambda-flops", type=float, default=0.05)
    p.add_argument("--bridge-threshold", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("runs/titnn"))
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    env = {"torch": torch.__version__, "python": platform.python_version(), "device": str(device),
           "platform": platform.platform()}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["capability"] = ".".join(map(str, torch.cuda.get_device_capability(device)))
    print("Environment:", json.dumps(env))
    out_dir = args.out / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if args.benchmark in ("routing", "both"):
        result = run_routing(args, device)
        (out_dir / "routing.json").write_text(json.dumps({"config": config, "environment": env, **result}, indent=2))
    if args.benchmark in ("compositional", "both"):
        result = run_compositional(args, device)
        (out_dir / "compositional.json").write_text(
            json.dumps({"config": config, "environment": env, **result}, indent=2))
    print(f"\nSaved results under {out_dir}")
    print("Read before quoting: synthetic toy tasks; MACs are counted from layer shapes, not measured speed; "
          "the information-theoretic cap holds only under the isolation listed in the docstring.")


if __name__ == "__main__":
    main()
