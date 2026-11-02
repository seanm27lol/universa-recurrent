"""C4 and T4: one recurrent controller trunk over the frozen backbone's states, with two different semantic modules.

**Example.** At the step after `FROM_TABLE(movies) END` for "Who is the director of the movie Sex, Drink and
Bloodshed?", the grammar's WHERE slot allows only predicate-family operators. The trunk reads the previous action, the
slot and the partial-plan features, updates its two GRU layers, attends over the question, schema and example tokens,
and the family classifier picks `predicate`. Then:
- **T4** runs only its predicate specialist (a two-layer MLP), which scores the predicate vocabulary (EQ wins) and
  emits pointer queries;
- **C4** runs its one shared MLP and masks its union vocabulary to the predicate family.

Everything else is literally the same module code: input projections, GRUs, cross-attention, embeddings, family
classifier, pointer scoring, masks, losses.

**The trunk** (width d = 512, backbone width D = 2560 for Qwen3.5-4B):

| Part | Computation |
|---|---|
| context memory | tokens: `LN(W_t x + E_region)`; schema items: `LN(W_i mean(x over the item's tokens) + E_kind)` |
| pointer keys | `K_item = W_ki memory_items`, `K_tok = W_kt memory_tokens` |
| initial state | `h0 = tanh(W_0 x_last)` for both GRU layers (the context's last token summarizes the request) |
| step input | `W_in [E_op(a_{t-1}); E_fam(f_{t-1}); E_slot; E_parent; W_f phi_t; W_a arg(a_{t-1})]` |
| recurrence | two stacked GRU layers, width 512 |
| cross-attention | 8 heads from the GRU output over the memory (tokens and items) |
| state | `s_t = LN(W_s [h_t; c_t])` (the 512-dimensional interface of the semantic module) |
| family classifier | `W_c s_t` over {schema, predicate, aggregate, projection, STOP}, masked by the grammar |
| pointers | `q . K / sqrt(d)`; the span end query is `q_end + W_e K_tok[start]` |

`phi_t` is `actions.Machine.features()` (slot, clause, family context, partial-plan counts, the stack of unfinished
subproblems).

**The semantic module** takes `s_t` and the chosen family and returns operator logits over the union vocabulary (the
other families' entries are -inf) and four pointer queries (table, column, span start, span end), each 512-dimensional:

| Arm | Module | Per-step active compute |
|---|---|---|
| T4 | four specialists `MLP_f: 512 -> 512 -> |V_f| + 4 x 512`, one per family; rows are grouped by family and only the chosen head runs on its group | one head |
| c4_active | one `MLP: 512 -> 512 -> |V| + 4 x 512` over the union vocabulary, masked to the family | about one head |
| c4_params | the same MLP with the hidden width that matches T4's total trainable parameters (within 1%) | larger |

The exact parameter and MAC tables come from `parameter_report` and `step_macs` (see docs/ARCHITECTURE.md).

**Limits.** The trunk never sees gold. Teacher forcing feeds it the gold prefix in training; greedy decoding feeds its
own choices at inference. Nothing here shows that T4's specialization helps; that is the question the study asks.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import actions as A
from . import grounding as G
from .encode import REGIONS

ARMS = ("T4", "c4_active", "c4_params")
N_POINTERS = 4                  # table, column, span start, span end
PTR_TABLE, PTR_COLUMN, PTR_START, PTR_END = range(N_POINTERS)
PTR_KIND_ID = {None: 0, "table": 1, "column": 2, "span": 3}
UNION_SIZE = len(A.UNION)
BOS_OP = UNION_SIZE             # the previous-operator id at step 0
BOS_FAMILY = A.N_CLASSES        # the previous-family id at step 0
NEG = -1e9


@dataclass(frozen=True)
class ControllerConfig:
    """Every architectural choice; identical for the arms except `arm` (and the c4_params width derived from it)."""
    arm: str
    backbone_width: int
    d_model: int = 512
    gru_layers: int = 2
    attention_heads: int = 8
    op_embed: int = 256
    family_embed: int = 64
    slot_embed: int = 64
    parent_embed: int = 128
    feature_embed: int = 128
    arg_embed: int = 256
    head_hidden: int = 512      # T4 specialists and c4_active
    c4_params_hidden: "int | None" = None  # set by `matched_config`
    grammar: str = A.GRAMMAR_VERSION
    # "v1": the pilot's controller, unchanged (its state dict and losses are the pilot's). "v2": relational grounding
    # (`grounding.py`): the schema encoder, item features and plan-relative pointer relations, all in the trunk.
    grounding: str = "v1"
    encoder_layers: int = 2     # grounding v2 only
    encoder_ff: int = 1024      # grounding v2 only

    def as_dict(self) -> dict:
        return asdict(self)


class MLP(nn.Module):
    """Two layers with a 512-dimensional input interface: Linear -> GELU -> Linear."""

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Multi-head attention from controller states to the request memory. Keys and values are projected once per
    request (`project`), so a decoding step costs the query projection, the scores, the weighted sum and the output
    projection."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def project(self, memory):
        return self.k(memory), self.v(memory)

    def forward(self, x, keys, values, pad):
        B, S, d = x.shape
        L, h = keys.shape[1], self.heads
        q = self.q(x).view(B, S, h, d // h).transpose(1, 2)
        k = keys.view(B, L, h, d // h).transpose(1, 2)
        v = values.view(B, L, h, d // h).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d // h)
        weights = scores.masked_fill(pad[:, None, None, :], NEG).softmax(-1)
        return self.o((weights @ v).transpose(1, 2).reshape(B, S, d))


class RelationLayer(nn.Module):
    """One pre-norm relation-aware attention layer: schema items attend to items and to question/evidence tokens,
    with a learned bias per head for the relation of each (query, key) pair; then a feed-forward block."""

    def __init__(self, d: int, heads: int, ff: int, n_relations: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.relation_bias = nn.Embedding(n_relations, heads)
        nn.init.zeros_(self.relation_bias.weight)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = MLP(d, ff, d)

    def forward(self, items, tokens, relations, pad):
        """items [B, N, d]; tokens [B, Tq, d] (keys only); relations [B, N, N + Tq] long; pad [B, N + Tq] bool."""
        B, N, d = items.shape
        h = self.heads
        x = self.norm(items)
        keys_in = torch.cat([x, tokens], dim=1)
        L = keys_in.shape[1]
        q = self.q(x).view(B, N, h, d // h).transpose(1, 2)
        k = self.k(keys_in).view(B, L, h, d // h).transpose(1, 2)
        v = self.v(keys_in).view(B, L, h, d // h).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d // h) + self.relation_bias(relations).permute(0, 3, 1, 2)
        weights = scores.masked_fill(pad[:, None, None, :], NEG).softmax(-1)
        items = items + self.o((weights @ v).transpose(1, 2).reshape(B, N, d))
        return items + self.ff(self.ff_norm(items))


class SchemaEncoder(nn.Module):
    """Grounding v2: relation-aware layers over the schema items (see `grounding.py`), then a final LayerNorm."""

    def __init__(self, cfg: "ControllerConfig"):
        super().__init__()
        self.layers = nn.ModuleList(RelationLayer(cfg.d_model, cfg.attention_heads, cfg.encoder_ff, G.N_RELATIONS)
                                    for _ in range(cfg.encoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, items, item_mask, tokens, token_mask, relations):
        pad = torch.cat([~item_mask, ~token_mask], dim=1)
        for layer in self.layers:
            items = layer(items, tokens, relations, pad)
        return self.norm(items)


class Trunk(nn.Module):
    """The shared controller trunk (identical code and configuration for C4 and T4)."""

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        d, D = cfg.d_model, cfg.backbone_width
        self.cfg = cfg
        self.token_proj = nn.Linear(D, d)
        self.item_proj = nn.Linear(D, d)
        self.region_embed = nn.Embedding(len(REGIONS), d)
        self.kind_embed = nn.Embedding(2, d)
        self.memory_norm = nn.LayerNorm(d)
        self.key_item = nn.Linear(d, d)
        self.key_token = nn.Linear(d, d)
        self.init_state = nn.Linear(D, cfg.gru_layers * d)
        self.op_embed = nn.Embedding(UNION_SIZE + 1, cfg.op_embed)
        self.family_embed = nn.Embedding(A.N_CLASSES + 1, cfg.family_embed)
        self.slot_embed = nn.Embedding(len(A.SLOT_KINDS), cfg.slot_embed)
        self.parent_embed = nn.Embedding(len(A.OP_NAMES) + 1, cfg.parent_embed)
        self.feature_proj = nn.Linear(A.N_FEATURES, cfg.feature_embed)
        self.arg_proj = nn.Linear(d, cfg.arg_embed)
        width = cfg.op_embed + cfg.family_embed + cfg.slot_embed + cfg.parent_embed + cfg.feature_embed + cfg.arg_embed
        self.input_proj = nn.Linear(width, d)
        self.gru = nn.GRU(d, d, num_layers=cfg.gru_layers, batch_first=True)
        self.attention = CrossAttention(d, cfg.attention_heads)
        self.state_proj = nn.Linear(2 * d, d)
        self.state_norm = nn.LayerNorm(d)
        self.family_head = nn.Linear(d, A.N_CLASSES)
        self.end_condition = nn.Linear(d, d)
        if cfg.grounding == "v2":
            # drawn from a generator forked off the seed, so the main stream is untouched: under one seed the v1 and
            # v2 controllers share every common initialization (trunk and semantic module), and the v2 parts are
            # identical across the arms
            seed = torch.initial_seed()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((seed * 1_000_003 + 2) % (2 ** 63))
                self.item_features = nn.Linear(G.N_ITEM_FEATURES, d)
                self.schema_encoder = SchemaEncoder(cfg)
            # one embedding per (pointer: table, column) and relation bit; the bias is (W_r^T q) . bits
            self.pointer_relation = nn.Parameter(torch.zeros(2, A.N_POINTER_RELATIONS, d))
        elif cfg.grounding != "v1":
            raise ValueError(f"grounding must be one of {A.GROUNDINGS}")

    # memory: once per request
    def memory(self, batch: "Batch") -> dict:
        # the backbone states arrive in their stored dtype (BF16) and are widened here, on the model's device: exact,
        # and four times less host-to-device traffic than widening first
        tok = self.token_proj(batch.tokens.float()) + self.region_embed(batch.token_types)
        itm = self.item_proj(batch.items) + self.kind_embed(batch.item_kinds)
        if self.cfg.grounding == "v2":
            itm = itm + self.item_features(batch.item_features)
        tok, itm = self.memory_norm(tok), self.memory_norm(itm)
        if self.cfg.grounding == "v2":
            d = self.cfg.d_model
            qe = tok.gather(1, batch.qe_index.unsqueeze(-1).expand(-1, -1, d))
            relations = torch.cat([batch.item_relations, batch.qe_relations.unsqueeze(1).expand(
                -1, itm.shape[1], -1)], dim=2)
            itm = self.schema_encoder(itm, batch.item_mask, qe, batch.qe_mask, relations)
        memory = torch.cat([tok, itm], dim=1)
        pad = torch.cat([~batch.token_mask, ~batch.item_mask], dim=1)
        h0 = torch.tanh(self.init_state(batch.summary)).view(-1, self.cfg.gru_layers, self.cfg.d_model)
        keys, values = self.attention.project(memory)
        return {"tokens": tok, "items": itm, "pad": pad, "keys": keys, "values": values,
                "key_item": self.key_item(itm), "key_token": self.key_token(tok),
                "h0": h0.transpose(0, 1).contiguous()}

    def arg_vectors(self, mem: dict, kind, a, b):
        """The previous action's pointer argument: the chosen item's memory vector, or the mean of the span's first
        and last token vectors; zero for actions without a pointer. kind/a/b: [B, S] long."""
        B, S = kind.shape
        d = self.cfg.d_model
        item = mem["items"].gather(1, a.clamp(min=0, max=mem["items"].shape[1] - 1).unsqueeze(-1).expand(B, S, d))
        tok_a = mem["tokens"].gather(1, a.clamp(min=0, max=mem["tokens"].shape[1] - 1).unsqueeze(-1).expand(B, S, d))
        tok_b = mem["tokens"].gather(1, b.clamp(min=0, max=mem["tokens"].shape[1] - 1).unsqueeze(-1).expand(B, S, d))
        is_item = ((kind == 1) | (kind == 2)).unsqueeze(-1)
        is_span = (kind == 3).unsqueeze(-1)
        return torch.where(is_item, item, torch.zeros_like(item)) + torch.where(is_span, (tok_a + tok_b) / 2,
                                                                                torch.zeros_like(item))

    def step_inputs(self, mem: dict, steps: dict):
        parts = [self.op_embed(steps["prev_op"]), self.family_embed(steps["prev_family"]),
                 self.slot_embed(steps["slot"]), self.parent_embed(steps["parent"]),
                 self.feature_proj(steps["features"]),
                 self.arg_proj(self.arg_vectors(mem, steps["prev_kind"], steps["prev_a"], steps["prev_b"]))]
        return self.input_proj(torch.cat(parts, dim=-1))

    def states(self, mem: dict, steps: dict, hidden=None):
        """[B, S, d] controller states for S steps (S = all steps with teacher forcing; 1 when decoding)."""
        x = self.step_inputs(mem, steps)
        h, hidden = self.gru(x, mem["h0"] if hidden is None else hidden)
        c = self.attention(h, mem["keys"], mem["values"], mem["pad"])
        return self.state_norm(self.state_proj(torch.cat([h, c], dim=-1))), hidden

    def pointer_scores(self, mem: dict, queries, start_index=None, relations=None):
        """queries [B, S, 4, d] -> (item scores for the table and column queries [B, S, 2, N], start [B, S, T],
        end [B, S, T]). The end query adds W_e K_tok[start] (the gold start in training, the chosen one decoding).
        Grounding v2: `relations` [B, S, N, R] (0/1, the step's plan-relative pointer relations) add
        (W_r^T q) . bits to each item's score."""
        scale = 1.0 / math.sqrt(self.cfg.d_model)
        items = torch.einsum("bspd,bnd->bspn", queries[:, :, :2], mem["key_item"]) * scale
        if self.cfg.grounding == "v2" and relations is not None:
            qr = torch.einsum("bspd,prd->bspr", queries[:, :, :2], self.pointer_relation) * scale
            items = items + torch.einsum("bspr,bsnr->bspn", qr, relations.float())
        start = torch.einsum("bsd,btd->bst", queries[:, :, PTR_START], mem["key_token"]) * scale
        end = None
        if start_index is not None:
            B, S = start_index.shape
            d, T = self.cfg.d_model, mem["key_token"].shape[1]
            index = start_index.clamp(min=0, max=T - 1)
            k_start = mem["key_token"].gather(1, index.unsqueeze(-1).expand(B, S, d))
            q_end = queries[:, :, PTR_END] + self.end_condition(k_start)
            end = torch.einsum("bsd,btd->bst", q_end, mem["key_token"]) * scale
        return items, start, end


class Specialists(nn.Module):
    """T4: four semantic specialists. Rows are grouped by their family and each head runs only on its own group."""

    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.heads = nn.ModuleList(MLP(d, cfg.head_hidden, len(A.FAMILY_OPS[f]) + N_POINTERS * d) for f in range(4))
        self.calls: list = []   # (family, rows) per head invocation, for work accounting and the tests

    def forward(self, state, family):
        R = state.shape[0]
        logits = state.new_full((R, UNION_SIZE), NEG)
        queries = state.new_zeros((R, N_POINTERS, self.d))
        for f in range(4):
            rows = (family == f).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            out = self.heads[f](state.index_select(0, rows))
            self.calls.append((f, int(rows.numel())))
            lo, hi = A.FAMILY_SLICES[f]
            logits[rows, lo:hi] = out[:, :hi - lo]
            queries[rows] = out[:, hi - lo:].reshape(-1, N_POINTERS, self.d)
        return logits, queries

    def executed(self, family) -> list:
        """The head each row runs (its family), or None for STOP."""
        return [int(f) if int(f) < 4 else None for f in family]


class Shared(nn.Module):
    """C4: one shared semantic MLP over the union vocabulary, masked to the chosen family."""

    def __init__(self, cfg: ControllerConfig, hidden: int):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.mlp = MLP(d, hidden, UNION_SIZE + N_POINTERS * d)
        self.calls: list = []
        family_of_entry = torch.tensor([f for f, _ in A.UNION])
        self.register_buffer("family_of_entry", family_of_entry, persistent=False)

    def forward(self, state, family):
        R = state.shape[0]
        rows = (family < 4).nonzero(as_tuple=True)[0]
        logits = state.new_full((R, UNION_SIZE), NEG)
        queries = state.new_zeros((R, N_POINTERS, self.d))
        if rows.numel():
            out = self.mlp(state.index_select(0, rows))
            self.calls.append(("shared", int(rows.numel())))
            own = self.family_of_entry.unsqueeze(0) == family.index_select(0, rows).unsqueeze(1)
            logits[rows] = out[:, :UNION_SIZE].masked_fill(~own, NEG)
            queries[rows] = out[:, UNION_SIZE:].reshape(-1, N_POINTERS, self.d)
        return logits, queries

    def executed(self, family) -> list:
        return ["shared" if int(f) < 4 else None for f in family]


def semantic_parameters(cfg: ControllerConfig, arm: str, hidden: "int | None" = None) -> int:
    d, P = cfg.d_model, N_POINTERS * cfg.d_model
    if arm == "T4":
        return sum(d * cfg.head_hidden + cfg.head_hidden + cfg.head_hidden * (len(A.FAMILY_OPS[f]) + P)
                   + len(A.FAMILY_OPS[f]) + P for f in range(4))
    h = hidden if hidden is not None else cfg.head_hidden
    return d * h + h + h * (UNION_SIZE + P) + UNION_SIZE + P


def c4_params_hidden(cfg: ControllerConfig) -> int:
    """The shared MLP's hidden width whose parameter count matches T4's four specialists (to the nearest unit)."""
    target = semantic_parameters(cfg, "T4")
    d, P = cfg.d_model, N_POINTERS * cfg.d_model
    return round((target - UNION_SIZE - P) / (d + 1 + UNION_SIZE + P))


def matched_config(arm: str, backbone_width: int, **overrides) -> ControllerConfig:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    cfg = ControllerConfig(arm=arm, backbone_width=backbone_width, **overrides)
    if arm == "c4_params":
        cfg = ControllerConfig(**{**cfg.as_dict(), "c4_params_hidden": c4_params_hidden(cfg)})
    return cfg


class Controller(nn.Module):
    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        if cfg.arm == "T4":
            self.semantic = Specialists(cfg)
        elif cfg.arm == "c4_active":
            self.semantic = Shared(cfg, cfg.head_hidden)
        elif cfg.arm == "c4_params":
            self.semantic = Shared(cfg, cfg.c4_params_hidden or c4_params_hidden(cfg))
        else:
            raise ValueError(cfg.arm)

    # teacher forcing
    def losses(self, batch: "Batch", weights: "dict | None" = None) -> dict:
        """Summed loss terms over the batch's valid steps (identical terms and targets for every arm):
        family CE (every step, STOP included), operator CE (steps whose gold family is not STOP) and pointer CE
        (table/column, or span start + span end). `total` = weighted sum; divide by `steps` (or by the effective
        batch's step count when accumulating) for the mean."""
        weights = weights or {"family": 1.0, "op": 1.0, "pointer": 1.0}
        st = batch.steps
        mem = self.trunk.memory(batch)
        state, _ = self.trunk.states(mem, st)
        valid = st["valid"]
        fam_logits = self.trunk.family_head(state).masked_fill(~st["legal_family"], NEG)
        fam_loss = F.cross_entropy(fam_logits[valid], st["family"][valid], reduction="sum")
        rows = valid & (st["family"] < 4)
        b_idx, s_idx = rows.nonzero(as_tuple=True)
        op_logits, q_rows = self.semantic(state[b_idx, s_idx], st["family"][b_idx, s_idx])
        op_logits = op_logits.masked_fill(~st["legal_union"][b_idx, s_idx], NEG)
        op_loss = F.cross_entropy(op_logits, st["union"][b_idx, s_idx], reduction="sum")
        B, S = valid.shape
        queries = state.new_zeros((B, S, N_POINTERS, self.cfg.d_model))
        queries[b_idx, s_idx] = q_rows
        items, start, end = self.trunk.pointer_scores(mem, queries, st["ptr_target"], st.get("ptr_rel"))
        kind = st["ptr_kind"]
        ptr_loss = state.new_zeros(())
        n_ptr = 0
        for k, slot in ((1, 0), (2, 1)):
            sel = valid & (kind == k)
            if sel.any():
                scores = items[:, :, slot][sel].masked_fill(~st["ptr_items"][sel], NEG)
                ptr_loss = ptr_loss + F.cross_entropy(scores, st["ptr_target"][sel], reduction="sum")
                n_ptr += int(sel.sum())
        sel = valid & (kind == 3)
        if sel.any():
            s_scores = start[sel].masked_fill(~st["span_start"][sel], NEG)
            e_scores = end[sel].masked_fill(~st["span_end"][sel], NEG)
            ptr_loss = ptr_loss + F.cross_entropy(s_scores, st["ptr_target"][sel], reduction="sum") \
                + F.cross_entropy(e_scores, st["ptr_end"][sel], reduction="sum")
            n_ptr += int(sel.sum())
        total = weights["family"] * fam_loss + weights["op"] * op_loss + weights["pointer"] * ptr_loss
        return {"total": total, "family": fam_loss, "op": op_loss, "pointer": ptr_loss, "steps": int(valid.sum()),
                "op_steps": int(rows.sum()), "pointer_steps": n_ptr}


# --- inputs ----------------------------------------------------------------------------------------------------------
@dataclass
class Prepared:
    """One request, ready for the controller. `steps` (teacher-forcing targets) only exist for training.

    The token states stay in the backbone's dtype (BF16), the dtype they are cached in, and are widened on the model's
    device (`Trunk.memory`), so the values are exactly the float32 ones. With `tokens_file` set they are not held in
    memory at all: `token_states()` reads them from the immutable encoding cache when a batch needs them. The training
    pool's states (about 126 GB) do not fit in this host's memory. `step_cache` holds the dense teacher-forcing
    tensors, built once at this example's own sizes."""
    task_id: str
    tokens: "torch.Tensor | None"  # [T, D] backbone states, or None when read from `tokens_file`
    token_types: torch.Tensor     # [T] region ids
    items: torch.Tensor           # [N, D] pooled item states (float32)
    item_kinds: torch.Tensor      # [N] 0 table, 1 column
    summary: torch.Tensor         # [D] the last token's state (float32)
    schema: A.SchemaItems
    spans: A.SpanIndex
    steps: "list | None" = None
    meta: dict = field(default_factory=dict)
    step_cache: "dict | None" = None
    tokens_file: "str | None" = None
    tokens_crc32: "int | None" = None  # of the states as read at prepare time; every later read is checked against it
    tokens_dtype: "str | None" = None  # the states' dtype at prepare time; every later read must match it and the shape
    grounding: str = "v1"
    item_features: "torch.Tensor | None" = None   # v2: [N, F] 0/1
    item_relations: "torch.Tensor | None" = None  # v2: [N, N] relation ids (uint8)
    qe_index: "torch.Tensor | None" = None        # v2: [Tq] indices of the question and evidence tokens
    qe_relations: "torch.Tensor | None" = None    # v2: [Tq] their relation ids (question or evidence token)

    @property
    def n_tokens(self) -> int:
        return int(self.token_types.shape[0])

    def token_states(self) -> torch.Tensor:
        if self.tokens is not None:
            return self.tokens
        from safetensors import safe_open
        with safe_open(self.tokens_file, framework="pt") as f:
            # get_tensor maps the file; clone() reads it into ordinary memory here (on the prefetch thread). A mapped
            # tensor copied to the GPU directly was read page by page during the copy: 5.3 s per update, measured.
            states = f.get_tensor("states").clone()
        expected = (self.n_tokens, int(self.items.shape[1]))
        if tuple(states.shape) != expected or (self.tokens_dtype and str(states.dtype) != self.tokens_dtype):
            raise RuntimeError(f"{self.task_id}: {self.tokens_file} holds {states.dtype} {tuple(states.shape)}, "
                               f"expected {self.tokens_dtype} {expected}")
        if self.tokens_crc32 is not None and _crc32(states) != self.tokens_crc32:
            raise RuntimeError(f"{self.task_id}: {self.tokens_file} changed since it was prepared (CRC32 mismatch)")
        return states

    def steps_dense(self) -> dict:
        if self.step_cache is None:
            self.step_cache = step_tensors(self.steps, self.items.shape[0], self.n_tokens, self.grounding)
        return self.step_cache


def _crc32(states: torch.Tensor) -> int:
    import zlib
    return zlib.crc32(states.contiguous().view(torch.uint8).numpy())


def grounding_fields(encoding) -> dict:
    """Grounding v2's per-request inputs: item features and relations, and the question/evidence token keys."""
    ctx = encoding.context
    question = ctx.text[slice(*ctx.regions["question"])]
    evidence = ctx.text[slice(*ctx.regions["evidence"])]
    qe, qe_rel = G.question_evidence_tokens(encoding.token_types)
    return {"item_features": torch.tensor(G.item_features(encoding.schema, question, evidence)),
            "item_relations": torch.tensor(G.item_relations(encoding.schema), dtype=torch.uint8),
            "qe_index": torch.tensor(qe, dtype=torch.long), "qe_relations": torch.tensor(qe_rel, dtype=torch.long)}


def as_grounding(example: Prepared, encoding, steps, grounding: str) -> Prepared:
    """The same request prepared for another grounding, sharing its state tensors (no second read of the states)."""
    import dataclasses
    extra = grounding_fields(encoding) if grounding == "v2" else {
        k: None for k in ("item_features", "item_relations", "qe_index", "qe_relations")}
    return dataclasses.replace(example, steps=steps, step_cache=None, grounding=grounding, **extra)


def prepare(encoding, steps=None, tokens_file: "str | None" = None, grounding: str = "v1") -> Prepared:
    """With `tokens_file` (the encoding's cache file), the token states are not kept; they are read per batch.
    Grounding v2 also derives the item relations and features (`grounding.py`) from the schema and the question."""
    states = encoding.states
    kinds = torch.tensor([0 if it.kind == "table" else 1 for it in encoding.schema.items], dtype=torch.long)
    items = encoding.item_vectors()
    dense = step_tensors(steps, items.shape[0], states.shape[0], grounding) if steps is not None else None
    extra = {}
    if grounding == "v2":
        extra = grounding_fields(encoding)
    elif grounding != "v1":
        raise ValueError(grounding)
    return Prepared(task_id=encoding.task_id, tokens=None if tokens_file else states,
                    token_types=torch.tensor(encoding.token_types), items=items, item_kinds=kinds,
                    summary=states[-1].float().clone(), schema=encoding.schema, spans=encoding.spans, steps=steps,
                    meta={"cost": dict(encoding.cost)}, step_cache=dense, tokens_file=tokens_file,
                    tokens_crc32=_crc32(states) if tokens_file else None, tokens_dtype=str(states.dtype),
                    grounding=grounding, **extra)


@dataclass
class Batch:
    task_ids: list
    tokens: torch.Tensor
    token_types: torch.Tensor
    token_mask: torch.Tensor
    items: torch.Tensor
    item_kinds: torch.Tensor
    item_mask: torch.Tensor
    summary: torch.Tensor
    steps: "dict | None" = None
    item_features: "torch.Tensor | None" = None   # grounding v2 (see Prepared)
    item_relations: "torch.Tensor | None" = None  # [B, N, N] long
    qe_index: "torch.Tensor | None" = None        # [B, Tq] long (padding points at token 0, masked)
    qe_relations: "torch.Tensor | None" = None    # [B, Tq] long
    qe_mask: "torch.Tensor | None" = None         # [B, Tq] bool

    def to(self, device):
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.__dict__.items() if k != "steps"}
        steps = None if self.steps is None else {k: v.to(device) for k, v in self.steps.items()}
        return Batch(**moved, steps=steps)


def _arg_fields(action: "A.Action | None"):
    if action is None or action.arg is None:
        return 0, 0, 0
    kind = PTR_KIND_ID[A.POINTER_KIND[action.op]]
    if kind == 3:
        return kind, action.arg[0], action.arg[1]
    return kind, action.arg, 0


def step_tensors(steps: list, n_items: int, n_tokens: int, grounding: str = "v1") -> dict:
    """Dense teacher-forcing tensors for one example (S steps). Grounding v2 adds `ptr_rel` [S, N, R]: each pointer
    step's plan-relative relation bits for every candidate item."""
    S = len(steps)
    out = {
        "prev_op": torch.full((S,), BOS_OP, dtype=torch.long), "prev_family": torch.full((S,), BOS_FAMILY,
                                                                                          dtype=torch.long),
        "prev_kind": torch.zeros(S, dtype=torch.long), "prev_a": torch.zeros(S, dtype=torch.long),
        "prev_b": torch.zeros(S, dtype=torch.long), "slot": torch.zeros(S, dtype=torch.long),
        "parent": torch.zeros(S, dtype=torch.long), "features": torch.zeros(S, A.N_FEATURES),
        "family": torch.zeros(S, dtype=torch.long), "union": torch.zeros(S, dtype=torch.long),
        "legal_family": torch.zeros(S, A.N_CLASSES, dtype=torch.bool),
        "legal_union": torch.zeros(S, UNION_SIZE, dtype=torch.bool),
        "ptr_kind": torch.zeros(S, dtype=torch.long), "ptr_target": torch.zeros(S, dtype=torch.long),
        "ptr_end": torch.zeros(S, dtype=torch.long), "ptr_items": torch.zeros(S, n_items, dtype=torch.bool),
        "span_start": torch.zeros(S, n_tokens, dtype=torch.bool), "span_end": torch.zeros(S, n_tokens,
                                                                                            dtype=torch.bool),
        "valid": torch.ones(S, dtype=torch.bool),
    }
    prev = None
    for t, st in enumerate(steps):
        if t > 0:
            out["prev_op"][t] = steps[t - 1].union_id
            out["prev_family"][t] = steps[t - 1].family
            out["prev_kind"][t], out["prev_a"][t], out["prev_b"][t] = _arg_fields(prev)
        out["slot"][t] = st.features["slot"]
        out["parent"][t] = st.features["parent_op"]
        out["features"][t] = torch.tensor(st.features["vector"])
        out["family"][t] = st.family
        out["union"][t] = max(st.union_id, 0)
        out["legal_family"][t] = torch.tensor(st.legal_families)
        if st.legal_union_ids:
            out["legal_union"][t, st.legal_union_ids] = True
        kind = PTR_KIND_ID[st.pointer]
        out["ptr_kind"][t] = kind
        if kind in (1, 2):
            out["ptr_target"][t] = st.target
            out["ptr_items"][t, list(st.pointer_legal)] = True
        elif kind == 3:
            out["ptr_target"][t], out["ptr_end"][t] = st.target
            out["span_start"][t, list(st.pointer_legal)] = True
            out["span_end"][t, list(st.end_legal)] = True
        prev = st.action
    if grounding == "v2":
        out["ptr_rel"] = relation_bits(steps, n_items)
    return out


def relation_bits(steps: list, n_items: int) -> torch.Tensor:
    """[S, N, R] bool from each step's `relations` {item: bitmask}."""
    rel = torch.zeros(len(steps), n_items, A.N_POINTER_RELATIONS, dtype=torch.bool)
    for t, st in enumerate(steps):
        for i, bits in (getattr(st, "relations", None) or {}).items():
            for r in range(A.N_POINTER_RELATIONS):
                if bits >> r & 1:
                    rel[t, i, r] = True
    return rel


# dense teacher-forcing tensors whose trailing axis is the example's item or token count (padded per batch); `ptr_rel`
# has the item axis second to last
_ITEM_AXIS, _TOKEN_AXIS = ("ptr_items",), ("span_start", "span_end")


def collate(examples: list, device="cpu", token_states: "dict | None" = None) -> Batch:
    """Pad a list of Prepared into one Batch on `device` (steps included when every example has them). The padded
    buffers are allocated on the device and each example is copied in once. `token_states` may supply already loaded
    states by task id (the training prefetcher)."""
    dev = torch.device(device)
    B = len(examples)
    T = max(e.n_tokens for e in examples)
    N = max(e.items.shape[0] for e in examples)
    D = examples[0].items.shape[1]
    states = [(token_states or {}).get(e.task_id) for e in examples]
    states = [s if s is not None else e.token_states() for s, e in zip(states, examples)]
    if len({s.dtype for s in states}) > 1:
        raise ValueError(f"a batch mixes state dtypes {sorted(str(s.dtype) for s in states)}; widening would be lossy")
    tokens = torch.zeros(B, T, D, dtype=states[0].dtype, device=dev)
    token_types = torch.zeros(B, T, dtype=torch.long, device=dev)
    token_mask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    items = torch.zeros(B, N, D, device=dev)
    item_kinds = torch.zeros(B, N, dtype=torch.long, device=dev)
    item_mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
    summary = torch.zeros(B, D, device=dev)
    for b, (e, st) in enumerate(zip(examples, states)):
        t, n = e.n_tokens, e.items.shape[0]
        tokens[b, :t] = st.to(dev, non_blocking=True)
        token_types[b, :t] = e.token_types.to(dev)
        token_mask[b, :t] = True
        items[b, :n] = e.items.to(dev)
        item_kinds[b, :n] = e.item_kinds.to(dev)
        item_mask[b, :n] = True
        summary[b] = e.summary.to(dev)
    steps = None
    if all(e.steps is not None for e in examples):
        per = [e.steps_dense() for e in examples]
        S = max(p["valid"].shape[0] for p in per)
        steps = {}
        for key in per[0]:
            tail = (N,) if key in _ITEM_AXIS else (T,) if key in _TOKEN_AXIS else \
                (N, A.N_POINTER_RELATIONS) if key == "ptr_rel" else tuple(per[0][key].shape[1:])
            buf = torch.zeros((B, S) + tail, dtype=per[0][key].dtype, device=dev)
            for b, p in enumerate(per):
                v = p[key]
                index = (b, slice(0, v.shape[0])) + tuple(slice(0, k) for k in v.shape[1:])
                buf[index] = v.to(dev)
            steps[key] = buf
    extra = {}
    if len({e.grounding for e in examples}) > 1:
        raise ValueError("a batch mixes grounding versions")
    if examples[0].grounding == "v2":
        Q = max(1, max(e.qe_index.shape[0] for e in examples))
        F = examples[0].item_features.shape[1]
        extra = {"item_features": torch.zeros(B, N, F, device=dev),
                 "item_relations": torch.zeros(B, N, N, dtype=torch.long, device=dev),
                 "qe_index": torch.zeros(B, Q, dtype=torch.long, device=dev),
                 "qe_relations": torch.zeros(B, Q, dtype=torch.long, device=dev),
                 "qe_mask": torch.zeros(B, Q, dtype=torch.bool, device=dev)}
        for b, e in enumerate(examples):
            n, q = e.items.shape[0], e.qe_index.shape[0]
            extra["item_features"][b, :n] = e.item_features.to(dev)
            extra["item_relations"][b, :n, :n] = e.item_relations.to(dev).long()
            extra["qe_index"][b, :q] = e.qe_index.to(dev)
            extra["qe_relations"][b, :q] = e.qe_relations.to(dev)
            extra["qe_mask"][b, :q] = True
    return Batch(task_ids=[e.task_id for e in examples], tokens=tokens, token_types=token_types,
                 token_mask=token_mask, items=items, item_kinds=item_kinds, item_mask=item_mask, summary=summary,
                 steps=steps, **extra)


# --- parameters and compute -------------------------------------------------------------------------------------------
def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(model: Controller) -> dict:
    sem = model.semantic
    out = {"arm": model.cfg.arm, "trunk": count(model.trunk), "semantic": count(sem), "total": count(model)}
    if isinstance(sem, Specialists):
        out["per_head"] = {A.FAMILIES[f]: count(sem.heads[f]) for f in range(4)}
        out["semantic_active_max"] = max(out["per_head"].values())
    else:
        out["semantic_active_max"] = out["semantic"]
        out["hidden"] = sem.mlp.fc1.out_features
    return out


def semantic_step_macs(cfg: ControllerConfig, arm: str, family: int) -> int:
    """Multiply-accumulates of the semantic module for one step of the given family (weights only)."""
    d, P = cfg.d_model, N_POINTERS * cfg.d_model
    if arm == "T4":
        return d * cfg.head_hidden + cfg.head_hidden * (len(A.FAMILY_OPS[family]) + P)
    h = cfg.head_hidden if arm == "c4_active" else (cfg.c4_params_hidden or c4_params_hidden(cfg))
    return d * h + h * (UNION_SIZE + P)


def step_macs(cfg: ControllerConfig, family: int, n_tokens: int, n_items: int, pointer: "str | None" = None) -> dict:
    """Per-step active compute, split into trunk, semantic module and pointer scoring (weights only; activations,
    softmax and embeddings lookups are not counted)."""
    d = cfg.d_model
    width = cfg.op_embed + cfg.family_embed + cfg.slot_embed + cfg.parent_embed + cfg.feature_embed + cfg.arg_embed
    L = n_tokens + n_items
    trunk = (A.N_FEATURES * cfg.feature_embed + d * cfg.arg_embed + width * d          # step input
             + cfg.gru_layers * 3 * (d * d + d * d)                                      # GRU gates
             + d * d + 2 * L * d + d * d                                                 # attention q, scores, values, out
             + 2 * d * d + d * A.N_CLASSES)                                              # state, family classifier
    semantic = semantic_step_macs(cfg, cfg.arm, family) if family < 4 else 0
    ptr = {None: 0, "table": n_items * d, "column": n_items * d, "span": 2 * n_tokens * d + d * d}[pointer]
    if cfg.grounding == "v2" and pointer in ("table", "column"):
        ptr += A.N_POINTER_RELATIONS * d + n_items * A.N_POINTER_RELATIONS  # (W_r^T q), then . bits per item
    return {"trunk": trunk, "semantic": semantic, "pointer": ptr, "total": trunk + semantic + ptr}


def request_macs(cfg: ControllerConfig, n_tokens: int, n_items: int, n_qe: int = 0) -> int:
    """Once per request: memory projections, pointer keys, attention keys and values, initial state; for grounding
    v2 also the item features and the schema encoder over the items and the n_qe question/evidence tokens."""
    d, D = cfg.d_model, cfg.backbone_width
    L = n_tokens + n_items
    total = (n_tokens + n_items) * D * d + (n_tokens + n_items) * d * d + 2 * L * d * d + D * cfg.gru_layers * d
    if cfg.grounding == "v2":
        keys = n_items + n_qe
        per_layer = (n_items * d * d + 2 * keys * d * d + 2 * n_items * keys * d + n_items * d * d
                     + 2 * n_items * d * cfg.encoder_ff)
        total += n_items * G.N_ITEM_FEATURES * d + cfg.encoder_layers * per_layer
    return total
