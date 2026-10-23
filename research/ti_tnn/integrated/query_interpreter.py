"""A learned interpreter for bounded until-questions (protocols/integration_test_v1.md, revision 2).

**Example.** "which is the first airport within 6 connections of country_12_city_0_airport that operator_3
operates ?" is parsed as follows:
- **start copy:** the `country_12_city_0_airport` token;
- **target copy:** the `operator_3` token;
- **decoder outputs:** `connects_to`, `operator_of`, `b6`, `<end>`.

The policy then binds the copied tokens to typed values, checks the query and its budget contract, and only
then runs anything.

**Architecture.** It reuses #48's interpreter (`interpreters.py`, word mode), unchanged in kind:
- a word embedding over the training vocabulary, `<unk>` otherwise;
- a bidirectional GRU encoder;
- an attention decoder.

It adds a second entity-copy head, so start and target are copied separately. The decoder's outputs are:
- the world's relations;
- the budget tokens `b1` to `b16`, `b_default` (no budget stated) and `<over_budget>` (a stated budget
  outside the contract);
- `<end>` and `<unsupported>`.

**Score:** the product of the chosen decoder-step probabilities and both copy probabilities. It is uncalibrated;
the serving context's certificate decides what is released.

**Training:** a fixed number of Adam updates on fixed-size batches. Entity dropout replaces the start and the
target token by `<unk>`, each with probability 0.1, so the copy heads learn to copy entities they never
saw. `interpreters.py` is frozen for the robustness study and is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import interpreters as I
from . import routing as J
from .contracts import content_identity

END, UNSUPPORTED = J.END, J.UNSUPPORTED
BUDGETS = tuple(f"b{n}" for n in range(1, 17))
DEFAULT, OVER = "b_default", "<over_budget>"
MAX_STEPS = 4  # two relations, a budget token, <end>


@dataclass(frozen=True)
class QueryExample:
    text: str
    outputs: tuple[str, ...]
    start_position: int | None
    target_position: int | None


@dataclass(frozen=True)
class QueryParse:
    outcome: str  # "query", "unsupported", "over_budget" or "malformed"
    relations: tuple[str, ...]
    budget: int | None  # 1-16 for a query; None otherwise
    default_budget: bool  # the question stated no budget (b_default -> 16)
    start: str
    target: str
    score: float
    step_probabilities: tuple[float, ...]
    start_probability: float
    target_probability: float


class QueryInterpreter(nn.Module):
    def __init__(self, n_words: int, outputs: tuple[str, ...], d: int = 48, hidden: int = 64):
        super().__init__()
        self.outputs = outputs
        h2 = 2 * hidden
        self.embed = nn.Embedding(n_words, d, padding_idx=0)
        self.encoder = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.start_head = nn.Linear(h2, 1)
        self.target_head = nn.Linear(h2, 1)
        self.start = nn.Linear(h2, h2)
        self.out_embed = nn.Embedding(len(outputs) + 1, d)
        self.cell = nn.GRUCell(d + h2, h2)
        self.attend = nn.Linear(h2, h2, bias=False)
        self.out = nn.Linear(2 * h2, len(outputs))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor | None = None,
                steps: int = MAX_STEPS):
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(ids), mask.sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        enc, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=mask.shape[1])
        start_logits = self.start_head(enc).squeeze(-1).masked_fill(~mask, float("-inf"))
        target_logits = self.target_head(enc).squeeze(-1).masked_fill(~mask, float("-inf"))
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        state = torch.tanh(self.start(pooled))
        prev = torch.full((mask.shape[0],), len(self.outputs), dtype=torch.long)
        logits = []
        for t in range(steps):
            scores = torch.bmm(enc, self.attend(state).unsqueeze(-1)).squeeze(-1).masked_fill(~mask, float("-inf"))
            context = (F.softmax(scores, -1).unsqueeze(-1) * enc).sum(1)
            state = self.cell(torch.cat([self.out_embed(prev), context], -1), state)
            step = self.out(torch.cat([state, context], -1))
            logits.append(step)
            prev = targets[:, t] if targets is not None else step.argmax(-1)
        return start_logits, target_logits, torch.stack(logits, 1)


def output_vocabulary(relations) -> tuple[str, ...]:
    return tuple(sorted(relations)) + BUDGETS + (DEFAULT, OVER, END, UNSUPPORTED)


def _batch(token_lists, vocab, drops=None):
    width = max(map(len, token_lists))
    ids = torch.zeros(len(token_lists), width, dtype=torch.long)
    mask = torch.zeros(len(token_lists), width, dtype=torch.bool)
    for i, toks in enumerate(token_lists):
        enc = vocab.encode(toks)
        for j in (drops[i] if drops else ()):
            enc[j] = 1  # <unk>
        ids[i, :len(toks)] = torch.tensor(enc)
        mask[i, :len(toks)] = True
    return ids, mask


def train(examples: list[QueryExample], vocab_texts: list[str], relations, seed: int, updates: int = 1500,
          batch: int = 64, lr: float = 3e-3, d: int = 48, hidden: int = 64, entity_dropout: float = 0.1):
    """Exactly `updates` optimizer steps. Returns (net, vocab, final loss)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    outputs = output_vocabulary(relations)
    index = {o: i for i, o in enumerate(outputs)}
    vocab = J.TokenVocab(vocab_texts)
    net = QueryInterpreter(len(vocab.words), outputs, d, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tokens = [I.normalize(e.text) for e in examples]
    order, k, loss = [], 0, torch.tensor(0.0)
    for _ in range(updates):
        if k + batch > len(order):
            order, k = torch.randperm(len(examples), generator=gen).tolist(), 0
        idx = order[k:k + batch]
        k += batch
        drops = []
        for i in idx:
            e = examples[i]
            drops.append({p for p in (e.start_position, e.target_position)
                          if p is not None and float(torch.rand(1, generator=gen)) < entity_dropout})
        ids, mask = _batch([tokens[i] for i in idx], vocab, drops)
        targets = torch.full((len(idx), MAX_STEPS), index[END], dtype=torch.long)
        weight = torch.zeros(len(idx), MAX_STEPS)
        for r, i in enumerate(idx):
            for t, o in enumerate(examples[i].outputs):
                targets[r, t] = index[o]
                weight[r, t] = 1.0
        start_logits, target_logits, logits = net(ids, mask, targets)
        step_loss = F.cross_entropy(logits.reshape(-1, len(outputs)), targets.reshape(-1), reduction="none")
        loss = (step_loss * weight.reshape(-1)).sum() / weight.sum()
        for head, field in ((start_logits, "start_position"), (target_logits, "target_position")):
            rows = [r for r, i in enumerate(idx) if getattr(examples[i], field) is not None]
            if rows:
                gold = torch.tensor([getattr(examples[idx[r]], field) for r in rows])
                loss = loss + F.cross_entropy(head[rows], gold)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, vocab, loss.item()


def identity(net: QueryInterpreter, vocab) -> str:
    """A content hash over the parameters, the vocabulary and the output vocabulary."""
    params = {k: v.detach().cpu().contiguous().numpy().tobytes().hex() for k, v in sorted(net.state_dict().items())}
    return content_identity({"kind": "ti_tnn_query_interpreter_v1", "params": params, "vocab": list(vocab.words),
                             "outputs": list(net.outputs)})


def _read(outputs: tuple[str, ...]) -> tuple[str, tuple[str, ...], int | None, bool]:
    """(outcome, relations, budget, default) from a decoded output sequence (stop symbol included)."""
    if outputs and outputs[0] == UNSUPPORTED:
        return "unsupported", (), None, False
    rels = []
    for i, o in enumerate(outputs):
        if o in BUDGETS or o in (DEFAULT, OVER):
            ok = rels and i + 1 < len(outputs) and outputs[i + 1] == END
            if not ok:
                return "malformed", tuple(rels), None, False
            if o == OVER:
                return "over_budget", tuple(rels), None, False
            return ("query", tuple(rels), 16 if o == DEFAULT else int(o[1:]), o == DEFAULT)
        if o in (END, UNSUPPORTED):
            return "malformed", tuple(rels), None, False
        rels.append(o)
    return "malformed", tuple(rels), None, False


@torch.no_grad()
def interpret(net: QueryInterpreter, vocab, texts: list[str], batch: int = 256) -> list[QueryParse | None]:
    """Greedy parses; None for a question the input contract cannot encode."""
    out: list[QueryParse | None] = [None] * len(texts)
    encodable = []
    for i, t in enumerate(texts):
        try:
            encodable.append((i, I.normalize(t)))
        except I.UnsupportedInput:
            pass
    for s in range(0, len(encodable), batch):
        chunk = encodable[s:s + batch]
        ids, mask = _batch([toks for _, toks in chunk], vocab)
        start_logits, target_logits, logits = net(ids, mask)
        sp, tp, step_p = F.softmax(start_logits, -1), F.softmax(target_logits, -1), F.softmax(logits, -1)
        for r, (i, toks) in enumerate(chunk):
            decoded, probs = [], []
            for t in range(MAX_STEPS):
                k = int(step_p[r, t].argmax())
                probs.append(float(step_p[r, t, k]))
                decoded.append(net.outputs[k])
                if decoded[-1] in (END, UNSUPPORTED):
                    break
            outcome, rels, budget, default = _read(tuple(decoded))
            si, ti = int(sp[r].argmax()), int(tp[r].argmax())
            score = float(torch.tensor(probs).prod()) * float(sp[r, si]) * float(tp[r, ti])
            out[i] = QueryParse(outcome, rels, budget, default, toks[si], toks[ti], score, tuple(probs),
                                float(sp[r, si]), float(tp[r, ti]))
    return out
