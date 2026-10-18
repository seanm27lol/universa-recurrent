"""Jev as a planning policy: a controlled-language question in, a typed program or an abstention out
(integration plan section 5.4, PR 6).

Example: "what is the operator of the airport of the largest city of country_3 ?"
- The interpreter proposes the program `largest_city_of, airport_of, operator_of`, in the order it runs
  (the reverse of the order the phrases appear), and binds `country_3` as its input.
- The runtime type-checks the program before anything executes, then runs it with the registered modules.
  Each module sees only its typed value.

The interpreter is a small bidirectional GRU encoder with two heads:
- **Entity head:** chooses which question token is the entity. The entity is copied, never generated, so
  an entity the interpreter never saw can still be bound.
- **Program decoder:** a GRU with attention that emits relations one at a time, then `<end>`. It can
  instead emit `<unsupported>` (a relation the system does not have) or `<ambiguous>` (a phrase with
  more than one reading) as its first output.

The policy (`propose`) answers only if four things hold:
- the interpreter proposed a program;
- the entity exists in the world;
- the runtime accepted the program and produced an answer;
- the calibrated confidence clears the registered threshold (`calibration.py`).

Every other case abstains, with the reason. No type mask is applied while decoding: an ill-typed proposal is
left for the runtime to reject, so questions like "the capital of <a city>" end in abstention rather than
being forced into some well-typed program.

The confidence is the product of the decoder's chosen-step probabilities and the entity head's
probability. It is a score, not a probability, until calibrated on a disjoint split.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import Budget, Program, content_identity
from .runtime import execute

PAD, UNK = "<pad>", "<unk>"
END, UNSUPPORTED, AMBIGUOUS = "<end>", "<unsupported>", "<ambiguous>"
MAX_STEPS = 3
POLICY_VERSION = "jev_policy_v1"


def tokenize(text: str) -> list[str]:
    return text.split()


class TokenVocab:
    """Word indices from training questions; anything else is <unk>."""

    def __init__(self, texts):
        self.words = [PAD, UNK] + sorted({w for t in texts for w in tokenize(t)})
        self.index = {w: i for i, w in enumerate(self.words)}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.index.get(w, 1) for w in tokens]


@dataclass(frozen=True)
class Example:
    """One supervised question: the target outputs (relations in execution order, then <end>; or a single
    <unsupported> / <ambiguous>) and the entity's token position."""

    text: str
    outputs: tuple[str, ...]
    entity_position: int


class Interpreter(nn.Module):
    def __init__(self, n_words: int, outputs: tuple[str, ...], d: int = 48, hidden: int = 64):
        super().__init__()
        self.outputs = outputs
        h2 = 2 * hidden
        self.embed = nn.Embedding(n_words, d, padding_idx=0)
        self.encoder = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.entity = nn.Linear(h2, 1)
        self.start = nn.Linear(h2, h2)
        self.out_embed = nn.Embedding(len(outputs) + 1, d)  # the last index is the start symbol
        self.cell = nn.GRUCell(d + h2, h2)
        self.attend = nn.Linear(h2, h2, bias=False)
        self.out = nn.Linear(2 * h2, len(outputs))

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None, steps: int = MAX_STEPS + 1):
        mask = ids != 0
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(ids), lengths, batch_first=True, enforce_sorted=False)
        enc, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=ids.shape[1])
        entity_logits = self.entity(enc).squeeze(-1).masked_fill(~mask, float("-inf"))
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        state = torch.tanh(self.start(pooled))
        prev = torch.full((ids.shape[0],), len(self.outputs), dtype=torch.long)
        logits = []
        for t in range(steps):
            scores = torch.bmm(enc, self.attend(state).unsqueeze(-1)).squeeze(-1).masked_fill(~mask, float("-inf"))
            context = (F.softmax(scores, -1).unsqueeze(-1) * enc).sum(1)
            state = self.cell(torch.cat([self.out_embed(prev), context], -1), state)
            step = self.out(torch.cat([state, context], -1))
            logits.append(step)
            prev = targets[:, t] if targets is not None else step.argmax(-1)
        return entity_logits, torch.stack(logits, 1)


def interpreter_identity(net: Interpreter, vocab: TokenVocab) -> str:
    params = {k: v.detach().cpu().contiguous().numpy().tobytes().hex() for k, v in sorted(net.state_dict().items())}
    return content_identity({"kind": "jev_interpreter_v1", "words": vocab.words, "outputs": list(net.outputs),
                             "params": params})


def _batch(examples, vocab, outputs, word_dropout=0.0, generator=None):
    index = {o: i for i, o in enumerate(outputs)}
    toks = [vocab.encode(tokenize(e.text)) for e in examples]
    width = max(map(len, toks))
    ids = torch.zeros(len(toks), width, dtype=torch.long)
    for i, t in enumerate(toks):
        ids[i, :len(t)] = torch.tensor(t)
    if word_dropout:
        drop = (torch.rand(ids.shape, generator=generator) < word_dropout) & (ids != 0)
        ids = ids.masked_fill(drop, 1)
    targets = torch.full((len(examples), MAX_STEPS + 1), index[END], dtype=torch.long)
    weight = torch.zeros(len(examples), MAX_STEPS + 1)
    for i, e in enumerate(examples):
        for t, o in enumerate(e.outputs):
            targets[i, t] = index[o]
            weight[i, t] = 1.0
    entity = torch.tensor([e.entity_position for e in examples])
    return ids, targets, weight, entity


def train_interpreter(examples: list[Example], relations: tuple[str, ...], seed: int, d: int = 48, hidden: int = 64,
                      epochs: int = 30, lr: float = 3e-3, batch: int = 64, word_dropout: float = 0.1):
    """Train on (question, outputs, entity position) examples. Returns (net, vocab, final loss)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    outputs = tuple(sorted(relations)) + (END, UNSUPPORTED, AMBIGUOUS)
    vocab = TokenVocab([e.text for e in examples])
    net = Interpreter(len(vocab.words), outputs, d, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=gen).tolist()
        for s in range(0, len(order), batch):
            chunk = [examples[i] for i in order[s:s + batch]]
            ids, targets, weight, entity = _batch(chunk, vocab, outputs, word_dropout, gen)
            entity_logits, logits = net(ids, targets)
            step_loss = F.cross_entropy(logits.reshape(-1, len(outputs)), targets.reshape(-1), reduction="none")
            loss = (step_loss * weight.reshape(-1)).sum() / weight.sum() + F.cross_entropy(entity_logits, entity)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net, vocab, loss.item()


@dataclass(frozen=True)
class Parse:
    """What the interpreter proposes for one question."""

    outcome: str  # "program", "unsupported", "ambiguous" or "malformed" (no stop symbol within MAX_STEPS + 1)
    steps: tuple[str, ...]
    entity: str
    score: float  # product of chosen-step probabilities and the entity probability; uncalibrated
    step_probabilities: tuple[float, ...]
    entity_probability: float


@torch.no_grad()
def interpret(net: Interpreter, vocab: TokenVocab, texts: list[str], batch: int = 256) -> list[Parse]:
    out = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        tokens = [tokenize(t) for t in chunk]
        ids, *_ = _batch([Example(t, (), 0) for t in chunk], vocab, net.outputs)
        entity_logits, logits = net(ids)
        entity_p = F.softmax(entity_logits, -1)
        step_p = F.softmax(logits, -1)
        for i, toks in enumerate(tokens):
            pos = int(entity_p[i].argmax())
            steps, probs, outcome = [], [], "malformed"
            for t in range(MAX_STEPS + 1):
                k = int(step_p[i, t].argmax())
                probs.append(float(step_p[i, t, k]))
                o = net.outputs[k]
                if o in (UNSUPPORTED, AMBIGUOUS):
                    outcome = "unsupported" if o == UNSUPPORTED else "ambiguous"
                    break
                if o == END:
                    outcome = "program" if steps else "malformed"
                    break
                steps.append(o)
            score = float(torch.tensor(probs).prod()) * float(entity_p[i, pos])
            out.append(Parse(outcome, tuple(steps), toks[pos], score, tuple(probs), float(entity_p[i, pos])))
    return out


def propose(parse: Parse, types, registry) -> dict:
    """The policy before thresholding: an answer with its execution, or an abstention with the reason.
    `types` are the type domains (to bind the entity); the policy never sees the facts."""
    if parse.outcome != "program":
        return {"decision": "abstain", "reason": parse.outcome, "answer": None, "execution": None}
    bound = [t for t in sorted(types) if types[t].contains(parse.entity)]
    if not bound:
        return {"decision": "abstain", "reason": "unknown entity", "answer": None, "execution": None}
    program = Program(parse.steps, types[bound[0]].value(parse.entity))
    ex = execute(program, registry, types, Budget(len(parse.steps)))
    if ex.status != "answered":
        return {"decision": "abstain", "reason": ex.decision.reason, "answer": None, "execution": ex}
    return {"decision": "answer", "reason": "", "answer": ex.answer.value, "execution": ex}
