"""Word- and character-level interpreters for the interpreter robustness study
(protocols/interpreter_robustness_v1.md). They extend #48's interpreter (`routing.py`) with a choice of how
words are encoded.

Both use #48's encoder and decoder: a bidirectional GRU over word vectors, an entity head that copies one
question token, and an attention decoder emitting relations in execution order, then `<end>`, or
`<unsupported>` / `<ambiguous>`. They differ only in how a word becomes a vector:
- **`word`**: an embedding per training-vocabulary word. Every other word is the single token `<unk>`, so
  different unfamiliar words become identical inputs.
- **`char`**: a bidirectional GRU over the word's characters. Distinct spellings are distinct input
  sequences. (A finite-dimensional encoder is not guaranteed to keep every distinction in its vectors; the
  guarantee is about the input sequence.)

**Input contract for `char`.** It removes the ways a shared unknown symbol could reappear:
- **Alphabet.** Questions are lowercased and split on whitespace, and every character must be in
  `ALPHABET`. A question with any other character, a word longer than `MAX_WORD` characters, or more than
  `MAX_WORDS` words is not encoded at all. The policy refuses it ("unsupported characters"); no character
  is ever mapped to a shared unknown symbol.
- **No truncation.** Words are never truncated.

**Training budget.** `train` runs a fixed number of optimizer updates on fixed-size batches, whatever the
training set's size. So arms with and without augmentation get the same budget: smaller sets simply
repeat examples more often.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import routing as J

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_'?.,"
CHAR_INDEX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 is padding
MAX_WORD, MAX_WORDS = 48, 40
MODES = ("word", "char")


class UnsupportedInput(ValueError):
    """The question contains characters outside the declared alphabet, or exceeds the length limits."""


def normalize(text: str) -> list[str]:
    tokens = text.lower().split()
    if not tokens or len(tokens) > MAX_WORDS:
        raise UnsupportedInput("empty or too many words")
    for t in tokens:
        if len(t) > MAX_WORD or any(c not in CHAR_INDEX for c in t):
            raise UnsupportedInput(f"unsupported word {t!r}")
    return tokens


class CharWords(nn.Module):
    """A word vector from its characters: a bidirectional GRU's final states, projected."""

    def __init__(self, d: int, char_dim: int = 24, hidden: int = 48):
        super().__init__()
        self.chars = nn.Embedding(len(ALPHABET) + 1, char_dim, padding_idx=0)
        self.gru = nn.GRU(char_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, d)

    def forward(self, chars: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:  # (N, L), (N,) -> (N, d)
        packed = nn.utils.rnn.pack_padded_sequence(self.chars(chars), lengths.clamp(min=1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)
        return self.proj(torch.cat([h[0], h[1]], -1))


@dataclass
class Batch:
    mask: torch.Tensor  # (B, T) word present
    ids: torch.Tensor | None = None  # word mode: (B, T)
    chars: torch.Tensor | None = None  # char mode: (B, T, L)


class Interpreter(nn.Module):
    """#48's interpreter with a pluggable word encoder (`mode` "word" or "char")."""

    def __init__(self, mode: str, n_words: int, outputs: tuple[str, ...], d: int = 48, hidden: int = 64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.outputs = mode, outputs
        h2 = 2 * hidden
        self.embed = nn.Embedding(n_words, d, padding_idx=0) if mode == "word" else None
        self.char_words = CharWords(d) if mode == "char" else None
        self.encoder = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.entity = nn.Linear(h2, 1)
        self.start = nn.Linear(h2, h2)
        self.out_embed = nn.Embedding(len(outputs) + 1, d)
        self.cell = nn.GRUCell(d + h2, h2)
        self.attend = nn.Linear(h2, h2, bias=False)
        self.out = nn.Linear(2 * h2, len(outputs))

    def words(self, batch: Batch) -> torch.Tensor:
        if self.mode == "word":
            return self.embed(batch.ids)
        B, T, L = batch.chars.shape
        flat = batch.chars.reshape(B * T, L)
        lengths = (flat != 0).sum(1)
        vecs = torch.zeros(B * T, self.char_words.proj.out_features)
        present = lengths > 0
        vecs[present] = self.char_words(flat[present], lengths[present])
        return vecs.reshape(B, T, -1)

    def forward(self, batch: Batch, targets: torch.Tensor | None = None, steps: int = J.MAX_STEPS + 1):
        mask = batch.mask
        packed = nn.utils.rnn.pack_padded_sequence(self.words(batch), mask.sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        enc, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=mask.shape[1])
        entity_logits = self.entity(enc).squeeze(-1).masked_fill(~mask, float("-inf"))
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
        return entity_logits, torch.stack(logits, 1)


def make_batch(mode: str, token_lists: list[list[str]], vocab: J.TokenVocab | None,
               drop_positions: list[set[int]] | None = None) -> Batch:
    """Encode token lists. `drop_positions` (word mode, training only) replaces those positions with <unk>."""
    width = max(map(len, token_lists))
    mask = torch.zeros(len(token_lists), width, dtype=torch.bool)
    for i, toks in enumerate(token_lists):
        mask[i, :len(toks)] = True
    if mode == "word":
        ids = torch.zeros(len(token_lists), width, dtype=torch.long)
        for i, toks in enumerate(token_lists):
            enc = vocab.encode(toks)
            for j in (drop_positions[i] if drop_positions else ()):
                enc[j] = 1
            ids[i, :len(toks)] = torch.tensor(enc)
        return Batch(mask, ids=ids)
    length = max(len(t) for toks in token_lists for t in toks)
    chars = torch.zeros(len(token_lists), width, length, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks):
            chars[i, j, :len(t)] = torch.tensor([CHAR_INDEX[c] for c in t])
    return Batch(mask, chars=chars)


def train(mode: str, examples: list[J.Example], vocab_texts: list[str], relations: tuple[str, ...], seed: int,
          updates: int = 1500, batch: int = 64, lr: float = 3e-3, d: int = 48, hidden: int = 64,
          entity_dropout: float = 0.1):
    """Train for exactly `updates` optimizer steps. The word vocabulary comes from `vocab_texts` (the ordinary
    training questions), so augmentation pseudo-words are <unk> to a word model, as unfamiliar test words are.
    In word mode, the entity token is replaced by <unk> with probability `entity_dropout`, so the entity head
    learns to copy entities it never saw. Relation words are never dropped: that would teach the decoder to
    guess unfamiliar relation words."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    outputs = tuple(sorted(relations)) + (J.END, J.UNSUPPORTED, J.AMBIGUOUS)
    index = {o: i for i, o in enumerate(outputs)}
    vocab = J.TokenVocab(vocab_texts) if mode == "word" else None
    net = Interpreter(mode, len(vocab.words) if vocab else 1, outputs, d, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tokens = [normalize(e.text) for e in examples]
    order, k, loss = [], 0, torch.tensor(0.0)
    for _ in range(updates):
        if k + batch > len(order):
            order, k = torch.randperm(len(examples), generator=gen).tolist(), 0
        idx = order[k:k + batch]
        k += batch
        drops = None
        if mode == "word" and entity_dropout:
            drops = [{examples[i].entity_position} if float(torch.rand(1, generator=gen)) < entity_dropout else set()
                     for i in idx]
        b = make_batch(mode, [tokens[i] for i in idx], vocab, drops)
        targets = torch.full((len(idx), J.MAX_STEPS + 1), index[J.END], dtype=torch.long)
        weight = torch.zeros(len(idx), J.MAX_STEPS + 1)
        for r, i in enumerate(idx):
            for t, o in enumerate(examples[i].outputs):
                targets[r, t] = index[o]
                weight[r, t] = 1.0
        entity = torch.tensor([examples[i].entity_position for i in idx])
        entity_logits, logits = net(b, targets)
        step_loss = F.cross_entropy(logits.reshape(-1, len(outputs)), targets.reshape(-1), reduction="none")
        loss = (step_loss * weight.reshape(-1)).sum() / weight.sum() + F.cross_entropy(entity_logits, entity)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, vocab, loss.item()


@torch.no_grad()
def interpret(net: Interpreter, vocab, texts: list[str], batch: int = 256) -> list[J.Parse | None]:
    """Parses in #48's format; None for a question the declared input contract cannot encode."""
    out: list[J.Parse | None] = [None] * len(texts)
    encodable = []
    for i, t in enumerate(texts):
        try:
            encodable.append((i, normalize(t)))
        except UnsupportedInput:
            pass
    for s in range(0, len(encodable), batch):
        chunk = encodable[s:s + batch]
        b = make_batch(net.mode, [toks for _, toks in chunk], vocab)
        entity_logits, logits = net(b)
        entity_p, step_p = F.softmax(entity_logits, -1), F.softmax(logits, -1)
        for r, (i, toks) in enumerate(chunk):
            pos = int(entity_p[r].argmax())
            steps, probs, outcome = [], [], "malformed"
            for t in range(J.MAX_STEPS + 1):
                k = int(step_p[r, t].argmax())
                probs.append(float(step_p[r, t, k]))
                o = net.outputs[k]
                if o in (J.UNSUPPORTED, J.AMBIGUOUS):
                    outcome = "unsupported" if o == J.UNSUPPORTED else "ambiguous"
                    break
                if o == J.END:
                    outcome = "program" if steps else "malformed"
                    break
                steps.append(o)
            score = float(torch.tensor(probs).prod()) * float(entity_p[r, pos])
            out[i] = J.Parse(outcome, tuple(steps), toks[pos], score, tuple(probs), float(entity_p[r, pos]))
    return out
