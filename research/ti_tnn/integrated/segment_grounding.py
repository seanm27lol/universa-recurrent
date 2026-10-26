"""F's learned segment grounding (protocols/factored_interpreter_v1.md, "F, the neural grounding model").

**Example.** The supplied grammar cuts "name the company operating the airfield of country_29_city_0 ." into
"the airfield of" and "the company operating".
- **Input.** F encodes each segment alone: its words through the training vocabulary (`<unk>` for any other word),
  plus the style (prefix or possessive).
- **Output.** A distribution over seven labels: the five relations, `<unsupported>` and `<ambiguous>`.
- **What F never sees:** the entity and the frame.

**The model:**
- a word embedding (d = 48);
- a one-layer bidirectional GRU (hidden 64 per direction);
- the final states of both directions, concatenated with a style embedding (d = 8);
- one linear layer to the seven labels.

**Training:**
- **The data:** the aligned segment presentations derived from the training pairs (`bench/factored_interpreter.py`).
- **Sampling:** every batch draws 64 presentations uniformly with replacement, so the source multiplicities are
  preserved, and the loss is unweighted cross-entropy.
- **The budget:** 1,500 Adam updates at learning rate 3e-3.
- **What is recorded:** every presentation actually drawn, and the token count, so that training compute is
  reported as realized, not assumed from the update count.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import routing as J

STYLES = ("prefix", "possessive")


def labels_for(relations) -> tuple[str, ...]:
    return tuple(sorted(relations)) + (J.UNSUPPORTED, J.AMBIGUOUS)


class SegmentGrounder(nn.Module):
    def __init__(self, n_words: int, n_labels: int = 7, d: int = 48, hidden: int = 64, style_d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_words, d, padding_idx=0)
        self.encoder = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.style = nn.Embedding(len(STYLES), style_d)
        self.out = nn.Linear(2 * hidden + style_d, n_labels)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(ids), lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.encoder(packed)  # (2, B, hidden): the final state of each direction
        return self.out(torch.cat([h[0], h[1], self.style(styles)], -1))


def encode(segments: list[tuple[str, tuple[str, ...]]], vocab: J.TokenVocab):
    """(ids, lengths, styles) for (style, tokens) segments."""
    width = max(len(t) for _, t in segments)
    ids = torch.zeros(len(segments), width, dtype=torch.long)
    for i, (_, toks) in enumerate(segments):
        ids[i, :len(toks)] = torch.tensor(vocab.encode(list(toks)))
    lengths = torch.tensor([len(t) for _, t in segments])
    styles = torch.tensor([STYLES.index(s) for s, _ in segments])
    return ids, lengths, styles


def train_grounder(presentations: list[tuple[str, tuple[str, ...], str]], vocab: J.TokenVocab, relations,
                   seed: int, updates: int = 1500, batch: int = 64, lr: float = 3e-3) -> dict:
    """Fit F on (style, tokens, label) presentations. Returns the model, its labels and the realized training:
    how often each presentation was drawn, the tokens processed, the final loss and the wall time."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    labels = labels_for(relations)
    index = {lab: i for i, lab in enumerate(labels)}
    net = SegmentGrounder(len(vocab.words), len(labels))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    draws = torch.zeros(len(presentations), dtype=torch.long)
    tokens, loss, t0 = 0, torch.tensor(0.0), time.time()
    for _ in range(updates):
        idx = torch.randint(len(presentations), (batch,), generator=gen)
        draws += torch.bincount(idx, minlength=len(presentations))
        chosen = [presentations[i] for i in idx.tolist()]
        ids, lengths, styles = encode([(s, t) for s, t, _ in chosen], vocab)
        tokens += int(lengths.sum())
        target = torch.tensor([index[lab] for _, _, lab in chosen])
        loss = F.cross_entropy(net(ids, lengths, styles), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return {"net": net, "labels": labels, "draws": draws.tolist(), "tokens": tokens, "loss": loss.item(),
            "seconds": round(time.time() - t0, 2), "updates": updates, "batch": batch,
            "parameters": sum(p.numel() for p in net.parameters())}


@torch.no_grad()
def ground(net: SegmentGrounder, vocab: J.TokenVocab, segments: list[tuple[str, tuple[str, ...]]],
           batch: int = 1024) -> list[list[float]]:
    """The full label distribution of each (style, tokens) segment, as Python floats (float32 softmax)."""
    out = []
    for s in range(0, len(segments), batch):
        chunk = segments[s:s + batch]
        ids, lengths, styles = encode(chunk, vocab)
        out += F.softmax(net(ids, lengths, styles), -1).tolist()
    return out


def forward_macs(n_tokens: int, d: int = 48, hidden: int = 64, style_d: int = 8, n_labels: int = 7) -> int:
    """Multiply-accumulates of one forward pass over a segment of `n_tokens` tokens, from the layer shapes: the
    bidirectional GRU (3 gates, input and recurrent matrices, both directions) and the output layer."""
    return n_tokens * 2 * 3 * (d * hidden + hidden * hidden) + (2 * hidden + style_d) * n_labels
