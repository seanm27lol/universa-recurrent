"""A+K: A's fitted interpreter, decoded under the supplied grammar's constraints inside its own recurrence
(protocols/factored_interpreter_v1.md, "A+K, constrained decoding inside the recurrence").

**Example.** The grammar finds two segments in "name the company operating the airfield of country_29_city_0 .", so
the program needs exactly two relations. A's entity pointer chooses `country_29_city_0`, a City.
- **Step 0:** only relations whose source is City are allowed, with `<unsupported>` and `<ambiguous>`. The
  highest-probability allowed output is chosen, say `airport_of`, and fed into the next GRU step.
- **Step 1:** only relations whose source is Airport are allowed, plus the two refusals.
- **Step 2:** `<end>` or a refusal.

**The score** is the product of the chosen outputs' **original** probabilities, from the softmax over every output,
times the entity pointer's probability. A forced `<end>` keeps whatever probability A gave it. Nothing is
renormalized over the allowed set.

**A grammar-admitted constrained policy sharing A's model and entity pointer.** When the grammar refuses a
question, A+K abstains with the grammar's reason. Those refusals are reported separately.

`decode(..., constrained=False)` applies no constraint and must reproduce A's own greedy decoding
(`interpreters.interpret`) exactly. A test checks this.

**Selection by logits.** The choice is the highest **logit** among the allowed outputs (`select`); ties go to the
earlier output index. The development pilot recorded at `0e1a3c9` chose the highest allowed **probability**
instead. Where two allowed outputs both underflow to probability 0 in float32, that rule ties and takes the earlier
index, whatever their logits. Each step now records its logits as well as its probabilities, and the scores still
use the original unmasked probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import interpreters as I
from . import routing as J


def select(logits: list[float], allowed: list[int]) -> int:
    """The allowed output with the highest logit; ties to the earlier index."""
    return max(allowed, key=lambda k: (logits[k], -k))


@torch.no_grad()
def decode(net: I.Interpreter, vocab, texts: list[str], slots: list[int | None], world, constrained: bool = True,
           batch: int = 256, stats: dict | None = None) -> list[dict | None]:
    """One record per text; None where `slots` is None (the grammar refused: no decoding).

    Each record holds the entity pointer's choice and probability, every step's full output distribution and logits,
    its allowed set and the chosen output, the outcome ("program", "unsupported", "ambiguous", "unknown entity" or
    "malformed"), the program, and the score. `stats["evaluated_row_steps"]`, when given, accumulates the decoder
    row-steps actually computed: every row of a batch, at every step the batch runs."""
    outputs = net.outputs
    index = {o: i for i, o in enumerate(outputs)}
    relation_outputs = [i for i, o in enumerate(outputs) if o in world.relations]
    refusals = [index[J.UNSUPPORTED], index[J.AMBIGUOUS]]
    type_of = {v: t for t in sorted(world.types) for v in world.types[t].values}
    records: list[dict | None] = [None] * len(texts)
    todo = [i for i, n in enumerate(slots) if n is not None]
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        toks = [I.normalize(texts[i]) for i in chunk]
        b = I.make_batch(net.mode, toks, vocab)
        mask = b.mask
        packed = nn.utils.rnn.pack_padded_sequence(net.words(b), mask.sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        enc, _ = net.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=mask.shape[1])
        entity_p = F.softmax(net.entity(enc).squeeze(-1).masked_fill(~mask, float("-inf")), -1)
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        state = torch.tanh(net.start(pooled))
        prev = torch.full((len(chunk),), len(outputs), dtype=torch.long)
        rows = []
        for r, i in enumerate(chunk):
            pos = int(entity_p[r].argmax())
            entity = toks[r][pos]
            known = entity in type_of
            rows.append({"entity": entity, "entity_probability": float(entity_p[r, pos]), "slots": slots[i],
                         "steps": [], "program": [], "outcome": None if (known or not constrained) else "unknown entity",
                         "type": type_of.get(entity), "done": constrained and not known})
        for t in range(J.MAX_STEPS + 1):
            if all(row["done"] for row in rows):
                break
            if stats is not None:
                stats["evaluated_row_steps"] = stats.get("evaluated_row_steps", 0) + len(rows)
            scores = torch.bmm(enc, net.attend(state).unsqueeze(-1)).squeeze(-1).masked_fill(~mask, float("-inf"))
            context = (F.softmax(scores, -1).unsqueeze(-1) * enc).sum(1)
            state = net.cell(torch.cat([net.out_embed(prev), context], -1), state)
            logits_t = net.out(torch.cat([state, context], -1))
            p = F.softmax(logits_t, -1)
            nxt = prev.clone()
            for r, row in enumerate(rows):
                if row["done"]:
                    continue
                dist, logits = p[r].tolist(), logits_t[r].tolist()
                if not constrained:
                    allowed = list(range(len(outputs)))
                elif t < row["slots"]:
                    allowed = sorted([k for k in relation_outputs if world.relations[outputs[k]].source == row["type"]]
                                     + refusals)
                else:
                    allowed = sorted([index[J.END]] + refusals)
                chosen = select(logits, allowed)
                row["steps"].append({"distribution": dist, "logits": logits, "allowed": allowed, "chosen": chosen})
                o = outputs[chosen]
                nxt[r] = chosen
                if o in (J.UNSUPPORTED, J.AMBIGUOUS):
                    row["outcome"], row["done"] = ("unsupported" if o == J.UNSUPPORTED else "ambiguous"), True
                elif o == J.END:
                    row["outcome"], row["done"] = ("program" if row["program"] else "malformed"), True
                else:
                    row["program"].append(o)
                    row["type"] = world.relations[o].target if o in world.relations else row["type"]
            prev = nxt
        for r, i in enumerate(chunk):
            row = rows[r]
            score = 1.0
            for st in row["steps"]:
                score *= st["distribution"][st["chosen"]]
            score *= row["entity_probability"]
            records[i] = {"entity": row["entity"], "entity_probability": row["entity_probability"],
                          "steps": row["steps"], "outcome": row["outcome"] or "malformed", "program": row["program"],
                          "score": score}
    return records
