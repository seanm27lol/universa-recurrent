"""Training-example retrieval: the same k worked examples for every arm, from the training pool only.

**Example.** Development request `bird23tf-04392` (`public_review_platform`) asks "Write down the number of running
business with each review count in Cave Creek city.", and its evidence mentions `review_count = 'High'`. The index
ranks the 5,071 training-pool questions by word overlap with the question plus evidence. The top hit (BM25 22.74) is
`bird23tf-02846` from `works_cycles`: "What is the class of the product with the most reviews?", whose evidence also
says "high". That example comes with its evidence and gold SQL, because training gold SQL is permitted material.
The match is lexical, not semantic, which is a limit of this retrieval. Development and reserved questions are never
in the index.

**Scoring: Okapi BM25** (Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009),
with k1 = 1.2 and b = 0.75. A document is one training-pool question plus its evidence, and the query is the
request's question plus its evidence. With N documents, df(t) of which contain term t, a term's weight is
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)) (the non-negative Lucene form). A document d of length |d|, in
an index whose average length is avgdl, then scores

    sum over distinct query terms t:  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

Terms are lowercase alphanumeric runs (Unicode letters and digits), minus a short English stopword list. Query
terms are summed in sorted order, so scores are bit-identical across runs. Ties are broken by task_id, ascending.

**Exclusions.** The request itself is never returned. Neither is any question from the request's own database
(`exclude_same_database=True`, the default). For development requests this changes nothing, because their databases
are not in the pool. For training requests it keeps the retrieved examples cross-database, as they are at
development time, so a controller trained on them sees the same kind of context it will be evaluated with.

**Identity.** `Index.identity` is a sha256 over the scoring version, the parameters, the stopword list and the
ordered (task_id, content_sha256) of every indexed question. The default index is built once per process.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from . import contracts as C

RETRIEVAL_VERSION = "bm25-v1"
K1, B = 1.2, 0.75
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = frozenset("""
a about above after all also an and any are as at be been being but by can could did do does doing during each few
for from had has have having he her here hers him his how i if in into is it its itself just me more most my no nor
not of off on once only or other our ours out over own same she should so some such than that the their theirs them
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours
""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Document:
    task_id: str
    db_id: str
    question: str
    evidence: str
    sql: str
    content_sha256: str


class Index:
    """A BM25 index over documents (question + evidence). Build once, query many times."""

    def __init__(self, documents, k1: float = K1, b: float = B):
        self.documents = tuple(sorted(documents, key=lambda d: d.task_id))
        self.k1, self.b = k1, b
        self.lengths, self.postings = [], {}
        for position, doc in enumerate(self.documents):
            counts = Counter(tokens(f"{doc.question} {doc.evidence}"))
            self.lengths.append(sum(counts.values()))
            for term, tf in sorted(counts.items()):
                self.postings.setdefault(term, []).append((position, tf))
        n = len(self.documents)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        self.idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        payload = {"version": RETRIEVAL_VERSION, "k1": k1, "b": b, "stopwords": sorted(STOPWORDS),
                   "documents": [(d.task_id, d.content_sha256) for d in self.documents]}
        self.identity = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def scores(self, text: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for term in sorted(set(tokens(text))):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for position, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[position] / self.avgdl)
                out[position] = out.get(position, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def search(self, text: str, k: int, exclude_task_id: str | None = None,
               exclude_db_id: str | None = None) -> list[tuple[Document, float]]:
        """The top k documents by (score descending, task_id ascending); zero-score documents fill any shortfall."""
        scored = self.scores(text)
        candidates = []
        for position, doc in enumerate(self.documents):
            if doc.task_id == exclude_task_id or (exclude_db_id is not None and doc.db_id == exclude_db_id):
                continue
            candidates.append((-scored.get(position, 0.0), doc.task_id, position))
        candidates.sort()
        return [(self.documents[p], -negative) for negative, _tid, p in candidates[:k]]


def documents_from_rows(rows) -> list[Document]:
    return [Document(task_id=r.task_id, db_id=r.db_id, question=r.question, evidence=r.evidence, sql=r.sql,
                     content_sha256=r.content_sha256) for r in rows]


@functools.lru_cache(maxsize=1)
def training_pool_index() -> Index:
    """The default index: every training-pool question of the pinned release (never development or reserved)."""
    from . import data

    pool = set(data.train_pool_databases())
    return Index(documents_from_rows(r for r in data.rows() if r.db_id in pool))


def retrieve(task: C.Task, k: int = C.Limits().retrieval_k, index: Index | None = None,
             exclude_same_database: bool = True) -> list[C.RetrievedExample]:
    """The k training examples every arm receives for this task (see the module docstring)."""
    index = index or training_pool_index()
    hits = index.search(f"{task.question} {task.evidence}", k, exclude_task_id=task.task_id,
                        exclude_db_id=task.db_id if exclude_same_database else None)
    return [C.RetrievedExample(task_id=d.task_id, db_id=d.db_id, question=d.question, evidence=d.evidence,
                               sql=d.sql, score=score) for d, score in hits]
