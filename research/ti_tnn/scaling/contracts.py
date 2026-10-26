"""Shared contracts for the TI-TNN scaling development study (Qwen3.8 comparison; not frozen).

**Example.** The development request `bird23tf-01234` asks a natural-language question about one training-release
database, say `movie_platform`. Every arm receives the same `Task`:
- the question and BIRD's evidence;
- the database's schema descriptions;
- permitted training-example retrieval;
- read-only tool access to the pinned snapshot.

No arm ever receives the `Gold` record. Only the scorer and the trainer read gold SQL. Each arm returns SQL, a plan
or an abstention, and the scorer runs the submitted SQL in full against the pinned snapshot.

This module holds only the shapes and pinned identities the components share. Behaviour lives elsewhere:

| Module | Owns |
|---|---|
| `data.py`, `schema.py`, `tools.py`, `scorer.py`, `retrieval.py`, `baseline_l.py` | the benchmark: sources, split, inventory, schema, tools, scoring, the no-fit control L |
| `ir.py`, `sql_to_ir.py`, `actions.py`, `encode.py`, `controller.py`, `train.py`, `infer_controller.py` | the relational IR and the C4/T4 controllers |
| `identity.py`, `ledger.py`, `records.py`, `runtime.py`, `llm_agent.py`, `cli.py` | identities, the GPU-hours ledger, request records, the Qwen runtime and harness, the CLI |
| `verify.py` | the independent verifier (written apart from the producers) |
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCALING = Path(__file__).resolve().parent
REPO = SCALING.parents[2]
# GPU runs execute from a separate clean worktree at the commit being run (so files edited during development cannot
# change a running producer); TI_TNN_SCALING_RUNS points every worktree at the one shared output directory.
RUNS = Path(os.environ.get("TI_TNN_SCALING_RUNS", REPO / "runs" / "ti_tnn_scaling"))
if not RUNS.is_absolute():
    raise RuntimeError(f"TI_TNN_SCALING_RUNS must be an absolute path, not {str(RUNS)!r}")
DATA_RAW = RUNS / "data" / "raw"
DATA = RUNS / "data"
PINS = SCALING / "pins"  # compact committed identities: split, inventory ids, snapshot hashes
DATABASES = DATA / "databases"  # the extracted training release: DATABASES/<db_id>/<db_id>.sqlite (+ database_description/)
RESERVED_DATABASES = DATA / "reserved_databases"  # the public benchmark's databases: hashed only, never queried by an arm
CACHE = RUNS / "cache"  # immutable encodings, keyed by model and context identity

STUDY = "ti_tnn_qwen38_scaling_dev"
STUDY_VERSION = "dev-1"  # a development protocol; nothing here is frozen

# --- pinned sources (verified before use; see data.py and runtime.py) ------------------------------------------------
MODELS = {
    "Q27": {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
    "B4": {"repo": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"},
}
BACKBONE = MODELS["B4"]  # the frozen backbone of C4 and T4 is the same released 4B model

TRAIN_RELEASE = {
    "repo": "birdsql/bird23-train-filtered", "revision": "4068469807b255fcfc0816bdd520946fe460d256",
    "files": {
        "data/train-00000-of-00001.jsonl": "2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8",
        "train_column_meaning.json": "7928ddb444fd6e1b96b13ef6a1a87f4b9cf138e8354873c0ed7ca2878d7a88db",
    },
    "rows": 6601, "databases": 69,
    # the databases themselves come from the official BIRD training archive
    "database_archive": {"url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip", "bytes": 8919543554,
                         "etag": "1FBA2891BD215A42CE1004736AB7F47F-400", "last_modified": "Tue, 11 Jul 2023 06:13:29 GMT"},
}
# The reserved public benchmark: pinned and hashed only. No arm is ever run on it in this work order, and its gold
# SQL is never read to design the representation or the training targets.
RESERVED_PUBLIC = {
    "repo": "birdsql/bird_mini_dev", "revision": "f65faf4ae3b638c1fa6df1d3370c8d92c8366301",
    "file": "data/mini_dev_sqlite-00000-of-00001.json",
    "file_sha256": "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2",
    "questions": 500, "dialect": "SQLite", "kind": "the original 500 SELECT-only Mini-Dev questions (no CRUD extension)",
    "database_archive": {"url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip", "bytes": 800943648,
                         "etag": "7BEB6DAB11E65F0FDE563E644A1EA319", "last_modified": "Thu, 20 Jun 2024 05:01:24 GMT"},
}
OFFICIAL_SCORER = {
    "repo": "https://github.com/bird-bench/mini_dev", "commit": "abd11b6db92a1c9f809b32f7564c7c71b34d67f0",
    "files": {"evaluation/evaluation_ex.py": "da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89",
              "evaluation/evaluation_utils.py": "f6943d249caac5aeaef9bce21d43dbf29dcef85a0c965a76df032a9542f308bf"},
    "rule": "EX = 1 when set(fetchall(predicted)) == set(fetchall(gold)) on the same SQLite database; errors and "
            "timeouts score 0",
    "meta_time_out_seconds": 30.0,
}

# --- the split (fixed before any model inference) --------------------------------------------------------------------
SPLIT = {
    "rule": "order the training release's databases by sha256(salt + ':' + db_id); the first round(0.2 * n) are "
            "development, the rest are the training pool. Split by database, never by question.",
    "salt": "ti-tnn-scaling-dev-split-v1",
    "dev_fraction": 0.2,
}
INVENTORY = {
    "rule": "within each development database, order questions by sha256(salt + ':' + task_id); then take one "
            "question per database per round, databases in split order, skipping exhausted databases. The first "
            "512 are the development inventory; its first 64 and first 16 are the nested subsets.",
    "salt": "ti-tnn-scaling-dev-inventory-v1",
    "size": 512, "subsets": (64, 16),
    "excluded": "questions whose gold SQL does not execute, or exceeds the scorer's timeout, on the pinned snapshot",
}
SMOKE_TRAIN = {
    "rule": "order training-pool questions by sha256(salt + ':' + task_id), keep those whose gold SQL compiles to "
            "the action grammar, and take the first 128",
    "salt": "ti-tnn-scaling-smoke-train-v1",
    "size": 128, "seed": 100, "max_updates": 100,
}


# --- what inference may see ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """One request, as every arm receives it. It holds no gold SQL, answers or evaluator-only metadata."""
    task_id: str
    db_id: str
    question: str
    evidence: str
    split: str  # "train" | "dev"


@dataclass(frozen=True)
class Gold:
    """Scorer and trainer only. Never passed to an inference path."""
    task_id: str
    sql: str


@dataclass(frozen=True)
class Snapshot:
    """One pinned, read-only SQLite database file."""
    db_id: str
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SchemaColumn:
    table: str
    name: str
    type: str                    # the declared SQLite type, as written (SQLite typing is dynamic)
    primary_key: bool
    description: str = ""        # BIRD database_description column_description
    value_description: str = ""  # BIRD database_description value_description
    meaning: str = ""            # train_column_meaning.json summary, when present


@dataclass(frozen=True)
class SchemaTable:
    name: str
    columns: tuple               # tuple[SchemaColumn, ...] in declaration order
    foreign_keys: tuple = ()     # tuple[(column, ref_table, ref_column), ...]


@dataclass(frozen=True)
class Schema:
    db_id: str
    snapshot_sha256: str
    tables: tuple                # tuple[SchemaTable, ...] in sqlite_master order


@dataclass(frozen=True)
class SchemaItem:
    """One pointer target in a serialized schema: a table, or one column of a table, with its character span."""
    kind: str                    # "table" | "column"
    table: str
    column: str                  # "" for a table
    start: int
    end: int


@dataclass(frozen=True)
class SerializedSchema:
    """The schema text every arm receives, and where each table and column appears in it."""
    db_id: str
    snapshot_sha256: str
    text: str
    items: tuple                 # tuple[SchemaItem, ...]


@dataclass(frozen=True)
class RetrievedExample:
    """A permitted training example: question, evidence and gold SQL from a training-pool database only."""
    task_id: str
    db_id: str
    question: str
    evidence: str
    sql: str
    score: float


# --- development limits (written into every run configuration before inference) ----------------------------------
@dataclass(frozen=True)
class Limits:
    """Development choices for profiling and the 16-question runs, not a final comparison protocol."""
    max_context_tokens: int = 32768          # prompt plus everything generated within one request
    max_new_tokens_thinking: int = 4096      # per assistant turn
    max_new_tokens_nonthinking: int = 1024   # per assistant turn
    max_generated_tokens_thinking: int = 16384     # per request, all turns
    max_generated_tokens_nonthinking: int = 4096   # per request, all turns
    max_tool_calls: int = 8                  # describe_schema + lookup_values + execute_select, per request
    max_repairs_llm: int = 1                 # Q27 and B4: at most one repair after a failed final SQL
    max_repairs_controller: int = 0          # C4 and T4: no repair in this work order
    request_timeout_seconds_4b: float = 600.0
    request_timeout_seconds_27b: float = 1800.0
    sql_timeout_seconds: float = 30.0        # tools and scorer (the official meta_time_out)
    preview_rows: int = 20                   # rows shown to a model; never the rows that are scored
    lookup_max_values: int = 20
    retrieval_k: int = 3                     # training examples given to every arm

    def as_dict(self) -> dict:
        return asdict(self)


# --- the request record (records.py validates it) -----------------------------------------------------------------
RECORD_FIELDS = (
    "task_id", "db_id", "arm", "config_identity", "input_identity", "snapshot_sha256",
    "attempts",          # every attempt: proposed plan or SQL, tool calls, observations, errors, repair flag
    "final",             # {"kind": "sql" | "abstain", "sql": ..., "reason": ...}
    "score",             # scorer result for the final output (None until scored)
    "tokens",            # {"prompt", "generated", "reasoning", "content", "tool"} per request
    "controller_work",   # C4/T4: steps, families selected, heads executed; None for other arms
    "elapsed_seconds", "peak_memory_bytes", "timed_out", "error",
)


@dataclass
class CostUnit:
    """The primary serving-cost unit, declared before any run: total end-to-end elapsed seconds across all requests
    divided by correctly completed requests, at batch size one, on this host, under the warm-cache policy below.
    CPU-side request work is included. Cold start, throughput, memory, preprocessing and training are reported
    separately."""
    warm_cache_policy: str = ("model loaded and one excluded warm-up request served; each pinned database file read "
                              "once before timing; the retrieval index built before timing")
    batch_size: int = 1
    numerator: str = "sum of per-request end-to-end elapsed seconds, all requested tasks"
    denominator: str = "requests whose final output scores correct under the pinned scorer"
    extra: dict = field(default_factory=dict)
