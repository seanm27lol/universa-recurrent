"""The frozen backbone: one forward pass of Qwen3.5-4B over a fixed context, cached by content, charged per request.

**Example.** For `bird23tf-00000` ("Who is the director of the movie Sex, Drink and Bloodshed?"), `build_context`
writes the text below (template `context-v1`); the backbone reads it once, without gradient; the controller then
points into it.

    <|im_start|>system
    Translate the question about a SQLite database into one SQLite query.<|im_end|>
    <|im_start|>user
    ### Examples from other databases
    Example 1 (database: ...)
    Question: ...
    Evidence: ...
    SQL: ...
    (three retrieved training examples)
    ### Question
    Who is the director of the movie Sex, Drink and Bloodshed?
    ### Evidence
    Sex, Drink and Bloodshed refers to movie title = 'Sex, Drink and Bloodshed';
    ### Database schema
    DATABASE movie_platform
    TABLE lists
      user_id INTEGER FK -> lists_users.user_id -- ...
    ...<|im_end|>
    <|im_start|>assistant

**Why this order.** The backbone is causal: a token's state sees only what precedes it. The question and evidence come
before the schema, so every schema item's state has read the question (schema linking happens inside the backbone).
The retrieved examples come first because they condition everything else. The question and evidence regions are the
literal pointer targets; each schema item's pooled vector (the mean of its tokens' states, located through the
`SerializedSchema` character spans) is a table or column pointer target. The last token's state is the request
summary that initializes the controller.

**Layer.** `layer=-1` (the default) is the final layer after the model's last RMSNorm (`last_hidden_state`). Any
other `l` is `hidden_states[l]` (0 = the token embeddings, `l` = the output of decoder layer `l`).

**Cache.** Only the immutable backbone states are cached, under `contracts.CACHE / "encodings"`, keyed by the sha256
of the canonical JSON of the backbone's identity and the exact token ids. The identity covers:
- the model repo, revision, layer and dtype;
- the encoder version;
- the device type and name;
- torch and transformers;
- the linear-attention implementation transformers actually bound (`runtime.bound_linear_attention`; on CPU the
  reference functions, which `reference_kernels` forces).

So states computed with different kernels, software or devices never share an entry. A file is written once (temporary
file, then an atomic link) and never overwritten; the cost record says whether a request computed its states or read
them.

**Kernels.** Importing this module imports `runtime`, which applies `TI_TNN_SCALING_LINEAR_KERNELS` (default
`reference`) before transformers loads. If transformers was imported earlier with other kernels, the identity still
records what was bound.

**Costs.** Every request returns `cost`: tokenization, forward (or read) seconds, tokens, bytes written and read,
cache hit, and the context's own identity. Nothing is shared silently between requests.

**Limits.** A context longer than `max_tokens` (32,768, the study's context limit) first loses retrieved examples,
then schema descriptions (`schema.serialize` with a character budget). If the bare schema still does not fit, the
request fails with `ContextTooLong`, which the inference path records as an abstention.
"""
from __future__ import annotations

import bisect
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import actions
from . import contracts as C
from . import runtime  # noqa: F401  (applies TI_TNN_SCALING_LINEAR_KERNELS before transformers is imported)

ENCODER_VERSION = "encode-v2"
TEMPLATE_VERSION = "context-v1"
DEFAULT_LAYER = -1
REGIONS = ("other", "examples", "question", "evidence", "schema")  # token type ids, in this order

SYSTEM_TEXT = "Translate the question about a SQLite database into one SQLite query."


class ContextTooLong(RuntimeError):
    """Even the bare schema does not fit the token limit."""


# --- the context -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Context:
    """The text the backbone reads, its regions and the absolute character span of every schema item."""
    text: str
    regions: dict               # name -> (start, end) character range
    item_spans: tuple           # ((start, end), ...) in SerializedSchema.items order
    examples_used: int
    template: str = TEMPLATE_VERSION

    def identity(self) -> str:
        return hashlib.sha256(json.dumps([self.template, self.text, list(self.item_spans)],
                                         ensure_ascii=False).encode()).hexdigest()


def build_context(task: C.Task, serialized: C.SerializedSchema, examples=()) -> Context:
    """Template context-v1 (see the module docstring)."""
    parts: list = []
    regions: dict = {}
    pos = 0

    def emit(text: str, region: "str | None" = None):
        nonlocal pos
        if region is not None:
            regions[region] = (pos, pos + len(text))
        parts.append(text)
        pos += len(text)

    emit(f"<|im_start|>system\n{SYSTEM_TEXT}<|im_end|>\n<|im_start|>user\n")
    start = pos
    emit("### Examples from other databases\n")
    for k, ex in enumerate(examples, 1):
        emit(f"Example {k} (database: {ex.db_id})\nQuestion: {ex.question}\nEvidence: {ex.evidence}\nSQL: {ex.sql}\n\n")
    regions["examples"] = (start, pos)
    emit("### Question\n")
    emit(task.question, "question")
    emit("\n### Evidence\n")
    emit(task.evidence, "evidence")
    emit("\n### Database schema\n")
    schema_start = pos
    emit(serialized.text, "schema")
    emit("<|im_end|>\n<|im_start|>assistant\n")
    item_spans = tuple((schema_start + it.start, schema_start + it.end) for it in serialized.items)
    return Context(text="".join(parts), regions=regions, item_spans=item_spans, examples_used=len(examples))


# --- tokenization ----------------------------------------------------------------------------------------------------
def tokenize(tokenizer, text: str) -> tuple:
    """(token ids, character offsets) with the fast tokenizer's offset mapping, no added special tokens."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def token_range(offsets: list, start: int, end: int) -> "tuple | None":
    """Inclusive token range of the tokens that overlap characters [start, end); None if none does."""
    ends = [o[1] for o in offsets]
    i = bisect.bisect_right(ends, start)
    j = i
    found = None
    while j < len(offsets) and offsets[j][0] < end:
        if offsets[j][1] > offsets[j][0] and offsets[j][1] > start:
            found = (found[0], j) if found else (j, j)
        j += 1
    return found


def span_index(context: Context, offsets: list) -> actions.SpanIndex:
    """The question and evidence regions as literal-pointer targets (absolute token indices, clipped offsets)."""
    regions = []
    for name in ("question", "evidence"):
        r0, r1 = context.regions[name]
        toks = []
        for k, (s, e) in enumerate(offsets):
            if e > s and s < r1 and e > r0:
                toks.append((k, max(s, r0) - r0, min(e, r1) - r0))
        regions.append(actions.Region(name, context.text[r0:r1], tuple(toks)))
    return actions.SpanIndex(regions)


def token_types(context: Context, offsets: list) -> list:
    """Region id of every token (REGIONS order; 0 for template text)."""
    out = []
    for s, e in offsets:
        kind = 0
        for name in ("examples", "question", "evidence", "schema"):
            r0, r1 = context.regions[name]
            if s < r1 and e > r0 and e > s:
                kind = REGIONS.index(name)
                break
        out.append(kind)
    return out


class MockTokenizer:
    """A deterministic stand-in with the fast tokenizer's interface (tests only): words, punctuation and the chat
    markers, ids by stable hashing into `vocab_size`."""
    _PIECE = re.compile(r"<\|[a-z_]+\|>|\s*\w+|\s*[^\w\s]|\s+", re.UNICODE)

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.name_or_path = f"mock-tokenizer-{vocab_size}"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids, offsets = [], []
        for m in self._PIECE.finditer(text):
            piece = m.group(0)
            ids.append(int(hashlib.sha256(piece.encode()).hexdigest(), 16) % self.vocab_size)
            offsets.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# --- the backbone ----------------------------------------------------------------------------------------------------
_KERNEL_FUNCTIONS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                     "causal_conv1d_update")


@contextlib.contextmanager
def reference_kernels(model):
    """Run Qwen3.5's linear-attention layers with their PyTorch reference functions while the block runs.

    transformers binds the flash-linear-attention kernels at import time when that package is installed, and those
    are Triton kernels that need a GPU. A CPU forward (tests, the tiny model) therefore swaps the module's functions for
    the reference implementations they wrap, and restores them afterwards. GPU runs never enter this block."""
    module = importlib.import_module(type(model).__module__)
    saved = {}
    for name in _KERNEL_FUNCTIONS:
        fn = getattr(module, name, None)
        if fn is not None and inspect.unwrap(fn) is not fn:
            saved[name] = fn
            setattr(module, name, inspect.unwrap(fn))
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(module, name, fn)


class Backbone:
    """A frozen decoder-only text model plus its tokenizer. `states(token_ids)` runs it once, without gradient."""

    def __init__(self, model, tokenizer, *, repo: str, revision: str, layer: int = DEFAULT_LAYER,
                 device: str = "cpu"):
        import torch
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.repo, self.revision, self.layer, self.device = repo, revision, int(layer), device
        self.dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        self.hidden_size = int(self.model.config.hidden_size)
        self._torch = torch
        import transformers
        cuda = device.startswith("cuda")
        self.platform = {"device_type": "cuda" if cuda else "cpu",
                         "device_name": torch.cuda.get_device_name(torch.device(device)) if cuda else "cpu",
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "linear_attention": runtime.bound_linear_attention() if cuda
                         else "reference (forced on CPU by reference_kernels)"}

    @classmethod
    def from_pinned(cls, device: str = "cuda", layer: int = DEFAULT_LAYER, model=None, tokenizer=None):
        """The pinned Qwen3.5-4B (contracts.BACKBONE) in BF16 from the local cache (no network). Pass `model` and
        `tokenizer` from an already loaded `runtime.Runtime("B4")` to share its weights instead of loading again."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        from . import identity
        snapshot = identity.model_snapshot_dir("B4")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.bfloat16,
                                                                local_files_only=True).to(device)
        text_model = model.model.language_model if hasattr(model, "model") else model
        return cls(text_model, tokenizer, repo=C.BACKBONE["repo"], revision=C.BACKBONE["revision"], layer=layer,
                   device=device)

    @classmethod
    def tiny_random(cls, tokenizer=None, *, hidden_size: int = 32, layers: int = 4, seed: int = 0,
                    layer: int = DEFAULT_LAYER):
        """A randomly initialised Qwen3.5 text model on CPU (tests): same class and interface as the pinned
        backbone's language model, never its weights."""
        import torch
        from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel
        tokenizer = tokenizer or MockTokenizer()
        vocab = len(tokenizer)
        cfg = Qwen3_5TextConfig(vocab_size=int(vocab), hidden_size=hidden_size, intermediate_size=2 * hidden_size,
                                num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=hidden_size // 4, linear_key_head_dim=8, linear_value_head_dim=8,
                                linear_num_key_heads=2, linear_num_value_heads=4,
                                layer_types=(["linear_attention"] * (layers - 1) + ["full_attention"]),
                                full_attention_interval=layers)
        torch.manual_seed(seed)
        model = Qwen3_5TextModel(cfg).float()
        digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]
        return cls(model, tokenizer, repo=f"tiny-random/qwen3_5-text-{hidden_size}x{layers}",
                   revision=f"seed{seed}-{digest}", layer=layer, device="cpu")

    def identity(self) -> dict:
        return {"repo": self.repo, "revision": self.revision, "layer": self.layer, "dtype": self.dtype,
                "hidden_size": self.hidden_size, "encoder": ENCODER_VERSION, **self.platform}

    def cache_key(self, token_ids: list) -> str:
        material = json.dumps([self.identity(), list(map(int, token_ids))], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode()).hexdigest()

    def states(self, token_ids: list):
        """[T, D] states of the chosen layer (backbone dtype, on CPU)."""
        torch = self._torch
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        guard = reference_kernels(self.model) if not self.device.startswith("cuda") else contextlib.nullcontext()
        with torch.inference_mode(), guard:
            if self.layer == -1:
                out = self.model(input_ids=ids, use_cache=False)
                h = out.last_hidden_state
            else:
                out = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
                h = out.hidden_states[self.layer]
        return h[0].detach().to("cpu").contiguous()


# --- the cache -------------------------------------------------------------------------------------------------------
def cache_path(key: str, root: "Path | None" = None) -> Path:
    root = Path(root) if root is not None else C.CACHE / "encodings"
    return root / key[:2] / f"{key}.safetensors"


def _write_once(path: Path, tensor, metadata: dict) -> int:
    from safetensors.torch import save_file
    if path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    save_file({"states": tensor}, str(tmp), metadata={k: str(v) for k, v in metadata.items()})
    try:
        os.link(tmp, path)  # atomic and never overwrites: fails if another writer won
        written = path.stat().st_size
    except FileExistsError:
        written = 0
    finally:
        tmp.unlink(missing_ok=True)
    return written


def cached_states(backbone: Backbone, token_ids: list, root: "Path | None" = None, use_cache: bool = True) -> tuple:
    """(states [T, D], cost dict). Reads the immutable cache entry, or computes and writes it once. With
    `use_cache=False` it always computes and neither reads nor writes (a timed serving request pays its own forward)."""
    from safetensors.torch import load_file
    key = backbone.cache_key(token_ids)
    path = cache_path(key, root)
    t0 = time.perf_counter()
    if not use_cache:
        if backbone.device.startswith("cuda"):
            backbone._torch.cuda.synchronize()
        states = backbone.states(token_ids)
        if backbone.device.startswith("cuda"):
            backbone._torch.cuda.synchronize()
        return states, {"cache_key": key, "cache_hit": False, "cache_used": False,
                        "forward_seconds": time.perf_counter() - t0, "bytes_written": 0, "bytes_read": 0}
    if path.exists():
        states = load_file(str(path))["states"]
        if tuple(states.shape) != (len(token_ids), backbone.hidden_size):
            raise RuntimeError(f"cache entry {path} has shape {tuple(states.shape)}")
        return states, {"cache_key": key, "cache_hit": True, "forward_seconds": 0.0,
                        "read_seconds": time.perf_counter() - t0, "bytes_written": 0,
                        "bytes_read": path.stat().st_size}
    if backbone.device.startswith("cuda"):
        backbone._torch.cuda.synchronize()
    t0 = time.perf_counter()
    states = backbone.states(token_ids)
    if backbone.device.startswith("cuda"):
        backbone._torch.cuda.synchronize()
    forward = time.perf_counter() - t0
    import torch
    import transformers
    meta = {**backbone.identity(), "tokens": len(token_ids), "torch": torch.__version__,
            "transformers": transformers.__version__, "device": backbone.device,
            "device_name": torch.cuda.get_device_name(0) if backbone.device.startswith("cuda") else "cpu"}
    t1 = time.perf_counter()
    written = _write_once(path, states, meta)
    return states, {"cache_key": key, "cache_hit": False, "forward_seconds": forward,
                    "write_seconds": time.perf_counter() - t1, "bytes_written": written, "bytes_read": 0}


# --- one request -----------------------------------------------------------------------------------------------------
@dataclass
class Encoding:
    """Everything the controller reads for one request. No gold: built from the Task, schema and examples only."""
    task_id: str
    context: Context
    token_ids: list
    offsets: list
    states: object               # torch.Tensor [T, D], backbone dtype
    item_ranges: list            # [(first token, last token)] per schema item
    token_types: list            # region id per token (REGIONS)
    spans: actions.SpanIndex
    schema: actions.SchemaItems
    cost: dict = field(default_factory=dict)

    def item_vectors(self):
        """[N, D] float32: the mean state over each schema item's tokens."""
        import torch
        states = self.states.float()
        out = torch.zeros(len(self.item_ranges), states.shape[1])
        for k, (i, j) in enumerate(self.item_ranges):
            out[k] = states[i:j + 1].mean(0)
        return out


def fit_context(task: C.Task, schema, examples, tokenizer, max_tokens: int) -> tuple:
    """(Context, SerializedSchema, token ids, offsets) within `max_tokens`: drop examples from the last, then shorten
    descriptions (needs a `contracts.Schema`); else ContextTooLong."""
    from . import schema as schema_mod
    full_schema = schema if isinstance(schema, C.Schema) else None
    serialized = schema_mod.serialize(schema) if full_schema is not None else schema
    examples = list(examples)
    while True:
        ctx = build_context(task, serialized, examples)
        ids, offsets = tokenize(tokenizer, ctx.text)
        if len(ids) <= max_tokens:
            return ctx, serialized, ids, offsets
        if examples:
            examples.pop()
            continue
        if full_schema is None:
            raise ContextTooLong(f"{task.task_id}: {len(ids)} tokens > {max_tokens} and no schema budget to cut")
        budget = int(len(serialized.text) * max_tokens / len(ids) * 0.9)
        shorter = schema_mod.serialize(full_schema, budget_chars=max(budget, 0))
        if shorter.text == serialized.text:
            raise ContextTooLong(f"{task.task_id}: {len(ids)} tokens > {max_tokens} with the bare schema")
        serialized = shorter


def encode_request(backbone: Backbone, task: C.Task, schema, examples=(), *, max_tokens: "int | None" = None,
                   cache_root: "Path | None" = None, use_cache: bool = True) -> Encoding:
    """Build the context, tokenize, run (or read) the frozen backbone and locate items and spans."""
    t0 = time.perf_counter()
    max_tokens = max_tokens or C.Limits().max_context_tokens
    ctx, serialized, ids, offsets = fit_context(task, schema, examples, backbone.tokenizer, max_tokens)
    tok_seconds = time.perf_counter() - t0
    states, cost = cached_states(backbone, ids, cache_root, use_cache=use_cache)
    ranges = []
    for (a, b) in ctx.item_spans:
        r = token_range(offsets, a, b)
        if r is None:
            raise RuntimeError(f"{task.task_id}: schema item at {a}:{b} has no tokens")
        ranges.append(r)
    cost.update({"tokens": len(ids), "tokenize_seconds": tok_seconds, "seconds": time.perf_counter() - t0,
                 "context_identity": ctx.identity(), "examples_used": ctx.examples_used,
                 "schema_chars": len(serialized.text), "template": TEMPLATE_VERSION, **backbone.identity()})
    return Encoding(task_id=task.task_id, context=ctx, token_ids=ids, offsets=offsets, states=states,
                    item_ranges=ranges, token_types=token_types(ctx, offsets), spans=span_index(ctx, offsets),
                    schema=actions.SchemaItems(serialized.items), cost=cost)


def encode_tasks(backbone: Backbone, tasks, *, schema_for=None, examples_for=None, k: "int | None" = None,
                 cache_root: "Path | None" = None, log=None) -> list:
    """Encodings for a task list (the CLI entry point). Defaults: the pinned schema (`schema.load_schema`) and the
    benchmark's retrieval (`retrieval.retrieve`, k = 3)."""
    from . import retrieval
    from . import schema as schema_mod
    k = C.Limits().retrieval_k if k is None else k
    schema_for = schema_for or (lambda t: schema_mod.load_schema(t.db_id))
    examples_for = examples_for or (lambda t: retrieval.retrieve(t, k=k))
    out = []
    for n, task in enumerate(tasks):
        out.append(encode_request(backbone, task, schema_for(task), examples_for(task), cache_root=cache_root))
        if log and (n + 1) % 16 == 0:
            log(f"encoded {n + 1}/{len(tasks)}")
    return out
