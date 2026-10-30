"""The frozen-backbone encoder, the C4/T4 controllers, training and decoding (CPU only, synthetic data).

The backbone here is a tiny randomly initialised Qwen3.5 text model (the same class as the pinned model's language
model, never its weights) with a deterministic mock tokenizer. What these tests establish:
- **The cache** is content-keyed, written once, never overwritten, and every request's cost record says whether it
  computed or read its states.
- **C4/T4 supervision equivalence.** Under one seed the arms have identical trunks, identical teacher-forcing targets
  and the same loss terms. Swapping T4's semantic module for C4's reproduces C4's losses exactly, so nothing else
  differs.
- **Selected-only execution.** In training and in decoding, T4 runs each specialist at most once per call, only on
  the rows of its family, and never runs a head that has no rows.
- **Batch equivalence.** Batched losses equal the sum of single-example losses; batched greedy decoding equals
  decoding each request alone.
- **Checkpoints.** Content-hashed files reload exactly; resuming reproduces the uninterrupted run bit for bit;
  tampering is detected.
- **Parameter matching.** c4_params matches T4's total trainable parameters within 1% at the real backbone width.
- **From the review of 641ff3c:** the cache key covers the kernels, device, software and encoder version; a grammar
  dead end abstains (`dead_end`) instead of crashing; one failing request does not stop an evaluation; uninterrupted
  runs give byte-identical checkpoints, and a resumed run equal weight and optimizer digests, with superseded log lines
  kept aside.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("sqlglot")
from scaling import actions as A  # noqa: E402
from scaling import contracts as C  # noqa: E402
from scaling import controller as K  # noqa: E402
from scaling import encode as E  # noqa: E402
from scaling import infer_controller as I  # noqa: E402
from scaling import schema as schema_mod  # noqa: E402
from scaling import sql_to_ir as X  # noqa: E402
from scaling import train as T  # noqa: E402

COLUMNS = {"movies": [("movie_id", "INTEGER", True), ("title", "TEXT", False), ("director", "TEXT", False),
                      ("year", "INT", False)],
           "ratings": [("rating_id", "INTEGER", True), ("movie_id", "INT", False), ("score", "INT", False)]}
SCHEMA = C.Schema(db_id="cinema", snapshot_sha256="0" * 64, tables=tuple(
    C.SchemaTable(name=t, columns=tuple(C.SchemaColumn(table=t, name=n, type=ty, primary_key=pk,
                                                       description=f"the {n} of a {t[:-1]}") for n, ty, pk in cols))
    for t, cols in COLUMNS.items()))
REQUESTS = [
    ("t1", "Who directed the movie Alpha?", "Alpha refers to title = 'Alpha';",
     "SELECT director FROM movies WHERE title = 'Alpha'"),
    ("t2", "How many movies did Jo direct after 1999?", "Jo refers to director = 'Jo';",
     "SELECT COUNT(*) FROM movies WHERE director = 'Jo' AND year > 1999"),
    ("t3", "Which movie has the highest average score?", "",
     "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
     "GROUP BY T1.title ORDER BY AVG(T2.score) DESC LIMIT 1"),
]
EXAMPLE = C.RetrievedExample(task_id="x0", db_id="other", question="How many rows?", evidence="",
                             sql="SELECT COUNT(*) FROM t", score=1.0)


@pytest.fixture(scope="module")
def backbone():
    return E.Backbone.tiny_random(E.MockTokenizer(), hidden_size=16, layers=2, seed=0)


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("encodings")


@pytest.fixture(scope="module")
def prepared(backbone, cache_root):
    out = []
    for task_id, question, evidence, sql in REQUESTS:
        task = C.Task(task_id=task_id, db_id="cinema", question=question, evidence=evidence, split="train")
        enc = E.encode_request(backbone, task, SCHEMA, [EXAMPLE], cache_root=cache_root)
        plan = X.convert(sql, enc.schema.index)
        out.append(K.prepare(enc, A.trace(plan, enc.schema, enc.spans)))
    return out


def models(seed=100):
    return {arm: T.build_model(arm, 16, seed) for arm in K.ARMS}


# --- the encoder ------------------------------------------------------------------------------------------------------
def test_cache_is_content_keyed_immutable_and_charged(backbone, tmp_path):
    task = C.Task(task_id="t9", db_id="cinema", question="Who?", evidence="", split="dev")
    first = E.encode_request(backbone, task, SCHEMA, [], cache_root=tmp_path)
    path = E.cache_path(first.cost["cache_key"], tmp_path)
    stamp = path.stat().st_mtime_ns
    second = E.encode_request(backbone, task, SCHEMA, [], cache_root=tmp_path)
    assert not first.cost["cache_hit"] and first.cost["bytes_written"] > 0
    assert second.cost["cache_hit"] and second.cost["bytes_written"] == 0 and path.stat().st_mtime_ns == stamp
    assert torch.equal(first.states, second.states)
    assert first.cost["cache_key"] == backbone.cache_key(first.token_ids)
    other = E.Backbone(backbone.model, backbone.tokenizer, repo=backbone.repo, revision=backbone.revision, layer=1)
    assert other.cache_key(first.token_ids) != first.cost["cache_key"]  # the layer is part of the key


def test_encoding_locates_items_and_literal_regions(prepared, backbone, cache_root):
    task = C.Task(task_id="t1", db_id="cinema", question=REQUESTS[0][1], evidence=REQUESTS[0][2], split="train")
    enc = E.encode_request(backbone, task, SCHEMA, [EXAMPLE], cache_root=cache_root)
    for it, (i, j) in zip(enc.schema.items, enc.item_ranges):
        assert enc.context.text[enc.offsets[i][0]:enc.offsets[j][1]].strip().startswith(it.column or it.table)
    assert [r.name for r in enc.spans.regions] == ["question", "evidence"]
    assert enc.spans.regions[0].text == REQUESTS[0][1] and enc.cost["tokens"] == len(enc.token_ids)


# --- C4 and T4 ---------------------------------------------------------------------------------------------------------
def test_c4_t4_supervision_is_identical(prepared):
    batch = K.collate(prepared)
    ms = models()
    trunk = ms["T4"].trunk.state_dict()
    for arm in ("c4_active", "c4_params"):
        other = ms[arm].trunk.state_dict()
        assert trunk.keys() == other.keys() and all(torch.equal(trunk[k], other[k]) for k in trunk)
    out = {arm: m.losses(batch) for arm, m in ms.items()}
    for arm in K.ARMS:
        assert set(out[arm]) == set(out["T4"])
        assert (out[arm]["steps"], out[arm]["op_steps"], out[arm]["pointer_steps"]) == \
            (out["T4"]["steps"], out["T4"]["op_steps"], out["T4"]["pointer_steps"])
        assert torch.equal(out[arm]["family"], out["T4"]["family"])  # the shared family classifier
    swapped = ms["T4"]
    swapped.semantic = ms["c4_active"].semantic                       # only the semantic module differs
    again = swapped.losses(batch)
    for key in ("total", "family", "op", "pointer"):
        assert torch.equal(again[key], out["c4_active"][key])


def test_semantic_modules_share_one_output_interface(prepared):
    ms = models()
    state = torch.randn(6, 512)
    family = torch.tensor([0, 1, 2, 3, 1, 4])
    for m in ms.values():
        logits, queries = m.semantic(state, family)
        assert logits.shape == (6, K.UNION_SIZE) and queries.shape == (6, K.N_POINTERS, 512)
        for r, f in enumerate(family.tolist()):
            live = (logits[r] > K.NEG / 2).nonzero().flatten().tolist()
            if f == 4:
                assert live == [] and not queries[r].any()            # STOP rows run no head
            else:
                lo, hi = A.FAMILY_SLICES[f]
                assert live and min(live) >= lo and max(live) < hi    # only the chosen family's operators


def test_t4_runs_only_the_selected_heads(prepared):
    model = models()["T4"]
    seen = []
    for f, head in enumerate(model.semantic.heads):
        head.register_forward_hook(lambda mod, inp, out, f=f: seen.append((f, inp[0].shape[0])))
    batch = K.collate(prepared)
    model.losses(batch)
    st = batch.steps
    want = {f: int((st["valid"] & (st["family"] == f)).sum()) for f in range(4)}
    assert sorted(seen) == sorted((f, n) for f, n in want.items() if n)  # one call per non-empty family
    seen.clear()
    calls_before = len(model.semantic.calls)
    decoded = I.decode_batch(model, prepared, max_steps=40)
    per_call = model.semantic.calls[calls_before:]
    assert all(n > 0 for _, n in per_call)
    heads = {}
    for d in decoded:
        for h, fam in zip(d.work["heads"], d.work["families"]):
            assert h == (None if fam == "STOP" else fam)
            if h is not None:
                heads[h] = heads.get(h, 0) + 1
    rows = {}
    for f, n in seen:
        rows[A.FAMILIES[f]] = rows.get(A.FAMILIES[f], 0) + n
    assert rows == heads                                              # every head row is a row of that family


def test_batched_and_single_paths_agree(prepared):
    for arm, model in models().items():
        batch = model.losses(K.collate(prepared))
        singles = [model.losses(K.collate([p])) for p in prepared]
        for key in ("total", "family", "op", "pointer"):
            assert torch.allclose(batch[key], sum(s[key] for s in singles), rtol=1e-5, atol=1e-4), (arm, key)
        together = I.decode_batch(model, prepared, max_steps=40)
        alone = [I.decode_one(model, p, max_steps=40) for p in prepared]
        for a, b in zip(together, alone):
            assert (a.actions, a.sql, a.abstain_reason) == (b.actions, b.sql, b.abstain_reason)
            assert (a.work["families"], a.work["heads"], a.work["macs"]) == (b.work["families"], b.work["heads"],
                                                                             b.work["macs"])


def test_decoding_is_legal_and_accounted(prepared):
    model = models()["c4_params"]
    for d in I.decode_batch(model, prepared, max_steps=40):
        assert (d.sql is None) == (d.abstain_reason is not None)
        assert d.abstain_reason in (None, "budget_exhausted", "invalid_plan", "unsupported_operation")
        assert d.work["steps"] == len(d.work["families"]) and d.work["macs"]["semantic"] > 0
        record = d.as_record()
        assert json.loads(json.dumps(record))["final"]["kind"] in ("sql", "abstain")


# --- training and checkpoints -----------------------------------------------------------------------------------------
def test_checkpoints_reload_exactly_and_resume_bit_for_bit(prepared, tmp_path):
    cfg = T.TrainConfig(arm="T4", updates=3, effective_batch=2, micro_batch=1, checkpoint_every=1)
    full = T.build_model("T4", 16, 100)
    run = T.fit(full, prepared, cfg, tmp_path / "full", log=None)
    assert [c["update"] for c in run["checkpoints"]] == [0, 1, 2]
    losses = [json.loads(line)["loss"] for line in open(run["log"])]
    assert losses[-1] < losses[0]
    first = run["checkpoints"][0]
    ckpt = T.load_checkpoint(first["path"])
    assert ckpt["sha256"] == first["sha256"] and ckpt["update"] == 0
    resumed = T.build_model("T4", 16, 7)                     # deliberately different initial weights
    T.fit(resumed, prepared, cfg, tmp_path / "resumed", resume_from=first["path"], log=None)
    final = full.state_dict()
    assert all(torch.equal(final[k], v) for k, v in resumed.state_dict().items())
    reloaded = T.model_from_checkpoint(T.load_checkpoint(run["checkpoints"][-1]["path"]))
    assert all(torch.equal(final[k], v) for k, v in reloaded.state_dict().items())
    opt = T.make_optimizer(reloaded, cfg)
    again = T.save_checkpoint(tmp_path / "again", reloaded, opt, 9, {"x": 1})
    assert T.save_checkpoint(tmp_path / "again", reloaded, opt, 9, {"x": 1})["path"] == again["path"]  # written once
    path = Path(first["path"])
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    tampered = tmp_path / path.name
    tampered.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        T.load_checkpoint(tampered)


def test_sampling_stream_is_a_pure_function():
    cfg = T.TrainConfig(arm="T4")
    assert T.batch_indices(128, 5, cfg) == T.batch_indices(128, 5, cfg)
    seen = [i for u in range(4) for i in T.batch_indices(128, u, cfg)]
    assert sorted(seen) == list(range(128))                  # four updates of 32 cover one epoch exactly


def test_c4_params_matches_t4_at_the_real_backbone_width():
    reports = {arm: K.parameter_report(T.build_model(arm, 2560, 100)) for arm in K.ARMS}
    t4, c4p, c4a = reports["T4"], reports["c4_params"], reports["c4_active"]
    assert t4["trunk"] == c4p["trunk"] == c4a["trunk"]
    assert abs(c4p["total"] - t4["total"]) / t4["total"] < 0.01
    assert c4a["semantic"] < c4p["semantic"]
    cfg = {arm: K.matched_config(arm, 2560) for arm in K.ARMS}
    for f in range(4):
        head = K.semantic_step_macs(cfg["T4"], "T4", f)
        assert head < K.semantic_step_macs(cfg["c4_active"], "c4_active", f) < \
            K.semantic_step_macs(cfg["c4_params"], "c4_params", f)


# --- the review of 641ff3c: cache identity, dead ends, per-request guard, deterministic checkpoints -------------------
def test_the_cache_key_covers_kernels_device_software_and_encoder(backbone):
    ident = backbone.identity()
    assert ident["encoder"] == E.ENCODER_VERSION and ident["device_type"] == "cpu"
    assert ident["linear_attention"].startswith("reference") and ident["torch"] == torch.__version__
    ids = [1, 2, 3]
    for field, value in (("linear_attention", {"torch_chunk_gated_delta_rule": "fla"}), ("device_type", "cuda"),
                         ("device_name", "NVIDIA GB10"), ("transformers", "0.0")):
        other = E.Backbone(backbone.model, backbone.tokenizer, repo=backbone.repo, revision=backbone.revision)
        other.platform = {**other.platform, field: value}
        assert other.cache_key(ids) != backbone.cache_key(ids), field


def test_a_dead_end_abstains_instead_of_crashing(prepared, monkeypatch):
    model = models()["T4"]
    monkeypatch.setattr(A.Machine, "legal_families", lambda self: [False] * len(A.FAMILIES + ("STOP",)))
    for d in I.decode_batch(model, prepared, max_steps=40):
        assert d.abstain_reason == "dead_end" and d.sql is None and d.work["steps"] == 0
    monkeypatch.undo()

    # a pointer operator with no legal target, or a target lookup that fails: abstain after that one decision
    monkeypatch.setattr(A.Machine, "options", lambda self: {"FROM_TABLE": ("table", ())})
    for d in I.decode_batch(model, prepared, max_steps=40):
        assert d.abstain_reason == "dead_end" and "no legal table target" in d.detail
        assert d.work["steps"] == len(d.work["families"]) == len(d.work["heads"]) == 1

    def lookup_fails(self, op):
        raise KeyError(op)
    monkeypatch.setattr(A.Machine, "options", lambda self: {"FROM_TABLE": ("table", (0,))})
    monkeypatch.setattr(A.Machine, "pointer_targets", lookup_fails)
    for d in I.decode_batch(model, prepared, max_steps=40):
        assert d.abstain_reason == "dead_end" and "KeyError" in d.detail and d.work["steps"] == 1


def test_one_failing_request_does_not_stop_the_evaluation(backbone, monkeypatch):
    from scaling import data, retrieval
    tasks = {tid: C.Task(task_id=tid, db_id="cinema", question=q, evidence=e, split="dev")
             for tid, q, e, _ in REQUESTS}
    monkeypatch.setattr(data, "task", lambda tid: tasks[tid])
    monkeypatch.setattr(schema_mod, "load_schema", lambda db: SCHEMA)
    monkeypatch.setattr(retrieval, "retrieve", lambda task: [EXAMPLE])
    real = E.encode_request

    def flaky(bb, task, *a, **kw):
        if task.task_id == "t2":
            raise RuntimeError("boom")
        return real(bb, task, *a, **kw)
    monkeypatch.setattr(E, "encode_request", flaky)
    out = I.evaluate(models()["c4_params"], list(tasks), backbone, score=False, max_steps=40, log=None)
    assert [r["task_id"] for r in out] == ["t1", "t2", "t3"]
    assert out[1]["final"] == {"kind": "abstain", "sql": None, "reason": "error: RuntimeError", "detail": "boom"}
    assert out[0]["controller_work"] is not None and out[2]["controller_work"] is not None


def test_checkpoint_bytes_agree_across_runs_and_resume_keeps_superseded_lines(prepared, tmp_path):
    import random

    import numpy as np
    cfg = T.TrainConfig(arm="T4", updates=3, effective_batch=2, micro_batch=1, checkpoint_every=1)
    first = T.fit(T.build_model("T4", 16, 100), prepared, cfg, tmp_path / "a", log=None)
    random.random(), np.random.rand()  # disturb the global generators, as another process would differ
    second = T.fit(T.build_model("T4", 16, 100), prepared, cfg, tmp_path / "b", log=None)
    assert [c["sha256"] for c in first["checkpoints"]] == [c["sha256"] for c in second["checkpoints"]]
    resumed = T.fit(T.build_model("T4", 16, 7), prepared, cfg, tmp_path / "a",
                    resume_from=first["checkpoints"][0]["path"], log=None)
    for key in ("model_digest", "optimizer_digest"):  # equal tensors; the file bytes may differ (pickle sharing)
        assert resumed["checkpoints"][-1][key] == first["checkpoints"][-1][key]
    kept = [json.loads(line)["update"] for line in open(tmp_path / "a" / "train_log.jsonl")]
    moved = [json.loads(line)["update"] for line in open(tmp_path / "a" / "train_log.superseded.jsonl")]
    assert kept == [0, 1, 2] and moved == [1, 2]


def test_smoke_records_pass_the_common_record_checks(backbone, monkeypatch):
    from scaling import cli, data, retrieval
    from scaling import records as R
    tasks = {tid: C.Task(task_id=tid, db_id="cinema", question=q, evidence=e, split="dev")
             for tid, q, e, _ in REQUESTS}
    monkeypatch.setattr(data, "task", lambda tid: tasks[tid])
    monkeypatch.setattr(schema_mod, "load_schema", lambda db: SCHEMA)
    monkeypatch.setattr(retrieval, "retrieve", lambda task: [EXAMPLE])
    snap = C.Snapshot(db_id="cinema", path="/nonexistent", sha256="0" * 64, bytes=1)
    evals = I.evaluate(models()["c4_params"], list(tasks), backbone, score=False, max_steps=40, use_cache=False,
                       log=None)
    assert all(e["states_sha256"] and e["encode_cost"]["cache_used"] is False for e in evals)
    recs = []
    for ev in evals:
        correct = 0
        score = {"correct": correct, "status": "scored"}
        recs.append(cli._controller_record(ev, "C4_params", "cfg", snap, score))
    summary = R.summarize(recs, list(tasks))
    assert summary["arm"] == "C4_params" and summary["requested"] == 3 and summary["tool_calls"] == 0


# --- the training-throughput repair: BF16 storage, cached step tensors, device collate, length groups, file-backed
def _old_step_tensors(examples):
    """The dense steps as collate built them before the repair: every example at the batch's N and T."""
    N = max(e.items.shape[0] for e in examples)
    T = max(e.n_tokens for e in examples)
    per = [K.step_tensors(e.steps, N, T) for e in examples]
    S = max(p["valid"].shape[0] for p in per)
    out = {}
    for key in per[0]:
        buf = torch.zeros((len(per), S) + tuple(per[0][key].shape[1:]), dtype=per[0][key].dtype)
        for b, p in enumerate(per):
            buf[b, :p[key].shape[0]] = p[key]
        out[key] = buf
    return out


def test_cached_step_tensors_pad_to_the_same_batch(prepared):
    new = K.collate(prepared).steps
    old = _old_step_tensors(prepared)
    assert new.keys() == old.keys() and all(torch.equal(new[k], old[k]) for k in old)


def test_bf16_storage_gives_exactly_the_float32_losses(prepared):
    model = models()["T4"]
    as_bf16 = [K.Prepared(**{**e.__dict__, "tokens": e.tokens.to(torch.bfloat16)}) for e in prepared]
    widened = [K.Prepared(**{**e.__dict__, "tokens": e.tokens.to(torch.bfloat16).float()}) for e in prepared]
    a, b = model.losses(K.collate(as_bf16)), model.losses(K.collate(widened))
    assert K.collate(as_bf16).tokens.dtype == torch.bfloat16
    assert all(torch.equal(a[k], b[k]) for k in ("total", "family", "op", "pointer"))


def test_file_backed_examples_train_exactly_like_in_memory_ones(prepared, backbone, cache_root, tmp_path):
    lazy = []
    for e in prepared:
        key = e.meta["cost"]["cache_key"]
        lazy.append(K.Prepared(**{**e.__dict__, "tokens": None,
                                  "tokens_file": str(E.cache_path(key, cache_root))}))
    assert all(torch.equal(x.token_states(), y.tokens) for x, y in zip(lazy, prepared))
    cfg = T.TrainConfig(arm="T4", updates=3, effective_batch=2, micro_batch=1, checkpoint_every=0)
    a = T.fit(T.build_model("T4", 16, 100), prepared, cfg, tmp_path / "memory", log=None)
    b = T.fit(T.build_model("T4", 16, 100), lazy, cfg, tmp_path / "file", log=None)
    for key in ("model_digest", "optimizer_digest"):
        assert a["checkpoints"][-1][key] == b["checkpoints"][-1][key]


def test_micro_batches_group_by_length_and_keep_the_update(prepared):
    groups = T.micro_batches(prepared, [2, 0, 1], 2)
    assert sorted(i for g in groups for i in g) == [0, 1, 2]
    lengths = [prepared[i].n_tokens for g in groups for i in g]
    assert lengths == sorted(lengths)


def test_the_prefetcher_reads_the_files_of_the_update(prepared, cache_root):
    lazy = [K.Prepared(**{**e.__dict__, "tokens": None,
                          "tokens_file": str(E.cache_path(e.meta["cost"]["cache_key"], cache_root))})
            for e in prepared]
    cfg = T.TrainConfig(arm="T4", updates=2, effective_batch=2, micro_batch=1)
    pf = T.Prefetcher(lazy, cfg, 0)
    try:
        for update in range(2):
            got = pf.get(update)
            wanted = {lazy[i].task_id for i in T.batch_indices(len(lazy), update, cfg)}
            assert set(got) == wanted
    finally:
        pf.close()
    assert T.Prefetcher(prepared, cfg, 0).get(0) == {}  # in-memory examples need no reads
