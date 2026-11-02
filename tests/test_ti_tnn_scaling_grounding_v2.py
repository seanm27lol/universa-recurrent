"""Relational grounding (grounding v2) on the CPU, with a tiny random backbone and a synthetic keyed schema.

- **The pilot's problematic joins.** A comparison in ON between one column reference and itself (`T1.x = T1.x`,
  and the other five comparison operators) is illegal under v2 and still legal under v1; the same predicate in WHERE
  stays legal.
- **Legitimate joins are preserved.** Foreign-key joins in either order, self-joins on the key and on a shared name
  (different occurrences), same-source comparisons beside a real key, undeclared-key joins, non-equality joins,
  LEFT JOIN with and without ON, and multi-key ON all encode under v2 to exactly v1's actions and decode to the same
  plan.
- **Relations.** The joined-table pointer marks the tables a declared key links to the plan's sources; the second
  operand of a join comparison marks the first operand's key partner and namesakes and the joined source's columns.
- **Arms.** Under v2, T4 and C4_params have identical trunks and targets; swapping the semantic module reproduces the
  other arm's losses exactly; T4 runs only the selected specialists; batched and single paths agree.
- **Accounting.** The trunk's added parameters and once-per-request MACs equal their closed forms; C4_params still
  matches T4's total parameters within 1%; the pointer MACs include the relation term.
- **v1 is unchanged:** a v1 model has none of the v2 modules, and v1 and v2 trunks share every common initialization
  under one seed.
"""
from __future__ import annotations

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
from scaling import grounding as G  # noqa: E402
from scaling import infer_controller as I  # noqa: E402
from scaling import ir  # noqa: E402
from scaling import schema as schema_mod  # noqa: E402
from scaling import sql_to_ir as X  # noqa: E402
from scaling import train as T  # noqa: E402

COLUMNS = {"movies": [("movie_id", "INTEGER", True), ("title", "TEXT", False), ("director", "TEXT", False),
                      ("year", "INT", False)],
           "ratings": [("rating_id", "INTEGER", True), ("movie_id", "INT", False), ("score", "INT", False),
                       ("critic", "TEXT", False)],
           "people": [("person_id", "INTEGER", True), ("name", "TEXT", False), ("mentor_id", "INT", False)],
           "critics": [("critic", "TEXT", True), ("outlet", "TEXT", False)]}
KEYS = {"ratings": (("movie_id", "movies", "movie_id"),), "people": (("mentor_id", "people", "person_id"),)}
SCHEMA = C.Schema(db_id="cinema", snapshot_sha256="0" * 64, tables=tuple(
    C.SchemaTable(name=t, columns=tuple(C.SchemaColumn(table=t, name=n, type=ty, primary_key=pk,
                                                       description=f"the {n}") for n, ty, pk in cols),
                  foreign_keys=KEYS.get(t, ())) for t, cols in COLUMNS.items()))
REQUESTS = [
    ("g1", "Who directed the movie Alpha?", "Alpha refers to title = 'Alpha';",
     "SELECT director FROM movies WHERE title = 'Alpha'"),
    ("g2", "Which movie has the highest average score?", "",
     "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
     "GROUP BY T1.title ORDER BY AVG(T2.score) DESC LIMIT 1"),
    ("g3", "Who mentored Jo?", "Jo refers to name = 'Jo';",
     "SELECT T2.name FROM people AS T1 INNER JOIN people AS T2 ON T1.mentor_id = T2.person_id WHERE T1.name = 'Jo'"),
]
EXAMPLE = C.RetrievedExample(task_id="x0", db_id="other", question="How many rows?", evidence="",
                             sql="SELECT COUNT(*) FROM t", score=1.0)
LEGITIMATE = [
    "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id",
    "SELECT T2.title FROM ratings AS T1 INNER JOIN movies AS T2 ON T2.movie_id = T1.movie_id",
    "SELECT T2.name FROM people AS T1 INNER JOIN people AS T2 ON T1.mentor_id = T2.person_id",   # self-join on the key
    "SELECT T1.name FROM people AS T1 INNER JOIN people AS T2 ON T1.name = T2.name",             # same column, two occurrences
    "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id AND T2.score = T2.rating_id",
    "SELECT T2.outlet FROM ratings AS T1 INNER JOIN critics AS T2 ON T1.critic = T2.critic",     # undeclared key
    "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.year > T2.score",          # non-equality
    "SELECT T1.title FROM movies AS T1 LEFT JOIN ratings AS T2 ON T1.movie_id = T2.movie_id",
    "SELECT T1.title FROM movies AS T1 LEFT JOIN ratings AS T2",
    "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id AND T1.year = T2.score",
    "SELECT T1.title FROM movies AS T1 WHERE T1.title = T1.title",                               # not a join predicate
    "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
    "INNER JOIN critics AS T3 ON T2.critic = T3.critic WHERE T2.score = 5",
]


def _enc(backbone, cache_root, task_id, question, evidence):
    task = C.Task(task_id=task_id, db_id="cinema", question=question, evidence=evidence, split="train")
    return E.encode_request(backbone, task, SCHEMA, [EXAMPLE], cache_root=cache_root)


@pytest.fixture(scope="module")
def backbone():
    return E.Backbone.tiny_random(E.MockTokenizer(), hidden_size=16, layers=2, seed=0)


@pytest.fixture(scope="module")
def encodings(backbone, tmp_path_factory):
    root = tmp_path_factory.mktemp("encodings")
    return [_enc(backbone, root, t, q, e) for t, q, e, _sql in REQUESTS]


@pytest.fixture(scope="module")
def prepared(encodings):
    out = {}
    for g in A.GROUNDINGS:
        out[g] = [K.prepare(enc, A.trace(X.convert(sql, enc.schema.index), enc.schema, enc.spans, g), grounding=g)
                  for enc, (_t, _q, _e, sql) in zip(encodings, REQUESTS)]
    return out


def _items():
    return A.SchemaItems(schema_mod.serialize(SCHEMA).items, SCHEMA)


def _col(items, table, column):
    return items.column_item[(table, column)]


def _replay_until_second_operand(items, grounding, table_a="ratings", table_b="ratings"):
    """FROM <table_a> T1 JOIN <table_b> T2, END, EQ, COL0(T1.movie_id): the machine at the ON's second operand."""
    m = A.Machine(items, None, grounding=grounding)
    for a in (A.Action("QUERY_SELECT"), A.Action("FROM_TABLE", items.table_item[table_a]),
              A.Action("JOIN_INNER_TABLE", items.table_item[table_b]), A.Action("END"), A.Action("EQ"),
              A.Action("COL0", _col(items, table_a, "movie_id"))):
        m.apply(a)
    return m


# --- the constraint -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("op", sorted(A.COMPARE_ACTIONS))
def test_a_same_reference_join_comparison_is_illegal_in_v2_only(op):
    items = _items()
    for grounding, legal in (("v1", True), ("v2", False)):
        m = A.Machine(items, None, grounding=grounding)
        for a in (A.Action("QUERY_SELECT"), A.Action("FROM_TABLE", items.table_item["ratings"]),
                  A.Action("JOIN_INNER_TABLE", items.table_item["movies"]), A.Action("END"), A.Action(op),
                  A.Action("COL0", _col(items, "ratings", "movie_id"))):
            m.apply(a)
        same = A.Action("COL0", _col(items, "ratings", "movie_id"))
        assert m.is_legal(same) is legal
        assert m.is_legal(A.Action("COL0", _col(items, "movies", "movie_id")))  # the real join stays legal
        assert m.is_legal(A.Action("COL0", _col(items, "ratings", "score")))    # another column of the same source


def test_the_pilot_self_join_tautology_is_rejected_but_the_self_join_is_not():
    """`violations T1 JOIN violations T2 ON T1.business_id = T1.business_id`, as seed 101's T4 wrote it, here on
    ratings: v2 refuses the second COL0 and offers COL1 (the other occurrence) for the same column."""
    items = _items()
    m = _replay_until_second_operand(items, "v2")
    rid = _col(items, "ratings", "movie_id")
    assert not m.is_legal(A.Action("COL0", rid))
    assert m.is_legal(A.Action("COL1", rid))                                    # T1.movie_id = T2.movie_id
    with pytest.raises(A.IllegalAction):
        m.apply(A.Action("COL0", rid))
    v1 = _replay_until_second_operand(items, "v1")
    v1.apply(A.Action("COL0", rid))                                             # v1 accepted it (the pilot's grammar)


def test_the_same_comparison_in_where_stays_legal():
    items = _items()
    m = A.Machine(items, None, grounding="v2")
    for a in (A.Action("QUERY_SELECT"), A.Action("FROM_TABLE", items.table_item["movies"]), A.Action("END"),
              A.Action("EQ"), A.Action("COL0", _col(items, "movies", "title"))):
        m.apply(a)
    assert m.is_legal(A.Action("COL0", _col(items, "movies", "title")))


@pytest.mark.parametrize("sql", LEGITIMATE)
def test_legitimate_joins_encode_identically_under_both_groundings(sql, encodings):
    enc = encodings[0]
    plan = X.convert(sql, enc.schema.index)
    v1 = A.encode(plan, enc.schema, enc.spans, "v1")
    v2 = A.encode(plan, enc.schema, enc.spans, "v2")
    assert v1 == v2
    assert A.decode(v2, enc.schema, enc.spans, "v2") == plan
    assert [s.action for s in A.trace(plan, enc.schema, enc.spans, "v2")[:-1]] == v1


def test_a_gold_plan_with_a_same_reference_join_is_not_representable_in_v2(encodings):
    """The one historical case (training question bird23tf-00003) has this shape: a real key and `T2.x = T2.x`."""
    enc = encodings[0]
    plan = X.convert("SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
                     "AND T2.score = T2.score", enc.schema.index)
    assert A.encode(plan, enc.schema, enc.spans, "v1")
    with pytest.raises(A.NotEncodable):
        A.encode(plan, enc.schema, enc.spans, "v2")


# --- relations ------------------------------------------------------------------------------------------------------------
def test_schema_keys_and_item_relations():
    items = _items()
    keys = items.keys
    rm, mm = _col(items, "ratings", "movie_id"), _col(items, "movies", "movie_id")
    assert keys.fk_out[rm] == {mm} and keys.fk_in[mm] == {rm} and keys.partners(mm) == {rm}
    assert mm in keys.pk and _col(items, "people", "person_id") in keys.pk
    assert keys.table_links["ratings"] == {"movies"} and "people" not in keys.table_links  # a self-link is no link
    rel = G.item_relations(items)
    name = {r: i for i, r in enumerate(G.RELATIONS)}
    t_movies, t_ratings = items.table_item["movies"], items.table_item["ratings"]
    assert rel[rm][mm] == name["column references column"] and rel[mm][rm] == name["column referenced by column"]
    assert rel[t_movies][mm] == name["table->own column"] and rel[mm][t_movies] == name["column->own table"]
    assert rel[t_movies][t_ratings] == name["table links to table"]
    assert rel[t_movies][rm] == name["table->column of a linked table"]
    crit_r, crit_c = _col(items, "ratings", "critic"), _col(items, "critics", "critic")
    assert rel[crit_r][crit_c] == name["column shares a name with column"]  # undeclared, by name only
    assert all(rel[i][i] == name["self"] for i in range(len(items.items)))
    feats = G.item_features(items, "Which movie has the highest score?", "score refers to ratings.score")
    f = {n: k for k, n in enumerate(G.ITEM_FEATURES)}
    assert feats[_col(items, "ratings", "score")][f["name in the evidence"]] == 1.0
    assert feats[mm][f["primary key"]] == 1.0 and feats[rm][f["foreign key (references)"]] == 1.0


def test_the_joined_table_pointer_marks_linked_tables():
    items = _items()
    m = A.Machine(items, None, grounding="v2")
    m.apply(A.Action("QUERY_SELECT"))
    m.apply(A.Action("FROM_TABLE", items.table_item["ratings"]))
    rel = m.pointer_relations("JOIN_INNER_TABLE")
    bit = {r: 1 << k for k, r in enumerate(A.POINTER_RELATIONS)}
    assert rel[items.table_item["movies"]] & bit["table linked by a declared foreign key to a source of this SELECT"]
    assert rel[items.table_item["ratings"]] & bit["table already a source of this SELECT"]
    assert rel[items.table_item["critics"]] & bit["table shares a column name with a source of this SELECT "
                                                  "(no declared key)"]
    assert items.table_item["people"] not in rel
    assert A.Machine(items, None, grounding="v1").pointer_relations("FROM_TABLE") == {}


def test_the_second_operand_is_scored_against_the_first():
    items = _items()
    m = _replay_until_second_operand(items, "v2", "ratings", "movies")
    rel = m.pointer_relations("COL0")
    bit = {r: 1 << k for k, r in enumerate(A.POINTER_RELATIONS)}
    mm = _col(items, "movies", "movie_id")
    assert rel[mm] & bit["column is a declared foreign-key partner of the comparison's first operand"]
    assert rel[mm] & bit["column has the first operand's name, on another table"]
    assert rel[mm] & bit["column belongs to the source this ON clause joins"]
    assert rel[_col(items, "movies", "title")] == bit["column belongs to the source this ON clause joins"]
    assert _col(items, "ratings", "movie_id") not in m.options()["COL0"][1]     # excluded, so never scored


def test_gold_steps_carry_their_relations(prepared):
    steps = [s for s in prepared["v2"][1].steps if s.pointer in ("table", "column")]
    assert all(isinstance(s.relations, dict) for s in steps)
    join_table = next(s for s in steps if s.op == "JOIN_INNER_TABLE")
    assert join_table.relations.get(join_table.target, 0) & 2                   # the gold joined table is key-linked
    dense = prepared["v2"][1].steps_dense()
    assert dense["ptr_rel"].shape == (len(prepared["v2"][1].steps), prepared["v2"][1].items.shape[0],
                                      A.N_POINTER_RELATIONS)
    assert "ptr_rel" not in prepared["v1"][1].steps_dense()


# --- the arms under v2 ---------------------------------------------------------------------------------------------------
def models(grounding="v2", seed=100):
    return {arm: T.build_model(arm, 16, seed, grounding) for arm in K.ARMS}


def test_v1_and_v2_trunks_share_their_common_initialization_and_v1_has_no_v2_modules():
    v1, v2 = T.build_model("T4", 16, 100, "v1"), T.build_model("T4", 16, 100, "v2")
    a, b = v1.trunk.state_dict(), v2.trunk.state_dict()
    assert set(a) < set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.startswith(("schema_encoder", "item_features", "pointer_relation")) for k in a)
    assert all(torch.equal(x, y) for x, y in zip(v1.semantic.state_dict().values(), v2.semantic.state_dict().values()))


def test_v2_supervision_is_identical_across_arms(prepared):
    batch = K.collate(prepared["v2"])
    ms = models()
    trunk = ms["T4"].trunk.state_dict()
    for arm in ("c4_active", "c4_params"):
        other = ms[arm].trunk.state_dict()
        assert trunk.keys() == other.keys() and all(torch.equal(trunk[k], other[k]) for k in trunk)
    out = {arm: m.losses(batch) for arm, m in ms.items()}
    for arm in K.ARMS:
        assert (out[arm]["steps"], out[arm]["op_steps"], out[arm]["pointer_steps"]) == \
            (out["T4"]["steps"], out["T4"]["op_steps"], out["T4"]["pointer_steps"])
        assert torch.equal(out[arm]["family"], out["T4"]["family"])
    swapped = ms["T4"]
    swapped.semantic = ms["c4_params"].semantic
    again = swapped.losses(batch)
    for key in ("total", "family", "op", "pointer"):
        assert torch.equal(again[key], out["c4_params"][key])


def test_v2_t4_runs_only_the_selected_heads(prepared):
    model = models()["T4"]
    seen = []
    for f, head in enumerate(model.semantic.heads):
        head.register_forward_hook(lambda mod, inp, out, f=f: seen.append((f, inp[0].shape[0])))
    batch = K.collate(prepared["v2"])
    model.losses(batch)
    st = batch.steps
    want = {f: int((st["valid"] & (st["family"] == f)).sum()) for f in range(4)}
    assert sorted(seen) == sorted((f, n) for f, n in want.items() if n)
    seen.clear()
    decoded = I.decode_batch(model, prepared["v2"], max_steps=40)
    rows = {}
    for f, n in seen:
        rows[A.FAMILIES[f]] = rows.get(A.FAMILIES[f], 0) + n
    heads = {}
    for d in decoded:
        assert d.work["grounding"] == "v2"
        for h, fam in zip(d.work["heads"], d.work["families"]):
            assert h == (None if fam == "STOP" else fam)
            if h is not None:
                heads[h] = heads.get(h, 0) + 1
    assert rows == heads


def test_v2_batched_and_single_paths_agree(prepared):
    for arm, model in models().items():
        batch = model.losses(K.collate(prepared["v2"]))
        singles = [model.losses(K.collate([p])) for p in prepared["v2"]]
        for key in ("total", "family", "op", "pointer"):
            assert torch.allclose(batch[key], sum(s[key] for s in singles), rtol=1e-5, atol=1e-4), (arm, key)
        together = I.decode_batch(model, prepared["v2"], max_steps=40)
        alone = [I.decode_one(model, p, max_steps=40) for p in prepared["v2"]]
        for a, b in zip(together, alone):
            assert (a.actions, a.sql, a.abstain_reason) == (b.actions, b.sql, b.abstain_reason)


def test_v2_decoding_never_emits_a_same_reference_join(prepared):
    for seed in range(4):  # random controllers explore the legal space
        for model in models(seed=seed).values():
            for d in I.decode_batch(model, prepared["v2"], max_steps=60):
                plan = d.plan
                if plan is None:
                    continue
                for node in ir.walk(plan):
                    if isinstance(node, ir.FromItem) and node.on is not None:
                        for c in ir.walk(node.on):
                            assert not (isinstance(c, ir.Compare) and c.left == c.right and
                                        isinstance(c.left, ir.Column))


def test_the_relation_term_moves_only_the_marked_items(prepared):
    model = models()["T4"]
    batch = K.collate(prepared["v2"][1:2])
    mem = model.trunk.memory(batch)
    q = torch.randn(1, 1, K.N_POINTERS, 512)
    rel = torch.zeros(1, 1, mem["items"].shape[1], A.N_POINTER_RELATIONS, dtype=torch.bool)
    rel[0, 0, 3, 1] = True
    base, _, _ = model.trunk.pointer_scores(mem, q, relations=rel)
    assert torch.equal(base, model.trunk.pointer_scores(mem, q)[0])            # zero-initialized: no effect yet
    with torch.no_grad():
        model.trunk.pointer_relation.normal_()
    moved, _, _ = model.trunk.pointer_scores(mem, q, relations=rel)
    diff = (moved - base).abs().sum(dim=(0, 1, 2))
    assert diff[3] > 0 and diff.sum() == diff[3]


# --- accounting ---------------------------------------------------------------------------------------------------------
def _encoder_parameters(cfg):
    d, ff, h = cfg.d_model, cfg.encoder_ff, cfg.attention_heads
    layer = 4 * (d * d + d) + G.N_RELATIONS * h + 2 * 2 * d + (d * ff + ff + ff * d + d)
    return G.N_ITEM_FEATURES * d + d + cfg.encoder_layers * layer + 2 * d + 2 * A.N_POINTER_RELATIONS * d


def test_v2_parameter_and_compute_accounting():
    reports = {(arm, g): K.parameter_report(T.build_model(arm, 2560, 100, g)) for arm in K.ARMS for g in A.GROUNDINGS}
    cfg2 = K.matched_config("T4", 2560, grounding="v2")
    added = _encoder_parameters(cfg2)
    for arm in K.ARMS:
        assert reports[(arm, "v2")]["trunk"] - reports[(arm, "v1")]["trunk"] == added
        assert reports[(arm, "v2")]["semantic"] == reports[(arm, "v1")]["semantic"]
    for g in A.GROUNDINGS:
        t4, c4p = reports[("T4", g)], reports[("c4_params", g)]
        assert t4["trunk"] == c4p["trunk"] and abs(c4p["total"] - t4["total"]) / t4["total"] < 0.01
    assert K.matched_config("c4_params", 2560, grounding="v2").c4_params_hidden == \
        K.matched_config("c4_params", 2560).c4_params_hidden
    cfg1 = K.matched_config("T4", 2560)
    n_tok, n_items, n_qe = 3000, 60, 40
    d, ff = cfg2.d_model, cfg2.encoder_ff
    keys = n_items + n_qe
    layer = n_items * d * d + 2 * keys * d * d + 2 * n_items * keys * d + n_items * d * d + 2 * n_items * d * ff
    extra = n_items * G.N_ITEM_FEATURES * d + cfg2.encoder_layers * layer
    assert K.request_macs(cfg2, n_tok, n_items, n_qe) - K.request_macs(cfg1, n_tok, n_items) == extra
    for pointer in ("table", "column"):
        a, b = K.step_macs(cfg1, 0, n_tok, n_items, pointer), K.step_macs(cfg2, 0, n_tok, n_items, pointer)
        assert b["pointer"] - a["pointer"] == A.N_POINTER_RELATIONS * d + n_items * A.N_POINTER_RELATIONS
        assert a["trunk"] == b["trunk"] and a["semantic"] == b["semantic"]
    assert K.step_macs(cfg1, 0, n_tok, n_items, "span") == K.step_macs(cfg2, 0, n_tok, n_items, "span")


def test_v2_checkpoints_resume_bit_for_bit(prepared, tmp_path):
    cfg = T.TrainConfig(arm="c4_params", updates=3, effective_batch=2, micro_batch=1, checkpoint_every=1)
    full = T.build_model("c4_params", 16, 100, "v2")
    run = T.fit(full, prepared["v2"], cfg, tmp_path / "full", log=None)
    resumed = T.build_model("c4_params", 16, 7, "v2")
    T.fit(resumed, prepared["v2"], cfg, tmp_path / "resumed", resume_from=run["checkpoints"][0]["path"], log=None)
    final = full.state_dict()
    assert all(torch.equal(final[k], v) for k, v in resumed.state_dict().items())
    reloaded = T.model_from_checkpoint(T.load_checkpoint(run["checkpoints"][-1]["path"]))
    assert reloaded.cfg.grounding == "v2"
    with pytest.raises(ValueError, match="grounding"):
        T.fit(T.build_model("c4_params", 16, 100, "v1"), prepared["v2"],
              T.TrainConfig(arm="c4_params", updates=1, effective_batch=2, micro_batch=1), tmp_path / "mixed",
              log=None)
