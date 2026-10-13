import json

import torch

from open_weight_lingua import marking_layers_knockout as mlk


def test_conditions_cover_the_grid_from_both_sides():
    names = mlk.conditions(48)
    assert names["program_blind"] == set(range(48)) and names["baseline"] == set()
    assert names["from_24"] == set(range(24, 48)) and names["until_24"] == set(range(24))
    assert len(names) == 2 + 2 * 7 and "from_0" not in names and "until_0" not in names


def test_block_layers_matches_knockout_and_is_layer_selective(model_factory):
    from test_answer_knockout import _eager

    from open_weight_lingua.answer_knockout import knockout

    model = _eager(model_factory(layers=4))
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])

    def logits(context):
        with torch.no_grad(), context:
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits

    base = logits(mlk.block_layers(model, set(), [6, 7], [1, 2]))
    late = logits(mlk.block_layers(model, {2, 3}, [6, 7], [1, 2]))
    assert torch.equal(late, logits(knockout(model, 2, [6, 7], [1, 2])))
    assert torch.equal(base[0, :6], late[0, :6]) and not torch.equal(base[0, 6:], late[0, 6:])
    early = logits(mlk.block_layers(model, {0, 1}, [6, 7], [1, 2]))
    assert not torch.equal(early, late)


def test_eligibility_keeps_only_prompts_where_the_mark_decides():
    base = {"last_line_assigns_asked": False, "depends_on_other": False, "values": {"answer": 1, "other": 2}}
    assert mlk.eligible(base)
    assert not mlk.eligible({**base, "last_line_assigns_asked": True})
    assert not mlk.eligible({**base, "depends_on_other": True})
    assert not mlk.eligible({**base, "values": {"answer": 1}})


def test_window_follows_the_frozen_rule():
    layers = 8  # grid 0..7
    names = list(mlk.conditions(layers))
    rows, results = [], []
    for split in ("calibration", "pilot"):
        rows.append({"split": split})
        said = {c: False for c in names}
        said["program_blind"] = True
        for k in (1, 2, 3):  # blocking from 1, 2 or 3 still reaches the marking layers (2 and 3)
            said[f"from_{k}"] = True
        for k in (3, 4, 5, 6, 7):  # blocking until 3 or later covers layer 2
            said[f"until_{k}"] = True
        results.append({c: {"answer": {"correct": not said[c], "logprob": -1.0}, "other": {"correct": said[c], "logprob": -1.0}}
                        for c in names})
    summary = mlk.summarize(rows, results, layers)
    assert summary["window"]["calibration"] == {"from_release": 4, "until_onset": 2}
    assert summary["all"]["program_blind"]["says_other"] == 1.0


def test_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    import transformers

    from test_answer_knockout import _eager
    from test_pilot_stages import TINY_COUNTS, _inputs

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": mlk.sha256_file(lock)}))
    seq = lambda item, t: {"ids": list(item["input_ids"]) + [t], "positions": [len(item["input_ids"]) - 1], "targets": [t]}  # noqa: E731
    monkeypatch.setattr(mlk, "route_row", lambda item, split, tok: {
        "id": item["id"], "group_id": item["group_id"], "split": split, "question": [0, 1], "span": [2, 3],
        "values": {"answer": 1, "other": 2}, "last_line_assigns_asked": False, "depends_on_other": False,
        "sequences": {"answer": seq(item, 10), "other": seq(item, 11)}})
    monkeypatch.setattr(mlk, "load_model", lambda path, role, device: _eager(model_factory(layers=8)))
    monkeypatch.setattr(mlk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(mlk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(mlk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(mlk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    brief = tmp_path / "a_later_brief.md"
    brief.write_text("a later phase reusing this module")
    args = mlk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu", "--protocol", str(brief)])
    run = RunDirectory(tmp_path / "layers")
    summary = mlk.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    inputs = sum(len(json.loads((tmp_path / n / "manifest.json").read_text())["inputs"]) for n in ("cal", "pil"))
    assert manifest["prompts"] == inputs and summary["layers"] == 8 and set(summary["window"]) == {"calibration", "pilot"}
    assert manifest["protocol_path"] == str(brief) and manifest["protocol_sha256"] == mlk.sha256_file(brief)
    assert mlk.parse_args(["--lock", "l", "--calibration-run", "c", "--pilot-run", "p"]).protocol == mlk.PROTOCOL
    first = json.loads((run.path / "results.jsonl").read_text().splitlines()[0])
    assert set(first) == {"id", *mlk.conditions(8)}
