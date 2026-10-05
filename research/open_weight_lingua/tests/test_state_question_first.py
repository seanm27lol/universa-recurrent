import json

import numpy as np

from open_weight_lingua import state_question_first as sq

ORIGINAL = "x = 9\ny = 8\ny = 3\ny = y + 3\nx = x + 1\nWhat is y? Reply with only the integer."
PROGRAM = "x = 9\ny = 8\ny = 3\ny = y + 3\nx = x + 1"


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_question_first_prompt_keeps_the_program_answer_and_split():
    manifest = {"inputs": [
        {"id": "calibration-0003-A-y", "group_id": "calibration-0003", "side": "A", "variable": "y",
         "prompt": ORIGINAL, "answer": "6", "input_ids": [1, 2]},
        {"id": "calibration-0200-B-x", "group_id": "calibration-0200", "side": "B", "variable": "x",
         "prompt": ORIGINAL.replace("What is y?", "What is x?"), "answer": "10", "input_ids": [1, 2]},
    ]}
    rows = sq.prompt_rows(manifest, "calibration")
    assert rows[0]["prompt"] == "What is y at the end of this program? Reply with only the integer.\n" + PROGRAM
    assert rows[0]["answer"] == "6" and rows[0]["original_prompt"] == ORIGINAL
    assert [r["part"] for r in rows] == ["train", "select"]
    assert sq.prompt_rows(manifest, "pilot")[0]["part"] == "test"


def test_boundaries_land_after_the_question_on_each_statement_from_line_two():
    prompt = sq.QUESTION_TEMPLATE.format(variable="y", program=PROGRAM)
    text = "<u>" + prompt + "<m>\n"
    ids = [ord(c) for c in text]
    found = sq.question_first_boundaries(CharTokenizer(), ids, prompt)
    assert [text[: p + 1].split("\n")[-1] for p in found] == ["y = 8", "y = 3", "y = y + 3", "x = x + 1"]


def _paired_records(groups=120, seed=0):
    """Each program appears twice, once per question, with identical labels; only `asked` differs."""
    rng = np.random.default_rng(seed)
    categories = ("literal", "arithmetic", "copy", "carried")
    records = []
    for g in range(groups):
        part = "train" if g < groups // 2 else "select" if g < 3 * groups // 4 else "test"
        lines = [{"line": i, **{v: {"value": int(rng.integers(0, 20)), "category": categories[(i + k + g) % 4],
                                    "computed": bool(rng.integers(0, 2)), "copy_guess": 0}
                                for k, v in enumerate(("x", "y"))}} for i in range(2, 6)]
        for asked in ("x", "y"):
            for label in lines:
                records.append({"id": f"g{g}-{asked}", "group_id": f"g{g}", "side": "A", "asked": asked,
                                "part": part, "position": label["line"], **label})
    return records


def _asked_code(records, d=32, seed=1):
    """Layer 1 codes only the asked variable's value; layer 0 is noise."""
    rng = np.random.default_rng(seed)
    code = rng.normal(size=(20, d))
    features = 0.05 * rng.normal(size=(len(records), 2, d))
    features[:, 1] += code[[r[r["asked"]]["value"] for r in records]]
    return features


def test_an_asked_only_code_passes_the_readings_and_shows_a_question_effect():
    records = _paired_records()
    features = _asked_code(records)
    grid, held_out = sq.survey(features, records, "cpu")
    gated = sq.gate_readings(features, records, grid, "cpu")
    assert gated["R1"]["layer"] == gated["R2"]["layer"] == 1
    assert gated["R1"]["supported"] and gated["R2"]["supported"]
    assert grid[1]["not_asked"]["x"]["test"]["overall"] < 0.3  # the other variable is not coded
    effect = sq.question_effect(records, held_out, 1, "arithmetic")
    assert effect["pairs"] > 0 and effect["ci95"][0] > 0.5
    assert effect["asked_accuracy"] > 0.9


def test_no_question_effect_when_both_conditions_read_alike():
    records = _paired_records()
    rng = np.random.default_rng(2)
    code = rng.normal(size=(20, 32))
    features = 0.05 * rng.normal(size=(len(records), 2, 32))
    features[:, 1, :16] += code[[r["x"]["value"] for r in records], :16]  # x coded whatever is asked
    _, held_out = sq.survey(features, records, "cpu")
    effect = sq.question_effect(records, held_out, 1, "arithmetic")
    assert effect["pairs"] > 0 and abs(effect["difference"]) < 0.1
    assert not effect["ci95"][0] > 0  # E6 would not hold


def test_permuted_control_disqualifies_a_reading(monkeypatch):
    records = _paired_records()
    features = _asked_code(records)
    grid, _ = sq.survey(features, records, "cpu")
    monkeypatch.setattr(sq, "CONTROL_MAX", -1.0)
    gated = sq.gate_readings(features, records, grid, "cpu")
    assert gated["R2"]["probe_supported"] is True and gated["R2"]["supported"] is False


def test_phase_six_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model, fixture tokens; the plumbing only, never a real-model result."""
    from test_pilot_stages import TINY_COUNTS, _inputs, _stub_load_model
    import transformers

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": sq.sha256_file(lock)}))
    monkeypatch.setattr(sq, "TRAIN_GROUPS", 2)
    monkeypatch.setattr(sq, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(sq, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(sq, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(sq, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(sq, "compatibility", lambda device: {})
    monkeypatch.setattr(sq, "question_first_boundaries",
                        lambda tok, ids, prompt: [min(i, len(ids) - 1) for i in range(len(prompt.split("\n")) - 2)])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: tokenizer)
    args = sq.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "question-first")
    results, summary = sq.run(args, run, {})
    assert summary["prompts"] == {"train": 8, "select": 4, "test": 16}
    assert len(results["generations"]) == 28 and len(results["boundaries"]) == summary["layers"]
    assert set(summary) >= {"gate_U6", "R1", "R2", "gate_G6", "effect_E6", "question_effects", "final_state"}
    assert summary["gate_G6"]["void"] == (not summary["gate_U6"]["passed"])
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["answer_convention"] == "rstrip" and manifest["boundaries"] == len(manifest["records"])
    assert (run.path / "raw" / "features.safetensors").is_file()
