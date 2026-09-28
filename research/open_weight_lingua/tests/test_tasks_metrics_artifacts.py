from dataclasses import replace
import json
from pathlib import Path
import pickle
import pytest
import torch
from open_weight_lingua.artifacts import (
    RunDirectory,
    load_numeric,
    save_numeric,
    write_json,
)
from open_weight_lingua.metrics import (
    answer_tokens,
    exact_integer,
    next_token_kl,
    sequence_log_probability,
)
from open_weight_lingua.preflight import read_lock, verify_models
from open_weight_lingua.tasks import (
    Statement,
    canonical,
    generate_groups,
    interpret,
    tokenize_groups,
)


def test_explicit_interpreter_and_pair_invariants():
    program = (
        Statement("x", "assign", 3),
        Statement("y", "assign", 8),
        Statement("x", "add", 2),
        Statement("y", "copy", "x"),
        Statement("y", "subtract", 1),
    )
    assert interpret(program) == {"x": 5, "y": 4}
    for bad in (
        (Statement("x", "run", "print(3)"),),
        (Statement("x", "add", 1),),
        (Statement("x", "assign", 20),),
    ):
        with pytest.raises(ValueError):
            interpret(bad)
    groups, stats = generate_groups("smoke", 8)
    assert groups == generate_groups("smoke", 8)[0]
    assert stats["groups"] == 8 and stats["prompt_variants"] == 32
    assert sum(g.affected == "x" for g in groups) == 4
    for group in groups:
        group.validate()
        assert 3 <= len(group.source) <= 6
        assert len(group.variants()) == 4
    with pytest.raises(ValueError):
        replace(groups[0], counterfactual=groups[0].source).validate()


def test_split_exclusions_without_opening_validation():
    smoke, _ = generate_groups("smoke", 8)
    seen = {
        canonical(program)
        for group in smoke
        for program in (group.source, group.counterfactual)
    }
    calibration, _ = generate_groups("calibration", 32, excluded_programs=seen)
    new = {
        canonical(program)
        for group in calibration
        for program in (group.source, group.counterfactual)
    }
    assert not seen & new


def test_input_only_alignment_and_duplicate_rejection(tokenizer):
    groups, _ = generate_groups("smoke", 8)
    with pytest.raises(ValueError, match="duplicated"):
        tokenize_groups(
            groups, tokenizer
        )  # fixture deliberately returns identical prompts

    class VariedTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [ord(c) for c in messages[0]["content"]] + [999]

        def decode(self, ids):
            return "prefix-boundary"

    rows = tokenize_groups(groups, VariedTokenizer())
    assert len(rows) == 32
    assert all(row["input_ids"][row["position"]] == 999 for row in rows)
    with pytest.raises(ValueError, match="duplicated"):
        tokenize_groups(
            groups, VariedTokenizer(), excluded_tokenized_prompts=[rows[0]["input_ids"]]
        )


def test_multi_token_scoring_includes_eos(tokenizer):
    ids = answer_tokens(tokenizer, "19")
    assert ids == [11, 19, 2]
    logits = torch.zeros((1, 6, 32))
    value = sequence_log_probability(logits, 3, ids)
    assert value == pytest.approx(-3 * torch.log(torch.tensor(32.0)).item())
    for text in ("19 ", "019", "the answer is 19", "19\n"):
        assert not exact_integer(text, "19")
    assert exact_integer("19", "19")
    assert next_token_kl(logits[0, 0], logits[0, 0]) == 0
    with pytest.raises(ValueError, match="finite"):
        next_token_kl(torch.tensor([float("nan")]), torch.ones(1))


def test_safe_numeric_and_overwrite_refusal(tmp_path):
    path = tmp_path / "vector.safetensors"
    original = torch.arange(8, dtype=torch.bfloat16)
    save_numeric(path, {"original": original})
    loaded = load_numeric(path)
    assert (
        torch.equal(loaded["original"], original)
        and loaded["original"].dtype == original.dtype
    )
    with pytest.raises(FileExistsError):
        save_numeric(path, {"original": original})
    with pytest.raises(ValueError, match="oversized"):
        load_numeric(path, max_bytes=2)
    unsafe = tmp_path / "unsafe.npy"
    unsafe.write_bytes(pickle.dumps({"not": "safe"}))
    with pytest.raises(ValueError, match="forbidden"):
        load_numeric(unsafe)
    disguised = tmp_path / "unsafe.safetensors"
    disguised.write_bytes(unsafe.read_bytes())
    with pytest.raises(Exception):
        load_numeric(disguised)
    run = RunDirectory(tmp_path / "run")
    with pytest.raises(FileExistsError):
        RunDirectory(run.path)
    write_json(run.path / "manifest.json", {"frozen": True})
    with pytest.raises(FileExistsError):
        write_json(run.path / "manifest.json", {})


def test_source_lock_immutable_and_local_hash_check(tmp_path):
    lock_path = Path(__file__).parents[1] / "configs/model-lock.json"
    lock = read_lock(lock_path)
    assert (
        lock["models"]["ar"]["revision"] == "e2c9e57eac213d37a31612087f645ab6332c1bb6"
    )
    lock["models"]["ar"]["revision"] = "main"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="immutable"):
        read_lock(path)
    fake_lock = {
        "models": {"ar": {"files": {"config.json": {"bytes": 2, "sha256": "0" * 64}}}}
    }
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        verify_models(fake_lock, {"ar": tmp_path})


def test_metadata_fixtures_match_source_lock():
    root = Path(__file__).parents[1]
    lock = read_lock(root / "configs/model-lock.json")
    from open_weight_lingua.artifacts import sha256_file

    for role in ("target", "av", "ar"):
        for name in ("config.json", "nla_meta.yaml"):
            path = root / "tests/fixtures/upstream" / f"{role}-{name}"
            if path.exists():
                assert (
                    sha256_file(path) == lock["models"][role]["files"][name]["sha256"]
                )
