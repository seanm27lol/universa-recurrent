"""CPU tests for the opt-in vLLM AV/AR backend.

The worker module is importable without vllm (heavy imports stay lazy), so job
validation, prompt/embeds construction, value-head math and the equivalence
comparison run on CPU with fixtures. GPU worker tests skip cleanly when vllm or
CUDA is unavailable.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from open_weight_lingua import runner, vllm_backend, vllm_equivalence, vllm_worker
from open_weight_lingua.nla_adapter import ar_prompt


def _base_job(tmp_path, mode="verbalize"):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model_dir / name).write_text("{}")
    return {
        "schema_version": 1,
        "mode": mode,
        "model_dir": str(model_dir),
        "output_dir": str(tmp_path / "out"),
        "rows": [{"id": "row-0"}],
        "metadata": {"width": 16},
    }


def test_job_validation(tmp_path):
    job = _base_job(tmp_path)
    model_dir, _output_dir, rows, _metadata = vllm_worker.validate_job(job, "verbalize")
    assert model_dir.is_dir() and rows[0]["id"] == "row-0"
    for field, value in [
        ("schema_version", 2),
        ("mode", "reconstruct"),
        ("model_dir", str(tmp_path / "missing")),
        ("output_dir", ""),
        ("rows", []),
    ]:
        broken = {**job, field: value}
        with pytest.raises(ValueError):
            vllm_worker.validate_job(broken, mode="verbalize")
    duplicate = {**job, "rows": [{"id": "row-0"}, {"id": "row-0"}]}
    with pytest.raises(ValueError, match="unique"):
        vllm_worker.validate_job(duplicate, "verbalize")


def test_embeds_prompt_marks_only_the_injected_row():
    ids = [3, 6, 7, 8, 4]
    row = torch.full((16,), 2.0, dtype=torch.bfloat16)
    prompt = vllm_worker.embeds_prompt(ids, 2, row, 16)
    assert prompt["prompt_token_ids"] == ids
    assert prompt["prompt_is_token_ids"] == [True, True, False, True, True]
    embeds = prompt["prompt_embeds"]
    assert embeds.shape == (5, 16) and embeds.dtype == torch.bfloat16
    assert torch.equal(embeds[2], row) and not embeds[:2].any() and not embeds[3:].any()
    with pytest.raises(ValueError, match="range"):
        vllm_worker.embeds_prompt(ids, 5, row, 16)
    with pytest.raises(ValueError, match="width"):
        vllm_worker.embeds_prompt(ids, 2, torch.zeros(8), 16)


def test_scaled_injection_matches_eager_math():
    vector = torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16)
    row = vllm_worker.scaled_injection(vector, 16, 150.0)
    expected = (vector.float() / vector.float().norm() * 150.0).to(torch.bfloat16)
    assert row.dtype == torch.bfloat16 and torch.equal(row, expected)
    assert float(row.float().norm()) == pytest.approx(150, rel=1e-3)
    with pytest.raises(ValueError):
        vllm_worker.scaled_injection(torch.zeros(16), 16, 150.0)
    with pytest.raises(ValueError):
        vllm_worker.scaled_injection(torch.ones(8), 16, 150.0)
    with pytest.raises(ValueError):
        vllm_worker.scaled_injection(torch.ones(16), 16, 0.0)


@pytest.mark.parametrize(
    "raw,tokens,status",
    [
        ("<explanation>A description</explanation><2>", [5, 2], "ok"),
        ("<explanation>missing close", [5, 2], "invalid_explanation_tags"),
        (
            "<explanation><explanation>nested</explanation>",
            [5, 2],
            "invalid_explanation_tags",
        ),
        ("<explanation> </explanation>", [5, 2], "invalid_explanation_tags"),
        ("<explanation>A</explanation>", [5, 5], "truncated"),
    ],
)
def test_parse_description_record_matches_eager_policy(raw, tokens, status):
    record = vllm_worker.parse_description_record(
        raw, tokens, max_new_tokens=2, eos_token_id=2, injection_position=3
    )
    assert record["status"] == status
    assert record["token_ids"] == tokens and record["injection_position"] == 3
    assert (record["description"] is None) == (status != "ok")


def test_ar_prompt_id_checks():
    vllm_worker.validate_ar_prompt_ids([20, 21, 22, 8, 9], (8, 9), None)
    with pytest.raises(ValueError, match="suffix"):
        vllm_worker.validate_ar_prompt_ids([20, 21, 22, 8, 7], (8, 9), None)
    with pytest.raises(ValueError, match="BOS"):
        vllm_worker.validate_ar_prompt_ids([20, 21, 22, 8, 9], (8, 9), 1)


def test_value_head_math_on_fixture_tensors(tmp_path):
    width = 16
    weight = torch.randn(width, width).to(torch.bfloat16)
    path = tmp_path / "value_head.safetensors"
    save_file({"weight": weight}, str(path))
    loaded = vllm_worker.read_value_head(path, width)
    assert torch.equal(loaded, weight)
    hidden = torch.randn(width)
    vector = vllm_worker.value_head_direction(hidden, loaded)
    expected = torch.nn.functional.linear(hidden.to(torch.bfloat16), weight).float()
    assert vector.dtype == torch.float32 and torch.equal(vector, expected)
    with pytest.raises(ValueError):
        vllm_worker.value_head_direction(torch.zeros(width), loaded)
    bad = tmp_path / "bad.safetensors"
    save_file({"weight": torch.eye(width)}, str(bad))
    with pytest.raises(ValueError, match="dtype"):
        vllm_worker.read_value_head(bad, width)
    save_file({"weight": torch.zeros(width, width, dtype=torch.bfloat16)}, str(bad))
    with pytest.raises(ValueError):
        vllm_worker.value_head_direction(torch.ones(width), torch.zeros(width, width))


def test_worker_module_never_imports_pipeline_or_transformers():
    import ast

    tree = ast.parse(Path(vllm_worker.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"open_weight_lingua", "transformers", "yaml"})
    assert "vllm" in imported  # lazily, inside the mode functions


def test_equivalence_compare_av_logic():
    eager = {
        "a": {"status": "ok", "token_ids": [1, 2, 3]},
        "b": {"status": "ok", "token_ids": [4, 5, 6]},
    }
    identical = vllm_equivalence.compare_av(
        eager, {k: dict(v) for k, v in eager.items()}
    )
    assert identical["token_identity_rate"] == 1.0 and identical["bar_met"]
    diverged = vllm_equivalence.compare_av(
        eager, {"a": {"status": "ok", "token_ids": [1, 9, 3]}, "b": dict(eager["b"])}
    )
    assert diverged["token_identity_rate"] == 0.5 and not diverged["bar_met"]
    row = next(r for r in diverged["per_row"] if r["id"] == "a")
    assert row["first_divergence"] == 1
    assert row["prefix_token_agreement"] == pytest.approx(1 / 3)
    shortened = vllm_equivalence.compare_av(
        eager, {"a": {"status": "ok", "token_ids": [1, 2]}, "b": dict(eager["b"])}
    )
    row = next(r for r in shortened["per_row"] if r["id"] == "a")
    assert not row["identical"] and row["first_divergence"] == 2


def test_equivalence_compare_ar_logic():
    generator = torch.Generator().manual_seed(0)
    eager = {f"r{i}": torch.randn(16, generator=generator) for i in range(3)}
    same = vllm_equivalence.compare_ar(eager, dict(eager))
    assert same["cosine"]["min"] == pytest.approx(1.0) and same["bar_met"]
    rotated = dict(eager)
    rotated["r1"] = -eager["r1"]
    worse = vllm_equivalence.compare_ar(eager, rotated)
    assert worse["cosine"]["min"] == pytest.approx(-1.0) and not worse["bar_met"]
    scaled = {key: value * 2 for key, value in eager.items()}
    norms = vllm_equivalence.compare_ar(eager, scaled)
    assert norms["bar_met"]  # cosine-only bar
    assert norms["norm_relative_error"]["max"] == pytest.approx(1.0)
    missing = vllm_equivalence.compare_ar(eager, {"r0": eager["r0"]})
    assert not missing["bar_met"] and missing["compared"] == 1


def test_runner_backend_flag_plumbing():
    args = runner.parse_args([])
    assert (args.av_backend, args.ar_backend) == ("eager", "eager")
    args = runner.parse_args(["--av-backend", "vllm", "--ar-backend", "vllm"])
    assert (args.av_backend, args.ar_backend) == ("vllm", "vllm")
    with pytest.raises(SystemExit):
        runner.parse_args(["--av-backend", "sglang"])


def test_manifest_backend_fields(metadata_factory):
    manifest = runner.build_manifest(
        {"models": {}, "sources": {}},
        Path(__file__),
        [],
        None,
        {"layer": 20},
        {},
        av_backend="vllm",
        ar_backend="eager",
    )
    assert manifest["av_backend"] == "vllm" and manifest["ar_backend"] == "eager"
    assert "NOT CLAIMED" in manifest["vllm_backend"]["cross_backend_equivalence"]
    default = runner.build_manifest(
        {"models": {}, "sources": {}}, Path(__file__), [], None, {"layer": 20}, {}
    )
    assert default["av_backend"] == "eager" and "vllm_backend" not in default


def test_ar_stage_vllm_maps_worker_rows(
    tmp_path, monkeypatch, tokenizer, metadata_factory
):
    """The vllm AR stage writes the same row schema without loading a model."""
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)

    class Run:
        path = run_dir

    meta = metadata_factory("ar")
    vector = torch.randn(16)
    rows = [{"id": "g-A-x", "group_id": "g"}, {"id": "g-A-y", "group_id": "g"}]
    activations = {
        "g-A-x": type("R", (), {"vector": torch.randn(16)})(),
        "g-A-y": type("R", (), {"vector": torch.randn(16)})(),
    }
    descriptions = {"g-A-x": "x is currently 5"}
    directions = {}
    completion = {
        "status": "COMPLETE_WITH_FAILURES",
        "model_forward_calls": 1,
        "rows": {"g-A-x": {"status": "ok", "seconds": 0.5}},
    }
    seen = {}

    def fake_batch(**kwargs):
        seen.update(kwargs)
        return {"g-A-x": vector}, completion

    monkeypatch.setattr(vllm_backend, "reconstruct_batch", fake_batch)
    runner._ar_stage(
        Run(),
        rows,
        activations,
        descriptions,
        directions,
        backend="vllm",
        paths={"ar": tmp_path},
        tokenizers={"ar": tokenizer},
        ar_meta=meta,
        device="cpu",
        timings={},
        calls={},
    )
    assert rows[0]["reconstruction"]["status"] == "ok"
    assert rows[0]["ar_seconds"] == 0.5
    assert rows[1]["reconstruction"] == {
        "status": "skipped",
        "reason": "AV description unavailable",
    }
    assert directions["g-A-x"] is vector
    assert (run_dir / "raw" / "g-A-x-direction.safetensors").is_file()
    assert (run_dir / "ar_vllm_worker.json").is_file()
    assert seen["items"] == [("g-A-x", "x is currently 5")]
    # The parent computed the AR prompt ids under its own tokenizer convention.
    assert seen["items"][0][0] == "g-A-x"
    assert ar_prompt(tokenizer, meta, "x is currently 5")


def test_av_stage_vllm_maps_worker_rows(
    tmp_path, monkeypatch, tokenizer, metadata_factory
):
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)

    class Run:
        path = run_dir

    rows = [{"id": "g-A-x"}, {"id": "g-A-y"}]
    activations = {
        "g-A-x": type("R", (), {"vector": torch.randn(16)})(),
        "g-A-y": type("R", (), {"vector": torch.randn(16)})(),
    }
    descriptions = {}
    records = {
        "g-A-x": {
            "raw_text": "<explanation>x is currently 5</explanation>",
            "description": "x is currently 5",
            "token_ids": [9, 2],
            "status": "ok",
            "injection_position": 2,
        },
        "g-A-y": {"status": "failed", "error": "simulated worker row failure"},
    }
    completion = {
        "status": "COMPLETE_WITH_FAILURES",
        "model_forward_calls": 2,
        "rows": {"g-A-x": {"seconds": 0.25}, "g-A-y": {"seconds": 0.5}},
    }
    seen = {}

    def fake_batch(**kwargs):
        seen.update(kwargs)
        return records, completion

    monkeypatch.setattr(vllm_backend, "verbalize_batch", fake_batch)
    runner._av_stage(
        Run(),
        rows,
        activations,
        descriptions,
        backend="vllm",
        paths={"av": tmp_path},
        tokenizers={"av": tokenizer},
        av_meta=metadata_factory("av"),
        device="cpu",
        timings={},
        calls={},
    )
    assert rows[0]["description"]["status"] == "ok"
    assert rows[0]["av_seconds"] == 0.25
    assert rows[1]["description"]["status"] == "failed"
    assert rows[1]["av_seconds"] == 0.5
    assert descriptions == {"g-A-x": "x is currently 5"}
    assert set(seen["vectors"]) == {"g-A-x", "g-A-y"}
    assert (run_dir / "av_vllm_worker.json").is_file()


def test_ar_checkpoint_inventory(tmp_path):
    layers = 2
    names = {"model.embed_tokens.weight"} | {
        f"model.layers.{i}.{name}"
        for i in range(layers)
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    }
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}
        )
    )
    vllm_worker.verify_ar_checkpoint_inventory(tmp_path, layers)
    # The released AR legitimately lacks these; presence is tolerated, other
    # extras are not.
    names.add("model.norm.weight")
    names.add("lm_head.weight")
    index.write_text(json.dumps({"weight_map": {name: "f" for name in names}}))
    vllm_worker.verify_ar_checkpoint_inventory(tmp_path, layers)
    names.add("model.layers.0.self_attn.o_proj.bias")
    index.write_text(json.dumps({"weight_map": {name: "f" for name in names}}))
    with pytest.raises(ValueError, match="unexpected"):
        vllm_worker.verify_ar_checkpoint_inventory(tmp_path, layers)
    names.remove("model.layers.0.self_attn.o_proj.bias")
    names.remove("model.layers.1.mlp.down_proj.weight")
    index.write_text(json.dumps({"weight_map": {name: "f" for name in names}}))
    with pytest.raises(ValueError, match="lacks"):
        vllm_worker.verify_ar_checkpoint_inventory(tmp_path, layers)


def test_worker_cli_rejects_a_bad_job_in_process(tmp_path, capsys):
    job = _base_job(tmp_path)
    job["metadata"] = {"width": "sixteen"}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert vllm_worker.main(["verbalize", "--job", str(path)]) == 1
    assert "width" in capsys.readouterr().err


def test_worker_cli_fails_loudly_on_a_bad_job(tmp_path):
    job = _base_job(tmp_path)
    job["metadata"] = {"width": "sixteen"}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    python = vllm_backend.DEFAULT_WORKER_PYTHON
    if not python.is_file():
        pytest.skip("worker venv not built; validation logic covered in-process")
    completed = subprocess.run(
        [str(python), str(vllm_backend.WORKER_SCRIPT), "verbalize", "--job", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "width" in completed.stderr
