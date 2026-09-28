from dataclasses import replace
import inspect
import json
from pathlib import Path
import shutil
import pytest
import torch
import yaml
from safetensors.torch import save_file
from open_weight_lingua.nla_adapter import (
    Verbalizer,
    Reconstructor,
    load_metadata,
    load_value_head,
    check_pair,
    av_prompt,
    ar_prompt,
)
from open_weight_lingua.geometry import unit_direction, restore_norm, direction_metrics

FIXTURES = Path(__file__).parent / "fixtures/upstream"


def metadata_dir(tmp_path, role):
    for name in ("config.json", "nla_meta.yaml"):
        shutil.copyfile(FIXTURES / f"{role}-{name}", tmp_path / name)
    return json.loads((FIXTURES / "target-config.json").read_text())


def test_actual_released_metadata(tmp_path):
    target = metadata_dir(tmp_path, "av")
    av = load_metadata(tmp_path, "av", target)
    metadata_dir(tmp_path, "ar")
    ar = load_metadata(tmp_path, "ar", target)
    check_pair(av, ar)
    assert (av.layer, av.width, av.injection_scale, ar.layers) == (20, 3584, 150, 21)
    assert ar.suffix_ids == (1318, 29, 366, 1708, 29)


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 3),
        ("extraction_layer_index", 19),
        ("role", "ar"),
        ("schema_version", 99),
        ("extraction", {"injection_scale": None, "mse_scale": 1}),
    ],
)
def test_metadata_mismatch(tmp_path, field, value):
    target = metadata_dir(tmp_path, "av")
    path = tmp_path / "nla_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    meta[field] = value
    path.write_text(yaml.safe_dump(meta))
    if field == "extraction_layer_index":
        av = load_metadata(tmp_path, "av", target)
        metadata_dir(tmp_path, "ar")
        with pytest.raises(ValueError, match="disagree"):
            check_pair(av, load_metadata(tmp_path, "ar", target))
    else:
        with pytest.raises(ValueError):
            load_metadata(tmp_path, "av", target)


def test_missing_metadata_and_bad_ar_depth(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        load_metadata(tmp_path, "ar", {})
    target = metadata_dir(tmp_path, "ar")
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["num_hidden_layers"] = 20
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="depth"):
        load_metadata(tmp_path, "ar", target)


def test_tokenizer_context_mismatch(tokenizer, metadata_factory):
    meta = metadata_factory()
    with pytest.raises(ValueError, match="context"):
        av_prompt(tokenizer, replace(meta, left_id=31))
    with pytest.raises(ValueError, match="tokenizer"):
        av_prompt(tokenizer, replace(meta, injection_id=31))
    with pytest.raises(ValueError, match="suffix"):
        ar_prompt(tokenizer, replace(metadata_factory("ar"), suffix_ids=(31,)), "text")


def test_injection_scale_only_one_embedding_and_no_cached_results(
    model_factory, tokenizer, metadata_factory
):
    av = Verbalizer(model_factory(), tokenizer, metadata_factory())
    first, second = torch.arange(1, 17).float(), torch.arange(16, 0, -1).float()
    a, b = av.build_embeddings(first), av.build_embeddings(second)
    assert ((a != b).any(-1)).nonzero().tolist() == [[0, av.position]]
    assert float(a[0, av.position].norm()) == pytest.approx(150, rel=1e-6)
    captures = []
    hook = av.model.register_forward_hook(
        lambda _, __, out: captures.append(out.logits.detach().clone())
    )
    av.verbalize(first, max_new_tokens=1)
    av.verbalize(second, max_new_tokens=1)
    av.verbalize(first, max_new_tokens=1)
    hook.remove()
    assert len(captures) == 3  # same token IDs, three actual fresh forwards
    assert not torch.equal(captures[0], captures[1])
    assert torch.equal(captures[0], captures[2])


def test_ar_required_head_and_raw_final_block(
    tmp_path, model_factory, tokenizer, metadata_factory
):
    model = model_factory(layers=2)
    with pytest.raises(ValueError, match="required value_head"):
        Reconstructor(
            model,
            tokenizer,
            metadata_factory("ar"),
            tmp_path / "value_head.safetensors",
        )
    assert not isinstance(model.model.norm, torch.nn.Identity)
    path = tmp_path / "value_head.safetensors"
    save_file({"weight": torch.eye(16)}, path)
    ar = Reconstructor(model, tokenizer, metadata_factory("ar"), path)
    observed = []
    handle = ar.model.model.layers[1].register_forward_hook(
        lambda _, __, out: observed.append(out[0, -1].detach().clone())
    )
    vector = ar.reconstruct("A description only")
    handle.remove()
    assert torch.equal(vector, observed[0])
    assert isinstance(ar.model.model.norm, torch.nn.Identity)
    assert isinstance(ar.model.lm_head, torch.nn.Identity)
    assert not ar.head.weight.requires_grad


@pytest.mark.parametrize(
    "state",
    [
        {"weight": torch.ones(3, 3)},
        {"weight": torch.eye(16), "bias": torch.zeros(16)},
        {"weight": torch.full((16, 16), float("nan"))},
    ],
)
def test_bad_value_head(tmp_path, state):
    path = tmp_path / "value_head.safetensors"
    save_file(state, path)
    with pytest.raises(ValueError):
        load_value_head(path, 16, torch.float32, "cpu")


def test_information_boundary():
    assert list(inspect.signature(Verbalizer.verbalize).parameters) == [
        "self",
        "vector",
        "max_new_tokens",
    ]
    assert list(inspect.signature(Reconstructor.reconstruct).parameters) == [
        "self",
        "description",
    ]
    assert list(inspect.signature(restore_norm).parameters) == [
        "direction",
        "retained_norm",
        "dtype",
    ]


@pytest.mark.parametrize(
    "vector",
    [
        torch.zeros(3),
        torch.tensor([float("nan"), 1, 2]),
        torch.tensor([float("inf"), 1, 2]),
        torch.ones(3) * 1e-20,
    ],
)
def test_invalid_direction(vector):
    with pytest.raises(ValueError):
        unit_direction(vector)


@pytest.mark.parametrize("norm", [0, -1, float("nan"), float("inf")])
def test_invalid_retained_norm(norm):
    with pytest.raises(ValueError):
        restore_norm(torch.ones(3), norm, dtype=torch.float32)


def test_norm_channel_and_metric_names():
    vector = torch.tensor([3.0, 4.0, 0.0])
    result = restore_norm(vector, 10, dtype=torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, torch.tensor([6, 8, 0], dtype=torch.bfloat16))
    diagnostics = direction_metrics(vector, torch.tensor([1.0, 2.0, 3.0]))
    assert diagnostics["unit_direction_squared_l2"] == pytest.approx(
        2 * (1 - diagnostics["cosine"]), abs=1e-6
    )


@pytest.mark.parametrize(
    "raw,status",
    [
        ("<explanation>A description</explanation><2>", "ok"),
        ("<explanation>missing close", "invalid_explanation_tags"),
        ("<explanation><explanation>nested</explanation>", "invalid_explanation_tags"),
        ("<explanation> </explanation>", "invalid_explanation_tags"),
    ],
)
def test_description_parsing_without_retries(
    raw, status, monkeypatch, model_factory, tokenizer, metadata_factory
):
    from types import SimpleNamespace

    model = model_factory()
    calls = []

    def forced_eos(**kwargs):
        calls.append(kwargs)
        logits = torch.zeros((1, kwargs["inputs_embeds"].shape[1], 32))
        logits[:, :, tokenizer.eos_token_id] = 1
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(model, "forward", forced_eos)
    monkeypatch.setattr(tokenizer, "decode", lambda ids, **kwargs: raw)
    result = Verbalizer(model, tokenizer, metadata_factory()).verbalize(torch.ones(16))
    assert result.status == status
    assert len(calls) == 1 and calls[0]["use_cache"] is False
    assert "input_ids" not in calls[0]
    assert result.description == ("A description" if status == "ok" else None)


def test_value_head_native_precision(tmp_path):
    path = tmp_path / "value_head.safetensors"
    save_file({"weight": torch.eye(16)}, path)
    with pytest.raises(ValueError, match="dtype"):
        load_value_head(path, 16, torch.bfloat16, "cpu")
