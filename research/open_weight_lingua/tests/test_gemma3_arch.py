"""Gemma-3 port: tiny random Gemma3 fixtures plus the real fetched sidecars.

The Gemma-shaped fixture models are random-weight software checks, like the
Qwen fixtures — nothing here is a measurement on the released Gemma-3 or NLA
weights. The nla_meta/config fixtures under fixtures/upstream/gemma3-* are
byte-identical copies of the public kitft sidecars and configs pinned in
configs/model-lock-gemma3-12b.json (asserted below). The gated target's
config.json is intentionally NOT fixture-cloned: it cannot be fetched until
the Gemma license is accepted, so the target text-stack shape is wrapped
inline from the real fetched AV config instead.
"""

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import pytest
import torch
import yaml
from safetensors.torch import save_file
from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
)

from open_weight_lingua.architectures import (
    ARCH_SPECS,
    decoder_layers,
    layers_dotted_path,
    spec_for_model_type,
    spec_for_repos,
    text_config_dict,
)
from open_weight_lingua.artifacts import sha256_file
from open_weight_lingua.metrics import answer_tokens
from open_weight_lingua.nla_adapter import (
    Metadata,
    Reconstructor,
    Verbalizer,
    ar_prompt,
    av_prompt,
    check_pair,
    load_metadata,
)
from open_weight_lingua.preflight import load_model, read_lock
from open_weight_lingua.target import TARGET_BUCKET, Site, Target, validate_site
from open_weight_lingua.tasks import generate_groups, tokenize_groups

FIXTURES = Path(__file__).parent / "fixtures/upstream"
PROJECT = Path(__file__).parents[1]
GEMMA_LOCK = PROJECT / "configs/model-lock-gemma3-12b.json"
QWEN_LOCK = PROJECT / "configs/model-lock.json"

# Real released sidecar values (kitft/nla-gemma3-12b-L32-*, pinned in the lock).
INJECTION_CHAR = "㈜"
INJECTION_ID = 246566
LEFT_ID = 236813
RIGHT_ID = 954
AR_SUFFIX_IDS = (1005, 236813, 655, 6011, 236813)


class TinyGemmaTokenizer:
    """Gemma-shaped stub: BOS-first encoding, real sidecar marker/suffix IDs.

    Unlike the Qwen TinyTokenizer (bos_token_id None), the live BOS check in
    ar_prompt fires here, matching the released Gemma convention.
    """

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3

    def encode(self, text, add_special_tokens=False):
        if text == INJECTION_CHAR:
            return [INJECTION_ID]
        if text.isdigit():
            return [10 + int(digit) for digit in text]
        ids = [20, 21, 22]
        if text.endswith("</text> <summary>"):
            ids += AR_SUFFIX_IDS
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i - 10) if 10 <= i < 20 else f"<{i}>" for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # BOS first (Gemma template), one marker flanked by the sidecar
        # neighbors, a content-dependent payload (distinct prompts must not
        # collide), then a shared generation-prompt trailer.
        content = messages[-1]["content"]
        payload = [(ord(char) % 200) + 40 for char in content]
        return [2, LEFT_ID, INJECTION_ID, RIGHT_ID, *payload, 50]


@pytest.fixture
def gemma_tokenizer():
    return TinyGemmaTokenizer()


def tiny_text_config(layers=6, vocab_size=64):
    config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        # Below the pinned bucket on purpose: the fixture exercises the
        # sliding-window path even at padded length.
        sliding_window=64,
        eos_token_id=1,
        bos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def tiny_causal_lm(layers=6, vocab_size=64, dtype=torch.float32):
    return Gemma3ForCausalLM(tiny_text_config(layers, vocab_size)).to(dtype).eval()


def tiny_conditional_generation(dtype=torch.float32):
    config = Gemma3Config(
        text_config=tiny_text_config().to_dict(),
        vision_config={
            "hidden_size": 8,
            "image_size": 16,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_hidden_layers": 2,
            "patch_size": 8,
            "num_channels": 3,
        },
    )
    config._attn_implementation = "eager"
    return Gemma3ForConditionalGeneration(config).to(dtype).eval()


def load_metadata_from_fixture(tmp_path, role):
    for name in ("config.json", "nla_meta.yaml"):
        shutil.copyfile(FIXTURES / f"gemma3-{role}-{name}", tmp_path / name)
    return load_metadata(tmp_path, role, gemma_target_config())


def gemma_target_config():
    """Gated-target config shape, wrapped inline from the real AV config.

    google/gemma-3-12b-it/config.json is license-gated; its text stack is the
    same Gemma3-12B stack the public AV checkpoint carries (3840 wide, 48
    blocks), so the fixture nests that real fetched text config under a
    gemma3 multimodal wrapper exactly as Gemma3Config serializes it.
    """
    text_config = json.loads((FIXTURES / "gemma3-av-config.json").read_text())
    return {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": text_config,
        "vision_config": {"model_type": "siglip_vision_model"},
    }


def tiny_metadata(role="av", layers=6):
    """Real sidecar conventions (marker/suffix/templates) at fixture width."""
    return Metadata(
        role,
        16,
        1,
        80000.0 if role == "av" else None,
        math.sqrt(16),
        INJECTION_CHAR,
        INJECTION_ID,
        LEFT_ID,
        RIGHT_ID,
        () if role == "av" else AR_SUFFIX_IDS,
        "Explain <concept>{injection_char}</concept>",
        "Summary: <text>{explanation}</text> <summary>",
        layers if role == "av" else 2,
    )


def test_real_sidecars_match_the_gemma_lock():
    lock = json.loads(GEMMA_LOCK.read_text())
    for role in ("av", "ar"):
        for name in ("config.json", "nla_meta.yaml"):
            fixture = FIXTURES / f"gemma3-{role}-{name}"
            assert (
                sha256_file(fixture) == lock["models"][role]["files"][name]["sha256"]
            )
    # The gated target config is pending; no fixture pretends otherwise.
    assert lock["models"]["target"]["files"]["config.json"]["sha256"] is None
    assert not (FIXTURES / "gemma3-target-config.json").exists()


def test_real_released_gemma_metadata(tmp_path):
    av = load_metadata_from_fixture(tmp_path, "av")
    ar = load_metadata_from_fixture(tmp_path, "ar")
    check_pair(av, ar)
    assert (av.width, av.layer, av.injection_scale) == (3840, 32, 80000.0)
    assert av.mse_scale == pytest.approx(math.sqrt(3840))
    assert av.injection_char == INJECTION_CHAR
    assert (av.injection_id, av.left_id, av.right_id) == (
        INJECTION_ID,
        LEFT_ID,
        RIGHT_ID,
    )
    assert ar.injection_scale is None
    assert ar.suffix_ids == AR_SUFFIX_IDS
    assert ar.layers == 33  # truncated to the extraction block plus one


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 3),
        ("extraction_layer_index", 31),
        ("role", "ar"),
        ("schema_version", 1),
        ("extraction", {"injection_scale": None, "mse_scale": 1}),
    ],
)
def test_gemma_metadata_mismatch(tmp_path, field, value):
    for name in ("config.json", "nla_meta.yaml"):
        shutil.copyfile(FIXTURES / f"gemma3-av-{name}", tmp_path / name)
    path = tmp_path / "nla_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    meta[field] = value
    path.write_text(yaml.safe_dump(meta))
    if field == "extraction_layer_index":
        av = load_metadata(tmp_path, "av", gemma_target_config())
        with pytest.raises(ValueError, match="disagree"):
            check_pair(av, load_metadata_from_fixture(tmp_path, "ar"))
    else:
        with pytest.raises(ValueError):
            load_metadata(tmp_path, "av", gemma_target_config())


def test_gemma_ar_depth_rule(tmp_path):
    for name in ("config.json", "nla_meta.yaml"):
        shutil.copyfile(FIXTURES / f"gemma3-ar-{name}", tmp_path / name)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["num_hidden_layers"] = 32
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="depth"):
        load_metadata(tmp_path, "ar", gemma_target_config())


def test_l32_is_a_local_attention_block():
    """The released extraction site sits in a sliding-window block.

    AV/target text stack: 48 blocks, full attention at every 6th index
    (5, 11, ..., 47); block 32 is sliding_attention with window 1024, wider
    than the pinned 128-token bucket, so no pinned forward ever truncates.
    The AR keeps blocks 0..32 with the same pattern.
    """
    av = json.loads((FIXTURES / "gemma3-av-config.json").read_text())
    layer_types = av["layer_types"]
    assert len(layer_types) == av["num_hidden_layers"] == 48
    full = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    assert full == [5, 11, 17, 23, 29, 35, 41, 47]
    assert layer_types[32] == "sliding_attention"
    assert av["sliding_window"] == 1024 > TARGET_BUCKET
    ar = json.loads((FIXTURES / "gemma3-ar-config.json").read_text())
    assert len(ar["layer_types"]) == ar["num_hidden_layers"] == 33
    assert ar["layer_types"][32] == "sliding_attention"


def test_architecture_registry_fails_closed():
    assert spec_for_model_type("qwen2").family == "qwen2.5-7b"
    assert spec_for_model_type("gemma3").family == "gemma3-12b"
    assert spec_for_model_type("gemma3_text").family == "gemma3-12b"
    for bad in (None, "llama", "gemma2"):
        with pytest.raises(ValueError, match="unsupported architecture"):
            spec_for_model_type(bad)
    assert (
        spec_for_repos(dict(ARCH_SPECS["gemma3-12b"].repos)).family == "gemma3-12b"
    )
    with pytest.raises(ValueError, match="non-immutable"):
        spec_for_repos(
            {**dict(ARCH_SPECS["gemma3-12b"].repos), "target": "someone/else"}
        )


def test_text_config_unwraps_only_multimodal():
    unwrapped = text_config_dict(gemma_target_config())
    assert unwrapped["model_type"] == "gemma3_text"
    assert (unwrapped["hidden_size"], unwrapped["num_hidden_layers"]) == (3840, 48)
    flat = json.loads((FIXTURES / "gemma3-av-config.json").read_text())
    assert text_config_dict(flat) is flat
    qwen = json.loads((FIXTURES / "target-config.json").read_text())
    assert text_config_dict(qwen) is qwen
    with pytest.raises(KeyError):
        text_config_dict({"model_type": "gemma3"})


def test_wrapped_target_capture_patch_greedy_and_scores(gemma_tokenizer):
    """The real gemma3 target hook path: model.language_model.layers.N."""
    target = Target(tiny_conditional_generation(), gemma_tokenizer)
    assert layers_dotted_path(target.model) == "model.language_model.layers"
    assert len(decoder_layers(target.model)) == 6
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record, checks, _ = target.identity_gate(ids, mask, Site(1, 2))
    assert record.hook_path == "model.language_model.layers.1"
    assert record.hidden_state_index == 2
    assert all(item["bitwise_equal"] for item in checks.values())
    with pytest.raises(ValueError, match="final norm"):
        target.capture(ids, mask, Site(5, 2))
    score = target.score_answer(ids, mask, record.site, "19", record.vector)
    assert score["token_ids_including_eos"] == [11, 19, 1]
    assert score["log_probability"] < 0
    assert score["suffix_drift_relative"] == 0.0  # same-shape gate, bitwise


def test_wrapped_target_padded_matches_unpadded(gemma_tokenizer):
    """Sliding-window pads stay exactly masked: bucketed == unpadded logits."""
    target = Target(tiny_conditional_generation(), gemma_tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    with torch.inference_mode():
        unpadded = target.model(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).logits
    padded = target.forward(ids, mask)
    assert padded.shape[1] == TARGET_BUCKET
    assert torch.allclose(
        unpadded.float(), padded[0, :3].unsqueeze(0).float(), atol=1e-5, rtol=1e-5
    )
    assert torch.equal(unpadded.argmax(-1), padded[0, :3].argmax(-1).unsqueeze(0))


def test_validate_site_rejects_unaudited_architecture():
    from types import SimpleNamespace

    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    ids = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="unsupported architecture"):
        validate_site(model, ids, mask, Site(1, 2))


def test_embedding_scale_matches_injection_convention(gemma_tokenizer):
    """Gemma3 scales rows by sqrt(hidden) inside the lookup; the AV injection
    overwrites one post-scale slot, so the injected norm is exactly the
    sidecar injection_scale — the pinned upstream recipe's convention."""
    model = tiny_causal_lm(layers=6, vocab_size=262208)
    embedding = model.get_input_embeddings()
    assert float(embedding.embed_scale) == pytest.approx(4.0)  # sqrt(16)
    with torch.inference_mode():
        scaled_row = embedding(torch.tensor([[3]]))[0, 0]
        raw_row_norm = float(embedding.weight[3].norm())
    assert float(scaled_row.norm()) == pytest.approx(4.0 * raw_row_norm, rel=1e-5)
    av = Verbalizer(model, gemma_tokenizer, tiny_metadata("av"))
    assert av.position == 2
    embeds = av.build_embeddings(torch.arange(1, 17).float())
    assert float(embeds[0, av.position].norm()) == pytest.approx(80000.0, rel=1e-4)
    with torch.inference_mode():
        reference = embedding(torch.tensor([av.prompt_ids]))
    unmodified = torch.ones(embeds.shape[:2], dtype=torch.bool)
    unmodified[0, av.position] = False
    assert torch.equal(embeds[unmodified], reference[unmodified])


def test_av_prompt_and_ar_prompt_gemma_conventions(tmp_path, gemma_tokenizer):
    av, ar = (
        load_metadata_from_fixture(tmp_path, "av"),
        load_metadata_from_fixture(tmp_path, "ar"),
    )
    ids, position = av_prompt(gemma_tokenizer, av)
    assert ids[position] == INJECTION_ID
    assert ids[position - 1] == LEFT_ID and ids[position + 1] == RIGHT_ID
    assert ids[0] == gemma_tokenizer.bos_token_id  # gemma template BOS
    ar_ids = ar_prompt(gemma_tokenizer, ar, "a description")
    assert tuple(ar_ids[-5:]) == AR_SUFFIX_IDS
    assert ar_ids[0] == gemma_tokenizer.bos_token_id
    with pytest.raises(ValueError, match="suffix"):
        ar_prompt(gemma_tokenizer, replace(ar, suffix_ids=(1,)), "x")


def test_ar_reconstructor_on_gemma3_text_stack(tmp_path, gemma_tokenizer):
    model = tiny_causal_lm(layers=2, vocab_size=262208)
    meta = tiny_metadata("ar")
    with pytest.raises(ValueError, match="required value_head"):
        Reconstructor(model, gemma_tokenizer, meta, tmp_path / "value_head.safetensors")
    assert not isinstance(model.model.norm, torch.nn.Identity)
    path = tmp_path / "value_head.safetensors"
    save_file({"weight": torch.eye(16)}, path)
    ar = Reconstructor(model, gemma_tokenizer, meta, path)
    observed = []
    handle = model.model.layers[1].register_forward_hook(
        lambda _, __, out: observed.append(out[0][0, -1].detach().clone())
    )
    vector = ar.reconstruct("A description only")
    handle.remove()
    assert torch.equal(vector, observed[0])
    assert isinstance(model.model.norm, torch.nn.Identity)
    assert isinstance(model.lm_head, torch.nn.Identity)


def test_answer_tokens_and_tokenize_groups_gemma_stub(gemma_tokenizer):
    assert answer_tokens(gemma_tokenizer, "19") == [11, 19, 1]
    groups, _ = generate_groups("smoke", 2)
    rows = tokenize_groups(groups, gemma_tokenizer)
    assert len(rows) == 8
    for row in rows:
        assert row["position"] == len(row["input_ids"]) - 1
        assert row["attention_mask"] == [1] * len(row["input_ids"])


def test_gemma_lock_pending_fails_closed_and_completion_passes(tmp_path):
    lock = json.loads(GEMMA_LOCK.read_text())
    assert lock["schema_version"] == 1
    assert {entry["license"] for entry in lock["models"].values()} == {"gemma"}
    assert lock["models"]["target"]["gated"] == "manual"
    pending = [
        name
        for name, file in lock["models"]["target"]["files"].items()
        if file["sha256"] is None
    ]
    assert sorted(pending) == [
        "README.md",
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    for entry in lock["models"].values():
        assert entry["download_bytes"] == sum(
            file["bytes"] for file in entry["files"].values()
        )
        for file in entry["files"].values():
            if file["sha256"] is not None:
                assert len(file["sha256"]) == 64
    # The pipeline's gate refuses the pending lock with the actionable path.
    with pytest.raises(ValueError, match="unresolved gated-file hashes"):
        read_lock(GEMMA_LOCK)
    # A completed lock (every hash pinned) passes and resolves the family.
    for file in lock["models"]["target"]["files"].values():
        if file["sha256"] is None:
            file["sha256"] = "0" * 64
    completed = tmp_path / "completed-lock.json"
    completed.write_text(json.dumps(lock))
    resolved = read_lock(completed)
    assert (
        spec_for_repos(
            {role: entry["repo_id"] for role, entry in resolved["models"].items()}
        ).family
        == "gemma3-12b"
    )
    # The Qwen lock is untouched by the generalization and still resolves.
    qwen = read_lock(QWEN_LOCK)
    assert (
        spec_for_repos(
            {role: entry["repo_id"] for role, entry in qwen["models"].items()}
        ).family
        == "qwen2.5-7b"
    )
    # Both locks pin the same audited inference/training source revisions.
    assert lock["sources"] == qwen["sources"]


def test_complete_gemma3_lock_refuses_a_complete_lock(tmp_path):
    import subprocess
    import sys

    completed = tmp_path / "lock.json"
    completed.write_text(QWEN_LOCK.read_text())
    script = PROJECT / "scripts/complete_gemma3_lock.py"
    result = subprocess.run(
        [sys.executable, str(script), "--lock", str(completed)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "already complete" in result.stdout + result.stderr


def test_load_model_tiny_gemma3_ar_checkpoint(tmp_path):
    """AR truncation convention on the gemma3_text stack: no norm/lm_head."""
    model = tiny_causal_lm(layers=2)
    state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in ("model.norm.weight", "lm_head.weight")
    }
    model.config.save_pretrained(tmp_path)
    save_file(state, tmp_path / "model.safetensors")
    loaded = load_model(tmp_path, "ar", "cpu")
    assert type(loaded).__name__ == "Gemma3ForCausalLM"
    with pytest.raises(ValueError, match="does not load completely"):
        load_model(tmp_path, "av", "cpu")
