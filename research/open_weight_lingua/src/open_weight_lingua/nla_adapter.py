"""Local Transformers alternative to the pinned NLA SGLang recipe.

Fixed-template embedding injection; fresh, cache-free greedy forwards. AR sees
only description text. Norm restoration lives outside both pretrained adapters.
Architecture/conventions: kitft/nla-inference, pinned in model-lock.json.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from string import Formatter
import torch
import yaml
from safetensors.torch import load_file

from .architectures import decoder_layers, stack_config, text_config_dict, text_stack
from .geometry import unit_direction


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Metadata:
    role: str
    width: int
    layer: int
    injection_scale: float | None
    mse_scale: float
    injection_char: str
    injection_id: int
    left_id: int
    right_id: int
    suffix_ids: tuple[int, ...]
    av_template: str
    ar_template: str
    layers: int


def load_metadata(directory: Path, role: str, target_config: dict) -> Metadata:
    """Read actual sidecars; missing/contradictory fields stop execution."""
    directory = Path(directory)
    try:
        meta = yaml.safe_load((directory / "nla_meta.yaml").read_text())
        cfg = json.loads((directory / "config.json").read_text())
        _require(role in ("av", "ar") and meta["role"] == role, "wrong NLA role")
        _require(
            meta["kind"] == "nla_model" and meta["schema_version"] == 2,
            "unsupported metadata schema",
        )
        width, layer = meta["d_model"], meta["extraction_layer_index"]
        _require(
            type(width) is int and type(layer) is int, "width/layer must be integers"
        )
        target_text = text_config_dict(target_config)
        _require(
            cfg["model_type"] == target_text["model_type"],
            "NLA backbone/target text-stack architecture mismatch",
        )
        _require(
            width == cfg["hidden_size"] == target_text["hidden_size"],
            "metadata width mismatch",
        )
        _require(
            0 <= layer < target_text["num_hidden_layers"] - 1,
            "extraction layer mismatch",
        )
        native_dtype = cfg.get("dtype", cfg.get("torch_dtype"))
        _require(native_dtype == "bfloat16", "released NLA weights must remain BF16")
        expected_layers = (
            layer + 1 if role == "ar" else target_text["num_hidden_layers"]
        )
        _require(cfg["num_hidden_layers"] == expected_layers, "backbone depth mismatch")
        if role == "ar":
            _require(
                meta["critic"]["extraction_layer_index"] == layer,
                "AR extraction depth mismatch",
            )
        extraction, tokens, templates = (
            meta["extraction"],
            meta["tokens"],
            meta["prompt_templates"],
        )
        scale, mse_scale = extraction["injection_scale"], float(extraction["mse_scale"])
        _require(
            math.isfinite(mse_scale) and math.isclose(mse_scale, math.sqrt(width)),
            "MSE scale mismatch",
        )
        if role == "av":
            _require(
                isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0,
                "missing or invalid AV injection scale",
            )
        else:
            _require(scale is None, "AR must not inject an activation")
        for key, field in (("av", "injection_char"), ("ar", "explanation")):
            fields = [
                name
                for _, name, _, _ in Formatter().parse(templates[key])
                if name is not None
            ]
            _require(fields == [field], f"unexpected fields in {key} template")
        suffix = tokens["critic_suffix_ids"]
        if role == "ar":
            _require(
                isinstance(suffix, list)
                and len(suffix) > 0
                and all(type(i) is int for i in suffix),
                "AR terminal-token convention missing",
            )
        return Metadata(
            role,
            width,
            layer,
            scale,
            mse_scale,
            tokens["injection_char"],
            tokens["injection_token_id"],
            tokens["injection_left_neighbor_id"],
            tokens["injection_right_neighbor_id"],
            tuple(suffix or ()),
            templates["av"],
            templates["ar"],
            cfg["num_hidden_layers"],
        )
    except (KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        raise ValueError(f"missing/invalid {role} metadata: {exc}") from exc


def check_pair(av: Metadata, ar: Metadata):
    for name in (
        "width",
        "layer",
        "mse_scale",
        "injection_char",
        "injection_id",
        "left_id",
        "right_id",
        "av_template",
        "ar_template",
    ):
        _require(
            getattr(av, name) == getattr(ar, name), f"AV/AR metadata disagree: {name}"
        )


def av_prompt(tokenizer, meta: Metadata):
    _require(
        tokenizer.encode(meta.injection_char, add_special_tokens=False)
        == [meta.injection_id],
        "injection marker tokenizer mismatch",
    )
    _require(meta.injection_id != tokenizer.unk_token_id, "injection marker is UNK")
    content = meta.av_template.format(injection_char=meta.injection_char)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    positions = [i for i, token in enumerate(ids) if token == meta.injection_id]
    _require(len(positions) == 1, "expected one injection marker")
    p = positions[0]
    _require(
        0 < p < len(ids) - 1
        and ids[p - 1] == meta.left_id
        and ids[p + 1] == meta.right_id,
        "injection marker context mismatch",
    )
    return ids, p


def ar_prompt(tokenizer, meta: Metadata, description: str):
    _require(
        type(description) is str and bool(description.strip()),
        "AR needs a nonempty description",
    )
    ids = tokenizer.encode(
        meta.ar_template.format(explanation=description), add_special_tokens=True
    )
    _require(
        tuple(ids[-len(meta.suffix_ids) :]) == meta.suffix_ids,
        "AR suffix-token mismatch",
    )
    if tokenizer.bos_token_id is not None:
        _require(ids[0] == tokenizer.bos_token_id, "AR BOS convention mismatch")
    return ids


@dataclass
class DescriptionRecord:
    raw_text: str
    description: str | None
    token_ids: list[int]
    status: str
    injection_position: int


class Verbalizer:
    """No prompt/truth parameter, result cache, retries, or text selection."""

    def __init__(self, model, tokenizer, metadata: Metadata):
        _require(metadata.role == "av", "AV metadata required")
        _require(
            stack_config(model).hidden_size == metadata.width
            and len(decoder_layers(model)) == metadata.layers,
            "AV model/metadata mismatch",
        )
        self.model = model.eval().requires_grad_(False)
        self.tokenizer, self.metadata = tokenizer, metadata
        self.prompt_ids, self.position = av_prompt(tokenizer, metadata)

    @torch.inference_mode()
    def build_embeddings(self, vector):
        _require(vector.shape == (self.metadata.width,), "AV vector width mismatch")
        embedding = self.model.get_input_embeddings()
        ids = torch.tensor([self.prompt_ids], device=embedding.weight.device)
        # The architecture's own embedding scaling (exactly one for Qwen2,
        # sqrt(hidden) applied inside the lookup for Gemma3) is part of the
        # lookup; the injected slot replaces one post-lookup row, matching the
        # pinned upstream injection convention. Do the injection math in fp32.
        embeds = embedding(ids).float().clone()
        embeds[0, self.position] = (
            unit_direction(vector).to(embeds.device) * self.metadata.injection_scale
        )
        return embeds.to(embedding.weight.dtype)

    @torch.inference_mode()
    def verbalize(self, vector, *, max_new_tokens=200) -> DescriptionRecord:
        _require(
            1 <= max_new_tokens <= 200,
            "AV budget must be within the frozen 200-token ceiling",
        )
        prefix = self.build_embeddings(vector)
        generated = []
        embedding = self.model.get_input_embeddings()
        for _ in range(max_new_tokens):
            # A fresh full forward preserves this exact injection on every pass.
            suffix = embedding(
                torch.tensor([generated], device=prefix.device, dtype=torch.long)
            )
            embeds = torch.cat((prefix, suffix), dim=1)
            out = self.model(
                inputs_embeds=embeds,
                attention_mask=torch.ones(
                    embeds.shape[:2], device=prefix.device, dtype=torch.long
                ),
                use_cache=False,
            )
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        matches = list(
            re.finditer(r"<explanation>\s*(.*?)\s*</explanation>", raw, re.DOTALL)
        )
        if (
            len(generated) == max_new_tokens
            and generated[-1] != self.tokenizer.eos_token_id
        ):
            status, description = "truncated", None
        elif (
            len(matches) != 1
            or raw.count("<explanation>") != 1
            or raw.count("</explanation>") != 1
            or not matches[0].group(1).strip()
        ):
            status, description = "invalid_explanation_tags", None
        else:
            status, description = "ok", matches[0].group(1).strip()
        return DescriptionRecord(raw, description, generated, status, self.position)


def load_value_head(path: Path, width: int, dtype, device):
    """Mandatory separate safe tensor: no random-head or LM-head fallback."""
    if not Path(path).is_file():
        raise ValueError("missing required value_head.safetensors")
    state = load_file(str(path), device="cpu")
    if set(state) != {"weight"} or state["weight"].shape != (width, width):
        raise ValueError("AR value head must contain only a width-by-width weight")
    if (
        not state["weight"].is_floating_point()
        or not torch.isfinite(state["weight"]).all()
    ):
        raise ValueError("invalid AR value head")
    if state["weight"].dtype != dtype:
        raise ValueError("AR value-head dtype differs from native backbone dtype")
    head = torch.nn.Linear(width, width, bias=False, dtype=dtype)
    head.load_state_dict(state, strict=True)
    return head.to(device).eval().requires_grad_(False)


class Reconstructor:
    def __init__(self, backbone, tokenizer, metadata: Metadata, head_path: Path):
        _require(metadata.role == "ar", "AR metadata required")
        stack = text_stack(backbone)
        _require(
            stack_config(backbone).hidden_size == metadata.width
            and len(stack.layers) == metadata.layer + 1 == metadata.layers,
            "AR backbone width/depth mismatch",
        )
        parameter = next(backbone.parameters())
        # Require/load the head before changing the caller's backbone.
        self.head = load_value_head(
            head_path, metadata.width, parameter.dtype, parameter.device
        )
        backbone.lm_head = torch.nn.Identity()
        stack.norm = torch.nn.Identity()
        self.model = backbone.eval().requires_grad_(False)
        self.tokenizer, self.metadata = tokenizer, metadata
        ar_prompt(tokenizer, metadata, "tokenizer preflight")

    @torch.inference_mode()
    def reconstruct(self, description: str) -> torch.Tensor:
        """Return AR's direction-bearing vector; no source norm/activation input."""
        ids = ar_prompt(self.tokenizer, self.metadata, description)
        device = next(self.model.parameters()).device
        out = text_stack(self.model)(
            input_ids=torch.tensor([ids], device=device), use_cache=False
        )
        vector = self.head(out.last_hidden_state[0, -1]).float().cpu()
        unit_direction(vector)  # fail closed before passing a degenerate direction
        return vector
