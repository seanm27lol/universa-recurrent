"""Audited architecture registry: hook paths and pinned shapes per family.

The pipeline is written against whole decoder-block outputs, so the only
architecture-specific facts it may assume are: the config ``model_type``
markers, the attribute path from the loaded model to the module owning the
block list, the audited text-stack width/depth, and the released NLA
extraction block. Each supported family pins those facts here; any other
``model_type`` or repository trio fails closed. Facts were verified against
the released configs and the pinned Transformers 4.57.6 implementation on
2026-09-22 (see reports/gemma3_port_readiness.md).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    family: str
    """Registry key, e.g. "qwen2.5-7b" or "gemma3-12b"."""
    repos: dict
    """Role -> the one pinned Hugging Face repository per role."""
    target_model_type: str
    """model_type on the released target's top-level config."""
    text_model_type: str
    """model_type of the text stack, shared by the released NLA pair."""
    hidden_size: int
    """Audited text-stack width."""
    num_hidden_layers: int
    """Audited full target/AV text depth (the AR keeps extraction_layer + 1)."""
    extraction_layer: int
    """Audited released NLA extraction block index."""
    target_stack_path: tuple
    """Loaded target: attribute path to the module owning .layers/.norm."""
    text_stack_path: tuple
    """Loaded text-only causal LM (the NLA pair): the same path."""
    scaled_embedding: bool
    """The architecture's embedding lookup multiplies rows by sqrt(hidden).

    Gemma3 applies the scale inside the embedding module's forward
    (Gemma3TextScaledWordEmbedding, embed_scale = hidden_size**0.5), so
    model.get_input_embeddings()(ids) already returns scaled rows and the
    AV injection overwrites one post-scale slot — the same convention as
    the pinned upstream recipe (kitft/nla-inference resolve_embed_scale:
    raw weight rows times sqrt(hidden) with the injection written into the
    scaled stream). Qwen2 rows are unscaled (scale exactly one). Capture
    and patch never see this: both hook post-residual block outputs.
    """


ARCH_SPECS = {
    "qwen2.5-7b": ArchSpec(
        family="qwen2.5-7b",
        repos={
            "target": "Qwen/Qwen2.5-7B-Instruct",
            "av": "kitft/nla-qwen2.5-7b-L20-av",
            "ar": "kitft/nla-qwen2.5-7b-L20-ar",
        },
        target_model_type="qwen2",
        text_model_type="qwen2",
        hidden_size=3584,
        num_hidden_layers=28,
        extraction_layer=20,
        target_stack_path=("model",),
        text_stack_path=("model",),
        scaled_embedding=False,
    ),
    "gemma3-12b": ArchSpec(
        family="gemma3-12b",
        repos={
            # The official google/gemma-3-12b-it is gated-manual (anonymous
            # HTTP 401, 2026-09-22); the target pins the public unsloth mirror
            # instead. The five weight shards and both tokenizer blobs carry
            # identical LFS sha256 in both repos' API records (byte-identical
            # content); the divergence is confined to four small config files.
            # The lock records the official revision alongside for later
            # reconciliation. Gemma Terms of Use apply regardless of source.
            "target": "unsloth/gemma-3-12b-it",
            "av": "kitft/nla-gemma3-12b-L32-av",
            "ar": "kitft/nla-gemma3-12b-L32-ar",
        },
        # The target ships Gemma3ForConditionalGeneration: the text stack sits
        # under model.language_model. The released NLA pair is text-only
        # Gemma3ForCausalLM (model_type gemma3_text), where the stack is model,
        # exactly like Qwen2ForCausalLM.
        target_model_type="gemma3",
        text_model_type="gemma3_text",
        hidden_size=3840,
        num_hidden_layers=48,
        extraction_layer=32,
        target_stack_path=("model", "language_model"),
        text_stack_path=("model",),
        scaled_embedding=True,
    ),
}


def spec_for_model_type(model_type) -> ArchSpec:
    """The one audited spec matching a target or text-stack model_type."""
    matches = [
        spec
        for spec in ARCH_SPECS.values()
        if model_type in (spec.target_model_type, spec.text_model_type)
    ]
    if model_type is None or len(matches) != 1:
        raise ValueError(f"unsupported architecture for this pipeline: {model_type!r}")
    return matches[0]


def spec_for_repos(repos: dict) -> ArchSpec:
    """The one audited spec whose pinned repository trio matches exactly."""
    matches = [spec for spec in ARCH_SPECS.values() if dict(spec.repos) == dict(repos)]
    if len(matches) != 1:
        raise ValueError("wrong model or non-immutable model revision")
    return matches[0]


def _stack_path(model) -> tuple:
    spec = spec_for_model_type(getattr(model.config, "model_type", None))
    if model.config.model_type == spec.target_model_type:
        return spec.target_stack_path
    return spec.text_stack_path


def text_stack(model):
    """The loaded module owning .layers/.norm/embed_tokens; fails closed."""
    module = model
    for name in _stack_path(model):
        module = getattr(module, name, None)
        if module is None:
            raise ValueError(
                f"audited text-stack path is absent on {type(model).__name__}"
            )
    if not hasattr(module, "layers"):
        raise ValueError(f"text stack of {type(model).__name__} has no block list")
    return module


def decoder_layers(model):
    """The decoder block list of an audited loaded model."""
    return text_stack(model).layers


def layers_dotted_path(model) -> str:
    """The recorded hook path prefix, e.g. "model.language_model.layers"."""
    return ".".join((*_stack_path(model), "layers"))


def stack_config(model):
    """The config object carrying hidden_size/depth for the loaded model."""
    config = model.config
    spec_for_model_type(getattr(config, "model_type", None))
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def text_config_dict(config: dict) -> dict:
    """Unwrap a raw target config dict to its text-stack config.

    Gemma3ForConditionalGeneration nests the text stack under text_config;
    text-only configs (Qwen2, Gemma3ForCausalLM) are returned unchanged.
    KeyError on missing keys is deliberate: callers wrap it as an invalid
    metadata/config failure.
    """
    model_type = config["model_type"]
    spec = spec_for_model_type(model_type)
    if model_type == spec.target_model_type != spec.text_model_type:
        nested = config["text_config"]
        if not isinstance(nested, dict):
            raise ValueError("target config text_config is not a mapping")
        return nested
    return config
