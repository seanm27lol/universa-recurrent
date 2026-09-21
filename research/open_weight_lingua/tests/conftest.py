import math
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from open_weight_lingua.nla_adapter import Metadata


class TinyTokenizer:
    """Deliberately multi-token integers; no downloaded tokenizer required."""

    eos_token_id = 2
    unk_token_id = None
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        if text == "㈎":
            return [7]
        if text.isdigit():
            return [10 + int(digit) for digit in text]
        if text.endswith("</text> <summary>"):
            return [20, 21, 22, 8, 9]
        return [20, 21, 22]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i - 10) if 10 <= i < 20 else f"<{i}>" for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [3, 6, 7, 8, 4]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(73)


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


@pytest.fixture
def model_factory():
    def make(layers=3, dtype=torch.float32):
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        return Qwen2ForCausalLM(config).to(dtype).eval()

    return make


@pytest.fixture
def metadata_factory():
    def make(role="av"):
        return Metadata(
            role,
            16,
            1,
            150.0 if role == "av" else None,
            math.sqrt(16),
            "㈎",
            7,
            6,
            8,
            () if role == "av" else (8, 9),
            "Explain <concept>{injection_char}</concept>",
            "Summary: <text>{explanation}</text> <summary>",
            3 if role == "av" else 2,
        )

    return make
