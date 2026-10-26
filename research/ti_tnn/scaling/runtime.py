"""The Qwen runtime for the ordinary baselines (B4, Q27) and the frozen backbone: pinned loading, per-turn generation,
tool-call parsing and exact token, time and memory accounting.

**Example.** Q27 receives a question, its schema and three retrieved training examples, with three tools available. In
thinking mode the chat template ends the prompt with `<think>\\n`. The model then writes reasoning, `</think>`, and then
either a `<tool_call>` block or a final answer, and ends the turn with `<|im_end|>`.

For each turn, `Runtime.generate_turn` returns:
- the generated text, split into reasoning and content;
- the parsed tool calls;
- token counts: prompt, generated, reasoning, content;
- the finish reason: `eos`, `max_new_tokens` (truncated) or `timeout`;
- the time to first token, the decode time and the decode rate.

**Pinned loading.** Weights, tokenizer and chat template are read from the local cache at the revision in `contracts.MODELS`,
with no network. The precision is BF16 (`dtype=torch.bfloat16`). Any other precision would be a separately named
configuration, never a silent substitute.

**Sampling.** The settings come from the official model cards, keyed by (model, thinking), and are recorded in every
run's configuration. Presence penalty follows the OpenAI/vLLM convention: subtract the penalty once from the logit of
every token already generated in this turn.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from . import contracts as C
from . import identity

# The model cards' recommended settings (Qwen3.8-27B: thinking / instruct; Qwen3.5-4B: thinking for precise coding /
# instruct for general tasks). SQL generation is precise coding, so B4 thinking uses the card's coding setting.
SAMPLING = {
    ("Q27", True): {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                    "repetition_penalty": 1.0, "source": "Qwen3.8-27B card: thinking mode"},
    ("Q27", False): {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                     "repetition_penalty": 1.0, "source": "Qwen3.8-27B card: instruct (non-thinking) mode"},
    ("B4", True): {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                   "repetition_penalty": 1.0, "source": "Qwen3.5-4B card: thinking mode for precise coding tasks"},
    ("B4", False): {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                    "repetition_penalty": 1.0, "source": "Qwen3.5-4B card: instruct (non-thinking) mode, general tasks"},
}

TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


@dataclass
class GenerationConfig:
    model: str                      # "Q27" | "B4"
    thinking: bool
    max_new_tokens: int
    reasoning_effort: str | None = None   # Q27 only: "xhigh" (default) | "medium" | "low"
    sampling: dict = field(default_factory=dict)

    @classmethod
    def official(cls, model: str, thinking: bool, limits: C.Limits, reasoning_effort: str | None = None):
        if reasoning_effort is not None and model != "Q27":
            raise ValueError("reasoning_effort is a Qwen3.8 template option")
        return cls(model=model, thinking=thinking, reasoning_effort=reasoning_effort,
                   max_new_tokens=limits.max_new_tokens_thinking if thinking else limits.max_new_tokens_nonthinking,
                   sampling=dict(SAMPLING[(model, thinking)]))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Turn:
    text: str
    reasoning: str
    content: str
    tool_calls: list
    prompt_tokens: int
    generated_tokens: int
    reasoning_tokens: int
    content_tokens: int
    finish_reason: str              # "eos" | "max_new_tokens" | "timeout"
    seconds: float
    time_to_first_token: float | None
    decode_tokens_per_second: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def parse_tool_calls(content: str) -> list[dict]:
    """Qwen's XML tool-call format (`qwen3_coder`): <tool_call><function=NAME><parameter=P>V</parameter>...</function></tool_call>."""
    calls = []
    for name, body in TOOL_CALL.findall(content):
        calls.append({"name": name, "arguments": {k: v for k, v in PARAMETER.findall(body)}})
    return calls


def split_reasoning(text: str, thinking: bool) -> tuple[str, str]:
    """(reasoning, content). In thinking mode the prompt already opened `<think>`, so everything before `</think>` is
    reasoning; with no `</think>` (truncated mid-thought), all of it is reasoning."""
    if not thinking:
        return "", text
    if "</think>" in text:
        r, c = text.split("</think>", 1)
        return r.strip("\n"), c.lstrip("\n")
    return text, ""


class Runtime:
    """One loaded model. Loading happens inside the caller's GPU reservation, so it is charged."""

    def __init__(self, key: str, device: str = "cuda", attn_implementation: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.key, self.device = key, device
        self.snapshot = identity.model_snapshot_dir(key)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        kwargs = {"dtype": torch.bfloat16, "local_files_only": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(self.snapshot, **kwargs).to(device).eval()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0
        self.dtype = str(next(self.model.parameters()).dtype)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        self.chat_template_sha256 = identity.sha256_file(self.snapshot / "chat_template.jinja") \
            if (self.snapshot / "chat_template.jinja").exists() else None

    def describe(self) -> dict:
        import torch
        out = {"key": self.key, "repo": C.MODELS[self.key]["repo"], "revision": C.MODELS[self.key]["revision"],
               "dtype": self.dtype, "parameters": self.parameters, "load_seconds": self.load_seconds,
               "chat_template_sha256": self.chat_template_sha256, "model_class": type(self.model).__name__}
        if self.device.startswith("cuda"):
            out["memory_allocated_bytes"] = torch.cuda.memory_allocated()
            out["memory_reserved_bytes"] = torch.cuda.memory_reserved()
        return out

    def render(self, messages: list[dict], tools: list[dict] | None, cfg: GenerationConfig) -> str:
        kw = {"enable_thinking": cfg.thinking}
        if cfg.reasoning_effort is not None:
            kw["reasoning_effort"] = cfg.reasoning_effort
        return self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate_turn(self, messages: list[dict], tools: list[dict] | None, cfg: GenerationConfig, seed: int,
                      deadline: float | None = None, max_new_tokens: int | None = None) -> Turn:
        """One assistant turn. `deadline` is a time.time() wall-clock limit for the whole request."""
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        prompt = self.render(messages, tools, cfg)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        budget = cfg.max_new_tokens if max_new_tokens is None else min(cfg.max_new_tokens, max_new_tokens)
        stamps: list[float] = []

        class Presence(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty = penalty

            def __call__(self, input_ids, scores):
                stamps.append(time.perf_counter())
                if self.penalty and input_ids.shape[1] > n_prompt:
                    gen = input_ids[:, n_prompt:]
                    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
                    scores = scores - self.penalty * mask.to(scores.dtype)
                return scores

        class Deadline(StoppingCriteria):
            hit = False

            def __call__(self, input_ids, scores, **kwargs):
                if deadline is not None and time.time() > deadline:
                    Deadline.hit = True
                    return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        s = cfg.sampling
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=max(1, budget), do_sample=True, temperature=s["temperature"], top_p=s["top_p"],
                top_k=s["top_k"], min_p=s["min_p"] or None, repetition_penalty=s["repetition_penalty"],
                logits_processor=LogitsProcessorList([Presence(s["presence_penalty"])]),
                stopping_criteria=StoppingCriteriaList([Deadline()]), eos_token_id=[self.im_end],
                pad_token_id=self.tokenizer.pad_token_id or self.im_end)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        gen_ids = out[0, n_prompt:].tolist()
        ended = bool(gen_ids) and gen_ids[-1] == self.im_end
        body = gen_ids[:-1] if ended else gen_ids
        text = self.tokenizer.decode(body, skip_special_tokens=False)
        reasoning, content = split_reasoning(text, cfg.thinking)
        if cfg.thinking and self.think_end in body:
            n_reason = body.index(self.think_end)
        elif cfg.thinking:
            n_reason = len(body)
        else:
            n_reason = 0
        finish = "eos" if ended else ("timeout" if Deadline.hit else "max_new_tokens")
        ttft = (stamps[0] - t0) if stamps else None
        decode = (seconds - ttft) if ttft is not None else None
        rate = ((len(gen_ids) - 1) / decode) if decode and len(gen_ids) > 1 else None
        return Turn(text=text, reasoning=reasoning, content=content, tool_calls=parse_tool_calls(content),
                    prompt_tokens=n_prompt, generated_tokens=len(gen_ids), reasoning_tokens=n_reason,
                    content_tokens=len(body) - n_reason, finish_reason=finish, seconds=seconds,
                    time_to_first_token=ttft, decode_tokens_per_second=rate)
