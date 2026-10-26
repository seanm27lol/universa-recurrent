"""The Qwen runtime for the ordinary baselines (B4, Q27) and the frozen backbone: pinned loading, per-turn generation,
tool-call parsing and exact token, time and memory accounting.

**Example.** Q27 receives a question, its schema and three retrieved training examples, with three tools available. In
thinking mode the chat template ends the prompt with `<think>\\n`; the model writes its reasoning, then `</think>`,
then a `<tool_call>` block or a final answer, and ends the turn with `<|im_end|>`.

For each turn, `Runtime.generate_turn` returns:
- the text, split into reasoning and content, and the parsed tool calls;
- this turn's prompt (prefill) tokens, generated tokens, and the reasoning/content split;
- the finish reason: `eos`, `max_new_tokens` (truncated), `timeout` or `budget`;
- the time to first token, measured with a GPU synchronisation, and the decode rate.

**Token counting:**
- The generated count includes the stop token.
- In thinking mode, everything before `</think>` is reasoning. `</think>` and what follows it count as content.

**Pinned loading:**
- Weights, tokenizer and chat template come from the local cache at the pinned revision; the files are checked
  against `pins/models_v1.json`. There is no network access.
- The precision is BF16. Any other precision would be a separately named configuration, never a silent substitute.
- A GPU model can only be loaded inside a `ledger.Reservation`.

**Linear-attention kernels.** Qwen3.5 and Qwen3.8 put three gated-delta-rule (linear attention) layers before every
full-attention layer. transformers chooses their kernel when it is imported:
- `flash-linear-attention` (Triton), when the package imports;
- otherwise its reference PyTorch implementation, which is correct but much slower.

The configuration is named by `TI_TNN_SCALING_LINEAR_KERNELS`:
- `reference`, the default, which blocks the `fla` import so the reference path runs;
- `fla`, which uses flash-linear-attention 0.6.0.

**The 4B profiles chose `reference`:**
- prefill held steady at 2,500–2,700 tokens/s from 1k to 28.7k tokens;
- `fla` took 6.7 s at 4k and 13.5 s at 28.7k against 1.6 s and 11.5 s, apparently compiling its Triton kernels for
  every new sequence length;
- decode was only about 4% faster with `fla`;
- the fixed-prompt logits agree: identical top-20 set, total-variation distance 0.014.

The backend that actually ran is recorded. Causal conv1d always uses the reference implementation, because
`causal-conv1d` has no wheel for this platform.

**Sampling** follows the official model cards, keyed by (model, thinking), and is recorded in every configuration.
Presence penalty follows the OpenAI/vLLM convention: subtract it once from the logit of every token already
generated in this turn.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field

from . import contracts as C
from . import identity, ledger

KERNELS = os.environ.get("TI_TNN_SCALING_LINEAR_KERNELS", "reference")  # chosen after the 4B profiles
if KERNELS not in ("fla", "reference"):
    raise RuntimeError(f"TI_TNN_SCALING_LINEAR_KERNELS must be 'fla' or 'reference', not {KERNELS!r}")
if KERNELS == "reference":
    sys.modules["fla"] = None  # transformers then selects its reference PyTorch implementation


def linear_attention_backend() -> dict:
    """What transformers will use for the gated delta rule and causal conv1d in this process."""
    blocked = "fla" in sys.modules and sys.modules["fla"] is None
    fla = (not blocked) and importlib.util.find_spec("fla") is not None
    version = None
    if fla:
        import importlib.metadata as md
        version = md.version("flash-linear-attention")
    return {"configuration": KERNELS, "gated_delta_rule": f"flash-linear-attention {version}" if fla else "reference",
            "causal_conv1d": "reference"}


def bound_linear_attention() -> dict:
    """What transformers actually bound for Qwen3.5's linear-attention functions, read from the kernel decorator's
    closure. Call it after the model is loaded. It reports what ran, whatever the import order was."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    out = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        fn = getattr(m, name, None)
        cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ())))) if fn else {}
        impl = cells.get("implementation")
        out[name] = (f"{impl.__module__}.{impl.__name__}" if impl is not None else "unknown") + \
            (" (optimized)" if cells.get("is_new_implementation") else " (reference)")
    return out


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
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass
class GenerationConfig:
    model: str                      # "Q27" | "B4"
    thinking: bool
    max_new_tokens: int
    reasoning_effort: str | None = None   # Q27 thinking only: "xhigh" | "medium" | "low" (None: the template default)
    sampling: dict = field(default_factory=dict)

    @classmethod
    def official(cls, model: str, thinking: bool, limits: C.Limits, reasoning_effort: str | None = None):
        if reasoning_effort is not None and (model != "Q27" or not thinking):
            raise ValueError("reasoning_effort is a Qwen3.8 template option and applies only in thinking mode")
        return cls(model=model, thinking=thinking, reasoning_effort=reasoning_effort,
                   max_new_tokens=limits.max_new_tokens_thinking if thinking else limits.max_new_tokens_nonthinking,
                   sampling=dict(SAMPLING[(model, thinking)]))

    def template_options(self) -> dict:
        """The chat-template options as they actually apply (template defaults made explicit)."""
        out = {"enable_thinking": self.thinking}
        if self.model == "Q27":
            out["reasoning_effort"] = (self.reasoning_effort or "xhigh") if self.thinking else "not applied (thinking off)"
            out["preserve_thinking"] = True  # the Qwen3.8 template's default; earlier reasoning is kept in context
        else:
            out["reasoning_effort"] = "not supported by the Qwen3.5 template"
            out["preserve_thinking"] = "not supported by the Qwen3.5 template (only the latest turn's reasoning is kept)"
        return out

    def as_dict(self) -> dict:
        return {**asdict(self), "template_options": self.template_options()}


@dataclass
class Turn:
    text: str
    reasoning: str
    content: str
    tool_calls: list
    prompt_tokens: int              # this turn's prefill (the whole conversation so far)
    generated_tokens: int           # includes the stop token, if one was generated
    reasoning_tokens: int
    content_tokens: int
    finish_reason: str              # "eos" | "max_new_tokens" | "timeout" | "budget"
    stop_token: str | None
    seconds: float
    time_to_first_token: float | None
    decode_tokens_per_second: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def parse_tool_calls(content: str) -> list[dict]:
    """Qwen's XML tool-call format (`qwen3_coder`): <tool_call><function=NAME><parameter=P>V</parameter>...</function></tool_call>."""
    return [{"name": name, "arguments": {k: v for k, v in PARAMETER.findall(body)}}
            for name, body in TOOL_CALL.findall(content)]


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
    """One loaded model. On a GPU it must be loaded inside the caller's reservation, so loading is charged."""

    def __init__(self, key: str, device: str = "cuda", attn_implementation: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        if device.startswith("cuda") and ledger.active() is None:
            raise RuntimeError("load a GPU model only inside a ledger.Reservation")
        self.key, self.device = key, device
        self.model_identity = identity.model_identity(key)
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
        self.stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        tpl = self.snapshot / "chat_template.jinja"
        self.chat_template_sha256 = identity.sha256_file(tpl) if tpl.exists() else None

    def describe(self) -> dict:
        import torch
        out = {"key": self.key, "repo": C.MODELS[self.key]["repo"], "revision": C.MODELS[self.key]["revision"],
               "dtype": self.dtype, "parameters": self.parameters, "load_seconds": self.load_seconds,
               "chat_template_sha256": self.chat_template_sha256, "model_class": type(self.model).__name__,
               "linear_attention": linear_attention_backend(), "linear_attention_bound": bound_linear_attention(),
               "stop_tokens": dict(zip(STOP_TOKENS, self.stop_ids))}
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
        """One assistant turn. `deadline` is a time.monotonic() limit for the whole request."""
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        prompt = self.render(messages, tools, cfg)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        budget = cfg.max_new_tokens if max_new_tokens is None else min(cfg.max_new_tokens, max_new_tokens)
        if budget <= 0:
            return Turn("", "", "", [], n_prompt, 0, 0, 0, "budget", None, 0.0, None, None)
        cuda = self.device.startswith("cuda")
        stamps: list[float] = []

        class Presence(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty = penalty

            def __call__(self, input_ids, scores):
                if not stamps and cuda:
                    torch.cuda.synchronize()  # the prefill is done once its logits exist
                stamps.append(time.perf_counter())
                if self.penalty and input_ids.shape[1] > n_prompt:
                    gen = input_ids[:, n_prompt:]
                    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
                    scores = scores - self.penalty * mask.to(scores.dtype)
                return scores

        class Deadline(StoppingCriteria):
            hit = False

            def __call__(self, input_ids, scores, **kwargs):
                if deadline is not None and time.monotonic() > deadline:
                    Deadline.hit = True
                    return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        s = cfg.sampling
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=budget, do_sample=True, temperature=s["temperature"], top_p=s["top_p"],
                top_k=s["top_k"], min_p=s["min_p"] or None, repetition_penalty=s["repetition_penalty"],
                logits_processor=LogitsProcessorList([Presence(s["presence_penalty"])]),
                stopping_criteria=StoppingCriteriaList([Deadline()]), eos_token_id=self.stop_ids,
                pad_token_id=self.tokenizer.pad_token_id or self.stop_ids[0])
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        gen_ids = out[0, n_prompt:].tolist()
        stop = gen_ids[-1] if gen_ids and gen_ids[-1] in self.stop_ids else None
        body = gen_ids[:-1] if stop is not None else gen_ids
        text = self.tokenizer.decode(body, skip_special_tokens=False)
        reasoning, content = split_reasoning(text, cfg.thinking)
        if cfg.thinking and self.think_end in body:
            n_reason = body.index(self.think_end)
        elif cfg.thinking:
            n_reason = len(body)
        else:
            n_reason = 0
        finish = "eos" if stop is not None else ("timeout" if Deadline.hit else "max_new_tokens")
        ttft = (stamps[0] - t0) if stamps else None
        decode = (seconds - ttft) if ttft is not None else None
        rate = ((len(gen_ids) - 1) / decode) if decode and len(gen_ids) > 1 else None
        return Turn(text=text, reasoning=reasoning, content=content, tool_calls=parse_tool_calls(content),
                    prompt_tokens=n_prompt, generated_tokens=len(gen_ids), reasoning_tokens=n_reason,
                    content_tokens=len(body) - n_reason, finish_reason=finish,
                    stop_token=self.tokenizer.convert_ids_to_tokens(stop) if stop is not None else None,
                    seconds=seconds, time_to_first_token=ttft, decode_tokens_per_second=rate)
