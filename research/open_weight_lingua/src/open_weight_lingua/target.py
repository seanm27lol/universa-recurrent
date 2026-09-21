"""Capture and replace one Qwen block-output vector, with no KV cache.

A full forward pass is recomputed. All other prompt states remain available;
this is neither whole-state replacement nor a fast pause/resume engine.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import torch

from .metrics import answer_tokens, sequence_log_probability

SUFFIX_DRIFT_BOUND = 1e-1
"""Frozen bound on cross-length relative drift at the scored prefix site.

Justified from smoke-test data and frozen before the pilot (brief §4:
investigate on smoke-test data and document a justified bound before the
pilot; never relax tolerances after inspecting validation results). In the
2026-09-21 GB10 smoke probe (run smoke-20260921T185833Z-c264a184, raw
measurements retained in the suffix-probe results), appending 1-5 answer
tokens moved the prefix-site vector by at most 3.09e-2 in
||drift||₂/||original||₂ terms: BF16 cuBLAS kernel re-selection on sm_121
changes the GEMM reduction order, and the reassociation compounds over
blocks. CPU fp32 collapses the drift to ~4e-6, and equal-length pairs stay
bitwise identical, so this bound covers only known kernel-reduction noise,
with a safety factor of about three over the measured maximum.
"""


@dataclass(frozen=True)
class Site:
    layer: int
    position: int
    row: int = 0


@dataclass
class ActivationRecord:
    vector: torch.Tensor
    site: Site
    token_ids: list[int]
    attention_mask: list[int]
    hook_path: str
    hidden_state_index: int

    @property
    def norm(self) -> float:
        return float(torch.linalg.vector_norm(self.vector.float()))


def last_nonpadding(attention_mask: torch.Tensor, row: int = 0) -> int:
    if attention_mask.ndim != 2 or not 0 <= row < attention_mask.shape[0]:
        raise ValueError("invalid attention-mask shape or row")
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("attention mask must be binary")
    positions = attention_mask[row].nonzero().flatten()
    if not len(positions):
        raise ValueError("all-padding input")
    return int(positions[-1])


def output_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("unsupported block-output structure")


def replace_output(output, tensor):
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        if hasattr(output, "_fields"):
            return type(output)(tensor, *output[1:])
        return (tensor, *output[1:])
    if hasattr(output, "last_hidden_state"):
        result = copy.copy(output)
        result.last_hidden_state = tensor
        if isinstance(result, dict):
            result["last_hidden_state"] = tensor
        return result
    raise TypeError("unsupported block-output structure")


def validate_site(model, ids, mask, site):
    if model.config.model_type != "qwen2":
        raise ValueError("only the audited Qwen2 architecture is supported")
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError("IDs and mask must have matching batch/sequence dimensions")
    last_nonpadding(mask, site.row)
    if not 0 <= site.layer < len(model.model.layers):
        raise ValueError("layer outside model")
    if not 0 <= site.position < ids.shape[1] or mask[site.row, site.position] != 1:
        raise ValueError("patch position is padding or outside prompt")


@contextmanager
def block_hook(model, site, *, replacement=None, captures=None, audit=False):
    """Scope a hook to one forward; preserve dtype, structure and other vectors."""

    def hook(_module, _args, output):
        hidden = output_tensor(output)
        if captures is not None:
            captures.append(hidden[site.row, site.position].detach().clone())
        if replacement is None:
            return None
        if replacement.shape != hidden[site.row, site.position].shape:
            raise ValueError("replacement width mismatch")
        if replacement.dtype != hidden.dtype or not torch.isfinite(replacement).all():
            raise ValueError("replacement must be finite and in native dtype")
        patched = hidden.clone()
        value = replacement.to(device=hidden.device)
        patched[site.row, site.position] = value
        if not torch.equal(patched[site.row, site.position], value):
            raise RuntimeError("native replacement was not installed exactly")
        if audit:
            unmodified = torch.ones(
                hidden.shape[:2], dtype=torch.bool, device=hidden.device
            )
            unmodified[site.row, site.position] = False
            if not torch.equal(patched[unmodified], hidden[unmodified]):
                raise RuntimeError("patch changed an unintended token")
        return replace_output(output, patched)

    handle = model.model.layers[site.layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class Target:
    def __init__(self, model, tokenizer):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer

    def tensors(self, ids, mask):
        device = next(self.model.parameters()).device
        return (
            torch.as_tensor(ids, dtype=torch.long, device=device),
            torch.as_tensor(mask, dtype=torch.long, device=device),
        )

    @torch.inference_mode()
    def forward(self, ids, mask, site=None, replacement=None, *, noop=False):
        if site is None:
            if replacement is not None or noop:
                raise ValueError("a hook requires an explicit site")
            return self.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
        validate_site(self.model, ids, mask, site)
        with block_hook(self.model, site, replacement=replacement, audit=True):
            return self.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits

    @torch.inference_mode()
    def capture(self, ids, mask, site: Site, *, check_index=True) -> ActivationRecord:
        validate_site(self.model, ids, mask, site)
        if site.layer == len(self.model.model.layers) - 1 and check_index:
            raise ValueError(
                "last block hidden-state entry includes final norm; unsupported index check"
            )
        captured = []
        with block_hook(self.model, site, captures=captured):
            out = self.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=check_index,
            )
        if len(captured) != 1:
            raise RuntimeError("capture hook must fire exactly once")
        if check_index and not torch.equal(
            captured[0], out.hidden_states[site.layer + 1][site.row, site.position]
        ):
            raise RuntimeError("block hook disagrees with hidden_states[layer+1]")
        return ActivationRecord(
            captured[0].cpu(),
            site,
            ids[site.row].tolist(),
            mask[site.row].tolist(),
            f"model.layers.{site.layer}",
            site.layer + 1,
        )

    def greedy(self, ids, mask, site, replacement=None, *, max_tokens=8):
        if ids.shape[0] != 1 or site.row != 0:
            raise ValueError("generation supports batch size one")
        # Remove trailing padding before appending. Keep the original patch index.
        length = last_nonpadding(mask) + 1
        work_ids, work_mask = ids[:, :length].clone(), mask[:, :length].clone()
        generated, terminated = [], False
        for _ in range(max_tokens):
            logits = self.forward(
                work_ids,
                work_mask,
                site if replacement is not None else None,
                replacement,
            )
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                terminated = True
                break
            work_ids = torch.cat((work_ids, work_ids.new_tensor([[token]])), dim=1)
            work_mask = torch.cat((work_mask, work_mask.new_ones((1, 1))), dim=1)
        visible = generated[:-1] if terminated else generated
        return {
            "text": self.tokenizer.decode(visible, skip_special_tokens=False),
            "token_ids": generated,
            "terminated": terminated,
        }

    def score_answer(self, ids, mask, site, answer, original_vector, replacement=None):
        if ids.shape[0] != 1 or site.row != 0:
            raise ValueError("scoring supports batch size one")
        suffix = answer_tokens(self.tokenizer, answer)
        length = last_nonpadding(mask) + 1
        extended = torch.cat((ids[:, :length], ids.new_tensor([suffix])), dim=1)
        extended_mask = torch.cat(
            (mask[:, :length], mask.new_ones((1, len(suffix)))), dim=1
        )
        # Capture before patching; appended answers may not leak into the prefix.
        observed = self.capture(extended, extended_mask, site, check_index=False).vector
        # Same-length causality gate: a deterministic dummy suffix of equal
        # length must reproduce the site vector bitwise. Equal-length forwards
        # select identical kernels, so any difference is genuine suffix-content
        # dependence, not reduction-order noise (measured bitwise zero across
        # eight same-length pairs in the 2026-09-21 GB10 smoke probe).
        dummy_id = getattr(self.tokenizer, "pad_token_id", None)
        if dummy_id is None:
            dummy_id = self.tokenizer.eos_token_id
        if dummy_id is None:
            dummy_id = 0
        dummy_suffix = [dummy_id] * len(suffix)
        if dummy_suffix == suffix:
            raise RuntimeError(
                "dummy suffix coincides with the answer suffix; "
                "the causality gate cannot discriminate a leak"
            )
        dummy = torch.cat((ids[:, :length], ids.new_tensor([dummy_suffix])), dim=1)
        # Same total length, so the extended mask applies unchanged.
        observed_dummy = self.capture(
            dummy, extended_mask, site, check_index=False
        ).vector
        if not torch.equal(observed, observed_dummy):
            raise RuntimeError("answer suffix changed the causal prefix activation")
        # Cross-length, only kernel-reduction drift is expected; the frozen
        # justified bound is documented on SUFFIX_DRIFT_BOUND.
        drift = float(
            torch.linalg.vector_norm(observed.float() - original_vector.float())
            / torch.linalg.vector_norm(original_vector.float())
        )
        if not drift <= SUFFIX_DRIFT_BOUND:
            raise RuntimeError(
                f"cross-length suffix drift {drift:.6g} exceeds the frozen "
                f"pre-pilot bound {SUFFIX_DRIFT_BOUND:.6g}"
            )
        logits = self.forward(
            extended,
            extended_mask,
            site if replacement is not None else None,
            replacement,
        )
        return {
            "token_ids_including_eos": suffix,
            "log_probability": sequence_log_probability(logits, length, suffix),
            "suffix_drift_relative": drift,
        }

    def identity_gate(self, ids, mask, site):
        record = self.capture(ids, mask, site)
        reference = self.forward(ids, mask)
        results = {}
        for name, logits in (
            ("repeat", self.forward(ids, mask)),
            ("noop", self.forward(ids, mask, site, noop=True)),
            ("raw_restore", self.forward(ids, mask, site, record.vector)),
        ):
            close = torch.allclose(
                reference.float(), logits.float(), atol=1e-5, rtol=1e-5
            )
            same_tokens = torch.equal(reference.argmax(-1), logits.argmax(-1))
            results[name] = {
                "bitwise_equal": torch.equal(reference, logits),
                "max_absolute_logit_error": float(
                    (reference.float() - logits.float()).abs().max()
                ),
                "within_tolerance": close,
                "argmax_equal": same_tokens,
            }
            if not close or not same_tokens:
                raise RuntimeError(f"identity gate failed: {name}: {results[name]}")
        original = self.greedy(ids, mask, site)
        restored = self.greedy(ids, mask, site, record.vector)
        if original != restored:
            raise RuntimeError("raw restoration changed greedy answer")
        return record, results, original
