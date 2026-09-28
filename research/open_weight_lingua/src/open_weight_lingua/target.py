"""Capture and replace one block-output vector, with no KV cache.

A full forward pass is recomputed. All other prompt states remain available;
this is neither whole-state replacement nor a fast pause/resume engine.
Every target forward is right-padded to the fixed TARGET_BUCKET length so
GEMM shapes never vary with sequence length (kernel-shape pinning).
Architecture-specific facts (model_type markers, block-list paths) come from
the audited registry in architectures.py; anything outside it fails closed.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import torch

from .architectures import decoder_layers, layers_dotted_path
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

2026-09-21 update, pre-pilot: that justification was falsified on the pilot
distribution (pilot-0000-A-x measured 2.58e-1, a prompt-conditioned heavy
tail the smoke sample missed). The bound is kept frozen as an untouched
backstop; the operative fix is TARGET_BUCKET kernel-shape pinning, which
removes cross-length drift by construction. Never re-fitted to pilot data.
"""

TARGET_BUCKET = 128
"""Fixed right-padded sequence length for every target-model forward.

Kernel-shape pinning, established pre-pilot on 2026-09-21 after the first
pilot attempt's identity gate failed on pilot-0000-A-x: unpadded BF16
cuBLAS kernel re-selection across sequence lengths moved the pinned prefix
site by up to 2.6e-1 relative on pilot prompts, flipping a near-tie greedy
argmax. fp32 collapses the difference to ~3e-6, so the prefix is
mathematically length-independent and the drift is pure kernel noise — but
with a prompt-conditioned heavy tail that SUFFIX_DRIFT_BOUND, justified on
smoke data only, does not cover. Padding every forward to one bucket pins
all GEMM shapes: same-shape forwards are bitwise deterministic on this
backend, and masked pad positions contribute exactly zero (finite-minimum
additive bias before a float32 softmax, bitwise-proven in the 2026-09-21
suffix probe). The longest pilot prompt is 79 tokens; 79 + 8 generation +
3 answer suffix = 90 <= 128. The runner checks the fit loudly before any
inference, and B never changes silently. AV/AR models are separate and keep
their own templates.
"""


def pad_to_bucket(ids, mask, bucket=TARGET_BUCKET):
    """Right-pad ids/mask (zeros, masked out) to the fixed bucket length."""
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError("IDs and mask must have matching batch/sequence dimensions")
    length = ids.shape[1]
    if length > bucket:
        raise ValueError(
            f"input length {length} exceeds the pinned bucket {bucket}"
        )
    if length == bucket:
        return ids, mask
    pad = bucket - length
    return (
        torch.cat((ids, ids.new_zeros((ids.shape[0], pad))), dim=1),
        torch.cat((mask, mask.new_zeros((mask.shape[0], pad))), dim=1),
    )


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
    layers = decoder_layers(model)  # fails closed on unaudited architectures
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError("IDs and mask must have matching batch/sequence dimensions")
    last_nonpadding(mask, site.row)
    if not 0 <= site.layer < len(layers):
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

    handle = decoder_layers(model)[site.layer].register_forward_hook(hook)
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
            ids, mask = pad_to_bucket(ids, mask)
            return self.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
        validate_site(self.model, ids, mask, site)
        ids, mask = pad_to_bucket(ids, mask)
        with block_hook(self.model, site, replacement=replacement, audit=True):
            return self.model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits

    @torch.inference_mode()
    def capture(self, ids, mask, site: Site, *, check_index=True) -> ActivationRecord:
        validate_site(self.model, ids, mask, site)
        if site.layer == len(decoder_layers(self.model)) - 1 and check_index:
            raise ValueError(
                "last block hidden-state entry includes final norm; unsupported index check"
            )
        bucket_ids, bucket_mask = pad_to_bucket(ids, mask)
        captured = []
        with block_hook(self.model, site, captures=captured):
            out = self.model(
                input_ids=bucket_ids,
                attention_mask=bucket_mask,
                use_cache=False,
                output_hidden_states=check_index,
            )
        if len(captured) != 1:
            raise RuntimeError("capture hook must fire exactly once")
        if check_index and not torch.equal(
            captured[0], out.hidden_states[site.layer + 1][site.row, site.position]
        ):
            raise RuntimeError("block hook disagrees with hidden_states[layer+1]")
        # The record describes the logical input; the bucket is mechanical.
        return ActivationRecord(
            captured[0].cpu(),
            site,
            ids[site.row].tolist(),
            mask[site.row].tolist(),
            f"{layers_dotted_path(self.model)}.{site.layer}",
            site.layer + 1,
        )

    def greedy(self, ids, mask, site, replacement=None, *, max_tokens=8):
        if ids.shape[0] != 1 or site.row != 0:
            raise ValueError("generation supports batch size one")
        # Pinned-shape generation: every step scores the same bucket, generated
        # tokens fill successive pad slots, and the argmax is read at the last
        # non-padding position. The patch stays at the original site index.
        work_ids, work_mask = pad_to_bucket(ids, mask)
        work_ids, work_mask = work_ids.clone(), work_mask.clone()
        position = last_nonpadding(work_mask)
        generated, terminated = [], False
        for _ in range(max_tokens):
            logits = self.forward(
                work_ids,
                work_mask,
                site if replacement is not None else None,
                replacement,
            )
            token = int(logits[0, position].argmax())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                terminated = True
                break
            if position + 1 >= TARGET_BUCKET:
                raise RuntimeError(
                    f"greedy generation exceeds the pinned bucket {TARGET_BUCKET}"
                )
            work_ids[0, position + 1] = token
            work_mask[0, position + 1] = 1
            position += 1
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
        ids, mask = pad_to_bucket(ids, mask)
        length = last_nonpadding(mask) + 1
        if length + len(suffix) > TARGET_BUCKET:
            raise RuntimeError(
                f"answer suffix on prefix length {length} exceeds the pinned "
                f"bucket {TARGET_BUCKET}"
            )
        # The suffix is written into pad slots; prefix positions keep their
        # indices, so sequence_log_probability scores the same absolute slots.
        extended = ids.clone()
        extended[0, length : length + len(suffix)] = ids.new_tensor(suffix)
        extended_mask = mask.clone()
        extended_mask[0, length : length + len(suffix)] = 1
        # Capture before patching; appended answers may not leak into the prefix.
        observed = self.capture(extended, extended_mask, site, check_index=False).vector
        # Same-length causality gate: a deterministic dummy suffix of equal
        # length must reproduce the site vector bitwise. Equal-shape forwards
        # select identical kernels, so any difference is genuine suffix-content
        # dependence, not reduction-order noise (measured bitwise zero across
        # eight same-length pairs in the 2026-09-21 GB10 smoke probe; under
        # TARGET_BUCKET pinning every forward is same-shape by construction).
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
        dummy = ids.clone()
        dummy[0, length : length + len(suffix)] = ids.new_tensor(dummy_suffix)
        # Same bucket shape and content length, so the extended mask applies.
        observed_dummy = self.capture(
            dummy, extended_mask, site, check_index=False
        ).vector
        if not torch.equal(observed, observed_dummy):
            raise RuntimeError("answer suffix changed the causal prefix activation")
        # Same-shape comparison against the prefix-only capture: kernel-shape
        # pinning makes the expected drift exactly zero. The frozen
        # SUFFIX_DRIFT_BOUND remains as an untouched backstop (never re-fitted
        # after its smoke-data justification was falsified on pilot prompts).
        drift = float(
            torch.linalg.vector_norm(observed.float() - original_vector.float())
            / torch.linalg.vector_norm(original_vector.float())
        )
        if not drift <= SUFFIX_DRIFT_BOUND:
            raise RuntimeError(
                f"prefix-site suffix drift {drift:.6g} exceeds the frozen "
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

    def greedy_identity(self, ids, mask, site, pinned_vector, unpatched=None, patched=None):
        """Backstop comparison of unpatched and pinned-patch greedy trajectories.

        Under TARGET_BUCKET kernel-shape pinning every forward shares one
        shape, so both paths are bitwise the same computation and "exact" is
        the expectation; a divergence signals backend nondeterminism or a
        hook/site bug, not tolerable drift. The drift-measurement path is
        retained from the pre-pinning repair purely as documented evidence:
        the unpatched site vector at the divergence-step sequence must stay
        within the frozen SUFFIX_DRIFT_BOUND of the pinned vector, and beyond
        the bound the gate raises. drift_diverged records are counted in run
        summaries, never hidden and never a pilot-decision input.
        """
        if unpatched is None:
            unpatched = self.greedy(ids, mask, site)
        if patched is None:
            patched = self.greedy(ids, mask, site, pinned_vector)
        if unpatched == patched:
            return {"status": "exact"}
        p0_ids, p1_ids = unpatched["token_ids"], patched["token_ids"]
        divergence_step = next(
            (
                index
                for index, (left, right) in enumerate(zip(p0_ids, p1_ids))
                if left != right
            ),
            min(len(p0_ids), len(p1_ids)),
        )
        # Rebuild the sequence both paths scored at the diverging step: the
        # prompt plus the shared generated prefix (greedy never appends EOS).
        # capture re-pads it to the bucket, so shapes stay pinned.
        length = last_nonpadding(mask) + 1
        prefix = [
            token
            for token in p0_ids[:divergence_step]
            if token != self.tokenizer.eos_token_id
        ]
        extended = torch.cat((ids[:, :length], ids.new_tensor([prefix])), dim=1)
        extended_mask = torch.cat(
            (mask[:, :length], mask.new_ones((1, len(prefix)))), dim=1
        )
        current = self.capture(extended, extended_mask, site, check_index=False).vector
        drift = float(
            torch.linalg.vector_norm(current.float() - pinned_vector.float())
            / torch.linalg.vector_norm(pinned_vector.float())
        )
        evidence = {
            "divergence_step": divergence_step,
            "p0_text": unpatched["text"],
            "p1_text": patched["text"],
            "p0_token_ids": p0_ids,
            "p1_token_ids": p1_ids,
            "drift_relative": drift,
        }
        if not drift <= SUFFIX_DRIFT_BOUND:
            raise RuntimeError(
                f"greedy trajectories diverged at step {divergence_step} with "
                f"prefix-site drift {drift:.6g} beyond the frozen pre-pilot "
                f"bound {SUFFIX_DRIFT_BOUND:.6g}"
            )
        return {"status": "drift_diverged", "evidence": evidence}

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
        comparison = self.greedy_identity(ids, mask, site, record.vector, original, restored)
        # Greedy trajectories compare whole generated token lists; bitwise_equal
        # means identical trajectories. Under kernel-shape pinning "exact" is
        # the expectation; drift_diverged is retained backstop evidence.
        results["greedy"] = {
            "bitwise_equal": comparison["status"] == "exact",
            **comparison,
        }
        return record, results, original
