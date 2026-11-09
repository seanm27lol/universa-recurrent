import pytest
import torch

from native.config import NativeConfig
from native.generation import greedy_generate
from native.model import NativeModel


MODES = ("dense", "shared", "untyped", "typed")


def make_model(mode):
    torch.manual_seed(41)
    return NativeModel(
        NativeConfig(
            vocab_size=64,
            context_length=32,
            d_model=16,
            n_heads=4,
            d_slot=8,
            d_ff=32,
            mode=mode,
            n_state_classes=5,
        )
    ).eval()


def concatenate(values):
    if values[0] is None:
        return None
    return torch.cat(values, dim=1)


@pytest.mark.parametrize("mode", MODES)
def test_full_and_incremental_agree_across_document_boundaries(mode):
    model = make_model(mode)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions = torch.tensor([[0, 1, 0, 1, 2]])
    docs = torch.tensor([[4, 4, 8, 8, 8]])
    kwargs = {"positions": positions, "doc_ids": docs}
    with torch.no_grad():
        full = model(ids, **kwargs, return_aux=True, return_trace=True)
        cache = None
        pieces = []
        for index in range(ids.shape[1]):
            step = model(
                ids[:, index : index + 1],
                positions=positions[:, index : index + 1],
                doc_ids=docs[:, index : index + 1],
                cache=cache,
                use_cache=True,
                return_aux=True,
                return_trace=True,
            )
            cache = step.cache
            pieces.append(step)
    for name in ("logits", "state_logits", "h", "slots", "route_logits", "route_probs"):
        expected = getattr(full, name)
        if expected is not None:
            actual = concatenate([getattr(piece, name) for piece in pieces])
            assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-6), name
    if full.route_indices is not None:
        actual = concatenate([piece.route_indices for piece in pieces])
        assert torch.equal(actual, full.route_indices)
    assert torch.equal(cache.doc_ids, docs)
    assert torch.equal(cache.positions, positions)
    assert cache.length == ids.shape[1]


@pytest.mark.parametrize("mode", MODES)
def test_cache_preserves_left_padding_mask(mode):
    model = make_model(mode)
    ids = torch.tensor([[0, 1, 2, 3]])
    positions = torch.tensor([[0, 0, 1, 2]])
    docs = torch.tensor([[7, 7, 7, 7]])
    mask = torch.tensor([[True, False, False, False]])
    with torch.no_grad():
        full = model(
            ids,
            positions=positions,
            doc_ids=docs,
            key_padding_mask=mask,
            return_trace=True,
        )
        cache = None
        logits = []
        hidden = []
        for index in range(ids.shape[1]):
            output = model(
                ids[:, index : index + 1],
                positions=positions[:, index : index + 1],
                doc_ids=docs[:, index : index + 1],
                key_padding_mask=mask[:, index : index + 1],
                cache=cache,
                use_cache=True,
                return_trace=True,
            )
            cache = output.cache
            logits.append(output.logits)
            hidden.append(output.h)
    assert torch.equal(cache.key_padding_mask, mask)
    assert torch.allclose(torch.cat(logits, dim=1), full.logits, atol=3e-6, rtol=3e-6)
    assert torch.allclose(torch.cat(hidden, dim=1), full.h, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode", MODES)
def test_new_cached_document_matches_isolated_execution(mode):
    model = make_model(mode)
    with torch.no_grad():
        prefix = model(
            torch.tensor([[1, 2]]),
            positions=torch.tensor([[0, 1]]),
            doc_ids=torch.tensor([[3, 3]]),
            use_cache=True,
        )
        cached = model(
            torch.tensor([[4, 5]]),
            positions=torch.tensor([[0, 1]]),
            doc_ids=torch.tensor([[9, 9]]),
            cache=prefix.cache,
            use_cache=True,
            return_trace=True,
        )
        isolated = model(
            torch.tensor([[4, 5]]),
            positions=torch.tensor([[0, 1]]),
            doc_ids=torch.tensor([[9, 9]]),
            return_trace=True,
        )
    assert torch.allclose(cached.logits, isolated.logits, atol=3e-6, rtol=3e-6)
    assert torch.allclose(cached.h, isolated.h, atol=3e-6, rtol=3e-6)
    assert torch.allclose(cached.slots, isolated.slots, atol=3e-6, rtol=3e-6)
    assert cached.cache.doc_ids.tolist() == [[3, 3, 9, 9]]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("use_cache", [False, True])
def test_complete_greedy_generation_smoke(mode, use_cache):
    model = make_model(mode)
    ids = torch.tensor([[1, 2], [3, 4]])
    generated = greedy_generate(model, ids, max_new_tokens=3, use_cache=use_cache)
    assert generated.shape == (2, 5)
    assert torch.equal(generated[:, :2], ids)
    reference = greedy_generate(model, ids, max_new_tokens=3, use_cache=not use_cache)
    assert torch.equal(generated, reference)


@pytest.mark.parametrize("use_cache", [False, True])
def test_greedy_generation_ignores_left_and_right_prompt_padding(use_cache):
    model = make_model("typed")
    padded = torch.tensor([[0, 1, 2], [1, 2, 0]])
    mask = torch.tensor([[True, False, False], [False, False, True]])
    generated = greedy_generate(
        model,
        padded,
        max_new_tokens=2,
        key_padding_mask=mask,
        use_cache=use_cache,
    )
    reference = greedy_generate(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        use_cache=use_cache,
    )
    assert torch.equal(generated[0, 1:], reference[0])
    assert torch.equal(generated[1, :2], reference[0, :2])
    assert torch.equal(generated[1, 3:], reference[0, 2:])


def test_model_generate_and_zero_length_completion():
    model = make_model("typed")
    ids = torch.tensor([[1, 2]])
    assert model.generate(ids, max_new_tokens=2).shape == (1, 4)
    assert torch.equal(greedy_generate(model, ids, max_new_tokens=0), ids)
