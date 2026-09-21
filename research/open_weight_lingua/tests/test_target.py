import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput
from open_weight_lingua.target import (
    SUFFIX_DRIFT_BOUND,
    ActivationRecord,
    Target,
    Site,
    block_hook,
    last_nonpadding,
    output_tensor,
    replace_output,
)


def test_index_native_restoration_two_inputs(model_factory, tokenizer):
    target = Target(model_factory(dtype=torch.bfloat16), tokenizer)
    records = []
    for tokens in ([3, 4, 5], [6, 7, 8]):
        ids, mask = target.tensors([tokens], [[1, 1, 1]])
        record, checks, _ = target.identity_gate(ids, mask, Site(1, 2))
        assert record.hidden_state_index == 2
        assert record.vector.dtype == torch.bfloat16
        assert all(item["bitwise_equal"] for item in checks.values())
        assert not target.model.model.layers[1]._forward_hooks
        records.append(record.vector)
    assert not torch.equal(*records)


def test_single_site_and_other_rows_unchanged(model_factory):
    model = model_factory()
    site = Site(1, 2, row=1)
    captured = []
    before = model.model.layers[1].register_forward_hook(
        lambda _, __, out: captured.append(output_tensor(out).clone())
    )
    replacement = torch.ones(16)
    with (
        torch.inference_mode(),
        block_hook(model, site, replacement=replacement, audit=True),
    ):
        after = model.model.layers[1].register_forward_hook(
            lambda _, __, out: captured.append(output_tensor(out).clone())
        )
        model(input_ids=torch.tensor([[3, 4, 5], [5, 6, 7]]), use_cache=False)
        after.remove()
    before.remove()
    changed = (captured[0] != captured[1]).any(-1)
    assert changed.nonzero().tolist() == [[1, 2]]
    assert torch.equal(captured[1][1, 2], replacement)


def test_exception_cleanup(model_factory):
    model = model_factory()
    with pytest.raises(RuntimeError, match="intentional"):
        with block_hook(model, Site(1, 0)):
            raise RuntimeError("intentional")
    assert not model.model.layers[1]._forward_hooks
    with pytest.raises(ValueError, match="width"):
        with block_hook(model, Site(1, 0), replacement=torch.ones(7)):
            model(input_ids=torch.tensor([[3, 4]]), use_cache=False)
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize(
    "mask,expected", [([0, 0, 1, 1], 3), ([1, 1, 0, 0], 1), ([1, 0, 1, 0], 2)]
)
def test_padding(mask, expected, model_factory, tokenizer):
    tensor = torch.tensor([mask])
    assert last_nonpadding(tensor) == expected
    target = Target(model_factory(), tokenizer)
    record = target.capture(torch.tensor([[3, 4, 5, 6]]), tensor, Site(1, expected))
    assert record.attention_mask == mask


def test_bad_site_and_dtype(model_factory, tokenizer):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 0]], [[1, 1, 0]])
    for site in (Site(1, 2), Site(3, 0), Site(1, -1), Site(1, 1, 2)):
        with pytest.raises(ValueError):
            target.capture(ids, mask, site)
    with pytest.raises(ValueError, match="native dtype"):
        target.forward(ids, mask, Site(1, 1), torch.ones(16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="all-padding"):
        last_nonpadding(torch.zeros((1, 3)))


def test_tuple_and_model_output_preserved():
    tensor = torch.zeros(1, 2, 16)
    replacement = torch.ones_like(tensor)
    marker = object()
    assert replace_output((tensor, marker), replacement)[1] is marker
    original = BaseModelOutput(last_hidden_state=tensor, hidden_states=(tensor,))
    out = replace_output(original, replacement)
    assert type(out) is BaseModelOutput
    assert out[0] is replacement and out.hidden_states is original.hidden_states
    assert original.last_hidden_state is tensor


def test_answer_suffix_is_causal_and_patch_stays_at_prefix(model_factory, tokenizer):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))
    score = target.score_answer(
        ids, mask, record.site, "19", record.vector, record.vector
    )
    assert score["token_ids_including_eos"] == [11, 19, 2]
    assert score["log_probability"] < 0
    seen = []
    handle = target.model.model.layers[1].register_forward_hook(
        lambda _, __, out: seen.append(output_tensor(out).detach().clone())
    )
    target.greedy(ids, mask, record.site, record.vector, max_tokens=3)
    handle.remove()
    assert len(seen) >= 1
    # Prefix's pre-patch causal vector remains fixed as suffix grows.
    assert all(torch.allclose(row[0, 2], record.vector) for row in seen)


def test_score_answer_records_suffix_drift(model_factory, tokenizer):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))
    score = target.score_answer(ids, mask, record.site, "19", record.vector)
    assert score["token_ids_including_eos"] == [11, 19, 2]
    # CPU fixture drift is far below the frozen justified bound.
    assert 0 <= score["suffix_drift_relative"] < 1e-3


def _fake_record(vector, ids, mask, site):
    return ActivationRecord(
        vector, site, ids[0].tolist(), mask[0].tolist(), "fake", site.layer + 1
    )


def test_same_length_dummy_gate_rejects_suffix_dependence(
    model_factory, tokenizer, monkeypatch
):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))

    def leaking_capture(capture_ids, capture_mask, site, *, check_index=True):
        # The captured vector depends on suffix content: a genuine causal leak.
        vector = torch.full((16,), float(sum(capture_ids[0].tolist())))
        return _fake_record(vector, capture_ids, capture_mask, site)

    monkeypatch.setattr(target, "capture", leaking_capture)
    with pytest.raises(
        RuntimeError, match="answer suffix changed the causal prefix activation"
    ):
        target.score_answer(ids, mask, record.site, "19", record.vector)


def test_suffix_drift_bound_rejects_excess_drift(
    model_factory, tokenizer, monkeypatch
):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))

    def drifting_capture(capture_ids, capture_mask, site, *, check_index=True):
        # Equal-length captures agree bitwise but sit far from the original.
        vector = torch.zeros_like(record.vector)
        return _fake_record(vector, capture_ids, capture_mask, site)

    monkeypatch.setattr(target, "capture", drifting_capture)
    with pytest.raises(RuntimeError, match="suffix drift"):
        target.score_answer(ids, mask, record.site, "19", record.vector)


def test_suffix_drift_within_bound_passes(model_factory, tokenizer, monkeypatch):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))

    def kernel_noise_capture(capture_ids, capture_mask, site, *, check_index=True):
        vector = record.vector * 1.01
        return _fake_record(vector, capture_ids, capture_mask, site)

    monkeypatch.setattr(target, "capture", kernel_noise_capture)
    score = target.score_answer(ids, mask, record.site, "19", record.vector)
    assert score["suffix_drift_relative"] == pytest.approx(0.01, rel=1e-3)
    assert score["suffix_drift_relative"] <= SUFFIX_DRIFT_BOUND
    assert score["token_ids_including_eos"] == [11, 19, 2]
    assert score["log_probability"] < 0


def test_dummy_suffix_token_resolution(model_factory, tokenizer, monkeypatch):
    target = Target(model_factory(), tokenizer)
    ids, mask = target.tensors([[3, 4, 5]], [[1, 1, 1]])
    record = target.capture(ids, mask, Site(1, 2))
    calls = []

    def recording_capture(capture_ids, capture_mask, site, *, check_index=True):
        calls.append(capture_ids[0].tolist())
        return _fake_record(record.vector, capture_ids, capture_mask, site)

    monkeypatch.setattr(target, "capture", recording_capture)
    target.score_answer(ids, mask, record.site, "19", record.vector)
    assert calls[0][-3:] == [11, 19, 2]
    # TinyTokenizer has no pad token; the dummy falls back to EOS.
    assert calls[1][-3:] == [2, 2, 2]
    calls.clear()
    tokenizer.pad_token_id = 7
    target.score_answer(ids, mask, record.site, "19", record.vector)
    assert calls[1][-3:] == [7, 7, 7]
