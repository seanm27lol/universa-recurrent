"""Compact CPU checks for pass-two learned-state writer suppression."""
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'research' / 'ti_tnn'))
from native import diagnose_learned_state as D  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.generation import greedy_generate  # noqa: E402
from native.model import NativeModel  # noqa: E402
from native.structured import Instruction, format_prompt  # noqa: E402


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        self.text = text
        return type('Encoded', (), {
            'ids': list(range(len(text))),
            'offsets': [(index, index + 1) for index in range(len(text))],
        })()

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.text[index] for index in ids)


def _model(mode):
    return NativeModel(NativeConfig(
        vocab_size=32,
        context_length=16,
        d_model=8,
        n_heads=2,
        d_slot=4,
        d_ff=16,
        mode=mode,
    )).cpu().eval()


def test_final_instruction_anchor_uses_last_overlapping_prompt_token():
    program = [Instruction('set', 0, value=7), Instruction('add', 1, src=2)]
    prompt = format_prompt(program, (1, 2, 3, 4))
    record = {
        'example_id': 'toy',
        'program': [str(instruction) for instruction in program],
        'prompt': prompt,
    }
    tokenizer = _CharacterTokenizer()
    anchor = D.final_instruction_anchor(record, tokenizer)

    start = prompt.index('add r1 r2\n')
    assert anchor.instruction_index == 1
    assert anchor.instruction == 'add r1 r2'
    assert anchor.token_index == start + len('add r1 r2') - 1
    assert anchor.token_id == anchor.token_index
    assert (anchor.instruction_char_start, anchor.instruction_char_end) == (
        start, start + len('add r1 r2'),
    )
    assert (anchor.token_char_start, anchor.token_char_end) == (
        anchor.token_index, anchor.token_index + 1,
    )


@pytest.mark.parametrize('mode', ['dense', 'untyped', 'typed'])
def test_writer_hook_suppresses_only_selected_pass_two_after_state(mode):
    torch.manual_seed(17)
    model = _model(mode)
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    anchor = 5
    with torch.no_grad():
        baseline = model(input_ids, return_trace=True)
        with D.suppress_after_state_update(model, anchor) as state:
            intervened = model(input_ids, return_trace=True)

    assert state['completed_forwards'] == 1
    assert state['targeted_forwards'] == 1
    assert state['suppressed_writer_updates'] == 4
    assert torch.equal(intervened.slots[:, :, 0], baseline.slots[:, :, 0])
    assert torch.equal(intervened.h[:, :, 1], baseline.h[:, :, 1])
    assert torch.equal(intervened.slots[:, anchor, 1], intervened.slots[:, anchor, 0])
    other_tokens = torch.arange(input_ids.shape[1]) != anchor
    assert torch.equal(
        intervened.slots[:, other_tokens, 1], baseline.slots[:, other_tokens, 1],
    )
    assert not torch.equal(baseline.slots[:, anchor, 1], baseline.slots[:, anchor, 0])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cached_suffix_and_restored_ordinary_path_have_no_direct_instrumentation():
    torch.manual_seed(23)
    model = _model('dense')
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    writers = [model.slot_writer]
    modules = [model, *writers]
    ordinary_pre_hooks = [dict(module._forward_pre_hooks) for module in modules]
    ordinary_hooks = [dict(module._forward_hooks) for module in modules]

    with torch.no_grad():
        baseline = greedy_generate(model, input_ids, 3, positions=positions, use_cache=True)
        assert '_write_slots' not in model.__dict__
        with D.suppress_after_state_update(model, 4) as state:
            greedy_generate(model, input_ids, 3, positions=positions, use_cache=True)
        restored = greedy_generate(model, input_ids, 3, positions=positions, use_cache=True)

    assert torch.equal(restored, baseline)
    assert state['forwards'] == state['completed_forwards'] == 3
    assert state['targeted_forwards'] == 1
    assert state['suppressed_writer_updates'] == 4
    assert state['local_index'] is None
    assert '_write_slots' not in model.__dict__
    assert [dict(module._forward_pre_hooks) for module in modules] == ordinary_pre_hooks
    assert [dict(module._forward_hooks) for module in modules] == ordinary_hooks
    assert all(parameter.grad is None for parameter in model.parameters())
