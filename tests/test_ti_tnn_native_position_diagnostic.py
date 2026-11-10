"""CPU-only checks for packed-position and causal token alignment."""
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'research' / 'ti_tnn'))
from native.data import EOS_TOKEN_ID, IGNORE_INDEX, SOURCE_IDS  # noqa: E402
from native.diagnose_positions import locate_packed_document  # noqa: E402


def test_packed_document_mapping_and_answer_alignment_cpu():
    arrays = {
        'input_ids': np.full((2, 8), EOS_TOKEN_ID, dtype=np.int32),
        'labels': np.full((2, 8), IGNORE_INDEX, dtype=np.int32),
        'positions': np.tile(np.arange(8, dtype=np.int32), (2, 1)),
        'doc_ids': np.full((2, 8), -1, dtype=np.int32),
        'key_padding_mask': np.ones((2, 8), dtype=np.bool_),
        'source_ids': np.full((2, 8), -1, dtype=np.int8),
    }
    prompt = (10, 11, 12)
    answer = (13, EOS_TOKEN_ID)
    selection = (1, np.arange(3, 8))
    arrays['input_ids'][selection] = prompt + answer
    arrays['labels'][selection] = (IGNORE_INDEX, *prompt[1:], *answer)
    arrays['doc_ids'][selection] = 7
    arrays['key_padding_mask'][selection] = False
    arrays['source_ids'][selection] = SOURCE_IDS['structured']

    aligned = locate_packed_document(arrays, 7, prompt, answer)

    assert (aligned.row, aligned.column, aligned.start_position) == (1, 3, 3)
    assert aligned.document_ids == prompt + answer
    assert aligned.other_target_ids == prompt[1:]
    assert aligned.answer_ids == answer
    assert aligned.metadata()['answer_tokens_including_eos'] == 2
    assert aligned.metadata()['other_tokens_excluding_document_start'] == 2

    arrays['positions'][1, 3] = 0
    with pytest.raises(ValueError, match='absolute column 3'):
        locate_packed_document(arrays, 7, prompt, answer)
