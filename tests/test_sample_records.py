from pathlib import Path
import pytest
from universa_recurrent.lingua import load_record
from universa_recurrent.verification import verify_record


@pytest.mark.parametrize('mode', ['full', 'compact'])
def test_retained_sample_record(mode):
    root = Path(__file__).resolve().parents[1]
    record = load_record(root / 'examples' / 'traces' / f'flow_{mode}.json')
    check = verify_record(record)
    assert check.accepted
    assert check.intermediate_checks == (35 if mode == 'full' else 0)
