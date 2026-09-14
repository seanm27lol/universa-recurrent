"""Run the actual frozen-checkpoint and prior-report pipeline on a tiny CPU fixture."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_overflow_continuation as p
import benchmark_state_continuation as old
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_study import calibrate, load_estimators, write_json
from universa_recurrent.neural.v2_train import train_v2_checkpoint


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp('overflow')
    torch.set_num_threads(1)
    cp = root/'weights-7100.pt'
    train_v2_checkpoint(cp, device_name='cpu', seed=7100, train_size=16, calibration_size=8,
        epochs=1, batch_size=8, hidden_dim=8, candidate_embedding_dim=2, steps=8)
    engines, meta, dev, _ = load_estimators(cp, 'cpu')
    _, cal = calibrate(engines, StructuredFlowDataset(8, seed=7200), dev, .75, 8)
    cal['checkpoint_sha256'] = meta['checkpoint_sha256']
    cal['model_sources'] = {e.name:e.checkpoint_sha256 for e in engines if e.structured}
    claim = root/'study-7100/calibration.json'; write_json(claim, cal)
    previous = root/'previous'
    assert old.main(['--replication-dir', str(root), '--output-dir', str(previous), '--device', 'cpu',
        '--n', '12', '--calibration-size', '8', '--bits', '8', '12', '16']) == 0
    return root, cp, claim, previous


def test_full_cli_preserves_references_and_writes_per_example_arrays(experiment, tmp_path):
    root, cp, claim, previous = experiment
    before = tuple(p.sha256_file(f) for f in (cp, claim, previous/'seed-7100.json', previous/'seed-7100.npz'))
    output = tmp_path/'results'
    command = ['--replication-dir', str(root), '--previous-dir', str(previous),
               '--output-dir', str(output), '--device', 'cpu']
    assert p.main(command) == 0
    worker = p.read_json(output/'seed-7100.json')
    assert worker['all_exact_fallback_gates_passed'] and worker['no_refitting']
    assert len(worker['rows']) == 72 and len(worker['raw_gates']) == 6
    assert all(r['status'] == 'finite' for r in worker['rows'])
    with np.load(output/'seed-7100.npz', allow_pickle=False) as data:
        for r in worker['rows']:
            if r['mode'] != 'overflow_safe': continue
            key = f'{r["model"]}__cut{r["cut"]}__{r["method"]}'
            assert np.isclose(data[key+'__final_estimate_mse_to_reference'].mean(), r['metrics']['final_estimate_mse_to_reference'])
            assert r['clipped_or_fallback_subset']['claim_changes'] == 0
            assert r['storage']['fallback_n'] == r['fallback_n']
            assert r['executed_updates'] == r['depth'] - r['cut']
    assert before == tuple(p.sha256_file(f) for f in (cp, claim, previous/'seed-7100.json', previous/'seed-7100.npz'))
    assert p.main(command) == 2
    assert not list(output.glob('*.pt'))


def test_wrong_report_reference_rejected(experiment, tmp_path):
    _, cp, claim, previous = experiment
    report = p.read_json(previous/'seed-7100.json'); report['checkpoint_sha256'] = '0'*64
    path = tmp_path/'seed-7100.json'; write_json(path, report)
    with pytest.raises(ValueError, match='does not match'):
        p.worker(cp, claim, path, tmp_path/'out.json', 'cpu')


def test_numeric_archive_hash_rejected(tmp_path):
    f = tmp_path/'a.npz'; np.savez(f, x=np.ones(3))
    with pytest.raises(ValueError, match='hash mismatch'): p.safe_arrays(f, '0'*64)


def test_failed_case_cannot_be_ranked():
    good = dict(model='shared', cut=2, method='named_8bit_overflow_safe', status='finite',
        metrics=dict(final_estimate_mse_to_reference=0.01, max_absolute_estimate_change=.1, claim_index_changed_n=1))
    bad = {**good, 'status':'nonfinite_continuation', 'metrics':None}
    s = p.summarize([{'rows':[good], 'n':10}, {'rows':[bad], 'n':10}])[0]
    assert s['failed_checkpoints'] == 1 and s['mean_final_change'] is None and s['changed_claims'] is None


def test_missing_cli_and_references(tmp_path):
    assert p.main([]) == 2
    assert p.main(['--replication-dir',str(tmp_path), '--previous-dir',str(tmp_path), '--output-dir',str(tmp_path/'o')]) == 2
