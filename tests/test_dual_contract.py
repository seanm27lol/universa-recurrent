"""Counterexamples before performance claims: numerical estimate != structural claim."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from universa_recurrent.neural.dual_output import (
    ClaimPolicy, FixedEstimator, construct_output, fit_coverage_policy, metrics,
)
from universa_recurrent.neural.dual_lingua import make_record, verify_record
from universa_recurrent.neural.dual_study import (
    load_estimators, collect, run_study, write_json, read_json, reserved_seeds,
    dataset_fingerprint, calibrate,
)
from universa_recurrent.neural.dual_cli import main, summarize_runs
from universa_recurrent.neural.v2_train import train_v2_checkpoint
from universa_recurrent.neural.data import StructuredFlowDataset


def example(p=0.6):
    # Two valid candidates on perpendicular lines. Their nontrivial mixture is
    # on neither line: this is the exact counterexample the interface must keep.
    return torch.tensor([[[2.,0.],[0.,2.]]]), torch.tensor([[p,1-p]])


def record(p=0.6, threshold=0.5):
    states, probabilities = example(p)
    policy = ClaimPolicy(threshold)
    out = construct_output(states, probabilities, policy)
    return make_record(out, policy, torch.zeros(1,2), torch.ones(1,2),
        ['horizontal','vertical'], torch.tensor([[[0.,1.]], [[1.,0.]]]),
        model_name='shared', checkpoint_sha256='a'*64, calibration_sha256='b'*64, trained_depth=2)


@pytest.mark.parametrize('threshold', [None,0.,0.5,0.6,0.8,1.])
def test_threshold_cannot_change_estimate(threshold):
    states, probabilities = example()
    output = construct_output(states, probabilities, ClaimPolicy(threshold))
    torch.testing.assert_close(output['estimate'], torch.tensor([[1.2,0.8]]))
    assert output['estimate'].data_ptr() != output['claim_state'].data_ptr()
    r = record(threshold=threshold)
    assert verify_record(r)['accepted']
    assert np.linalg.norm(np.array(r['library']['boundaries'][0]) @ np.array(r['estimate'])) > 0


def test_tied_candidates_first_index_and_cutoff_inclusive():
    r = record(p=0.5, threshold=0.5)
    assert r['structure_claim']['index'] == 0
    assert verify_record(r)['accepted']


@pytest.mark.parametrize('target', [0.,0.25,0.5,0.75,1.])
def test_coverage_rule_includes_all_ties(target):
    probabilities = torch.tensor([[0.9,0.1],[0.7,0.3],[0.7,0.3],[0.5,0.5]])
    p = fit_coverage_policy(probabilities, target)
    states = torch.ones(4,2,3)
    result = construct_output(states, probabilities, p)
    achieved = float(result['claim_mask'].float().mean())
    assert achieved >= target
    if target == 0:
        assert not result['claim_mask'].any()
    if target == .5:
        assert achieved == .75


@pytest.mark.parametrize('value', [-.1,1.1,float('nan'),float('inf'),True,'0.7'])
def test_invalid_coverage_and_cutoff(value):
    with pytest.raises(ValueError):
        ClaimPolicy(.5, value)
    with pytest.raises(ValueError):
        ClaimPolicy(value)


@pytest.mark.parametrize('p', [[[0.6,0.6]], [[-0.1,1.1]], [[float('nan'),0.5]], [[float('inf'),0.]], []])
def test_invalid_probabilities(p):
    states, _ = example()
    with pytest.raises(ValueError):
        construct_output(states, torch.tensor(p), ClaimPolicy(.5))


def test_no_claim_requested_is_not_full_abstention():
    states,p = example()
    row = metrics(construct_output(states,p,ClaimPolicy(1.),'mixture'),
                  torch.zeros(1,2),torch.tensor([0]),'mixture')
    assert row['claims_requested'] is False
    assert row['coverage'] is None and row['abstained_n'] is None


def test_zero_and_full_claim_denominators():
    states,p = example()
    rejected = metrics(construct_output(states,p,ClaimPolicy(None)),torch.zeros(1,2),
                       torch.tensor([0]),'mixture_with_claim')
    assert rejected['wrong_claim_rate_claimed'] is None
    assert rejected['wrong_claim_rate_all'] == 0.
    wrong = metrics(construct_output(states,p,ClaimPolicy(0.)),torch.zeros(1,2),
                    torch.tensor([1]),'mixture_with_claim')
    assert wrong['wrong_claim_n'] == 1
    assert wrong['wrong_claim_rate_claimed'] == wrong['wrong_claim_rate_all'] == 1
    assert wrong['estimate_mse_on_abstained'] is None
    assert wrong['claim_candidate_mse_on_claimed'] != wrong['estimate_mse_on_claimed']


@pytest.mark.parametrize('mutation', [
    lambda r: r['estimate'].__setitem__(0, 10.),
    lambda r: r['scope'].__setitem__('estimate_has_structural_certificate', True),
    lambda r: r['scope'].__setitem__('neural_update_replayed', 0),
    lambda r: r['probabilities'].__setitem__(0, -0.1),
    lambda r: r['probabilities'].__setitem__(0, float('nan')),
    lambda r: r['candidate_states'][0].__setitem__(1, 1.),
    lambda r: r['structure_claim'].__setitem__('index', 1),
    lambda r: r['structure_claim'].__setitem__('index', False),
    lambda r: r['structure_claim'].__setitem__('name','vertical'),
    lambda r: r['structure_claim'].__setitem__('state',r['estimate']),
    lambda r: r['policy'].__setitem__('threshold',.99),
    lambda r: r['policy'].__setitem__('threshold',True),
    lambda r: r['policy'].__setitem__('target_coverage',None),
    lambda r: r['mask'].__setitem__(0, 0.2),
    lambda r: r.__setitem__('trained_depth',True),
    lambda r: r.__setitem__('arbitrary_extra_claim',True),
    lambda r: r['library']['names'].__setitem__(1,'horizontal'),
    lambda r: r.__setitem__('checkpoint_sha256','not a hash'),
])
def test_tampered_records_rejected(mutation):
    r = record()
    mutation(r)
    assert not verify_record(r)['accepted']


def test_abstention_cannot_have_a_claim():
    r = record(threshold=1.)
    r['structure_claim'] = {'index':0,'name':'horizontal','state':[2.,0.]}
    assert not verify_record(r)['accepted']


def test_json_nonfinite_duplicate_and_overwrite(tmp_path):
    path = tmp_path/'x.json'
    write_json(path, {'a':1})
    with pytest.raises(FileExistsError):
        write_json(path,{'a':2})
    assert read_json(path) == {'a':1}
    for text in ('{"a":NaN}', '{"a":1,"a":2}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp('dual-weights')/'v2.pt'
    train_v2_checkpoint(path,device_name='cpu',seed=401,train_size=48,calibration_size=24,
        epochs=1,batch_size=16,hidden_dim=12,candidate_embedding_dim=4,steps=2)
    return path


def test_all_estimators_share_output_contract_and_no_padding_leak(checkpoint):
    engines,_,device,_ = load_estimators(checkpoint,'cpu')
    data = StructuredFlowDataset(8, seed=7001)
    changed = data.observed + (1-data.mask)*999
    for engine in engines:
        if not engine.structured:
            continue
        a = collect(engine,data.observed,data.mask,ClaimPolicy(.5),'mixture_with_claim',4)
        b = collect(engine,changed,data.mask,ClaimPolicy(.99),'mixture_with_claim',4)
        torch.testing.assert_close(a['estimate'],b['estimate'],rtol=1e-5,atol=1e-6)
        validate = verify_record(record())
        assert validate['accepted']


def test_calibration_same_rule_for_every_structured_model(checkpoint):
    engines,_,device,_ = load_estimators(checkpoint,'cpu')
    dataset = StructuredFlowDataset(24,seed=7002)
    policies, report = calibrate(engines,dataset,device,.75,12)
    assert set(policies) == {e.name for e in engines if e.structured}
    assert all(r['calibration_metrics']['coverage'] >= .75 for r in report['models'].values())
    assert report['data']['sha256'] == dataset_fingerprint(dataset)


@pytest.mark.parametrize('seed', [402,403])
def test_study_refuses_trained_or_calibrated_seed(checkpoint,tmp_path,seed):
    with pytest.raises(ValueError,match='overlaps'):
        run_study(checkpoint,tmp_path/'bad',device_name='cpu',test_seed=seed,repeats=0)
    assert not (tmp_path/'bad').exists()


def test_study_refuses_same_test_calibration_seed(checkpoint,tmp_path):
    with pytest.raises(ValueError,match='differ'):
        run_study(checkpoint,tmp_path/'bad',calibration_seed=7000,test_seed=7000)


@pytest.mark.parametrize('name,value', [('repeats',-1),('warmup',True),('batch_size',0),('n',False)])
def test_invalid_run_arguments_do_not_create_results(checkpoint,tmp_path,name,value):
    with pytest.raises(ValueError):
        run_study(checkpoint,tmp_path/'bad',**{name:value})
    assert not (tmp_path/'bad').exists()


def test_full_compare_roundtrip_same_timed_quality_and_bound_record(checkpoint,tmp_path,monkeypatch):
    destination=tmp_path/'study'
    assert main(['compare','--checkpoint',str(checkpoint),'--output-dir',str(destination),
        '--device','cpu','--calibration-seed','7100','--test-seed','7101',
        '--calibration-size','24','--n','24','--batch-size','12','--warmup','0','--repeats','2']) == 0
    evaluation=read_json(destination/'eval.json')
    benchmark=read_json(destination/'benchmark.json')
    assert evaluation['dataset'] == benchmark['dataset']
    assert benchmark['timed_output_verified_against_evaluation']
    quality={r['model']+'/'+r['output_mode']:r for r in evaluation['rows']}
    for row in benchmark['rows']:
        ref=quality[row['model']+'/'+row['output_mode']]
        assert row['estimate_mse'] == ref['estimate_mse']
        assert row.get('coverage') == ref.get('coverage')
        assert len(row['samples_ms']) == 2
    for model in {r['model'] for r in evaluation['rows']}:
        if model+'/mixture' in quality:
            assert quality[model+'/mixture']['estimate_mse'] == quality[model+'/mixture_with_claim']['estimate_mse']
    from universa_recurrent.neural.v2 import MultiHypothesisRecurrentNet
    def forbidden(*args,**kwargs):
        raise AssertionError('checker must not replay the neural model')
    monkeypatch.setattr(MultiHypothesisRecurrentNet,'rollout',forbidden)
    r=read_json(destination/'lingua.json')
    assert verify_record(r,checkpoint=checkpoint,calibration=destination/'calibration.json')['accepted']
    altered=copy.deepcopy(r);altered['calibration_sha256']='f'*64
    assert not verify_record(altered,calibration=destination/'calibration.json')['accepted']
    with pytest.raises(ValueError,match='overwrite'):
        run_study(checkpoint,destination)
    assert read_json(destination/'COMPLETED.json')['verified']


def test_replicate_seed_collision_rejected_before_files(tmp_path):
    assert main(['replicate','--output-dir',str(tmp_path/'no'),'--seeds','100','101']) == 2
    assert not (tmp_path/'no').exists()


def test_reservation_requires_metadata():
    with pytest.raises(ValueError):
        reserved_seeds({})


def test_summary_refuses_missing_replication_identity(tmp_path):
    p=tmp_path/'run';p.mkdir()
    write_json(p/'eval.json', {'dataset':{'sha256':'x'},'rows':[]})
    write_json(p/'PLAN.json', {'checkpoint_sha256':'a'})
    write_json(p/'COMPLETED.json',{'verified':True})
    with pytest.raises(ValueError,match='training_seed'):
        summarize_runs([p])


def test_extreme_finite_geometry_does_not_pass_via_infinite_tolerance():
    r = record()
    r['candidate_states'] = [[1e300,1e300],[0.,2.]]
    r['estimate'] = [6e299,6e299]
    with np.errstate(over='ignore', invalid='ignore'):
        assert not verify_record(r)['accepted']


def test_metrics_fail_closed_on_squared_error_overflow():
    with pytest.raises(ValueError,match='derived'):
        metrics({'estimate':torch.tensor([[1e300]],dtype=torch.float64)},
                torch.tensor([[0.]],dtype=torch.float64),torch.tensor([0]),'estimate_only')
