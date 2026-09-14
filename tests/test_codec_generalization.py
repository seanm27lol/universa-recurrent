"""New examples must not silently refit ranges or change the continuation budget."""
from __future__ import annotations
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
p=importlib.import_module('benchmark_codec_generalization')


@pytest.fixture(scope='module')
def refs(tmp_path_factory):
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.dual_study import load_estimators, calibrate, write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    root=tmp_path_factory.mktemp('codec-heldout');rep=root/'rep';rep.mkdir()
    cp=rep/'weights-6100.pt';torch.set_num_threads(1)
    train_v2_checkpoint(cp,device_name='cpu',seed=6100,train_size=16,
        calibration_size=8,epochs=1,batch_size=8,hidden_dim=8,
        candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,meta,device,_=load_estimators(cp,'cpu')
    _,claim=calibrate(engines,StructuredFlowDataset(8,seed=31000),device,.75,8)
    claim['checkpoint_sha256']=meta['checkpoint_sha256']
    claim['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    cal=rep/'study-6100/calibration.json';write_json(cal,claim)
    prev=root/'previous'
    assert p.prior.main(['--replication-dir',str(rep),'--output-dir',str(prev),
        '--device','cpu','--n','8','--calibration-size','8'])==0
    return root,rep,prev,cp,cal


def worker_args(refs,tmp_path,**updates):
    root,rep,prev,cp,cal=refs
    args=dict(checkpoint=cp,claim_calibration=cal,previous_report=prev/'seed-6100.json',
        worker_output=tmp_path/'worker.json',device='cpu',data_seeds=[71000],n=8,
        conditions=list(p.CONDITIONS),models=list(p.MODELS))
    args.update(updates)
    return SimpleNamespace(**args)


@pytest.mark.parametrize('seed',[61000,62000,32000,46000,36000])
def test_rejects_old_evaluation_seeds(seed):
    with pytest.raises(ValueError,match='overlaps'):
        p.validate_seeds([seed])


@pytest.mark.parametrize('seeds',[[],[True],[1.2],[-1],[71000,71000]])
def test_bad_seed_lists(seeds):
    with pytest.raises(ValueError):p.validate_seeds(seeds)


def test_reserved_training_seed():
    with pytest.raises(ValueError):p.validate_seeds([71000],{71000})


@pytest.mark.parametrize('condition,noise,observe',[
    ('matched',.05,.7),('noise_x3',.15,.7),('sparse_040',.05,.4)])
def test_explicit_condition_parameters(condition,noise,observe):
    v=p.settings_for({'noise_std':.05,'observe_probability':.7},condition)
    assert v['noise_std']==pytest.approx(noise)
    assert v['observe_probability']==observe


def test_unknown_condition():
    with pytest.raises(ValueError):p.settings_for({},'unknown')


def test_conditions_pair_truth_not_independent_samples():
    ds=[p.StructuredFlowDataset(8,seed=71000,**p.settings_for(
        {'noise_std':.05,'observe_probability':.7},c)) for c in p.CONDITIONS]
    assert all(torch.equal(ds[0].truth,d.truth) and torch.equal(ds[0].label,d.label) for d in ds)
    assert not torch.equal(ds[0].observed,ds[1].observed)
    assert not torch.equal(ds[0].mask,ds[2].mask)


@pytest.fixture(scope='module')
def complete(refs,tmp_path_factory):
    out=tmp_path_factory.mktemp('complete-heldout')
    return p.worker(worker_args(refs,out))


def test_actual_worker_preserves_all_budget_and_reporting(complete):
    w=complete
    assert w['all_raw_restoration_gates_passed'] and w['no_refitting'] and w['no_training']
    assert len(w['cohorts'])==3 and len(w['raw_gates'])==18
    assert len(w['rows'])==108
    for row in w['rows']:
        assert row['status']=='finite'
        assert row['executed_updates']==row['depth']-row['cut']
        assert row['metrics']['n']==8
        if row['method']=='raw_state':
            assert row['metrics']['final_estimate_mse_to_reference']==0
            assert row['metrics']['claim_index_changed_n']==0
        if row['storage']:
            assert row['storage']['tag_bytes_per_state']==1
            assert row['storage']['fallback_n']==row['fallback_n']


def test_never_refits_a_codec(refs,tmp_path,monkeypatch):
    import continuation_codec
    def forbidden(*a,**k):pytest.fail('test data leaked into codec calibration')
    monkeypatch.setattr(continuation_codec.StateCodec,'fit',forbidden)
    result=p.worker(worker_args(refs,tmp_path,conditions=['matched'],models=['fixed_depth_4']))
    assert result['no_refitting']


@pytest.mark.parametrize('key,value',[('model','wrong'),('cut',7),('depth',99),('names',['x','y'])])
def test_wrong_manifest_site(refs,key,value):
    old=p.read_json(refs[2]/'seed-6100.json')
    model=SimpleNamespace(config=SimpleNamespace(steps=4,num_structures=2,latent_dim=2))
    bad=copy.deepcopy(old);bad['codecs']['fixed_depth_4__cut2__named_16bit']['site'][key]=value
    with pytest.raises(ValueError,match='site'):
        p.frozen_codec(bad,model,'fixed_depth_4',2,16,old['checkpoint_sha256'],
                       ['balanced_flow','wrong_diagonal_constraint'])


def test_old_checkpoint_or_policy_mismatch(refs,tmp_path):
    old=p.read_json(refs[2]/'seed-6100.json');old['checkpoint_sha256']='0'*64
    path=tmp_path/'old.json';p.write_json(path,old)
    with pytest.raises(ValueError,match='does not match'):
        p.worker(worker_args(refs,tmp_path,previous_report=path))


def test_new_seed_overlaps_custom_old_seed(refs,tmp_path):
    with pytest.raises(ValueError,match='overlaps'):
        p.worker(worker_args(refs,tmp_path,data_seeds=[6100]))


def test_duplicates_inside_new_cohort_fail(refs,tmp_path,monkeypatch):
    original=p.StructuredFlowDataset
    def duplicate(*a,**k):
        data=original(*a,**k);data.observed[1]=data.observed[0];data.mask[1]=data.mask[0];return data
    monkeypatch.setattr(p,'StructuredFlowDataset',duplicate)
    with pytest.raises(ValueError,match='duplicate inputs'):
        p.worker(worker_args(refs,tmp_path))


def test_numerical_failures_not_silently_averaged(complete):
    w=copy.deepcopy(complete);w['rows'][0].update(status='nonfinite_continuation',metrics=None)
    result=p.summarize([complete,w]);failed=[r for r in result if r['failed_checkpoints']]
    assert len(failed)==1 and failed[0]['failed_checkpoints']==1
    assert failed[0]['mean_final_mse'] is None and failed[0]['claims_changed'] is None


def test_mismatched_worker_cases_fail(complete):
    bad=copy.deepcopy(complete);bad['rows'].pop()
    with pytest.raises(ValueError,match='different cases'):p.summarize([complete,bad])


def test_cli_actual_parent_child_roundtrip(refs,tmp_path):
    args=['--replication-dir',str(refs[1]),'--previous-dir',str(refs[2]),
          '--output-dir',str(tmp_path/'result'),'--device','cpu','--n','4',
          '--data-seeds','71000','--conditions','matched','--models','fixed_depth_4']
    assert p.main(args)==0
    summary=p.read_json(tmp_path/'result/summary.json')
    assert summary['checkpoint_runs']==1 and summary['all_raw_restoration_gates_passed']
    assert summary['failed_cases']==0 and len(summary['rows'])==18
    assert len(list((tmp_path/'result').glob('*.npz')))==1
    assert not list((tmp_path/'result').rglob('*.pt'))
    assert p.main(args)==2


@pytest.mark.parametrize('args',[
    [],['--n','1'],['--data-seeds','62000'],['--models','shared','shared'],
    ['--conditions','matched','matched']])
def test_bad_cli_stops(args):assert p.main(args)==2


def test_nonfinite_control_kept_in_worker_report(refs,tmp_path,monkeypatch):
    original=p.prior.continue_from
    def continuation(model,state,restored,policy,**kwargs):
        if not np.any(restored):
            raise p.prior.NumericalContinuationError('synthetic nonfinite control')
        return original(model,state,restored,policy,**kwargs)
    monkeypatch.setattr(p.prior,'continue_from',continuation)
    result=p.worker(worker_args(refs,tmp_path,conditions=['matched'],models=['fixed_depth_4']))
    failed=[r for r in result['rows'] if r['status']!='finite']
    assert len(failed)==3
    assert all(r['method']=='zero_dynamic' and r['metrics'] is None for r in failed)


def test_raw_restoration_failure_is_fatal(refs,tmp_path,monkeypatch):
    original=p.prior.continue_from
    def continuation(*args,**kwargs):
        out,steps=original(*args,**kwargs)
        out['estimate']=out['estimate']+1
        return out,steps
    monkeypatch.setattr(p.prior,'continue_from',continuation)
    with pytest.raises(ValueError):
        p.worker(worker_args(refs,tmp_path,conditions=['matched'],models=['fixed_depth_4']))
