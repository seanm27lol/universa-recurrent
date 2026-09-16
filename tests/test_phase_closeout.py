"""Final validation must not quietly change the target or hide a negative result."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib
import json
import sys
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
close = importlib.import_module('phase_closeout')


def test_locked_protocol_has_one_primary_and_fixed_stop():
    p = close.protocol()
    assert p['data_seeds'] == [91000, 92000]
    assert p['selection_data_seed'] not in p['data_seeds']
    assert p['primary']['order'] == 'linear'
    assert len(p['primary']['families']) == 9
    assert 'Close this phase either way' in p['decision']
    assert p['n_per_seed'] * len(p['data_seeds']) == 512


@pytest.mark.parametrize('key,value', [('data_seeds',[87000,92000]),('n_per_seed',1024),('training_seeds',[6100])])
def test_protocol_top_level_mutation_rejected(tmp_path,monkeypatch,key,value):
    p=close.protocol();p[key]=value
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p));monkeypatch.setattr(close,'PROTOCOL',path)
    with pytest.raises(ValueError,match='locked protocol'):close.protocol()


@pytest.mark.parametrize('key,value', [('order','full_quadratic'),('cut',3),('radius',.0625),
                                      ('relative_rms_limit_for_limited_utility',1.2),('scales',[.25])])
def test_primary_mutation_rejected(tmp_path,monkeypatch,key,value):
    p=close.protocol();p['primary'][key]=value
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p));monkeypatch.setattr(close,'PROTOCOL',path)
    with pytest.raises(ValueError):close.protocol()


def test_utility_is_not_close_prediction():
    p=close.protocol();p['uncertainty']['repeats']=100
    blocks=[np.array([[.49]*8,[1.]*8]),np.array([[.49]*8,[1.]*8])]
    r=close.primary_result(blocks,p)
    assert r['relative_rms_error']==pytest.approx(.7)
    assert r['limited_utility_criterion_met'] is True
    assert r['close_prediction_criterion_met'] is False


def test_each_seed_block_must_meet_criterion():
    p=close.protocol();p['uncertainty']['repeats']=100
    blocks=[np.array([[.01]*8,[1.]*8]),np.array([[.81]*8,[1.]*8])]
    r=close.primary_result(blocks,p)
    assert r['relative_rms_error'] < .8
    assert r['limited_utility_criterion_met'] is False


def test_zero_energy_is_not_an_automatic_success():
    p=close.protocol();p['uncertainty']['repeats']=10
    r=close.primary_result([np.zeros((2,8)),np.zeros((2,8))],p)
    assert r['relative_rms_error'] is None and r['uncertainty'] is None
    assert not r['limited_utility_criterion_met']


def test_bootstrap_reproducible_and_paired():
    block=np.array([[1.,4.,9.,16.],[4.,16.,36.,64.]])
    a=close.paired_bootstrap([block,block],repeats=50,seed=2)
    b=close.paired_bootstrap([block,block],repeats=50,seed=2)
    assert a==b and a['one_sided_upper_95']==.5


@pytest.mark.parametrize('loss', [np.array([[-1.],[1.]]),np.array([[float('nan')],[1.]])])
def test_invalid_loss_rejected(loss):
    with pytest.raises(ValueError):close.paired_bootstrap([loss],repeats=1)


def test_reference_source_mutation_fails(tmp_path,monkeypatch):
    scripts=tmp_path/'scripts';scripts.mkdir();path=scripts/'example.py';path.write_text('before')
    p={'frozen_scripts_sha256':{'example.py':close.digest(path)}}
    monkeypatch.setattr(close,'ROOT',tmp_path)
    close.check_sources(p);path.write_text('after')
    with pytest.raises(ValueError,match='frozen helper'):close.check_sources(p)


def test_nonfinite_worker_remains_failed(tmp_path):
    path=tmp_path/'seed-6100.json';npz=path.with_suffix('.npz');np.savez(npz,x=np.array([1.]))
    w={'format':'universa-recurrent.local-response.v1.worker','training_seed':6100,
       'all_raw_gates_passed':True,'per_example_file':npz.name,'per_example_sha256':close.digest(npz),
       'fits':[{'status':'nonfinite_probe_or_prediction'}],'rows':[]}
    path.write_text(json.dumps(w))
    meta,scores,inputs=close.recompute_worker(path)
    assert meta['failed'] is True and not scores


def test_numeric_archive_change_is_rejected(tmp_path):
    path=tmp_path/'seed-6100.json';npz=path.with_suffix('.npz');np.savez(npz,x=np.array([1.]))
    w={'format':'universa-recurrent.local-response.v1.worker','training_seed':6100,
       'all_raw_gates_passed':True,'per_example_file':npz.name,'per_example_sha256':'0'*64}
    path.write_text(json.dumps(w))
    with pytest.raises(ValueError,match='hash mismatch'):close.recompute_worker(path)


def test_writes_refuse_overwrite(tmp_path):
    path=tmp_path/'x.json';close.write(path,{'a':1})
    with pytest.raises(FileExistsError):close.write(path,{'a':2})


def test_missing_reference_stops_before_start(tmp_path):
    with pytest.raises(ValueError,match='missing checkpoint'):
        close.references(tmp_path,tmp_path,close.protocol())


def test_unsafe_json_is_rejected(tmp_path):
    path=tmp_path/'x.json';path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):close.read(path)
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError):close.read(path)


def test_primary_function_does_not_call_model():
    import inspect
    text=inspect.getsource(close.primary_result)+inspect.getsource(close.paired_bootstrap)
    assert 'subprocess' not in text and 'torch' not in text


def test_actual_cpu_worker_loss_audit(tmp_path):
    # CPU CI uses the real current evaluator; no production checkpoint is copied.
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    bench=pytest.importorskip('benchmark_local_response')
    import benchmark_state_continuation as prior
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.dual_study import calibrate,load_estimators,write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    cp=tmp_path/'weights-7100.pt';cal=tmp_path/'calibration.json'
    train_v2_checkpoint(cp,device_name='cpu',seed=7100,train_size=16,calibration_size=8,
        epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,meta,device,_=load_estimators(cp,'cpu')
    _,policy=calibrate(engines,StructuredFlowDataset(8,seed=7200),device,.75,8)
    policy['checkpoint_sha256']=meta['checkpoint_sha256']
    policy['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    write_json(cal,policy)
    old=tmp_path/'old.json'
    args=SimpleNamespace(checkpoint=cp,claim_calibration=cal,device='cpu',calibration_seed=61000,
        test_seed=62000,calibration_size=8,n=4,models=['shared','fixed_depth_4'],bits=[16],rotation_seed=63000,output=old)
    write_json(old,prior.run_worker(args))
    path=tmp_path/'seed-7100.json'
    args=SimpleNamespace(checkpoint=cp,claim_calibration=cal,previous_report=old,worker_output=path,
        models=['shared','fixed_depth_4'],test_seed=91000,n=2,device='cpu')
    write_json(path,bench.worker(args))
    audited,scores,inputs=close.recompute_worker(path)
    assert audited['rows_recomputed']==2112 and audited['failed'] is False
    assert len(scores)==192 and inputs['observed'].shape==(2,5)
    # A reported error cannot be silently replaced by a favorable new number.
    w=close.read(path);w['rows'][0]['metrics']['prediction_mse']+=1
    path.write_text(json.dumps(w))
    with pytest.raises(ValueError,match='score mismatch'):close.recompute_worker(path)
