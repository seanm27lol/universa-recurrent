"""Composition need not be additive; parser fidelity and controlled comparisons do."""
import copy
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
p = importlib.import_module('benchmark_edit_composition')
from mathematical_edits import Edit


def base_manifest():
    return dict(kind='named', bits=16, site={'num_structures':2,'latent_dim':2},
                ranges=[2.,3.,4.,5.,6.,7.])


@pytest.mark.parametrize('scale',p.SCALES)
def test_fixed_plan_and_target_fields(scale):
    pairs=p.pair_plan(base_manifest(),scale)
    assert len(pairs)==7 and len({x[0] for x in pairs})==7
    assert pairs[0][1].delta==scale*.125*2
    assert pairs[-1][1].delta == -pairs[-1][2].delta
    assert pairs[-2][1].candidate != pairs[-2][2].candidate
    assert pairs[-2][1].delta == pairs[-2][2].delta


@pytest.mark.parametrize('scale',[0,True,-1,2,np.nan])
def test_invalid_scales(scale):
    with pytest.raises(ValueError):p.pair_plan(base_manifest(),scale)


@pytest.mark.parametrize('kind',['cycle','evidence'])
def test_b_starts_at_base_and_parser_agrees(kind):
    x=np.arange(18,dtype=np.float32).reshape(3,6)/8
    original=x.copy();a=Edit(kind,0,0 if kind=='cycle' else None,.2)
    b=Edit(kind,1,0 if kind=='cycle' else None,-.3)
    raw=p.pair_states(x,a,b,['a','b'],2,parsed=False)
    parsed=p.pair_states(x,a,b,['a','b'],2,parsed=True)
    assert np.array_equal(x,original)
    assert all(np.array_equal(raw[k],parsed[k]) for k in raw)
    ia=0 if kind=='cycle' else 4
    assert np.array_equal(raw['b'][:,ia],x[:,ia])


def test_predictor_has_no_joint_argument():
    assert list(inspect.signature(p.additive_prediction).parameters)==['base','a','b']
    y0=np.array([[1.,2.]]);a=y0+.5;b=y0-.2
    np.testing.assert_allclose(p.additive_prediction(y0,a,b),y0+.3)


def test_affine_response_is_additive():
    x=np.arange(10,dtype=float).reshape(5,2)
    result=p.interaction_metrics(x,x+2,x-1,x+1)
    assert result['additive_error_mse']==0 and result['relative_rms_interaction']==0


def test_nonlinear_response_has_interaction():
    x=np.arange(10,dtype=float).reshape(5,2)
    result=p.interaction_metrics(x*x,(x+1)**2,(x+2)**2,(x+3)**2)
    assert result['additive_error_mse']==16 and result['worst_absolute_interaction']==4


def test_cancellation_never_gets_fake_zero_relative_error():
    x=np.zeros((4,5))
    result=p.interaction_metrics(x,x+1,x+2,x)
    assert result['relative_rms_interaction'] is None
    assert result['joint_effect_mse']==0 and result['additive_error_mse']==9


@pytest.mark.parametrize('bad',[np.ones((2,3)),np.array([[np.nan,0]]),np.array([1.,2.])])
def test_invalid_response(bad):
    with pytest.raises(ValueError):p.additive_prediction(np.zeros((1,2)),bad,np.zeros((1,2)))


@pytest.fixture(scope='module')
def reference_files(tmp_path_factory):
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.dual_study import calibrate,load_estimators,write_json
    root=tmp_path_factory.mktemp('composition');cp=root/'weights-7100.pt'
    torch.set_num_threads(1)
    train_v2_checkpoint(cp,device_name='cpu',seed=7100,train_size=16,calibration_size=8,
                       epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8)
    engines,meta,device,_=load_estimators(cp,'cpu')
    _,cal=calibrate(engines,StructuredFlowDataset(8,seed=7200),device,.75,8)
    cal['checkpoint_sha256']=meta['checkpoint_sha256']
    cal['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    claim=root/'study-7100/calibration.json';write_json(claim,cal)
    olddir=root/'old';olddir.mkdir()
    oldargs=SimpleNamespace(checkpoint=cp,claim_calibration=claim,device='cpu',n=8,calibration_size=8,
        calibration_seed=61000,test_seed=62000,models=['shared','fixed_depth_4'],bits=[16],
        rotation_seed=63000,output=olddir/'seed-7100.json')
    write_json(oldargs.output,p.prior.run_worker(oldargs))
    return root,cp,claim,oldargs.output


def args_for(refs,tmp_path,**changes):
    _,cp,claim,old=refs
    opts=dict(checkpoint=cp,claim_calibration=claim,previous_report=old,worker_output=tmp_path/'worker.json',
        device='cpu',n=4,test_seed=85000,models=['fixed_depth_4'],scales=[.25])
    opts.update(changes)
    return SimpleNamespace(**opts)


def test_actual_worker_and_metrics(reference_files,tmp_path):
    before=[p.sha256_file(f) for f in reference_files[1:]]
    result=p.worker(args_for(reference_files,tmp_path,models=['shared','fixed_depth_4']))
    assert len(result['rows'])==56 and len(result['raw_gates'])==4
    assert result['no_training'] and result['no_refitting']
    with np.load(tmp_path/'worker.npz',allow_pickle=False) as arrays:
        for row in result['rows']:
            prefix=f"{row['model']}__cut{row['cut']}"
            key=prefix+f"__{row['family']}__scale{row['scale']:g}__{row['horizon']}"
            for path,info in row['paths'].items():
                assert info['status']=='finite'
                x0=arrays[prefix+'__'+row['horizon']+'__'+path+'__baseline__estimate'].astype(float)
                xa,xb,xab=[arrays[key+'__'+path+'__'+which+'__estimate'].astype(float) for which in ('a','b','joint')]
                expected=((xab-xa-xb+x0)**2).mean()
                assert np.isclose(info['metrics']['additive_error_mse'],expected,rtol=1e-8,atol=1e-16)
            assert row['prediction_uses_joint_outcome'] is False
            assert (row['executed_updates_per_evaluation']==0)==(row['horizon']=='readout_at_cut')
        assert len(arrays['observed'])==4
    assert before==[p.sha256_file(f) for f in reference_files[1:]]


def test_raw_parser_mismatch_fails(reference_files,tmp_path,monkeypatch):
    fn=p.pair_states
    def changed(*args,**kwargs):
        result=fn(*args,**kwargs)
        if kwargs['parsed']:result['a'][0,0]+=1
        return result
    monkeypatch.setattr(p,'pair_states',changed)
    with pytest.raises(ValueError,match='parser'):p.worker(args_for(reference_files,tmp_path))


def test_frozen_wrong_site_rejected(reference_files,tmp_path):
    root,cp,claim,old=reference_files
    data=p.read_json(old)
    data['codecs']['fixed_depth_4__cut2__named_16bit']['site']['cut']=3
    altered=tmp_path/'altered.json';p.write_json(altered,data)
    with pytest.raises(ValueError,match='site'):p.worker(args_for((root,cp,claim,altered),tmp_path))


@pytest.mark.parametrize('seed',[83000,7101,7200,61000])
def test_seed_overlap_rejected(reference_files,tmp_path,seed):
    with pytest.raises(ValueError):p.worker(args_for(reference_files,tmp_path,test_seed=seed))


def test_nonfinite_edited_result_is_retained(reference_files,tmp_path,monkeypatch):
    fn=p.continue_path;count=0
    def once(*a,**kw):
        nonlocal count
        count+=1
        if count==6:raise p.prior.NumericalContinuationError('deliberate numerical fixture')
        return fn(*a,**kw)
    monkeypatch.setattr(p,'continue_path',once)
    result=p.worker(args_for(reference_files,tmp_path))
    bad=[r for r in result['rows'] if r['paths']['raw']['status']!='finite']
    assert len(bad)==1 and bad[0]['codec_comparison'] is None
    assert bad[0]['paths']['raw']['metrics'] is None


def test_parent_subprocess_roundtrip_and_overwrite_guard(reference_files,tmp_path):
    root,_,_,old=reference_files;out=tmp_path/'result'
    args=['--replication-dir',str(root),'--previous-dir',str(old.parent),'--output-dir',str(out),
          '--device','cpu','--n','4','--scales','.25','--models','fixed_depth_4']
    assert p.main(args)==0
    summary=p.read_json(out/'summary.json')
    assert summary['case_rows']==28 and summary['failed_path_cases']==0
    assert summary['all_raw_gates_passed'] and summary['checkpoint_runs']==1
    assert p.main(args)==2
    assert not list(out.glob('*.pt'))


@pytest.mark.parametrize('extra',[[],['--models','shared','shared'],['--scales','.5','.5'],['--n','1']])
def test_bad_cli(extra):assert p.main(extra)==2
