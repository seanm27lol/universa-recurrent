"""Local rules may fail to predict nonlinear behavior; leakage and algebra may not."""
from pathlib import Path
from types import SimpleNamespace
import importlib, inspect, sys
import numpy as np
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
lr=importlib.import_module('local_response')

@pytest.mark.parametrize('width',[1,2,6])
@pytest.mark.parametrize('radius',[.125,.0625])
def test_quadratic_exact_and_counted(width,radius):
    rng=np.random.default_rng(91);batch=3;output=5
    g=rng.normal(size=(batch,output,width));h=rng.normal(size=(batch,output,width,width));h=(h+h.swapaxes(-1,-2))/2
    b=rng.normal(size=(batch,output));u=lr.stencil(width,radius)
    y=b[None]+np.einsum('bnd,pd->pbn',g,u)+.5*np.einsum('bnij,pi,pj->pbn',h,u,u)
    rule=lr.build_rule(u,y);v=rng.normal(size=(batch,width))
    actual=b+np.einsum('bnd,bd->bn',g,v)+.5*np.einsum('bnij,bi,bj->bn',h,v,v)
    np.testing.assert_allclose(rule.predict(v),actual,rtol=1e-10,atol=1e-10)
    assert len(u)==1+2*width+2*width*(width-1)
    assert rule.stored_bytes_per_input()==b[0].nbytes+g[0].nbytes+h[0].nbytes
    np.testing.assert_allclose(rule.predict(v,order='linear'),b+np.einsum('bnd,bd->bn',g,v),atol=1e-10)
    np.testing.assert_array_equal(rule.predict(v,order='none'),b)

def test_cross_term_is_not_diagonal_curvature():
    u=lr.stencil(2,.125);y=(u[:,0]*u[:,1])[:,None,None]
    rule=lr.build_rule(u,y);v=np.array([[1.,2.]])
    assert rule.predict(v).item()==2
    assert rule.predict(v,order='diagonal_quadratic').item()==0
    assert 'model' not in inspect.signature(rule.predict).parameters
    assert 'target' not in inspect.signature(lr.build_rule).parameters

def test_zero_baseline_and_cancellation_statistic():
    u=lr.stencil(2,.125);rule=lr.build_rule(u,np.zeros((len(u),3,5)))
    np.testing.assert_array_equal(rule.predict(np.zeros((3,2))),rule.base)
    result=lr.score(rule.base,rule.base,rule.base)
    assert result['relative_rms_error'] is None
    assert result['prediction_mse']==0

@pytest.mark.parametrize('width,radius',[(0,.1),(17,.1),(True,.1),(2,0),(2,float('nan')),(2,2),(2,True)])
def test_invalid_stencil(width,radius):
    with pytest.raises(ValueError):lr.stencil(width,radius)

@pytest.mark.parametrize('bad',['order','shape','nan','partial','reverse'])
def test_invalid_rule_data(bad):
    u=lr.stencil(2,.125);y=np.ones((len(u),2,3))
    if bad=='partial':u=u[:-1];y=y[:-1]
    elif bad=='reverse':u=u[::-1]
    elif bad=='nan':y[0,0,0]=np.nan
    elif bad=='shape':y=y[0]
    else:u[1,1]=.1
    with pytest.raises(ValueError):lr.build_rule(u,y)

def test_no_mutation_or_target_hidden_in_rule():
    u=lr.stencil(2,.125);y=np.ones((len(u),2,3));before=y.copy();rule=lr.build_rule(u,y)
    v=np.ones((2,2));rule.predict(v)
    np.testing.assert_array_equal(y,before);np.testing.assert_array_equal(v,1)
    assert set(vars(rule))=={'base','jacobian','hessian','radius'}

def test_queries_are_outside_stencil_and_prescribed():
    plans=lr.query_plan();assert len(plans)==33
    for _,_,u in plans:
        if not u.any():continue
        for radius in (.125,.0625):
            assert not any(np.array_equal(u,p) for p in lr.stencil(6,radius))
    assert [x[0] for x in plans[:11]].count('common_evidence_control')==1
    np.testing.assert_array_equal(np.stack([x[2] for x in plans]),np.stack([x[2] for x in lr.query_plan()]))

@pytest.fixture(scope='module')
def refs(tmp_path_factory):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.dual_study import calibrate,load_estimators,write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    import benchmark_state_continuation as prior
    root=tmp_path_factory.mktemp('response');cp=root/'weights-7100.pt';cal=root/'study-7100/calibration.json'
    train_v2_checkpoint(cp,device_name='cpu',seed=7100,train_size=16,calibration_size=8,epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,meta,device,_=load_estimators(cp,'cpu')
    _,policy=calibrate(engines,StructuredFlowDataset(8,seed=7200),device,.75,8)
    policy['checkpoint_sha256']=meta['checkpoint_sha256'];policy['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured};write_json(cal,policy)
    old=root/'previous/seed-7100.json';old.parent.mkdir()
    args=SimpleNamespace(checkpoint=cp,claim_calibration=cal,device='cpu',calibration_seed=61000,test_seed=62000,calibration_size=8,n=4,models=['shared','fixed_depth_4'],bits=[16],rotation_seed=63000,output=old)
    write_json(old,prior.run_worker(args))
    return root,cp,cal,old

@pytest.mark.parametrize('bad',['model','count','seed'])
def test_arguments_are_rejected(bad):
    pytest.importorskip('torch');import benchmark_local_response as bench
    args=SimpleNamespace(test_seed=87000,n=4,models=['shared'])
    if bad=='model':args.models=['shared','shared']
    if bad=='count':args.n=1
    if bad=='seed':args.test_seed=85000
    with pytest.raises(ValueError):bench.validate_args(args)

def test_real_cpu_worker_and_saved_prediction_arithmetic(refs,tmp_path):
    import benchmark_local_response as bench
    root,cp,cal,old=refs;out=tmp_path/'worker.json'
    w=bench.worker(SimpleNamespace(checkpoint=cp,claim_calibration=cal,previous_report=old,worker_output=out,models=['fixed_depth_4'],test_seed=87000,n=2,device='cpu'))
    assert w['all_raw_gates_passed'];assert len(w['fits'])==8
    assert all(f['status']=='finite' and f['probe_evaluations']==73 for f in w['fits'])
    assert len(w['rows'])==1056
    with np.load(out.with_suffix('.npz'),allow_pickle=False) as a:
        for fit in w['fits']:
            prefix=f"{fit['model']}__cut{fit['cut']}__{fit['horizon']}__radius{fit['radius']:g}"
            rule=lr.build_rule(a[prefix+'__offsets'],a[prefix+'__probes'])
            np.testing.assert_array_equal(rule.jacobian,a[prefix+'__jacobian'])
            np.testing.assert_array_equal(rule.hessian,a[prefix+'__hessian'])
            row=next(r for r in w['rows'] if r['cut']==fit['cut'] and r['horizon']==fit['horizon'] and r['radius']==fit['radius'] and r['query_index']==0 and r['order']=='full_quadratic')
            key=f"{fit['model']}__cut{fit['cut']}"
            actual=a[key+f"__{fit['horizon']}__query0__actual"]
            pred=rule.predict(a[key+'__query0__actual_u'])
            assert lr.score(pred,actual,rule.base)==row['metrics']

def test_subprocess_cli_and_overwrite(refs,tmp_path):
    pytest.importorskip('torch');import subprocess,os,json
    root,cp,cal,old=refs;out=tmp_path/'run'
    cmd=[sys.executable,str(SCRIPTS/'benchmark_local_response.py'),'--replication-dir',str(root),'--previous-dir',str(old.parent),'--output-dir',str(out),'--device','cpu','--models','fixed_depth_4','--n','2']
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((out/'summary.json').read_text())
    assert report['all_raw_gates_passed'] and report['failed_fits']==0 and report['checkpoint_runs']==1
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and 'overwrite' in result.stderr

@pytest.mark.parametrize('kind',['bad_order','bad_displacement','nonfinite'])
def test_prediction_rejects_bad_input(kind):
    u=lr.stencil(2,.125);rule=lr.build_rule(u,np.ones((len(u),2,3)))
    if kind=='bad_order':
        with pytest.raises(ValueError):rule.predict(np.zeros((2,2)),order='cubic')
    elif kind=='bad_displacement':
        with pytest.raises(ValueError):rule.predict(np.zeros((1,2)))
    else:
        with pytest.raises(ValueError):rule.predict(np.full((2,2),np.inf))

def test_nonfinite_probes_record_failed_fits(refs,tmp_path,monkeypatch):
    import benchmark_local_response as bench
    root,cp,cal,old=refs
    original=bench.response
    def failed(model,point,state,policy,horizon):
        if not np.array_equal(state,point.vector()):
            raise bench.prior.NumericalContinuationError('injected failed probe/target')
        return original(model,point,state,policy,horizon)
    monkeypatch.setattr(bench,'response',failed)
    w=bench.worker(SimpleNamespace(checkpoint=cp,claim_calibration=cal,previous_report=old,worker_output=tmp_path/'worker.json',models=['fixed_depth_4'],test_seed=87000,n=2,device='cpu'))
    assert all(f['status']=='nonfinite_probe_or_prediction' for f in w['fits'])
    assert all(r['status']=='nonfinite_target' for r in w['rows'])
    assert w['all_raw_gates_passed']
