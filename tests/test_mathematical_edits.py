"""Known-field language correctness is a gate, not proof of latent semantics."""
import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
edit=importlib.import_module('mathematical_edits')
from continuation_codec import StateCodec
from overflow_state_codec import OverflowStateCodec
NAMES=['balanced_flow','wrong_diagonal_constraint']

def base():
    site=dict(checkpoint_sha256='a'*64,model='shared',cut=2,depth=4,num_structures=2,latent_dim=2,
              names=NAMES,calibration_sha256='b'*64)
    return StateCodec.fit(np.arange(60,dtype=np.float32).reshape(10,6),kind='named',bits=16,site=site).manifest()


@pytest.mark.parametrize('op',edit.edit_plan(base()))
def test_parser_matches_independent_numerical_edits_without_mutation(op):
    x=np.random.default_rng(42).normal(size=(13,6)).astype(np.float32);saved=x.copy()
    cmd=edit.render(op,NAMES,2)
    assert edit.parse(cmd,NAMES,2)==op
    assert np.array_equal(edit.direct_edit(x,op,NAMES,2),edit.apply_command(x,cmd,NAMES,2))
    assert x.tobytes()==saved.tobytes()


@pytest.mark.parametrize('cmd',[
 'cycle missing[0] += 1','cycle balanced_flow[2] += 1','evidence balanced_flow += nan',
 'all_evidence += 1e500','__import__("os")','cycle balanced_flow[0] *= 2','identity;print(1)',
 'evidence balanced_flow += inf','identity\n','x'*513,'all_evidence += -1e99',None,
])
def test_unsafe_undefined_or_unbounded_commands_rejected(cmd):
    with pytest.raises(ValueError):edit.parse(cmd,NAMES,2)


@pytest.mark.parametrize('kind',['cycle','evidence'])
def test_target_specificity_control_is_not_the_requested_edit(kind):
    op=edit.Edit(kind,0,0 if kind=='cycle' else None,.5)
    wrong=edit.wrong_candidate(op,NAMES,2)
    a=np.zeros((3,6),np.float32)
    assert not np.array_equal(edit.direct_edit(a,op,NAMES,2),edit.direct_edit(a,wrong,NAMES,2))
    assert wrong.delta==op.delta


def test_decoder_accepts_no_context_and_uses_immutable_frozen_manifest():
    import inspect
    assert list(inspect.signature(edit.decode_and_edit).parameters)==['payloads','codec','command']
    b=base();codec=OverflowStateCodec.from_manifest(b);manifest=codec.manifest()
    x=np.array([[1,2,3,4,5,6],[1e3,0,0,0,0,0]],np.float32)
    payloads,fb=codec.encode(x)
    result=edit.decode_and_edit(payloads,codec,'cycle balanced_flow[0] += 0.125')
    assert np.array_equal(result,edit.direct_edit(codec.decode(payloads),edit.Edit('cycle',0,0,.125),NAMES,2))
    assert np.array_equal(result[fb],edit.direct_edit(x,edit.Edit('cycle',0,0,.125),NAMES,2)[fb])
    assert codec.manifest()==manifest


def test_neutral_shift_preserves_immediate_probabilities():
    x=np.array([[1,2,3,4,0,1],[0,0,0,0,2,2]],np.float32)
    op=edit.Edit('all_evidence',delta=math.log(2));q=np.stack([np.eye(5)[:,:2]]*2)
    result=edit.primitive_error(x,edit.direct_edit(x,op,NAMES,2),op,q,NAMES,2)
    assert result['candidate_delta_max_error']==0
    assert result['probability_rule_max_error']<1e-7


def test_cycle_geometry_prediction():
    x=np.zeros((3,6),np.float32);q=np.stack([np.eye(5)[:,:2]]*2);op=edit.Edit('cycle',1,1,.25)
    r=edit.primitive_error(x,edit.direct_edit(x,op,NAMES,2),op,q,NAMES,2)
    assert r=={'candidate_delta_max_error':0.,'probability_rule_max_error':0.}


@pytest.mark.parametrize('x',[np.zeros((2,5),np.float32),np.zeros((2,6),np.float64),np.full((2,6),np.nan,np.float32)])
def test_invalid_state_rejected(x):
    with pytest.raises(ValueError):edit.apply_command(x,'identity',NAMES,2)


def test_wrong_codec_and_extraneous_arguments_rejected():
    codec=OverflowStateCodec.from_manifest(base());p,_=codec.encode(np.zeros((2,6),np.float32))
    with pytest.raises(ValueError):edit.decode_and_edit([b'x'*len(p[0])],codec,'identity')
    with pytest.raises(TypeError):edit.decode_and_edit(p,codec,'identity',original=np.zeros((2,6)))


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.dual_study import load_estimators,calibrate,write_json
    prior=importlib.import_module('benchmark_state_continuation')
    root=tmp_path_factory.mktemp('math-edit-fixture');repl=root/'repl';repl.mkdir();previous=root/'previous';previous.mkdir()
    cp=repl/'weights-5242.pt'
    train_v2_checkpoint(cp,device_name='cpu',seed=5242,train_size=16,calibration_size=8,epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,meta,device,_=load_estimators(cp,'cpu')
    _,artifact=calibrate(engines,StructuredFlowDataset(8,seed=54000),device,.75,8)
    artifact['checkpoint_sha256']=meta['checkpoint_sha256'];artifact['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    cal=repl/'study-5242/calibration.json';write_json(cal,artifact)
    args=SimpleNamespace(checkpoint=cp,claim_calibration=cal,output=previous/'seed-5242.json',device='cpu',models=['shared','fixed_depth_4'],bits=[16],n=4,calibration_size=8,calibration_seed=61000,test_seed=62000,rotation_seed=63000)
    old=prior.run_worker(args);write_json(args.output,old)
    return repl,previous,cp,cal


def test_real_frozen_checkpoint_cli_roundtrip_and_report_arithmetic(references,tmp_path):
    pipeline=importlib.import_module('benchmark_mathematical_edits');repl,prev,cp,cal=references
    out=tmp_path/'results'
    args=['--replication-dir',str(repl),'--previous-dir',str(prev),'--output-dir',str(out),'--device','cpu','--n','4']
    before=cp.read_bytes()
    assert pipeline.main(args)==0
    w=pipeline.read_json(out/'seed-5242.json');summary=pipeline.read_json(out/'summary.json')
    assert summary['failed_cases']==0 and summary['all_raw_gates_passed']
    assert len(w['rows'])==160 and len(w['raw_gates'])==4
    a=np.load(out/'seed-5242.npz',allow_pickle=False)
    for row in w['rows']:
        assert row['executed_updates']==(8 if row['model']=='shared' else 4)-row['cut']
        k=f"{row['model']}__cut{row['cut']}__edit{row['edit_index']}__"
        raw=a[k+'direct_float32__estimate'].astype(float);actual=a[k+row['mode']+'__estimate'].astype(float)
        assert np.isclose(row['metrics']['final_mse_to_direct'],((raw-actual)**2).mean())
        if row['mode']=='direct_float32':assert row['metrics']['final_mse_to_direct']==0
    assert cp.read_bytes()==before
    assert pipeline.main(args)==2


def test_no_future_data_available_to_parser(monkeypatch):
    x=np.arange(12,dtype=np.float32).reshape(2,6)
    monkeypatch.setattr(StateCodec,'fit',lambda *a,**k:pytest.fail('refitting'))
    assert edit.apply_command(x,'identity',NAMES,2).tobytes()==x.tobytes()


@pytest.mark.parametrize('seed',[62000,71000,72000,46000])
def test_old_seeds_refused(seed,tmp_path):
    pytest.importorskip('torch')
    pipeline=importlib.import_module('benchmark_mathematical_edits')
    assert pipeline.main(['--replication-dir',str(tmp_path),'--previous-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),'--test-seed',str(seed)])==2


def test_effect_metric_separates_base_error_from_treatment():
    torch=pytest.importorskip('torch');pipeline=importlib.import_module('benchmark_mathematical_edits')
    def output(value):
        return dict(estimate=torch.tensor([[value]*5]),claim_index=torch.tensor([-1]),top_route=torch.tensor([0]))
    # Each path changes by one, even though compression shifted its starting point.
    result=pipeline.metrics(output(3.),output(2.),output(2.),output(1.))
    assert result['final_mse_to_direct']==1
    assert result['effect_disagreement_mse']==0


def test_missing_direct_comparison_is_explicit():
    torch=pytest.importorskip('torch');pipeline=importlib.import_module('benchmark_mathematical_edits')
    x=dict(estimate=torch.zeros(2,5))
    result=pipeline.metrics(x,None,x,x)
    assert result['comparison_available'] is False
    assert 'final_mse_to_direct' not in result


def test_numerical_failure_is_kept_as_failed_row(references,tmp_path,monkeypatch):
    pipeline=importlib.import_module('benchmark_mathematical_edits');repl,prev,cp,cal=references
    original=pipeline.prior.continue_from;calls=0
    def fail_one(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==3:raise pipeline.prior.NumericalContinuationError('injected experimental failure')
        return original(*a,**kw)
    monkeypatch.setattr(pipeline.prior,'continue_from',fail_one)
    args=SimpleNamespace(checkpoint=cp,claim_calibration=cal,previous_report=prev/'seed-5242.json',
        worker_output=tmp_path/'out.json',device='cpu',n=4,test_seed=83000,models=['shared'])
    result=pipeline.worker(args)
    failures=[r for r in result['rows'] if r['status']=='nonfinite_continuation']
    assert len(failures)==1 and failures[0]['metrics'] is None
    assert result['all_raw_gates_passed']
