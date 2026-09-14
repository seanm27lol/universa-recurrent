"""Full-state restoration is mandatory; lossy reconstruction may honestly fail."""
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
p=importlib.import_module('benchmark_state_continuation')
from universa_recurrent.neural.data import StructuredFlowDataset, candidate_bases
from universa_recurrent.neural.v2 import MultiHypothesisRecurrentNet, UntiedMultiHypothesisNet, V2Config
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output


def make(untied=False):
    torch.set_num_threads(1);torch.manual_seed(123)
    _,q,_=candidate_bases()
    model=(UntiedMultiHypothesisNet if untied else MultiHypothesisRecurrentNet)(torch.from_numpy(q), V2Config(hidden_dim=8,candidate_embedding_dim=2,steps=4)).eval()
    data=StructuredFlowDataset(7,seed=543)
    return model,data


@pytest.mark.parametrize('untied',[False,True])
@pytest.mark.parametrize('cut',[1,2,3])
def test_exact_boundary_matches_unmodified_rollout(untied,cut):
    model,data=make(untied);policy=ClaimPolicy(.5)
    weights=copy.deepcopy(model.state_dict());original_obs=data.observed.clone()
    h=model.rollout(data.observed,data.mask)
    expected=construct_output(h['states'][-1],h['route_probabilities'][-1],policy)
    s=p.pause(model,data.observed,data.mask,cut);v=s.vector()
    out,executed=p.continue_from(model,s,v,policy)
    assert executed==4-cut
    p.compare_outputs(out,expected)
    assert np.array_equal(s.vector(),v)
    assert torch.equal(data.observed,original_obs)
    assert all(torch.equal(value,model.state_dict()[k]) for k,value in weights.items())
    assert s.retained_bytes_per_example()==4*(5+5+8+2)


def test_pause_does_not_call_future_rollout(monkeypatch):
    model,data=make();calls=[];step=model._step
    def count(*a,**kw): calls.append(1);return step(*a,**kw)
    monkeypatch.setattr(model,'_step',count)
    monkeypatch.setattr(model,'rollout',lambda *a,**k:pytest.fail('future rollout leaked into pause'))
    s=p.pause(model,data.observed,data.mask,2)
    assert len(calls)==2
    p.continue_from(model,s,s.vector(),ClaimPolicy(None))
    assert len(calls)==4


def test_decoder_state_used_not_original_dynamic(monkeypatch):
    model,data=make();s=p.pause(model,data.observed,data.mask,2)
    step=model._step;seen=[]
    def check(context,obs,mask,coords,prior,logits):
        seen.append((coords.clone(),logits.clone()))
        return step(context,obs,mask,coords,prior,logits)
    monkeypatch.setattr(model,'_step',check)
    p.continue_from(model,s,np.zeros_like(s.vector()),ClaimPolicy(None))
    assert torch.count_nonzero(seen[0][0])==0 and torch.count_nonzero(seen[0][1])==0


def test_missing_logits_and_wrong_model_fail():
    model,data=make();s=p.pause(model,data.observed,data.mask,2)
    with pytest.raises(ValueError):p.continue_from(model,s,s.vector()[:,:4],ClaimPolicy(None))
    other,_=make()
    with pytest.raises(ValueError):p.continue_from(other,s,s.vector(),ClaimPolicy(None))


@pytest.mark.parametrize('cut',[0,4,5,True,1.5])
def test_invalid_cut(cut):
    model,data=make()
    with pytest.raises(ValueError):p.pause(model,data.observed,data.mask,cut)


def test_stop_at_cut_is_explicit_zero_remaining_budget():
    model,data=make();s=p.pause(model,data.observed,data.mask,2)
    out,steps=p.continue_from(model,s,s.vector(),ClaimPolicy(None),stop_at_cut=True)
    assert steps==0
    h=model.rollout(data.observed,data.mask,steps=2)
    expected=construct_output(h['states'][-1],h['route_probabilities'][-1],ClaimPolicy(None))
    p.compare_outputs(out,expected)


def test_controls_erase_only_named_fields_and_shuffle_once():
    original=np.arange(42,dtype=np.float32).reshape(7,6)
    choices={name:(val,stop) for name,_,val,stop in p.state_cases(original,original,
      {'num_structures':2,'latent_dim':2},[8],12)}
    assert (choices['erase_coordinates'][0][:,:4]==0).all()
    assert np.array_equal(choices['erase_coordinates'][0][:,4:],original[:,4:])
    assert (choices['erase_current_logits'][0][:,4:]==0).all()
    assert np.array_equal(choices['other_input_dynamic'][0][1:],original[:-1])
    assert choices['stop_at_cut'][1]
    assert p.cuts_for(4)==[1,2,3] and p.cuts_for(8)==[2,4,7]


@pytest.fixture(scope='module')
def fixture_files(tmp_path_factory):
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    from universa_recurrent.neural.dual_study import calibrate, load_estimators,write_json
    root=tmp_path_factory.mktemp('state-continuation')
    cp=root/'weights-7100.pt'
    train_v2_checkpoint(cp,device_name='cpu',seed=7100,train_size=16,calibration_size=8,
                       epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8)
    engines,metadata,device,_=load_estimators(cp,'cpu')
    _,cal=calibrate(engines,StructuredFlowDataset(8,seed=7200),device,.75,8)
    cal['checkpoint_sha256']=metadata['checkpoint_sha256']
    cal['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    file=root/'study-7100/calibration.json';write_json(file,cal)
    return root,cp,file


def test_real_parent_worker_cli_and_npz(fixture_files,tmp_path):
    root,cp,cal=fixture_files;before=(p.sha256_file(cp),p.sha256_file(cal))
    out=tmp_path/'out'
    args=['--replication-dir',str(root),'--output-dir',str(out),'--device','cpu',
          '--n','8','--calibration-size','8','--models','fixed_depth_4','--bits','4','16']
    assert p.main(args)==0
    summary=p.read_json(out/'summary.json')
    assert summary['all_restoration_gates_passed'] and summary['checkpoint_runs']==1
    w=p.read_json(out/'seed-7100.json')
    assert len(w['rows'])==30  # three cuts times ten methods
    for row in w['rows']:
        if row['method']=='raw_f32':
            assert row['metrics']['state_mse']==0
            assert row['metrics']['claim_index_changed_n']==0
        if row['method']!='stop_at_cut':
            assert row['executed_updates']==row['depth']-row['cut']
    with np.load(out/w['per_example_file'],allow_pickle=False) as a:
        assert len(a['test_truth'])==8
        for row in w['rows']:
            key=f'{row["model"]}__cut{row["cut"]}__{row["method"]}'
            assert np.isclose(a[key+'__task_mse'].mean(),row['metrics']['task_mse'])
    assert p.sha256_file(out/w['per_example_file'])==w['per_example_sha256']
    assert before==(p.sha256_file(cp),p.sha256_file(cal))
    assert p.main(args)==2
    assert not list(out.glob('*.pt'))


@pytest.mark.parametrize('field,value',[('calibration_seed',7101),('test_seed',7102),('test_seed',7200),('calibration_seed',62000)])
def test_split_overlap_fails(fixture_files,tmp_path,field,value):
    _,cp,cal=fixture_files
    args=dict(checkpoint=cp,claim_calibration=cal,device='cpu',n=4,calibration_size=4,
        calibration_seed=61000,test_seed=62000,models=['fixed_depth_4'],bits=[8],rotation_seed=33,output=tmp_path/'x.json')
    args[field]=value
    with pytest.raises(ValueError):p.run_worker(SimpleNamespace(**args))


def test_cli_rejects_missing_input_and_duplicate_settings(tmp_path):
    assert p.main(['--device','cpu'])==2
    assert p.main(['--models','shared','shared'])==2
    assert p.main(['--bits','4','4'])==2
    assert p.main(['--replication-dir',str(tmp_path),'--output-dir',str(tmp_path/'new')])==2


def test_nonfinite_continuation_is_explicit(monkeypatch):
    model,data=make();s=p.pause(model,data.observed,data.mask,2)
    step=model._step
    def overflow(*args):
        coords,state,logits,rms,progress=step(*args)
        return torch.full_like(coords,float('inf')),state,logits,rms,progress
    monkeypatch.setattr(model,'_step',overflow)
    with pytest.raises(p.NumericalContinuationError):
        p.continue_from(model,s,s.vector(),ClaimPolicy(None))


def test_summary_does_not_hide_failed_checkpoint():
    good=dict(model='shared',cut=2,method='named_4bit',metrics=dict(n=10,
        final_estimate_mse_to_reference=.01,task_mse=.1,claim_index_changed_n=2))
    bad={**good,'metrics':None,'status':'nonfinite_continuation'}
    row=p.summarize([{'rows':[good]},{'rows':[bad]}])[0]
    assert row['failed_checkpoint_n']==1
    assert row['task_mse']['mean'] is None
    assert row['task_mse']['per_checkpoint']==[.1,None]
