"""Same answers before speed claims; records must not overstate their guarantees."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
from universa_recurrent.neural.data import candidate_bases, StructuredFlowDataset
from universa_recurrent.neural.v2 import (MultiHypothesisRecurrentNet, UntiedMultiHypothesisNet,
    AmbientRecurrentControl, DirectMultiHypothesisNet, V2Config)
from universa_recurrent.neural.final_state import final_candidates, final_ambient, FinalStateEstimator
from universa_recurrent.neural.dual_output import FixedEstimator, ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import calibrate, write_json, load_estimators
from universa_recurrent.neural.dual_lingua import make_record as endpoint_record
from universa_recurrent.neural.retention_lingua import make_record, verify_record, dumps, loads
from universa_recurrent.neural.retention_study import compare_outputs, run_worker, main, summarize
from universa_recurrent.neural.v2_train import train_v2_checkpoint


def model_for(kind='shared', steps=4):
    _, bases, _ = candidate_bases()
    torch.manual_seed(19)
    config = V2Config(hidden_dim=12, candidate_embedding_dim=4, steps=steps)
    if kind == 'ambient':
        return AmbientRecurrentControl(5, 12, steps).eval()
    cls = {'shared': MultiHypothesisRecurrentNet, 'untied': UntiedMultiHypothesisNet,
           'direct': DirectMultiHypothesisNet}[kind]
    return cls(torch.tensor(bases), config).eval()


@pytest.mark.parametrize('kind', ['shared','untied','ambient'])
@pytest.mark.parametrize('batch', [1,7])
@pytest.mark.parametrize('steps', [1,2,4])
def test_endpoints_equal_all_reference_fields(kind, batch, steps):
    model = model_for(kind)
    data = StructuredFlowDataset(batch, seed=47)
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():
        reference = model.rollout(data.observed, data.mask, steps=steps)
    if kind == 'ambient':
        actual = final_ambient(model, data.observed, data.mask, steps=steps)
        torch.testing.assert_close(actual, reference[-1], atol=1e-7, rtol=1e-6)
        assert not actual.requires_grad
    else:
        actual = final_candidates(model, data.observed, data.mask, steps=steps)
        expected = {key: val if key == 'prior_logits' else val[-1] for key,val in reference.items()}
        compare_outputs(actual, expected)
        assert all(not v.requires_grad for v in actual.values())
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize('kind', ['shared','untied','ambient'])
def test_final_path_never_calls_rollout_or_stack(kind):
    model = model_for(kind)
    data = StructuredFlowDataset(2, seed=49)
    with patch.object(model, 'rollout', side_effect=AssertionError('history allocated')), \
         patch('torch.stack', side_effect=AssertionError('stack allocated')):
        if kind == 'ambient':
            final_ambient(model, data.observed, data.mask)
        else:
            final_candidates(model, data.observed, data.mask)


@pytest.mark.parametrize('kind', ['shared','untied','ambient','direct'])
@pytest.mark.parametrize('threshold', [None,0.,0.5,1.])
def test_adapter_preserves_estimate_and_discrete_claims(kind, threshold):
    model = model_for(kind)
    _, bases, _ = candidate_bases()
    engine = FixedEstimator(kind,model,torch.tensor(bases),kind)
    final = FinalStateEstimator.from_reference(engine)
    assert final.model is model
    data = StructuredFlowDataset(7, seed=777)
    mode = 'estimate_only' if kind == 'ambient' else 'mixture_with_claim'
    policy = ClaimPolicy(threshold)
    compare_outputs(final.predict(data.observed,data.mask,policy,mode),
                    engine.predict(data.observed,data.mask,policy,mode))


@pytest.mark.parametrize('bad', [0,-1,5,True,1.5])
def test_invalid_depth_rejected(bad):
    data = StructuredFlowDataset(1,seed=55)
    with pytest.raises(ValueError):
        final_candidates(model_for(), data.observed, data.mask, steps=bad)


def test_input_checks_and_zero_tie():
    model = model_for()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    observed = torch.zeros(2,5)
    mask = torch.ones_like(observed)
    final = final_candidates(model,observed,mask)
    assert (final['route_probabilities'] == .5).all()
    assert (final['states'] == 0).all()
    a = construct_output(final['states'],final['route_probabilities'],ClaimPolicy(.5))
    assert (a['claim_index'] == 0).all()
    observed[0,0] = float('nan')
    with pytest.raises(ValueError): final_candidates(model,observed,mask)
    with pytest.raises(ValueError): final_candidates(model,observed[:0],mask[:0])
    with pytest.raises(ValueError): final_candidates(model.train(),torch.zeros(2,5),mask)


def valid_record(full=True):
    model = model_for()
    names, _, boundaries = candidate_bases()
    data = StructuredFlowDataset(1,seed=86)
    with torch.inference_mode(): history = model.rollout(data.observed,data.mask)
    policy=ClaimPolicy(0.)
    prediction=construct_output(history['states'][-1],history['route_probabilities'][-1],policy)
    endpoint=endpoint_record(prediction,policy,data.observed,data.mask,names,boundaries,
        model_name='shared',checkpoint_sha256='a'*64,calibration_sha256='b'*64,trained_depth=4)
    return make_record(endpoint,history=history if full else None,bases=model.bases)


@pytest.mark.parametrize('full',[False,True])
def test_record_roundtrip_and_scope(full):
    result=verify_record(loads(dumps(valid_record(full))))
    assert result['accepted'], result
    assert result['retained_steps_checked'] == (4 if full else 0)
    assert not result['learned_transitions_verified']
    assert not result['estimate_structurally_certified']


@pytest.mark.parametrize('field', ['coordinates','states','route_logits','route_probabilities','residuals','progress','bases'])
def test_earlier_trajectory_tampering_rejected(field):
    record=valid_record()
    array=record['trajectory'][field]
    while isinstance(array[0],list): array=array[0]
    array[0] += .4
    assert not verify_record(record)['accepted']


def test_scope_forgery_and_missing_history_rejected():
    r=valid_record(); r['scope']['learned_transitions_verified']=True
    assert not verify_record(r)['accepted']
    r=valid_record(False); r['retention']='trajectory'
    assert not verify_record(r)['accepted']
    r=valid_record(); r['trajectory']['progress'][0][0]=float('inf')
    assert not verify_record(r)['accepted']
    with pytest.raises(ValueError): loads(b'{"a":1,"a":2}')
    with pytest.raises(ValueError): loads(b'{"a":NaN}')


def test_checker_does_not_replay_neural_model():
    r=valid_record()
    with patch.object(MultiHypothesisRecurrentNet,'rollout',side_effect=AssertionError('replay')), \
         patch.object(MultiHypothesisRecurrentNet,'_step',side_effect=AssertionError('replay')):
        assert verify_record(r)['accepted']


def test_comparison_rejects_changed_claim_even_when_estimate_matches():
    a={'estimate':torch.zeros(1,5),'claim_mask':torch.tensor([False])}
    b={'estimate':torch.zeros(1,5),'claim_mask':torch.tensor([True])}
    with pytest.raises(ValueError): compare_outputs(a,b)


def test_missing_weights_and_duplicate_batch_cli(tmp_path):
    assert main(['--replication-dir',str(tmp_path),'--output-dir',str(tmp_path/'new'),'--device','cpu']) == 2
    assert main(['--replication-dir',str(tmp_path),'--output-dir',str(tmp_path/'new2'),'--batch-sizes','1','1']) == 2


def test_tiny_frozen_worker_and_aggregation(tmp_path):
    checkpoint=tmp_path/'weights-100.pt'
    train_v2_checkpoint(checkpoint,device_name='cpu',seed=100,train_size=16,calibration_size=8,
        epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,metadata,device,_=load_estimators(checkpoint,'cpu')
    calibration=StructuredFlowDataset(8,seed=200)
    _,artifact=calibrate(engines,calibration,device,.75,8)
    artifact['checkpoint_sha256']=metadata['checkpoint_sha256']
    artifact['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    cal=tmp_path/'calibration.json'; write_json(cal,artifact)
    from universa_recurrent.neural.train import sha256_file
    before=sha256_file(checkpoint)
    args=SimpleNamespace(deterministic=False,checkpoint=checkpoint,calibration=cal,device='cpu',
        test_seed=300,n=8,batch_size=3,warmup=0,repeats=1,order_seed=400,audit_samples=1,audit_repeats=1)
    result=run_worker(args)
    assert sha256_file(checkpoint)==before
    assert len(result['inference_rows']) == 10
    assert all(v['accepted'] for a in result['lingua_costs'] for v in
               [a['checkpoint_binding_single_calls']['endpoint']['result'],a['checkpoint_binding_single_calls']['trajectory']['result']])
    summary=summarize([result])
    assert len(summary)==5 and all(r['checkpoint_pairs']==1 for r in summary)
    assert result['dataset_sha256']


@pytest.mark.parametrize('kind',['shared','untied','ambient'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA equivalence needs a GPU')
def test_cuda_final_equivalence(kind):
    model=model_for(kind).to('cuda')
    data=StructuredFlowDataset(9,seed=12345)
    observed, mask=data.observed.cuda(),data.mask.cuda()
    with torch.inference_mode(): reference=model.rollout(observed,mask)
    if kind == 'ambient':
        compare_outputs({'state':final_ambient(model,observed,mask)}, {'state':reference[-1]})
    else:
        expected={k:v if k == 'prior_logits' else v[-1] for k,v in reference.items()}
        compare_outputs(final_candidates(model,observed,mask),expected)


def test_masked_values_do_not_change_final_state():
    model=model_for()
    data=StructuredFlowDataset(5,seed=124)
    observed=data.observed.clone()
    observed[data.mask == 0] = 99999.
    compare_outputs(final_candidates(model,observed,data.mask),final_candidates(model,data.observed,data.mask))
