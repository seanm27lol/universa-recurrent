"""Batching must not hide a bad worksheet behind good neighbors."""
from __future__ import annotations
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
batch = importlib.import_module('batch_receipt_verifier')
pipeline = importlib.import_module('benchmark_batched_pipeline')
original = importlib.import_module('benchmark_verified_pipeline')


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    from universa_recurrent.neural.dual_study import load_estimators, calibrate, write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    root = tmp_path_factory.mktemp('batch-receipts')
    cp = root/'weights-6100.pt'
    torch.set_num_threads(1)
    train_v2_checkpoint(cp, device_name='cpu', seed=6100, train_size=16,
        calibration_size=8, epochs=1, batch_size=8, hidden_dim=8,
        candidate_embedding_dim=2, steps=8, include_controls=True)
    engines, meta, device, _ = load_estimators(cp,'cpu')
    data=StructuredFlowDataset(8,seed=6500)
    policies, artifact=calibrate(engines,data,device,.75,8)
    artifact['checkpoint_sha256']=meta['checkpoint_sha256']
    artifact['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    cal=root/'study-6100/calibration.json';write_json(cal,artifact)
    context=batch.PreparedVerifier.prepare(cp,cal)
    by_name={e.name:e for e in engines}
    groups={}
    for model in ('shared','fixed_depth_4','direct'):
        engine=by_name[model]
        for kind in (('endpoint',) if model=='direct' else ('endpoint','trajectory')):
            output, history, _=original.infer(engine,policies[model],data.observed,data.mask,kind)
            payloads=original.generate_receipts(engine,meta,policies[model],data.observed,data.mask,
                output,history,original.digest(cal))
            expected=batch.expected_inputs(data.observed,data.mask,model,kind)
            groups[(model,kind)]=(payloads,expected,output,history)
    return SimpleNamespace(root=root,cp=cp,cal=cal,context=context,
        verifier=batch.BatchPreparedVerifier(context),groups=groups,engines=by_name,
        meta=meta,policies=policies,data=data)


@pytest.mark.parametrize('model,kind', [('shared','endpoint'),('shared','trajectory'),
    ('fixed_depth_4','endpoint'),('fixed_depth_4','trajectory'),('direct','endpoint')])
@pytest.mark.parametrize('count',[1,4,8])
def test_generator_bytes_and_all_verdicts(fixture,model,kind,count):
    f=fixture; payloads,expectations,output,history=f.groups[(model,kind)]
    generated=batch.generate_payloads(f.engines[model],f.meta,f.policies[model],
        f.data.observed,f.data.mask,output,history,original.digest(f.cal))
    assert generated==payloads
    old=batch.scalar_batch(f.context,payloads[:count],expected_count=count,expected=expectations[:count])
    new=f.verifier.verify_payloads(payloads[:count],expected_count=count,expected=expectations[:count])
    assert old['accepted'] and new['accepted'] and new['checked_count']==count
    assert new['vectorized_records']==count
    assert pipeline.verdict_signature(old)==pipeline.verdict_signature(new)
    for left,right in zip(old['verdicts'],new['verdicts']):
        assert left['checks']==right['checks']


MUTATIONS = [
 ('bad_format',lambda r:r.__setitem__('format','bad')),
 ('outer_extra',lambda r:r.__setitem__('x',1)),
 ('scope_bool',lambda r:r['scope'].__setitem__('diagnostics_checked',1)),
 ('claim_certificate',lambda r:r['scope'].__setitem__('estimate_structurally_certified',True)),
 ('end_extra',lambda r:r['endpoint'].__setitem__('x',1)),
 ('end_format',lambda r:r['endpoint'].__setitem__('format','bad')),
 ('end_scope',lambda r:r['endpoint']['scope'].__setitem__('execution_authenticated',True)),
 ('checkpoint',lambda r:r['endpoint'].__setitem__('checkpoint_sha256','0'*64)),
 ('calibration',lambda r:r['endpoint'].__setitem__('calibration_sha256','0'*64)),
 ('depth_bool',lambda r:r['endpoint'].__setitem__('trained_depth',True)),
 ('depth',lambda r:r['endpoint'].__setitem__('trained_depth',99)),
 ('model',lambda r:r['endpoint'].__setitem__('model','missing')),
 ('policy',lambda r:r['endpoint']['policy'].__setitem__('threshold',None)),
 ('coverage_bool',lambda r:r['endpoint']['policy'].__setitem__('target_coverage',True)),
 ('name',lambda r:r['endpoint']['library']['names'].__setitem__(0,'different')),
 ('boundary',lambda r:r['endpoint']['library']['boundaries'][0][0].__setitem__(0,999)),
 ('estimate',lambda r:r['endpoint']['estimate'].__setitem__(0,123)),
 ('huge_estimate',lambda r:r['endpoint']['estimate'].__setitem__(0,1e308)),
 ('invalid_mask',lambda r:r['endpoint']['mask'].__setitem__(0,2)),
 ('no_observation',lambda r:r['endpoint'].__setitem__('mask',[0]*5)),
 ('bad_p',lambda r:r['endpoint']['probabilities'].__setitem__(0,-.01)),
 ('non_normalized',lambda r:r['endpoint'].__setitem__('probabilities',[.1,.1])),
 ('bad_state',lambda r:r['endpoint']['candidate_states'][0].__setitem__(0,77)),
 ('wrong_state_shape',lambda r:r['endpoint']['candidate_states'][0].append(1)),
 ('missing_mask',lambda r:r['endpoint'].pop('mask')),
 ('false_history',lambda r:r.__setitem__('trajectory',{})),
]


@pytest.mark.parametrize('name,edit',MUTATIONS,ids=[v[0] for v in MUTATIONS])
def test_one_bad_final_record_matches_scalar(fixture,name,edit):
    payloads,expectations,*_=fixture.groups[('shared','endpoint')]
    r=copy.deepcopy(original.loads(payloads[-1]));edit(r)
    bad=payloads[:-1]+[original.dumps(r)]
    a=batch.scalar_batch(fixture.context,bad,expected_count=8,expected=expectations)
    b=fixture.verifier.verify_payloads(bad,expected_count=8,expected=expectations)
    assert [v['accepted'] for v in a['verdicts']]==[v['accepted'] for v in b['verdicts']]
    assert b['checked_count']==8 and all(v['accepted'] for v in b['verdicts'][:-1])
    assert not b['accepted']


@pytest.mark.parametrize('field',list(original.FIELDS)+['bases'])
def test_trajectory_tamper_in_last_record(fixture,field):
    payloads,expected,*_=fixture.groups[('shared','trajectory')]
    r=original.loads(payloads[-1]);a=np.asarray(r['trajectory'][field]);a.flat[0]+=10
    r['trajectory'][field]=a.tolist();bad=payloads[:-1]+[original.dumps(r)]
    scalar=batch.scalar_batch(fixture.context,bad,expected_count=8,expected=expected)
    vector=fixture.verifier.verify_payloads(bad,expected_count=8,expected=expected)
    assert not scalar['accepted'] and not vector['accepted']
    assert all(v['accepted'] for v in vector['verdicts'][:-1])


@pytest.mark.parametrize('payload',[b'{"a":1,"a":2}',b'{"a":NaN}',b'null',b'[]',b'{',b'\xff',b'{"a":1e309}',None])
def test_malformed_last_is_not_dropped(fixture,payload):
    valid,expected,*_=fixture.groups[('direct','endpoint')]
    bad=valid[:-1]+[payload]
    result=fixture.verifier.verify_payloads(bad,expected_count=8,expected=expected)
    assert result['checked_count']==8 and not result['accepted']
    assert all(v['accepted'] for v in result['verdicts'][:-1])


@pytest.mark.parametrize('change',['missing','extra','duplicate','reverse','empty'])
def test_external_request_count_order_and_inputs(fixture,change):
    ps,expected,*_=fixture.groups[('direct','endpoint')]
    bad={'missing':ps[:-1],'extra':ps+ps[-1:],'duplicate':ps[:-1]+ps[:1],
         'reverse':ps[::-1],'empty':[]}[change]
    result=fixture.verifier.verify_payloads(bad,expected_count=8,expected=expected)
    assert not result['accepted'] and result['checked_count']==len(bad)


@pytest.mark.parametrize('offset',[0, .25, .49,.51,.99,1.01,2,-.99])
def test_numerical_boundary_defers_to_unchanged_checker(fixture,offset):
    ps,_,*_=fixture.groups[('direct','endpoint')]
    r=original.loads(ps[0]);e=r['endpoint']
    target=np.einsum('k,kn->n',e['probabilities'],e['candidate_states'])
    e['estimate']=target.tolist();e['estimate'][0]+=offset*(1e-7+1e-6*abs(target[0]))
    a=fixture.context.verify(r)
    b=fixture.verifier.verify_payloads([original.dumps(r)],expected_count=1)
    assert a['accepted']==b['accepted']
    if abs(offset)>.51: assert b['scalar_fallback_records']==1


def test_mixed_models_and_retention_keep_order(fixture):
    ps=[];expected=[]
    for key in fixture.groups:
        a,b,*_=fixture.groups[key];ps+=a[:2];expected+=b[:2]
    result=fixture.verifier.verify_payloads(ps,expected_count=len(ps),expected=expected)
    assert result['accepted'] and result['vectorized_records']==10


def test_valid_records_do_not_call_scalar_verifier_or_neural_replay(fixture,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('no scalar/math replay on fast records')
    ps,expected,*_=fixture.groups[('shared','trajectory')]
    monkeypatch.setattr(batch.PreparedVerifier,'verify',forbidden)
    monkeypatch.setattr(original,'infer',forbidden)
    result=fixture.verifier.verify_payloads(ps,expected_count=8,expected=expected)
    assert result['accepted'] and result['vectorized_records']==8


def test_immutable_reference_manifest_copy(fixture):
    copy_=fixture.context.manifest();copy_['models'].clear()
    ps,ex,*_=fixture.groups[('shared','endpoint')]
    assert fixture.verifier.verify_payloads(ps,expected_count=8,expected=ex)['accepted']
    with pytest.raises(AttributeError):fixture.verifier.reference=None


def test_chunking_checks_every_member(fixture):
    ps,ex,*_=fixture.groups[('shared','trajectory')]
    result=fixture.verifier.verify_payloads(ps*17,expected_count=136,expected=ex*17)
    assert result['accepted'] and result['vectorized_records']==136


def test_randomized_differential_corruptions(fixture):
    rng=np.random.default_rng(42);ps,_,*_=fixture.groups[('shared','trajectory')]
    for _ in range(120):
        r=original.loads(ps[int(rng.integers(8))]);field=rng.choice(list(original.FIELDS))
        a=np.asarray(r['trajectory'][field]);a.flat[int(rng.integers(a.size))]+=rng.choice([1e-10,1e-6,.1,1e200])
        r['trajectory'][field]=a.tolist()
        assert fixture.context.verify(r)['accepted']==fixture.verifier.verify_payloads([original.dumps(r)],expected_count=1)['accepted']


def test_worker_actual_cpu_end_to_end(fixture):
    args=SimpleNamespace(checkpoint=fixture.cp,calibration=fixture.cal,device='cpu',cpu_threads=1,
        test_seed=6600,record_counts=[1,4],models=['shared','fixed_depth_4','direct'],order_seed=9,repeats=1,warmup=0)
    r=pipeline.worker(args)
    assert len(r['rows'])==10 and len(r['saved_receipts'])==5
    for row in r['rows']:
        assert row['per_record_verdicts_agree'] and row['identical_receipt_bytes']
        assert row['rejection_tests']['both_rejected_every_trial']
        for mode,samples in row['samples'].items():
            assert len(samples)==1
            s=samples[0]
            assert s['checked_count']==s['accepted_count']==row['record_count']
            assert s['total_ms']>=sum(s['stages_ms'].values())


def test_parent_cli_and_refuses_overwrite(fixture,tmp_path):
    out=tmp_path/'new'
    args=['--replication-dir',str(fixture.root),'--output-dir',str(out),
          '--models','direct','--record-counts','1','2','--repeats','1','--warmup','0',
          '--test-seed','6600','--device','cpu']
    assert pipeline.main(args)==0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['all_outputs_and_receipts_agree'] and summary['checkpoint_runs']==1
    assert pipeline.main(args)==2


@pytest.mark.parametrize('bad',[0,-1,True,1.5,4097])
def test_invalid_request_count(bad,fixture):
    with pytest.raises(ValueError):fixture.verifier.verify_payloads([],expected_count=bad)


def test_finite_logits_with_overflow_cannot_fast_pass(fixture):
    ps,*_=fixture.groups[('shared','trajectory')]
    r=original.loads(ps[0]);h=r['trajectory'];e=r['endpoint']
    h['route_logits']=[[1e308,-1e308] for _ in h['route_logits']]
    h['route_probabilities']=[[1.,0.] for _ in h['route_probabilities']]
    e['probabilities']=[1.,0.];e['estimate']=list(e['candidate_states'][0])
    e['structure_claim']={'index':0,'name':e['library']['names'][0], 'state':list(e['candidate_states'][0])}
    assert not fixture.context.verify(r)['accepted']
    result=fixture.verifier.verify_payloads([original.dumps(r)],expected_count=1)
    assert not result['accepted'] and result['scalar_fallback_records']==1


def test_ties_and_rejection_policy(fixture):
    # A deliberately tied but consistent endpoint exercises exactly the legacy
    # first-index decision rule, not a new scientific rule for claim confidence.
    ps,*_=fixture.groups[('shared','endpoint')]
    r=original.loads(ps[0]);e=r['endpoint'];e['probabilities']=[.5,.5]
    e['estimate']=np.mean(e['candidate_states'],axis=0).tolist()
    threshold=e['policy']['threshold']
    e['structure_claim']=({'index':0,'name':e['library']['names'][0],
        'state':e['candidate_states'][0]} if threshold is not None and .5>=threshold else None)
    assert fixture.context.verify(r)['accepted']
    assert fixture.verifier.verify_payloads([original.dumps(r)],expected_count=1)['accepted']


def test_batch_bytes_limit(fixture,monkeypatch):
    ps,*_=fixture.groups[('direct','endpoint')]
    monkeypatch.setattr(batch,'MAX_BATCH_BYTES',5)
    with pytest.raises(ValueError,match='bytes limit'):
        fixture.verifier.verify_payloads(ps,expected_count=8)


def test_input_mutation_is_not_a_cached_verdict(fixture):
    ps,expected,*_=fixture.groups[('shared','endpoint')]
    assert fixture.verifier.verify_payloads(ps,expected_count=8,expected=expected)['accepted']
    r=original.loads(ps[-1]);r['endpoint']['estimate'][0]+=1
    bad=ps[:-1]+[original.dumps(r)]
    assert not fixture.verifier.verify_payloads(bad,expected_count=8,expected=expected)['accepted']


def test_changed_reference_bytes_require_new_preparation(fixture,tmp_path):
    cp=tmp_path/'weights.pt';cp.write_bytes(fixture.cp.read_bytes())
    cal=tmp_path/'cal.json';cal.write_bytes(fixture.cal.read_bytes())
    verifier=batch.BatchPreparedVerifier.prepare(cp,cal)
    ps,*_=fixture.groups[('shared','endpoint')]
    cal.write_text('{}')
    # The original pinned edition remains valid; it does not follow path changes.
    assert verifier.verify_payloads(ps,expected_count=8)['accepted']
    with pytest.raises(ValueError):batch.BatchPreparedVerifier.prepare(cp,cal)


def test_legacy_single_record_parser_resource_limit(fixture,monkeypatch):
    # Limits apply before invoking a huge parser, preserving per-record result.
    class NotBytes: pass
    result=fixture.verifier.verify_payloads([NotBytes()],expected_count=1)
    assert not result['accepted'] and result['checked_count']==1
