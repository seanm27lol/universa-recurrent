"""Bound host conversion without changing any receipt or hiding runtime pauses."""
from __future__ import annotations
import gc
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
bounded=importlib.import_module('benchmark_bounded_receipts')
original=bounded.original


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    from universa_recurrent.neural.dual_study import calibrate,load_estimators,write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    root=tmp_path_factory.mktemp('bounded')
    cp=root/'weights-6100.pt'
    torch.set_num_threads(1)
    train_v2_checkpoint(cp,device_name='cpu',seed=6100,train_size=16,calibration_size=8,
        epochs=1,batch_size=8,hidden_dim=8,candidate_embedding_dim=2,steps=8,include_controls=True)
    engines,meta,device,_=load_estimators(cp,'cpu')
    data=StructuredFlowDataset(65,seed=6600)
    policies,artifact=calibrate(engines,data,device,.75,65)
    artifact['checkpoint_sha256']=meta['checkpoint_sha256']
    artifact['model_sources']={e.name:e.checkpoint_sha256 for e in engines if e.structured}
    cal=root/'study-6100/calibration.json';write_json(cal,artifact)
    return SimpleNamespace(root=root,cp=cp,cal=cal,meta=meta,data=data,
        policies=policies,engines={e.name:e for e in engines})


CASES=[('shared','endpoint'),('shared','trajectory'),('fixed_depth_4','endpoint'),
       ('fixed_depth_4','trajectory'),('direct','endpoint')]


def inputs(f,model,kind,count):
    engine=f.engines[model];obs,mask=f.data.observed[:count],f.data.mask[:count]
    output,history,_=original.infer(engine,f.policies[model],obs,mask,kind)
    return (engine,f.meta,f.policies[model],obs,mask,output,history,original.digest(f.cal))


@pytest.mark.parametrize('model,kind',CASES)
@pytest.mark.parametrize('count',[1,17,65])
@pytest.mark.parametrize('chunk_size',[1,16,64,4096])
def test_exact_individual_bytes(fixture,model,kind,count,chunk_size):
    args=inputs(fixture,model,kind,count)
    old=original.generate_receipts(*args)
    new=bounded.generate_bounded(*args,chunk_size=chunk_size)
    assert old==new and len(new)==count


@pytest.mark.parametrize('value',[0,-1,True,1.5,4097])
def test_bad_chunk_size_rejected(fixture,value):
    with pytest.raises(ValueError,match='chunk_size'):
        bounded.generate_bounded(*inputs(fixture,'shared','trajectory',17),chunk_size=value)


def test_chunk_limit_is_on_host_conversion_not_gpu_inference(fixture,monkeypatch):
    args=inputs(fixture,'shared','trajectory',65)
    calls=[];real=bounded.generate_payloads
    def observed(*a,**kw):
        calls.append(len(a[3]))
        assert a[3].device.type=='cpu' and a[0].bases.device.type=='cpu'
        assert all(t.device.type=='cpu' for t in a[5].values())
        assert all(t.device.type=='cpu' for t in a[6].values())
        return real(*a,**kw)
    monkeypatch.setattr(bounded,'generate_payloads',observed)
    bounded.generate_bounded(*args,chunk_size=16)
    assert calls==[16,16,16,16,1]


def test_no_mutation_of_tensors_or_weights(fixture):
    args=inputs(fixture,'shared','trajectory',17)
    engine,_,_,obs,mask,output,history,_=args
    tensors=[obs,mask,*output.values(),*history.values(),*engine.model.state_dict().values()]
    copies=[v.clone() for v in tensors]
    bounded.generate_bounded(*args,chunk_size=16)
    assert all(torch.equal(a,b) for a,b in zip(tensors,copies))


def test_length_mismatch_is_rejected(fixture):
    args=list(inputs(fixture,'shared','trajectory',17))
    args[5]=dict(args[5],estimate=args[5]['estimate'][:1])
    with pytest.raises(ValueError,match='length'):bounded.generate_bounded(*args)


def test_observer_restores_callbacks_even_on_failure():
    callbacks=list(gc.callbacks);settings=(gc.isenabled(),gc.get_threshold(),gc.get_freeze_count())
    with pytest.raises(RuntimeError):
        with bounded.GCProbe():raise RuntimeError('test exception')
    assert gc.callbacks==callbacks
    assert settings==(gc.isenabled(),gc.get_threshold(),gc.get_freeze_count())


def test_observer_captures_collection_without_subtraction():
    with bounded.GCProbe() as probe:
        probe.stage='test_collection'
        start=time.perf_counter_ns()
        cycle=[];cycle.append(cycle);del cycle
        gc.collect(0)  # Deliberate test trigger; the benchmark never forces collection.
        end=time.perf_counter_ns()
        info=probe.observations(start,end)
    assert info['events'] and info['observed_gc_ms']>=0
    assert all(e['stage']=='test_collection' and e['duration_ms']>=0 for e in info['events'])


def test_cpu_worker_all_paths_and_separate_diagnostics(fixture):
    f=fixture
    args=SimpleNamespace(checkpoint=f.cp,calibration=f.cal,device='cpu',cpu_threads=1,
        test_seed=6700,record_counts=[1,17],models=['shared','fixed_depth_4','direct'],
        order_seed=9,repeats=1,diagnostic_repeats=1,warmup=0)
    result=bounded.worker(args)
    assert len(result['rows'])==10 and result['gc_policy_unchanged']
    for row in result['rows']:
        assert row['identical_outputs_and_bytes'] and row['rejection_tests']['both_rejected_every_trial']
        for mode in bounded.MODES:
            assert len(row['timings'][mode]['samples_ms'])==1
            plain=row['samples'][mode][0];diagnostic=row['diagnostics'][mode][0]
            assert 'gc' not in plain and 'gc' in diagnostic
            for sample in [plain,diagnostic]:
                assert sample['checked_count']==sample['accepted_count']==row['record_count']
                assert sample['total_ms']>=sum(sample['stages_ms'].values())
                assert sample.get('gc',{}).get('observed_gc_ms',0)<=sample['total_ms']


def test_parent_subprocess_and_refuse_overwrite(fixture,tmp_path):
    out=tmp_path/'out'
    args=['--replication-dir',str(fixture.root),'--output-dir',str(out),'--device','cpu',
          '--models','direct','--record-counts','1','3','--repeats','1','--warmup','0',
          '--diagnostic-repeats','1','--test-seed','6700']
    assert bounded.main(args)==0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['checkpoint_runs']==1 and summary['all_outputs_and_bytes_agree']
    assert summary['all_gc_policies_unchanged']
    assert not list(out.rglob('*.pt'))
    assert bounded.main(args)==2


def test_missing_reference_and_duplicate_counts(tmp_path):
    args=['--replication-dir',str(tmp_path),'--output-dir',str(tmp_path/'out')]
    assert bounded.main(args)==2
    assert bounded.main(args+['--record-counts','1','1'])==2
