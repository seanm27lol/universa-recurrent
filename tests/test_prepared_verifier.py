"""Amortize fixed-file setup, never reuse a previous receipt's verdict."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import pytest

torch = pytest.importorskip('torch')
PATH = Path(__file__).resolve().parents[1]/'scripts/benchmark_verifier_setup.py'
spec = importlib.util.spec_from_file_location('benchmark_verifier_setup', PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


@pytest.fixture(scope='module')
def sample(tmp_path_factory):
    from universa_recurrent.neural.v2_train import train_v2_checkpoint, load_v2_checkpoint
    from universa_recurrent.neural.dual_study import run_study, read_json
    from universa_recurrent.neural.retention_lingua import make_record
    root = tmp_path_factory.mktemp('prepared')
    cp = root/'model.pt'
    torch.set_num_threads(1)
    train_v2_checkpoint(cp, device_name='cpu', seed=4800, train_size=16, calibration_size=8,
                        epochs=1, batch_size=8, hidden_dim=8, steps=2, include_controls=True)
    run_study(cp, root/'study', device_name='cpu', calibration_size=8, n=8,
              calibration_seed=5000, test_seed=6000, batch_size=8, warmup=0, repeats=0)
    endpoint = read_json(root/'study/lingua.json')
    model, _, _, _ = load_v2_checkpoint(cp, device_name='cpu')
    observed = torch.tensor([endpoint['observed']])
    mask = torch.tensor([endpoint['mask']])
    with torch.inference_mode():
        trajectory = make_record(endpoint, history=model.rollout(observed, mask), bases=model.bases)
    return cp, root/'study/calibration.json', make_record(endpoint), trajectory


@pytest.mark.parametrize('kind',[2,3])
def test_full_existing_properties_still_run_and_match(sample, kind):
    from universa_recurrent.neural.retention_lingua import verify_record
    cp,cal,*_ = sample
    old=verify_record(sample[kind], checkpoint=cp, calibration=cal)
    new=mod.PreparedVerifier.prepare(cp,cal).verify(sample[kind])
    assert old['accepted'] and new['accepted']
    assert new['checks']==old['checks']
    assert new['checkpoint_bound'] and new['calibration_bound']
    assert not new['learned_transitions_verified'] and not new['estimate_structurally_certified']


@pytest.mark.parametrize('mutation',[
    'estimate','probabilities','candidate','claim','hash','cal_hash','depth','model',
    'policy','library','scope','mask','trajectory_state','trajectory_residual','trajectory_logit',
])
def test_changed_record_is_not_a_cached_pass(sample,mutation):
    cp,cal,_,trajectory=sample
    context=mod.PreparedVerifier.prepare(cp,cal)
    assert context.verify(trajectory)['accepted']
    r=copy.deepcopy(trajectory);e=r['endpoint']
    if mutation=='estimate': e['estimate'][0]+=1
    if mutation=='probabilities': e['probabilities']=[.1,.1]
    if mutation=='candidate': e['candidate_states'][0][0]+=1
    if mutation=='claim': e['structure_claim']={'index':True,'name':'bad','state':e['estimate']}
    if mutation=='hash':e['checkpoint_sha256']='a'*64
    if mutation=='cal_hash':e['calibration_sha256']='b'*64
    if mutation=='depth':e['trained_depth']+=1
    if mutation=='model':e['model']='absent'
    if mutation=='policy':e['policy']['target_coverage']=.123
    if mutation=='library':e['library']['names'][0]='other'
    if mutation=='scope':e['scope']['execution_authenticated']=True
    if mutation=='mask':e['mask'][0]=.5
    if mutation=='trajectory_state':r['trajectory']['states'][0][0][0]+=1
    if mutation=='trajectory_residual':r['trajectory']['residuals'][0][0]+=1
    if mutation=='trajectory_logit':r['trajectory']['route_logits'][0][0]+=10
    assert not context.verify(r)['accepted']


def test_no_files_or_network_replay_after_preparation(sample,monkeypatch):
    from universa_recurrent.neural import v2_train
    from universa_recurrent.neural.v2 import MultiHypothesisRecurrentNet
    cp,cal,r,_=sample
    context=mod.PreparedVerifier.prepare(cp,cal)
    def forbidden(*a,**kw):raise AssertionError('expensive source loading or replay in check')
    monkeypatch.setattr(v2_train,'load_v2_checkpoint',forbidden)
    monkeypatch.setattr(MultiHypothesisRecurrentNet,'rollout',forbidden)
    monkeypatch.setattr(mod,'digest',forbidden)
    assert context.verify(r)['accepted']


def test_snapshot_is_immutable_and_does_not_claim_live_file_binding(sample,tmp_path):
    from universa_recurrent.neural.retention_lingua import verify_record
    cp,cal,r,_=sample
    local=tmp_path/'cal.json';local.write_bytes(cal.read_bytes())
    prepared=mod.PreparedVerifier.prepare(cp,local)
    exposed=prepared.manifest();exposed['models']['shared']['depth']=999
    assert prepared.verify(r)['accepted']
    with pytest.raises(AttributeError):prepared._snapshot=b'{}'
    local.write_bytes(b'{}')
    assert prepared.verify(r)['accepted']  # intentionally pinned old edition
    assert not verify_record(r,checkpoint=cp,calibration=local)['accepted']
    with pytest.raises(ValueError):mod.PreparedVerifier.prepare(cp,local)


def test_wrong_calibration_checkpoint_refused(sample,tmp_path):
    cp,cal,_,_=sample
    d=json.loads(cal.read_text());d['checkpoint_sha256']='a'*64
    wrong=tmp_path/'cal.json';wrong.write_text(json.dumps(d))
    with pytest.raises(ValueError):mod.PreparedVerifier.prepare(cp,wrong)


def test_file_change_during_prepare_is_rejected(sample,monkeypatch):
    cp,cal,_,_=sample;original=mod.digest;count=0
    def changed(path):
        nonlocal count
        count+=1
        return original(path) if count<3 else '0'*64
    monkeypatch.setattr(mod,'digest',changed)
    with pytest.raises(ValueError,match='changed'):mod.PreparedVerifier.prepare(cp,cal)


@pytest.mark.parametrize('kind',[2,3])
def test_worker_smoke_and_counts(sample,kind):
    cp,cal,_,_=sample;r=sample[kind]
    worker={'training_seed':4800,'checkpoint_sha256':mod.digest(cp),'calibration_sha256':mod.digest(cal),
            'lingua_costs':[{'model':'shared','sample_count':1,
                'sample_records':{'endpoint':[sample[2]],'trajectory':[sample[3]]}}]}
    outcome=mod.benchmark_worker(worker,cp,cal,[1,2],1,7)
    assert outcome['accepted_decisions_and_existing_checks_agree']
    assert len(outcome['rows'])==4
    for row in outcome['rows']:
        assert row['source_record_pool_size'] == 1
        assert row['distinct_records_in_batch'] == 1
        assert set(row['timings'])=={'files_per_record','prepare_once_per_batch','reuse_prepared_snapshot'}
        for t in row['timings'].values():assert t['median_ms']>0


def test_zip_duplicate_members_rejected(tmp_path):
    path=tmp_path/'reports.zip'
    with ZipFile(path,'w') as z:
        z.writestr('a.json','{}')
        with pytest.warns(UserWarning):z.writestr('a.json','{}')
    with pytest.raises(ValueError,match='duplicate'):mod.read_workers(path)


def test_no_workers_rejected(tmp_path):
    path=tmp_path/'reports.zip'
    with ZipFile(path,'w') as z:z.writestr('empty.json','{}')
    with pytest.raises(ValueError):mod.read_workers(path)


def test_hash_function_and_timing_statistics(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'abc')
    assert mod.digest(p)==hashlib.sha256(b'abc').hexdigest()
    assert mod.stats([1,2,3])['median_ms']==2


def test_main_cli_writes_complete_without_weights(sample,tmp_path):
    cp,cal,_,_=sample
    root=tmp_path/'rep';(root/'study-4800').mkdir(parents=True)
    (root/'weights-4800.pt').write_bytes(cp.read_bytes())
    (root/'study-4800/calibration.json').write_bytes(cal.read_bytes())
    w={'format':'universa-recurrent.retention-study.v1.worker','batch_size':1024,'training_seed':4800,
       'dataset_sha256':'a'*64,'checkpoint_sha256':mod.digest(cp),'calibration_sha256':mod.digest(cal),
       'lingua_costs':[{'model':'shared','sample_count':1,'sample_records':{'endpoint':[sample[2]],'trajectory':[sample[3]]}}]}
    archive=tmp_path/'records.zip'
    with ZipFile(archive,'w') as z:z.writestr('worker.json',json.dumps(w))
    out=tmp_path/'out'
    args=['--reports',str(archive),'--replication-dir',str(root),'--output-dir',str(out),
          '--record-counts','1','--repeats','1']
    assert mod.main(args)==0
    summary = json.loads((out/'summary.json').read_text())
    assert summary['all_checks_agreed']
    assert summary['repeats'] == 1
    assert summary['warning'].startswith('1 measured repeats per condition;')
    assert not list(out.glob('*.pt'))
    assert mod.main(args)==2  # exclusive output directory
