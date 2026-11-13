"""CPU orchestration fixtures: no neural models, real GPU reservation or training."""
import copy
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/ti_tnn'))
from native import checkpoint as C, data as D, ledger, train, run_mixed_learning as R
from native.config import NativeConfig


def protocol_fixture(root):
    p = {'format': R.FORMAT, 'modes': list(R.MODES), 'objective': 'full_sequence',
         'initialization': {'kind': 'fresh_random', 'parent_checkpoint': None, 'seed': 20261108},
         'model_seed': 20261108, 'sampler_seed': 20261108, 'endpoint_per_run': dict(R.ENDPOINT),
         'lambda_balance': .01, 'lambda_route': .1, 'lambda_state': .1,
         'optimizer': {'learning_rate': .0003, 'weight_decay': .01, 'betas': [.9, .999], 'eps': 1e-8, 'grad_clip': 1.},
         'scheduler': {'warmup_fraction': .02, 'final_lr_fraction': .1},
         'checkpoint_every': 500, 'dev_every': 1000, 'dev_batches': 16,
         'preparation_sha256': 'data-manifest', 'artifact_sha256': {},
         'capability_loss_tokens': {'wikipedia': 28877, 'code': 32640},
         'resource_projection': {'per_model_per_condition_seconds': {m: {'total': 300.} for m in R.MODES},
             'common_seconds': {'fixture': 60.}, 'shutdown_allowance_hours': .05},
         'execution': {'study_id': R.STUDY_ID, 'source_commit': 'source', 'source_files': {},
             'environment_prefix': sys.prefix, 'python': sys.executable, 'installed_versions': {},
             'preparation': str(root/'data'), 'output': str(root/'comparison'),
             'allocation_hours': R.ALLOCATION, 'native_cap_hours': 12., 'training_configs': {}}}
    for mode in R.MODES:
        args = R.arguments(p, root/'data', root/'unused', mode)
        model = NativeConfig.from_json(args.config).to_dict();model['mode'] = mode
        sampler = SimpleNamespace(data_identity='data', order_identity='order')
        p['execution']['training_configs'][mode] = train.fresh_run_config(args, NativeConfig(**model),
            root/'data', sampler, sampler, 19862, 40001786, 'code')
    return p


def evaluation_fixture(p, checkpoint):
    return {'complete': True, 'checkpoint_sha256': checkpoint['sha256'], 'preparation_sha256': p['preparation_sha256'],
        'capability': {source: {'teacher_forced': {'complete': True, 'sequences': 128,
            'sources': {source: {'loss_tokens': n}}}, 'continuation': {'complete': True, 'examples': 64,
            'token_decisions': 2048, 'missing': 0}} for source,n in p['capability_loss_tokens'].items()},
        'synthetic_diagnostics': {'complete': True, 'checkpoint_sha256': checkpoint['sha256'],
            'populations': {name: {'score': {'complete_answers': {'correct': 0, 'total': 128},
                 'registers': {'total': 512}}, 'answer_ce': {'complete': True}, 'state_readout': {'complete': True}}
                 for name in R.POPULATIONS}, 'reconstruction': {'agreement': True}}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    p = protocol_fixture(tmp_path)
    (tmp_path/'data').mkdir();R.write_json(tmp_path/'data/manifest.json',{})
    path, lock = tmp_path/'ledger.jsonl', tmp_path/'gpu.lock';lock.touch()
    path.write_text(json.dumps({'event': 'start', 'reservation': 'history', 't': 1., 'cap_hours': ledger.CAP_HOURS})+'\n'
        +json.dumps({'event': 'end', 'reservation': 'history', 't': 2., 'elapsed': ledger.HISTORICAL_HOURS*3600,
                     'status': 'complete'})+'\n')
    monkeypatch.setattr(ledger, 'validate_shared_reservation_paths', lambda: (path, lock))
    monkeypatch.setattr(ledger._base, 'foreign_gpu_processes', lambda: [])
    monkeypatch.setattr(R, 'preflight', lambda *a: p)
    monkeypatch.setattr(R, 'check_storage', lambda *a: 128*1024**3)
    calls=[]
    def fake_train(args):
        binding=args.mixed_learning_binding
        config=train.bind_mixed_learning_owner(binding['expected_config'], binding, ledger.active())
        calls.append(args)
        folder=Path(args.output);final=folder/'checkpoints/final-update-00019862.pt'
        final.parent.mkdir(parents=True,exist_ok=True);final.write_bytes(b'CPU fixture, not model weights')
        R.write_json(folder/'run_config.json',config)
        return {'mode':args.mode,'loss_mode':args.loss_mode,'step':19862,'planned_updates':19862,'cursor':158896,
            'max_loss_tokens':40000000,'consumed_loss_tokens':40001786,
            'final_checkpoint':{'path':str(final),'sha256':D.sha256_file(final)}}
    def fake_load(path, **kw):
        assert D.sha256_file(path)==kw['expected_sha256']
        config=R.read_json(Path(path).parents[1]/'run_config.json')
        assert config==kw['expected_config']
        assert config['code_identity']==kw['expected_code_identity']
        assert config['train_data_identity']==kw['expected_data_identity']
        assert config['order_identity']==kw['expected_order_identity']
        assert kw['expected_identities']=={'dev_data':config['dev_data_identity'],
            'data_manifest':D.sha256_file(Path(config['data'])/'manifest.json')}
        return {'step':19862,'cursor':158896,'consumed_loss_tokens':40001786}
    def fake_evaluate(checkpoint, data, tokenizer, output):
        output.mkdir(parents=True,exist_ok=False)
        report=evaluation_fixture(p,checkpoint);R.write_json(output/'evaluation.json',report)
        return report
    monkeypatch.setattr(train,'run_training',fake_train)
    monkeypatch.setattr(C,'load_checkpoint',fake_load)
    monkeypatch.setattr(R,'evaluate_endpoint',fake_evaluate)
    return SimpleNamespace(p=p,root=tmp_path,ledger=path,lock=lock,calls=calls,
        run=lambda **kw:R.run(tmp_path/'protocol.json',tmp_path/'tokenizer','protocol',**kw))


def test_real_margin_blocks_over_budget_without_any_mutation(harness,monkeypatch):
    for component in harness.p['resource_projection']['per_model_per_condition_seconds'].values():component['total']=3.1860077914841516*3600/4
    harness.p['resource_projection']['common_seconds']={'reserved_setup':540.}
    before=harness.ledger.read_bytes();lock=harness.lock.read_bytes()
    monkeypatch.setattr(ledger,'Reservation',lambda **kw:pytest.fail('blocked/dry run reserved'))
    result=harness.run()
    assert result['status']=='blocked_budget' and result['margin']==.5
    assert result['required_remaining_hours']==pytest.approx(5.054011687226227)
    assert result['allocation_shortfall_hours']==pytest.approx(.304011687226227)
    with pytest.raises(ledger.BudgetExceeded):harness.run(execute=True,authorize_hours=4.75)
    assert not harness.calls and harness.ledger.read_bytes()==before and harness.lock.read_bytes()==lock
    assert not (harness.root/'comparison').exists()


def test_dry_run_is_read_only_even_when_budget_fits(harness,monkeypatch):
    before=harness.ledger.read_bytes()
    monkeypatch.setattr(ledger,'Reservation',lambda **kw:pytest.fail('dry run reserved'))
    result=harness.run()
    assert result['status']=='dry_run' and not result['training_launched'] and not result['reservation_created']
    assert not harness.calls and harness.ledger.read_bytes()==before and not (harness.root/'comparison').exists()


@pytest.mark.parametrize('hours',[None,4,5,True,float('nan')])
def test_explicit_exact_authorization_required(harness,hours):
    with pytest.raises(PermissionError):harness.run(execute=True,authorize_hours=hours)
    assert not harness.calls


def test_four_arms_fixed_exposure_and_duplicate_rejection(harness):
    result=harness.run(execute=True,authorize_hours=4.75)
    assert result['complete'] and list(result['arms'])==list(R.MODES)
    for mode,args in zip(R.MODES,harness.calls):
        assert args.mode==mode and args.max_loss_tokens==40000000 and args.loss_mode=='full_sequence'
        assert args.model_seed==args.data_seed==20261108
        assert args.lambda_route==args.lambda_state==.1 and args.lambda_balance==.01
        assert args.mixed_learning_binding['expected_config']['training']['planned_consumed_loss_tokens']==40001786
        assert args.mixed_learning_binding['expected_config']['training']['planned_updates']==19862
    with pytest.raises(FileExistsError):harness.run(execute=True,authorize_hours=4.75)
    assert harness.run(execute=True,authorize_hours=4.75,resume=True)['status']=='already_complete'
    assert len(harness.calls)==4


def test_failure_and_recovery_keep_prior_cost_and_incomplete_arms(harness,monkeypatch):
    real=R.evaluate_endpoint
    monkeypatch.setattr(R,'evaluate_endpoint',lambda *a:(_ for _ in ()).throw(RuntimeError('fixture evaluation failure')))
    with pytest.raises(RuntimeError,match='fixture evaluation failure'):harness.run(execute=True,authorize_hours=4.75)
    first=harness.root/'comparison/attempt-0001.json';before=first.read_bytes();status=json.loads(before)
    assert not status['complete'] and status['arms']['dense']['status']=='failed'
    assert all(not arm['complete'] for arm in status['arms'].values())
    prior=ledger.usage(harness.ledger)['native_used_hours']
    monkeypatch.setattr(R,'evaluate_endpoint',real)
    result=harness.run(execute=True,authorize_hours=4.75,resume=True)
    assert result['complete'] and result['plan']['prior_charged_hours']==pytest.approx(prior)
    assert len(harness.calls)==4 and first.read_bytes()==before
    assert ledger.usage(harness.ledger)['native_used_hours']>=prior


def test_final_checkpoint_result_gap_does_not_retrain(harness,monkeypatch):
    original=train.run_training;saved={}
    def interrupt(args):
        saved.update(original(args));raise RuntimeError('after final journal')
    monkeypatch.setattr(train,'run_training',interrupt)
    with pytest.raises(RuntimeError,match='after final journal'):harness.run(execute=True,authorize_hours=4.75)
    final=Path(saved['final_checkpoint']['path']);before=final.read_bytes()
    data=harness.root/'data'
    manifest={**saved['final_checkpoint'],'bytes':len(before)}
    monkeypatch.setattr(C,'recover',lambda *a,**kw:SimpleNamespace(checkpoint_path=final,manifest=manifest,requires_replay=False,invalid_checkpoints=[]))
    monkeypatch.setattr(train,'run_training',lambda *a:pytest.fail('final checkpoint retrained'))
    recovered=R.recover_final_result(harness.p,R.owner(harness.p,'protocol'),'dense',final.parents[2])
    assert recovered['recovered_final_descriptor_only'] and final.read_bytes()==before
    monkeypatch.setattr(C,'recover',lambda *a,**kw:SimpleNamespace(checkpoint_path=final,manifest=manifest,requires_replay=True,invalid_checkpoints=[]))
    with pytest.raises(RuntimeError,match='refusing overwrite'):
        R.recover_final_result(harness.p,R.owner(harness.p,'protocol'),'dense',final.parents[2])


@pytest.mark.parametrize('field',['code_identity','train_data_identity','dev_data_identity','order_identity','model','training'])
def test_config_identity_attacks_rejected_and_owner_stable(harness,field):
    config=harness.p['execution']['training_configs']['dense'];identity=R.owner(harness.p,'protocol')
    binding={'owner':identity,'expected_config':config};reservation=SimpleNamespace(extra={'mixed_learning_study':identity})
    with pytest.raises(C.CheckpointIdentityMismatch):train.bind_mixed_learning_owner({**config,field:'changed'},binding,reservation)
    assert train.bind_mixed_learning_owner(config,binding,reservation)==train.bind_mixed_learning_owner(config,binding,reservation)
    with pytest.raises(RuntimeError,match='authorized mixed-learning owner'):
        train.bind_mixed_learning_owner(config,binding,SimpleNamespace(extra={'mixed_learning_study':identity,'initial_state_study':identity}))


def test_prior_cost_native_cap_claim_and_lock_fail_closed(harness):
    assert not R.projection(harness.p,{},4.5,12.)['fits']
    assert not R.projection(harness.p,{},0.,.1)['fits']
    identity=R.owner(harness.p,'protocol')
    with harness.ledger.open('a') as f:f.write(json.dumps({'event':'start','reservation':'open','t':3.,'extra':{'mixed_learning_study':identity}})+'\n')
    with pytest.raises(RuntimeError,match='open comparison'):harness.run()
    with pytest.raises(RuntimeError,match='different immutable'):
        R.prior_costs(harness.ledger,{**identity,'output':'another'})


def test_active_lock_and_under_lock_budget_race_stop_before_training(harness,monkeypatch):
    with harness.lock.open('r') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='conflicting'):harness.run()
    original=ledger.Reservation
    class Race(original):
        def __enter__(self):
            result=super().__enter__();self.used_before=ledger.CAP_HOURS-.01;return result
    monkeypatch.setattr(ledger,'Reservation',Race)
    with pytest.raises(ledger.BudgetExceeded):harness.run(execute=True,authorize_hours=4.75)
    assert not harness.calls
    attempt=R.read_json(harness.root/'comparison/attempt-0001.json')
    assert attempt['status']=='failed' and attempt['reservation_id']
    assert any(e.get('reservation')==attempt['reservation_id'] for e in ledger.read(harness.ledger))


@pytest.mark.parametrize('part',['language','generation','register','readout'])
def test_incomplete_evaluation_never_completes_arm(harness,part):
    ck={'sha256':'weights'};r=evaluation_fixture(harness.p,ck)
    if part=='language':r['capability']['code']['teacher_forced']['complete']=False
    elif part=='generation':r['capability']['code']['continuation']['missing']=1
    elif part=='register':r['synthetic_diagnostics']['populations'].pop('dev')
    else:r['synthetic_diagnostics']['populations']['dev']['state_readout']['complete']=False
    with pytest.raises(RuntimeError):R.validate_evaluation(harness.p,r,ck)


def test_changed_completion_artifact_cannot_be_skipped(harness):
    harness.run(execute=True,authorize_hours=4.75)
    path=harness.root/'comparison/arms/dense/training_result.json';path.write_text('{}')
    with pytest.raises(RuntimeError,match='identity mismatch'):harness.run(resume=True)


def test_cli_success_reports_completion_without_key_error(harness,monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['runner','--protocol','fixture','--tokenizer','fixture',
        '--expected-protocol-sha256','protocol','--execute','--authorize-hours','4.75'])
    assert R.main()==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='complete' and report['complete']


def test_resume_completed_between_checks_does_not_reserve(harness,monkeypatch):
    output=harness.root/'comparison';output.mkdir();R.write_json(output/'claim.json',R.owner(harness.p,'protocol'))
    calls=iter([{},dict.fromkeys(R.MODES,{})])
    monkeypatch.setattr(R,'completed_arms',lambda *a:next(calls))
    monkeypatch.setattr(ledger,'Reservation',lambda **kw:pytest.fail('completed comparison reserved'))
    assert harness.run(execute=True,authorize_hours=4.75,resume=True)['status']=='already_complete'
    assert not list(output.glob('attempt-*'))


@pytest.mark.parametrize('attack',[None,'protocol','source','environment','data','tokenizer','sampler','model','endpoint','objective'])
def test_read_only_preflight_enforces_frozen_identities(tmp_path,monkeypatch,attack):
    p=protocol_fixture(tmp_path)
    data=tmp_path/'data';data.mkdir();R.write_json(data/'manifest.json',{})
    tokenizer=tmp_path/'tokenizer';tokenizer.write_bytes(b'fixture tokenizer')
    p['preparation_sha256']=D.sha256_file(data/'manifest.json')
    p['tokenizer_sha256']=D.sha256_file(tokenizer)
    p['execution']['output']=str(tmp_path/'ti_tnn_native_mixed_learning_v1')
    configs=copy.deepcopy(p['execution']['training_configs'])
    monkeypatch.setattr(R.I,'source_commit',lambda *a,**kw:'source')
    monkeypatch.setattr(R.I,'locked_environment',lambda *a:{})
    monkeypatch.setattr(R,'training_configs',lambda *a:configs)
    monkeypatch.setattr(ledger,'shared_runs',lambda:tmp_path/'shared')
    monkeypatch.setattr(ledger,'Reservation',lambda **kw:pytest.fail('preflight reserved'))
    monkeypatch.setattr(train,'run_training',lambda *a:pytest.fail('preflight trained'))
    if attack=='source':p['execution']['source_commit']='changed'
    elif attack=='environment':p['execution']['installed_versions']={'changed':'version'}
    elif attack=='data':(data/'manifest.json').write_text('{"changed":true}')
    elif attack=='tokenizer':tokenizer.write_bytes(b'changed')
    elif attack=='sampler':p['execution']['training_configs']['typed']['order_identity']='changed'
    elif attack=='model':p['execution']['training_configs']['shared']['model']['hidden_size']=999
    elif attack=='endpoint':p['endpoint_per_run']['updates']-=1
    elif attack=='objective':p['lambda_state']=0
    protocol=tmp_path/'protocol.json';R.write_json(protocol,p)
    digest=D.sha256_file(protocol) if attack!='protocol' else 'changed'
    if attack is None:assert R.preflight(protocol,tokenizer,digest)==p
    else:
        with pytest.raises((RuntimeError,ValueError)):R.preflight(protocol,tokenizer,digest)
    assert not Path(p['execution']['output']).exists()
