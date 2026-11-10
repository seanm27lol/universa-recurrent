import json
from pathlib import Path

import pytest

from native import prepare_program_protocol as P
from native import run_program_study
from native.program_study import _make_example as make_example
from native.structured import Instruction


REPO = Path(__file__).parents[1]
BEFORE = {'native_remaining_hours': 8.214045098002895}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def study_fixture(tmp_path):
    root = tmp_path / 'data'
    tokenizer = root / 'tokenizer.json'
    write_json(tokenizer, {'fixture': 'tokenizer-only; no tokenization is performed here'})
    manifest = {
        'position_policy': 'document',
        'train_epoch': {'loss_tokens_per_epoch': 348165, 'updates_per_epoch': 212, 'batch_size': 8},
        'splits': {'dev': {'rows': 51}},
        'audit_file': str(root / 'audit.json'),
        'tokenizer': {'path': str(tokenizer), 'sha256': P.sha256_file(tokenizer)},
        'source_identity': {'fixture': 'self-contained executable programs'},
        'selections': {'evaluation_example_ids': {}, 'intervention_example_ids': {}},
    }
    operations = ('set', 'add', 'multiply', 'copy')
    for split in P.SPLITS:
        records = []
        for index in range(158 if split == 'train' else 128):
            operation = operations[index % 4]
            instruction = (Instruction('set', 0, value=index % 32) if operation == 'set'
                           else Instruction(operation, 0, src=1))
            initial = (index % 24, (index // 4) % 24, (index // 8) % 24, (index // 16) % 24)
            example = make_example(split, [Instruction('copy', 1, src=0), instruction], initial)
            records.append(example.to_record())
        (root / 'structured').mkdir(parents=True, exist_ok=True)
        (root / 'structured' / f'{split}.jsonl').write_text(
            ''.join(json.dumps(record) + '\n' for record in records))
        manifest['selections']['evaluation_example_ids'][split] = [
            record['example_id'] for record in records[:128]]
        selected = records[-32:] if split == 'train' else records[:32]
        manifest['selections']['intervention_example_ids'][split] = [
            record['example_id'] for record in selected]
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'audit.json', {'fixture': True})
    write_json(root / 'selections.json', manifest['selections'])
    reference = tmp_path / 'reference'
    measured = {
        'dense': (23379.84890727775, 222.49587756663095, 7.680006882059388,
                  303.7991662049899, 0.010124170978087932),
        'untyped': (26040.885961792912, 199.75971660995856, 8.010237285867333,
                    284.30868265707977, 0.012363662011921406),
        'typed': (25002.740884884694, 208.05398991855327, 7.3681500618113205,
                  297.1710329030175, 0.012401053449138999),
    }
    for mode, (throughput, training, checkpoint, wall, dev_batch) in measured.items():
        arm = reference / 'arms' / mode
        write_json(arm / 'train-result.json', {
            'actual_loss_tokens_per_second': throughput,
            'training_seconds': training,
            'checkpoint_seconds': checkpoint,
            'dev': {'forward_seconds': dev_batch * 2, 'batches': 2},
        })
        write_json(arm / 'status.json', {'wall_seconds': wall})
        write_json(arm / 'evaluation.json', {
            'splits': {split: {'generation': {'evaluated': 32}} for split in ('train', 'dev')},
        })
    return root, tokenizer, reference


def test_measured_projection_covers_all_work_and_preserves_margin(tmp_path):
    root, _, reference = study_fixture(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    result = P.compute_projection(manifest, reference, before=BEFORE)
    assert result['epochs_per_arm'] == 16
    assert result['per_arm_seconds']['training'] == pytest.approx(5570640 / 23379.84890727775)
    assert result['per_arm_seconds']['generation_answer_loss_and_state_evaluation'] == pytest.approx(
        640 * 1.2773264519164513 * 1.5)
    assert result['per_arm_seconds']['targeted_interventions'] == pytest.approx(
        160 * 1.2773264519164513 * 1.5)
    assert result['per_arm_seconds']['supplemental_intervention_baselines'] == pytest.approx(
        30 * 1.2773264519164513 * 1.5)
    assert result['per_arm_seconds']['independent_rescoring'] > 0
    assert result['measurement_inputs']['independent_scoring']['scored_outputs_per_arm'] == 830
    assert result['measurement_inputs']['independent_scoring']['supplemental_baseline_generations_per_arm'] == 30
    assert result['per_arm_seconds']['periodic_and_final_dev'] > 0
    assert result['per_arm_seconds']['checkpoint'] == pytest.approx(8.010237285867333)
    assert result['total_seconds_all_arms'] == pytest.approx(3 * result['per_arm_seconds']['total'])
    assert result['projected_total_hours'] == pytest.approx(result['total_seconds_all_arms'] / 3600)
    assert result['allocation_request_hours'] == 2.5
    assert result['planning_margin_fraction'] >= 0.2
    assert result['required_hours_with_shutdown'] <= result['allocation_request_hours']
    assert result['fits_live_native_ledger']


def test_protocol_matches_dry_runner_and_refuses_overwrite(tmp_path):
    root, tokenizer, reference = study_fixture(tmp_path)
    output = tmp_path / 'study_protocol.json'
    protocol = P.prepare(root, REPO, tokenizer, reference, before=BEFORE, output=output)
    run_program_study.validate_protocol(protocol)
    plan = run_program_study.projection(protocol, BEFORE)
    assert plan['fits_budget']
    assert protocol['status'] == 'proposed_not_authorized_or_launched'
    assert protocol['allocation_request_hours'] == 2.5
    assert protocol['data']['manifest_sha256'] == P.sha256_file(root / 'manifest.json')
    assert all((REPO / 'research/ti_tnn/native' / name).is_file()
               for name in protocol['source_files_sha256'])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        P.prepare(root, REPO, tokenizer, reference, before=BEFORE, output=output)
    assert output.read_bytes() == original


def test_live_ledger_shortfall_blocks_protocol_creation(tmp_path):
    root, tokenizer, reference = study_fixture(tmp_path)
    output = tmp_path / 'study_protocol.json'
    with pytest.raises(RuntimeError, match='does not fit'):
        P.prepare(root, REPO, tokenizer, reference, before={'native_remaining_hours': 0.1}, output=output)
    assert not output.exists()
