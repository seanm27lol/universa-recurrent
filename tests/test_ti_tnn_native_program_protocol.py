import json
from pathlib import Path

import pytest

from native import prepare_program_protocol as P
from native import run_program_study


DATA = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization')
REPO = Path(__file__).parents[1]
TOKENIZER = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native/cache/tokenizer/tokenizer.json')
BEFORE = {'native_remaining_hours': 8.214045098002895}


def test_measured_projection_covers_all_work_and_preserves_margin():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    result = P.compute_projection(manifest, P.REFERENCE_RUN, before=BEFORE)
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
    output = tmp_path / 'study_protocol.json'
    protocol = P.prepare(DATA, REPO, TOKENIZER, before=BEFORE, output=output)
    run_program_study.validate_protocol(protocol)
    plan = run_program_study.projection(protocol, BEFORE)
    assert plan['fits_budget']
    assert protocol['status'] == 'proposed_not_authorized_or_launched'
    assert protocol['allocation_request_hours'] == 2.5
    assert protocol['data']['manifest_sha256'] == P.sha256_file(DATA / 'manifest.json')
    assert all((REPO / 'research/ti_tnn/native' / name).is_file()
               for name in protocol['source_files_sha256'])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        P.prepare(DATA, REPO, TOKENIZER, before=BEFORE, output=output)
    assert output.read_bytes() == original


def test_live_ledger_shortfall_blocks_protocol_creation(tmp_path):
    output = tmp_path / 'study_protocol.json'
    with pytest.raises(RuntimeError, match='does not fit'):
        P.prepare(DATA, REPO, TOKENIZER, before={'native_remaining_hours': 0.1}, output=output)
    assert not output.exists()
