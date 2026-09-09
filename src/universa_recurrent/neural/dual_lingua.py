"""A small lab notebook for an estimate and a separately scoped structural claim.

The NumPy checker does not run the neural model. It checks weighted-sum arithmetic,
candidate feasibility, and the declared threshold rule. With trusted local files
it binds that record to checkpoint/calibration bytes, not to remote execution.
"""
from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np

FORMAT = 'universa-recurrent.dual-lingua.v1'
SCOPE = {'estimate_is_weighted_mixture': True, 'estimate_has_structural_certificate': False,
         'claim_means_candidate_constraint_only': True, 'route_correctness_proven': False,
         'neural_update_replayed': False, 'execution_authenticated': False}


def _plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return np.asarray(value).tolist()


def make_record(prediction: dict, policy, observed, mask, names: list[str], boundaries,
                *, model_name: str, checkpoint_sha256: str, calibration_sha256: str,
                trained_depth: int) -> dict:
    if prediction['estimate'].shape[0] != 1:
        raise ValueError('one-example records only')
    accepted = bool(prediction['claim_mask'][0].item())
    index = int(prediction['top_route'][0].item())
    states = _plain(prediction['candidate_states'][0])
    claim = ({'index': index, 'name': names[index], 'state': states[index]} if accepted else None)
    return {'format': FORMAT, 'scope': dict(SCOPE), 'model': model_name,
            'checkpoint_sha256': checkpoint_sha256, 'calibration_sha256': calibration_sha256,
            'trained_depth': trained_depth, 'policy': policy.as_dict(),
            'observed': _plain(observed[0]), 'mask': _plain(mask[0]),
            'library': {'names': list(names), 'boundaries': _plain(boundaries)},
            'candidate_states': states, 'probabilities': _plain(prediction['probabilities'][0]),
            'estimate': _plain(prediction['estimate'][0]), 'structure_claim': claim}


def _array(value, dimensions: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.ndim != dimensions or array.size < 1 or array.size > 1_000_000:
        raise ValueError('invalid numeric array shape/type/size')
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError('nonfinite record value')
    return array


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def verify_record(record: dict, *, checkpoint: Path | None = None,
                  calibration: Path | None = None) -> dict:
    """Verify declared properties only. Hardcoded tolerances cannot be enlarged by a record."""
    checks = {}
    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(name)
    try:
        fields = {'format','scope','model','checkpoint_sha256','calibration_sha256',
                  'trained_depth','policy','observed','mask','library','candidate_states',
                  'probabilities','estimate','structure_claim'}
        require('schema', isinstance(record, dict) and set(record) == fields and record['format'] == FORMAT)
        require('scope', record['scope'] == SCOPE and all(type(record['scope'][k]) is bool for k in SCOPE))
        require('identifiers', _hash(record['checkpoint_sha256']) and _hash(record['calibration_sha256'])
                and isinstance(record['model'], str) and bool(record['model']))
        depth = record['trained_depth']
        require('depth', type(depth) is int and depth > 0)
        library = record['library']
        require('library_schema', isinstance(library, dict) and set(library) == {'names','boundaries'})
        names = library['names']
        require('names', isinstance(names, list) and len(names) >= 2 and
                all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names))
        states = _array(record['candidate_states'], 2)
        boundaries = _array(library['boundaries'], 3)
        probabilities = _array(record['probabilities'], 1)
        estimate = _array(record['estimate'], 1)
        observed = _array(record['observed'], 1)
        mask = _array(record['mask'], 1)
        k, n = states.shape
        require('dimensions', k == len(names) and probabilities.shape == (k,) and
                boundaries.shape[0] == k and boundaries.shape[-1] == n and
                estimate.shape == observed.shape == mask.shape == (n,))
        require('observations', np.isin(mask, [0.,1.]).all() and mask.sum() >= 1
                and np.all(observed[mask == 0] == 0))
        require('probabilities', ((probabilities >= 0) & (probabilities <= 1)).all()
                and abs(float(probabilities.sum()) - 1) <= 2e-6)
        # Small, fixed float32-aware tolerances. This is not a formal proof.
        residuals = np.linalg.norm(np.einsum('kmn,kn->km', boundaries, states), axis=1)
        scales = np.linalg.norm(boundaries, axis=(1,2)) * np.linalg.norm(states, axis=1)
        require('finite_geometry', np.isfinite(residuals).all() and np.isfinite(scales).all())
        require('candidate_constraints', np.all(residuals <= 1e-6 + 1e-5 * scales))
        mixture = np.einsum('k,kn->n', probabilities, states)
        require('mixture_arithmetic', np.allclose(estimate, mixture, rtol=1e-6, atol=1e-7))
        policy = record['policy']
        require('policy_schema', isinstance(policy, dict) and set(policy) == {'threshold','target_coverage'})
        threshold, coverage = policy['threshold'], policy['target_coverage']
        for value in ((coverage,) if threshold is None else (coverage, threshold)):
            require('policy_values', type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1)
        require('coverage_value', type(coverage) in (float,int) and math.isfinite(coverage) and 0 <= coverage <= 1)
        top = int(np.argmax(probabilities))
        accepted = threshold is not None and probabilities[top] >= threshold
        claim = record['structure_claim']
        if accepted:
            require('claim_schema', isinstance(claim, dict) and set(claim) == {'index','name','state'})
            require('claim_selection', type(claim['index']) is int and claim['index'] == top and claim['name'] == names[top])
            claimed_state = _array(claim['state'], 1)
            require('claim_tensor_not_estimate', claimed_state.shape == (n,) and
                    np.allclose(claimed_state, states[top], atol=1e-7, rtol=1e-6))
        else:
            require('abstention', claim is None)
        if checkpoint is not None:
            from .train import sha256_file
            from .v2_train import load_v2_checkpoint
            require('checkpoint_bytes', sha256_file(checkpoint) == record['checkpoint_sha256'])
            # Deserialization/metadata validation only. No forward, rollout, or solve.
            model, controls, metadata, _ = load_v2_checkpoint(checkpoint, device_name='cpu')
            require('checkpoint_library', metadata['names'] == names and np.array_equal(
                    np.asarray(_plain(metadata['boundaries']), dtype=np.float64), boundaries))
            require('checkpoint_model', record['model'] == 'shared' or record['model'] in controls)
            source = model if record['model'] == 'shared' else controls[record['model']]
            from .v2 import DirectMultiHypothesisNet
            expected_depth = 1 if isinstance(source, DirectMultiHypothesisNet) else source.config.steps
            require('checkpoint_depth', depth == expected_depth)
        if calibration is not None:
            from .train import sha256_file
            from .dual_study import read_json, FORMAT as STUDY_FORMAT
            require('calibration_bytes', sha256_file(calibration) == record['calibration_sha256'])
            artifact = read_json(calibration)
            require('calibration_format', artifact.get('format') == STUDY_FORMAT + '.calibration')
            require('calibration_checkpoint', artifact.get('checkpoint_sha256') == record['checkpoint_sha256'])
            entry = artifact['models'][record['model']]
            require('calibration_policy', entry['policy'] == policy and
                    artifact['model_sources'][record['model']] == record['checkpoint_sha256'])
        return {'accepted': True, 'reason': 'mixture arithmetic and separate candidate claim checked',
                'checks': checks, 'checkpoint_bound': checkpoint is not None,
                'calibration_bound': calibration is not None, 'neural_replay': False,
                'estimate_structurally_certified': False, 'route_correctness_proven': False}
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, OSError, RuntimeError) as exc:
        return {'accepted': False, 'reason': str(exc), 'checks': checks}
