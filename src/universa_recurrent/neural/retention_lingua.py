"""An endpoint receipt, optionally accompanied by the numerical worksheets.

A trajectory lets us check each candidate's geometry and recorded residuals.
It does NOT prove that a learned update produced those intermediate states.
This checker uses NumPy arithmetic, not a neural replay or solver.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .dual_lingua import _array, _plain, verify_record as verify_endpoint

FORMAT = 'universa-recurrent.retention-lingua.v1'
SCOPE = {'candidate_geometry_checked': True, 'diagnostics_checked': True,
         'learned_transitions_verified': False, 'route_correctness_proven': False,
         'execution_authenticated': False, 'estimate_structurally_certified': False}
FIELDS = ('coordinates', 'states', 'route_logits', 'route_probabilities', 'residuals', 'progress')


def make_record(endpoint: dict, *, history: dict | None = None, bases=None) -> dict:
    """One example. Conversion includes device-to-host copies when needed."""
    trajectory = None
    if history is not None:
        if bases is None or history['states'].shape[1] != 1:
            raise ValueError('a full record requires bases and a one-example history')
        trajectory = {key: _plain(history[key][:, 0]) for key in FIELDS}
        trajectory['bases'] = _plain(bases)
    return {'format': FORMAT, 'scope': dict(SCOPE),
            'retention': 'trajectory' if history is not None else 'endpoint',
            'endpoint': endpoint, 'trajectory': trajectory}


def dumps(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def loads(data: bytes) -> dict:
    """Reject ambiguous or nonfinite JSON before checking arithmetic."""
    if not isinstance(data, bytes) or len(data) > 10_000_000:
        raise ValueError('record must be bytes smaller than 10 MB')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON literal: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def verify_record(record: dict, *, checkpoint: Path | None = None,
                  calibration: Path | None = None) -> dict:
    checks = {}
    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(name)
    def close(name, actual, expected):
        require(name, actual.shape == expected.shape and np.isfinite(expected).all()
                and np.allclose(actual, expected, atol=1e-6, rtol=1e-5))
    try:
        require('schema', isinstance(record, dict) and set(record) ==
                {'format', 'scope', 'retention', 'endpoint', 'trajectory'} and record['format'] == FORMAT)
        require('scope', record['scope'] == SCOPE and
                all(type(record['scope'][k]) is bool for k in SCOPE))
        endpoint = verify_endpoint(record['endpoint'], checkpoint=checkpoint, calibration=calibration)
        require('endpoint', endpoint['accepted'])
        require('retention', record['retention'] in ('endpoint', 'trajectory'))
        if record['retention'] == 'endpoint':
            require('no_history_claim', record['trajectory'] is None)
            count = 0
        else:
            h = record['trajectory']
            require('trajectory_schema', isinstance(h, dict) and set(h) == set(FIELDS) | {'bases'})
            states = _array(h['states'], 3)
            coords = _array(h['coordinates'], 3)
            bases = _array(h['bases'], 3)
            p = _array(h['route_probabilities'], 2)
            logits = _array(h['route_logits'], 2)
            residuals = _array(h['residuals'], 2)
            progress = _array(h['progress'], 2)
            end = record['endpoint']
            boundary = _array(end['library']['boundaries'], 3)
            obs, mask = _array(end['observed'], 1), _array(end['mask'], 1)
            t, k, n = states.shape
            require('dimensions', t == end['trained_depth'] and k == len(end['library']['names'])
                    and n == len(obs) and coords.shape[:2] == (t, k)
                    and bases.shape == (k, n, coords.shape[-1])
                    and all(a.shape == (t, k) for a in (p, logits, residuals, progress)))
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                close('basis_orthonormal', np.einsum('knd,kne->kde', bases, bases),
                      np.broadcast_to(np.eye(bases.shape[-1]), (k, bases.shape[-1], bases.shape[-1])))
                bq = np.einsum('kmn,knd->kmd', boundary, bases)
                close('basis_constraints', bq, np.zeros_like(bq))
                close('coordinate_decode', states, np.einsum('knd,tkd->tkn', bases, coords))
                bz = np.einsum('kmn,tkn->tkm', boundary, states)
                scale = np.linalg.norm(boundary, axis=(1, 2))[None] * np.linalg.norm(states, axis=-1)
                require('every_candidate_feasible', np.isfinite(scale).all() and
                        np.all(np.linalg.norm(bz, axis=-1) <= 1e-6 + 1e-5 * scale))
                expected_rms = np.linalg.norm(mask[None, None] * (states - obs[None, None]), axis=-1) / np.sqrt(mask.sum())
                close('observed_residuals', residuals, expected_rms)
                initial_rms = np.linalg.norm(mask * obs) / np.sqrt(mask.sum())
                previous = np.concatenate((np.full((1, k), initial_rms), expected_rms[:-1]), axis=0)
                close('progress_arithmetic', progress, previous - expected_rms)
                require('probabilities_valid', ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(-1), 1, atol=2e-6))
                exp = np.exp(logits - logits.max(-1, keepdims=True))
                close('logit_readout', p, exp / exp.sum(-1, keepdims=True))
                close('endpoint_states', states[-1], _array(end['candidate_states'], 2))
                close('endpoint_probabilities', p[-1], _array(end['probabilities'], 1))
            count = t
        return {'accepted': True, 'checks': checks, 'retained_steps_checked': count,
                'checkpoint_bound': endpoint['checkpoint_bound'], 'calibration_bound': endpoint['calibration_bound'],
                'learned_transitions_verified': False, 'estimate_structurally_certified': False,
                'reason': 'Endpoint arithmetic and retained candidate geometry checked; no neural replay.'}
    except (ValueError, TypeError, KeyError, IndexError, FloatingPointError, OverflowError, OSError, RuntimeError) as error:
        return {'accepted': False, 'checks': checks, 'reason': str(error)}
