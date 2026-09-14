"""Check a stack of worksheets, returning a verdict for EVERY worksheet.

Only arithmetic is grouped. Schemas, artifact identities, and the request's
expected inputs remain per-record checks. Near a numerical tolerance boundary,
or for unusual shapes, defer to the unchanged prepared scalar checker.
This opt-in research adapter neither changes neural weights nor caches PASSes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark_verifier_setup import PreparedVerifier
from universa_recurrent.neural import dual_lingua as endpoint
from universa_recurrent.neural import retention_lingua as receipt

MAX_RECORDS = 4096
MAX_BATCH_BYTES = 64_000_000
CHUNK = 64
ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError,
          FloatingPointError, RuntimeError, RecursionError)
END_FIELDS = {'format', 'scope', 'model', 'checkpoint_sha256', 'calibration_sha256',
              'trained_depth', 'policy', 'observed', 'mask', 'library',
              'candidate_states', 'probabilities', 'estimate', 'structure_claim'}
OUTER_FIELDS = {'format', 'scope', 'retention', 'endpoint', 'trajectory'}
TRAJECTORY_CHECKS = ('trajectory_schema', 'dimensions', 'basis_orthonormal',
    'basis_constraints', 'coordinate_decode', 'every_candidate_feasible',
    'observed_residuals', 'progress_arithmetic', 'probabilities_valid',
    'logit_readout', 'endpoint_states', 'endpoint_probabilities')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def rejected(reason):
    return {'accepted': False, 'reason': str(reason), 'checks': {}}


def _array(value, shape):
    a = endpoint._array(value, len(shape))
    require(a.shape == shape, 'unexpected array shape')
    return a


def _eligible(record, snapshot, boundary):
    """Fast-path eligibility is NOT a verdict; all geometry is checked later."""
    require(isinstance(record, dict) and set(record) == OUTER_FIELDS, 'receipt schema')
    require(record['format'] == receipt.FORMAT, 'receipt format')
    require(record['scope'] == receipt.SCOPE and
            all(type(v) is bool for v in record['scope'].values()), 'receipt scope')
    e = record['endpoint']
    require(isinstance(e, dict) and set(e) == END_FIELDS, 'endpoint schema')
    require(e['format'] == endpoint.FORMAT, 'endpoint format')
    require(e['scope'] == endpoint.SCOPE and
            all(type(v) is bool for v in e['scope'].values()), 'endpoint scope')
    for key in ('checkpoint_sha256', 'calibration_sha256', 'library'):
        require(e[key] == snapshot[key], 'pinned ' + key)
    require(type(e['model']) is str and e['model'] in snapshot['models'], 'pinned model')
    model = snapshot['models'][e['model']]
    require(type(e['trained_depth']) is int and e['trained_depth'] == model['depth']
            and e['trained_depth'] > 0, 'pinned depth')
    policy = e['policy']
    require(isinstance(policy, dict) and set(policy) == {'threshold', 'target_coverage'}
            and policy == model['policy'], 'pinned policy')
    for key in ('threshold', 'target_coverage'):
        value = policy[key]
        if key == 'threshold' and value is None:
            continue
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
                'policy numeric value')
    k, _, n = boundary.shape
    arrays = {'s': _array(e['candidate_states'], (k, n)),
              'p': _array(e['probabilities'], (k,)), 'mu': _array(e['estimate'], (n,)),
              'x': _array(e['observed'], (n,)), 'mask': _array(e['mask'], (n,))}
    p = arrays['p']
    top = int(p.argmax())
    accepted = policy['threshold'] is not None and p[top] >= policy['threshold']
    claim = e['structure_claim']
    if accepted:
        require(isinstance(claim, dict) and set(claim) == {'index', 'name', 'state'}, 'claim schema')
        require(type(claim['index']) is int and claim['index'] == top and
                claim['name'] == snapshot['library']['names'][top], 'claim selection')
        arrays['claim'] = _array(claim['state'], (n,))
    else:
        require(claim is None, 'abstention')
        arrays['claim'] = arrays['s'][top]
    kind, history, t = record['retention'], record['trajectory'], e['trained_depth']
    require(kind in ('endpoint', 'trajectory'), 'retention')
    d = 0
    if kind == 'endpoint':
        require(history is None, 'discarded history must not be claimed')
    else:
        require(isinstance(history, dict) and set(history) == set(receipt.FIELDS) | {'bases'},
                'trajectory schema')
        q = endpoint._array(history['bases'], 3)
        require(q.shape[:2] == (k, n), 'basis shape')
        d = q.shape[-1]
        arrays.update(q=q, hs=_array(history['states'], (t, k, n)),
            a=_array(history['coordinates'], (t, k, d)),
            hp=_array(history['route_probabilities'], (t, k)),
            logits=_array(history['route_logits'], (t, k)),
            rms=_array(history['residuals'], (t, k)),
            progress=_array(history['progress'], (t, k)))
    require(sum(a.size for a in arrays.values()) <= 65536, 'large record: scalar fallback')
    return (kind, t, d), arrays


def _all(a):
    return np.asarray(a).reshape(len(a), -1).all(axis=1)


def _close(actual, expected, atol=1e-6, rtol=1e-5):
    # A conservative interior of the OLD tolerance. Anything outside this fast
    # acceptance region goes to the old checker, not to automatic rejection.
    return _all(np.isfinite(expected) & np.isfinite(actual) &
                (np.abs(actual - expected) <= .5 * (atol + rtol * np.abs(expected))))


def _geometry(arrays, boundary, kind):
    """Vectorized candidate checks, one boolean per record; never one batch mean."""
    bnorm = np.linalg.norm(boundary, axis=(1, 2))
    s, p, mu, x, mask = (arrays[key] for key in ('s', 'p', 'mu', 'x', 'mask'))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        valid = _all(np.isin(mask, [0, 1])) & (mask.sum(-1) >= 1)
        valid &= _all((mask != 0) | (x == 0))
        valid &= _all((p >= 0) & (p <= 1)) & (np.abs(p.sum(-1) - 1) <= 1e-6)
        bz = np.einsum('kmn,bkn->bkm', boundary, s)
        scale = bnorm[None] * np.linalg.norm(s, axis=-1)
        valid &= _all(np.isfinite(scale) & np.isfinite(bz).all(-1) &
                      (np.linalg.norm(bz, axis=-1) <= .5 * (1e-6 + 1e-5 * scale)))
        mixture = np.einsum('bk,bkn->bn', p, s)
        valid &= _close(mu, mixture, 1e-7, 1e-6)
        selected = s[np.arange(len(s)), p.argmax(-1)]
        valid &= _close(arrays['claim'], selected, 1e-7, 1e-6)
        if kind == 'endpoint':
            return valid
        hs, q, a, hp, logits, rms, progress = (arrays[key] for key in
            ('hs', 'q', 'a', 'hp', 'logits', 'rms', 'progress'))
        gram = np.einsum('bknd,bkne->bkde', q, q)
        valid &= _close(gram, np.broadcast_to(np.eye(q.shape[-1]), gram.shape))
        bq = np.einsum('kmn,bknd->bkmd', boundary, q)
        valid &= _close(bq, np.zeros_like(bq))
        valid &= _close(hs, np.einsum('bknd,btkd->btkn', q, a))
        bz = np.einsum('kmn,btkn->btkm', boundary, hs)
        scale = bnorm[None, None] * np.linalg.norm(hs, axis=-1)
        valid &= _all(np.isfinite(scale) & np.isfinite(bz).all(-1) &
                      (np.linalg.norm(bz, axis=-1) <= .5 * (1e-6 + 1e-5 * scale)))
        expected = (np.linalg.norm(mask[:, None, None] * (hs - x[:, None, None]), axis=-1)
                    / np.sqrt(mask.sum(-1))[:, None, None])
        valid &= _close(rms, expected)
        initial = np.linalg.norm(mask * x, axis=-1) / np.sqrt(mask.sum(-1))
        previous = np.concatenate((np.broadcast_to(initial[:, None, None],
            (len(s), 1, s.shape[1])), expected[:, :-1]), axis=1)
        valid &= _close(progress, previous - expected)
        valid &= _all((hp >= 0) & (hp <= 1))
        valid &= _close(hp.sum(-1), np.ones(hp.shape[:2]), 2e-6, 1e-5)
        shifted = logits - logits.max(-1, keepdims=True)
        valid &= _all(np.isfinite(shifted))  # scalar checker raises on subtract overflow
        exp = np.exp(shifted)
        valid &= _close(hp, exp / exp.sum(-1, keepdims=True))
        valid &= _close(hs[:, -1], s) & _close(hp[:, -1], p)
        return valid


def expected_inputs(observed, mask, model: str, retention: str) -> list[dict]:
    """Caller-owned request expectations, NEVER derived from received receipts."""
    def plain(a):
        return a.detach().cpu().tolist() if hasattr(a, 'detach') else np.asarray(a).tolist()
    xs, masks = plain(observed), plain(mask)
    require(len(xs) == len(masks) > 0, 'expected input length mismatch')
    return [dict(observed=x, mask=m, model=model, retention=retention) for x, m in zip(xs, masks)]


def finish(verdicts, records, expected_count, expected):
    """Count and request-order checks are also used by the scalar control."""
    if expected is not None:
        require(len(expected) == expected_count, 'expectation count mismatch')
        for i, (v, record) in enumerate(zip(verdicts, records)):
            try:
                require(i < len(expected), 'unexpected extra record')
                e, wanted = record['endpoint'], expected[i]
                require(e['observed'] == wanted['observed'] and e['mask'] == wanted['mask']
                        and e['model'] == wanted['model'] and record['retention'] == wanted['retention'],
                        'request input/model/retention mismatch')
            except ERRORS as error:
                verdicts[i] = rejected(error)
    complete = len(verdicts) == expected_count
    return {'accepted': complete and all(v['accepted'] for v in verdicts),
            'complete': complete, 'expected_count': expected_count,
            'checked_count': len(verdicts), 'accepted_count': sum(v['accepted'] for v in verdicts),
            'input_bound': expected is not None, 'verdicts': verdicts}


def _parse(payloads, expected_count):
    require(type(expected_count) is int and 1 <= expected_count <= MAX_RECORDS,
            'expected_count must be in [1,4096]')
    require(isinstance(payloads, (list, tuple)) and len(payloads) <= MAX_RECORDS, 'batch limit')
    require(sum(len(x) for x in payloads if isinstance(x, bytes)) <= MAX_BATCH_BYTES, 'batch bytes limit')
    records, errors = [], []
    for payload in payloads:
        try:
            r = receipt.loads(payload)
            records.append(r); errors.append(None)
        except ERRORS as error:
            records.append(None); errors.append(rejected(error))
    return records, errors


def scalar_batch(reference, payloads, *, expected_count, expected=None):
    """Prepared sequential control with exactly the same external request checks."""
    records, errors = _parse(payloads, expected_count)
    verdicts = []
    for r, error in zip(records, errors):
        try:
            verdicts.append(error if error is not None else reference.verify(r))
        except ERRORS as exc:
            verdicts.append(rejected(exc))
    return finish(verdicts, records, expected_count, expected)


@dataclass(frozen=True, slots=True)
class BatchPreparedVerifier:
    reference: PreparedVerifier

    def __post_init__(self):
        require(isinstance(self.reference, PreparedVerifier), 'trusted PreparedVerifier required')

    @classmethod
    def prepare(cls, checkpoint: Path, calibration: Path):
        return cls(PreparedVerifier.prepare(checkpoint, calibration))

    def verify_payloads(self, payloads: Sequence[bytes], *, expected_count: int,
                        expected: list[dict] | None = None) -> dict:
        records, verdicts = _parse(payloads, expected_count)
        snapshot = self.reference.manifest()  # one fresh copy, never caller-editable cache
        boundary = endpoint._array(snapshot['library']['boundaries'], 3)
        groups = defaultdict(list)
        fallback = set()
        for i, r in enumerate(records):
            if verdicts[i] is not None:
                continue
            try:
                key, arrays = _eligible(r, snapshot, boundary)
                groups[key].append((i, arrays))
            except ERRORS:
                fallback.add(i)
        snapshot_sha = hashlib.sha256(self.reference._snapshot).hexdigest()
        fast = 0
        for (kind, depth, _), items in groups.items():
            for offset in range(0, len(items), CHUNK):
                block = items[offset:offset + CHUNK]
                try:
                    packed = {key: np.stack([a[key] for _, a in block]) for key in block[0][1]}
                    valid = _geometry(packed, boundary, kind)
                except ERRORS:
                    valid = [False] * len(block)
                for (i, _), passed in zip(block, valid):
                    if not passed:
                        fallback.add(i); continue
                    names = ('schema', 'scope', 'endpoint', 'retention') + (
                        ('no_history_claim',) if kind == 'endpoint' else TRAJECTORY_CHECKS)
                    verdicts[i] = dict(accepted=True, checks=dict.fromkeys(names, True),
                        retained_steps_checked=0 if kind == 'endpoint' else depth,
                        checkpoint_bound=True, calibration_bound=True,
                        learned_transitions_verified=False, estimate_structurally_certified=False,
                        reason='Grouped arithmetic and per-record pinned reference checks passed.',
                        binding_scope='trusted immutable snapshot; current disk state is NOT rechecked',
                        snapshot_sha256=snapshot_sha)
                    fast += 1
        for i in sorted(fallback):
            try:
                verdicts[i] = self.reference.verify(records[i])
            except ERRORS as error:
                verdicts[i] = rejected(error)
        require(all(v is not None for v in verdicts), 'internal missing verdict')
        result = finish(verdicts, records, expected_count, expected)
        result.update(vectorized_records=fast, scalar_fallback_records=len(fallback))
        return result


def generate_payloads(engine, metadata, policy, observed_cpu, mask_cpu,
                      output, history, calibration_sha) -> list[bytes]:
    """Batch host conversions, preserve the EXACT historical per-record JSON.

    Public constants remain repeated in the wire format. No packed protocol,
    quantization, omitted fields, neural update, or discarded check is added.
    """
    plain = endpoint._plain
    fields = ('candidate_states', 'probabilities', 'estimate', 'claim_mask', 'top_route')
    host = {key: plain(output[key]) for key in fields}
    observations, masks = plain(observed_cpu), plain(mask_cpu)
    count = len(observations)
    require(count == len(masks) and all(len(v) == count for v in host.values()), 'generation count')
    library = {'names': list(metadata['names']), 'boundaries': plain(metadata['boundaries'])}
    h = None if history is None else {key: plain(history[key]) for key in receipt.FIELDS}
    bases = None if h is None else plain(engine.bases)
    result = []
    for i in range(count):
        states, index = host['candidate_states'][i], host['top_route'][i]
        claim = ({'index': index, 'name': library['names'][index], 'state': states[index]}
                 if host['claim_mask'][i] else None)
        e = {'format': endpoint.FORMAT, 'scope': dict(endpoint.SCOPE), 'model': engine.name,
             'checkpoint_sha256': metadata['checkpoint_sha256'], 'calibration_sha256': calibration_sha,
             'trained_depth': engine.depth, 'policy': policy.as_dict(),
             'observed': observations[i], 'mask': masks[i], 'library': library,
             'candidate_states': states, 'probabilities': host['probabilities'][i],
             'estimate': host['estimate'][i], 'structure_claim': claim}
        trajectory = None if h is None else {
            **{key: [step[i] for step in h[key]] for key in receipt.FIELDS}, 'bases': bases}
        result.append(receipt.dumps({'format': receipt.FORMAT, 'scope': dict(receipt.SCOPE),
            'retention': 'endpoint' if h is None else 'trajectory', 'endpoint': e, 'trajectory': trajectory}))
    return result
