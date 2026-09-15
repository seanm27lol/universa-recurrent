"""A tiny, typed mathematical edit language over already-known state fields.

`cycle balanced_flow[0] += 0.125` changes one cycle coefficient.
`evidence balanced_flow += 0.6931471805599453` changes one current logit.
These names are supplied by the model's schema, not discovered neuron meanings.
No eval, neural call, original input, future state or answer is used by the parser.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import re
import numpy as np


@dataclass(frozen=True)
class Edit:
    kind: str
    candidate: int | None = None
    axis: int | None = None
    delta: float = 0.0


def schema(names, latent_dim):
    if (not isinstance(names, (list, tuple)) or len(names) < 2
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', n) for n in names)
            or type(latent_dim) is not int or latent_dim < 1):
        raise ValueError('distinct token-safe candidate names and positive latent dimension required')
    return len(names), latent_dim


def validate(edit, names, latent_dim):
    k, d = schema(names, latent_dim)
    if not isinstance(edit, Edit) or edit.kind not in ('cycle', 'evidence', 'all_evidence', 'identity'):
        raise ValueError('unknown mathematical operation')
    if (type(edit.delta) not in (int, float) or not math.isfinite(edit.delta)
            or abs(edit.delta) > float(np.finfo(np.float32).max)):
        raise ValueError('finite float32-sized delta required')
    if edit.kind in ('cycle', 'evidence'):
        if type(edit.candidate) is not int or not 0 <= edit.candidate < k:
            raise ValueError('candidate outside schema')
    elif edit.candidate is not None:
        raise ValueError('this operation does not address a candidate')
    if edit.kind == 'cycle':
        if type(edit.axis) is not int or not 0 <= edit.axis < d:
            raise ValueError('cycle axis outside schema')
    elif edit.axis is not None:
        raise ValueError('only a cycle edit has an axis')
    if edit.kind == 'identity' and edit.delta != 0:
        raise ValueError('identity cannot change a value')
    return edit


def render(edit, names, latent_dim):
    validate(edit, names, latent_dim)
    value = format(float(edit.delta), '.17g')
    if edit.kind == 'identity':
        return 'identity'
    if edit.kind == 'cycle':
        return f'cycle {names[edit.candidate]}[{edit.axis}] += {value}'
    if edit.kind == 'evidence':
        return f'evidence {names[edit.candidate]} += {value}'
    return f'all_evidence += {value}'


def parse(command, names, latent_dim):
    schema(names, latent_dim)
    if not isinstance(command, str) or not 1 <= len(command) <= 512 or not command.isascii():
        raise ValueError('bounded ASCII command required')
    if command == 'identity':
        return Edit('identity')
    numeric = r'([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)'
    match = re.fullmatch(r'cycle ([A-Za-z_][A-Za-z_0-9]*)\[([0-9]+)\] \+= ' + numeric, command)
    if match:
        name, axis, delta = match.groups()
        if name not in names:
            raise ValueError('unknown candidate name')
        return validate(Edit('cycle', names.index(name), int(axis), float(delta)), names, latent_dim)
    match = re.fullmatch(r'evidence ([A-Za-z_][A-Za-z_0-9]*) \+= ' + numeric, command)
    if match:
        name, delta = match.groups()
        if name not in names:
            raise ValueError('unknown candidate name')
        return validate(Edit('evidence', names.index(name), None, float(delta)), names, latent_dim)
    match = re.fullmatch(r'all_evidence \+= ' + numeric, command)
    if match:
        return validate(Edit('all_evidence', delta=float(match.group(1))), names, latent_dim)
    raise ValueError('command does not follow the mathematical grammar')


def values(vector, names, latent_dim):
    k, d = schema(names, latent_dim)
    x = np.asarray(vector)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != k*(d+1) or not len(x) or not np.isfinite(x).all():
        raise ValueError('complete finite float32 working-state matrix required')
    return x.copy()


def direct_edit(vector, edit, names, latent_dim):
    """Numerical reference uses flat indices; it does not parse a description."""
    validate(edit, names, latent_dim)
    x = values(vector, names, latent_dim)
    start = len(names)*latent_dim
    with np.errstate(over='raise', invalid='raise'):
        if edit.kind == 'cycle':
            x[:, edit.candidate*latent_dim + edit.axis] += np.float32(edit.delta)
        elif edit.kind == 'evidence':
            x[:, start + edit.candidate] += np.float32(edit.delta)
        elif edit.kind == 'all_evidence':
            x[:, start:] += np.float32(edit.delta)
    return x


def apply_command(vector, command, names, latent_dim):
    """An independently indexed named-field path, with no neural/model access."""
    op = parse(command, names, latent_dim)
    x = values(vector, names, latent_dim)
    k, d = len(names), latent_dim
    coordinates = x[:, :k*d].reshape(len(x), k, d)
    evidence = x[:, k*d:]
    with np.errstate(over='raise', invalid='raise'):
        if op.kind == 'cycle':
            coordinates[:, names.index(names[op.candidate]), op.axis] += np.float32(op.delta)
        elif op.kind == 'evidence':
            evidence[:, names.index(names[op.candidate])] += np.float32(op.delta)
        elif op.kind == 'all_evidence':
            for index in range(k):
                evidence[:, index] += np.float32(op.delta)
    return x


def decode_and_edit(payloads, codec, command):
    """Only numerical description bytes, pinned metadata and the edit command."""
    site = codec.manifest()['base']['site']
    return apply_command(codec.decode(payloads), command, site['names'], site['latent_dim'])


def wrong_candidate(edit, names, latent_dim):
    validate(edit, names, latent_dim)
    return (None if edit.candidate is None else
            replace(edit, candidate=(edit.candidate+1) % len(names)))


def edit_plan(base):
    """Fixed calibration-relative cycle deltas and fixed log-odds deltas; no tuning."""
    site = base['site']; names, d = site['names'], site['latent_dim']
    if base['kind'] != 'named' or base['bits'] != 16:
        raise ValueError('frozen named 16-bit base required')
    result = [Edit('identity'), Edit('all_evidence', delta=math.log(2))]
    for k in range(len(names)):
        for j in range(d):
            for sign in (-1, 1):
                result.append(Edit('cycle', k, j, sign*.125*base['ranges'][k*d+j]))
        for sign in (-1, 1):
            result.append(Edit('evidence', k, None, sign*math.log(2)))
    for op in result:
        validate(op, names, d)
    return result


def primitive_error(before, after, edit, bases, names, latent_dim):
    """Check only the immediate named operation, not a downstream causal theory.

    A cycle delta predicts a candidate-vector change delta*Q[:,axis].
    A logit delta predicts the softmax change. Float32 addition has rounding.
    """
    k, d = schema(names, latent_dim)
    b, a = values(before, names, d).astype(float), values(after, names, d).astype(float)
    q = np.asarray(bases, dtype=float)
    if q.ndim != 3 or q.shape[0] != k or q.shape[2] != d or not np.isfinite(q).all():
        raise ValueError('basis dimensions must match named fields')
    bc = b[:, :k*d].reshape(-1,k,d); ac = a[:, :k*d].reshape(-1,k,d)
    expected_delta = np.zeros((len(b),k,q.shape[1]))
    if edit.kind == 'cycle':
        expected_delta[:,edit.candidate] = float(np.float32(edit.delta))*q[edit.candidate,:,edit.axis]
    actual_delta = np.einsum('knd,bkd->bkn',q,ac-bc)
    expected_logits = b[:,k*d:].copy()
    if edit.kind == 'evidence':expected_logits[:,edit.candidate] += float(np.float32(edit.delta))
    if edit.kind == 'all_evidence':expected_logits += float(np.float32(edit.delta))
    def softmax(v):
        p = np.exp(v-v.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    return dict(candidate_delta_max_error=float(np.max(np.abs(actual_delta-expected_delta))),
                probability_rule_max_error=float(np.max(np.abs(softmax(a[:,k*d:])-softmax(expected_logits)))))
