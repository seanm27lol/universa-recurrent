"""A tiny mathematical state description, not a learned language interpreter.

Coordinates say how much of each known cycle is present; route logits are the
model's current evidence numbers, not certified probabilities. We quantize these
named fields, or an equal-dimensional rotated numeric control, using calibration
states only. The decoder receives only the bytes and this fixed codec manifest.
It never receives the original observation, future states, or the final answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

FORMAT = 'universa-recurrent.state-codec.v1'
BITS = (4, 8, 12, 16)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def finite_matrix(value, width=None):
    x = np.asarray(value)
    if (x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 256
            or x.dtype.kind != 'f' or not np.isfinite(x).all()
            or (width is not None and x.shape[1] != width)):
        raise ValueError('expected finite floating [examples, state fields]')
    return x.astype(np.float64)


@dataclass(frozen=True, slots=True, init=False)
class StateCodec:
    """Pinned calibration metadata. A hash identifies it, not an execution proof."""
    _manifest: bytes

    @classmethod
    def fit(cls, calibration, *, kind, bits=None, site, rotation_seed=0):
        x = finite_matrix(calibration)
        width = x.shape[1]
        if kind not in ('raw_f32', 'named', 'rotated'):
            raise ValueError('unknown codec kind')
        if kind == 'raw_f32':
            if bits is not None:
                raise ValueError('raw_f32 does not accept a quantization bit width')
        elif type(bits) is not int or bits not in BITS:
            raise ValueError('quantization bits must be 4, 8, 12 or 16')
        if type(rotation_seed) is not int or rotation_seed < 0:
            raise ValueError('rotation seed must be a nonnegative integer')
        # A random coordinate system is a matched numerical-compression control.
        # It is NOT a second neural architecture or a no-structure control.
        matrix = np.eye(width)
        if kind == 'rotated':
            matrix, _ = np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(width, width)))
        with np.errstate(over='raise', invalid='raise'):
            rotated = x @ matrix
            ranges = np.maximum(np.abs(rotated).max(0) * 1.05, 1e-8)
        meta = dict(format=FORMAT, kind=kind, bits=bits, width=width, site=site,
                    rotation_seed=rotation_seed if kind == 'rotated' else None,
                    matrix=matrix.tolist() if kind == 'rotated' else None,
                    ranges=ranges.tolist() if kind != 'raw_f32' else None,
                    calibration_n=len(x), field_order='candidate-major coordinates, then route logits',
                    scale_rule='per-field calibration max absolute value times 1.05, floor 1e-8')
        result = object.__new__(cls)
        object.__setattr__(result, '_manifest', canonical(meta))
        return result

    def manifest(self):
        return json.loads(self._manifest)

    @property
    def identity(self):
        return hashlib.sha256(self._manifest).digest()

    def encode(self, values):
        m = self.manifest()
        x = finite_matrix(values, m['width'])
        clipped = np.zeros_like(x, dtype=bool)
        if m['kind'] == 'raw_f32':
            with np.errstate(over='raise', invalid='raise'):
                v = x.astype('<f4')
            buffers = [row.tobytes() for row in v]
        else:
            matrix = np.eye(m['width']) if m['matrix'] is None else np.array(m['matrix'])
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                y = x @ matrix
                ranges = np.array(m['ranges'])
                clipped = np.abs(y) > ranges
                limit = (1 << (m['bits'] - 1)) - 1
                integers = np.rint(np.clip(y / ranges, -1, 1) * limit).astype(np.int64) + limit
            bitrows = ((integers[..., None] >> np.arange(m['bits'])) & 1).reshape(len(x), -1)
            buffers = [row.tobytes() for row in np.packbits(bitrows.astype(np.uint8), axis=1, bitorder='little')]
        # Each record carries its full codec hash; this overhead is counted.
        payloads = [self.identity + row for row in buffers]
        return payloads, dict(clipped_fields=int(clipped.sum()),
                              clipped_examples=int(clipped.any(-1).sum()),
                              examples=len(x), fields_per_example=m['width'])

    def decode(self, payloads):
        """No state/context parameter: only the description and fixed shared metadata."""
        m = self.manifest()
        if not isinstance(payloads, (list, tuple)) or not payloads:
            raise ValueError('nonempty sequence of encoded states required')
        data_bytes = 4 * m['width'] if m['kind'] == 'raw_f32' else math.ceil(m['width'] * m['bits'] / 8)
        for p in payloads:
            if type(p) is not bytes or len(p) != 32 + data_bytes or p[:32] != self.identity:
                raise ValueError('wrong codec, length or state record type')
        if m['kind'] == 'raw_f32':
            restored = np.stack([np.frombuffer(p[32:], dtype='<f4') for p in payloads]).copy()
        else:
            packed = np.stack([np.frombuffer(p[32:], dtype=np.uint8) for p in payloads])
            bits = np.unpackbits(packed, axis=1, bitorder='little')
            used = m['width'] * m['bits']
            if bits[:, used:].any():
                raise ValueError('nonzero padding in state record')
            codes = (bits[:, :used].reshape(len(payloads), m['width'], m['bits'])
                     * (1 << np.arange(m['bits']))).sum(-1)
            limit = (1 << (m['bits'] - 1)) - 1
            if (codes > 2 * limit).any():
                raise ValueError('reserved fixed-point symbol')
            y = (codes.astype(np.float64) - limit) * np.array(m['ranges']) / limit
            matrix = np.eye(m['width']) if m['matrix'] is None else np.array(m['matrix'])
            restored = (y @ matrix.T).astype(np.float32)
        if not np.isfinite(restored).all():
            raise ValueError('decoded state is nonfinite')
        return restored

    def storage(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('positive state count required')
        m = self.manifest()
        values = 4 * m['width'] if m['kind'] == 'raw_f32' else math.ceil(m['bits'] * m['width'] / 8)
        return dict(value_bytes_per_state=values, record_bytes_per_state=values + 32,
                    shared_codec_manifest_bytes=len(self._manifest),
                    total_codec_and_records_bytes=len(self._manifest) + count * (values + 32),
                    excludes='Unchanged observations, masks, encoder context, prior logits, model weights and structure library.')


def describe(values, *, names, latent_dim, cut):
    """Deterministic readable rendering; it is NOT an extra channel to the decoder."""
    x = finite_matrix(np.asarray(values, dtype=np.float32).reshape(1, -1))[0]
    k = len(names)
    if type(latent_dim) is not int or latent_dim < 1 or len(x) != k * (latent_dim + 1):
        raise ValueError('names/latent dimension do not match the state')
    coords, logits = x[:k * latent_dim].reshape(k, latent_dim), x[k * latent_dim:]
    parts = [f'After update {cut}; z_k = Q_k a_k; route weights = softmax(evidence).']
    for name, a, logit in zip(names, coords, logits):
        parts.append(f'{name}: cycle coordinates={a.tolist()}; evidence={float(logit):.7g}.')
    parts.append('Candidate geometry is known; correctness of the chosen structure is not certified.')
    return '\n'.join(parts)
