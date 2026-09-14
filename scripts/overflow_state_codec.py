"""Use a compact state when it fits; otherwise preserve ALL original float32 fields.

A ruler cannot measure past its endpoint. Store the original reading instead of
pretending it equals the endpoint. Ranges are copied from the PREVIOUS experiment,
never fitted to test values. In-range rounding is unchanged and can affect decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np
from continuation_codec import StateCodec, canonical, FORMAT as BASE_FORMAT

FORMAT = 'universa-recurrent.overflow-state-codec.v1'
FIELDS = {'format', 'kind', 'bits', 'width', 'site', 'rotation_seed', 'matrix',
          'ranges', 'calibration_n', 'field_order', 'scale_rule'}


def validate_manifest(meta: dict) -> None:
    """Validate a numerical manifest; no code, pickle, or fitted model is loaded."""
    if not isinstance(meta, dict) or set(meta) != FIELDS:
        raise ValueError('invalid base manifest fields')
    if meta['format'] != BASE_FORMAT or meta['kind'] not in ('named', 'rotated'):
        raise ValueError('a named or rotated frozen quantizer is required')
    width, bits = meta['width'], meta['bits']
    if type(width) is not int or not 1 <= width <= 256 or type(bits) is not int or bits not in (8, 12, 16):
        raise ValueError('invalid width or bit budget')
    if type(meta['calibration_n']) is not int or meta['calibration_n'] < 1:
        raise ValueError('invalid calibration count')
    if (meta['field_order'] != 'candidate-major coordinates, then route logits' or
        meta['scale_rule'] != 'per-field calibration max absolute value times 1.05, floor 1e-8'):
        raise ValueError('unsupported quantizer convention')
    ranges = np.asarray(meta['ranges'])
    if (ranges.dtype.kind not in 'fiu' or ranges.shape != (width,) or
        not np.isfinite(ranges).all() or (ranges < 1e-8).any() or
        (ranges > np.finfo(np.float32).max).any()):
        raise ValueError('invalid frozen ranges')
    site = meta['site']
    required = {'checkpoint_sha256', 'model', 'cut', 'depth', 'num_structures',
                'latent_dim', 'names', 'calibration_sha256'}
    if not isinstance(site, dict) or set(site) != required:
        raise ValueError('invalid site')
    for key in ('cut', 'depth', 'num_structures', 'latent_dim'):
        if type(site[key]) is not int or site[key] < 1:
            raise ValueError('invalid site dimension')
    if not 1 <= site['cut'] < site['depth'] or width != site['num_structures'] * (site['latent_dim'] + 1):
        raise ValueError('site and state width disagree')
    names = site['names']
    if (not isinstance(names, list) or len(names) != site['num_structures'] or
        not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names)):
        raise ValueError('invalid candidate names')
    if not isinstance(site['model'], str) or not site['model']:
        raise ValueError('missing model')
    for key in ('checkpoint_sha256', 'calibration_sha256'):
        if not isinstance(site[key], str) or re.fullmatch('[0-9a-f]{64}', site[key]) is None:
            raise ValueError('invalid artifact hash')
    if meta['kind'] == 'named':
        if meta['matrix'] is not None or meta['rotation_seed'] is not None:
            raise ValueError('named coordinates cannot carry a rotation')
    else:
        q = np.asarray(meta['matrix'])
        if (q.dtype.kind not in 'fiu' or q.shape != (width, width) or
            not np.isfinite(q).all() or (np.abs(q) > 1.000001).any() or
            not np.allclose(q.T @ q, np.eye(width), atol=1e-10, rtol=1e-10) or
            type(meta['rotation_seed']) is not int or meta['rotation_seed'] < 0):
            raise ValueError('invalid rotation')


@dataclass(frozen=True, slots=True, init=False)
class OverflowStateCodec:
    """Immutable codec. A 32-byte identity plus one tag precedes each payload.

    Tag 0: the historical packed quantized values. Tag 1: original little-endian
    float32 values. The identity selects a manifest, NOT a payload signature.
    """
    _base_bytes: bytes
    _manifest_bytes: bytes

    @classmethod
    def from_manifest(cls, meta: dict) -> 'OverflowStateCodec':
        validate_manifest(meta)
        base_bytes = canonical(meta)
        result = object.__new__(cls)
        object.__setattr__(result, '_base_bytes', base_bytes)
        object.__setattr__(result, '_manifest_bytes', canonical({
            'format': FORMAT, 'base': meta, 'tag_bytes': 1,
            'tags': {'0': 'quantized', '1': 'original_float32'},
            'fallback': 'whole state iff any transformed field exceeds frozen range',
        }))
        return result

    def manifest(self) -> dict:
        return json.loads(self._manifest_bytes)

    def base_codec(self) -> StateCodec:
        # The historical class has no deserialization API. Construct only AFTER
        # validation above; immutable bytes are the complete numerical metadata.
        result = object.__new__(StateCodec)
        object.__setattr__(result, '_manifest', self._base_bytes)
        return result

    @property
    def identity(self) -> bytes:
        return hashlib.sha256(self._manifest_bytes).digest()

    def _values(self, values) -> np.ndarray:
        x = np.asarray(values)
        width = self.base_codec().manifest()['width']
        if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != width or
            not 1 <= len(x) <= 1_000_000 or not np.isfinite(x).all()):
            raise ValueError('finite nonempty float32 matrix with all state fields required')
        return x

    def overflow_mask(self, values) -> np.ndarray:
        x = self._values(values)
        meta = self.base_codec().manifest()
        q = np.eye(meta['width']) if meta['matrix'] is None else np.asarray(meta['matrix'])
        return (np.abs(x.astype(np.float64) @ q) > np.asarray(meta['ranges'])).any(axis=1)

    def encode(self, values) -> tuple[list[bytes], np.ndarray]:
        x = self._values(values)
        overflow = self.overflow_mask(x)
        # The baseline may clip, but none of its clipped payloads are emitted.
        packed, _ = self.base_codec().encode(x)
        prefix = self.identity
        payloads = [prefix + (b'\x01' + row.astype('<f4').tobytes() if use_raw
                             else b'\x00' + quantized[32:])
                    for row, use_raw, quantized in zip(x, overflow, packed)]
        return payloads, overflow

    def decode(self, payloads: list[bytes]) -> np.ndarray:
        if not isinstance(payloads, (list, tuple)) or not 1 <= len(payloads) <= 1_000_000:
            raise ValueError('nonempty encoded state sequence required')
        base = self.base_codec()
        width = base.manifest()['width']
        decoded = []
        for payload in payloads:
            if type(payload) is not bytes or len(payload) < 33 or payload[:32] != self.identity:
                raise ValueError('wrong codec identity or frame')
            tag, body = payload[32], payload[33:]
            if tag == 0:
                row = base.decode([base.identity + body])[0]
            elif tag == 1:
                if len(body) != width * 4:
                    raise ValueError('wrong raw length')
                row = np.frombuffer(body, dtype='<f4').astype(np.float32, copy=True)
                if not np.isfinite(row).all() or not self.overflow_mask(row[None])[0]:
                    raise ValueError('nonfinite or unnecessary raw fallback')
            else:
                raise ValueError('unknown fallback tag')
            decoded.append(row)
        return np.stack(decoded)

    def storage(self, payloads: list[bytes]) -> dict:
        # Parse before reporting any counts. All framing/manifest costs count.
        self.decode(payloads)
        sizes = np.asarray([len(p) for p in payloads])
        fallback_n = sum(p[32] == 1 for p in payloads)
        return dict(states=len(payloads), fallback_n=fallback_n,
            fallback_fraction=fallback_n / len(payloads), identity_bytes_per_state=32,
            tag_bytes_per_state=1, mean_value_bytes=float(sizes.mean() - 33),
            mean_record_bytes=float(sizes.mean()), total_records_bytes=int(sizes.sum()),
            shared_manifest_bytes=len(self._manifest_bytes),
            total_codec_and_records_bytes=int(sizes.sum()) + len(self._manifest_bytes))
