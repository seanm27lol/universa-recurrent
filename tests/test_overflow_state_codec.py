"""The fallback must restore every original field, not just the overflowing one."""
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continuation_codec import StateCodec
from overflow_state_codec import OverflowStateCodec


def manifest(kind='named', bits=8):
    site = dict(checkpoint_sha256='a'*64, calibration_sha256='b'*64, model='shared',
                cut=2, depth=8, num_structures=2, latent_dim=2, names=['A', 'B'])
    return StateCodec.fit(np.ones((8, 6), dtype=np.float32), kind=kind, bits=bits,
                          rotation_seed=21, site=site).manifest()


@pytest.mark.parametrize('kind', ['named', 'rotated'])
@pytest.mark.parametrize('bits', [8, 12, 16])
def test_exact_overflow_and_unchanged_inrange(kind, bits):
    codec = OverflowStateCodec.from_manifest(manifest(kind, bits))
    x = np.array([[0, 0, 0, 0, 0, 0], [100, -.0, .127, -7, .2, -9]], dtype=np.float32)
    before = x.tobytes()
    payloads, mask = codec.encode(x)
    assert mask.tolist() == [False, True]
    restored = codec.decode(payloads)
    assert restored[1].tobytes() == x[1].tobytes()
    base, _ = codec.base_codec().encode(x)
    assert np.array_equal(restored[0], codec.base_codec().decode(base)[0])
    assert x.tobytes() == before
    counted = codec.storage(payloads)
    assert counted['fallback_n'] == 1
    assert counted['total_records_bytes'] == sum(map(len, payloads))
    assert counted['tag_bytes_per_state'] == 1
    assert len(payloads[1]) == 32 + 1 + 24
    assert len(payloads[0]) == 32 + 1 + (6 * bits + 7)//8


def test_exact_range_boundary_and_next_float():
    m = manifest(); m['ranges'] = [1.0]*6
    codec = OverflowStateCodec.from_manifest(m)
    x = np.ones((4, 6), dtype=np.float32)
    x[1] = -1
    x[2, 0] = np.nextafter(np.float32(1), np.float32(np.inf))
    x[3, 0] = np.nextafter(np.float32(-1), np.float32(-np.inf))
    _, use_raw = codec.encode(x)
    assert use_raw.tolist() == [False, False, True, True]


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(invalid):
    c = OverflowStateCodec.from_manifest(manifest()); x = np.zeros((1, 6), dtype=np.float32)
    x[0, 0] = invalid
    with pytest.raises(ValueError): c.encode(x)


@pytest.mark.parametrize('shape', [(1, 5), (6,), (0, 6), (1, 2, 3)])
def test_missing_fields_empty_and_shape_rejected(shape):
    with pytest.raises(ValueError): OverflowStateCodec.from_manifest(manifest()).encode(np.zeros(shape, dtype=np.float32))


def test_no_silent_float64_downcast():
    with pytest.raises(ValueError): OverflowStateCodec.from_manifest(manifest()).encode(np.ones((1, 6)))


def test_immutable_ranges_and_site():
    meta = manifest(); c = OverflowStateCodec.from_manifest(meta); identity = c.identity
    meta['ranges'][0] = 100; meta['site']['cut'] = 7
    c.manifest()['base']['ranges'][0] = 0
    assert c.identity == identity
    with pytest.raises(FrozenInstanceError): c._base_bytes = b'bad'


@pytest.mark.parametrize('change', ['range', 'matrix', 'width', 'bits', 'cut', 'hash', 'extra', 'bool', 'kind'])
def test_malformed_manifest_rejected(change):
    m = manifest()
    if change == 'range': m['ranges'][0] = float('inf')
    elif change == 'matrix': m['matrix'] = np.eye(6).tolist()
    elif change == 'width': m['width'] = 5
    elif change == 'bits': m['bits'] = 32
    elif change == 'cut': m['site']['cut'] = 8
    elif change == 'hash': m['site']['checkpoint_sha256'] = 'bad'
    elif change == 'extra': m['unused'] = 'hidden payload'
    elif change == 'bool': m['bits'] = True
    else: m['kind'] = 'raw_f32'
    with pytest.raises(ValueError): OverflowStateCodec.from_manifest(m)


@pytest.mark.parametrize('defect', ['identity', 'tag', 'short', 'trailing', 'reserved', 'nonfinite_raw', 'fake_raw'])
def test_bad_frame_rejected(defect):
    c = OverflowStateCodec.from_manifest(manifest())
    p = c.encode(np.zeros((1, 6), dtype=np.float32))[0][0]
    if defect == 'identity': p = b'x' + p[1:]
    elif defect == 'tag': p = p[:32] + b'\x02' + p[33:]
    elif defect == 'short': p = p[:-1]
    elif defect == 'trailing': p += b'\0'
    elif defect == 'reserved': p = p[:33] + b'\xff' + p[34:]
    elif defect == 'nonfinite_raw': p = p[:32] + b'\x01' + np.full(6, np.nan, dtype='<f4').tobytes()
    else: p = p[:32] + b'\x01' + np.zeros(6, dtype='<f4').tobytes()
    with pytest.raises(ValueError): c.decode([p])


def test_different_cut_cannot_decode():
    a = OverflowStateCodec.from_manifest(manifest()); m = manifest(); m['site']['cut'] = 3
    b = OverflowStateCodec.from_manifest(m)
    with pytest.raises(ValueError): b.decode(a.encode(np.zeros((1, 6), dtype=np.float32))[0])


def test_rotated_full_precision_fallback_is_not_rotated_float32():
    c = OverflowStateCodec.from_manifest(manifest('rotated', 12))
    x = np.random.default_rng(3).normal(size=(20, 6)).astype(np.float32)*100
    p, mask = c.encode(x)
    assert mask.all()
    assert c.decode(p).tobytes() == x.tobytes()
