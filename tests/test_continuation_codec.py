"""The decoder must not need the answer, context, or a hidden original state."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
c = importlib.import_module('continuation_codec')


def values(n=11, d=6):
    return np.random.default_rng(123).normal(size=(n, d)).astype(np.float32)


def codec(kind='named', bits=8, d=6):
    return c.StateCodec.fit(values(128, d), kind=kind,
            bits=None if kind=='raw_f32' else bits, site={'cut': 2}, rotation_seed=33)


def test_raw_lossless_and_real_byte_count():
    x=values(); x[0,0]=-0.
    obj=codec('raw_f32'); payloads, stats=obj.encode(x)
    back=obj.decode(payloads)
    assert back.tobytes()==x.tobytes()
    assert len(payloads[0]) == 32+4*x.shape[1]
    report=obj.storage(len(x))
    assert report['total_codec_and_records_bytes']==len(obj._manifest)+sum(map(len,payloads))
    assert stats['clipped_fields']==0


@pytest.mark.parametrize('bits', c.BITS)
@pytest.mark.parametrize('kind', ['named','rotated'])
@pytest.mark.parametrize('d', [1,3,6])
def test_quantized_roundtrip_and_symbol_budget(bits,kind,d):
    obj=codec(kind,bits,d); x=values(9,d)
    payloads, counts=obj.encode(x)
    back=obj.decode(payloads)
    assert back.shape==x.shape and back.dtype==np.float32 and np.isfinite(back).all()
    assert len(payloads[0])-32 == (d*bits+7)//8
    assert obj.storage(len(x))['record_bytes_per_state']==len(payloads[0])
    assert counts['clipped_fields']>=0
    # Quantization error bound in transform coordinates for unclipped inputs.
    if counts['clipped_fields']==0:
        matrix=np.eye(d) if obj.manifest()['matrix'] is None else np.array(obj.manifest()['matrix'])
        errors=np.abs((back-x).astype(float)@matrix)
        bound=np.array(obj.manifest()['ranges'])/((1<<(bits-1))-1)/2 + 2e-6
        assert (errors<=bound).all()


@pytest.mark.parametrize('bits', c.BITS)
def test_equal_payload_controls_with_honest_metadata(bits):
    named, rotated=codec('named',bits),codec('rotated',bits)
    assert named.storage(17)['record_bytes_per_state']==rotated.storage(17)['record_bytes_per_state']
    # Rotated coordinate map is additional shared side information, not free.
    assert rotated.storage(17)['shared_codec_manifest_bytes']>named.storage(17)['shared_codec_manifest_bytes']


def test_calibration_clipping_and_no_test_refit():
    obj=c.StateCodec.fit(np.ones((8,6),np.float32),kind='named',bits=8,site={})
    before=obj._manifest
    p, stats=obj.encode(np.full((3,6),100.,np.float32))
    assert stats['clipped_examples']==3 and stats['clipped_fields']==18
    assert np.max(obj.decode(p))<=1.051
    assert obj._manifest==before


def test_immutable_metadata_copy():
    obj=codec(); h=obj.identity
    m=obj.manifest(); m['ranges'][0]=99999
    assert obj.identity==h
    with pytest.raises(Exception): obj._manifest=b'changed'


def test_wrong_cut_or_codec_is_rejected():
    obj=codec(); p,_=obj.encode(values())
    other=c.StateCodec.fit(values(128),kind='named',bits=8,site={'cut':3},rotation_seed=33)
    with pytest.raises(ValueError,match='wrong codec'): other.decode(p)


@pytest.mark.parametrize('bad', [[], [b''], [bytearray(b'a')], [b'a'*1000]])
def test_malformed_records(bad):
    with pytest.raises(ValueError): codec().decode(bad)


def test_reserved_symbol_and_padding_rejected():
    obj=codec('named',4,3); p,_=obj.encode(values(1,3))
    b=bytearray(p[0]); b[-1]|=128
    with pytest.raises(ValueError,match='padding'): obj.decode([bytes(b)])
    b=bytearray(p[0]); b[32]|=15
    with pytest.raises(ValueError,match='reserved'): obj.decode([bytes(b)])


@pytest.mark.parametrize('kind', ['raw_f32','named','rotated'])
@pytest.mark.parametrize('bad', [np.array([[np.nan]],np.float32),np.array([[np.inf]],np.float32),np.ones((0,6)),np.ones((6,)),np.ones((2,6),int)])
def test_invalid_numeric_calibration(kind,bad):
    with pytest.raises(ValueError): c.StateCodec.fit(bad,kind=kind,bits=None if kind=='raw_f32' else 8,site={})


def test_raw_nonfinite_payload_rejected():
    obj=codec('raw_f32'); p=obj.identity+np.full(6,np.nan,dtype='<f4').tobytes()
    with pytest.raises(ValueError,match='nonfinite'): obj.decode([p])


@pytest.mark.parametrize('b', [True,0,3,24,None])
def test_invalid_bits(b):
    with pytest.raises(ValueError): codec('named',b)


def test_raw_overflow_rejected():
    obj=codec('raw_f32')
    with pytest.raises(FloatingPointError): obj.encode(np.full((1,6),1e300))


def test_renderer_describes_known_fields_not_neural_semantics():
    text=c.describe(np.arange(6,dtype=np.float32), names=['A','B'],latent_dim=2,cut=2)
    assert 'cycle coordinates' in text and 'softmax' in text and 'not certified' in text
    with pytest.raises(ValueError): c.describe(np.arange(5),names=['A','B'],latent_dim=2,cut=2)
