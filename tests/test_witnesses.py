import copy
import json
import numpy as np
import pytest
from universa_recurrent.examples import flow_example
from universa_recurrent.recurrence import solve
from universa_recurrent.verification import check_projection, check_optimality, verify_record
from universa_recurrent.lingua import load_record, save_record


@pytest.fixture
def record():
    ex=flow_example()
    return solve(ex.problem,ex.candidates[0],trace_mode="full").record


def test_verifier_does_not_resolve(record,monkeypatch):
    def fail(*a,**kw):
        raise AssertionError("Checker must not solve or factorize")
    for name in ("solve","svd","lstsq","eigh","eigvalsh","pinv"):
        monkeypatch.setattr(np.linalg,name,fail)
    result=verify_record(record)
    assert result.accepted
    assert result.intermediate_checks == record["solver"]["iterations"]


def test_json_roundtrip_and_no_overwrite(record,tmp_path):
    p=tmp_path/'trace.json'
    save_record(record,p)
    assert verify_record(load_record(p)).accepted
    with pytest.raises(FileExistsError):
        save_record(record,p)
    broken=copy.deepcopy(record)
    broken['final']['state'][0]=float('nan')
    with pytest.raises(ValueError):
        save_record(broken,tmp_path/'bad.json')
    p.write_text('{"x": NaN}')
    with pytest.raises(ValueError):
        load_record(p)


@pytest.mark.parametrize("kind", ["final_state","final_multiplier","proposal","step_output","step_multiplier","order","count","claim","schema","missing","boundary_shape","nonfinite","objective","mode"])
def test_corrupted_record_refused(record,kind):
    r=copy.deepcopy(record)
    if kind == 'final_state': r['final']['state'][0] += 1
    elif kind == 'final_multiplier': r['final']['multiplier'][0] += 1
    elif kind == 'proposal': r['events'][0]['projection_witness']['proposal'][0] += 1
    elif kind == 'step_output': r['events'][0]['projection_witness']['output'][0] += 1
    elif kind == 'step_multiplier': r['events'][0]['projection_witness']['multiplier'][0] += 1
    elif kind == 'order': r['events'][0]['step'] = 99
    elif kind == 'count': r['solver']['iterations'] += 1
    elif kind == 'claim': r['claim'] = 'the model thinks like a human'
    elif kind == 'schema': r['schema'] = 'future'
    elif kind == 'missing': del r['final']
    elif kind == 'boundary_shape': r['problem']['boundary_shape'][1] += 1
    elif kind == 'nonfinite': r['final']['state'][0] = float('inf')
    elif kind == 'objective': r['events'][0]['objective'] += 1
    elif kind == 'mode': r['mode'] = 'unknown'
    assert not verify_record(r).accepted


def test_compact_scope_is_final_only():
    ex=flow_example()
    r=solve(ex.problem,ex.candidates[0]).record
    v=verify_record(r)
    assert v.accepted and v.intermediate_checks == 0
    assert 'final claim only' in v.reason
    # Demonstrates the declared limitation, not a promised history check.
    r['events'][0]['objective'] += 0.1
    assert verify_record(r).accepted


def test_same_endpoint_different_history_fails_full_check(record):
    r=copy.deepcopy(record)
    # Leave the final endpoint untouched, corrupt an intermediate proposal.
    r['events'][3]['projection_witness']['proposal'][2] += 0.25
    assert not verify_record(r).accepted


@pytest.mark.parametrize("bad", [None, [], {}, {"schema":"universa-recurrent.trace.v1"}])
def test_malformed_records(bad):
    assert not verify_record(bad).accepted


def test_invalid_witnesses():
    assert not check_projection([[1.,1.]],[1.,2.],[1.,2.],[0.]).accepted
    assert not check_projection([[1.,1.]],[1.,2.],[float('nan'),2.],[0.]).accepted
    assert not check_optimality([[1.]],[1.],[[0.]],[0.],[0.],ridge=-1).accepted
