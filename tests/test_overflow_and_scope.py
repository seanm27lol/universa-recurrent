import numpy as np
from universa_recurrent.examples import flow_example
from universa_recurrent.recurrence import solve
from universa_recurrent.verification import check_projection,check_optimality,verify_record


def test_overflow_never_passes():
    with np.errstate(over='ignore',invalid='ignore'):
        check=check_projection([[1e308]],[1e308],[1e308],[1e308])
        assert not check.accepted
        check=check_optimality([[1e308]],[1e308],[[1e308]],[1e308],[1e308],ridge=1.)
        assert not check.accepted


def test_cannot_relabel_compact_as_full_evidence():
    ex=flow_example()
    r=solve(ex.problem,ex.candidates[0]).record
    r['intermediate_evidence']='full_numeric_witnesses'
    assert not verify_record(r).accepted
