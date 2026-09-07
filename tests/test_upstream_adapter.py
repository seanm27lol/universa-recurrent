import numpy as np
import pytest
from universa_recurrent.structures import incidence_matrix
from universa_recurrent.structures.universa_adapter import compile_from_universa
from universa_recurrent.verification import check_projection


def test_pinned_universa_adapter():
    pytest.importorskip("universa",reason="Optional commit-pinned Universa dependency not installed")
    b=incidence_matrix(3,[(0,1),(1,2),(2,0)])
    c=compile_from_universa("upstream-cycle",b)
    z,lam=c.projection_witness([1.,2.,3.])
    np.testing.assert_allclose(z,[2.,2.,2.],atol=1e-10)
    assert check_projection(b,[1.,2.,3.],z,lam).accepted
