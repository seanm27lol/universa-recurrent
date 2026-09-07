import numpy as np
import pytest
from universa_recurrent.structures import CompiledConstraint, incidence_matrix
from universa_recurrent.verification import check_projection


def test_incidence_signs():
    np.testing.assert_array_equal(incidence_matrix(3, [(0,1), (1,2)]), [[-1,0],[1,-1],[0,1]])


@pytest.mark.parametrize("nodes,edges", [(0,[(0,1)]),(2,[]),(2,[(0,0)]),(2,[(0,2)]),(True,[(0,1)]),(2,[(False,1)])])
def test_bad_graphs(nodes, edges):
    with pytest.raises(ValueError):
        incidence_matrix(nodes, edges)


@pytest.mark.parametrize("seed", range(10))
def test_random_projection_witness(seed):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(3,7))
    c = CompiledConstraint.compile("random", b)
    v = rng.normal(size=7)
    z, lam = c.projection_witness(v)
    reference = v - b.T @ np.linalg.solve(b @ b.T, b @ v)
    np.testing.assert_allclose(z, reference, atol=1e-10)
    assert check_projection(b, v, z, lam).accepted
    np.testing.assert_allclose(c.project(z), z, atol=1e-10)


@pytest.mark.parametrize("b", [np.zeros((0,4)), np.zeros((2,4)), np.eye(4), np.array([[1.,2.,3.],[2.,4.,6.]])])
def test_rank_edges(b):
    c = CompiledConstraint.compile("edgecase", b)
    v = np.arange(b.shape[1], dtype=float)
    z, lam = c.projection_witness(v)
    assert check_projection(b, v, z, lam).accepted
    assert c.latent_dimension == b.shape[1] - c.rank


@pytest.mark.parametrize("b", [[[float('nan')]], [[float('inf')]], [], [1,2], np.zeros((2,0))])
def test_bad_boundaries(b):
    with pytest.raises(ValueError):
        CompiledConstraint.compile("bad", b)


def test_no_factorization_in_repeated_projection(monkeypatch):
    b = incidence_matrix(3, [(0,1),(1,2),(2,0)])
    c = CompiledConstraint.compile("cycle", b)
    def fail(*a, **k):
        raise AssertionError("Unexpected repeated decomposition")
    monkeypatch.setattr(np.linalg, "svd", fail)
    for _ in range(5):
        c.project([1,2,3])
        c.projection_witness([1,2,3])


def test_copied_readonly_cache():
    b = np.array([[1.,-1.]])
    c = CompiledConstraint.compile("copy", b)
    key = c.fingerprint
    b[0,0] = 99
    assert c.boundary[0,0] == 1
    assert c.fingerprint == key
    assert CompiledConstraint.compile("changed", b).fingerprint != key
    with pytest.raises(ValueError):
        c.boundary[0,0] = 4
