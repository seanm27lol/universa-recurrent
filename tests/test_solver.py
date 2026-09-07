import numpy as np
import pytest
from universa_recurrent.examples import flow_example
from universa_recurrent.recurrence import ReconstructionProblem, direct_solve, solve
from universa_recurrent.structures import CompiledConstraint
from universa_recurrent.verification import verify_record


def test_demo_converges_matches_direct_and_descends():
    ex = flow_example()
    c = ex.candidates[0]
    result = solve(ex.problem, c, trace_mode="full")
    assert result.iterations < 256
    assert result.stop_reason == "stationarity_tolerance"
    np.testing.assert_allclose(result.state, direct_solve(ex.problem, c), atol=1e-7)
    losses = [e["objective"] for e in result.record["events"]]
    assert np.all(np.diff(losses) <= 1e-12)
    assert verify_record(result.record).accepted


@pytest.mark.parametrize("seed", range(8))
def test_independent_kkt_reference(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(8,5))
    b = rng.normal(size=(2,5))
    y = rng.normal(size=8)
    p = ReconstructionProblem(a, y, ridge=0.2)
    c = CompiledConstraint.compile("random", b)
    # Separate ambient-coordinate reference, not the reduced producer.
    h = a.T @ a + p.ridge * np.eye(5)
    kkt = np.block([[h, b.T],[b, np.zeros((2,2))]])
    answer = np.linalg.solve(kkt, np.r_[a.T @ y, np.zeros(2)])[:5]
    np.testing.assert_allclose(direct_solve(p,c), answer, atol=1e-9)
    result = solve(p,c,max_steps=2000, trace_mode="full")
    np.testing.assert_allclose(result.state, answer, atol=1e-7)
    assert verify_record(result.record).accepted


def test_budget_is_not_convergence():
    ex = flow_example()
    result = solve(ex.problem, ex.candidates[0], max_steps=1)
    assert result.stop_reason == "iteration_budget"
    assert not verify_record(result.record).accepted


def test_zero_budget_and_fixed_depth():
    ex=flow_example()
    zero=solve(ex.problem,ex.candidates[0],max_steps=0)
    assert zero.iterations == 0
    assert zero.stop_reason == "iteration_budget"
    fixed=solve(ex.problem,ex.candidates[0],max_steps=64,adaptive=False)
    assert fixed.iterations == 64
    assert verify_record(fixed.record).accepted


@pytest.mark.parametrize("b", [np.eye(3), np.zeros((0,3)), np.zeros((2,3))])
def test_zero_dim_and_unconstrained(b):
    p=ReconstructionProblem(np.eye(3),[1,2,3],ridge=0.1)
    c=CompiledConstraint.compile("edge",b)
    out=solve(p,c,trace_mode="full")
    np.testing.assert_allclose(out.state,direct_solve(p,c),atol=1e-10)
    assert verify_record(out.record).accepted


@pytest.mark.parametrize("ridge", [0,-1,float('nan'),float('inf'),True])
def test_bad_ridge(ridge):
    with pytest.raises(ValueError):
        ReconstructionProblem(np.eye(2), [1,2], ridge=ridge)


@pytest.mark.parametrize("kwargs", [{"max_steps":-1},{"max_steps":True},{"tolerance":0},{"trace_mode":"magic"},{"adaptive":1}])
def test_bad_solver_inputs(kwargs):
    ex=flow_example()
    with pytest.raises(ValueError):
        solve(ex.problem,ex.candidates[0],**kwargs)


def test_dimension_mismatch():
    p=ReconstructionProblem(np.eye(2),[1,2])
    with pytest.raises(ValueError):
        solve(p,CompiledConstraint.compile("bad",np.eye(3)))
