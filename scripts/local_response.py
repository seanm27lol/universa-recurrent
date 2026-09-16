"""A local worksheet: individual sensitivities, curvature, and interaction terms.

The model is sampled before prediction. A fitted rule contains numbers only;
predict() cannot call the model or inspect the target answer. This is ordinary
finite-difference local approximation, not a learned language interpreter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def finite(value, ndim, name):
    x = np.asarray(value)
    if x.ndim != ndim or not x.size or x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError('invalid finite ' + name)
    return x.astype(np.float64)


def stencil(width: int, radius: float) -> np.ndarray:
    if type(width) is not int or not 1 <= width <= 16:
        raise ValueError('width must be an integer in [1,16]')
    if type(radius) not in (int, float) or not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError('radius must be finite and in (0,1]')
    eye = np.eye(width) * radius
    offsets = [np.zeros(width)]
    for i in range(width): offsets.extend([eye[i], -eye[i]])
    for i in range(width):
        for j in range(i + 1, width):
            offsets.extend([eye[i] + eye[j], eye[i] - eye[j], -eye[i] + eye[j], -eye[i] - eye[j]])
    return np.stack(offsets)


@dataclass(frozen=True)
class LocalRule:
    base: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    radius: float

    def predict(self, displacement, *, order='full_quadratic'):
        """Normalized actual displacement, shape [examples, fields]. No model argument."""
        u = finite(displacement, 2, 'displacement')
        if u.shape != (len(self.base), self.jacobian.shape[-1]):
            raise ValueError('displacement dimensions differ from rule')
        if order not in ('none', 'linear', 'diagonal_quadratic', 'full_quadratic'):
            raise ValueError('unknown response approximation')
        y = self.base.copy()
        with np.errstate(over='raise', invalid='raise'):
            if order != 'none': y += np.einsum('bnd,bd->bn', self.jacobian, u)
            if order == 'diagonal_quadratic':
                y += .5 * np.einsum('bnd,bd->bn', np.diagonal(self.hessian, axis1=-2, axis2=-1), u*u)
            if order == 'full_quadratic':
                y += .5 * np.einsum('bnij,bi,bj->bn', self.hessian, u, u)
        if not np.isfinite(y).all(): raise ValueError('nonfinite prediction')
        return y

    def stored_bytes_per_input(self):
        return sum(x[0].nbytes for x in (self.base, self.jacobian, self.hessian))


def build_rule(offsets, responses) -> LocalRule:
    """Only stencil responses, never the evaluated target displacement or answer.

    Responses are [probes, examples, output coordinates]. Central differences
    approximate derivatives in normalized state coordinates. Inference remains
    float32; coefficients and arithmetic are float64. No exact-derivative claim.
    """
    u = finite(offsets, 2, 'stencil'); y = finite(responses, 3, 'responses')
    width = u.shape[1]
    if len(u) < 3: raise ValueError('incomplete stencil')
    radius = float(u[1, 0])
    expected = stencil(width, radius)
    if u.shape != expected.shape or not np.array_equal(u, expected) or len(y) != len(u):
        raise ValueError('responses require complete canonical stencil order')
    base = y[0].copy(); jac = np.empty((*base.shape, width)); hess = np.zeros((*base.shape, width, width))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for i in range(width):
            plus, minus = y[1 + 2*i:3 + 2*i]
            jac[..., i] = (plus - minus) / (2*radius)
            hess[..., i, i] = (plus - 2*base + minus) / radius**2
        p = 1 + 2*width
        for i in range(width):
            for j in range(i + 1, width):
                pp, pm, mp, mm = y[p:p+4]; p += 4
                mixed = (pp - pm - mp + mm) / (4*radius**2)
                hess[..., i, j] = mixed; hess[..., j, i] = mixed
    if not np.isfinite(jac).all() or not np.isfinite(hess).all(): raise ValueError('nonfinite coefficients')
    return LocalRule(base, jac, hess, radius)


def query_plan(width=6):
    """Fixed new signed/dense combinations and exact no-op/gauge controls.

    No tuning on the composition cohort. Deltas are dimensionless; frozen scales
    are applied by the runner. Random directions are reproducible protocol values.
    """
    if width != 6: raise ValueError('pilot requires six named fields')
    vectors = [
        ('two_cycles_0', [1, 1, 0, 0, 0, 0]),
        ('two_cycles_1', [0, 0, 1, -1, 0, 0]),
        ('cross_candidate', [1, 0, -1, 0, 0, 0]),
        ('cycle_evidence_0', [-1, 0, 0, 0, 1, 0]),
        ('cycle_evidence_1', [0, 0, 1, 0, 0, -1]),
        ('common_evidence_control', [0, 0, 0, 0, 1, 1]),
        ('zero_control', [0, 0, 0, 0, 0, 0]),
    ]
    rng = np.random.default_rng(88000)
    for index in range(4):
        v = rng.uniform(-1, 1, width); v /= np.max(np.abs(v))
        vectors.append((f'dense_{index}', v.tolist()))
    return [(name, scale, np.asarray(vector, dtype=np.float64)*scale)
            for scale in (.25, .5, 1.) for name, vector in vectors]


def score(prediction, actual, baseline):
    p, a, b = [finite(x, 2, 'prediction/actual/baseline') for x in (prediction, actual, baseline)]
    if p.shape != a.shape or p.shape != b.shape: raise ValueError('output shapes differ')
    error = (p-a)**2; energy = float(((a-b)**2).mean())
    return dict(prediction_mse=float(error.mean()), effect_mse=energy,
                relative_rms_error=float(np.sqrt(error.mean()/energy)) if energy > 1e-12 else None,
                relative_floor_mse=1e-12, worst_component_error=float(np.abs(p-a).max()),
                p99_max_component_error=float(np.quantile(np.abs(p-a).max(1), .99)))
