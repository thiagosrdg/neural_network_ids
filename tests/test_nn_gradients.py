"""Gradient checks (float64) and shape tests for the NumPy layers and the MLP (T05).

Each check compares the hand-written `backward` with central finite differences from
`neural_ids.gradcheck` and requires a relative error below 1e-5.

The ReLU kink: ReLU is not differentiable at 0. If a pre-activation z lies within the
finite-difference step of 0, the nudges z ± (a tiny change) land on different sides of the
kink and the numerical gradient is wrong, even when `backward` is right. These tests keep every
pre-activation farther from 0 than KINK_MARGIN:
- The ReLU-only test builds its input as sign(x)·(|x| + 0.1), so |x| ≥ 0.1 by construction.
- The MLP tests run the forward pass first and assert, with `assert_away_from_kink`, that every
  Dense output that feeds a ReLU satisfies |z| > KINK_MARGIN. The seeds are fixed, so this holds
  every run; if someone changes a seed and a value lands near 0, the test fails with that
  message instead of a confusing gradient error.
Why 1e-3 is enough: nudging one parameter by EPS = 1e-6 moves any pre-activation by about
EPS·|input| ≈ 1e-6 · 3 (the inputs are standard normal), which is far below 1e-3.
"""

from collections.abc import Callable

import numpy as np
import pytest

from neural_ids.gradcheck import (
    assert_gradients_close,
    gradient_mismatches,
    numerical_gradient,
    relative_error,
)
from neural_ids.models import logreg
from neural_ids.nn import functional
from neural_ids.nn.functional import bce_loss, sigmoid
from neural_ids.nn.layers import BCEWithLogits, Dense, ReLU
from neural_ids.nn.model import MLP

EPS = 1e-6  # finite-difference step
TOL = 1e-5  # maximum relative error (spec)
KINK_MARGIN = 1e-3  # every ReLU input must satisfy |z| > KINK_MARGIN (see module docstring)


def check(analytic: np.ndarray, loss_fn: Callable[[np.ndarray], float], w: np.ndarray) -> float:
    """Compare `analytic` with the numerical gradient of `loss_fn` at `w`, entry by entry.

    Fails if any entry breaks |analytic - numeric| <= 1e-8 + 1e-5·|numeric|. Returns the max
    relative error, which each test also keeps below TOL.
    """
    numeric = numerical_gradient(loss_fn, w, eps=EPS)
    return assert_gradients_close(analytic, numeric)


def assert_away_from_kink(mlp: MLP, x: np.ndarray) -> None:
    """Assert that every Dense output that feeds a ReLU is farther from 0 than KINK_MARGIN."""
    h = x
    for layer, nxt in zip(mlp.layers, mlp.layers[1:] + [None], strict=True):
        h = layer.forward(h)
        if isinstance(nxt, ReLU):
            closest = float(np.min(np.abs(h)))
            assert closest > KINK_MARGIN, (
                f"a pre-activation is {closest:.2e} from the ReLU kink; change the seed"
            )


# ---------------------------------------------------------------- shared stable math


def test_logreg_and_layers_share_one_implementation() -> None:
    assert logreg.sigmoid is functional.sigmoid
    assert logreg.bce_loss is functional.bce_loss


def test_bce_loss_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        bce_loss(np.zeros((4, 1)), np.zeros(4))


# ---------------------------------------------------------------- Dense


def test_dense_he_init() -> None:
    layer = Dense(1000, 500, np.random.default_rng(0))
    assert layer.W.shape == (1000, 500) and layer.W.dtype == np.float64
    assert layer.b.shape == (500,) and np.all(layer.b == 0.0)
    # He: std = sqrt(2 / fan_in) = sqrt(2 / 1000) ≈ 0.0447; 500 000 draws, so 2% is loose.
    assert layer.W.std() == pytest.approx(np.sqrt(2 / 1000), rel=0.02)
    assert abs(layer.W.mean()) < 1e-3


def test_dense_forward_shape_and_value() -> None:
    rng = np.random.default_rng(1)
    layer = Dense(5, 3, rng)
    layer.b[:] = [1.0, 2.0, 3.0]
    x = rng.normal(size=(4, 5))
    out = layer.forward(x)
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out, x @ layer.W + layer.b)


def test_dense_backward_shapes() -> None:
    rng = np.random.default_rng(2)
    layer = Dense(5, 3, rng)
    layer.forward(rng.normal(size=(4, 5)))
    grad_x = layer.backward(rng.normal(size=(4, 3)))
    assert grad_x.shape == (4, 5)
    assert layer.dW.shape == (5, 3)
    assert layer.db.shape == (3,)


def test_dense_backward_before_forward_raises() -> None:
    layer = Dense(2, 2, np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="forward"):
        layer.backward(np.ones((1, 2)))


def _dense_setup() -> tuple[Dense, np.ndarray, np.ndarray]:
    """A Dense layer, an input (4, 5), and fixed weights g (4, 3) for the loss L = sum(out · g)."""
    rng = np.random.default_rng(3)
    layer = Dense(5, 3, rng)
    layer.b[:] = rng.normal(size=3)  # nonzero bias, so db is tested on a general point
    x = rng.normal(size=(4, 5))  # x: (batch, in)
    g = rng.normal(size=(4, 3))  # dL/d(out) for L = sum(out * g)
    return layer, x, g


def test_dense_gradcheck() -> None:
    layer, x, g = _dense_setup()
    layer.forward(x)
    grad_x = layer.backward(g)
    dW, db = layer.dW.copy(), layer.db.copy()

    def loss_w(w: np.ndarray) -> float:
        return float(np.sum((x @ w + layer.b) * g))

    def loss_b(b: np.ndarray) -> float:
        return float(np.sum((x @ layer.W + b) * g))

    def loss_x(xx: np.ndarray) -> float:
        return float(np.sum((xx @ layer.W + layer.b) * g))

    assert check(dW, loss_w, layer.W) < TOL
    assert check(db, loss_b, layer.b) < TOL
    assert check(grad_x, loss_x, x) < TOL


def test_gradcheck_catches_a_wrong_gradient() -> None:
    """A check that never fails proves nothing: a dW off by 1% must be rejected."""
    layer, x, g = _dense_setup()
    layer.forward(x)
    layer.backward(g)
    wrong_dW = layer.dW * 1.01

    def loss_w(w: np.ndarray) -> float:
        return float(np.sum((x @ w + layer.b) * g))

    numeric = numerical_gradient(loss_w, layer.W, eps=EPS)
    assert relative_error(wrong_dW, numeric) > TOL  # about 0.01 / 2.01 ≈ 5e-3
    with pytest.raises(AssertionError, match="gradient entries"):
        assert_gradients_close(wrong_dW, numeric)


def test_per_element_check_catches_a_small_wrong_entry() -> None:
    """One tiny entry of dW is wrong next to large correct entries.

    Feature 0 is scaled by 1e-6, so row 0 of dW is about 1e-6 while other entries reach about 5.
    Flipping the sign of dW[0, 0] is a 200% error in that entry, but the old global check divides
    by the largest magnitude in the whole array: relative_error gives 1.62e-7 here, below the
    1e-5 limit, so it passes the bug. The per-element rule catches it.
    """
    layer, x, g = _dense_setup()
    x[:, 0] *= 1e-6  # x: (4, 5); feature 0 is tiny, so dW[0, :] is tiny
    layer.forward(x)
    layer.backward(g)

    def loss_w(w: np.ndarray) -> float:
        return float(np.sum((x @ w + layer.b) * g))

    numeric = numerical_gradient(loss_w, layer.W, eps=EPS)
    assert assert_gradients_close(layer.dW, numeric) < TOL  # the correct gradient passes

    wrong_dW = layer.dW.copy()
    wrong_dW[0, 0] *= -1.0
    assert relative_error(wrong_dW, numeric) < TOL  # the old check alone would miss it
    bad = gradient_mismatches(wrong_dW, numeric)
    assert bad.sum() == 1 and bad[0, 0]
    with pytest.raises(AssertionError, match=r"1 of 15 gradient entries.*\(0, 0\)"):
        assert_gradients_close(wrong_dW, numeric)


# ---------------------------------------------------------------- ReLU


def test_relu_forward_values() -> None:
    out = ReLU().forward(np.array([[-2.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(out, [[0.0, 0.0, 3.0]])


def test_relu_backward_uses_the_mask() -> None:
    relu = ReLU()
    relu.forward(np.array([[-2.0, 0.0, 3.0]]))
    grad = relu.backward(np.array([[5.0, 5.0, 5.0]]))
    np.testing.assert_array_equal(grad, [[0.0, 0.0, 5.0]])  # derivative at 0 taken as 0


def test_relu_gradcheck() -> None:
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(4, 6))
    x = np.sign(raw) * (np.abs(raw) + 0.1)  # |x| ≥ 0.1: no value near the kink
    g = rng.normal(size=(4, 6))
    relu = ReLU()
    relu.forward(x)
    grad_x = relu.backward(g)
    assert grad_x.shape == x.shape
    assert check(grad_x, lambda xx: float(np.sum(np.maximum(xx, 0.0) * g)), x) < TOL


# ---------------------------------------------------------------- BCEWithLogits


def _bce_batch() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(6, 1)) * 3.0  # (B, 1), the shape the MLP produces
    y = np.array([0, 1, 1, 0, 1, 0], dtype=np.float64)  # (B,)
    return logits, y


def test_bce_with_logits_value_matches_functional() -> None:
    logits, y = _bce_batch()
    loss = BCEWithLogits().forward(logits, y)
    assert loss == pytest.approx(bce_loss(logits[:, 0], y))


def test_bce_with_logits_gradient_shape_and_formula() -> None:
    logits, y = _bce_batch()
    fn = BCEWithLogits()
    fn.forward(logits, y)
    grad = fn.backward()
    assert grad.shape == (6, 1)  # same shape as the logits, never (B, B)
    np.testing.assert_allclose(grad[:, 0], (sigmoid(logits[:, 0]) - y) / 6)


def test_bce_with_logits_does_not_broadcast_to_b_by_b() -> None:
    """(B, 1) − (B,) silently becomes (B, B). The loss must differ from that wrong value."""
    logits, y = _bce_batch()
    broadcast_loss = float(
        np.mean(np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
    )  # logits * y is (6, 6) here: this is the bug
    loss = BCEWithLogits().forward(logits, y)
    assert loss != pytest.approx(broadcast_loss)
    # Labels shaped like the logits, (B, 1), are rejected rather than guessed.
    with pytest.raises(ValueError, match="y must have shape"):
        BCEWithLogits().forward(logits, y[:, None])
    with pytest.raises(ValueError, match="logits must have shape"):
        BCEWithLogits().forward(np.zeros((6, 2)), y)


def test_bce_with_logits_gradcheck() -> None:
    logits, y = _bce_batch()
    fn = BCEWithLogits()
    fn.forward(logits, y)
    grad = fn.backward()
    assert check(grad, lambda z: BCEWithLogits().forward(z, y), logits) < TOL


def test_bce_with_logits_is_stable_for_huge_logits() -> None:
    fn = BCEWithLogits()
    loss = fn.forward(np.array([[1000.0], [-1000.0]]), np.array([0.0, 1.0]))
    assert loss == pytest.approx(1000.0)
    assert np.all(np.isfinite(fn.backward()))


# ---------------------------------------------------------------- MLP


def test_mlp_layer_structure_and_output_shape() -> None:
    rng = np.random.default_rng(6)
    mlp = MLP([29, 64, 32, 1], rng)
    kinds = [type(layer).__name__ for layer in mlp.layers]
    assert kinds == ["Dense", "ReLU", "Dense", "ReLU", "Dense"]  # no ReLU on the logits
    assert mlp.forward(rng.normal(size=(7, 29))).shape == (7, 1)


def test_mlp_parameter_count_matches_architecture() -> None:
    # docs/ARCHITECTURE.md: (29·64 + 64) + (64·32 + 32) + (32·1 + 1) = 1920 + 2080 + 33
    assert MLP([29, 64, 32, 1], np.random.default_rng(0)).n_parameters == 4033


def test_mlp_rejects_too_few_layer_sizes() -> None:
    with pytest.raises(ValueError):
        MLP([29], np.random.default_rng(0))


def test_mlp_parameters_and_gradients_same_order_and_shapes() -> None:
    rng = np.random.default_rng(7)
    mlp = MLP([5, 4, 3, 1], rng)
    x = rng.normal(size=(4, 5))
    mlp.backward(np.ones_like(mlp.forward(x)))
    params, grads = mlp.parameters(), mlp.gradients()
    assert [p.shape for p in params] == [(5, 4), (4,), (4, 3), (3,), (3, 1), (1,)]
    assert [g.shape for g in grads] == [p.shape for p in params]
    dense = [layer for layer in mlp.layers if isinstance(layer, Dense)]
    assert all(g is a for g, a in zip(grads[0::2], [d.dW for d in dense], strict=True))
    assert all(p is a for p, a in zip(params[0::2], [d.W for d in dense], strict=True))


def test_mlp_parameters_are_the_models_own_arrays() -> None:
    """T06's optimizer updates parameters in place, so parameters() must not return copies."""
    rng = np.random.default_rng(8)
    mlp = MLP([5, 4, 3, 1], rng)
    x = rng.normal(size=(4, 5))
    before = mlp.forward(x).copy()
    last_bias = mlp.parameters()[-1]  # (1,), the output bias
    last_bias += 2.5  # in-place change through the returned array
    np.testing.assert_allclose(mlp.forward(x), before + 2.5)


def _mlp_gradcheck(layer_sizes: list[int], batch: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mlp = MLP(layer_sizes, rng)
    for p in mlp.parameters()[1::2]:
        p[:] = rng.normal(scale=0.1, size=p.shape)  # nonzero biases: test a general point
    x = rng.normal(size=(batch, layer_sizes[0]))  # x: (batch, features)
    y = rng.integers(0, 2, size=batch).astype(np.float64)  # y: (batch,)
    assert_away_from_kink(mlp, x)

    loss_fn = BCEWithLogits()
    loss_fn.forward(mlp.forward(x), y)
    grad_x = mlp.backward(loss_fn.backward())  # dL/dx: (batch, features)
    analytic = [g.copy() for g in mlp.gradients()]

    def loss_of_input(xx: np.ndarray) -> float:
        return BCEWithLogits().forward(mlp.forward(xx), y)

    assert check(grad_x, loss_of_input, x) < TOL  # the input gradient (used by evasion attacks)

    for p, g in zip(mlp.parameters(), analytic, strict=True):
        original = p.copy()

        def loss_at(w: np.ndarray, p: np.ndarray = p) -> float:
            p[...] = w  # write the candidate value into the model's own array
            return BCEWithLogits().forward(mlp.forward(x), y)

        err = check(g, loss_at, original)
        p[...] = original
        assert err < TOL, f"parameter {p.shape}: relative error {err:.2e}"


def test_mlp_gradcheck_tiny() -> None:
    _mlp_gradcheck([5, 4, 3, 1], batch=4, seed=9)


def test_mlp_gradcheck_architecture_size() -> None:
    _mlp_gradcheck([29, 64, 32, 1], batch=3, seed=15)  # closest |z| is 1.4e-2
