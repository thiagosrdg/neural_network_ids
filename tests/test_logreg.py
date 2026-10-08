"""Tests for `neural_ids.models.logreg`: stability, gradient check, and training.

Synthetic arrays only: these tests check the code, not detection.
"""

import math
import warnings

import numpy as np
import pytest
from sklearn.metrics import log_loss

from neural_ids.gradcheck import numerical_gradient, relative_error
from neural_ids.models.logreg import LogisticRegression, bce_loss, sigmoid
from neural_ids.utils import set_seed


def _blobs(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Gaussian blobs (not perfectly separable), float64."""
    rng = set_seed(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, d)) + np.where(y[:, None] == 1, 0.8, -0.8)  # (n, d)
    return x, y


def test_sigmoid_matches_naive_formula_where_it_is_safe() -> None:
    z = np.linspace(-30, 30, 121)
    np.testing.assert_allclose(sigmoid(z), 1 / (1 + np.exp(-z)), rtol=1e-12)
    assert sigmoid(np.array([0.0]))[0] == 0.5


def test_sigmoid_extreme_logits_no_overflow() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an overflow in np.exp would raise here
        p = sigmoid(np.array([-1000.0, 1000.0]))
    np.testing.assert_array_equal(p, [0.0, 1.0])


def test_bce_matches_sklearn_log_loss() -> None:
    rng = set_seed(0)
    z = rng.normal(scale=3, size=200)
    y = rng.integers(0, 2, size=200)
    assert bce_loss(z, y) == pytest.approx(log_loss(y, sigmoid(z)), rel=1e-9)


def test_bce_and_gradient_finite_at_logits_plus_minus_1000() -> None:
    z = np.array([1000.0, -1000.0, 1000.0, -1000.0])
    y = np.array([1, 0, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss = bce_loss(z, y)
        # right answers cost 0, confidently wrong answers cost |z| = 1000 each
        assert loss == pytest.approx((0 + 0 + 1000 + 1000) / 4)
        # gradient through the model: one feature, w = 1000, so logits are ±1000
        model = LogisticRegression(n_features=1)
        model.w = np.array([1000.0])
        x = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        dw, db = model.gradients(x, y)
    assert math.isfinite(loss)
    assert np.all(np.isfinite(dw)) and math.isfinite(db)
    np.testing.assert_allclose(dw, [(0 + 0 + 1 + 1) / 4])  # xᵀ(p − y)/n with p − y = [0,0,1,-1]


def test_first_loss_is_ln2_with_zero_weights() -> None:
    x, y = _blobs(100, 4, seed=1)
    model = LogisticRegression(n_features=4)
    assert model.loss(x, y) == pytest.approx(math.log(2), abs=1e-12)
    model.fit(x, y, x, y, set_seed(0))
    assert model.history["train_loss"][0] == pytest.approx(math.log(2), abs=1e-12)


def test_gradient_check_float64() -> None:
    x, y = _blobs(30, 5, seed=2)
    model = LogisticRegression(n_features=5)
    rng = set_seed(3)
    model.w = rng.normal(size=5)
    model.b = 0.3
    dw, db = model.gradients(x, y)

    def loss_of(theta: np.ndarray) -> float:  # theta = [w..., b]
        return bce_loss(x @ theta[:-1] + theta[-1], y)

    theta = np.append(model.w, model.b)  # (6,) float64
    numeric = numerical_gradient(loss_of, theta)
    assert relative_error(np.append(dw, db), numeric) < 1e-6


def test_update_moves_against_gradient() -> None:
    model = LogisticRegression(n_features=2, lr=0.5)
    model.w = np.array([1.0, -1.0])
    model.b = 2.0
    model.update(np.array([0.2, -0.4]), 1.0)
    np.testing.assert_allclose(model.w, [0.9, -0.8])
    assert model.b == pytest.approx(1.5)


def test_one_step_lowers_the_loss() -> None:
    x, y = _blobs(200, 3, seed=4)
    model = LogisticRegression(n_features=3, lr=0.1)
    before = model.loss(x, y)
    model.update(*model.gradients(x, y))
    assert model.loss(x, y) < before


def test_training_loss_goes_down_and_predicts_well() -> None:
    x, y = _blobs(500, 3, seed=5)
    model = LogisticRegression(n_features=3, epochs=10).fit(x, y, x, y, set_seed(6))
    losses = model.history["train_loss"]
    assert len(losses) == 11
    assert losses[-1] < 0.5 * losses[0]
    proba = model.predict_proba(x)
    assert proba.shape == (500,) and np.all((proba >= 0) & (proba <= 1))
    assert np.mean(model.predict(x) == y) > 0.9


def test_same_seed_same_weights() -> None:
    x, y = _blobs(100, 2, seed=7)
    a = LogisticRegression(2, epochs=3).fit(x, y, x, y, set_seed(8))
    b = LogisticRegression(2, epochs=3).fit(x, y, x, y, set_seed(8))
    np.testing.assert_array_equal(a.w, b.w)
