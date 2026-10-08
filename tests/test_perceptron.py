"""Tests for `neural_ids.models.perceptron` on small synthetic arrays."""

import numpy as np

from neural_ids.models.perceptron import Perceptron, step
from neural_ids.utils import set_seed


def _separable(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs, far apart: a line separates them."""
    rng = set_seed(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, 2)) + np.where(y[:, None] == 1, 3.0, -3.0)  # (n, 2)
    return x, y


def test_step() -> None:
    np.testing.assert_array_equal(step(np.array([-1.0, 0.0, 2.0])), [0, 0, 1])


def test_update_rule_by_hand() -> None:
    model = Perceptron(n_features=2, lr=1.0)
    x = np.array([[1.0, 2.0], [3.0, -1.0]])
    y = np.array([1, 0])
    # zero weights -> both predicted 0; errors y - ŷ = [1, 0]
    model.update(x, y)
    np.testing.assert_allclose(model.w, [0.5, 1.0])  # 1.0 * ([1,2]*1 + [3,-1]*0) / 2
    assert model.b == 0.5


def test_correct_batch_does_not_change_weights() -> None:
    model = Perceptron(n_features=2, lr=1.0)
    model.w = np.array([1.0, 0.0])
    model.update(np.array([[2.0, 0.0], [-2.0, 0.0]]), np.array([1, 0]))
    np.testing.assert_array_equal(model.w, [1.0, 0.0])
    assert model.b == 0.0


def test_learns_separable_data() -> None:
    x, y = _separable(400, seed=0)
    model = Perceptron(n_features=2, epochs=10).fit(x, y, x, y, set_seed(1))
    assert np.mean(model.predict(x) == y) > 0.98
    assert len(model.history["train_error"]) == 11  # epoch 0 + 10 epochs
    assert model.history["train_error"][-1] < model.history["train_error"][0]


def test_same_seed_same_weights() -> None:
    x, y = _separable(200, seed=2)
    a = Perceptron(2, epochs=3).fit(x, y, x, y, set_seed(5))
    b = Perceptron(2, epochs=3).fit(x, y, x, y, set_seed(5))
    np.testing.assert_array_equal(a.w, b.w)
