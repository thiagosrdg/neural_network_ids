"""Tests for `neural_ids.gradcheck` on functions with known gradients."""

import numpy as np
import pytest

from neural_ids.gradcheck import numerical_gradient, relative_error
from neural_ids.utils import set_seed


def test_quadratic_gradient() -> None:
    # L(w) = sum(w²) -> dL/dw = 2w
    w = set_seed(0).normal(size=(3, 4))  # float64
    grad = numerical_gradient(lambda v: float(np.sum(v**2)), w)
    assert relative_error(grad, 2 * w) < 1e-8


def test_does_not_modify_input() -> None:
    w = np.array([1.0, -2.0, 3.0])
    numerical_gradient(lambda v: float(np.sum(np.sin(v))), w)
    np.testing.assert_array_equal(w, [1.0, -2.0, 3.0])


def test_rejects_float32() -> None:
    with pytest.raises(TypeError):
        numerical_gradient(lambda v: float(np.sum(v)), np.zeros(3, dtype=np.float32))


def test_relative_error_is_scale_free_and_safe_at_zero() -> None:
    a = np.array([1.0, 2.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(1e6 * a, 1e6 * a * (1 + 1e-9)) < 1e-8
    assert relative_error(np.zeros(2), np.zeros(2)) == 0.0
