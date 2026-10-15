"""Tests for `neural_ids.nn.optim.SGD`: momentum checked against a hand calculation."""

import numpy as np
import pytest

from neural_ids.nn.model import MLP
from neural_ids.nn.optim import SGD


def test_two_momentum_steps_match_hand_calculation() -> None:
    # lr = 0.1, momentum = 0.9, PyTorch form: v = 0.9·v + g ; p = p − 0.1·v
    p = np.array([1.0, -2.0])
    opt = SGD([p], lr=0.1, momentum=0.9)

    # step 1: v = 0 + [0.5, 1.0] = [0.5, 1.0]; p = [1 − 0.05, −2 − 0.1] = [0.95, −2.1]
    opt.step([np.array([0.5, 1.0])])
    np.testing.assert_allclose(p, [0.95, -2.1], rtol=0, atol=1e-15)

    # step 2: v = 0.9·[0.5, 1.0] + [0.5, −1.0] = [0.95, −0.1]
    #         p = [0.95 − 0.095, −2.1 + 0.01] = [0.855, −2.09]
    opt.step([np.array([0.5, -1.0])])
    np.testing.assert_allclose(p, [0.855, -2.09], rtol=0, atol=1e-15)
    np.testing.assert_allclose(opt.velocity[0], [0.95, -0.1], rtol=0, atol=1e-15)


def test_zero_momentum_is_plain_gradient_descent() -> None:
    p = np.array([[1.0, 2.0], [3.0, 4.0]])
    g = np.array([[1.0, -1.0], [0.5, 0.0]])
    opt = SGD([p], lr=0.5, momentum=0.0)
    opt.step([g])
    opt.step([g])
    np.testing.assert_allclose(p, [[0.0, 3.0], [2.5, 4.0]])


def test_step_updates_the_model_arrays_in_place() -> None:
    model = MLP([3, 4, 1], np.random.default_rng(0))
    params = model.parameters()
    before = [p.copy() for p in params]
    ids = [id(p) for p in params]
    opt = SGD(params, lr=0.1, momentum=0.9)

    x = np.random.default_rng(1).normal(size=(5, 3))  # (5, 3)
    model.forward(x)
    model.backward(np.ones((5, 1)))
    opt.step(model.gradients())  # read fresh after backward

    after = model.parameters()
    assert [id(p) for p in after] == ids  # same objects: nothing was rebound
    assert all(not np.array_equal(a, b) for a, b in zip(after, before, strict=True))


def test_step_checks_gradient_count_and_shapes() -> None:
    opt = SGD([np.zeros(3), np.zeros((2, 2))], lr=0.1, momentum=0.9)
    with pytest.raises(ValueError, match="2 parameters"):
        opt.step([np.zeros(3)])
    with pytest.raises(ValueError, match="shape"):
        opt.step([np.zeros(3), np.zeros((2, 3))])


@pytest.mark.parametrize(("lr", "momentum"), [(0.0, 0.9), (-0.1, 0.9), (0.1, -0.1), (0.1, 1.0)])
def test_rejects_bad_hyperparameters(lr: float, momentum: float) -> None:
    with pytest.raises(ValueError):
        SGD([np.zeros(2)], lr=lr, momentum=momentum)
