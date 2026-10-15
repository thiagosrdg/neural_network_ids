"""Cross-check: the NumPy MLP (T05–T06) and the PyTorch MLP compute exactly the same thing.

With the same weights in float64:
- the logits match (atol 1e-8);
- PyTorch's autograd gradients match our hand-written `backward` (rtol 1e-6), for the binary
  loss and for softmax cross-entropy (whose gradient is p − onehot(y), see docs/notes/softmax.md);
- a few steps of our `SGD` (momentum 0.9) and `torch.optim.SGD(momentum=0.9)` on the same
  batches give the same weights.
If one of these fails, either our backprop or the weight copy is wrong.
Synthetic random inputs only: these tests check code, not detection.
"""

import numpy as np
import pytest
import torch

from neural_ids.nn.functional import sigmoid
from neural_ids.nn.layers import BCEWithLogits
from neural_ids.nn.model import MLP
from neural_ids.nn.optim import SGD
from neural_ids.torch_models import MLPClassifier, copy_numpy_weights
from neural_ids.utils import set_seed

ATOL_LOGITS = 1e-8
RTOL_GRAD = 1e-6
SIZES = [29, 16, 8]  # inputs and hidden layers; the output size is added per test


def _pair(n_out: int, seed: int = 0) -> tuple[MLP, MLPClassifier]:
    """A NumPy MLP and a float64 PyTorch MLP holding the same weights.

    The NumPy output layer starts at He × 0.01; we rescale it to plain He so the logits and
    gradients are not tiny (a check on near-zero numbers proves little).
    """
    rng = set_seed(seed)
    np_mlp = MLP([*SIZES, n_out], rng)
    np_mlp.parameters()[-2][...] = rng.normal(0.0, np.sqrt(2.0 / SIZES[-1]), (SIZES[-1], n_out))
    np_mlp.parameters()[-1][...] = rng.normal(0.0, 0.1, n_out)  # non-zero output bias
    n_classes = 2 if n_out == 1 else n_out
    torch_mlp = MLPClassifier(SIZES[0], tuple(SIZES[1:]), n_classes).double()
    copy_numpy_weights(np_mlp, torch_mlp)
    return np_mlp, torch_mlp


def _torch_grads(model: MLPClassifier) -> list[np.ndarray]:
    """[dW1, db1, ...] in the NumPy layout: weight gradients transposed back to (in, out)."""
    grads = []
    for linear in model.linear_layers():
        grads += [linear.weight.grad.numpy().T, linear.bias.grad.numpy()]
    return grads


# --- The weight copy -----------------------------------------------------------------------


def test_copy_puts_transposed_weights_in_each_linear_layer() -> None:
    np_mlp, torch_mlp = _pair(1)
    params = np_mlp.parameters()
    for i, linear in enumerate(torch_mlp.linear_layers()):
        w, b = params[2 * i], params[2 * i + 1]  # W: (in, out), b: (out,)
        assert linear.weight.shape == (w.shape[1], w.shape[0])  # nn.Linear: (out, in)
        np.testing.assert_array_equal(linear.weight.detach().numpy(), w.T)
        np.testing.assert_array_equal(linear.bias.detach().numpy(), b)


def test_copy_is_a_copy_not_shared_memory() -> None:
    np_mlp, torch_mlp = _pair(1)
    np_mlp.parameters()[0][0, 0] += 1.0  # change the NumPy weight afterwards
    w0 = torch_mlp.linear_layers()[0].weight
    assert w0[0, 0].item() != np_mlp.parameters()[0][0, 0]


def test_copy_refuses_a_different_architecture() -> None:
    np_mlp = MLP([29, 16, 1], np.random.default_rng(0))
    with pytest.raises(ValueError):
        copy_numpy_weights(np_mlp, MLPClassifier(29, (8,), 2).double())


# --- Forward and backward ------------------------------------------------------------------


@pytest.mark.parametrize("n_out", [1, 4])
def test_logits_match(n_out: int) -> None:
    np_mlp, torch_mlp = _pair(n_out)
    x = set_seed(1).normal(size=(50, SIZES[0]))  # (50, 29) float64
    np_logits = np_mlp.forward(x)  # (50, n_out)
    with torch.no_grad():
        t_logits = torch_mlp(torch.from_numpy(x)).numpy()  # (50, n_out)
    assert t_logits.dtype == np.float64
    np.testing.assert_allclose(t_logits, np_logits, rtol=0, atol=ATOL_LOGITS)


def test_binary_gradients_match_autograd() -> None:
    """BCEWithLogits: our backward against autograd of nn.BCEWithLogitsLoss (mean)."""
    np_mlp, torch_mlp = _pair(1)
    rng = set_seed(2)
    x = rng.normal(size=(40, SIZES[0]))  # (40, 29)
    y = rng.integers(0, 2, 40)  # (40,)

    loss_fn = BCEWithLogits()
    np_loss = loss_fn.forward(np_mlp.forward(x), y.astype(np.float64))
    np_mlp.backward(loss_fn.backward())

    t_loss = torch.nn.BCEWithLogitsLoss()(
        torch_mlp(torch.from_numpy(x)), torch.from_numpy(y).double().unsqueeze(1)
    )
    t_loss.backward()

    assert t_loss.item() == pytest.approx(np_loss, rel=1e-12)
    for g_np, g_t in zip(np_mlp.gradients(), _torch_grads(torch_mlp), strict=True):
        np.testing.assert_allclose(g_t, g_np, rtol=RTOL_GRAD, atol=1e-12)


def test_softmax_cross_entropy_gradients_match_autograd() -> None:
    """Multi-class: dL/dlogits = (softmax(z) − onehot(y)) / B, then our MLP.backward."""
    k = 4
    np_mlp, torch_mlp = _pair(k)
    rng = set_seed(3)
    x = rng.normal(size=(40, SIZES[0]))  # (40, 29)
    y = rng.integers(0, k, 40)  # (40,) class indices

    z = np_mlp.forward(x)  # (40, K) logits
    z_shift = z - z.max(axis=1, keepdims=True)  # subtract the row max: exp cannot overflow
    p = np.exp(z_shift) / np.exp(z_shift).sum(axis=1, keepdims=True)  # (40, K) softmax
    np_loss = -np.log(p[np.arange(40), y]).mean()
    onehot = np.eye(k)[y]  # (40, K)
    np_mlp.backward((p - onehot) / 40)  # the whole derivation of docs/notes/softmax.md

    t_logits = torch_mlp(torch.from_numpy(x))
    t_logits.retain_grad()  # keep dL/dlogits too (autograd frees non-leaf grads by default)
    t_loss = torch.nn.CrossEntropyLoss()(t_logits, torch.from_numpy(y))
    t_loss.backward()

    assert t_loss.item() == pytest.approx(np_loss, rel=1e-12)
    np.testing.assert_allclose(t_logits.grad.numpy(), (p - onehot) / 40, rtol=RTOL_GRAD)
    for g_np, g_t in zip(np_mlp.gradients(), _torch_grads(torch_mlp), strict=True):
        np.testing.assert_allclose(g_t, g_np, rtol=RTOL_GRAD, atol=1e-12)


def test_sigmoid_matches_softmax_with_two_logits() -> None:
    """Binary is the K = 2 case: softmax([0, z])[1] = sigmoid(z)."""
    z = np.linspace(-5, 5, 11)
    two = np.stack([np.zeros_like(z), z], axis=1)  # (11, 2)
    p = np.exp(two) / np.exp(two).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p[:, 1], sigmoid(z), rtol=1e-12)


# --- The optimizer -------------------------------------------------------------------------


def test_sgd_momentum_steps_match_torch_sgd() -> None:
    """5 steps on the same batches: our SGD(0.9) and torch.optim.SGD(0.9) end with equal weights."""
    np_mlp, torch_mlp = _pair(1)
    lr, momentum, steps = 0.05, 0.9, 5
    np_opt = SGD(np_mlp.parameters(), lr=lr, momentum=momentum)
    t_opt = torch.optim.SGD(torch_mlp.parameters(), lr=lr, momentum=momentum)
    loss_np, loss_t = BCEWithLogits(), torch.nn.BCEWithLogitsLoss()
    rng = set_seed(4)
    batches = [(rng.normal(size=(32, SIZES[0])), rng.integers(0, 2, 32)) for _ in range(steps)]

    for x, y in batches:  # the same batches, in the same order, for both
        loss_np.forward(np_mlp.forward(x), y.astype(np.float64))
        np_mlp.backward(loss_np.backward())
        np_opt.step(np_mlp.gradients())

        t_opt.zero_grad()
        loss_t(torch_mlp(torch.from_numpy(x)), torch.from_numpy(y).double().unsqueeze(1)).backward()
        t_opt.step()

    np_params = np_mlp.parameters()
    for i, linear in enumerate(torch_mlp.linear_layers()):
        np.testing.assert_allclose(linear.weight.detach().numpy().T, np_params[2 * i], atol=1e-10)
        np.testing.assert_allclose(linear.bias.detach().numpy(), np_params[2 * i + 1], atol=1e-10)
    # The weights really moved, so the comparison above is not trivially true.
    assert not np.allclose(np_params[0], _pair(1)[0].parameters()[0])
