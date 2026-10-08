"""Numerical gradient check: compare a hand-derived gradient with finite differences.

Central difference for one parameter w_i:
    dL/dw_i ≈ (L(w + eps·e_i) - L(w - eps·e_i)) / (2·eps)
The error is O(eps²), much smaller than the O(eps) of the one-sided version. Use float64:
in float32 the rounding error of L (about 1e-7 relative) divided by 2·eps swamps the result.

Cost: 2 loss evaluations per parameter, so O(P · cost(L)). Fine for checking small models,
far too slow for training (backpropagation gets all P derivatives for about one evaluation).
"""

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    loss_fn: Callable[[np.ndarray], float], w: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference gradient of `loss_fn` at `w`.

    Args:
        loss_fn: maps a parameter array of the same shape as `w` to a scalar loss.
        w: float64 parameters, any shape. Not modified.
        eps: step size.

    Returns:
        Array with the same shape as `w`: the estimated dL/dw.
    """
    if w.dtype != np.float64:
        raise TypeError(f"gradient checks need float64, got {w.dtype}")
    w = w.copy()
    grad = np.zeros_like(w)
    flat_w = w.reshape(-1)  # a view: changing flat_w changes w
    flat_g = grad.reshape(-1)
    for i in range(flat_w.size):  # loop over parameters, not samples
        old = flat_w[i]
        flat_w[i] = old + eps
        plus = loss_fn(w)
        flat_w[i] = old - eps
        minus = loss_fn(w)
        flat_w[i] = old
        flat_g[i] = (plus - minus) / (2 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max(|a| + |b|, tiny): scale-free, so it works for small and large gradients."""
    num = np.max(np.abs(a - b))
    den = max(float(np.max(np.abs(a) + np.abs(b))), 1e-12)
    return float(num / den)
