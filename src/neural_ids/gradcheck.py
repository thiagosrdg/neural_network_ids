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
    """max |a - b| / max(|a| + |b|, tiny): one summary number for a whole array.

    Good for reporting, too weak as the only test: the denominator is the largest magnitude in
    the array, so a wrong small entry next to large correct ones barely moves it. Use
    `assert_gradients_close` to decide pass or fail.
    """
    num = np.max(np.abs(a - b))
    den = max(float(np.max(np.abs(a) + np.abs(b))), 1e-12)
    return float(num / den)


def gradient_mismatches(
    analytic: np.ndarray, numeric: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> np.ndarray:
    """Boolean mask, True where |analytic - numeric| > atol + rtol·|numeric|.

    The same rule as `np.testing.assert_allclose(analytic, numeric, rtol, atol)`: each entry is
    judged against its own size, so a small entry cannot hide behind large ones. `atol` covers
    entries whose true value is about 0, where a relative test alone is impossible to pass.
    Time and memory O(P) for P entries.
    """
    if analytic.shape != numeric.shape:
        raise ValueError(f"shapes differ: {analytic.shape} vs {numeric.shape}")
    return np.abs(analytic - numeric) > atol + rtol * np.abs(numeric)


def assert_gradients_close(
    analytic: np.ndarray, numeric: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> float:
    """Fail if any entry breaks the per-element rule; otherwise return the max relative error.

    Raises:
        AssertionError: naming how many entries fail and the first few (index, analytic, numeric).

    Returns:
        `relative_error(analytic, numeric)`, kept as the summary number for reports.
    """
    bad = gradient_mismatches(analytic, numeric, rtol, atol)
    if bad.any():
        shown = [
            f"{tuple(int(i) for i in idx)}: {analytic[tuple(idx)]:.6e} vs {numeric[tuple(idx)]:.6e}"
            for idx in np.argwhere(bad)[:5]
        ]
        raise AssertionError(
            f"{int(bad.sum())} of {bad.size} gradient entries break "
            f"|a - n| <= {atol:g} + {rtol:g}·|n| (analytic vs numeric): " + "; ".join(shown)
        )
    return relative_error(analytic, numeric)
