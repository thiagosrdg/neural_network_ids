"""Stable math shared by every model: the sigmoid and binary cross-entropy on logits.

There is exactly one copy of these formulas in the project. `models/logreg.py` (T04) and
`nn/layers.py` (T05) both import them from here, so a fix in one place fixes every model.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function 1 / (1 + e^(−z)). Any shape -> same shape, in [0, 1].

    The naive formula computes e^(−z), which overflows for z < −709 in float64. Both branches
    below only compute e^(−|z|) ≤ 1, which never overflows.
    """
    e = np.exp(-np.abs(z))  # same shape as z, values in (0, 1]
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def bce_loss(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy computed from logits, without ever taking log(0).

    BCE = −[y·log(p) + (1 − y)·log(1 − p)] with p = sigmoid(z) simplifies to
        max(z, 0) − z·y + log(1 + e^(−|z|))
    (the same formula as PyTorch `BCEWithLogitsLoss`).

    Args:
        logits: (n,) raw scores z.
        y: (n,) labels in {0, 1}. Must have the same shape as `logits`.

    Returns:
        The mean loss over the n examples, a float ≥ 0.

    Raises:
        ValueError: if the shapes differ. Without this check, (n, 1) and (n,) would broadcast
            silently to (n, n) and return a wrong but plausible number.
    """
    if logits.shape != y.shape:
        raise ValueError(f"logits {logits.shape} and y {y.shape} must have the same shape")
    per_example = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(per_example))
