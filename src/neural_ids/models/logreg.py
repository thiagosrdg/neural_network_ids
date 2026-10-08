"""Logistic regression: one neuron with a sigmoid, trained by mini-batch gradient descent.

    z = x · w + b            (the logit: a raw score in (-inf, inf))
    p = sigmoid(z)           (probability of "attack")
    L = mean BCE(z, y)       (binary cross-entropy)
    dL/dw = xᵀ(p − y) / n,   dL/db = mean(p − y)     (derivation: docs/notes/logreg.md)

The loss takes logits, not probabilities, like PyTorch's `BCEWithLogitsLoss`: computing
log(sigmoid(z)) in one stable formula never takes log(0), even for z = ±1000.
"""

from typing import Self

import numpy as np

from neural_ids.models.batching import minibatch_indices


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function 1 / (1 + e^(−z)). (n,) -> (n,) in [0, 1].

    The naive formula computes e^(−z), which overflows for z < −709 in float64. Both branches
    below only compute e^(−|z|) ≤ 1, which never overflows.
    """
    e = np.exp(-np.abs(z))  # (n,) in (0, 1]
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def bce_loss(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy computed from logits, without ever taking log(0).

    BCE = −[y·log(p) + (1 − y)·log(1 − p)] with p = sigmoid(z) simplifies to
        max(z, 0) − z·y + log(1 + e^(−|z|))
    (the same formula as PyTorch `BCEWithLogitsLoss`).

    Args:
        logits: (n,) raw scores z.
        y: (n,) labels in {0, 1}.

    Returns:
        The mean loss over the n examples, a float ≥ 0.
    """
    per_example = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(per_example))


class LogisticRegression:
    """Binary logistic regression. Weights start at zero, so the first loss is ln 2."""

    def __init__(
        self, n_features: int, lr: float = 0.1, epochs: int = 30, batch_size: int = 64
    ) -> None:
        self.w = np.zeros(n_features, dtype=np.float64)  # (d,)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Mean BCE per epoch on the full splits; index 0 is before training.
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def logits(self, x: np.ndarray) -> np.ndarray:
        """z = x · w + b. x: (n, d) -> (n,)."""
        return x @ self.w + self.b

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean BCE of the current weights on (x, y)."""
        return bce_loss(self.logits(x), y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of attack. x: (n, d) -> (n,) in [0, 1]."""
        return sigmoid(self.logits(x))

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Labels in {0, 1}: 1 where predict_proba >= threshold. x: (n, d) -> (n,) int64."""
        return (self.predict_proba(x) >= threshold).astype(np.int64)

    def gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Gradient of the mean BCE with respect to w and b.

        Args:
            x: (B, d) features of the batch.
            y: (B,) labels in {0, 1}.

        Returns:
            (dw, db): dw has shape (d,), db is a float.
        """
        err = self.predict_proba(x) - y  # (B,) = p − y
        dw = x.T @ err / x.shape[0]  # (d, B) @ (B,) -> (d,)
        db = float(np.mean(err))
        return dw, db

    def update(self, dw: np.ndarray, db: float) -> None:
        """One gradient-descent step: move against the gradient, scaled by the learning rate."""
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def _record(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        self.history["train_loss"].append(self.loss(x, y))
        self.history["val_loss"].append(self.loss(x_val, y_val))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        rng: np.random.Generator,
    ) -> Self:
        """Mini-batch gradient descent on (x, y); the validation split is only measured.

        Time O(epochs · n · d), memory O(n + d).
        """
        self._record(x, y, x_val, y_val)
        for _ in range(self.epochs):
            for idx in minibatch_indices(x.shape[0], self.batch_size, rng):
                dw, db = self.gradients(x[idx], y[idx])  # x[idx]: (B, d)
                self.update(dw, db)
            self._record(x, y, x_val, y_val)
        return self
