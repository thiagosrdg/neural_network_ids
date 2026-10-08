"""Perceptron: one neuron with a step activation, trained with the perceptron rule.

    prediction = step(x · w + b),   step(z) = 1 if z > 0 else 0

The step has no useful slope (0 everywhere, undefined at 0), so gradient descent cannot train
it. The perceptron rule instead moves the weights toward each misclassified example:
    w ← w + lr · (y − ŷ) · x
(y − ŷ) is +1 for a missed attack, −1 for a false alarm, and 0 when correct.
Here it is applied to a mini-batch at once (the average of the per-example updates), so the
code has no Python loop over samples.
"""

from typing import Self

import numpy as np

from neural_ids.models.batching import minibatch_indices


def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step: (n,) float scores -> (n,) int64 in {0, 1}; step(0) = 0."""
    return (z > 0).astype(np.int64)


class Perceptron:
    """Binary perceptron trained for a fixed number of epochs (no early stopping)."""

    def __init__(
        self, n_features: int, lr: float = 0.1, epochs: int = 20, batch_size: int = 64
    ) -> None:
        self.w = np.zeros(n_features, dtype=np.float64)  # (d,)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Error rate (fraction misclassified) per epoch; index 0 is before training.
        self.history: dict[str, list[float]] = {"train_error": [], "val_error": []}

    def decision(self, x: np.ndarray) -> np.ndarray:
        """Scores x · w + b. x: (n, d) -> (n,)."""
        return x @ self.w + self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Labels in {0, 1}. x: (n, d) -> (n,) int64."""
        return step(self.decision(x))

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Apply the perceptron rule to one mini-batch.

        Args:
            x: (B, d) features of the batch.
            y: (B,) labels in {0, 1}.
        """
        err = y - self.predict(x)  # (B,) in {-1, 0, +1}
        self.w += self.lr * (x.T @ err) / x.shape[0]  # (d, B) @ (B,) -> (d,)
        self.b += self.lr * float(np.mean(err))

    def _record(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        self.history["train_error"].append(float(np.mean(self.predict(x) != y)))
        self.history["val_error"].append(float(np.mean(self.predict(x_val) != y_val)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        rng: np.random.Generator,
    ) -> Self:
        """Train on (x, y); the validation split is only measured, never trained on.

        Time O(epochs · n · d), memory O(n + d).
        """
        self._record(x, y, x_val, y_val)
        for _ in range(self.epochs):
            for idx in minibatch_indices(x.shape[0], self.batch_size, rng):
                self.update(x[idx], y[idx])  # x[idx]: (B, d)
            self._record(x, y, x_val, y_val)
        return self
