"""Regression losses for the NumPy networks (T07).

`MSELoss` is the autoencoder's loss: how far the reconstruction is from the input. It follows
the same contract as `BCEWithLogits` in `nn/layers.py`: `forward` returns a float and caches what
`backward` needs, and `backward` returns dL/d(prediction). The 1/(B·d) of the mean enters once,
here, so the layers never divide by the batch size again.
"""

import numpy as np


class MSELoss:
    """Mean squared error over every element: L = mean((pred − target)²).

    Same definition as PyTorch `nn.MSELoss()` (reduction "mean"): the mean is over all B·d
    numbers, not a sum per flow averaged over the batch.
    """

    def __init__(self) -> None:
        self._diff: np.ndarray | None = None  # (B, d): pred − target, cached by forward

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Mean squared error. pred, target: same shape, for example (B, d). Returns a float.

        Time and memory O(B · d).

        Raises:
            ValueError: if the shapes differ. (B, d) against (B,) or (d,) would broadcast
                without any error and give a wrong but plausible number.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred {pred.shape} and target {target.shape} must have the same shape"
            )
        self._diff = pred - target  # (B, d)
        return float(np.mean(self._diff**2))

    def backward(self) -> np.ndarray:
        """dL/d(pred) = 2 · (pred − target) / (B · d), same shape as pred. Time O(B · d)."""
        if self._diff is None:
            raise RuntimeError("call forward before backward")
        return 2.0 * self._diff / self._diff.size  # .size = B · d; the only division by it
