"""Majority-class baseline: always predict the most common training label."""

from typing import Self

import numpy as np


class MajorityClassifier:
    """Predicts the most frequent class of the training labels for every input.

    It ignores the features, so it shows the score a model gets "for free". With 79% normal
    flows it reaches about 79% accuracy and catches no attack (recall 0).
    """

    def __init__(self) -> None:
        self.majority: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Learn the most common label. x: (n, d) is unused; y: (n,) non-negative ints."""
        self.majority = int(np.argmax(np.bincount(y)))  # ties -> smallest label
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return (n,) int64 filled with the majority label."""
        if self.majority is None:
            raise RuntimeError("call fit before predict")
        return np.full(x.shape[0], self.majority, dtype=np.int64)
