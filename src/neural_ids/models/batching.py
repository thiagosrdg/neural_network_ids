"""Mini-batch index generation shared by the NumPy models."""

import numpy as np


def minibatch_indices(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle 0..n-1 and cut it into batches of `batch_size` (the last one may be smaller).

    A new shuffle every epoch stops the model from seeing the rows in the same order, which
    would bias each epoch toward the last rows it saw. O(n) time and memory.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    order = rng.permutation(n)  # (n,)
    return [order[start : start + batch_size] for start in range(0, n, batch_size)]
