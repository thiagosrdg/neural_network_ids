"""Autoencoder: a network trained to reproduce its own input through a narrow middle layer.

For layer_sizes [29, 16, 4, 16, 29]:
    x (B, 29) → Dense → ReLU → (B, 16) → Dense → ReLU → bottleneck (B, 4)
              → Dense → ReLU → (B, 16) → Dense → reconstruction (B, 29), linear
The encoder is everything up to the bottleneck, the decoder everything after it.

It is an `MLP` with a symmetric shape: the same `Dense` and `ReLU` layers, ReLU between layers
and none after the last one. The output is linear because the standardized features can be
negative, which a ReLU output could never produce. The last layer starts small (He × 0.01, see
`nn/model.py`), so the first reconstruction is close to 0, the mean of every standardized
feature, and the first loss is close to mean(x²).
"""

from collections.abc import Sequence

import numpy as np

from neural_ids.nn.layers import Dense, ReLU
from neural_ids.nn.model import MLP


class Autoencoder(MLP):
    """Symmetric MLP whose output size equals its input size; adds `encode`."""

    def __init__(self, layer_sizes: Sequence[int], rng: np.random.Generator) -> None:
        """layer_sizes: [d, ..., bottleneck, ..., d], an odd count of at least 3, symmetric.

        Raises:
            ValueError: the sizes are not symmetric or have no middle layer.
        """
        sizes = list(layer_sizes)
        if len(sizes) < 3 or len(sizes) % 2 == 0:
            raise ValueError(f"need an odd number (>= 3) of layer sizes, got {sizes}")
        if sizes != sizes[::-1]:
            raise ValueError(f"layer sizes must be symmetric (d, ..., d), got {sizes}")
        super().__init__(sizes, rng)

    @property
    def bottleneck_size(self) -> int:
        """Width of the middle layer (4 for [29, 16, 4, 16, 29])."""
        return self.layer_sizes[len(self.layer_sizes) // 2]

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Bottleneck outputs after their ReLU. x: (B, d) -> (B, bottleneck_size).

        Runs the encoder half only: the first len(layer_sizes) // 2 Dense+ReLU pairs.
        """
        n_dense = len(self.layer_sizes) // 2  # Dense layers before the bottleneck
        seen = 0
        for layer in self.layers:
            x = layer.forward(x)
            if isinstance(layer, Dense):
                seen += 1
            elif isinstance(layer, ReLU) and seen == n_dense:
                return x  # (B, bottleneck_size)
        raise AssertionError("unreachable: a symmetric autoencoder has a ReLU after the bottleneck")
