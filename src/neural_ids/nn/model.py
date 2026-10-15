"""Multi-layer perceptron (MLP) built from `Dense` and `ReLU` layers.

For layer_sizes [29, 64, 32, 1]:
    x (B, 29) → Dense → ReLU → (B, 64) → Dense → ReLU → (B, 32) → Dense → logits (B, 1)
There is no activation after the last layer: the sigmoid lives inside `BCEWithLogits`.
"""

from collections.abc import Sequence

import numpy as np

from neural_ids.nn.layers import Dense, ReLU


class MLP:
    """Dense layers with ReLU between them; the last layer outputs raw logits."""

    def __init__(self, layer_sizes: Sequence[int], rng: np.random.Generator) -> None:
        """Build the layers. layer_sizes = [inputs, hidden..., outputs], at least two numbers."""
        if len(layer_sizes) < 2:
            raise ValueError(f"need at least input and output sizes, got {list(layer_sizes)}")
        self.layer_sizes = list(layer_sizes)
        self.layers: list[Dense | ReLU] = []
        pairs = list(zip(layer_sizes[:-1], layer_sizes[1:], strict=True))
        for i, (n_in, n_out) in enumerate(pairs):
            self.layers.append(Dense(n_in, n_out, rng))
            if i < len(pairs) - 1:  # ReLU between layers, never on the logits
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, layer_sizes[0]) -> logits (B, layer_sizes[-1]). Time O(B · P)."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out: dL/d(logits), same shape as the output of `forward`.

        Fills every Dense layer's dW and db and returns dL/dx, shape (B, layer_sizes[0]).
        Time O(B · P), the same order as `forward`.
        """
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def _dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    def parameters(self) -> list[np.ndarray]:
        """[W1, b1, W2, b2, ...]: the model's own arrays, not copies.

        Changing them in place (for example `p -= lr * g`) changes the model. Rebinding a
        name (`p = p - lr * g`) does not.
        """
        return [p for layer in self._dense_layers() for p in layer.parameters()]

    def gradients(self) -> list[np.ndarray]:
        """[dW1, db1, dW2, db2, ...], in the same order as `parameters()`.

        `backward` stores new gradient arrays, so call `gradients()` again after every
        `backward`; a list kept from an earlier call still holds the old gradients.
        """
        return [g for layer in self._dense_layers() for g in layer.gradients()]

    @property
    def n_parameters(self) -> int:
        """Total number of weights and biases."""
        return sum(p.size for p in self.parameters())
