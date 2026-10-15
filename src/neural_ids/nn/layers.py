"""Layers with a forward and a backward pass: the building blocks of the MLP and the autoencoder.

Every layer follows one contract (the same idea as PyTorch's `autograd.Function`):
    forward(x)        computes the output and caches what backward needs
    backward(grad)    takes dL/d(output), stores parameter gradients, returns dL/d(input)
Chaining the backward calls from the loss to the input is backpropagation: the chain rule
applied one layer at a time. The equations with shapes are in `docs/notes/backprop.md`.

The 1/B of the mean loss enters once, in `BCEWithLogits.backward`; the layers never divide by
the batch size again.
"""

import numpy as np

from neural_ids.nn.functional import bce_loss, sigmoid


class Dense:
    """Fully connected layer: out = x · W + b.

    Attributes:
        W: (in_features, out_features) weights, He-initialized (times `init_scale`).
        b: (out_features,) biases, zeros.
        dW, db: gradients of the loss with respect to W and b, set by `backward`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init_scale: float = 1.0,
    ) -> None:
        # He initialization: W ~ N(0, 2 / fan_in). ReLU zeroes about half of its inputs, and
        # the factor 2 makes up for that, so activations keep a similar size layer after layer.
        # `init_scale` multiplies the standard deviation (the MLP's output layer uses 0.01).
        std = init_scale * np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, std, size=(in_features, out_features))  # (in, out) float64
        self.b = np.zeros(out_features)  # (out,)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None  # input cached by forward, (B, in)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Affine map of a batch. x: (B, in) -> (B, out). Caches x for `backward`.

        Time O(B · in · out), memory O(B · out) plus the cached input.
        """
        self._x = x  # (B, in): backward needs it for dW = xᵀ · grad_out
        return x @ self.W + self.b  # (B, in) @ (in, out) + (out,) -> (B, out)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backward pass. grad_out: (B, out) = dL/d(out).

        Sets self.dW (in, out) and self.db (out,) to new arrays.

        Returns:
            dL/dx with shape (B, in).
        """
        if self._x is None:
            raise RuntimeError("call forward before backward")
        self.dW = self._x.T @ grad_out  # (in, B) @ (B, out) -> (in, out); sums over the batch
        self.db = grad_out.sum(axis=0)  # (B, out) -> (out,)
        return grad_out @ self.W.T  # (B, out) @ (out, in) -> (B, in)

    def parameters(self) -> list[np.ndarray]:
        """[W, b]: the layer's own arrays, not copies."""
        return [self.W, self.b]

    def gradients(self) -> list[np.ndarray]:
        """[dW, db], in the same order as `parameters()`."""
        return [self.dW, self.db]


class ReLU:
    """Rectified linear unit: max(x, 0), element by element. No parameters."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None  # (B, n) bool: where the input was > 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, n) -> (B, n). Caches the mask x > 0 for `backward`."""
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out: (B, n) -> dL/dx (B, n).

        The derivative is 1 where x > 0 and 0 elsewhere (we take 0 at the kink x = 0).
        """
        if self._mask is None:
            raise RuntimeError("call forward before backward")
        return grad_out * self._mask  # gradient passes where the unit was active, else 0


class BCEWithLogits:
    """Mean binary cross-entropy on logits, with gradient (sigmoid(z) − y) / B.

    The MLP outputs logits of shape (B, 1) while labels have shape (B,). Subtracting them
    directly would broadcast to (B, B) without any error, so `forward` checks both shapes and
    works on a flat (B,) copy of the logits; `backward` returns the gradient in the logits'
    original shape.
    """

    def __init__(self) -> None:
        self._z: np.ndarray | None = None  # (B,) flattened logits
        self._y: np.ndarray | None = None  # (B,) labels
        self._shape: tuple[int, ...] = ()  # original logits shape, (B, 1) or (B,)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Mean loss. logits: (B, 1) or (B,); y: (B,) labels in {0, 1}. Returns a float.

        Raises:
            ValueError: if y is not 1-D, or logits is not (B, 1) or (B,) for the same B.
        """
        if y.ndim != 1:
            raise ValueError(f"y must have shape (B,), got {y.shape}")
        batch = y.shape[0]
        if logits.shape not in {(batch,), (batch, 1)}:
            raise ValueError(
                f"logits must have shape ({batch}, 1) or ({batch},), got {logits.shape}"
            )
        self._shape = logits.shape
        self._z = logits.reshape(batch)  # (B,): now the same shape as y
        self._y = y
        return bce_loss(self._z, y)

    def backward(self) -> np.ndarray:
        """dL/d(logits) = (sigmoid(z) − y) / B, in the shape the logits had in `forward`."""
        if self._z is None or self._y is None:
            raise RuntimeError("call forward before backward")
        grad = (sigmoid(self._z) - self._y) / self._z.shape[0]  # (B,); the only 1/B
        return grad.reshape(self._shape)
