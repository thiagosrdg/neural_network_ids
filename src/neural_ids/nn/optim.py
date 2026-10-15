"""Optimizers: rules that turn gradients into parameter updates.

`SGD` is stochastic gradient descent with momentum, in the form PyTorch uses
(`torch.optim.SGD`, dampening 0, no Nesterov), so T08 can compare the two step by step:
    v ← momentum · v + g
    p ← p − lr · v
The velocity v is a running sum of past gradients in which each older gradient counts
`momentum` times less. Directions that stay the same from step to step add up and speed up;
directions that flip sign cancel out. With momentum 0 this is plain gradient descent.
"""

from collections.abc import Sequence

import numpy as np


class SGD:
    """Mini-batch SGD with momentum. Updates the given arrays in place.

    Pass the model's own arrays (`model.parameters()`), and at every step pass the gradients
    read again after `backward` (`model.gradients()`): `backward` stores new gradient arrays.
    """

    def __init__(self, params: Sequence[np.ndarray], lr: float, momentum: float = 0.0) -> None:
        """params: the arrays to update; lr > 0; momentum in [0, 1)."""
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p) for p in self.params]  # one per parameter, same shape

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """One update of every parameter. grads: same order and shapes as `params`.

        Time and extra memory O(P), P = total number of parameters (the velocity).
        """
        if len(grads) != len(self.params):
            raise ValueError(
                f"expected gradients for {len(self.params)} parameters, got {len(grads)}"
            )
        for p, g, v in zip(self.params, grads, self.velocity, strict=True):
            if g.shape != p.shape:
                raise ValueError(f"gradient shape {g.shape} != parameter shape {p.shape}")
            v *= self.momentum  # in place: keep a decayed share of the old velocity
            v += g  # v = momentum · v + g
            p -= self.lr * v  # in place: the model's own array changes
