"""PyTorch versions of the networks. T08: the MLP classifier (T09 adds the autoencoder).

`MLPClassifier` is the same network as the NumPy `MLP` (T05–T06):
    x (B, 29) → Linear → ReLU [→ Dropout] → ... → Linear → logits (B, out)
- binary (n_classes = 2): one output, the attack logit; the sigmoid lives in BCEWithLogitsLoss.
- multi-class (n_classes = K > 2): K outputs, one logit per class; the softmax lives in
  CrossEntropyLoss.
It also starts the same way (He init, last layer × OUTPUT_INIT_SCALE, zero biases), so the first
loss is close to ln 2 (binary) or ln K (multi-class).

One layout difference matters when copying weights: our `Dense.W` is (in, out) and computes
x · W, while `nn.Linear.weight` is (out, in) and computes x · weightᵀ. So weight = Wᵀ.
"""

import torch
from torch import nn

from neural_ids.nn.model import MLP, OUTPUT_INIT_SCALE


class MLPClassifier(nn.Module):
    """Dense layers with ReLU (and optional dropout) between them; raw logits out.

    Attributes:
        layer_sizes: [n_features, *hidden, n_outputs], n_outputs = 1 for binary, else K.
        n_classes: 2 for binary, K for multi-class.
        net: the `nn.Sequential` holding the layers.
    """

    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...],
        n_classes: int,
        dropout: float = 0.0,
    ) -> None:
        """n_classes >= 2; dropout in [0, 1) (0 = no Dropout layers at all)."""
        super().__init__()  # required: registers this object as a Module (like Java's super())
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        n_out = 1 if n_classes == 2 else n_classes
        self.n_classes = n_classes
        self.dropout = dropout
        self.layer_sizes = [n_features, *hidden, n_out]
        layers: list[nn.Module] = []
        pairs = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True))
        for i, (n_in, n_next) in enumerate(pairs):
            layers.append(nn.Linear(n_in, n_next))
            if i < len(pairs) - 1:  # between layers, never after the logits
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))  # active only in model.train() mode
        # nn.Sequential registers every layer, so model.parameters() and state_dict() find them.
        self.net = nn.Sequential(*layers)
        self._init_like_numpy()

    def _init_like_numpy(self) -> None:
        """He-normal weights (std √(2/fan_in)), last layer × OUTPUT_INIT_SCALE, zero biases.

        PyTorch's own default for nn.Linear is a uniform init that would give a different first
        loss; we use the T06 init so both frameworks start the same way. It draws from the
        global torch generator, which `set_seed` seeds.
        """
        linears = self.linear_layers()
        with torch.no_grad():  # init is not part of training: autograd must not record it
            for i, linear in enumerate(linears):
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")  # fan_in mode
                if i == len(linears) - 1:
                    linear.weight.mul_(OUTPUT_INIT_SCALE)
                nn.init.zeros_(linear.bias)

    def linear_layers(self) -> list[nn.Linear]:
        """The nn.Linear layers in order (the counterpart of the NumPy Dense layers)."""
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) -> logits (B, 1) for binary or (B, K) for multi-class.

        Called through `model(x)`, which also runs PyTorch's hooks. Time O(B · P).
        """
        return self.net(x)


def copy_numpy_weights(src: MLP, dst: MLPClassifier) -> None:
    """Copy every weight and bias of a NumPy MLP into a PyTorch MLPClassifier, in place.

    Dense.W is (in, out); nn.Linear.weight is (out, in), so the weight is transposed. The
    values are copied (no shared memory) and converted to dst's dtype: call `dst.double()`
    first for an exact float64 cross-check.

    Raises:
        ValueError: the layer sizes differ, or a shape does not match after the transpose.
            `copy_` broadcasts silently, so the shapes are checked explicitly.

    Time and memory O(P), P = number of parameters.
    """
    if src.layer_sizes != dst.layer_sizes:
        raise ValueError(f"layer sizes differ: NumPy {src.layer_sizes}, torch {dst.layer_sizes}")
    params = src.parameters()  # [W1, b1, W2, b2, ...]
    with torch.no_grad():  # writing into parameters is not an operation to differentiate
        for i, linear in enumerate(dst.linear_layers()):
            w = torch.from_numpy(params[2 * i].T)  # (in, out) -> (out, in); a view, no copy yet
            b = torch.from_numpy(params[2 * i + 1])  # (out,)
            if w.shape != linear.weight.shape or b.shape != linear.bias.shape:
                raise ValueError(f"layer {i}: shape mismatch after transpose")
            linear.weight.copy_(w)  # copies the values into the parameter's own memory
            linear.bias.copy_(b)
