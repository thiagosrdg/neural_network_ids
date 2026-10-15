# Softmax with cross-entropy: why the gradient is p − onehot(y)

Code: `torch.nn.CrossEntropyLoss` in `src/neural_ids/train_torch.py`. Test that checks this note
against autograd: `tests/test_torch_crosscheck.py::test_softmax_cross_entropy_gradients_match_autograd`.

## From two classes to K
The binary model has one logit z and `p(attack) = sigmoid(z)`. With K classes the model outputs
K logits `z = (z_1, ..., z_K)`, one score per class, and **softmax** turns them into K
probabilities that are positive and sum to 1:

    p_k = exp(z_k) / Σ_j exp(z_j)

**Cross-entropy** for one row with true class y is `L = −log p_y`: the loss is small when the
model gives the true class a high probability. The batch loss is the mean over the B rows.

## The derivative, for one row
Write `S = Σ_j exp(z_j)`, so `log p_y = z_y − log S` and `L = −z_y + log S`.

- `∂(−z_y)/∂z_k` is −1 when k = y and 0 otherwise. That is `−onehot(y)_k`.
- `∂(log S)/∂z_k = (1/S) · exp(z_k) = p_k`.

Together:

    ∂L/∂z_k = p_k − onehot(y)_k        →  as a vector: dL/dz = p − onehot(y)

For the mean over a batch, every row gets a factor 1/B: `dL/dZ = (P − Y) / B`, shape (B, K),
where P is the (B, K) matrix of probabilities and Y the one-hot labels. This is the only 1/B,
just as in `BCEWithLogits.backward` (see `backprop.md`).

## A small example
Logits (2, 0, −1), true class 0: `p ≈ (0.844, 0.114, 0.042)`, gradient `≈ (−0.156, 0.114, 0.042)`.
Gradient descent subtracts the gradient: the true class's logit goes up, the others go down,
and each wrong class goes down in proportion to the probability it wrongly received.

## Why combine softmax and cross-entropy in one function
- The derivative of softmax on its own is a K×K matrix (Jacobian). Multiplied by the
  cross-entropy gradient, almost everything cancels and leaves `p − onehot(y)`. Computing the
  two together skips that matrix: O(K) per row instead of O(K²).
- Numerical safety: `exp(1000)` overflows. Since `softmax(z) = softmax(z − max(z))`, the code
  subtracts the row maximum first, and `log p_y = z_y − logsumexp(z)` never takes `log(0)`.
  This is why `CrossEntropyLoss` takes raw logits and the model has no softmax layer.

## Binary is the K = 2 case
With two logits (0, z): `p_1 = exp(z) / (1 + exp(z)) = sigmoid(z)`, and `p − onehot(y)` becomes
`sigmoid(z) − y` for the attack logit: the same formula as `BCEWithLogits` in T05.

## Class weights
`CrossEntropyLoss(weight=w)` multiplies row i's loss by `w[y_i]` and divides by `Σ_i w[y_i]`
instead of B. The gradient for row i becomes `w[y_i] · (p_i − onehot(y_i)) / Σ w`: rows of rare
classes push harder. With `w_c = n / (K · count_c)` (fitted on the training split), every class
contributes the same total weight.
