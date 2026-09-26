# T05 — MLP from scratch, part 1: layers and backpropagation

Phase: 1 (Build) · Depends on: T04

## Goal
Implement the building blocks of a multi-layer perceptron (MLP) — a dense layer, ReLU, and a
combined sigmoid + cross-entropy loss computed from logits — with forward and backward passes,
and prove that the gradients are correct. The autoencoder in T07 reuses these blocks.

## Why it matters
- ML: backpropagation is the chain rule applied layer by layer. After you write it once, no
  framework is magic.
- Security: attackers use these same gradients to craft inputs that fool models (a phase 2
  extension). You cannot defend what you do not understand.

## Deliverables
- `src/neural_ids/nn/layers.py`:
  - `Dense(in_features, out_features, rng)` with He initialization; `forward(x)` caches the
    input; `backward(grad_out)` stores `dW` and `db` and returns the gradient for the input.
  - `ReLU` with `forward` and `backward`.
  - `BCEWithLogits`: the loss computed from logits in a numerically stable way, with the
    gradient `sigmoid(z) − y`.
- `src/neural_ids/nn/model.py`: `MLP(layer_sizes=[d, 64, 32, 1], rng)` with `forward`,
  `backward`, and access to parameters and gradients.
- `tests/test_nn_gradients.py`: gradient checks (float64) for each layer and for the full MLP
  on a tiny batch, with relative error below 1e-5; shape tests.
- `docs/notes/backprop.md`: the backward equations with shapes — `dW = xᵀ · grad_out`,
  `db = sum of grad_out over the batch`, `grad_x = grad_out · Wᵀ`, and for ReLU
  `grad · (x > 0)`.

## TODO(human)
- `Dense.forward` and `Dense.backward`.
- `ReLU.backward`.

## Acceptance criteria
- All gradient-check and shape tests pass.
- No Python loops over samples.

## Out of scope
The training loop (T06).

## Learning check
1. For a batch of 32 and a layer 29 → 64 (29 features in contract v1), what are the shapes of
   `x`, `W`, `grad_out`, and `dW`?
2. Why does He initialization scale the weights by `sqrt(2 / fan_in)` for ReLU networks?
3. What is the Big-O cost of one forward pass through the whole MLP?
