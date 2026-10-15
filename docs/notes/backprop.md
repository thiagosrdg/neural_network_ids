# Backpropagation with shapes

Code: `src/neural_ids/nn/layers.py`, `src/neural_ids/nn/model.py`. Tests: `tests/test_nn_gradients.py`.

## The idea
A network is a chain of functions: `L = loss(f3(f2(f1(x))))`. The chain rule says
`dL/dx = dL/df3 · df3/df2 · df2/df1 · df1/dx`. Backpropagation evaluates this product from the
left (the loss end): each layer receives `grad_out = dL/d(its output)` and returns
`dL/d(its input)`, which becomes the `grad_out` of the layer below. Each layer only needs what it
cached in `forward`.

Notation: B = batch size, `grad_out` always has the shape of the layer's output.

## Dense: out = x · W + b
| Quantity | Formula | Shape |
|---|---|---|
| input `x` | cached in forward | (B, in) |
| `W`, `b` | parameters | (in, out), (out,) |
| `grad_out` | dL/d(out) | (B, out) |
| `dW` | `xᵀ · grad_out` | (in, B)·(B, out) = (in, out) |
| `db` | sum of `grad_out` over the batch (axis 0) | (out,) |
| `grad_x` | `grad_out · Wᵀ` | (B, out)·(out, in) = (B, in) |

Why a sum over the batch: every example uses the same `W` and `b`, so their gradient is the sum
of each example's contribution. The matrix product `xᵀ · grad_out` performs that sum.

Why there is no `/B`: the loss is a mean, so `1/B` is already inside `grad_out` (it comes from
the loss gradient below). Dividing again would make the gradients B times too small.

A shape check that catches most mistakes: a gradient always has the shape of the thing it is the
gradient of (`dW` like `W`, `grad_x` like `x`). There is only one way to multiply the arrays you
have and get that shape.

## ReLU: out = max(x, 0)
`grad_x = grad_out · (x > 0)`, element by element, shape (B, n). The derivative is 1 where the
unit was active and 0 elsewhere. At exactly x = 0 ReLU has a kink (no derivative); we use 0.
Gradient checks must keep every ReLU input away from 0 (see the docstring of the test file).

## BCEWithLogits: L = mean BCE(sigmoid(z), y)
`dL/dz = (sigmoid(z) − y) / B`, shape (B, 1) like the logits (derivation of `p − y` in
`docs/notes/logreg.md`). This is where the `1/B` of the mean enters.

Shape trap: logits are (B, 1) and labels are (B,). In NumPy, `(B, 1) − (B,)` broadcasts to
(B, B) without an error. `BCEWithLogits.forward` checks the shapes and flattens the logits to
(B,) before any arithmetic.

## The full MLP backward for [d, 64, 32, 1]
```text
g = (sigmoid(z) − y) / B                    (B, 1)
g = Dense3.backward(g)   dW3 (32, 1)        (B, 32)
g = ReLU2.backward(g)                       (B, 32)
g = Dense2.backward(g)   dW2 (64, 32)       (B, 64)
g = ReLU1.backward(g)                       (B, 64)
g = Dense1.backward(g)   dW1 (d, 64)        (B, d)   = dL/dx, the input gradient
```
Cost: one backward pass is O(B · P) time, about two forward passes, where P is the number of
parameters (4,033 for d = 29). The numerical check costs 2·P forward passes, which is fine for
testing and far too slow for training.

Security link: the last line, dL/dx, says how to change each input feature to change the loss.
An evasion attack follows it (in the direction that lowers the "attack" score) to disguise a
malicious flow; that is a phase 2 topic.
