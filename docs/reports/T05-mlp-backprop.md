# T05 — MLP from scratch, part 1: layers and backpropagation · Report

- Date: 2026-10-15
- Status: done
- Learning mode: guided (chosen by the user for this task)
- Commit: see `git log` (`feat: add NumPy layers, MLP, and gradient-checked backpropagation`)

## Summary
NumPy layers with forward and backward passes (`Dense`, `ReLU`, `BCEWithLogits`) and an `MLP` built
from them. Every backward pass, and the full MLP's parameter and input gradients, match finite
differences in float64, entry by entry. The stable sigmoid and BCE now have one shared implementation.

## What was built
- `src/neural_ids/nn/functional.py` — the only `sigmoid` and `bce_loss` (moved from `logreg.py`); `bce_loss` rejects mismatched shapes
- `src/neural_ids/nn/layers.py` — `Dense` (He init), `ReLU`, `BCEWithLogits` (flattens (B, 1) logits after a shape check)
- `src/neural_ids/nn/model.py` — `MLP(layer_sizes, rng)`: `forward`, `backward`, `parameters()`, `gradients()`, `n_parameters`
- `src/neural_ids/models/logreg.py` — imports and re-exports the shared functions
- `src/neural_ids/gradcheck.py` — per-element rule `|a − n| <= 1e-8 + 1e-5·|n|` (`gradient_mismatches`,
  `assert_gradients_close`), now used by every gradient test, T04's included
- `tests/test_nn_gradients.py` — 24 tests (+2 in `test_gradcheck.py`); `docs/notes/backprop.md` — equations with shapes; `docs/ARCHITECTURE.md` — new T05 section
- Dependencies added: none

## My part (TODO(human))
- `Dense.forward`, `Dense.backward`, `ReLU.backward` — written by Claude (guided mode, explained line by line in chat)

## Verification (real output)
- Data used: none (random float64 arrays from fixed seeds); no training in this task
- Tests: 298 passed — `uv run pytest -q` (24 in `tests/test_nn_gradients.py`)
- Lint and format: `All checks passed!`, `66 files already formatted`
- Gradient checks: every entry passes the per-element rule. Max relative error (limit 1e-5): Dense 8.95e-11 ·
  ReLU 8.64e-11 · BCEWithLogits 1.66e-10 · MLP [5,4,3,1] 1.80e-10 · MLP [29,64,32,1] 6.90e-10 ·
  deliberately wrong dW (×1.01) 4.98e-03, rejected
- Hidden-error test: one tiny dW entry (about 1e-6) with its sign flipped beside entries near 5 — old global
  `relative_error` 1.62e-07 (would pass), per-element rule flags exactly that entry
- `MLP([29, 64, 32, 1]).n_parameters == 4033`, as in `docs/ARCHITECTURE.md`
- Acceptance criteria: all pass (gradient and shape tests pass; the only loops run over layers and, in the check, over parameters)
- `ml-reviewer`: no must-fix. Applied: identity test for `gradients()`, gradient check of the MLP's input gradient dL/dx.

## Key concepts
- Backprop = chain rule per layer: `dW = xᵀ·g`, `db = Σ_batch g`, `grad_x = g·Wᵀ`, ReLU `g·(x > 0)`.
- The 1/B of the mean loss enters once, in `(sigmoid(z) − y)/B`; layers never divide again.
- A gradient has the shape of what it differentiates; that alone fixes the order of each matrix product.
- (B, 1) − (B,) broadcasts silently to (B, B); shapes are checked before any arithmetic.
- Finite differences fail at ReLU's kink: every pre-activation is kept > 1e-3 from 0 (step 1e-6).
- One global error number lets a small wrong entry hide behind large ones; judge each entry by its own size.

## Decisions and deviations from the spec
- `parameters()` returns the model's own arrays (T06's optimizer updates in place); `backward` assigns
  new `dW`/`db`, so `gradients()` must be read again after each `backward` (docstring and test).
- `BCEWithLogits` accepts logits (B, 1) or (B,) but labels only (B,); a (B, 1) label array raises.
- The [29,64,32,1] check uses seed 15: seed 10 put a pre-activation 0.99986e-3 from 0 and the kink guard
  stopped the test, as designed.

## Problems and open questions
- Closed in this task: `relative_error` alone could hide a small wrong entry (reviewer note on T04 code);
  pass or fail now comes from the per-element rule, and `relative_error` stays as the reported summary.

## Suggested next step
- T06 — MLP from scratch, part 2: training loop and debugging.
