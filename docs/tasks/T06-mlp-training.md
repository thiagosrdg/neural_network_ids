# T06 — MLP from scratch, part 2: training loop and debugging

Phase: 1 (Build) · Depends on: T05

## Goal
Train the NumPy MLP with mini-batch SGD with momentum, use the standard debugging checks, and
compare it with logistic regression on the synthetic data.

## Why it matters
- ML: most neural-network bugs are training-loop bugs. Two cheap checks (the initial loss and
  overfitting one batch) catch most of them early.
- Security: a model earns its place in a detection pipeline only if it clearly beats a
  simpler, easier-to-explain baseline.

## Deliverables
- `src/neural_ids/nn/optim.py`: `SGD(params, lr, momentum)`.
- `src/neural_ids/train_numpy.py`, run with `uv run python -m neural_ids.train_numpy`:
  - a config dataclass (learning rate, batch size, epochs, hidden sizes, seed, patience);
  - an epoch loop that reshuffles with the seeded generator;
  - train loss, validation loss, and validation F1 per epoch;
  - early stopping on validation loss, with the best weights saved to `models/mlp_numpy.npz`;
  - loss curves saved to `reports/figures/mlp_numpy_loss.png`.
- Sanity checks (tests or a `--sanity` flag): the initial loss is about ln 2, and the model can
  overfit a single batch of 64 samples to a loss near zero.
- A validation comparison table (labeled synthetic): logistic regression versus MLP (accuracy,
  precision, recall, F1).
- Optional experiment: regenerate the data with `difficulty=0` and `difficulty=1` and compare
  the gap between the two models. Explain what you see.

## TODO(human)
- The momentum update in `SGD.step`.
- The body of one training epoch: forward → loss → backward → step.

## Acceptance criteria
- Both sanity checks pass.
- Training runs end to end on CPU; the curves and the best weights are saved.
- Running the command twice with the same seed gives the same numbers.
- The report includes the comparison table from a real run.

## Out of scope
The autoencoder (T07), PyTorch (T08).

## Learning check
1. What does momentum do? Use a physical analogy.
2. What does a learning rate that is 10 times too large look like on the loss curve?
3. Why reshuffle the data every epoch?
