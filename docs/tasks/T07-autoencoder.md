# T07 — Autoencoder from scratch: anomaly detection without labels

Phase: 1 (Build) · Depends on: T06

## Goal
Reuse your NumPy layers to build an autoencoder: a network trained to reproduce its own input
through a narrow middle layer. Train it on normal flows only, and flag flows that it
reconstructs badly as anomalies.

## Why it matters
- ML: the same layers and backpropagation, with a different target (the input itself) and a
  different loss (mean squared error). You see how general your building blocks are.
- Security: this is the mode that works with traffic collected by any person, because it needs
  only normal traffic and no attack labels. It is the classic trade-off of anomaly-based
  detection: it can catch new attacks, but it raises more false alarms than a supervised model,
  and "unusual" is not always "malicious".

## Deliverables
- `src/neural_ids/nn/losses.py`: `MSELoss` with forward and backward.
- `src/neural_ids/nn/autoencoder.py`: `Autoencoder(layer_sizes=[d, 16, 4, 16, d], rng)` built
  from your `Dense` and `ReLU` layers, with a linear output layer.
- `src/neural_ids/train_autoencoder_numpy.py` (`uv run python -m
  neural_ids.train_autoencoder_numpy`): trains on normal flows of the training split only;
  early stopping on the reconstruction loss of normal validation flows; saves
  `models/ae_numpy.npz` and the loss curve.
- `src/neural_ids/anomaly.py`: `reconstruction_error(model, X) -> np.ndarray` (one value per
  flow) and `threshold_from_normal(errors, quantile=0.99) -> float`.
- A results table (labeled synthetic): the false-positive rate on normal test flows and the
  detection rate for each attack class on test flows; a histogram of reconstruction errors for
  normal versus attack flows.
- `tests/test_autoencoder.py`: MSE gradient check (float64); the model can overfit a small
  batch; the threshold uses normal validation errors only.

## TODO(human)
- `MSELoss.forward` and `MSELoss.backward`.
- `reconstruction_error` and `threshold_from_normal`.

## Acceptance criteria
- The gradient check and the tests pass.
- No attack flow is used for training or for choosing the threshold (the `ml-reviewer`
  subagent confirms).
- The results table and the histogram come from a real run and are labeled synthetic.

## Out of scope
PyTorch (T09).

## Learning check
1. Why does the middle layer have to be smaller than the input?
2. Why choose the threshold from normal validation flows only?
3. On your own home network, which normal events could look "unusual" to this model?
