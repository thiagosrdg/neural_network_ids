# T04 — Baselines: majority class, perceptron, logistic regression · Report

- Date: 2026-10-08
- Status: done
- Learning mode: guided (chosen by the user for this task)
- Commit: not committed yet

## Summary
Three NumPy models (majority, perceptron, logistic regression) are trained on train and compared on val with
from-scratch metrics, confusion matrices, and the label-noise ceiling. The test split is not loaded.

## What was built
- `src/neural_ids/splits.py` — `load_split`: `allow_pickle=False`; `SchemaError` on another `schema_version` or `feature_names`
- `src/neural_ids/metrics.py` — confusion matrix, accuracy, precision, recall, F1 (`zero_division=0`), `label_noise_ceiling`
- `src/neural_ids/gradcheck.py` — central-difference gradient (float64 only) and relative error
- `src/neural_ids/models/` — `batching.py`, `baselines.py`, `perceptron.py`, `logreg.py`; `train_baselines.py` — the command
- Hash link: `synthetic.json` gets `csv_sha256`, every `.npz` gets `input_sha256` (`utils.file_sha256`); ceiling only on a match
- 7 new test files, 48 new tests; `docs/notes/logreg.md` (derivation); `docs/ARCHITECTURE.md`. Dependencies added: none

## My part (TODO(human))
- `sigmoid`, `bce_loss`, `gradients`, `update`, `Perceptron.update` — written by Claude (guided mode, explained in chat)

## Verification (real output)
- Data used: synthetic (`data/processed/{train,val}.npz` from 20000 SYNTHETIC flows); test split not used
- Tests: 272 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`, `59 files already formatted`; data rebuilt (synthetic → preprocess), same CSV bytes
- Main command: `uv run python -m neural_ids.train_baselines` (seed 42, 30 epochs, lr 0.1, batch 64). Validation, SYNTHETIC:

  | model | accuracy | precision | recall | F1 | TN | FP | FN | TP |
  |---|---|---|---|---|---|---|---|---|
  | majority | 0.786 | 0.000 | 0.000 | 0.000 | 2358 | 0 | 643 | 0 |
  | perceptron | 0.905 | 0.720 | 0.913 | 0.805 | 2130 | 228 | 56 | 587 |
  | logistic regression | 0.917 | 0.790 | 0.835 | 0.812 | 2215 | 143 | 106 | 537 |
  | noise ceiling (perfect model) | - | 0.970 | 0.921 | - | | | | |

  69 flipped labels fall in val. Logreg loss: train 0.6931 → 0.2469, val 0.6931 → 0.2489.
- Acceptance criteria: all pass (gradient check < 1e-6 in float64; first loss ln 2; `reports/figures/logreg_loss.png` goes down; logreg beats majority on recall and F1).
- `ml-reviewer`: no must-fix. Fixed: train/val `source` match, 0/1 input check, stale `synthetic.json` (hash link).

## How to read the noise ceiling
2.5% of the labels were flipped at random, so even a perfect model scores recall 0.921 and precision 0.970
against the val labels. On held-out data, a model can pass the recall ceiling only by flagging more flows,
which lowers its precision (the perceptron: recall 0.913, precision 0.720). No feature can predict a random
flip, so a model that clearly beats both numbers on held-out data is a strong sign of leakage.

## Key concepts
- BCE on logits, `max(z,0) − z·y + log1p(e^−|z|)`, never takes log(0) or overflows `exp`.
- Sigmoid + BCE gives dL/dz = p − y, so dL/dw = Xᵀ(p − y)/n (`docs/notes/logreg.md`).
- The perceptron rule never settles when no line separates the data: see `reports/figures/perceptron_errors.png`.

## Decisions and deviations from the spec
- The perceptron rule is applied per mini-batch (the mean of per-sample updates): no Python loop over samples.
- The perceptron curve shows the error rate (no smooth loss) to keep the "every run saves a curve" rule.

## Problems and open questions
- Closed: synthetic flow IDs repeat across runs, so the ceiling now needs `csv_sha256 == input_sha256` (tested).
- Later: perceptron with the best weights kept ("pocket"), threshold tuning on val, class weights.

## Suggested next step
- T05 — MLP from scratch, part 1: layers and backpropagation.
