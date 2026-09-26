# T04 — Baselines: majority class, perceptron, logistic regression

Phase: 1 (Build) · Depends on: T03

## Goal
Build the simplest models first, all in NumPy: a majority-class baseline, a perceptron, and
logistic regression (one neuron with a sigmoid) trained with gradient descent. Target:
`y_binary` (attack or normal).

## Why it matters
- ML: a neural network must beat simple baselines to justify its complexity. Logistic
  regression is exactly one neuron of the MLP you will build next.
- Security: a baseline shows what "free" performance looks like. If 80% of flows are normal,
  "everything is normal" scores 80% accuracy and catches zero attacks.

## Deliverables
- `src/neural_ids/metrics.py` (from scratch): accuracy, precision, recall, F1, and the
  confusion matrix, with "attack" as the positive class. Tests compare with `sklearn.metrics`.
- `src/neural_ids/gradcheck.py`: numerical gradient with central differences.
- `src/neural_ids/models/baselines.py`: `MajorityClassifier`.
- `src/neural_ids/models/perceptron.py`: step activation and the perceptron update rule,
  trained for a fixed number of epochs.
- `src/neural_ids/models/logreg.py`: a stable `sigmoid`, a stable binary cross-entropy
  (`bce_loss`), and `LogisticRegression` with mini-batch gradient descent, `predict_proba`,
  and `predict`.
- Command `uv run python -m neural_ids.train_baselines`: a validation metrics table for the
  three models (labeled synthetic) and the loss curve `reports/figures/logreg_loss.png`.
- `docs/notes/logreg.md`: the derivation of the gradient `dL/dw = Xᵀ(p − y) / n` in 5–10 lines
  (chain rule), and why a sigmoid with cross-entropy gives this simple form.

## TODO(human)
- `sigmoid` and `bce_loss` (both numerically stable).
- The gradient and update step inside `LogisticRegression.fit`.
- The perceptron update rule.

## Acceptance criteria
- Gradient check: relative error below 1e-6 in float64.
- With zero initial weights, the first loss is about ln 2 ≈ 0.693 (sanity check).
- The loss curve goes down and is saved.
- A validation table for the three models; logistic regression clearly beats the majority
  baseline on recall and F1.

## Out of scope
Hidden layers (T05).

## Learning check
1. Why does the gradient of cross-entropy with a sigmoid simplify to `(p − y)`?
2. Why does the perceptron never settle when the data is not linearly separable?
3. What is the Big-O cost of one training epoch in terms of `n` and `d`?
