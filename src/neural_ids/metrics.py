"""Binary classification metrics, from scratch. "Attack" (label 1) is the positive class.

Confusion matrix layout (the same as `sklearn.metrics.confusion_matrix` with labels [0, 1]):

                predicted normal   predicted attack
    normal            TN                 FP          (FP = false alarm)
    attack            FN                 TP          (FN = missed attack)

- precision = TP / (TP + FP): of the flows we flagged, how many were attacks.
- recall    = TP / (TP + FN): of the attacks, how many we flagged (detection rate).
- F1        = harmonic mean of precision and recall.
When a denominator is 0 (for example, a model that never predicts "attack" has TP + FP = 0),
the metric is 0, like scikit-learn with `zero_division=0`.

Every function is O(n) time and O(1) extra memory beyond its boolean masks.
"""

import numpy as np


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Both inputs must be (n,) arrays of 0/1 (not class indices or probabilities)."""
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"expected two (n,) arrays, got {y_true.shape} and {y_pred.shape}")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"{name} must hold only 0 and 1 (binary labels)")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] as a (2, 2) int64 array. Inputs: (n,) arrays of 0/1."""
    _check(y_true, y_pred)
    t = y_true.astype(bool)
    p = y_pred.astype(bool)
    tn = np.sum(~t & ~p)
    fp = np.sum(~t & p)
    fn = np.sum(t & ~p)
    tp = np.sum(t & p)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions equal to the label."""
    _check(y_true, y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP); 0 when no flow is predicted as attack."""
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN); 0 when there are no attacks."""
    _, (fn, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(tp, tp + fn)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """2·TP / (2·TP + FP + FN), the harmonic mean of precision and recall; 0 when undefined."""
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """All four scalar metrics in one dict (keys: accuracy, precision, recall, f1)."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
    }


def label_noise_ceiling(y_noisy: np.ndarray, flipped: np.ndarray) -> dict[str, float]:
    """Attack recall and precision of a perfect model, scored against noisy labels.

    A perfect model predicts the true class. Where a label was flipped, the true class is the
    opposite of the label, so the perfect prediction is `y_noisy XOR flipped`. Scoring it
    against `y_noisy` shows the best numbers these labels allow.

    Use this for evaluation only: `flipped` is the answer key and must never reach training.

    Args:
        y_noisy: (n,) 0/1 labels as stored in the split (with label noise).
        flipped: (n,) bool, True where the binary label was flipped.

    Returns:
        {"recall": ..., "precision": ...}
    """
    _check(y_noisy, flipped)
    perfect = (y_noisy.astype(bool) ^ flipped.astype(bool)).astype(np.int64)  # (n,)
    return {"recall": recall(y_noisy, perfect), "precision": precision(y_noisy, perfect)}
