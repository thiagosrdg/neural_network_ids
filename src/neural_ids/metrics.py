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


# --- Multi-class metrics (T08) -------------------------------------------------------------
# Labels are class indices 0..K-1 (index into `class_names`). For class c, "positive" means
# "class c" and every other class counts as negative (one-vs-rest), so precision, recall, and
# F1 per class use the same formulas as above. Macro-F1 is the plain mean of the K F1 values:
# every class counts the same, so a model cannot hide a bad rare class behind the big normal
# class (accuracy can). Time O(n + K²), memory O(K²).


def multiclass_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(K, K) int64 counts: row = true class, column = predicted class (sklearn layout).

    Inputs: (n,) integer class indices in [0, n_classes).
    """
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"expected two (n,) arrays, got {y_true.shape} and {y_pred.shape}")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
            raise ValueError(f"{name} must hold class indices in [0, {n_classes})")
    # Each (true, pred) pair maps to one cell index true·K + pred; bincount counts them all
    # in one vectorized pass (no Python loop over rows).
    cells = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)  # (n,)
    return np.bincount(cells, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 (each (K,) float64) and support (K,) int64.

    A metric with a zero denominator is 0, like sklearn with `zero_division=0`.
    """
    cm = multiclass_confusion(y_true, y_pred, n_classes)  # (K, K)
    tp = np.diag(cm).astype(np.float64)  # (K,)
    predicted = cm.sum(axis=0)  # (K,) column sums: how often each class was predicted
    support = cm.sum(axis=1)  # (K,) row sums: how many rows have each true label

    def div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    return {
        "precision": div(tp, predicted.astype(np.float64)),
        "recall": div(tp, support.astype(np.float64)),
        "f1": div(2 * tp, (predicted + support).astype(np.float64)),  # 2TP / (2TP + FP + FN)
        "support": support,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of the per-class F1 scores."""
    return float(per_class_report(y_true, y_pred, n_classes)["f1"].mean())


def recall_ceiling_per_class(
    y_class: np.ndarray, flipped: np.ndarray, n_classes: int
) -> np.ndarray:
    """Recall per class of a perfect model, scored against the noisy labels. (K,) float64.

    The synthetic generator only flips a label to a class of the OTHER group (normal ↔ attack),
    so a flipped row's label is always wrong for a perfect model. Its best recall for class c
    is therefore the share of rows labeled c that were not flipped. NaN for a class with no
    rows. Evaluation only: `flipped` is the answer key and must never reach training.

    Args:
        y_class: (n,) noisy class indices as stored in the split.
        flipped: (n,) bool, True where the label was flipped.
    """
    labeled = np.bincount(y_class, minlength=n_classes).astype(np.float64)  # (K,)
    kept = np.bincount(y_class[~flipped.astype(bool)], minlength=n_classes)  # (K,)
    return np.divide(kept, labeled, out=np.full(n_classes, np.nan), where=labeled > 0)
