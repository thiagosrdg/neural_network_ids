"""Tests for `neural_ids.metrics`: our metrics must match `sklearn.metrics` exactly.

Synthetic labels only: these tests check the metric code, not detection.
"""

import warnings

import numpy as np
import pytest
from sklearn import metrics as skm

from neural_ids.metrics import (
    accuracy,
    binary_report,
    confusion_matrix,
    f1,
    label_noise_ceiling,
    precision,
    recall,
)
from neural_ids.synthetic import perfect_model_scores
from neural_ids.utils import set_seed


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_match_sklearn_on_random_labels(seed: int) -> None:
    rng = set_seed(seed)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    )
    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))
    assert precision(y_true, y_pred) == pytest.approx(skm.precision_score(y_true, y_pred))
    assert recall(y_true, y_pred) == pytest.approx(skm.recall_score(y_true, y_pred))
    assert f1(y_true, y_pred) == pytest.approx(skm.f1_score(y_true, y_pred))


def test_no_predicted_attacks_gives_zero_precision_and_f1_like_sklearn() -> None:
    """The majority baseline predicts no attacks: TP + FP = 0, so precision is 0/0."""
    y_true = np.array([0, 0, 1, 0, 1, 0])
    y_pred = np.zeros_like(y_true)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # our code must not warn or produce NaN
        report = binary_report(y_true, y_pred)
    assert report["precision"] == skm.precision_score(y_true, y_pred, zero_division=0) == 0.0
    assert report["f1"] == skm.f1_score(y_true, y_pred, zero_division=0) == 0.0
    assert report["recall"] == skm.recall_score(y_true, y_pred, zero_division=0) == 0.0
    assert report["accuracy"] == pytest.approx(4 / 6)


def test_no_true_attacks_gives_zero_recall_like_sklearn() -> None:
    y_true = np.zeros(5, dtype=np.int64)
    y_pred = np.array([0, 1, 0, 0, 0])
    assert recall(y_true, y_pred) == skm.recall_score(y_true, y_pred, zero_division=0) == 0.0
    assert f1(y_true, y_pred) == skm.f1_score(y_true, y_pred, zero_division=0) == 0.0


def test_confusion_matrix_layout_by_hand() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), [[1, 2], [1, 1]])


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        accuracy(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("bad", [np.array([0, 2, 1]), np.array([0.0, 0.7, 1.0])])
def test_non_binary_inputs_raise(bad: np.ndarray) -> None:
    """Class indices (y_class) or probabilities passed by mistake must fail loudly."""
    with pytest.raises(ValueError, match="0 and 1"):
        confusion_matrix(np.array([0, 1, 1]), bad)


def test_label_noise_ceiling_by_hand() -> None:
    # 4 labeled attacks; one of them is really normal (flipped), one labeled normal is a
    # real attack (flipped). Perfect model: predicts the true class.
    y_noisy = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    flipped = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    ceiling = label_noise_ceiling(y_noisy, flipped)
    # perfect predictions: [1,1,1,0,1,0,0,0] -> TP 3, FN 1, FP 1
    assert ceiling == {"recall": 0.75, "precision": 0.75}


def test_label_noise_ceiling_matches_count_formula() -> None:
    """Cross-check with the independent count formula in `neural_ids.synthetic`."""
    rng = set_seed(3)
    y_noisy = rng.integers(0, 2, size=1000)
    flipped = rng.random(1000) < 0.05
    true_attack = y_noisy.astype(bool) ^ flipped
    expected = perfect_model_scores(
        n_true_attack=int(true_attack.sum()),
        n_attack_to_normal=int((true_attack & flipped).sum()),
        n_normal_to_attack=int((~true_attack & flipped).sum()),
    )
    ceiling = label_noise_ceiling(y_noisy, flipped)
    assert ceiling["recall"] == pytest.approx(expected["attack_recall"])
    assert ceiling["precision"] == pytest.approx(expected["attack_precision"])


def test_no_flips_gives_perfect_ceiling() -> None:
    y = np.array([0, 1, 1, 0])
    assert label_noise_ceiling(y, np.zeros(4, dtype=bool)) == {"recall": 1.0, "precision": 1.0}
