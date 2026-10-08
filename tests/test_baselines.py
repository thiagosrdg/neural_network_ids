"""Tests for `neural_ids.models.baselines.MajorityClassifier`."""

import numpy as np
import pytest

from neural_ids.metrics import recall
from neural_ids.models.baselines import MajorityClassifier


def test_predicts_training_majority_for_any_input() -> None:
    x = np.zeros((5, 3))
    y = np.array([0, 0, 0, 1, 1])
    model = MajorityClassifier().fit(x, y)
    pred = model.predict(np.ones((4, 3)))
    np.testing.assert_array_equal(pred, [0, 0, 0, 0])
    assert pred.dtype == np.int64
    assert recall(np.array([1, 0, 1, 0]), pred) == 0.0  # catches no attack


def test_majority_attack_when_attacks_dominate() -> None:
    model = MajorityClassifier().fit(np.zeros((3, 1)), np.array([1, 1, 0]))
    np.testing.assert_array_equal(model.predict(np.zeros((2, 1))), [1, 1])


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        MajorityClassifier().predict(np.zeros((1, 1)))
