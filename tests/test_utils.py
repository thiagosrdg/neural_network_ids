"""Tests for `neural_ids.utils`.

pytest finds every function whose name starts with `test_` and runs it. A test passes when
it finishes without an exception; `assert` raises an exception when its condition is false.
This is like JUnit in Java, but without classes or annotations.
"""

import random

import numpy as np

from neural_ids.utils import set_seed


def test_same_seed_gives_same_numbers() -> None:
    """Reproducibility: the same seed must give exactly the same numbers."""
    # Two separate Generators made from the same seed.
    a = set_seed(123).normal(size=(5, 3))  # a: (5, 3)
    b = set_seed(123).normal(size=(5, 3))  # b: (5, 3)

    # The result must be a real NumPy Generator, not the legacy global state.
    assert isinstance(set_seed(123), np.random.Generator)
    # Exact equality (not "close"): the same seed must repeat every bit.
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_numbers() -> None:
    """Different seeds must give different numbers (otherwise the seed is ignored)."""
    a = set_seed(1).normal(size=(5, 3))  # a: (5, 3)
    b = set_seed(2).normal(size=(5, 3))  # b: (5, 3)

    # `np.array_equal` returns True only if every element matches.
    assert not np.array_equal(a, b)


def test_python_random_is_seeded() -> None:
    """`set_seed` must also seed Python's built-in `random` module."""
    set_seed(7)
    first = [random.random() for _ in range(3)]  # three floats in [0, 1)
    set_seed(7)
    second = [random.random() for _ in range(3)]

    # After re-seeding with the same value, `random` must repeat the same sequence.
    assert first == second
