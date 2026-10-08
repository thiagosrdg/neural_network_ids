"""Tests for `neural_ids.splits.load_split`: contract checks when loading arrays."""

from pathlib import Path

import numpy as np
import pytest

from neural_ids.schema import FEATURE_NAMES, SCHEMA_VERSION, SchemaError
from neural_ids.splits import load_split


def _write(path: Path, **overrides: np.ndarray) -> Path:
    n, d = 4, len(FEATURE_NAMES)
    arrays = {
        "X": np.zeros((n, d), dtype=np.float32),
        "y_binary": np.array([0, 1, 0, 1]),
        "y_class": np.array([0, 1, 0, 2]),
        "class_names": np.array(["normal", "a", "b"]),
        "feature_names": np.array(FEATURE_NAMES, dtype=str),
        "schema_version": np.array(SCHEMA_VERSION),
        "source": np.array("SYNTHETIC"),
        "flow_id": np.array([f"synth-{i:07d}" for i in range(n)]),
        "input_sha256": np.array("ab" * 32),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


def test_loads_valid_file(tmp_path: Path) -> None:
    split = load_split(_write(tmp_path / "ok.npz"))
    assert split.x.shape == (4, len(FEATURE_NAMES))
    assert split.class_names == ("normal", "a", "b")
    assert split.source == "SYNTHETIC"
    assert split.flow_id.shape == (4,)
    assert split.input_sha256 == "ab" * 32


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "v.npz", schema_version=np.array("0.9"))
    with pytest.raises(SchemaError, match="schema_version"):
        load_split(path)


def test_reordered_feature_names_raise(tmp_path: Path) -> None:
    names = list(FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    path = _write(tmp_path / "f.npz", feature_names=np.array(names))
    with pytest.raises(SchemaError, match="feature_names"):
        load_split(path)


def test_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "m.npz"
    np.savez(path, X=np.zeros((1, len(FEATURE_NAMES))))
    with pytest.raises(SchemaError, match="missing"):
        load_split(path)


def test_object_array_is_refused(tmp_path: Path) -> None:
    """allow_pickle=False: an array of Python objects must not be loaded."""
    path = _write(tmp_path / "o.npz", flow_id=np.array([{"a": 1}] * 4, dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        load_split(path)
