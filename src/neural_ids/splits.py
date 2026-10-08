"""Load the train/validation/test arrays written by `neural_ids.preprocess`.

Every model task reads its data through `load_split`, which refuses files that do not match
the current feature contract: a model trained on columns in a different order (or computed by
an older schema) would give wrong answers with no error message.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neural_ids.schema import FEATURE_NAMES, SCHEMA_VERSION, SchemaError

_REQUIRED_KEYS = (
    "X",
    "y_binary",
    "y_class",
    "class_names",
    "feature_names",
    "schema_version",
    "source",
    "flow_id",
    "input_sha256",
)


@dataclass(frozen=True)
class Split:
    """One split of the dataset. `flow_id` and `source` are metadata, never model inputs."""

    x: np.ndarray  # (n, n_features) float32
    y_binary: np.ndarray  # (n,) int64, 1 = attack
    y_class: np.ndarray  # (n,) int64, index into class_names
    class_names: tuple[str, ...]
    flow_id: np.ndarray  # (n,) str
    source: str  # "SYNTHETIC" or a user label
    input_sha256: str  # SHA-256 of the flows CSV that preprocess read


def load_split(path: Path) -> Split:
    """Load one `.npz` split and check it against the feature contract.

    Args:
        path: a file written by `uv run python -m neural_ids.preprocess`.

    Returns:
        A `Split` with `x` of shape (n, len(FEATURE_NAMES)).

    Raises:
        SchemaError: a key is missing, `schema_version` differs from `SCHEMA_VERSION`, or
            `feature_names` differs from `FEATURE_NAMES` (names or order).
        ValueError: the file holds Python objects (refused by `allow_pickle=False`).
    """
    # allow_pickle=False: an object array would need pickle, which can run arbitrary code.
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise SchemaError(f"{path}: missing arrays {missing}")
        version = str(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"{path}: schema_version {version!r} != code SCHEMA_VERSION {SCHEMA_VERSION!r}; "
                "rebuild the arrays with `neural_ids.preprocess`"
            )
        names = tuple(str(name) for name in data["feature_names"])
        if names != FEATURE_NAMES:
            raise SchemaError(
                f"{path}: feature_names differ from FEATURE_NAMES (names or order); "
                "rebuild the arrays with `neural_ids.preprocess`"
            )
        x = data["X"]  # (n, n_features)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise SchemaError(f"{path}: X has shape {x.shape}, expected (n, {len(FEATURE_NAMES)})")
        return Split(
            x=x,
            y_binary=data["y_binary"],
            y_class=data["y_class"],
            class_names=tuple(str(name) for name in data["class_names"]),
            flow_id=data["flow_id"],
            source=str(data["source"]),
            input_sha256=str(data["input_sha256"]),
        )
