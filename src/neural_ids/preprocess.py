"""Preprocessing: from a flows table to the float32 arrays the models read, without leakage.

Steps for one table (the same for training, validation, test, and any future capture):
1. `validate_flows`: the table must follow the feature contract (else `SchemaError`).
2. Keep only `schema.FEATURE_NAMES`, in contract order. Metadata (addresses, ports,
   timestamps, flow IDs) and labels never reach the model inputs.
3. `log1p` on heavy-tailed features (`LOG_FEATURES`).
4. Standardize every feature that is not one-hot: z = (x - mean) / scale.

Data leakage means fitting a statistic on rows the model is later evaluated on. Every learned
value here (means, scales, the class names) comes from the training split only, and is saved
to JSON so the same numbers are applied to every later table.

Command:
    uv run python -m neural_ids.preprocess --input data/processed/synthetic.csv
writes `data/processed/{train,val,test}.npz` and `models/preprocessor.json`.
"""

import argparse
import json
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from neural_ids.schema import (
    FEATURE_NAMES,
    ONE_HOT_GROUPS,
    SCHEMA_VERSION,
    SchemaError,
    validate_flows,
)
from neural_ids.utils import set_seed

NORMAL_CLASS = "normal"  # index 0 of every class_names; all rows with is_attack == 0

# One-hot columns are already 0 or 1: no log, no scaling.
ONE_HOT_FEATURES: tuple[str, ...] = tuple(
    name for group in ONE_HOT_GROUPS.values() for name in group
)

# log1p(x) = log(1 + x): defined at 0 (all these features are >= 0 by contract) and close to x
# for small x. It turns "a few flows are 1000x bigger" into "a few flows are ~7 units bigger",
# so the largest flows do not control the mean, the scale, and later the gradients.
LOG_FEATURES: tuple[str, ...] = (
    # Duration: one-packet probes last 0 s, long sessions up to 1800 s (orders of magnitude).
    "duration_s",
    # Packet and byte counts: a probe sends 1 packet and 40 bytes; a download sends
    # 10^4 packets and 10^7 bytes. Classic heavy tail.
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    # Packet lengths: usually 40-1500 bytes, but a capture taken on the sending host with
    # segmentation offload (TSO/GSO) shows IP lengths up to ~64 KB, so the tail is long.
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_min",
    "pkt_len_max",
    # Gaps: microseconds in floods, up to 120 s in idle or slow (slowloris) flows.
    "iat_mean_s",
    "iat_std_s",
    "iat_max_s",
    # TCP flag counts: 0-1 in a probe, thousands in a long session or a flood.
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    # Rates: 0 for one-packet flows, above 10^7 bytes/s for floods and bulk transfers.
    "bytes_per_s",
    "packets_per_s",
    # Ratio: 0 (no reply), about 1 (balanced), 50+ (amplification, big download).
    "bwd_fwd_bytes_ratio",
)

_FORMAT = "neural_ids.preprocessor"  # marks our JSON files


# --- split ---------------------------------------------------------------------------------


def split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split into train, validation, and test tables, stratified by `label`.

    Stratified: each label keeps about the same share in every split. Per label with n_c rows,
    round(n_c * test_fraction) rows go to test, round(n_c * val_fraction) to validation, and
    the rest to train. Rows keep their original order inside each split.

    Phase 2 note: with real captures a random split leaks (flows from one session land in
    train and test). Replace this function with a split by capture or time window there.

    Args:
        df: (n_flows, n_columns) table with a `label` column.
        val_fraction, test_fraction: each >= 0, sum < 1.
        seed: passed to `set_seed`; the same seed gives the same split.

    Returns:
        (train, val, test), each with a fresh 0..k-1 index.

    Complexity: O(n) time (one permutation per label), O(n) memory for the index arrays.
    """
    if "label" not in df.columns:
        raise ValueError("split needs a 'label' column to stratify by")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            f"need val_fraction >= 0, test_fraction >= 0, and a sum < 1; "
            f"got {val_fraction} and {test_fraction}"
        )
    rng = set_seed(seed)
    labels = df["label"].to_numpy()  # labels: (n,)
    parts: tuple[list[np.ndarray], ...] = ([], [], [])  # train, val, test row positions
    for label in np.unique(labels):  # a loop over classes, not over samples
        rows = rng.permutation(np.flatnonzero(labels == label))  # rows: (n_c,)
        n_test = round(len(rows) * test_fraction)
        n_val = round(len(rows) * val_fraction)
        parts[2].append(rows[:n_test])
        parts[1].append(rows[n_test : n_test + n_val])
        parts[0].append(rows[n_test + n_val :])
    train, val, test = (
        df.iloc[np.sort(np.concatenate(part))].reset_index(drop=True) for part in parts
    )
    return train, val, test


# --- Standardizer --------------------------------------------------------------------------


class Standardizer:
    """Per-column standardization z = (x - mean) / scale, written from scratch.

    Like scikit-learn's `StandardScaler`: population std (ddof=0), and scale 1 for constant
    columns, so a value never seen in training becomes "its distance to the training value"
    instead of an enormous number.

    Attributes (after `fit`):
        mean_: (d,) float64 per-column mean of the fit data.
        scale_: (d,) float64 per-column std, or 1.0 for constant columns.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Self:
        """Store the per-column mean and scale of `x` (the training split only).

        A column is constant when its variance is within floating-point error of 0, by the
        rule scikit-learn uses (`_is_constant_feature`, after Chan, Golub, and LeVeque):
            var <= n * eps * var + (n * mean * eps) ** 2,   eps = float64 machine epsilon.
        A plain `std == 0` test misses constant nonzero columns: 14000 copies of log1p(1)
        give np.std about 1e-16, and dividing by that turns any new value into about 1e15.

        Args:
            x: (n, d) array, n >= 1. Converted to float64.

        Returns:
            self, so `Standardizer().fit(x)` can be chained.

        Complexity: O(n * d) time, O(n * d) memory for the float64 copy.
        """
        x = np.asarray(x, dtype=np.float64)  # x: (n, d)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"fit needs a 2-D array with at least one row; got shape {x.shape}")
        n = x.shape[0]
        mean = x.mean(axis=0)  # mean: (d,)
        var = x.var(axis=0)  # var: (d,), population variance (ddof=0)
        eps = np.finfo(np.float64).eps
        constant = var <= n * eps * var + (n * mean * eps) ** 2  # constant: (d,) bool
        self.mean_ = mean
        self.scale_ = np.where(constant, 1.0, np.sqrt(var))  # scale_: (d,)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return (x - mean_) / scale_, with the statistics stored by `fit`.

        Broadcasting: x has shape (n, d) and mean_ has shape (d,). NumPy lines up the last
        axes and repeats mean_ for each of the n rows, so no Python loop is needed.

        Args:
            x: (n, d) array with the same d as the fit data.

        Returns:
            (n, d) float64 array.

        Complexity: O(n * d) time and memory.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted: call fit first")
        x = np.asarray(x, dtype=np.float64)  # x: (n, d)
        if x.ndim != 2 or x.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"expected (n, {self.mean_.shape[0]}) with {self.mean_.shape[0]} columns; "
                f"got shape {x.shape}"
            )
        return (x - self.mean_) / self.scale_  # (n, d) - (d,) -> (n, d)

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-ready parameters. Python floats print exactly, so a round trip is lossless."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted: call fit first")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Self:
        """Rebuild a fitted Standardizer from `to_dict` output."""
        mean = np.asarray(data["mean"], dtype=np.float64)
        scale = np.asarray(data["scale"], dtype=np.float64)
        if mean.ndim != 1 or mean.shape != scale.shape:
            raise ValueError(
                f"mean and scale must be 1-D of equal length: {mean.shape}, {scale.shape}"
            )
        if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("mean must be finite and scale finite and > 0")
        s = cls()
        s.mean_, s.scale_ = mean, scale
        return s


# --- Preprocessor --------------------------------------------------------------------------


class Preprocessor:
    """Flows table -> (n, 29) float32 model inputs: select features, log1p, standardize.

    The fitted object is self-contained: the feature order, the log1p list, the standardized
    list, and their means and scales are stored on the object and in the JSON file. After
    `load`, `transform` uses only these stored values, never the module constants, so a later
    change to `LOG_FEATURES` cannot change how an old model's inputs are computed.
    """

    def __init__(self) -> None:
        # Read the module constants now (at creation), not at import time.
        self.schema_version = SCHEMA_VERSION
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        self.log1p_features: tuple[str, ...] = tuple(LOG_FEATURES)
        self.standardized_features: tuple[str, ...] = tuple(
            name for name in FEATURE_NAMES if name not in ONE_HOT_FEATURES
        )
        self.standardizer = Standardizer()

    def _columns(self, names: tuple[str, ...]) -> np.ndarray:
        """Positions of `names` inside `feature_names`: (len(names),) int."""
        return np.array([self.feature_names.index(name) for name in names], dtype=np.int64)

    def _logged_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Validate `df`, select the features in order, apply log1p: (n, d) float64."""
        validate_flows(df)  # also guarantees finite values >= 0, so log1p is safe
        x = df[list(self.feature_names)].to_numpy(dtype=np.float64)  # x: (n, 29)
        cols = self._columns(self.log1p_features)
        x[:, cols] = np.log1p(x[:, cols])
        return x

    def fit(self, train: pd.DataFrame) -> Self:
        """Fit the scaling on the TRAINING split only. O(n * d) time and memory."""
        x = self._logged_matrix(train)  # x: (n_train, 29)
        self.standardizer.fit(x[:, self._columns(self.standardized_features)])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Model inputs for any flows table, with the stored training statistics.

        Args:
            df: flows table that passes `validate_flows`; extra metadata and labels are
                ignored (they never reach the output).

        Returns:
            (n, 29) float32 array, columns in `feature_names` order.

        Complexity: O(n * d) time and memory.
        """
        if self.standardizer.mean_ is None:
            raise RuntimeError("Preprocessor is not fitted: call fit first")
        x = self._logged_matrix(df)  # x: (n, 29)
        cols = self._columns(self.standardized_features)
        x[:, cols] = self.standardizer.transform(x[:, cols])
        return x.astype(np.float32)

    def save(self, path: Path) -> None:
        """Write the fitted parameters to JSON (no pickle)."""
        params = self.standardizer.to_dict()  # raises if not fitted
        data = {
            "format": _FORMAT,
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "log1p_features": list(self.log1p_features),
            "standardized_features": list(self.standardized_features),
            **params,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a saved preprocessor; refuse another schema version or feature order."""
        data = json.loads(Path(path).read_text())
        if data.get("format") != _FORMAT:
            raise ValueError(f"{path} is not a neural_ids preprocessor file")
        if data["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(
                f"preprocessor was fitted for schema version {data['schema_version']!r}; "
                f"this code uses {SCHEMA_VERSION!r}. Refit it on data built for this version."
            )
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise SchemaError("preprocessor feature order differs from the contract")
        pre = cls()
        pre.schema_version = data["schema_version"]
        pre.feature_names = tuple(data["feature_names"])
        pre.log1p_features = tuple(data["log1p_features"])
        pre.standardized_features = tuple(data["standardized_features"])
        unknown = set(pre.log1p_features + pre.standardized_features) - set(pre.feature_names)
        if unknown:
            raise ValueError(f"unknown feature names in {path}: {sorted(unknown)}")
        pre.standardizer = Standardizer.from_dict(data)
        if len(pre.standardizer.mean_) != len(pre.standardized_features):
            raise ValueError("mean/scale length differs from the standardized feature list")
        return pre


# --- targets -------------------------------------------------------------------------------


def _is_attack(df: pd.DataFrame) -> np.ndarray:
    """The `is_attack` column as (n,) int64, checked to be 0 or 1."""
    if "is_attack" not in df.columns or "label" not in df.columns:
        raise ValueError("targets need the 'label' and 'is_attack' columns")
    is_attack = df["is_attack"].to_numpy()
    if not np.isin(is_attack, (0, 1)).all():
        raise ValueError("is_attack must be 0 or 1 in every row")
    return is_attack.astype(np.int64)


def class_names_from(train: pd.DataFrame) -> tuple[str, ...]:
    """Class names from the TRAINING split: ("normal", sorted attack labels...).

    Benign versus attack comes from `is_attack`, never from label names (docs/features.md):
    every benign row is class "normal", whatever its label (web, dns, ...).
    """
    is_attack = _is_attack(train)
    attacks = sorted(set(train["label"].to_numpy()[is_attack == 1].tolist()))
    if NORMAL_CLASS in attacks:
        raise ValueError(f"an attack row uses the reserved label {NORMAL_CLASS!r}")
    return (NORMAL_CLASS, *attacks)


def make_targets(df: pd.DataFrame, class_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Binary and multiclass targets.

    Returns:
        y_binary: (n,) int64, equal to `is_attack`.
        y_class: (n,) int64 index into `class_names`; 0 ("normal") where is_attack == 0,
            otherwise the index of the row's label. An attack label missing from
            `class_names` (not seen in training) raises ValueError.

    Complexity: O(n) time (hash lookup per row, vectorized by pandas).
    """
    if class_names[0] != NORMAL_CLASS:
        raise ValueError(f"class_names[0] must be {NORMAL_CLASS!r}")
    y_binary = _is_attack(df)  # (n,)
    index = pd.Index(class_names[1:]).get_indexer(df["label"]) + 1  # (n,); 0 = not found
    unknown = (y_binary == 1) & (index == 0)
    if unknown.any():
        missing = sorted(set(df["label"].to_numpy()[unknown].tolist()))
        raise ValueError(f"attack label(s) not seen in training: {missing}")
    y_class = np.where(y_binary == 1, index, 0).astype(np.int64)  # (n,)
    return y_binary, y_class


# --- exact copies --------------------------------------------------------------------------


def _feature_hashes(df: pd.DataFrame) -> np.ndarray:
    """One 64-bit hash per row of the raw feature values: (n,) uint64."""
    x = pd.DataFrame(df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64))  # same dtype everywhere
    return pd.util.hash_pandas_object(x, index=False).to_numpy()


def count_exact_copies(reference: pd.DataFrame, other: pd.DataFrame) -> int:
    """Rows of `other` whose feature vector also appears exactly in `reference`.

    Metadata is ignored. With independent synthetic flows a copy is not leakage (one-packet
    flows are often identical); with real captures a high share suggests near-duplicate
    flows from the same session in both splits.

    Complexity: O(n + m) hashing, O(m log n) for np.isin; O(n + m) memory.
    """
    return int(np.isin(_feature_hashes(other), _feature_hashes(reference)).sum())


# --- command -------------------------------------------------------------------------------

SYNTHETIC_ID_PREFIX = "synth-"  # every flow_id written by `neural_ids.synthetic`


def data_source(df: pd.DataFrame) -> str:
    """ "SYNTHETIC" when every flow comes from `neural_ids.synthetic`, else "unspecified".

    Uses the `flow_id` metadata only for this report label; it never reaches the model.
    """
    ids = df["flow_id"].astype(str)
    return "SYNTHETIC" if ids.str.startswith(SYNTHETIC_ID_PREFIX).all() else "unspecified"


def main(argv: list[str] | None = None) -> None:
    """Split a flows CSV, fit on train, write {train,val,test}.npz and the preprocessor JSON."""
    parser = argparse.ArgumentParser(description="Split and preprocess a flows table.")
    parser.add_argument("--input", type=Path, required=True, help="flows CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--preprocessor", type=Path, default=Path("models/preprocessor.json"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source", help="data source label stored in the arrays (default: detected)"
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    source = args.source or data_source(df)
    splits = dict(
        zip(
            ("train", "val", "test"),
            split(df, args.val_fraction, args.test_fraction, args.seed),
            strict=True,
        )
    )
    pre = Preprocessor().fit(splits["train"])  # training split only
    class_names = class_names_from(splits["train"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"input: {args.input} ({len(df)} flows, source: {source}), seed={args.seed}")
    print(f"classes: {list(class_names)}")
    for name, part in splits.items():
        x = pre.transform(part)  # x: (n, 29) float32
        y_binary, y_class = make_targets(part, class_names)
        np.savez(
            args.out_dir / f"{name}.npz",
            X=x,
            y_binary=y_binary,
            y_class=y_class,
            class_names=np.array(class_names, dtype=str),  # fixed-width Unicode, no pickle
            feature_names=np.array(FEATURE_NAMES, dtype=str),
            schema_version=np.array(SCHEMA_VERSION),  # 0-d Unicode: checked when loading
            source=np.array(source),
            # Metadata, never part of X: row i of X is flow flow_id[i]. Later tasks use it for
            # the label-noise ceiling (flipped_flow_ids) and to look up misclassified flows.
            flow_id=part["flow_id"].astype(str).to_numpy(dtype=str),  # (n,) fixed-width Unicode
        )
        counts = np.bincount(y_class, minlength=len(class_names)).tolist()
        print(
            f"  {name:<5} X {x.shape} y_binary {y_binary.shape} y_class {y_class.shape} "
            f"({len(part) / len(df):.1%}; per class {counts})"
        )
    for name in ("val", "test"):
        copies = count_exact_copies(splits["train"], splits[name])
        print(
            f"  {name} rows with exact copies of their features in train: "
            f"{copies}/{len(splits[name])} ({copies / len(splits[name]):.1%})"
        )
    pre.save(args.preprocessor)
    print(f"arrays -> {args.out_dir}/{{train,val,test}}.npz; preprocessor -> {args.preprocessor}")


if __name__ == "__main__":
    main()
