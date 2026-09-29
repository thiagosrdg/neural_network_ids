"""The feature contract in code: the only interface between data and models.

`docs/features.md` explains each feature in words; this module fixes the names, order, dtypes,
and allowed ranges, and `validate_flows` checks that a flows table follows them.

Rules (see CLAUDE.md):
- Change this contract only when a task says so, and then bump `SCHEMA_VERSION`. A model
  bundle stores the version it was trained with and refuses tables built for another one.
- Identifiers (IP addresses, exact ports, timestamps, flow IDs) are metadata, never features.
- Labels (`label`, `is_attack`) are never features either: a label used as an input is
  target leakage (the model would read the answer instead of learning it).
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

# Flow timeouts in seconds. They are part of the contract, not user options: a different
# timeout splits flows differently and so changes the feature values. Changing one needs a new
# SCHEMA_VERSION.
IDLE_TIMEOUT_S = 120.0  # a gap longer than this between two packets ends the flow
ACTIVE_TIMEOUT_S = 1800.0  # a packet more than this after the flow's first packet starts a new flow

# IP-layer packet length limits in bytes.
MIN_IP_LEN = 20  # smallest IPv4 header (IPv6 packets are at least 40)
MAX_IP_LEN = 65575  # IPv6: 65535 (largest payload length) + 40 (fixed header)

# Metadata: used for labeling and reports, never as model inputs.
METADATA_COLUMNS: tuple[str, ...] = (
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "start_time",
    "end_time",
)

# Allowed but not required (added by T02 and T11). Never model inputs.
OPTIONAL_COLUMNS: tuple[str, ...] = ("label", "is_attack", "capture_id")


@dataclass(frozen=True)
class FeatureSpec:
    """One model input: its column name, dtype, and allowed closed range [min, max].

    `dtype` "int64" means the column must have an integer dtype. "float64" accepts any integer
    or float dtype, because a CSV written by another tool or by hand can store whole numbers
    without a decimal point (for example `0`), and pandas then reads that column as int64.
    """

    name: str
    dtype: Literal["int64", "float64"]
    min: float
    max: float


_INF = math.inf

# The model inputs, in the fixed order the models see them. 29 features in v1.
FEATURES: tuple[FeatureSpec, ...] = (
    # Protocol, one-hot (exactly one is 1).
    FeatureSpec("proto_tcp", "int64", 0, 1),
    FeatureSpec("proto_udp", "int64", 0, 1),
    FeatureSpec("proto_icmp", "int64", 0, 1),
    FeatureSpec("proto_other", "int64", 0, 1),
    # Time.
    FeatureSpec("duration_s", "float64", 0.0, ACTIVE_TIMEOUT_S),
    # Volume per direction. The first packet defines "forward", so fwd_packets >= 1.
    FeatureSpec("fwd_packets", "int64", 1, _INF),
    FeatureSpec("bwd_packets", "int64", 0, _INF),
    FeatureSpec("fwd_bytes", "int64", MIN_IP_LEN, _INF),
    FeatureSpec("bwd_bytes", "int64", 0, _INF),
    # IP-layer packet lengths over both directions.
    FeatureSpec("pkt_len_mean", "float64", MIN_IP_LEN, MAX_IP_LEN),
    # Largest population std of values inside [a, b] is (b - a) / 2.
    FeatureSpec("pkt_len_std", "float64", 0.0, (MAX_IP_LEN - MIN_IP_LEN) / 2),
    FeatureSpec("pkt_len_min", "int64", MIN_IP_LEN, MAX_IP_LEN),
    FeatureSpec("pkt_len_max", "int64", MIN_IP_LEN, MAX_IP_LEN),
    # Inter-arrival times over both directions; each gap is <= IDLE_TIMEOUT_S.
    FeatureSpec("iat_mean_s", "float64", 0.0, IDLE_TIMEOUT_S),
    FeatureSpec("iat_std_s", "float64", 0.0, IDLE_TIMEOUT_S / 2),
    FeatureSpec("iat_max_s", "float64", 0.0, IDLE_TIMEOUT_S),
    # TCP flag counts over both directions (0 for non-TCP).
    FeatureSpec("syn_count", "int64", 0, _INF),
    FeatureSpec("ack_count", "int64", 0, _INF),
    FeatureSpec("fin_count", "int64", 0, _INF),
    FeatureSpec("rst_count", "int64", 0, _INF),
    FeatureSpec("psh_count", "int64", 0, _INF),
    FeatureSpec("urg_count", "int64", 0, _INF),
    # Rates (0 when duration_s is 0).
    FeatureSpec("bytes_per_s", "float64", 0.0, _INF),
    FeatureSpec("packets_per_s", "float64", 0.0, _INF),
    # Direction balance.
    FeatureSpec("bwd_fwd_bytes_ratio", "float64", 0.0, _INF),
    # Destination port class, one-hot (exactly one is 1).
    FeatureSpec("dst_port_class_well_known", "int64", 0, 1),
    FeatureSpec("dst_port_class_registered", "int64", 0, 1),
    FeatureSpec("dst_port_class_dynamic", "int64", 0, 1),
    FeatureSpec("dst_port_class_none", "int64", 0, 1),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURES)

# Each group must have exactly one 1 in every row.
ONE_HOT_GROUPS: dict[str, tuple[str, ...]] = {
    "proto": ("proto_tcp", "proto_udp", "proto_icmp", "proto_other"),
    "dst_port_class": (
        "dst_port_class_well_known",
        "dst_port_class_registered",
        "dst_port_class_dynamic",
        "dst_port_class_none",
    ),
}


class SchemaError(ValueError):
    """A flows table does not follow the feature contract."""


def validate_flows(df: pd.DataFrame) -> None:
    """Check that a flows table follows the feature contract; raise `SchemaError` if not.

    Input:
        df: (n_flows, n_columns) table. It must contain every name in `METADATA_COLUMNS` and
            `FEATURE_NAMES`; it may contain names in `OPTIONAL_COLUMNS`; nothing else.
            Column order does not matter. Metadata and optional columns are checked for
            presence only.

    Checks, each raising `SchemaError` with a message that names the column(s):
        0. Duplicate column names.
        1. Missing columns (required but absent).
        2. Extra columns (not metadata, feature, or optional).
        3. Dtype of every feature: spec "int64" needs an integer dtype; spec "float64"
           accepts integer or float dtypes. Anything else (strings, bools, objects) fails.
        4. Non-finite values in any feature: NaN, +inf, -inf.
        5. Range: every value v of a feature must satisfy spec.min <= v <= spec.max.
        6. One-hot groups (`ONE_HOT_GROUPS`): exactly one column is 1 in every row.
        7. Row consistency (rows whose values are each in range but impossible together):
           - `dst_port_class_none` is 1 exactly when the protocol has no ports
             (`proto_tcp` and `proto_udp` are both 0);
           - every TCP flag count is 0 when `proto_tcp` is 0;
           - `bwd_bytes` is 0 exactly when `bwd_packets` is 0;
           - `pkt_len_min <= pkt_len_mean <= pkt_len_max`.

    Returns:
        None when the table is valid.

    Complexity: O(n_flows * n_features) time, vectorized per column; O(n_flows) extra memory
    for one boolean mask at a time.
    """
    # 0. Duplicates would hide inside the sets below, and df[name] would return a table.
    duplicated = df.columns[df.columns.duplicated()]
    if len(duplicated):
        raise SchemaError(f"duplicate column names: {sorted(set(duplicated))}")

    # 1-2. Columns, as set differences: O(n_columns), no work per row.
    present = set(df.columns)
    required = set(METADATA_COLUMNS) | set(FEATURE_NAMES)
    missing = required - present
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")
    extra = present - required - set(OPTIONAL_COLUMNS)
    if extra:
        raise SchemaError(f"unexpected columns (not in the contract): {sorted(extra)}")

    for spec in FEATURES:
        col = df[spec.name]  # col: (n_flows,)

        # 3. Dtype. A bool column must never pass as a count or a one-hot value, so it is
        #    excluded explicitly from the integer case.
        is_int = pd.api.types.is_integer_dtype(col) and not pd.api.types.is_bool_dtype(col)
        is_float = pd.api.types.is_float_dtype(col)
        ok = is_int if spec.dtype == "int64" else (is_int or is_float)
        if not ok:
            allowed = "an integer dtype" if spec.dtype == "int64" else "an integer or float dtype"
            raise SchemaError(f"column {spec.name!r} has dtype {col.dtype}; expected {allowed}")

        values = col.to_numpy(dtype=np.float64)  # values: (n_flows,)

        # 4. Finite. Must come before the range check: NaN fails every comparison, and
        #    +inf passes `v <= inf` for features whose max is inf.
        bad = ~np.isfinite(values)
        if bad.any():
            raise SchemaError(
                f"column {spec.name!r} has {int(bad.sum())} non-finite value(s) (NaN or inf)"
            )

        # 5. Range [spec.min, spec.max], both ends included.
        bad = (values < spec.min) | (values > spec.max)
        if bad.any():
            first = values[bad][0]
            raise SchemaError(
                f"column {spec.name!r} has {int(bad.sum())} value(s) outside "
                f"[{spec.min}, {spec.max}]; first bad value: {first}"
            )

    # 6. One-hot groups. Every value is now 0 or 1, so "exactly one 1" means row sum == 1.
    for group, columns in ONE_HOT_GROUPS.items():
        row_sums = df[list(columns)].to_numpy().sum(axis=1)  # row_sums: (n_flows,)
        bad = row_sums != 1
        if bad.any():
            raise SchemaError(
                f"one-hot group {group!r} ({', '.join(columns)}) must have exactly one 1 per "
                f"row; {int(bad.sum())} row(s) do not, first at position {int(np.argmax(bad))}"
            )

    _check_row_consistency(df)


def _check_row_consistency(df: pd.DataFrame) -> None:
    """Check 7 of `validate_flows`: relations between features in the same row.

    Assumes checks 0-6 passed. O(n_flows) time per rule, vectorized.
    """
    no_ports = (df["proto_tcp"] + df["proto_udp"]).to_numpy() == 0  # no_ports: (n_flows,)
    _raise_if_any(
        df["dst_port_class_none"].to_numpy() != no_ports.astype(np.int64),
        "dst_port_class_none must be 1 exactly when proto_tcp and proto_udp are both 0",
    )

    not_tcp = df["proto_tcp"].to_numpy() == 0
    for name in ("syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"):
        _raise_if_any(not_tcp & (df[name].to_numpy() != 0), f"{name} must be 0 for non-TCP flows")

    _raise_if_any(
        (df["bwd_packets"].to_numpy() == 0) != (df["bwd_bytes"].to_numpy() == 0),
        "bwd_bytes must be 0 exactly when bwd_packets is 0",
    )

    lo = df["pkt_len_min"].to_numpy(dtype=np.float64)  # lo, mean, hi: (n_flows,)
    mean = df["pkt_len_mean"].to_numpy(dtype=np.float64)
    hi = df["pkt_len_max"].to_numpy(dtype=np.float64)
    tol = 1e-9 * hi  # floating-point rounding in the mean
    _raise_if_any(
        (lo > hi) | (mean < lo - tol) | (mean > hi + tol),
        "pkt_len_min <= pkt_len_mean <= pkt_len_max must hold",
    )


def _raise_if_any(bad: np.ndarray, rule: str) -> None:
    """Raise `SchemaError` naming the rule, the number of bad rows, and the first one."""
    if bad.any():
        raise SchemaError(
            f"{rule}; {int(bad.sum())} row(s) break it, first at position {int(np.argmax(bad))}"
        )
