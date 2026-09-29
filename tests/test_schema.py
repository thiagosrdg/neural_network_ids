"""Tests for the feature contract: `neural_ids.schema` and `docs/features.md`."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_ids.schema import (
    ACTIVE_TIMEOUT_S,
    FEATURE_NAMES,
    FEATURES,
    IDLE_TIMEOUT_S,
    MAX_IP_LEN,
    METADATA_COLUMNS,
    MIN_IP_LEN,
    ONE_HOT_GROUPS,
    OPTIONAL_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
    validate_flows,
)

FEATURES_DOC = Path(__file__).resolve().parents[1] / "docs" / "features.md"


# ---------- helpers ----------


def make_valid_flows() -> pd.DataFrame:
    """Two hand-made valid flows: a TCP handshake plus close, and a one-packet ICMP echo.

    Returns a (2, 37) table: 8 metadata columns + 29 features. Addresses come from
    documentation ranges.
    """
    meta = {
        "flow_id": ["f0", "f1"],
        "src_ip": ["192.0.2.10", "2001:db8::1"],
        "dst_ip": ["198.51.100.20", "2001:db8::2"],
        "src_port": [51000, 0],
        "dst_port": [443, 0],
        "protocol": [6, 58],
        "start_time": [1_000.0, 2_000.0],
        "end_time": [1_000.5, 2_000.0],
    }
    feats = {
        "proto_tcp": [1, 0],
        "proto_udp": [0, 0],
        "proto_icmp": [0, 1],
        "proto_other": [0, 0],
        "duration_s": [0.5, 0.0],
        "fwd_packets": [4, 1],
        "bwd_packets": [3, 0],
        "fwd_bytes": [220, 104],
        "bwd_bytes": [160, 0],
        "pkt_len_mean": [380 / 7, 104.0],
        "pkt_len_std": [5.0, 0.0],
        "pkt_len_min": [52, 104],
        "pkt_len_max": [60, 104],
        "iat_mean_s": [0.5 / 6, 0.0],
        "iat_std_s": [0.01, 0.0],
        "iat_max_s": [0.2, 0.0],
        "syn_count": [2, 0],
        "ack_count": [6, 0],
        "fin_count": [2, 0],
        "rst_count": [0, 0],
        "psh_count": [1, 0],
        "urg_count": [0, 0],
        "bytes_per_s": [760.0, 0.0],
        "packets_per_s": [14.0, 0.0],
        "bwd_fwd_bytes_ratio": [160 / 220, 0.0],
        "dst_port_class_well_known": [1, 0],
        "dst_port_class_registered": [0, 0],
        "dst_port_class_dynamic": [0, 0],
        "dst_port_class_none": [0, 1],
    }
    df = pd.DataFrame({**meta, **feats})
    # Force the contract dtypes so the helper itself is exact.
    for spec in FEATURES:
        df[spec.name] = df[spec.name].astype(spec.dtype)
    return df


def feature_doc_rows() -> dict[str, list[str]]:
    """Parse the feature table in docs/features.md: feature name -> list of cell texts."""
    rows: dict[str, list[str]] = {}
    for line in FEATURES_DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name = cells[0].strip("`")
        if name in FEATURE_NAMES:
            rows[name] = cells
    return rows


# ---------- the contract itself (pass now) ----------


def test_version_and_timeouts() -> None:
    assert SCHEMA_VERSION == "1.0"
    assert IDLE_TIMEOUT_S == 120.0
    assert ACTIVE_TIMEOUT_S == 1800.0


def test_29_unique_features_in_fixed_order() -> None:
    assert len(FEATURES) == 29
    assert len(set(FEATURE_NAMES)) == 29
    assert FEATURE_NAMES[0] == "proto_tcp"
    assert FEATURE_NAMES[-1] == "dst_port_class_none"


def test_specs_are_well_formed() -> None:
    for spec in FEATURES:
        assert spec.dtype in ("int64", "float64"), spec.name
        assert spec.min <= spec.max, spec.name
        assert spec.min >= 0, spec.name  # every v1 feature is a count, size, time, or rate


def test_ranges_follow_timeouts_and_ip_lengths() -> None:
    specs = {s.name: s for s in FEATURES}
    assert specs["duration_s"].max == ACTIVE_TIMEOUT_S
    assert specs["iat_max_s"].max == IDLE_TIMEOUT_S
    for name in ("pkt_len_min", "pkt_len_max", "pkt_len_mean"):
        assert (specs[name].min, specs[name].max) == (MIN_IP_LEN, MAX_IP_LEN) == (20, 65575)


def test_no_identifiers_or_labels_in_features() -> None:
    # Identifiers let the model memorize hosts; labels as inputs are target leakage.
    forbidden = set(METADATA_COLUMNS) | set(OPTIONAL_COLUMNS)
    assert forbidden.isdisjoint(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        for part in ("_ip", "mac", "time", "flow_id", "label", "attack", "capture"):
            assert part not in name, f"{name} looks like an identifier or a label"
        # Port *class* is allowed; an exact port number is not.
        assert not name.endswith("_port"), name


def test_one_hot_groups_are_features_and_disjoint() -> None:
    members = [c for group in ONE_HOT_GROUPS.values() for c in group]
    assert len(members) == len(set(members))
    assert set(members) <= set(FEATURE_NAMES)


def test_features_doc_lists_every_feature() -> None:
    rows = feature_doc_rows()
    assert set(rows) == set(FEATURE_NAMES)


# ---------- validate_flows (fail until TODO(human) is done) ----------


def test_valid_table_passes() -> None:
    assert validate_flows(make_valid_flows()) is None


def test_optional_columns_are_allowed() -> None:
    df = make_valid_flows()
    df["label"] = ["normal", "ping_sweep"]
    df["is_attack"] = [0, 1]
    df["capture_id"] = ["cap-a", "cap-b"]
    assert validate_flows(df) is None


def test_column_order_does_not_matter() -> None:
    df = make_valid_flows()
    assert validate_flows(df[df.columns[::-1]]) is None


def test_float_feature_accepts_int_dtype() -> None:
    # A CSV column of whole numbers is read back as int64.
    df = make_valid_flows()
    df["duration_s"] = df["duration_s"].round().astype("int64")
    assert validate_flows(df) is None


@pytest.mark.parametrize("column", ["syn_count", "duration_s", "src_ip", "end_time"])
def test_missing_column_raises(column: str) -> None:
    df = make_valid_flows().drop(columns=[column])
    with pytest.raises(SchemaError, match=column):
        validate_flows(df)


def test_extra_column_raises() -> None:
    df = make_valid_flows()
    df["payload_hex"] = ["00", "00"]
    with pytest.raises(SchemaError, match="payload_hex"):
        validate_flows(df)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("syn_count", [2.0, 0.0]),  # int feature stored as float
        ("proto_tcp", [True, False]),  # bools are not integers here
        ("duration_s", ["0.5", "0.0"]),  # strings
    ],
)
def test_wrong_dtype_raises(column: str, values: list) -> None:
    df = make_valid_flows()
    df[column] = values
    with pytest.raises(SchemaError, match=column):
        validate_flows(df)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("column", ["duration_s", "bytes_per_s", "bwd_fwd_bytes_ratio"])
def test_non_finite_raises(column: str, bad: float) -> None:
    # bytes_per_s and bwd_fwd_bytes_ratio have max = inf, so only a finiteness check catches +inf.
    df = make_valid_flows()
    df.loc[1, column] = bad
    with pytest.raises(SchemaError, match=column):
        validate_flows(df)


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("duration_s", -0.001),
        ("duration_s", ACTIVE_TIMEOUT_S + 1),
        ("iat_max_s", IDLE_TIMEOUT_S + 0.5),
        ("pkt_len_min", MIN_IP_LEN - 1),
        ("pkt_len_max", MAX_IP_LEN + 1),
        ("fwd_packets", 0),
        ("rst_count", -1),
        ("proto_udp", 2),
    ],
)
def test_out_of_range_raises(column: str, bad: float) -> None:
    df = make_valid_flows()
    dtype = df[column].dtype
    df[column] = df[column].astype("float64")
    df.loc[0, column] = bad
    df[column] = df[column].astype(dtype)
    with pytest.raises(SchemaError, match=column):
        validate_flows(df)


def test_values_at_range_limits_pass() -> None:
    df = make_valid_flows()
    df.loc[0, "duration_s"] = ACTIVE_TIMEOUT_S
    df.loc[0, "iat_max_s"] = IDLE_TIMEOUT_S
    df.loc[0, "pkt_len_max"] = MAX_IP_LEN
    assert validate_flows(df) is None


def test_one_hot_with_two_ones_raises() -> None:
    df = make_valid_flows()
    df.loc[0, "proto_udp"] = 1  # row 0 is now both TCP and UDP
    with pytest.raises(SchemaError, match="proto"):
        validate_flows(df)


def test_one_hot_with_no_ones_raises() -> None:
    df = make_valid_flows()
    df.loc[1, "dst_port_class_none"] = 0  # row 1 has no port class
    with pytest.raises(SchemaError, match="dst_port_class"):
        validate_flows(df)


def test_duplicate_column_raises() -> None:
    df = make_valid_flows()
    df = pd.concat([df, df[["syn_count"]]], axis=1)
    with pytest.raises(SchemaError, match="syn_count"):
        validate_flows(df)


@pytest.mark.parametrize(
    ("row", "changes", "match"),
    [
        # TCP flow marked as portless.
        (0, {"dst_port_class_well_known": 0, "dst_port_class_none": 1}, "dst_port_class_none"),
        # ICMP flow with a port class.
        (1, {"dst_port_class_none": 0, "dst_port_class_dynamic": 1}, "dst_port_class_none"),
        # ICMP flow with TCP flags.
        (1, {"syn_count": 5}, "syn_count"),
        # Backward bytes without backward packets, and the reverse.
        (0, {"bwd_packets": 0}, "bwd_bytes"),
        (1, {"bwd_bytes": 500}, "bwd_bytes"),
        # Impossible length statistics.
        (0, {"pkt_len_min": 61}, "pkt_len_min"),
        (0, {"pkt_len_mean": 70.0}, "pkt_len_mean"),
    ],
)
def test_inconsistent_row_raises(row: int, changes: dict, match: str) -> None:
    df = make_valid_flows()
    for column, value in changes.items():
        df.loc[row, column] = value
    with pytest.raises(SchemaError, match=match):
        validate_flows(df)


def test_empty_table_with_right_columns_passes() -> None:
    df = make_valid_flows().iloc[0:0]
    assert validate_flows(df) is None


# ---------- docs/features.md (fails until TODO(human) is done) ----------


def test_at_least_10_attack_explanations() -> None:
    rows = feature_doc_rows()
    filled = [name for name, cells in rows.items() if cells[-1] and "TODO(human)" not in cells[-1]]
    assert len(filled) >= 10, f"only {len(filled)} of 10 'why' cells filled: {filled}"


def test_features_doc_states_version() -> None:
    assert f"Schema version: {SCHEMA_VERSION}" in FEATURES_DOC.read_text(encoding="utf-8")


def test_helper_is_consistent() -> None:
    # Guards the test helper itself: 8 metadata + 29 features, contract dtypes.
    df = make_valid_flows()
    assert list(df.columns) == [*METADATA_COLUMNS, *FEATURE_NAMES]
    assert df["fwd_packets"].dtype == np.int64
