"""Tests for `neural_ids.features`: `FlowPackets` and `compute_flow_features`.

The expected values are computed by hand (the arithmetic is in the comments), not by calling
NumPy, so the tests check the definitions in `docs/features.md`, not the implementation.
Times start at 1000.0 and use gaps that are exact binary fractions (0.25, 0.5), so the float
arithmetic is exact.
"""

import math

import numpy as np
import pandas as pd
import pytest

from neural_ids.features import (
    ACK,
    FIN,
    PSH,
    RST,
    SYN,
    URG,
    FlowPackets,
    compute_flow_features,
    effective_times,
    flow_record,
)
from neural_ids.schema import FEATURE_NAMES, FEATURES, validate_flows


def make_flow(
    times: list[float],
    direction: list[int],
    ip_len: list[int],
    flags: list[int] | None = None,
    protocol: int = 6,
    dst_port: int = 443,
) -> FlowPackets:
    """Build a `FlowPackets` from short Python lists (flags default to 0)."""
    return FlowPackets(
        timestamps=np.array(times, dtype=np.float64),
        direction=np.array(direction),
        ip_len=np.array(ip_len),
        tcp_flags=np.array(flags if flags is not None else [0] * len(times)),
        protocol=protocol,
        dst_port=dst_port,
    )


# ---------- Hand-computed examples ----------


def handshake_flow() -> FlowPackets:
    """TCP 3-way handshake, one request, one response (client → server port 443).

    #  time     dir  ip_len  flags
    1  1000.00  fwd    60    SYN
    2  1000.25  bwd    60    SYN+ACK
    3  1000.50  fwd    52    ACK
    4  1000.75  fwd   152    PSH+ACK   (request, 100 bytes of data)
    5  1001.25  bwd   552    PSH+ACK   (response, 500 bytes of data)
    """
    return make_flow(
        times=[1000.0, 1000.25, 1000.5, 1000.75, 1001.25],
        direction=[1, -1, 1, 1, -1],
        ip_len=[60, 60, 52, 152, 552],
        flags=[SYN, SYN | ACK, ACK, PSH | ACK, PSH | ACK],
        dst_port=443,
    )


def test_handshake_exact_values() -> None:
    f = compute_flow_features(handshake_flow())
    expected_ints = {
        "proto_tcp": 1,
        "proto_udp": 0,
        "proto_icmp": 0,
        "proto_other": 0,
        "fwd_packets": 3,
        "bwd_packets": 2,
        "fwd_bytes": 264,  # 60 + 52 + 152
        "bwd_bytes": 612,  # 60 + 552
        "pkt_len_min": 52,
        "pkt_len_max": 552,
        "syn_count": 2,  # SYN, SYN+ACK
        "ack_count": 4,  # every packet except the first
        "fin_count": 0,
        "rst_count": 0,
        "psh_count": 2,
        "urg_count": 0,
        "dst_port_class_well_known": 1,
        "dst_port_class_registered": 0,
        "dst_port_class_dynamic": 0,
        "dst_port_class_none": 0,
    }
    for name, value in expected_ints.items():
        assert f[name] == value, name
    assert f["duration_s"] == 1.25
    assert f["pkt_len_mean"] == pytest.approx(175.2)  # 876 / 5
    # Deviations from 175.2: -115.2, -115.2, -123.2, -23.2, 376.8.
    # Squares sum to 184236.8; divided by n = 5 → 36847.36.
    assert f["pkt_len_std"] == pytest.approx(math.sqrt(36847.36))
    # Gaps: 0.25, 0.25, 0.25, 0.5 → mean 1.25 / 4 = 0.3125.
    assert f["iat_mean_s"] == 0.3125
    # Deviations: -0.0625 (x3), +0.1875 → squares 3 * 0.00390625 + 0.03515625 = 0.046875;
    # divided by 4 → 0.01171875.
    assert f["iat_std_s"] == pytest.approx(math.sqrt(0.01171875))
    assert f["iat_max_s"] == 0.5
    assert f["bytes_per_s"] == pytest.approx(700.8)  # 876 / 1.25
    assert f["packets_per_s"] == 4.0  # 5 / 1.25
    assert f["bwd_fwd_bytes_ratio"] == pytest.approx(612 / 264)


def test_single_syn_probe() -> None:
    """`nmap -sS` probe to a filtered port: one 44-byte SYN, no answer."""
    f = compute_flow_features(make_flow([1000.0], [1], [44], [SYN], dst_port=8080))
    assert f["fwd_packets"] == 1 and f["bwd_packets"] == 0
    assert f["fwd_bytes"] == 44 and f["bwd_bytes"] == 0
    assert f["pkt_len_mean"] == 44.0 and f["pkt_len_std"] == 0.0
    assert f["pkt_len_min"] == 44 and f["pkt_len_max"] == 44
    assert f["syn_count"] == 1 and f["ack_count"] == 0 and f["rst_count"] == 0
    # One packet: no gaps and no duration, so every time and rate feature is 0.
    for name in ("duration_s", "iat_mean_s", "iat_std_s", "iat_max_s"):
        assert f[name] == 0.0, name
    assert f["bytes_per_s"] == 0.0 and f["packets_per_s"] == 0.0
    assert f["bwd_fwd_bytes_ratio"] == 0.0
    assert f["dst_port_class_registered"] == 1  # 8080 is in 1024-49151


def test_dns_query_and_response() -> None:
    """UDP query (60 bytes) and response (140 bytes) half a second later."""
    f = compute_flow_features(
        make_flow([1000.0, 1000.5], [1, -1], [60, 140], protocol=17, dst_port=53)
    )
    assert f["proto_udp"] == 1 and f["proto_tcp"] == 0
    assert f["fwd_packets"] == 1 and f["bwd_packets"] == 1
    assert f["duration_s"] == 0.5
    assert f["iat_mean_s"] == 0.5 and f["iat_max_s"] == 0.5
    assert f["iat_std_s"] == 0.0  # one gap: population std is 0
    assert f["pkt_len_mean"] == 100.0 and f["pkt_len_std"] == 40.0
    assert f["bytes_per_s"] == 400.0  # 200 / 0.5
    assert f["packets_per_s"] == 4.0  # 2 / 0.5
    assert f["bwd_fwd_bytes_ratio"] == pytest.approx(140 / 60)
    for name in ("syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"):
        assert f[name] == 0, name
    assert f["dst_port_class_well_known"] == 1


# ---------- Edge cases ----------


def test_backwards_timestamp_gives_zero_gap() -> None:
    """Times 10, 12, 11, 13 → effective times 10, 12, 12, 13 → gaps 2, 0, 1."""
    f = compute_flow_features(
        make_flow([10.0, 12.0, 11.0, 13.0], [1, -1, 1, -1], [40, 40, 40, 40], protocol=17)
    )
    assert f["duration_s"] == 3.0
    assert f["iat_mean_s"] == 1.0
    assert f["iat_std_s"] == pytest.approx(math.sqrt(2 / 3))  # deviations 1, -1, 0
    assert f["iat_max_s"] == 2.0
    assert f["packets_per_s"] == pytest.approx(4 / 3)


def test_effective_times_is_cumulative_max() -> None:
    t = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
    np.testing.assert_array_equal(effective_times(t), [10.0, 12.0, 12.0, 12.0, 13.0])


def test_zero_duration_with_several_packets() -> None:
    """Three packets with the same timestamp: rates are 0, not infinite."""
    f = compute_flow_features(make_flow([5.0, 5.0, 5.0], [1, 1, -1], [100, 100, 60], protocol=17))
    assert f["duration_s"] == 0.0
    assert f["iat_mean_s"] == f["iat_std_s"] == f["iat_max_s"] == 0.0
    assert f["bytes_per_s"] == 0.0 and f["packets_per_s"] == 0.0
    assert f["fwd_packets"] == 2 and f["bwd_bytes"] == 60


def test_flag_bits_are_tcp_header_values() -> None:
    """FIN 0x01, SYN 0x02, RST 0x04, PSH 0x08, ACK 0x10, URG 0x20 (as in the TCP header)."""
    assert (FIN, SYN, RST, PSH, ACK, URG) == (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)
    # An Xmas-scan packet (FIN+PSH+URG = 0x29) counts once in each of its three flags.
    f = compute_flow_features(make_flow([0.0], [1], [40], [0x29]))
    assert (f["fin_count"], f["psh_count"], f["urg_count"]) == (1, 1, 1)
    assert (f["syn_count"], f["ack_count"], f["rst_count"]) == (0, 0, 0)


@pytest.mark.parametrize(
    ("protocol", "dst_port", "proto_col", "port_col"),
    [
        (6, 0, "proto_tcp", "dst_port_class_well_known"),  # port 0 counts as well-known
        (6, 1023, "proto_tcp", "dst_port_class_well_known"),
        (17, 1024, "proto_udp", "dst_port_class_registered"),
        (17, 49151, "proto_udp", "dst_port_class_registered"),
        (6, 49152, "proto_tcp", "dst_port_class_dynamic"),
        (17, 65535, "proto_udp", "dst_port_class_dynamic"),
        (1, 0, "proto_icmp", "dst_port_class_none"),
        (58, 0, "proto_icmp", "dst_port_class_none"),  # ICMPv6
        (47, 0, "proto_other", "dst_port_class_none"),  # GRE
    ],
)
def test_one_hot_protocol_and_port_class(
    protocol: int, dst_port: int, proto_col: str, port_col: str
) -> None:
    f = compute_flow_features(
        make_flow([0.0], [1], [40], [0], protocol=protocol, dst_port=dst_port)
    )
    for group_prefix, hot in (("proto_", proto_col), ("dst_port_class_", port_col)):
        for name in FEATURE_NAMES:
            if name.startswith(group_prefix):
                assert f[name] == (1 if name == hot else 0), name


def test_keys_order_and_python_types() -> None:
    """Keys follow FEATURES; int64 features are Python int (never bool), others Python float."""
    f = compute_flow_features(handshake_flow())
    assert tuple(f) == FEATURE_NAMES
    for spec in FEATURES:
        expected = int if spec.dtype == "int64" else float
        assert type(f[spec.name]) is expected, spec.name


# ---------- flow_record (metadata + features) ----------


def test_flow_record_metadata_uses_effective_times() -> None:
    pkts = make_flow([10.0, 12.0, 11.0], [1, -1, 1], [40, 40, 40], protocol=17, dst_port=53)
    row = flow_record(pkts, flow_id="f1", src_ip="192.0.2.1", dst_ip="198.51.100.2", src_port=50000)
    assert row["start_time"] == 10.0
    assert row["end_time"] == 12.0  # the last effective time, not the last raw time (11.0)
    assert row["duration_s"] == row["end_time"] - row["start_time"]
    assert row["protocol"] == 17 and row["dst_port"] == 53 and row["src_port"] == 50000
    validate_flows(pd.DataFrame([row]))


# ---------- FlowPackets input checks ----------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"times": []}, "at least one packet"),
        ({"direction": [-1, 1]}, "first packet"),
        ({"direction": [1, 0]}, "direction"),
        ({"ip_len": [60]}, "same length"),
        ({"ip_len": [19, 60]}, "ip_len"),
        ({"ip_len": [60, 65576]}, "ip_len"),
        ({"flags": [SYN, 0x100]}, "tcp_flags"),
        ({"protocol": 17, "flags": [SYN, 0]}, "non-TCP"),
        ({"protocol": 1, "dst_port": 80}, "port 0"),
        ({"dst_port": 70000}, "dst_port"),
        ({"times": [0.0, 120.5]}, "idle timeout"),
        ({"times": [0.0, math.nan]}, "finite"),
    ],
)
def test_flowpackets_rejects_bad_input(kwargs: dict, message: str) -> None:
    args = {
        "times": [0.0, 1.0],
        "direction": [1, -1],
        "ip_len": [60, 60],
        "flags": [0, 0],
        "protocol": 6,
        "dst_port": 80,
    }
    args.update(kwargs)
    if args["times"] == []:
        args.update(direction=[], ip_len=[], flags=[])
    with pytest.raises(ValueError, match=message):
        make_flow(**args)


@pytest.mark.parametrize(
    ("field", "value"),
    [("ip_len", [60.9, 60.0]), ("direction", [True, False]), ("tcp_flags", [2.0, 0.0])],
)
def test_flowpackets_rejects_non_integer_fields(field: str, value: list) -> None:
    """Floats and bools are not silently cast (60.9 would become 60)."""
    args = {
        "timestamps": np.array([0.0, 1.0]),
        "direction": np.array([1, -1]),
        "ip_len": np.array([60, 60]),
        "tcp_flags": np.array([0, 0]),
        "protocol": 6,
        "dst_port": 80,
    }
    args[field] = np.array(value)
    with pytest.raises(ValueError, match="integer dtype"):
        FlowPackets(**args)


def test_flowpackets_rejects_scalar_timestamps() -> None:
    with pytest.raises(ValueError, match="1-D"):
        FlowPackets(np.float64(0.0), np.array([1]), np.array([40]), np.array([0]), 17, 53)


def test_flowpackets_rejects_duration_over_active_timeout() -> None:
    times = [100.0 * i for i in range(20)]  # gaps of 100 s, duration 1900 s
    with pytest.raises(ValueError, match="active timeout"):
        make_flow(times, [1] * 20, [40] * 20, protocol=17)


def test_flowpackets_accepts_idle_gap_of_exactly_120() -> None:
    """A gap of exactly IDLE_TIMEOUT_S stays in the flow (the rule is `gap > 120`)."""
    f = compute_flow_features(make_flow([0.0, 120.0], [1, -1], [40, 40], protocol=17))
    assert f["iat_max_s"] == 120.0
