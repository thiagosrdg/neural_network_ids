"""The one feature function: the packets of one flow → the 29 numbers of the contract.

Every data source goes through `compute_flow_features`: synthetic traffic (T02), public
datasets, and user captures (T10). One implementation means the numbers are comparable, so a
model trained on one source sees the same kind of input from another (no training–serving
skew). Definitions and edge cases: `docs/features.md`.

The input is `FlowPackets`: four aligned arrays, one entry per packet, in processing (file)
order. They are never sorted by time; instead, `effective_times` applies the contract rule
`t_eff = max(t, t_last)`, so a timestamp that goes backwards gives a gap of 0.
"""

from dataclasses import dataclass

import numpy as np

from neural_ids.schema import (
    ACTIVE_TIMEOUT_S,
    FEATURE_NAMES,
    IDLE_TIMEOUT_S,
    MAX_IP_LEN,
    MIN_IP_LEN,
)

# TCP flag bits, with their values in the TCP header's flags byte. T10 can copy the flags
# value that Scapy reads (`int(pkt[TCP].flags)`) without translation.
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20

PROTO_ICMP = 1
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMPV6 = 58


@dataclass(frozen=True)
class FlowPackets:
    """The packets of one flow, in processing (file) order, both directions together.

    Attributes (n = number of packets, n >= 1):
        timestamps: (n,) float64 Unix seconds, as read; may go backwards.
        direction: (n,) int64, +1 forward, -1 backward. The first packet is forward.
        ip_len: (n,) int64 IP-layer length in bytes, MIN_IP_LEN..MAX_IP_LEN.
        tcp_flags: (n,) int64 flags byte (FIN 0x01 ... URG 0x20); all 0 for non-TCP.
        protocol: IP protocol number (6 TCP, 17 UDP, 1 ICMP, 58 ICMPv6, ...).
        dst_port: destination port of the first packet; 0 for protocols without ports.

    The checks in `__post_init__` make every `FlowPackets` a possible flow under the contract,
    including the timeouts: every effective gap <= IDLE_TIMEOUT_S and duration <=
    ACTIVE_TIMEOUT_S. Invalid input raises `ValueError`.
    """

    timestamps: np.ndarray
    direction: np.ndarray
    ip_len: np.ndarray
    tcp_flags: np.ndarray
    protocol: int
    dst_port: int

    def __post_init__(self) -> None:
        # Normalize dtypes once, so the feature code can rely on them. A frozen dataclass
        # blocks normal assignment; object.__setattr__ is the documented way around it.
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("a flow needs at least one packet, as 1-D arrays")
        object.__setattr__(self, "timestamps", timestamps)
        for name in ("direction", "ip_len", "tcp_flags"):
            values = np.asarray(getattr(self, name))
            # Casting floats or bools to int64 would hide bad parsing (60.9 → 60), so only
            # integer dtypes are accepted.
            if values.dtype.kind not in "iu":
                raise ValueError(f"{name} must have an integer dtype, not {values.dtype}")
            object.__setattr__(self, name, values.astype(np.int64))
        _check_packets(self)


def _check_packets(p: FlowPackets) -> None:
    """Raise `ValueError` if `p` is not a possible flow under the contract. O(n) time."""
    n = p.timestamps.shape[0]
    for name in ("direction", "ip_len", "tcp_flags"):
        if getattr(p, name).shape != (n,):
            raise ValueError(f"{name} must have the same length as timestamps ({n})")
    if not np.isfinite(p.timestamps).all():
        raise ValueError("timestamps must be finite")
    if not np.isin(p.direction, (1, -1)).all():
        raise ValueError("direction values must be +1 or -1")
    if p.direction[0] != 1:
        raise ValueError("the first packet defines 'forward', so direction[0] must be +1")
    if ((p.ip_len < MIN_IP_LEN) | (p.ip_len > MAX_IP_LEN)).any():
        raise ValueError(f"ip_len values must be in {MIN_IP_LEN}..{MAX_IP_LEN}")
    if ((p.tcp_flags < 0) | (p.tcp_flags > 0xFF)).any():
        raise ValueError("tcp_flags values must fit in one byte (0..255)")
    if p.protocol != PROTO_TCP and (p.tcp_flags != 0).any():
        raise ValueError("tcp_flags must be 0 for non-TCP flows")
    if not 0 <= p.protocol <= 255:
        raise ValueError("protocol must be 0..255")
    if not 0 <= p.dst_port <= 65535:
        raise ValueError("dst_port must be 0..65535")
    if p.protocol not in (PROTO_TCP, PROTO_UDP) and p.dst_port != 0:
        raise ValueError("protocols without ports (ICMP, other) must use dst_port 0")
    t = effective_times(p.timestamps)
    if n > 1 and np.diff(t).max() > IDLE_TIMEOUT_S:
        raise ValueError(f"a gap exceeds the idle timeout ({IDLE_TIMEOUT_S} s)")
    if t[-1] - t[0] > ACTIVE_TIMEOUT_S:
        raise ValueError(f"the duration exceeds the active timeout ({ACTIVE_TIMEOUT_S} s)")


def effective_times(timestamps: np.ndarray) -> np.ndarray:
    """Contract rule `t_eff = max(t, t_last)`: the running (cumulative) maximum.

    Input: timestamps (n,) float64 in processing order. Output: (n,) float64, never
    decreasing, so every gap is >= 0. O(n) time and memory.
    """
    return np.maximum.accumulate(timestamps)


def compute_flow_features(pkts: FlowPackets) -> dict[str, int | float]:
    """Compute every feature in `schema.FEATURES` for one flow.

    Input: a `FlowPackets` with n >= 1 packets.
    Output: {feature name: value} in `FEATURE_NAMES` order. int64 features are Python `int`
        (one-hot values are 0 or 1, never bool); float64 features are Python `float`. A list
        of these dicts therefore becomes a table whose dtypes match the contract.

    Complexity: O(n) time and O(n) memory; every step is a vectorized NumPy pass.
    """
    t = effective_times(pkts.timestamps)  # t: (n,)
    gaps = np.diff(t)  # gaps: (n-1,), all >= 0
    lengths = pkts.ip_len  # lengths: (n,)
    fwd = pkts.direction == 1  # fwd: (n,) bool mask

    n = int(lengths.shape[0])
    fwd_bytes = int(lengths[fwd].sum())
    bwd_bytes = int(lengths[~fwd].sum())
    duration = float(t[-1] - t[0])

    values: dict[str, int | float] = {
        **_protocol_one_hot(pkts.protocol),
        "duration_s": duration,
        "fwd_packets": int(fwd.sum()),
        "bwd_packets": int(n - fwd.sum()),
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        # Population std (ddof=0, NumPy's default), as the contract requires.
        "pkt_len_mean": float(lengths.mean()),
        "pkt_len_std": float(lengths.std()),
        "pkt_len_min": int(lengths.min()),
        "pkt_len_max": int(lengths.max()),
        **_iat_features(gaps),
        **_flag_counts(pkts.tcp_flags),
        **_rate_features(fwd_bytes + bwd_bytes, n, duration),
        # fwd_bytes >= 20 always (the first packet is forward), so no division by zero.
        "bwd_fwd_bytes_ratio": bwd_bytes / fwd_bytes,
        **_port_class_one_hot(pkts.protocol, pkts.dst_port),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def _iat_features(gaps: np.ndarray) -> dict[str, float]:
    """Inter-arrival times (IAT): mean, population std, and max of the gaps.

    Input: gaps (n-1,) float64, all >= 0. With no gaps (a one-packet flow) all three are 0;
    with one gap the std is 0 by the formula. Zero-duration flows have all gaps 0, so they
    give 0 with no special case.
    """
    if gaps.size == 0:
        return {"iat_mean_s": 0.0, "iat_std_s": 0.0, "iat_max_s": 0.0}
    return {
        "iat_mean_s": float(gaps.mean()),
        "iat_std_s": float(gaps.std()),
        "iat_max_s": float(gaps.max()),
    }


def _rate_features(total_bytes: int, n_packets: int, duration: float) -> dict[str, float]:
    """Bytes and packets per second over the whole flow; both 0 when the duration is 0.

    Dividing by a zero (or near-zero) duration would give inf or huge, meaningless rates,
    so the contract fixes the rates at 0 for zero-duration flows.
    """
    if duration == 0.0:
        return {"bytes_per_s": 0.0, "packets_per_s": 0.0}
    return {"bytes_per_s": total_bytes / duration, "packets_per_s": n_packets / duration}


def _flag_counts(flags: np.ndarray) -> dict[str, int]:
    """Number of packets with each flag bit set. flags: (n,) int64. O(n) per flag."""
    bits = {"syn": SYN, "ack": ACK, "fin": FIN, "rst": RST, "psh": PSH, "urg": URG}
    return {f"{name}_count": int(np.count_nonzero(flags & bit)) for name, bit in bits.items()}


def _protocol_one_hot(protocol: int) -> dict[str, int]:
    """One-hot group `proto`: exactly one of the four columns is 1."""
    return {
        "proto_tcp": int(protocol == PROTO_TCP),
        "proto_udp": int(protocol == PROTO_UDP),
        "proto_icmp": int(protocol in (PROTO_ICMP, PROTO_ICMPV6)),
        "proto_other": int(protocol not in (PROTO_TCP, PROTO_UDP, PROTO_ICMP, PROTO_ICMPV6)),
    }


def _port_class_one_hot(protocol: int, dst_port: int) -> dict[str, int]:
    """One-hot group `dst_port_class`. "none" is set by protocol, never by port value."""
    has_ports = protocol in (PROTO_TCP, PROTO_UDP)
    return {
        "dst_port_class_well_known": int(has_ports and dst_port <= 1023),
        "dst_port_class_registered": int(has_ports and 1024 <= dst_port <= 49151),
        "dst_port_class_dynamic": int(has_ports and dst_port >= 49152),
        "dst_port_class_none": int(not has_ports),
    }


def flow_record(
    pkts: FlowPackets, *, flow_id: str, src_ip: str, dst_ip: str, src_port: int
) -> dict[str, str | int | float]:
    """One row of a flows table: the metadata columns followed by every feature.

    `start_time` and `end_time` are the first and last effective times, so
    `duration_s == end_time - start_time` holds exactly. Addresses and `src_port` are
    metadata only; they never reach `compute_flow_features`.
    """
    t = effective_times(pkts.timestamps)  # t: (n,)
    return {
        "flow_id": flow_id,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": pkts.dst_port,
        "protocol": pkts.protocol,
        "start_time": float(t[0]),
        "end_time": float(t[-1]),
        **compute_flow_features(pkts),
    }
