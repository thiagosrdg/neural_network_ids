"""Synthetic traffic: packet-level generators for normal and attack flows, and a dataset.

SYNTHETIC DATA PROVES THAT CODE WORKS, NOT THAT DETECTION WORKS. The generators follow simple
rules written here, so a model that separates them has learned these rules, not real attacks.

Each generator returns the packets of ONE flow (`FlowPackets`); features always come from
`neural_ids.features.compute_flow_features`, the same function used for real captures.

Generated flows follow the contract's flow rules, so T10 would split a capture of these
packets into the same flows:
- every gap <= 110 s and every duration <= 1700 s (margins under the 120 s and 1800 s
  timeouts);
- a TCP flow ends at its RST, or at the final ACK after the closing FIN, and nothing comes
  after that packet. The only exception is a probe with no answer (a filtered port), which
  ends by idle timeout.
Timestamps are rounded to whole microseconds, the default resolution of pcap files, so the
time features survive a pcap round trip up to float64 rounding (about 0.24 µs).

`difficulty` (0 to 1) changes the generated PACKETS (sizes, gaps, packet counts, and
realistic "hard" variants), never the computed features, so every row is a possible flow.
It also flips up to `LABEL_NOISE_AT_MAX` of the labels.

Run: uv run python -m neural_ids.synthetic --n 20000 --seed 42
→ data/processed/synthetic.csv, plus synthetic.json with how the table was made.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from neural_ids.features import (
    ACK,
    FIN,
    PROTO_TCP,
    PROTO_UDP,
    PSH,
    RST,
    SYN,
    FlowPackets,
    flow_record,
)
from neural_ids.schema import SCHEMA_VERSION, validate_flows
from neural_ids.utils import set_seed

NORMAL_CLASSES: tuple[str, ...] = ("web", "dns", "ssh_session")
ATTACK_CLASSES: tuple[str, ...] = ("syn_scan", "ssh_bruteforce", "udp_flood")
LABEL_NOISE_AT_MAX = 0.05  # share of flipped labels at difficulty = 1 (linear in difficulty)

BASE_TIME = 1_767_225_600.0  # 2026-01-01 00:00:00 UTC; flows start within one day after it
_MAX_GAP_S = 110.0  # under IDLE_TIMEOUT_S (120) with room for float rounding
_MAX_DURATION_S = 1700.0  # under ACTIVE_TIMEOUT_S (1800)
_DOC_NETS = ("192.0.2.", "198.51.100.", "203.0.113.")  # documentation ranges (RFC 5737)

# IPv4 packet sizes in bytes (IP header 20 + transport header).
SYN_LINUX = 60  # TCP header 40: MSS, SACK, timestamps, window scale options
SYN_NMAP = 44  # TCP header 24: `nmap -sS` sends only the MSS option
TCP_HDR = 52  # TCP header 32: 20 + timestamps option; a pure ACK
RST_LEN = 40  # TCP header 20, no options
UDP_HDR = 28  # UDP header 8
MSS = 1448  # TCP payload per full segment (1500 MTU - 52)
MAX_UDP_LEN = 1500  # one Ethernet MTU, no fragmentation

EPHEMERAL_PORTS = (32768, 61000)  # Linux client ports [low, high): registered and dynamic
_COMMON_PORTS = (21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 3306, 3389, 5900, 8080)


class _Builder:
    """Collects the packets of one flow as (gap before, direction, ip_len, flags) arrays."""

    def __init__(self) -> None:
        self._parts: list[tuple[np.ndarray, ...]] = []

    def add(
        self,
        gap: float | np.ndarray,
        direction: int | np.ndarray,
        ip_len: int | np.ndarray,
        flags: int | np.ndarray = 0,
    ) -> None:
        """Append k packets; scalars are broadcast to the length of the array arguments."""
        arrays = (np.atleast_1d(np.asarray(x)) for x in (gap, direction, ip_len, flags))
        self._parts.append(tuple(np.broadcast_arrays(*arrays)))

    def build(self, t0: float, protocol: int, dst_port: int) -> FlowPackets:
        """Turn gaps into timestamps starting at t0. O(n) time and memory."""
        columns = zip(*self._parts, strict=True)  # 4 columns, each a tuple of arrays
        gaps, direction, ip_len, flags = (np.concatenate(col) for col in columns)
        gaps = np.clip(gaps.astype(np.float64), 0.0, _MAX_GAP_S)  # gaps: (n,)
        gaps[0] = 0.0  # the first packet starts the flow
        timestamps = np.round(t0 + np.cumsum(gaps), 6)  # whole microseconds, as in pcap
        return FlowPackets(
            timestamps=timestamps,
            direction=direction,
            ip_len=ip_len,
            tcp_flags=flags,
            protocol=protocol,
            dst_port=dst_port,
        )


def _lerp(easy: float, hard: float, difficulty: float) -> float:
    """Linear interpolation: `easy` at difficulty 0, `hard` at difficulty 1."""
    return easy + (hard - easy) * difficulty


def _tcp_open(b: _Builder, rtt: float, syn_len: int = SYN_LINUX) -> None:
    """3-way handshake: SYN (fwd), SYN+ACK (bwd) one RTT later, ACK (fwd)."""
    b.add([0.0, rtt, 0.0005], [1, -1, 1], [syn_len, syn_len, TCP_HDR], [SYN, SYN | ACK, ACK])


def _tcp_close(b: _Builder, rtt: float, closer: int) -> None:
    """FIN+ACK from `closer`, the closing FIN+ACK from the other side, then the final ACK.

    The final ACK is the flow's last packet under the contract's TCP close rule.
    """
    b.add(
        [0.001, rtt / 2, rtt / 2],
        [closer, -closer, closer],
        TCP_HDR,
        [FIN | ACK, FIN | ACK, ACK],
    )


def _ssh_setup(b: _Builder, rtt: float, rng: np.random.Generator, jitter: float) -> None:
    """SSH version banners and key exchange: 8 packets, alternating sides.

    Payload sizes are those of a typical OpenSSH exchange; `jitter` (0 = identical every
    time, as with one attack tool) scales random changes to them.
    """
    payload = np.array([21, 21, 1392, 1080, 48, 564, 16, 52])  # payload: (8,) bytes
    payload = payload + (jitter * rng.normal(0.0, 0.2, 8) * payload).astype(np.int64)
    direction = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    b.add(rtt / 2, direction, TCP_HDR + np.clip(payload, 1, MSS), PSH | ACK)


# ---------- Normal traffic ----------


def web(rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME) -> FlowPackets:
    """One HTTP(S) connection: handshake, request, larger response, idle, client close.

    Hard variants:
    - closed port (probability 0.3 * difficulty): the server's stack answers the SYN with
      RST+ACK, like a closed-port probe of `syn_scan`;
    - seen mid-connection (probability 0.2 * difficulty): the capture started during a
      download, so the first packet seen comes from the server. "Forward" is then server →
      client and `dst_port` is the client's ephemeral port (a documented contract limitation),
      so a benign flow gets a registered or dynamic port class.
    """
    port = int(rng.choice([80, 443, 8080, 8443], p=[0.2, 0.7, 0.05, 0.05]))
    rtt = float(rng.uniform(0.005, 0.1))
    b = _Builder()
    roll = rng.random()
    if roll < 0.3 * difficulty:
        b.add([0.0, rtt], [1, -1], [SYN_LINUX, RST_LEN], [SYN, RST | ACK])
        return b.build(t0, PROTO_TCP, port)
    if roll < 0.5 * difficulty:
        k = int(min(1 + rng.geometric(0.1), 300))  # rest of the download, server → client
        flags = np.full(k, ACK)
        flags[-1] = PSH | ACK
        b.add(rng.exponential(0.002, k), 1, TCP_HDR + MSS, flags)
        b.add(rtt / 2, -1, TCP_HDR, ACK)  # the client ACKs
        _tcp_close(b, rtt, closer=-1)  # the client (backward here) closes
        return b.build(t0, PROTO_TCP, int(rng.integers(*EPHEMERAL_PORTS)))

    _tcp_open(b, rtt)
    b.add(rng.uniform(0.001, 0.05), 1, TCP_HDR + rng.integers(150, 900), PSH | ACK)  # request
    k = int(min(1 + rng.geometric(0.15), 200))  # response segments
    sizes = np.full(k, TCP_HDR + MSS)  # sizes: (k,)
    sizes[-1] = TCP_HDR + rng.integers(200, MSS + 1)
    gaps = np.r_[rtt + rng.uniform(0.005, 0.3), rng.exponential(0.0005, k - 1)]  # server time
    flags = np.full(k, ACK)
    flags[-1] = PSH | ACK
    b.add(gaps, -1, sizes, flags)
    b.add(rtt / 2, 1, TCP_HDR, ACK)  # client ACKs the response
    b.add(rng.exponential(_lerp(1.0, 20.0, difficulty)), 1, TCP_HDR, ACK)  # idle keep-alive
    _tcp_close(b, rtt, closer=1)
    return b.build(t0, PROTO_TCP, port)


def dns(rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME) -> FlowPackets:
    """One UDP query to port 53 and one response.

    Hard variants:
    - no answer (probability 0.3 * difficulty): the client retries 0-2 times on the same
      socket, 5 s apart (the `dig` default timeout), like a tiny UDP flood;
    - response only (probability 0.2 * difficulty): the capture missed the query, so the one
      packet seen goes from the server to the client's ephemeral port (registered or dynamic
      class), with no reply, like a one-packet UDP flood.
    """
    rtt = float(rng.uniform(0.002, 0.15))
    query = UDP_HDR + int(rng.integers(29, 80))  # 12-byte DNS header + question
    answer = query + int(rng.integers(16, 400))
    b = _Builder()
    roll = rng.random()
    if roll < 0.3 * difficulty:
        tries = int(rng.integers(1, 4))
        b.add(np.r_[0.0, np.full(tries - 1, 5.0)], 1, query)
    elif roll < 0.5 * difficulty:
        b.add(0.0, 1, answer)
        return b.build(t0, PROTO_UDP, int(rng.integers(*EPHEMERAL_PORTS)))
    else:
        b.add([0.0, rtt], [1, -1], [query, answer])
    return b.build(t0, PROTO_UDP, 53)


def ssh_session(
    rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME
) -> FlowPackets:
    """An interactive SSH session: handshake, key exchange, human typing, client close.

    Each keystroke is a small client packet echoed by the server; sometimes the server sends
    command output instead. Gaps between keystrokes are log-normal (irregular, human) with
    occasional long pauses. With higher difficulty sessions can be as short as 3 keystrokes,
    which makes them look more like brute force.

    Hard variant (probability 0.4 * difficulty): a short benign SSH connection (see
    `_short_ssh`: a mistyped password, `git fetch`, or a small `scp`), which looks like one
    brute-force connection when seen as a single flow.
    """
    rtt = float(rng.uniform(0.01, 0.15))
    b = _Builder()
    _tcp_open(b, rtt)
    _ssh_setup(b, rtt, rng, jitter=1.0)
    if rng.random() < 0.4 * difficulty:
        _short_ssh(b, rtt, rng)
        _tcp_close(b, rtt, closer=1)
        return b.build(t0, PROTO_TCP, 22)

    m = int(rng.integers(round(_lerp(30, 3, difficulty)), 400))  # keystrokes
    typing = rng.lognormal(np.log(0.25), 1.0, m)  # typing: (m,) seconds
    pauses = rng.random(m) < 0.05
    typing[pauses] += rng.uniform(5.0, 100.0, int(pauses.sum()))
    typing = np.minimum(typing, _MAX_GAP_S)
    typing = typing[np.cumsum(typing + rtt) <= _MAX_DURATION_S - 100.0]  # room for the rest
    m = typing.size

    output = rng.random(m) < 0.15  # server prints command output
    echo = np.where(output, TCP_HDR + rng.integers(37, MSS + 1, m), TCP_HDR + 36)
    gaps = np.stack([typing, np.full(m, rtt)], axis=1).ravel()  # (2m,) keystroke, echo, ...
    sizes = np.stack([np.full(m, TCP_HDR + 36), echo], axis=1).ravel()
    b.add(gaps, np.tile([1, -1], m), sizes, PSH | ACK)
    _tcp_close(b, rtt, closer=1)
    return b.build(t0, PROTO_TCP, 22)


def _short_ssh(b: _Builder, rtt: float, rng: np.random.Generator) -> None:
    """The rest of a short benign SSH connection, after the key exchange (before the close).

    - mistyped password: one failed attempt, the same packets and server delay as one
      brute-force attempt, then a correct password and a few keystrokes;
    - `git fetch`: key login, one request, a burst of response segments;
    - small `scp` upload: key login, a burst of data segments, one reply.

    Seen as one flow, a mistyped password and a brute-force connection look almost the same;
    only the number of attempts over many flows (host-window features) separates them.
    """
    kind = rng.choice(["mistyped_password", "git_fetch", "scp_upload"])
    password = TCP_HDR + 84 + int(rng.integers(0, 80))
    if kind == "mistyped_password":
        # Human typing time, then the server's failure delay (as in `ssh_bruteforce`).
        b.add([rng.uniform(1.0, 8.0), rng.uniform(0.5, 2.5)], [1, -1], [password, TCP_HDR + 36])
        login = (rng.uniform(1.0, 8.0), password)  # retyped password
    else:
        login = (0.005, TCP_HDR + int(rng.integers(300, 500)))  # public key, no typing
    b.add([login[0], rtt], [1, -1], [login[1], TCP_HDR + 20], PSH | ACK)  # auth success
    b.add([0.002, rtt], [1, -1], [TCP_HDR + 100, TCP_HDR + 60], PSH | ACK)  # open channel

    if kind == "mistyped_password":
        m = int(rng.integers(0, 8))  # a few keystrokes, each echoed (0: logged in, then left)
        gaps = np.stack([rng.lognormal(np.log(0.25), 1.0, m), np.full(m, rtt)], axis=1).ravel()
        b.add(gaps, np.tile([1, -1], m), TCP_HDR + 36, PSH | ACK)
    else:
        k = int(rng.integers(1, 20))  # data segments
        data_dir = -1 if kind == "git_fetch" else 1
        b.add(0.002, -data_dir, TCP_HDR + int(rng.integers(100, 400)), PSH | ACK)  # request
        b.add(rng.exponential(0.001, k), data_dir, TCP_HDR + MSS, ACK)  # the data burst
        b.add(rtt, -data_dir, TCP_HDR + 36, PSH | ACK)  # exit status / acknowledgment


# ---------- Attacks ----------


def syn_scan(
    rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME
) -> FlowPackets:
    """One `nmap -sS` probe to one port (a scan is many calls, one flow per port).

    - open port: SYN → SYN+ACK → RST. The scanner's kernel never sent the SYN (nmap crafts
      it), so it resets the unexpected SYN+ACK; the handshake never completes.
    - closed port: SYN → RST+ACK from the target.
    - filtered port: SYN, nothing back. nmap retransmits with a new source port
      (`sport = base_port + tryno`), so a retry is a separate one-packet flow.

    Hard variant (probability 0.5 * difficulty): `nmap -sT` connect scan. The OS makes the
    connection, so the SYN is a normal 60-byte one, and an open port gets the final ACK of a
    full handshake before nmap closes it with RST+ACK.
    """
    state = rng.choice(["open", "closed", "filtered"], p=[0.1, 0.6, 0.3])
    common = rng.random() < 0.5
    port = int(rng.choice(_COMMON_PORTS)) if common else int(rng.integers(1, 65536))
    rtt = float(rng.uniform(0.001, 0.1))
    connect = rng.random() < 0.5 * difficulty
    syn_len = SYN_LINUX if connect else SYN_NMAP

    b = _Builder()
    b.add(0.0, 1, syn_len, SYN)
    if state == "closed":
        b.add(rtt, -1, RST_LEN, RST | ACK)
    elif state == "open":
        b.add(rtt, -1, syn_len, SYN | ACK)
        if connect:
            b.add([0.0001, 0.0001], [1, 1], [TCP_HDR, RST_LEN], [ACK, RST | ACK])
        else:
            b.add(0.0001, 1, RST_LEN, RST)
    return b.build(t0, PROTO_TCP, port)


def ssh_bruteforce(
    rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME
) -> FlowPackets:
    """One connection of a password-guessing tool (for example hydra) against SSH.

    Handshake and key exchange with identical sizes every time, then 3-6 login attempts: a
    password packet and the server's "failed" reply after its fixed delay. After the last
    allowed attempt the server closes the connection. Machine timing: at difficulty 0 the
    gaps vary by 1%; at difficulty 1 by 40%, and packet sizes vary too.

    Low-and-slow at higher difficulty: as few as 1 attempt per connection, and the tool
    waits up to 5 s before each attempt (for example hydra's `-c`), which stays under
    per-connection limits such as sshd's MaxAuthTries. One such flow looks like a person who
    mistyped a password once (`ssh_session`); only host-window features, which count attempts
    over many flows, can tell them apart.
    """
    rtt = float(rng.uniform(0.005, 0.1))
    jitter = _lerp(0.01, 0.4, difficulty)
    b = _Builder()
    _tcp_open(b, rtt)
    _ssh_setup(b, rtt, rng, jitter=difficulty)

    k = int(rng.integers(round(_lerp(3, 1, difficulty)), 7))  # attempts in this connection
    delay = float(rng.uniform(0.5, 2.5))  # server's failure delay, fixed per connection
    reply_gaps = np.maximum(delay * (1.0 + jitter * rng.standard_normal(k)), 0.01)
    wait = _lerp(0.002, float(rng.uniform(0.2, 5.0)), difficulty)  # tool's wait per attempt
    send_gaps = np.maximum(wait * (1.0 + jitter * rng.standard_normal(k)), 0.0)
    extra = rng.integers(0, int(80 * difficulty) + 1, k)  # extra: (k,) password length change
    gaps = np.stack([send_gaps, reply_gaps], axis=1).ravel()  # (2k,)
    sizes = np.stack([TCP_HDR + 84 + extra, np.full(k, TCP_HDR + 36)], axis=1).ravel()
    b.add(gaps, np.tile([1, -1], k), sizes, PSH | ACK)
    _tcp_close(b, rtt, closer=-1)  # the server hangs up
    return b.build(t0, PROTO_TCP, 22)


def udp_flood(
    rng: np.random.Generator, difficulty: float = 0.5, t0: float = BASE_TIME
) -> FlowPackets:
    """Many forward UDP packets to one port at a high rate; the target never answers.

    At difficulty 0: 300-3000 identical packets at 1,000-20,000 packets/s with regular gaps.
    At difficulty 1: as few as 3 packets, rates down to 2 packets/s, random gaps and sizes,
    so small slow floods look like unanswered DNS queries.
    """
    port = int(rng.choice([53, 80, 123, 443, int(rng.integers(1, 65536))]))
    n = int(rng.integers(round(_lerp(300, 3, difficulty)), 3001))
    pps = float(np.exp(rng.uniform(np.log(_lerp(1000.0, 2.0, difficulty)), np.log(20000.0))))
    jitter = _lerp(0.05, 1.0, difficulty)
    gaps = np.maximum((1.0 / pps) * (1.0 + jitter * rng.standard_normal(n)), 0.0)  # (n,)
    gaps = gaps[np.cumsum(np.minimum(gaps, _MAX_GAP_S)) <= _MAX_DURATION_S]  # the first stays
    n = gaps.size

    base = int(rng.integers(UDP_HDR, MAX_UDP_LEN - 400))
    sizes = base + rng.integers(0, int(400 * difficulty) + 1, n)  # sizes: (n,)
    b = _Builder()
    b.add(gaps, 1, sizes)
    return b.build(t0, PROTO_UDP, port)


FlowGenerator = Callable[..., FlowPackets]
GENERATORS: dict[str, FlowGenerator] = {
    "web": web,
    "dns": dns,
    "ssh_session": ssh_session,
    "syn_scan": syn_scan,
    "ssh_bruteforce": ssh_bruteforce,
    "udp_flood": udp_flood,
}


# ---------- Dataset ----------


def make_dataset(
    n_flows: int,
    attack_fraction: float = 0.2,
    difficulty: float = 0.5,
    seed: int = 42,
    label_noise: float | None = None,
) -> pd.DataFrame:
    """A synthetic flows table that follows the contract (see `make_dataset_with_provenance`)."""
    df, _ = make_dataset_with_provenance(n_flows, attack_fraction, difficulty, seed, label_noise)
    return df


def make_dataset_with_provenance(
    n_flows: int,
    attack_fraction: float = 0.2,
    difficulty: float = 0.5,
    seed: int = 42,
    label_noise: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """A synthetic flows table that follows the contract, and how it was made.

    Args:
        n_flows: number of rows (flows), >= 1.
        attack_fraction: share of attack flows, 0 to 1 (rounded to a whole count); the
            attack and normal counts are split as evenly as possible over their 3 classes.
        difficulty: 0 to 1; see the module docstring.
        seed: seed for `set_seed`; the same seed gives the same table.
        label_noise: share of flipped labels, 0 to 1; None means
            `LABEL_NOISE_AT_MAX * difficulty`. Use 0 to see the true generating class (for
            example in plots); the flows are the same for any value.

    Returns:
        df: (n_flows, 39) DataFrame: 8 metadata columns, 29 features, `label` (class name),
            and `is_attack` (0 or 1). Rows are in random order. `validate_flows` passes.
        provenance: JSON-ready dict (see `_provenance`): the arguments, SCHEMA_VERSION, class
            counts before and after label noise, flips in each direction, and the scores a
            perfect model would get against the noisy labels.

    Complexity: O(total packets) time; one Python call per flow (flows have different
    lengths, so they cannot share one array), vectorized inside each flow.
    """
    if n_flows < 1:
        raise ValueError("n_flows must be >= 1")
    if not 0.0 <= attack_fraction <= 1.0:
        raise ValueError("attack_fraction must be in [0, 1]")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    if label_noise is None:
        label_noise = LABEL_NOISE_AT_MAX * difficulty
    if not 0.0 <= label_noise <= 1.0:
        raise ValueError("label_noise must be in [0, 1]")

    rng = set_seed(seed)
    n_attack = round(n_flows * attack_fraction)
    classes = rng.permutation(
        np.r_[_spread(NORMAL_CLASSES, n_flows - n_attack), _spread(ATTACK_CLASSES, n_attack)]
    )  # classes: (n_flows,) true class of each generated flow

    rows = [_synthetic_row(i, str(name), rng, difficulty) for i, name in enumerate(classes)]
    df = pd.DataFrame(rows)
    df["label"], df["is_attack"] = _noisy_labels(classes, label_noise, rng)
    validate_flows(df)
    args = {
        "n_flows": n_flows,
        "seed": seed,
        "attack_fraction": attack_fraction,
        "difficulty": difficulty,
        "label_noise": label_noise,
    }
    return df, _provenance(args, classes, df["label"].to_numpy())


def _provenance(args: dict, classes: np.ndarray, labels: np.ndarray) -> dict:
    """How a synthetic table was made, from the true classes and the (noisy) labels.

    Input: classes and labels, both (n,) class names. O(n) time.
    """
    true_attack = np.isin(classes, ATTACK_CLASSES)  # (n,) bool
    labeled_attack = np.isin(labels, ATTACK_CLASSES)  # (n,) bool
    names = NORMAL_CLASSES + ATTACK_CLASSES
    n_normal_to_attack = int((~true_attack & labeled_attack).sum())
    n_attack_to_normal = int((true_attack & ~labeled_attack).sum())
    return {
        "data": "SYNTHETIC (tests code, not detection)",
        **args,
        "schema_version": SCHEMA_VERSION,
        "class_counts_true": {name: int((classes == name).sum()) for name in names},
        "class_counts_labeled": {name: int((labels == name).sum()) for name in names},
        "flipped_normal_to_attack": n_normal_to_attack,
        "flipped_attack_to_normal": n_attack_to_normal,
        "perfect_model_vs_noisy_labels": perfect_model_scores(
            int(true_attack.sum()), n_attack_to_normal, n_normal_to_attack
        ),
    }


def perfect_model_scores(
    n_true_attack: int, n_attack_to_normal: int, n_normal_to_attack: int
) -> dict[str, float | None]:
    """Attack recall and precision of a model that always predicts the TRUE class.

    Scored against the noisy labels, even a perfect model loses points:
    - true positives = true attacks still labeled attack = A - a2n
    - recall = TP / labeled attacks = (A - a2n) / (A - a2n + n2a)
    - precision = TP / predicted attacks = (A - a2n) / A
    A value is None when its denominator is 0. O(1).
    """
    tp = n_true_attack - n_attack_to_normal
    labeled = tp + n_normal_to_attack
    return {
        "attack_recall": tp / labeled if labeled else None,
        "attack_precision": tp / n_true_attack if n_true_attack else None,
    }


def _spread(names: tuple[str, ...], count: int) -> np.ndarray:
    """`count` class names split as evenly as possible, for example 7 → 3, 2, 2."""
    sizes = [len(part) for part in np.array_split(np.arange(count), len(names))]
    return np.repeat(np.array(names), sizes)


def _synthetic_row(
    i: int, name: str, rng: np.random.Generator, difficulty: float
) -> dict[str, str | int | float]:
    """Generate one flow of class `name` and return its metadata + feature row.

    Addresses and ports are drawn independently of the class, so metadata cannot hint at the
    label (it is never a model input anyway).
    """
    t0 = BASE_TIME + float(rng.uniform(0.0, 86_400.0))
    pkts = GENERATORS[name](rng, difficulty, t0)
    return flow_record(
        pkts,
        flow_id=f"synth-{i:07d}",
        src_ip=_random_ip(rng),
        dst_ip=_random_ip(rng),
        src_port=int(rng.integers(1024, 65536)),
    )


def _random_ip(rng: np.random.Generator) -> str:
    """A host address from one of the documentation ranges, for example 198.51.100.7."""
    return f"{_DOC_NETS[int(rng.integers(len(_DOC_NETS)))]}{int(rng.integers(1, 255))}"


def _noisy_labels(
    classes: np.ndarray, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Flip a share `noise` of labels to a class of the other group.

    Input: classes (n,) true class names. Output: label (n,) str and is_attack (n,) int64,
    always consistent with each other (is_attack = 1 exactly for attack class names).
    """
    is_attack = np.isin(classes, ATTACK_CLASSES).astype(np.int64)  # (n,)
    flip = rng.random(classes.size) < noise  # (n,) bool
    pick = rng.integers(0, 3, classes.size)  # (n,) index of the replacement class
    replacement = np.where(
        is_attack == 1, np.array(NORMAL_CLASSES)[pick], np.array(ATTACK_CLASSES)[pick]
    )
    label = np.where(flip, replacement, classes)
    return label, np.where(flip, 1 - is_attack, is_attack)


# ---------- Command line ----------


def main(argv: list[str] | None = None) -> None:
    """Write a synthetic flows table to CSV and print the class counts."""
    parser = argparse.ArgumentParser(description="Generate a SYNTHETIC flows table (CSV).")
    parser.add_argument("--n", type=int, default=20_000, help="number of flows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attack-fraction", type=float, default=0.2)
    parser.add_argument("--difficulty", type=float, default=0.5, help="0 (easy) to 1 (hard)")
    parser.add_argument("--out", type=Path, default=Path("data/processed/synthetic.csv"))
    args = parser.parse_args(argv)

    df, prov = make_dataset_with_provenance(
        args.n, args.attack_fraction, args.difficulty, args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(prov, indent=2) + "\n")

    print(f"SYNTHETIC flows (they test code, not detection): {len(df)} rows -> {args.out}")
    print(f"provenance -> {json_path}")
    print(f"difficulty={args.difficulty} seed={args.seed} label_noise={prov['label_noise']:g}")
    print(f"  {'class':<15} {'group':<7} {'true':>7} {'labeled':>8}")
    for name in NORMAL_CLASSES + ATTACK_CLASSES:
        group = "attack" if name in ATTACK_CLASSES else "normal"
        true, labeled = prov["class_counts_true"][name], prov["class_counts_labeled"][name]
        print(f"  {name:<15} {group:<7} {true:>7} {labeled:>8}")
    print(f"  flipped normal -> attack: {prov['flipped_normal_to_attack']}")
    print(f"  flipped attack -> normal: {prov['flipped_attack_to_normal']}")
    scores = prov["perfect_model_vs_noisy_labels"]
    print("A perfect model (always the true class), scored against the noisy labels:")
    for key in ("attack_recall", "attack_precision"):
        value = scores[key]
        print(f"  {key:<17} {'n/a' if value is None else f'{value:.4f}'}")


if __name__ == "__main__":
    main()
