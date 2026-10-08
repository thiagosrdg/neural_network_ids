"""Tests for `neural_ids.synthetic`: generators, `make_dataset`, and the command.

All data here is synthetic: these tests check that the code follows the contract and that each
class has the signature it was written to have, not that anything detects real attacks.
"""

import json

import numpy as np
import pandas as pd
import pytest

from neural_ids.features import (
    ACK,
    FIN,
    PROTO_TCP,
    RST,
    SYN,
    FlowPackets,
    compute_flow_features,
    effective_times,
)
from neural_ids.schema import ACTIVE_TIMEOUT_S, IDLE_TIMEOUT_S, SCHEMA_VERSION, validate_flows
from neural_ids.synthetic import (
    ATTACK_CLASSES,
    GENERATORS,
    NORMAL_CLASSES,
    SSH_MAX_AUTH_TRIES,
    TCP_HDR,
    main,
    make_dataset,
    make_dataset_with_provenance,
    perfect_model_scores,
)
from neural_ids.utils import file_sha256, set_seed

DIFFICULTIES = (0.0, 0.5, 1.0)


def generate(name: str, difficulty: float, n: int = 60, seed: int = 0) -> list[FlowPackets]:
    rng = set_seed(seed)
    return [GENERATORS[name](rng, difficulty) for _ in range(n)]


def contract_end_index(p: FlowPackets) -> int:
    """Index of the packet after which the contract ends this flow (or n - 1 if none does).

    A plain re-implementation of the flow-end rules in `docs/features.md`, section 2, so the
    test does not reuse generator code: idle timeout, active timeout (the flow ends BEFORE
    packet i, so the end is i - 1), RST, and the final ACK after the closing FIN.
    """
    t_first = t_last = float(p.timestamps[0])
    first_fin_dir = closing_fin_dir = None
    for i in range(len(p.timestamps)):
        t = float(p.timestamps[i])
        if i > 0 and (t - t_last > IDLE_TIMEOUT_S or t - t_first > ACTIVE_TIMEOUT_S):
            return i - 1
        t_last = max(t, t_last)
        flags, d = int(p.tcp_flags[i]), int(p.direction[i])
        if p.protocol != PROTO_TCP:
            continue
        if flags & RST:
            return i
        if closing_fin_dir is not None and d != closing_fin_dir and flags & ACK:
            return i  # the final ACK
        if flags & FIN:
            if first_fin_dir is None:
                first_fin_dir = d
            elif d != first_fin_dir and closing_fin_dir is None:
                closing_fin_dir = d  # the closing FIN (never the final ACK itself)
    return len(p.timestamps) - 1


# ---------- Every generated flow follows the contract ----------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("name", NORMAL_CLASSES + ATTACK_CLASSES)
def test_generated_flows_pass_validate_flows(name: str, difficulty: float) -> None:
    rows = [compute_flow_features(p) for p in generate(name, difficulty)]
    df = pd.DataFrame(rows)
    meta = {
        "flow_id": [str(i) for i in range(len(df))],
        "src_ip": "192.0.2.1",
        "dst_ip": "198.51.100.1",
        "src_port": 50000,
        "dst_port": 0,
        "protocol": 6,
        "start_time": 0.0,
        "end_time": 0.0,
    }
    validate_flows(pd.concat([pd.DataFrame(meta), df], axis=1))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("name", NORMAL_CLASSES + ATTACK_CLASSES)
def test_generated_packets_stay_inside_contract_limits(name: str, difficulty: float) -> None:
    for p in generate(name, difficulty):
        t = effective_times(p.timestamps)
        assert (np.diff(t) <= 110.0 + 1e-6).all()  # generator margin under the 120 s timeout
        assert t[-1] - t[0] <= 1700.0 + 1e-6
        assert ((p.ip_len >= 20) & (p.ip_len <= 1500)).all()
        if p.protocol != PROTO_TCP:
            assert (p.tcp_flags == 0).all()


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("name", NORMAL_CLASSES + ATTACK_CLASSES)
def test_last_packet_is_the_contract_end(name: str, difficulty: float) -> None:
    """Nothing comes after the packet that ends the flow (RST or final ACK).

    TCP flows must END with that packet; the one exception is an unanswered probe (a filtered
    port: one SYN, no reply), which ends by idle timeout.
    """
    for p in generate(name, difficulty):
        n = len(p.timestamps)
        assert contract_end_index(p) == n - 1
        if p.protocol == PROTO_TCP:
            last = int(p.tcp_flags[-1])
            unanswered_probe = n == 1 and last == SYN
            closed = bool(last & RST) or last == ACK
            assert closed or unanswered_probe, f"{name}: last flags {last:#04x}"


def test_contract_end_checker_catches_packets_after_rst() -> None:
    """Guards the test helper: a packet after an RST must be reported."""
    p = FlowPackets(
        timestamps=np.array([0.0, 0.1, 0.2]),
        direction=np.array([1, -1, 1]),
        ip_len=np.array([44, 40, 40]),
        tcp_flags=np.array([SYN, RST | ACK, ACK]),
        protocol=PROTO_TCP,
        dst_port=80,
    )
    assert contract_end_index(p) == 1


def tcp_flow(direction: list[int], flags: list[int]) -> FlowPackets:
    n = len(flags)
    return FlowPackets(
        timestamps=np.arange(n) * 0.1,
        direction=np.array(direction),
        ip_len=np.full(n, 52),
        tcp_flags=np.array(flags),
        protocol=PROTO_TCP,
        dst_port=80,
    )


def test_contract_end_checker_catches_packets_after_final_ack() -> None:
    # SYN, SYN+ACK, ACK, FIN+ACK (fwd), FIN+ACK (bwd, closing FIN), ACK (fwd, final), ACK.
    p = tcp_flow(
        [1, -1, 1, 1, -1, 1, -1],
        [SYN, SYN | ACK, ACK, FIN | ACK, FIN | ACK, ACK, ACK],
    )
    assert contract_end_index(p) == 5


def test_contract_end_checker_closing_fin_is_never_the_final_ack() -> None:
    # The closing FIN+ACK carries ACK but must not end the flow; the next ACK does.
    p = tcp_flow([1, -1, 1], [FIN | ACK, FIN | ACK, ACK])
    assert contract_end_index(p) == 2


def test_contract_end_checker_ignores_retransmitted_fin() -> None:
    # FIN (fwd), retransmitted FIN (fwd, not the closing FIN), ACK (bwd): no closing FIN yet,
    # so nothing ends the flow; then FIN+ACK (bwd, closing FIN) and ACK (fwd, final).
    p = tcp_flow([1, 1, -1, -1, 1], [FIN | ACK, FIN | ACK, ACK, FIN | ACK, ACK])
    assert contract_end_index(p) == 4


# ---------- Class signatures (difficulty 0) ----------


def features_of(name: str, difficulty: float = 0.0, n: int = 200) -> pd.DataFrame:
    return pd.DataFrame([compute_flow_features(p) for p in generate(name, difficulty, n)])


def test_syn_scan_is_tiny_and_short() -> None:
    f = features_of("syn_scan")
    assert (f["fwd_packets"] + f["bwd_packets"] <= 3).all()
    assert (f["duration_s"] < 0.25).all()
    assert (f["pkt_len_max"] <= 44).all()  # header-only packets, no data
    assert (f["syn_count"] >= 1).all() and (f["fin_count"] == 0).all()
    assert (f["psh_count"] == 0).all()
    # Open (SYN+ACK then RST), closed (RST+ACK), and filtered (no reply) all occur.
    assert (f["syn_count"] == 2).any()
    assert ((f["syn_count"] == 1) & (f["rst_count"] == 1)).any()
    assert (f["bwd_packets"] == 0).any()


def test_syn_scan_one_port_per_flow() -> None:
    for p in generate("syn_scan", 1.0):
        assert (p.tcp_flags & SYN).astype(bool)[p.direction == 1].sum() == 1


def test_udp_flood_is_one_sided_and_fast() -> None:
    f = features_of("udp_flood")
    assert (f["bwd_packets"] == 0).all() and (f["proto_udp"] == 1).all()
    assert (f["fwd_packets"] >= 300).all()
    assert (f["packets_per_s"] >= 500).all()
    assert (f["pkt_len_std"] == 0).all()  # identical packets at difficulty 0


def test_ssh_bruteforce_attempts_are_regular_and_short() -> None:
    f = features_of("ssh_bruteforce")
    assert (f["duration_s"] < 30).all()
    for p in generate("ssh_bruteforce", 0.0):
        t = effective_times(p.timestamps)
        replies = (p.direction == -1) & (p.ip_len == TCP_HDR + 36)  # "login failed" packets
        gaps = (t[1:] - t[:-1])[replies[1:]]  # gaps: (k,) server delay per attempt
        assert gaps.std() / gaps.mean() < 0.05  # machine timing


def test_ssh_session_is_long_and_irregular() -> None:
    f = features_of("ssh_session")
    brute = features_of("ssh_bruteforce")
    assert f["duration_s"].median() > 10 * brute["duration_s"].median()
    assert (f["iat_std_s"] > f["iat_mean_s"]).mean() > 0.9  # human typing: high variation


def test_web_completes_handshake_and_close() -> None:
    f = features_of("web")
    assert (f["syn_count"] == 2).all() and (f["fin_count"] == 2).all()
    assert (f["rst_count"] == 0).all()
    assert f["bwd_fwd_bytes_ratio"].median() > 1  # responses are larger than requests


def test_dns_is_one_query_one_answer() -> None:
    f = features_of("dns")
    assert (f["fwd_packets"] == 1).all() and (f["bwd_packets"] == 1).all()
    assert (f["proto_udp"] == 1).all() and (f["dst_port_class_well_known"] == 1).all()


def test_difficulty_creates_hard_variants() -> None:
    """At difficulty 1, some normal flows look like attacks (closed port, unanswered query)."""
    web = features_of("web", 1.0)
    assert ((web["rst_count"] == 1) & (web["bwd_packets"] == 1)).any()  # like a closed probe
    dns = features_of("dns", 1.0)
    assert (dns["bwd_packets"] == 0).any()  # like a tiny UDP flood


def test_high_ports_are_not_an_attack_shortcut_at_difficulty_1() -> None:
    """Benign flows seen mid-connection point at the client's ephemeral port."""
    df = make_dataset(3000, difficulty=1.0, seed=9, label_noise=0.0)
    normal = df[df["is_attack"] == 0]
    assert normal["dst_port_class_dynamic"].mean() > 0.01
    assert ((normal["proto_udp"] == 1) & (normal["dst_port_class_well_known"] == 0)).any()


def test_label_noise_zero_keeps_true_classes_and_same_flows() -> None:
    noisy = make_dataset(1000, difficulty=1.0, seed=10)
    clean = make_dataset(1000, difficulty=1.0, seed=10, label_noise=0.0)
    labels = ["label", "is_attack"]
    pd.testing.assert_frame_equal(noisy.drop(columns=labels), clean.drop(columns=labels))
    assert (noisy["label"] != clean["label"]).any()


# ---------- make_dataset ----------


def test_make_dataset_follows_contract_and_counts() -> None:
    df = make_dataset(1000, attack_fraction=0.3, difficulty=0.0, seed=1)
    validate_flows(df)
    assert len(df) == 1000
    assert int(df["is_attack"].sum()) == 300  # no label noise at difficulty 0
    counts = df["label"].value_counts()
    for name in ATTACK_CLASSES:
        assert counts[name] == 100
    for name in NORMAL_CLASSES:
        assert abs(counts[name] - 700 / 3) < 1


def test_label_and_is_attack_agree() -> None:
    df = make_dataset(2000, difficulty=1.0, seed=2)
    assert (df["is_attack"] == df["label"].isin(ATTACK_CLASSES).astype(int)).all()


def test_label_noise_grows_with_difficulty() -> None:
    """At difficulty 1 about 5% of labels are flipped; compare with the true classes."""
    clean = make_dataset(4000, difficulty=0.0, seed=3)
    assert clean["label"].isin(NORMAL_CLASSES + ATTACK_CLASSES).all()
    noisy = make_dataset(4000, difficulty=1.0, seed=3)
    # Flipped rows: a syn_scan signature (<= 3 header-only packets, SYN) labeled as normal.
    probe_like = (noisy["syn_count"] >= 1) & (noisy["pkt_len_max"] <= 44)
    assert (noisy.loc[probe_like, "is_attack"] == 0).any()
    share = 1 - (noisy.loc[probe_like, "is_attack"] == 1).mean()
    assert 0.01 < share < 0.12


def test_same_seed_same_data_and_different_seed_different_data() -> None:
    a = make_dataset(300, seed=7)
    pd.testing.assert_frame_equal(a, make_dataset(300, seed=7))
    assert not a.equals(make_dataset(300, seed=8))


def test_metadata_times_match_duration() -> None:
    df = make_dataset(500, seed=4)
    assert (df["duration_s"] == df["end_time"] - df["start_time"]).all()


def test_addresses_come_from_documentation_ranges() -> None:
    df = make_dataset(300, seed=5)
    ranges = ("192.0.2.", "198.51.100.", "203.0.113.")
    for col in ("src_ip", "dst_ip"):
        assert df[col].map(lambda ip: ip.startswith(ranges)).all()


def test_csv_round_trip_still_validates(tmp_path) -> None:
    path = tmp_path / "flows.csv"
    make_dataset(300, seed=6).to_csv(path, index=False)
    validate_flows(pd.read_csv(path))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_flows": 0},
        {"attack_fraction": 1.5},
        {"difficulty": -0.1},
        {"difficulty": 2.0},
        {"label_noise": 1.5},
    ],
)
def test_make_dataset_rejects_bad_arguments(kwargs: dict) -> None:
    args = {"n_flows": 10} | kwargs
    with pytest.raises(ValueError):
        make_dataset(**args)


def test_command_writes_csv_and_prints_counts(tmp_path, capsys) -> None:
    out = tmp_path / "synthetic.csv"
    main(["--n", "200", "--seed", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed and "syn_scan" in printed
    validate_flows(pd.read_csv(out))


@pytest.mark.parametrize("name", NORMAL_CLASSES + ATTACK_CLASSES)
def test_timestamps_are_whole_microseconds(name: str) -> None:
    """pcap files store microseconds by default, so T10 would read the same times."""
    for p in generate(name, 0.5, n=20):
        np.testing.assert_array_equal(p.timestamps, np.round(p.timestamps, 6))


# ---------- Short benign SSH vs brute force ----------


def test_ssh_bruteforce_overlaps_short_ssh_sessions_at_difficulty_1() -> None:
    """Short benign SSH flows land inside the brute-force region at difficulty 1 only.

    Region: the central 90% of brute-force flows in (total packets, duration).
    """

    def share_inside(difficulty: float) -> float:
        brute = features_of("ssh_bruteforce", difficulty, n=300)
        session = features_of("ssh_session", difficulty, n=300)
        packets = brute["fwd_packets"] + brute["bwd_packets"]
        p_lo, p_hi = packets.quantile([0.05, 0.95])
        d_lo, d_hi = brute["duration_s"].quantile([0.05, 0.95])
        s_packets = session["fwd_packets"] + session["bwd_packets"]
        inside = s_packets.between(p_lo, p_hi) & session["duration_s"].between(d_lo, d_hi)
        return float(inside.mean())

    assert share_inside(0.0) == 0.0
    assert share_inside(1.0) > 0.05


def test_short_ssh_mistyped_password_has_one_failed_attempt() -> None:
    """At least one generated session carries the brute-force 'login failed' reply."""
    failed = [
        p
        for p in generate("ssh_session", 1.0, n=200)
        if ((p.direction == -1) & (p.ip_len == TCP_HDR + 36)).any() and len(p.ip_len) < 40
    ]
    assert failed


# ---------- Provenance ----------


def test_perfect_model_scores_by_hand() -> None:
    # 10 true attacks; 1 labeled normal, 3 normals labeled attack.
    # TP = 9; labeled attacks = 9 + 3 = 12 → recall 0.75; predicted 10 → precision 0.9.
    assert perfect_model_scores(10, 1, 3) == {"attack_recall": 0.75, "attack_precision": 0.9}
    assert perfect_model_scores(0, 0, 0) == {"attack_recall": None, "attack_precision": None}


def test_provenance_matches_the_table() -> None:
    df, prov = make_dataset_with_provenance(2000, difficulty=1.0, seed=11)
    assert prov["n_flows"] == 2000 and prov["seed"] == 11
    assert prov["label_noise"] == 0.05 and prov["schema_version"] == SCHEMA_VERSION
    labeled = df["label"].value_counts()
    for name, count in prov["class_counts_labeled"].items():
        assert labeled.get(name, 0) == count
    assert sum(prov["class_counts_true"].values()) == 2000
    true_attacks = sum(prov["class_counts_true"][name] for name in ATTACK_CLASSES)
    assert true_attacks == 400
    net = prov["flipped_normal_to_attack"] - prov["flipped_attack_to_normal"]
    assert int(df["is_attack"].sum()) == true_attacks + net
    assert prov["flipped_normal_to_attack"] > 0 and prov["flipped_attack_to_normal"] > 0
    flipped = set(prov["flipped_flow_ids"])
    assert len(flipped) == prov["flipped_normal_to_attack"] + prov["flipped_attack_to_normal"]
    assert flipped <= set(df["flow_id"])


def test_no_label_noise_gives_perfect_scores() -> None:
    _, prov = make_dataset_with_provenance(500, difficulty=0.0, seed=12)
    assert prov["flipped_normal_to_attack"] == prov["flipped_attack_to_normal"] == 0
    assert prov["perfect_model_vs_noisy_labels"] == {
        "attack_recall": 1.0,
        "attack_precision": 1.0,
    }


def test_command_writes_provenance_json(tmp_path, capsys) -> None:
    out = tmp_path / "synthetic.csv"
    main(["--n", "300", "--seed", "2", "--difficulty", "1", "--out", str(out)])
    prov = json.loads(out.with_suffix(".json").read_text())
    assert prov["n_flows"] == 300 and prov["difficulty"] == 1.0
    assert set(prov["perfect_model_vs_noisy_labels"]) == {"attack_recall", "attack_precision"}
    assert "attack_recall" in capsys.readouterr().out
    assert prov["csv_sha256"] == file_sha256(out)  # fingerprint of the CSV just written


# ---------- Artifact fixes: closer, failing script, spoofed flood ----------


SETUP_PACKETS = 11  # 3 handshake + 8 key-exchange packets; an 88-byte packet can occur there


def failures(p: FlowPackets) -> int:
    """'login failed' replies: 88-byte backward packets after the handshake and key exchange.

    (In ssh_session, keystroke echoes have the same size; the tests below only compare
    brute force and one-failure flows, where this does not matter.)
    """
    after = slice(SETUP_PACKETS, None)
    return int(((p.direction[after] == -1) & (p.ip_len[after] == TCP_HDR + 36)).sum())


def test_bruteforce_server_closes_only_at_max_auth_tries() -> None:
    """sshd hangs up after 6 failures; with fewer attempts the tool (forward) closes."""
    for seed in range(5):  # several seeds: the rule must not depend on one
        for p in generate("ssh_bruteforce", 1.0, n=200, seed=seed):
            first_fin = int(np.flatnonzero(p.tcp_flags & FIN)[0])
            server_closed = p.direction[first_fin] == -1
            assert server_closed == (failures(p) == SSH_MAX_AUTH_TRIES)


def test_packet_balance_does_not_reveal_the_class() -> None:
    """fwd_packets - bwd_packets takes the same values (0 and 2) in both SSH classes."""
    for name in ("ssh_bruteforce", "ssh_session"):
        f = features_of(name, 1.0, n=300)
        diff = set(f["fwd_packets"] - f["bwd_packets"])
        assert {0, 2} <= diff, name


def test_failing_script_matches_one_bruteforce_attempt() -> None:
    """A benign one-failure flow has the same packet structure as a 1-attempt brute force."""

    def one_failure(p: FlowPackets) -> bool:
        return failures(p) == 1

    def shape(p: FlowPackets) -> tuple:
        return (tuple(p.direction), tuple(p.tcp_flags))

    brute = {shape(p) for p in generate("ssh_bruteforce", 1.0, n=300) if one_failure(p)}
    benign = {shape(p) for p in generate("ssh_session", 1.0, n=300) if one_failure(p)}
    assert brute & benign  # at least one identical direction-and-flag sequence


def test_spoofed_flood_packets_are_single_packet_flows() -> None:
    f = features_of("udp_flood", 1.0, n=300)
    single = f["fwd_packets"] == 1
    assert single.mean() > 0.4  # common at difficulty 1
    assert (f.loc[single, "bwd_packets"] == 0).all()
    assert (features_of("udp_flood", 0.0)["fwd_packets"] >= 300).all()  # none at difficulty 0


def test_dns_retries_are_not_perfectly_regular() -> None:
    """A real retry timer varies by milliseconds; exact 5.000 s gaps would be an artifact."""
    f = features_of("dns", 1.0, n=400)
    retries = f[f["fwd_packets"] >= 3]
    assert len(retries) > 0 and (retries["iat_std_s"] > 0).all()


def test_two_packet_udp_flows_exist_in_both_classes() -> None:
    for name in ("dns", "udp_flood"):
        f = features_of(name, 1.0, n=600)
        assert (f["fwd_packets"] + f["bwd_packets"] == 2).any(), name


def test_regular_floods_stay_common_at_difficulty_1() -> None:
    f = features_of("udp_flood", 1.0, n=600)
    multi = f[f["fwd_packets"] >= 10]
    assert ((multi["iat_std_s"] / multi["iat_mean_s"]) < 0.1).mean() > 0.05
