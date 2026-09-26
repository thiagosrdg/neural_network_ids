# T01 — The feature contract: from packets to flows

Phase: 1 (Build) · Depends on: T00

## Before you start (human)
Open a capture of your own traffic in Wireshark (make one with `docs/GUIDE.md`, section 4).
Open Statistics → Conversations: each row is close to what we will call a flow. Pick one TCP
conversation and one UDP conversation (for example DNS) and note their packets, bytes, and
duration. You do not need to share the file.

## Goal
Define exactly what the neural network will see: which numbers describe one flow, how each is
computed from packet headers, and how edge cases are handled. This contract is the interface
between any capture and any model.

## Why it matters
- ML: a model only works on inputs computed exactly as they were in training. A precise,
  versioned contract prevents training–serving skew (features computed one way for training
  and another way in use), which breaks models without any error message.
- Security: header and timing features work on encrypted traffic and never need payloads.
  Leaving identifiers out stops the model from memorizing "this IP address was the attacker in
  my lab" instead of learning behavior. A study of the CIC-IDS2017 dataset found exactly this
  problem: timestamps and IP addresses let models take shortcuts.

## Deliverables
- `docs/features.md`, with:
  - Flow definition: bidirectional flows keyed by the 5-tuple; "forward" is the direction of the
    first packet seen. A TCP flow ends on RST, or after both sides have sent FIN and the final
    ACK arrived (the closing handshake stays inside the flow). Any flow also ends after an idle
    timeout (default 120 s) or an active timeout (default 1800 s). ICMP flows use port 0.
    IPv4 and IPv6 are both supported.
  - Metadata columns (never model inputs): `flow_id`, `src_ip`, `dst_ip`, `src_port`,
    `dst_port`, `protocol`, `start_time`, `end_time`.
  - A feature table (name, definition, unit, edge cases, and why it may reveal attacks),
    starting from this v1 list:
    - `proto_tcp`, `proto_udp`, `proto_icmp`, `proto_other` (one-hot)
    - `duration_s`
    - `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes` — bytes are IP-layer lengths:
      the IPv4 total length, or the IPv6 payload length + 40. Never the frame length.
    - `pkt_len_mean`, `pkt_len_std`, `pkt_len_min`, `pkt_len_max`
    - `iat_mean_s`, `iat_std_s`, `iat_max_s` (inter-arrival time: the time between two
      consecutive packets of the flow)
    - `syn_count`, `ack_count`, `fin_count`, `rst_count`, `psh_count`, `urg_count` (0 for non-TCP)
    - `bytes_per_s`, `packets_per_s`
    - `bwd_fwd_bytes_ratio`
    - `dst_port_class` as one-hot: `well_known` (0–1023), `registered` (1024–49151),
      `dynamic` (49152–65535), `none` (ICMP)
  - Explicit rules for the edge cases: one-packet flows, zero duration, flows with no backward
    packets, non-IP packets, and IP fragments.
  - A "Future features" section: host-window features (for example, how many different ports
    one source contacted in the last few seconds) and the sizes and directions of the first N
    packets of a flow.
- `src/neural_ids/schema.py`: `SCHEMA_VERSION = "1.0"`, the flow timeouts,
  `METADATA_COLUMNS`, `FEATURES` (names in a fixed order, each with dtype and allowed range),
  and `validate_flows(df) -> None`, which raises a clear error for missing, extra,
  wrongly typed, or out-of-range columns.
- `tests/test_schema.py`: a valid table passes; each kind of problem raises a clear error.

## TODO(human)
- The "why it may reveal attacks" column for at least 10 features, using what you know from
  Network+ and Nmap. For example: what does an `nmap -sS` scan look like in `syn_count`,
  `rst_count`, and `bwd_packets`?
- `validate_flows`.

## Acceptance criteria
- `docs/features.md` defines every feature precisely enough that two people would compute the
  same number from the same packets.
- Tests pass; `validate_flows` raises a clear error for each kind of problem.
- No identifier (IP, MAC, exact port, timestamp, flow ID) is in `FEATURES`.

## Out of scope
Code that computes features (T02) or reads captures (T10).

## Learning check
1. Why is one bidirectional flow more useful than two one-way flows for detection?
2. Why measure sizes at the IP layer and not the frame size? Think of Ethernet, Wi-Fi, and
   `tcpdump -i any` captures.
3. Should the exact destination port be a feature? What could the model learn by mistake?
