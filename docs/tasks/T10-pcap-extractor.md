# T10 — From capture files to flows

Phase: 1 (Build) · Depends on: T09

## Goal
Read Wireshark and tcpdump captures (pcap and pcapng), group packets into bidirectional flows
as defined in `docs/features.md`, and compute features with `compute_flow_features`. This is
the bridge from any person's capture to the networks.

## Why it matters
- ML: this is where training–serving consistency becomes real. The extractor feeds the same
  feature function as the synthetic data, so a model trained today can score tomorrow's capture.
- Security: you implement the flow logic that tools like Zeek and CICFlowMeter provide, and it
  is harder than it looks. A study of CIC-IDS2017 found that CICFlowMeter ended a flow at the
  first FIN packet, so the rest of each TCP connection became separate tiny flows ("TCP
  appendices", 25.9% of the dataset), and that it did not treat RST as the end of a flow. Our
  tests must prove we do not repeat these mistakes.

## Deliverables
- Scapy added as a dependency (explain why; note that it is GPL-2.0 licensed).
- `src/neural_ids/extract.py`:
  - Streams packets with Scapy's `PcapReader` (constant memory). It reads pcap and pcapng
    (checked with Scapy 2.7), the Ethernet, Linux cooked (`tcpdump -i any`), and raw IP link
    types, and both IPv4 and IPv6.
  - Keeps the reader behind a small interface (an iterator of parsed packet records: time,
    addresses, ports, protocol, IP length, TCP flags), so a faster library such as dpkt can
    replace Scapy for large public datasets in phase 2 without touching the flow logic.
  - `flow_key(...)`: the same key for both directions of a conversation.
  - A flow table with idle and active timeouts and TCP termination (RST, or both FINs plus the
    final ACK) that builds `FlowPackets` and calls `compute_flow_features`.
  - Counts of skipped packets by reason (non-IP, fragments, malformed) in a summary.
  - Command `uv run python -m neural_ids.extract <capture> --out <flows.csv> [--anonymize]`: it
    prints the summary (packets, flows, skipped packets, counts per protocol) and never packet
    contents. `--anonymize` replaces IP addresses with keyed hashes (HMAC-SHA256, with the key
    read from an environment variable).
- `tests/test_extract.py`: build captures in a temporary folder with Scapy (`wrpcap`,
  documentation IP ranges, explicit MAC addresses, nothing sent) and check exact results:
  - a full TCP connection (handshake, data, FIN close, final ACK) → 1 flow;
  - a SYN scan of 5 ports → 5 flows;
  - a DNS query and response → 1 flow;
  - an idle gap longer than the timeout → 2 flows;
  - an IPv6 connection, a Linux cooked capture, and a pcapng file.
- `tests/conftest.py`: an automatic fixture that replaces Scapy's sending functions (`send`,
  `sendp`, `sr`, `sr1`, `srp`, `srp1`) with functions that raise, so any accidental network
  activity during the tests fails them.
- `tests/test_policy.py`: fails if the source code in `src/` calls Scapy's sending or sniffing
  functions.
- A cross-check in the report: on one crafted capture, compare our flow count with
  `tshark -r <file> -q -z conv,tcp` and `-z conv,udp` (ask me before running tshark).

## TODO(human)
- `flow_key`: both directions must give the same key. Think about how to order the two endpoints.
- The TCP termination rule.

## Acceptance criteria
- All tests pass, including the exact flow counts above.
- Extraction streams in constant memory for packet data, and its time is O(P) for P packets.
- The output passes `validate_flows`, and no payload bytes appear anywhere.

## Out of scope
Labeled datasets from folders of captures (T11), scoring (T12), host-window features (phase 2).

## Learning check
1. Why does the idle timeout matter more for UDP than for TCP?
2. Where does a capture file record its link type, and why does the extractor need it?
3. How much memory does the flow table use, and how do the timeouts keep it bounded?
