# Feature contract

Schema version: 1.0

This document defines exactly what the models see: which numbers describe one flow and how
each one is computed from packet headers. `src/neural_ids/schema.py` holds the same contract in
code (names, order, dtypes, ranges) and `validate_flows` enforces it. Every feature is computed
by `neural_ids.features` (T02), for synthetic data, public datasets, and user captures alike.

Any change to this document or to `schema.py` that changes a feature value needs a new
`SCHEMA_VERSION`. That includes the timeouts.

## 1. Packets

**Which packets count.** Packets are read in file order. A packet enters a flow only if it is an
IPv4 or IPv6 packet that is not a fragment and has a valid IP length. Everything else is
skipped and counted by reason in the extractor's summary (T10):

| Skip reason | Rule |
|---|---|
| non-IP | The frame carries no IPv4 or IPv6 packet (for example ARP, STP, LLDP). |
| fragment | IPv4: the MF (more fragments) flag is set, or the fragment offset is > 0. IPv6: the packet has a Fragment extension header. **Every** fragment is skipped, including the first one. |
| malformed | IP length outside 20–65575 bytes, an IPv6 jumbogram (payload length 0 with a Jumbo Payload option), or headers too short to read the fields below. |

Why the first fragment is skipped too: it can hold only part of the transport header. `nmap -f`
sends 8-byte fragments, so the TCP flags byte (offset 13) is not in the first fragment and
cannot be trusted. Reassembly is listed under Future features. **Limitation:** traffic that is
fragmented on purpose is invisible to v1 flows, except for the fragment count in the summary.

**IP length** (`ip_len`, bytes), the size used by every byte feature:
- IPv4: the Total Length header field.
- IPv6: the Payload Length header field + 40 (the fixed header).
- Never the frame length or the captured length. Link layers differ between captures
  (Ethernet, Wi-Fi, `tcpdump -i any`). The header field also stays correct when the capture
  truncated packets with a snap length (`tcpdump -s`).

**Protocol.** IPv4: the Protocol field. IPv6: the Next Header value after skipping the
Hop-by-Hop, Routing, and Destination Options extension headers. 6 = TCP, 17 = UDP, 1 = ICMP,
58 = ICMPv6. Anything else is "other".

**Ports.** TCP and UDP: the header ports. ICMP, ICMPv6, and other protocols: port 0 for both
sides.

**Timestamps.** Every time is a float64 number of Unix seconds, as read from the capture.
Gaps, timeouts, and time features are computed from these float64 values. (float64 resolves
about 0.24 µs at current Unix times, so nanosecond pcapng timestamps lose their last digits;
this rule makes every implementation lose them the same way.)

Packets are processed in file order, not sorted by time. Each packet gets an
effective time `t_eff = max(t, t_last)`, where `t_last` is the effective time of the previous
packet in the same flow. A timestamp that goes backwards (possible in `tcpdump -i any`
captures, which merge several interfaces) therefore gives a gap of 0. No gap or duration is
ever negative. All time features use effective times.

## 2. Flows

**Key.** A flow is bidirectional and keyed by the 5-tuple (source IP, source port, destination
IP, destination port, protocol). Packets A→B and B→A belong to the same flow. ICMP and ICMPv6
use port 0, so all ICMP traffic between two hosts is one flow (types and IDs are ignored in v1).
IPv4 and IPv6 are both supported.

**Direction.** "Forward" is the direction of the first packet seen for the flow; "backward" is
the other one. **Limitation:** for a flow that was already open when the capture started, the
first packet seen may come from the server, so forward and backward are swapped compared with
"client → server".

**Flow end.** A packet ends its flow, or starts a new one, by these rules, checked in this order:
1. **Idle timeout** (`IDLE_TIMEOUT_S` = 120 s): if `t - t_last > 120`, the old flow ends
   *before* this packet, and the packet starts a new flow. A gap of exactly 120 s stays in the
   flow.
2. **Active timeout** (`ACTIVE_TIMEOUT_S` = 1800 s): if `t - t_first > 1800` (where `t_first`
   is the flow's first packet), the old flow ends before this packet, and the packet starts a
   new flow.
3. **TCP RST:** a packet with RST set ends the flow *after* being added to it.
4. **TCP close:** call the *closing FIN* the first packet with FIN set from the second
   direction to send a FIN (retransmitted FINs from the first direction do not count). The
   *final ACK* is the first packet **after** the closing FIN that has ACK set and comes from
   the other direction. The closing FIN itself is never the final ACK, even when it also
   carries ACK (as in a simultaneous close). The final ACK is added to the flow, and the flow
   ends. The whole closing handshake stays inside the flow. (Ending at the first FIN creates
   fake extra flows; see `ARCHITECTURE.md`.)

A later packet with the same 5-tuple starts a new flow. **Limitation:** a retransmitted FIN or
ACK that arrives after the close becomes its own small flow.

Consequences used as ranges in `schema.py`: `duration_s` ≤ 1800 and every gap ≤ 120.

## 3. Metadata columns (never model inputs)

| Column | Meaning |
|---|---|
| `flow_id` | Unique ID of the flow within the table (string). |
| `src_ip`, `dst_ip` | Addresses of the forward direction (source = sender of the first packet). |
| `src_port`, `dst_port` | Ports of the forward direction; 0 when the protocol has no ports. |
| `protocol` | IP protocol number (6, 17, 1, 58, …). |
| `start_time`, `end_time` | Effective times of the first and last packets (Unix seconds). |

Optional columns, allowed but not required: `label` (class name), `is_attack` (0 or 1), and
`capture_id` (which capture the flow came from). They are also never model inputs: a label used
as an input is target leakage. `is_attack` alone decides benign (0) versus attack (1); `label`
can name a benign subclass (`web`, `dns`, `ssh_session` in the synthetic data; `normal` in
folder datasets). Later tasks select normal flows with `is_attack == 0`, never by label name.

Identifiers are metadata only. With IP addresses, exact ports, timestamps, or flow IDs as
inputs, a model learns shortcuts ("this IP was the attacker in my lab") instead of behavior.

## 4. Features (29, in this order)

Notation: a flow has `n` packets with IP lengths `L_1..L_n` and effective times `t_1..t_n` in
processing order (both directions together). `n_f` and `n_b` are the forward and backward
packet counts. Gaps are `g_i = t_(i+1) - t_i` for i = 1..n-1.

**Standard deviation** always uses the population formula (ddof=0):
`std = sqrt(mean((x - mean(x))^2))`. NumPy's `np.std` uses ddof=0 by default, but pandas
`.std()` uses ddof=1 (divides by n-1), so pandas code must call `.std(ddof=0)`.

| Feature | Type | Definition | Unit | Range | Edge cases | Why it may reveal attacks |
|---|---|---|---|---|---|---|
| `proto_tcp` | int | 1 if protocol = 6, else 0 | — | 0–1 | One-hot group `proto`: exactly one of the four is 1 | Most scans and brute force use TCP; combined with the flag counts it separates a normal session from a probe. |
| `proto_udp` | int | 1 if protocol = 17, else 0 | — | 0–1 | — | UDP scans (`nmap -sU`) and amplification floods (DNS, NTP) are UDP; a UDP flow with many packets and no reply is unusual. |
| `proto_icmp` | int | 1 if protocol = 1 (ICMP) or 58 (ICMPv6), else 0 | — | 0–1 | — | Ping sweeps (`nmap -sn`) and ICMP floods; ICMP tunnels show up as ICMP flows with many or large packets. On a local Ethernet network `nmap -sn` uses ARP instead, which v1 skips as non-IP. |
| `proto_other` | int | 1 if the protocol is none of the above (for example GRE, ESP, SCTP) | — | 0–1 | — | Rare protocols (GRE, raw IP, or `nmap -sO` protocol scans) are uncommon on most LANs, so they stand out. |
| `duration_s` | float | `t_n - t_1` | s | 0–1800 | One packet: 0 | Scan probes and SYN floods last close to 0 s; slowloris-style DoS and a long-lived C2 channel keep one flow open for a long time. C2 beacons that open a new connection each time are many short flows, so one flow does not show them. |
| `fwd_packets` | int | `n_f`, packets in the forward direction | packets | ≥ 1 | Always ≥ 1 (the first packet is forward) | A probe sends 1–2 packets; brute force and floods send many from one side. |
| `bwd_packets` | int | `n_b`, packets in the backward direction | packets | ≥ 0 | No reply: 0 | A probe to a filtered port gets no reply (0); a SYN flood gets few replies compared with its forward packets. |
| `fwd_bytes` | int | Sum of `ip_len` over forward packets | bytes | ≥ 20 | — | Exfiltration and uploads send many bytes forward; scans send only headers (about 40–60 bytes per packet). |
| `bwd_bytes` | int | Sum of `ip_len` over backward packets | bytes | ≥ 0 | No reply: 0 | Large replies to tiny requests hint at amplification (DNS ANY, NTP monlist) or a big download. |
| `pkt_len_mean` | float | Mean of `L_1..L_n` | bytes | 20–65575 | One packet: `L_1` | Header-only packets (about 40–60 bytes) are typical of scans and floods; normal web traffic mixes small and large packets. |
| `pkt_len_std` | float | Population std of `L_1..L_n` | bytes | 0–32777.5 | One packet: 0 | Near 0 means identical packets, as in automated floods and scans; real sessions vary. |
| `pkt_len_min` | int | Min of `L_1..L_n` | bytes | 20–65575 | — | Very small minimums with no data packets suggest probes; in v1 fragments are skipped, so tiny `nmap -f` fragments do not reach it. |
| `pkt_len_max` | int | Max of `L_1..L_n` | bytes | 20–65575 | — | A max near the header size means no data was ever sent (probe or failed handshake); a max at the MTU means bulk transfer. |
| `iat_mean_s` | float | Mean of gaps `g_1..g_(n-1)` | s | 0–120 | One packet (no gaps): 0 | Floods have tiny gaps; slow DoS (slowloris) has long gaps inside one flow. Slow scans (`nmap -T0` waits 5 minutes between probes, `-T1` 15 seconds) put the wait between probes, and each probe is its own flow, so per-flow IAT features cannot see it; host-window features can. |
| `iat_std_s` | float | Population std of the gaps | s | 0–60 | Fewer than 2 gaps: 0 | Very regular gaps (low std) inside one flow suggest a machine: scripted brute force on one connection, or a long-lived C2 channel with a fixed heartbeat. C2 beacons that open a new connection each time spread their regular timing over many flows, where only host-window features see it. |
| `iat_max_s` | float | Max of the gaps | s | 0–120 | One packet: 0 | A long pause inside one flow is typical of slow DoS (slowloris) and of keep-alive C2 channels. |
| `syn_count` | int | Packets (both directions) with the SYN flag set | packets | ≥ 0 | Non-TCP: 0 | `nmap -sS` to a closed port: 1 SYN and 0 bytes of data; SYN floods: SYNs with no completed handshake. |
| `ack_count` | int | Packets with the ACK flag set | packets | ≥ 0 | Non-TCP: 0 | A normal TCP flow ACKs almost every packet; a SYN-only flow (ack_count 0 or 1) never completed the handshake. `nmap -sA` sends lone ACKs to map firewalls. |
| `fin_count` | int | Packets with the FIN flag set | packets | ≥ 0 | Non-TCP: 0 | `nmap -sF` and Xmas scans (`-sX`) send FIN without a prior SYN; a FIN with syn_count 0 is not normal TCP. |
| `rst_count` | int | Packets with the RST flag set | packets | ≥ 0 | Non-TCP: 0 | `nmap -sS` to a closed port: SYN answered by RST (rst_count 1, bwd_packets 1); an open port: SYN, SYN-ACK, then the scanner's RST. |
| `psh_count` | int | Packets with the PSH flag set | packets | ≥ 0 | Non-TCP: 0 | Xmas scans set PSH with FIN and URG in a packet with no data; normal PSH marks packets that carry application data. |
| `urg_count` | int | Packets with the URG flag set | packets | ≥ 0 | Non-TCP: 0 | URG is almost never used by modern applications; a non-zero count points to Xmas scans (`nmap -sX`) or crafted packets. |
| `bytes_per_s` | float | `(fwd_bytes + bwd_bytes) / duration_s` | bytes/s | ≥ 0 | `duration_s` = 0: 0 | Floods and bulk exfiltration have very high byte rates; low-and-slow attacks have very low rates. |
| `packets_per_s` | float | `n / duration_s` | packets/s | ≥ 0 | `duration_s` = 0: 0 | Packet floods (SYN, UDP, ICMP) have extreme packet rates even when bytes per packet are small. |
| `bwd_fwd_bytes_ratio` | float | `bwd_bytes / fwd_bytes` | — | ≥ 0 | No reply: 0 (`fwd_bytes` is never 0) | Much greater than 1: amplification or a big download; 0: no reply (filtered probe or flood); much less than 1: upload or exfiltration. |
| `dst_port_class_well_known` | int | 1 if the protocol is TCP or UDP and `dst_port` is 0–1023 | — | 0–1 | One-hot group `dst_port_class`; port 0 counts as well-known | Most service attacks (SSH, RDP, SMB, HTTP brute force) and default `nmap` scans target ports 0–1023. |
| `dst_port_class_registered` | int | 1 if TCP or UDP and `dst_port` is 1024–49151 | — | 0–1 | — | Databases, proxies, and many malware C2 servers use ports 1024–49151. |
| `dst_port_class_dynamic` | int | 1 if TCP or UDP and `dst_port` is 49152–65535 | — | 0–1 | — | Flows that start toward a dynamic (ephemeral) port are unusual for clients and can be back-connect shells or P2P traffic; a full port scan hits this class too. |
| `dst_port_class_none` | int | 1 if the protocol has no ports: ICMP, ICMPv6, or `proto_other` | — | 0–1 | Set by protocol, never by port value | Marks portless traffic (ICMP, other protocols), so the model does not mix ping sweeps with TCP or UDP port behavior. |

"Type" is the column dtype in the flows table: int = int64, float = float64. An integer dtype is
also accepted for float features: a CSV written by another tool or by hand can store whole
numbers without a decimal point, and pandas then reads that column as int64. (A CSV written by
pandas `to_csv` keeps `0.0`, which `read_csv` reads back as float64.)
`dst_port` is always the forward direction's destination port (the port of the first packet).

## 5. Edge-case rules (summary)

| Case | Rule |
|---|---|
| One-packet flow | `duration_s` = 0; IAT features = 0; `pkt_len_std` = 0; mean = min = max = `L_1`. |
| Zero duration (several packets, same time) | `bytes_per_s` = `packets_per_s` = 0 (dividing by a tiny duration would create huge, meaningless rates). IAT features = 0. |
| No backward packets | `bwd_packets` = `bwd_bytes` = 0; `bwd_fwd_bytes_ratio` = 0. |
| Timestamp goes backwards | Effective time `max(t, t_last)`: the gap is 0, never negative. |
| Non-IP packet | Skipped; counted in the summary as "non-IP". |
| IP fragment (any, including the first) | Skipped; counted as "fragment". |
| TCP or UDP port 0 | `dst_port_class_well_known` = 1. |
| Flow already open at capture start | "Forward" is still the direction of the first packet seen. |

`validate_flows` enforces: no duplicate column, every required column present, no unknown
column, dtypes, finite values (no NaN, +inf, or -inf), ranges, exactly one 1 in each one-hot
group per row, and row consistency: `dst_port_class_none` = 1 exactly for flows without ports,
TCP flag counts = 0 for non-TCP flows, `bwd_bytes` = 0 exactly when `bwd_packets` = 0, and
`pkt_len_min` ≤ `pkt_len_mean` ≤ `pkt_len_max`.

## 6. Future features (not in v1)

- **Host-window features:** statistics over all flows from one source in the last few seconds,
  for example how many different destination ports or hosts it contacted. Per-flow features miss
  attacks spread over many flows, such as port scans (one tiny flow per port), slow scans, and
  distributed brute force.
- **First N packets:** the sizes, directions, and gaps of the first N packets of a flow, as a
  sequence for a 1D CNN or GRU. The start of a conversation (handshake, first request) is often
  the most telling part.
- **IP reassembly:** rebuild fragmented packets so fragmented traffic (including `nmap -f`)
  becomes normal flows, and add features such as a fragment count per flow.
- **Finer ICMP flows:** key ICMP flows by type and identifier.
