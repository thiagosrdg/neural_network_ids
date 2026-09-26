# T02 — Flow features and synthetic traffic

Phase: 1 (Build) · Depends on: T01

## Goal
Write the one function that turns the packets of a flow into feature values, and a synthetic
traffic generator that produces packet lists for normal and attack flows. Together they give
a dataset that follows the contract, so we can build and test the networks before we have
real captures.

## Why it matters
- ML: one shared feature function guarantees that synthetic data, public datasets, and anyone's
  captures produce comparable numbers.
- Security: to fake an attack, you must know what it looks like on the wire. Writing the
  generator is a hands-on review of TCP handshakes, scans, brute force, and floods.

## Deliverables
- `src/neural_ids/features.py`:
  - `FlowPackets`: a small dataclass for one flow, with NumPy arrays in time order —
    `timestamps` (seconds), `direction` (+1 forward, −1 backward), `ip_len` (bytes), and
    `tcp_flags` (flag bits; 0 for non-TCP) — plus `protocol` and `dst_port`.
  - `compute_flow_features(pkts: FlowPackets) -> dict[str, float]`: every feature in
    `schema.FEATURES`, vectorized with NumPy, following the edge-case rules in `docs/features.md`.
- `src/neural_ids/synthetic.py`:
  - One packet-level generator per traffic class, each returning `FlowPackets`:
    - normal: `web` (TCP handshake, request, larger response, close), `dns` (one UDP query and
      one response), `ssh_session` (a long interactive session with irregular timing);
    - attack: `syn_scan` (TCP SYN to many ports; open ports answer SYN/ACK, closed ports answer
      RST/ACK), `ssh_bruteforce` (short, very regular login attempts), `udp_flood` (many
      forward packets at a high rate, no answers).
  - `make_dataset(n_flows, attack_fraction=0.2, difficulty=0.5, seed=...) -> pd.DataFrame`:
    metadata columns (addresses from documentation ranges), every feature, `label` (the class
    name), and `is_attack` (0 or 1). `difficulty` (0 to 1) adds overlap between classes and
    some label noise.
  - Command `uv run python -m neural_ids.synthetic --n 20000 --seed 42` → writes
    `data/processed/synthetic.csv` and prints the class counts. The table passes `validate_flows`.
- `tests/test_features.py`: hand-computed examples with exact expected values (a 3-way
  handshake with a small exchange; a single SYN; a DNS query and response) and the edge cases.
- `tests/test_synthetic.py`: every generated flow passes `validate_flows`; the same seed gives
  the same data; each class has its expected signature (for example, `syn_scan` flows are tiny
  and very short).
- `explore/synthetic_look.py`: 3–4 plots comparing the classes, saved to `reports/figures/`.

## TODO(human)
- The inter-arrival-time features and the rate features in `compute_flow_features`, with their
  edge cases.
- The `syn_scan` generator. Use what you know about `nmap -sS`: which packets travel for an
  open port, a closed port, and a filtered port?

## Acceptance criteria
- All tests pass, including the hand-computed examples.
- With `difficulty=0` the classes separate clearly in the plots; with `difficulty=1` they
  visibly overlap.
- The code and `docs/features.md` agree on every definition.

## Out of scope
Reading real captures (T10), models.

## Learning check
1. Why must synthetic data, public datasets, and user captures all go through
   `compute_flow_features`?
2. From one flow alone, can you tell an SSH brute-force attempt from a normal failed login?
   What extra information would help?
3. Why does good performance on synthetic data prove nothing about real networks?
