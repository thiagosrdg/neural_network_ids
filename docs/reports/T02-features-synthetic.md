# T02 — Flow features and synthetic traffic · Report

- Date: 2026-10-01
- Status: done
- Learning mode: guided (Claude wrote all code, including the spec's TODO(human) pieces)
- Commit: see `git log` (3 commits: `feat: add shared flow features…`, `feat: add short benign SSH…`, `fix: remove class artifacts…`)

## Summary
`compute_flow_features` turns the packets of one flow (`FlowPackets`) into the 29 contract
features; synthetic data and T10 captures share it. Six packet-level generators and
`make_dataset` produce a SYNTHETIC table (passes `validate_flows`) with provenance JSON.

## What was built
- `src/neural_ids/features.py` — `FlowPackets` (checks at creation), `effective_times`, `compute_flow_features`, `flow_record`
- `src/neural_ids/synthetic.py` — 6 generators, `make_dataset(_with_provenance)`, `perfect_model_scores`, CLI (CSV + JSON)
- `tests/test_features.py`, `tests/test_synthetic.py` — hand-computed examples, edge cases, flow ends, signatures, artifacts
- `explore/synthetic_look.py` (4 figures, true class) and `explore/shortcut_check.py` (random forest per pair)
- `docs/features.md`, `schema.py` docstring, `docs/ARCHITECTURE.md` — notes 7a–c, `is_attack` rule, `FlowPackets` order, SSH limitation
- Dependencies added: none

## My part (TODO(human))
- IAT and rate features; `syn_scan` generator — done by Claude (guided mode)

## Verification (real output)
- Data used: synthetic only (documentation IP ranges)
- Tests: 190 passed — `uv run pytest -q`; lint and format: `All checks passed!`, `38 files already formatted`
- Main command (`--n 20000 --seed 42`, difficulty 0.5, label_noise 0.025): true attacks 4000,
  labeled 4284; flipped normal → attack 392, attack → normal 108 (500 `flipped_flow_ids`).
  Perfect model vs noisy labels: attack recall 0.9085, precision 0.9730. CSV passes `validate_flows`.
- Shortcut check (`explore/shortcut_check.py`; difficulty 1, no label noise, 2000/class, 5-fold RF):

| Pair | Accuracy (std) | Top 3 features (importance) |
|---|---|---|
| ssh_bruteforce vs ssh_session | 0.925 (0.011) | fwd_packets 0.14, iat_max_s 0.13, ack_count 0.13 |
| syn_scan vs web | 0.899 (0.008) | pkt_len_std 0.15, pkt_len_mean 0.10, fin_count 0.07 |
| udp_flood vs dns | 0.914 (0.010) | fwd_bytes 0.18, pkt_len_mean 0.16, pkt_len_max 0.15 |
| hard subsets: SSH ≤ 20 pkts / scan ≤ 3 pkts / UDP no reply | 0.775 / 0.833 / 0.886 | timing; bytes_per_s, IAT; fwd_bytes |

- Target < 0.95. Before (user's check): 0.996, 0.902, 1.000. Exactly one failed SSH login (16 packets): 0.576 (majority 0.560),
  from `explore/shortcut_check.py` (corrected in T03; the earlier 0.508 came from a script not in the repo).
- Acceptance criteria: all pass. `ml-reviewer`: round 1, 6 should-fix; round 2, 5 should-fix; all fixed.

## Key concepts
- 2-D plots can overlap while the 29-D data separates; measure with a strong nonlinear model.
- An artifact is a rule that one class has and real traffic does not (closer side, RTT range).
- Label noise caps the score; clearly beating both ceilings on held-out data signals leakage.

## Decisions and deviations from the spec
- Approved notes 1–7, plan changes 1–5, additions (short SSH flows, provenance JSON).
- Artifacts removed: closer follows sshd MaxAuthTries (6) or client exit; shared RTT, program
  wait, send jitter, password size, and sshd delay; failing-script SSH flow; spoofed one-packet
  and 2+ packet floods; per-flow flood jitter; DNS retry timer jitter (no exact 5.000 s gaps).
- Remaining separation is real per-flow behavior: 2–6 failed logins, scans with no data or FIN.

## Problems and open questions
- At difficulty 0, high ports still occur only in attacks (intended: easy mode).
- `flipped_flow_ids` is an answer key (now in ARCHITECTURE.md): ceilings only, never training.
- Later: ICMP unreachable for filtered probes; SYN flood, slow-DoS generators; host windows.

## Suggested next step
- T03 — Preprocessing without leakage.
