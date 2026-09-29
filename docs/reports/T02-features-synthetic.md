# T02 — Flow features and synthetic traffic · Report

- Date: 2026-09-29
- Status: done
- Learning mode: guided (Claude wrote all code, including the spec's TODO(human) pieces)
- Commit: not committed yet

## Summary
`compute_flow_features` turns the packets of one flow (`FlowPackets`) into the 29 contract
features; synthetic data and T10 captures share it. Six packet-level generators (3 normal,
3 attack) and `make_dataset` produce a SYNTHETIC flows table that passes `validate_flows`.

## What was built
- `src/neural_ids/features.py` — `FlowPackets` (checks at creation), `effective_times`, `compute_flow_features`, `flow_record`
- `src/neural_ids/synthetic.py` — `web`, `dns`, `ssh_session`, `syn_scan`, `ssh_bruteforce`, `udp_flood`; `make_dataset`; CLI
- `tests/test_features.py`, `tests/test_synthetic.py` — hand-computed examples, edge cases, contract-end checker, signatures
- `explore/synthetic_look.py` — 4 figures in `reports/figures/synthetic_*.png`
- `docs/features.md`, `schema.py` docstring, `docs/ARCHITECTURE.md` — review notes 7a–c, `is_attack` rule, `FlowPackets` order
- Dependencies added: none

## My part (TODO(human))
- IAT and rate features with edge cases — done by Claude (guided mode)
- `syn_scan` generator — done by Claude (guided mode)

## Verification (real output)
- Data used: synthetic only (documentation IP ranges)
- Tests: 177 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`; `36 files already formatted`
- Main command: `uv run python -m neural_ids.synthetic --n 20000 --seed 42` → 20000 rows;
  web 5240, dns 5243, ssh_session 5247, syn_scan 1418, ssh_bruteforce 1437, udp_flood 1415;
  is_attack=1: 4270 (labels after 2.5% noise at difficulty 0.5); the CSV passes `validate_flows`
- Acceptance criteria: all pass — tests incl. hand-computed examples; figures separate at
  difficulty 0 and overlap at 1 (true class, no label noise); code and `features.md` agree
- `ml-reviewer`: 0 must-fix, 6 should-fix; all fixed (see below)

## Key concepts
- Effective time is a cumulative maximum: backwards timestamps give gap 0, never sorting.
- Rates and IAT use explicit edge cases (0 packets of gap, 0 duration) instead of inf/NaN.
- nmap -sS: open = SYN, SYN/ACK, RST (kernel reset); closed = RST/ACK; filtered = one SYN.
- Label noise and class overlap are different things; plots must show the true class.
- A feature available only to one group (high ports) is a shortcut, even in synthetic data.

## Decisions and deviations from the spec
- Approved notes 1–7 and plan changes 1–5 (file order, Python int/float, header flag bits,
  difficulty on packets, one flow per call, filtered = one SYN, `direction[0] == +1`,
  metadata from effective times, TCP flows end at RST or final ACK).
- `flow_record` (metadata + features) lives in `features.py` so T10 can reuse it.
- After review: benign mid-connection web flows and response-only DNS use the client's
  ephemeral port at higher difficulty (removes the "high port = attack" shortcut);
  `make_dataset(label_noise=...)` so plots use true classes; timestamps rounded to µs;
  `FlowPackets` rejects float/bool integer fields and scalar timestamps; more guard tests.

## Problems and open questions
- At difficulty 0, high ports still occur only in attacks (intended: easy mode).
- `ssh_bruteforce` stays a tight cluster even at difficulty 1.
- Ideas for later: ICMP "unreachable" answers to filtered probes (a separate ICMP flow);
  SYN flood and slow-DoS generators.

## Suggested next step
- T03 — Preprocessing without leakage.
