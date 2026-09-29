# T02 — Flow features and synthetic traffic · Report

- Date: 2026-09-29
- Status: done
- Learning mode: guided (Claude wrote all code, including the spec's TODO(human) pieces)
- Commit: see `git log` (`feat: add shared flow features and synthetic traffic generator`, then `feat: add short benign SSH flows and synthetic provenance`)

## Summary
`compute_flow_features` turns the packets of one flow (`FlowPackets`) into the 29 contract
features; synthetic data and T10 captures share it. Six packet-level generators (3 normal,
3 attack) and `make_dataset` produce a SYNTHETIC flows table that passes `validate_flows`,
with a provenance JSON next to the CSV.

## What was built
- `src/neural_ids/features.py` — `FlowPackets` (checks at creation), `effective_times`, `compute_flow_features`, `flow_record`
- `src/neural_ids/synthetic.py` — 6 generators; `make_dataset`, `make_dataset_with_provenance`, `perfect_model_scores`; CLI (CSV + JSON)
- `tests/test_features.py`, `tests/test_synthetic.py` — hand-computed examples, edge cases, contract-end checker, signatures, provenance
- `explore/synthetic_look.py` — 4 figures in `reports/figures/synthetic_*.png` (true class)
- `docs/features.md`, `schema.py` docstring, `docs/ARCHITECTURE.md` — notes 7a–c, `is_attack` rule, `FlowPackets` order, SSH limitation
- Dependencies added: none

## My part (TODO(human))
- IAT and rate features with edge cases — done by Claude (guided mode)
- `syn_scan` generator — done by Claude (guided mode)

## Verification (real output)
- Data used: synthetic only (documentation IP ranges)
- Tests: 183 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`; `37 files already formatted`
- Main command: `uv run python -m neural_ids.synthetic --n 20000 --seed 42` (difficulty 0.5,
  label_noise 0.025) → true attacks 4000 (1334/1333/1333), labeled attacks 4300;
  flipped normal → attack 410, attack → normal 110. A perfect model scored against the noisy
  labels: attack recall 0.9047, attack precision 0.9725. CSV passes `validate_flows`.
- SSH overlap: share of `ssh_session` flows inside the central 90% brute-force box
  (packets × duration), 300 flows each: 0.0 at difficulty 0, 0.043 at 0.5, 0.067 at 1
- Acceptance criteria: all pass — tests incl. hand-computed examples; figures separate at
  difficulty 0 and overlap at 1 (true class); code and `features.md` agree
- `ml-reviewer`: 0 must-fix, 6 should-fix; all fixed

## Key concepts
- Effective time is a cumulative maximum: backwards timestamps give gap 0, never sorting.
- nmap -sS: open = SYN, SYN/ACK, RST (kernel reset); closed = RST/ACK; filtered = one SYN.
- Label noise caps the measurable score: with noisy labels even a perfect model loses recall.
- One failed login per flow cannot separate brute force from a typo; counting needs host windows.

## Decisions and deviations from the spec
- Approved notes 1–7 and plan changes 1–5; `flow_record` lives in `features.py` for T10.
- After review: benign flows to ephemeral ports (mid-connection web, response-only DNS);
  `label_noise` argument so plots use true classes; µs timestamps; stricter `FlowPackets`.
- Addition 1: `ssh_session` short variants (mistyped password, `git fetch`, small `scp`);
  `ssh_bruteforce` goes low-and-slow with difficulty (1 attempt per connection, tool wait).
- Addition 2: `synthetic.json` provenance and perfect-model recall/precision.

## Problems and open questions
- At difficulty 0, high ports still occur only in attacks (intended: easy mode).
- The SSH overlap is partial (6.7% of sessions at difficulty 1); `git`/`scp` flows stay apart.
- Ideas for later: ICMP "unreachable" answers to filtered probes; SYN flood and slow DoS.

## Suggested next step
- T03 — Preprocessing without leakage.
