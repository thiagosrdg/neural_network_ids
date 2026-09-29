# T01 — The feature contract: from packets to flows · Report

- Date: 2026-09-28
- Status: done
- Learning mode: hands-on (both TODO(human) pieces skipped at my request)
- Commit: see `git log` (`feat: define feature contract v1 and validate_flows`)

## Summary
The model's input is now fixed and versioned: 29 header and timing features per flow (schema
1.0), defined in words in `docs/features.md` and in code in `schema.py`. `validate_flows`
rejects any flows table that breaks the contract, with a message that names the problem.

## What was built
- `docs/features.md` — packet rules, flow definition and end rules, metadata, 29-feature table, edge cases, future features
- `src/neural_ids/schema.py` — `SCHEMA_VERSION`, timeouts, IP length limits, metadata and optional columns, `FEATURES`, `ONE_HOT_GROUPS`, `SchemaError`, `validate_flows`
- `tests/test_schema.py` — contract, `validate_flows`, and doc-sync tests
- `docs/ARCHITECTURE.md` — fragments are skipped (rule and limitation)
- `README.md` — `uv python install 3.13` before `uv sync` (requested fix outside the spec)
- Dependencies added: none

## My part (TODO(human))
- "Why it may reveal attacks" column — skipped (done by Claude, all 29 rows)
- `validate_flows` — skipped (done by Claude, after hint 1)

## Verification (real output)
- Data used: none (hand-made two-flow table in the tests; documentation IP ranges)
- Tests: 54 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`; `30 files already formatted`
- Main command: `uv run python -c "from neural_ids.schema import FEATURES, SCHEMA_VERSION; print(SCHEMA_VERSION, len(FEATURES))"` → `1.0 29`
- Acceptance criteria: all pass (precise definitions; tests pass with a clear error per problem; no identifier in `FEATURES`)
- `ml-reviewer` subagent: 1 must-fix and 5 should-fix points, all addressed (see below)

## Key concepts
- A versioned feature contract prevents training–serving skew, which breaks models silently.
- Identifiers and labels as inputs cause shortcuts and target leakage.
- Timeouts are part of the contract: they change how flows split, so they change feature values.
- NaN fails every comparison and `+inf <= inf` is true, so finiteness needs its own check.
- Population std (ddof=0): NumPy's default, but pandas `.std()` needs `ddof=0`.

## Decisions and deviations from the spec
- Approved changes to the plan: optional columns `label`, `is_attack`, `capture_id`; NaN and ±inf
  rejected; every IP fragment skipped; ddof=0; timeouts in ranges (`duration_s` ≤ 1800,
  `iat_max_s` ≤ 120, lengths 20–65575); backward timestamps clamped; one-hot check;
  ICMPv6 in `proto_icmp`; port class `none` set by protocol.
- Float features also accept integer dtypes (CSV round trips turn `0.0` columns into int64).
- After the review: duplicate columns rejected; row-consistency checks (port class `none` vs.
  protocol, TCP flags 0 for non-TCP, `bwd_bytes`/`bwd_packets`, min ≤ mean ≤ max); the TCP close
  rule defines "closing FIN" and "final ACK" exactly; times are float64 Unix seconds.

## Problems and open questions
- The "25.9% of CIC-IDS2017" figure in `ARCHITECTURE.md` was not checked against its source in
  this task, and its base (all flows or TCP flows) is not stated; removed from `features.md`.
- Metadata and optional columns are checked for presence only (not dtype or values).

## Suggested next step
- T02 — Flow features and synthetic traffic.
