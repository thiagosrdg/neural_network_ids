# T12 — End to end, demo, and portfolio docs

Phase: 1 (Build) · Depends on: T11

## Before you start (human)
Capture 2–3 minutes of your own computer's traffic into `data/raw/` (see `docs/GUIDE.md`,
section 4). Only capture traffic you are allowed to capture.

## Goal
Finish phase 1 as a portfolio project: one command scores any capture, one command shows the
whole pipeline working on synthetic data, and the docs explain the architecture and how anyone
can train the model on their own traffic.

## Why it matters
- ML: an end-to-end test catches integration bugs that unit tests miss.
- Security: a detection tool must fail safely (clear errors, no payloads in the output,
  anonymized identifiers on request) and be honest about what its model was trained on.
- Portfolio: a visitor should understand and run the project in minutes, without any data.

## Deliverables
- `neural-ids score <capture or flows.csv> --bundle <dir> [--top 20] [--anonymize]`: validates
  the input, scores each flow (attack probability, predicted class, or reconstruction error),
  and prints a summary plus the most suspicious flows. It warns clearly when the bundle was
  trained on synthetic data.
- `neural-ids demo`: in a temporary folder, generates synthetic flows, trains both model types,
  writes a small crafted capture, and scores it, in about a minute on a laptop, under a banner
  that says "synthetic demo".
- Smoke tests: the demo, and your real capture scored with a bundle trained on synthetic data.
  In the report, show only aggregates (packets, flows, protocols, score distribution) — no IP
  addresses.
- `docs/ARCHITECTURE.md` updated to match the code: real layer sizes and parameter counts, the
  final commands, and any changed design decisions.
- `docs/MODEL_CARD_TEMPLATE.md`: a card that users fill in for the model they train (data,
  labels, splits, metrics, limitations).
- `README.md` for GitHub: what the project is, the demo, "Train it on your own traffic"
  (capture rules, the dataset folder convention, the three commands), an architecture summary
  with the diagram, limitations, privacy and authorization, citations, and third-party licenses.

## TODO(human)
- The warning logic and the input-validation messages in `score`.
- The README's "Limitations" and "What I learned" sections, in your own words.

## Acceptance criteria
- `uv run neural-ids demo` works in a fresh clone after `uv sync`.
- `score` works on a crafted capture and on your real capture; an unsupported link type fails
  with a clear message, not a stack trace.
- No payload bytes and no unanonymized IP addresses appear in the outputs or the report.
- All tests pass. Phase 1 is complete.

## Learning check
1. Why are the scores of a synthetically trained model meaningless on your real capture?
2. What does a user need to collect to train the classifier, and what for the autoencoder?
3. Which privacy risks remain even with `--anonymize`?
