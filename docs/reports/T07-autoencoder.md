# T07 — Autoencoder from scratch: anomaly detection without labels · Report

- Date: 2026-10-15 · Status: done · Learning mode: guided (chosen by the user) · Commit: see `git log` (`feat: add NumPy autoencoder and reconstruction-error anomaly detection`)

## Summary
A NumPy autoencoder [29, 16, 4, 16, 29] is trained on normal flows only and flags a flow when its reconstruction error
exceeds the 99th percentile of normal validation errors. No attack label is used before the final test scoring.

## What was built
- `nn/losses.py`: `MSELoss` (mean over B·d, gradient 2·diff/(B·d), strict shapes) · `nn/autoencoder.py`: `Autoencoder(MLP)`
  with a symmetric-size check and `encode` (bottleneck after ReLU)
- `anomaly.py`: `NormalScaler`, `reconstruction_error`, `threshold_from_normal`, `flag`, `detection_table` (with noise ceiling)
- `train_autoencoder_numpy.py`: `--val-only` (never opens test.npz); saves `models/ae_numpy.npz` (weights, scaler, threshold,
  `layer_sizes`, `schema_version`, `input_sha256`; loaded with `allow_pickle=False` and checked) and two figures
- `tests/test_autoencoder.py` (28 tests); `docs/ARCHITECTURE.md`: T07 section. Dependencies added: none
- Docs fix (note 6): T04 and T06 Commit lines and `TEMPLATE.md` now use `see git log (<message>)`
## My part (TODO(human))
- `MSELoss.forward/backward`, `reconstruction_error`, `threshold_from_normal`: written by Claude (guided mode, explained in chat)
## Verification (real output)
- Data: SYNTHETIC · `uv run pytest -q`: 356 passed · ruff: `All checks passed!`, `76 files already formatted`
- Tuning used `--val-only` only, choosing by the normal validation MSE (no attack labels). After the scaler fix: lr 0.1 →
  0.0047 (kept), lr 0.05 → 0.0051, lr 0.2 → 0.0104; an 800-epoch limit changed nothing (early stop at 219). Then the test
  split was scored once. The histogram code was fixed afterwards (see Decisions); a re-run with the same settings printed
  identical output (`diff`), so no number changed.
- `uv run python -m neural_ids.train_autoencoder_numpy`: trained on 11000 normal flows, stopped at epoch 219, kept epoch 199
  (normal val MSE 0.0047; first loss 0.8594). Bottleneck: 4 of 4 units active (share of flows per unit 0.998, 0.984, 0.109,
  0.995). Threshold 0.0346; normal val flows flagged 0.010. Test, SYNTHETIC (tests code, not detection):

  | rate | autoencoder | noise ceiling* | flows |
  |---|---|---|---|
  | false-positive rate (normal) | 0.007 | 0.008 | 2358 |
  | detection: ssh_bruteforce | 0.005 | 0.876 | 218 |
  | detection: syn_scan | 0.427 | 0.901 | 213 |
  | detection: udp_flood | 0.774 | 0.915 | 212 |
  | detection: all attacks | 0.398 | 0.897 | 643 |

  \* a perfect detector against the noisy test labels (`flipped_flow_ids`, evaluation only). Of the 17 flagged normal test
  flows, 7 are attacks that label noise marked normal: the false-positive rate on truly normal flows is 0.004 (10 of 2340).
- FPR 0.007 against a 1% budget: about 24 flags were expected, 17 came out. With binomial noise (standard deviation about
  4.8 flows) that is about 1.4 standard deviations, so it is consistent with the budget, not a large difference.
- Acceptance criteria: all pass. `ml-reviewer`: no attack flow in the scaler, the training, early stopping, or the
  threshold; fixes applied for its findings (normal-only scaler, NaN-safe table, `layer_sizes` dtype check).
## Key concepts
- Same layers and backprop as the MLP; only the target (the input) and the loss (MSE) change.
- The threshold is a false-positive budget, set on normal data only: anomaly detection needs no attack labels.
- "Unusual" is not "malicious": the score measures distance from normal traffic, not intent.
## Decisions and deviations from the spec
- Extra `NormalScaler`: the T03 preprocessor is fitted on all training rows (attacks too), so the autoencoder re-standardizes
  on normal training rows only. Columns constant in normal traffic keep scale 1.
- Error histogram: share of each class per bin, not `density=True`, which flattened high-error bins on the log axis.
- The same normal val rows pick the stopping epoch and set the threshold (reviewer note): this can make the threshold slightly
  low, but the test FPR came out below budget. About 18 of 2358 normal val rows are flipped attacks (count from the
  reviewer, evaluation only); kept on purpose, as in real captures.
## Problems and open questions · Suggested next step
- SSH brute force is almost never caught (0.005). Each attempt looks like a short SSH session, and T02 added short benign
  SSH flows on purpose. The histogram puts these errors at about 3e-3 to 2e-2, above most normal flows but below the
  threshold. Per-flow features cannot see "many attempts from one host", so host-window features (phase 2) are needed.
- `syn_scan` has a group just below the threshold (about 1.5e-2): a lower quantile catches more, at the cost of more false
  positives. Later: plot detection against false-positive rate (a ROC-style curve) instead of using one quantile.
- Next: T08 — PyTorch classifier and cross-check.
