# T03 — Preprocessing without leakage · Report

- Date: 2026-10-08
- Status: done
- Learning mode: hands-on, but the user asked Claude to write the TODO(human) piece too
- Commit: see `git log` (`feat: add leak-free preprocessing with saved preprocessor`)

## Summary
`preprocess.py` turns a flows table into `(n, 29)` float32 model inputs: validate, select the
contract features, `log1p` the 21 non-one-hot features, and standardize them with training
statistics. The fitted preprocessor is self-contained JSON. The command writes train/val/test arrays.

## What was built
- `src/neural_ids/preprocess.py` — `split` (stratified, seeded), `Standardizer`, `LOG_FEATURES`
  (each justified), `Preprocessor` (fit/transform/save/load), `class_names_from`,
  `make_targets`, `count_exact_copies`, `data_source`, CLI
- `tests/test_preprocess.py` — 34 tests: split, sklearn match, constant columns, train-only
  statistics (unit and command level), metadata never in X, JSON round trip, version/order
  checks, self-contained load, targets from `is_attack`, `flow_id` aligned with `X` rows
- `explore/shortcut_check.py` — "exactly one failed SSH login (16 packets)" subset (review note 5)
- `docs/ARCHITECTURE.md` — new "Preprocessing (T03)" section; T02 report number corrected
- Dependencies added: none

## My part (TODO(human))
- `Standardizer.fit` and `Standardizer.transform` — skipped (done by Claude, at the user's request)

## Verification (real output)
- Data used: synthetic (`data/processed/synthetic.csv`, 20000 flows, documentation IP ranges)
- Tests: 224 passed — `uv run pytest -q`
- Lint and format: `All checks passed!`, `41 files already formatted`
- Main command: `uv run python -m neural_ids.preprocess --input data/processed/synthetic.csv`
  - classes `['normal', 'ssh_bruteforce', 'syn_scan', 'udp_flood']`; source: SYNTHETIC
  - train X (13998, 29) 70.0% · val X (3001, 29) 15.0% · test X (3001, 29) 15.0%
  - exact feature copies in train: val 156/3001 (5.2%), test 151/3001 (5.0%), mostly
    one-packet flows (val: 154 of 156); only `urg_count` is constant in training (scale 1)
- `explore/shortcut_check.py`: exactly one failed SSH login: accuracy 0.576 (std 0.034),
  rows 341/268, majority 0.560 (SYNTHETIC; replaces the unreproducible 0.508 in the T02 report)
- Acceptance criteria: all pass. `ml-reviewer`: no leakage, no must-fix; 3 should-fix and 2
  doc nits, all fixed (arrays now store schema version and source; command-level train-only test)

## Key concepts
- Leakage: any statistic fitted on rows you evaluate on makes results look better than they are.
- Broadcasting: `(n, d) - (d,)` repeats the row vector, so no loop over samples.
- `std == 0` misses constant nonzero columns (np.std ≈ 1e-16); use a floating-point error bound.
- A saved preprocessor must not depend on current code constants, or old models silently change.

## Decisions and deviations from the spec
- Constant columns get scale 1 by sklearn's `_is_constant_feature` rule, not an epsilon (notes).
- `log1p` on all 21 non-one-hot features, including packet lengths (TSO captures reach ~64 KB).
- `split` uses our own NumPy code with `set_seed`, not `train_test_split` (seed rule; easy swap).
- `class_names` come from train; an attack label not seen in training raises.
- Arrays also store `feature_names`, `schema_version`, `source` (from `synth-` IDs or `--source`),
  and `flow_id` per row (metadata, never in `X`); the command never reads `synthetic.json`.

## Problems and open questions
- Later loaders (T04+) should check `schema_version` in the `.npz` files.
- Phase 2: replace `split` with a split by capture; watch the exact-copy share there.

## Suggested next step
- T04 — Baselines: majority class, perceptron, logistic regression.
