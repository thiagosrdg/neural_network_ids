# T03 — Preprocessing without leakage

Phase: 1 (Build) · Depends on: T02

## Goal
Turn a flows table into numeric arrays for the models, with every learned statistic coming from
the training split only, and save the fitted parameters so they work for any future capture.
Practice NumPy vectorization and broadcasting on the way.

## Why it matters
- ML: neural networks need similarly scaled inputs. Data leakage (using validation or test data
  while fitting) gives results that look great and are false.
- Security: a deployed detector only knows the past. The saved preprocessor must reject input
  that does not match the contract instead of producing silent garbage.

## Deliverables
- `src/neural_ids/preprocess.py`:
  - `split(df, val_fraction=0.15, test_fraction=0.15, seed)`: stratified by `label`.
  - `Standardizer` (from scratch): `fit(X)` stores the per-column mean and standard deviation
    (with a small epsilon); `transform(X)` uses broadcasting.
  - `Preprocessor`: takes only `schema.FEATURES` (never metadata), applies `log1p` to
    heavy-tailed features (counts, bytes, durations, rates — list them and justify each), then
    standardizes every feature that is not one-hot. Output `float32`.
  - `save(path)` and `load(path)` as JSON, including `SCHEMA_VERSION` and the feature order;
    `load` refuses a different schema version.
- Command `uv run python -m neural_ids.preprocess --input data/processed/synthetic.csv`:
  writes `data/processed/{train,val,test}.npz` with `X` (n, d), `y_binary` (n,), `y_class` (n,)
  — `normal` for every normal label, otherwise the attack label — and `class_names`; prints the
  shapes; saves `models/preprocessor.json`.
- Tests: shapes; no NaN or inf; statistics come from training data only (build a case where
  train and validation differ strongly); our `Standardizer` matches scikit-learn's
  `StandardScaler` within 1e-6; JSON round trip; a schema-version mismatch raises an error;
  metadata columns never reach `X`.

## Watch out
With real captures (phase 2), a random split leaks: flows from the same session land in both
training and test data. There we will split by capture or by time window, so keep `split`
easy to replace.

## TODO(human)
- `Standardizer.fit` and `Standardizer.transform` (vectorized, no loops).

## Acceptance criteria
- The command prints shapes with about a 70/15/15 split and the same `d` for every split.
- All tests pass, and the `ml-reviewer` subagent finds no leakage.

## Out of scope
Models.

## Learning check
1. Explain the broadcasting in `(X - mean) / std` with the shapes `(n, d)` and `(d,)`.
2. Why `log1p` for byte counts and rates? What does it do to a flow with 0 bytes and to one
   with 10⁹ bytes?
3. What is the time and memory complexity of fitting the standardizer?
