# T09 — PyTorch autoencoder, model bundle, and train command

Phase: 1 (Build) · Depends on: T08

## Goal
Port the autoencoder to PyTorch, define one safe "model bundle" format, and write the command
that trains either model on any flows table that follows the contract. In phase 2 we point
this command at real data.

## Why it matters
- ML: a model is useless without the exact preprocessing, feature order, class names, and
  threshold used in training. A bundle keeps them together and versioned.
- Security: the bundle loads without pickle and checks the schema version and a checksum, so a
  wrong or corrupted file fails loudly instead of producing silent garbage.

## Deliverables
- `Autoencoder(nn.Module)` in `torch_models.py`, cross-checked against the NumPy version (same
  weights → same reconstructions).
- `src/neural_ids/bundle.py`: `save_bundle(dir, model, preprocessor, meta)` and
  `load_bundle(dir)`. A bundle folder holds `model.pt` (the `state_dict` only) and
  `bundle.json`: `SCHEMA_VERSION`, feature order, preprocessor parameters, model type and layer
  sizes, class names, threshold, a description of the training data (synthetic or real), and
  the SHA-256 of `model.pt`. Loading checks the schema version and the checksum.
- A CLI entry point `neural-ids` (standard-library `argparse`, declared in `pyproject.toml`)
  with `train --flows <csv> --mode classifier|autoencoder [--label-column label] --out
  models/<name>`. It validates the table with `validate_flows`, splits it, fits the
  preprocessor on the training split, trains, evaluates on validation, and writes the bundle.
- Tests: bundle round trip; a changed schema version or checksum is rejected; `train` runs end
  to end on a small synthetic table in both modes.

## TODO(human)
- The checksum verification in `load_bundle`.
- Choosing the autoencoder threshold inside `train` (reuse your T07 logic).

## Acceptance criteria
- `uv run neural-ids train --flows data/processed/synthetic.csv --mode autoencoder --out
  models/ae_synth` works, and so does the same command with `--mode classifier`.
- All tests pass; nothing is loaded with pickle (the `ml-reviewer` subagent confirms).

## Out of scope
Reading captures (T10).
