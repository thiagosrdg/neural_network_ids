# T11 — Bring your own data: labeled datasets from captures

Phase: 1 (Build) · Depends on: T10

## Goal
Let anyone turn a folder of their own captures into a labeled flows table that `train`
accepts, with one sub-folder per label. Make training split by capture, so flows from the same
capture never end up in both training and test data.

## Why it matters
- ML: flows from one capture are strongly related (same hosts, same sessions). A random split
  puts near-copies in training and test data and inflates the results; a split by capture (a
  "group split") gives honest numbers.
- Security: a dataset manifest with a checksum for every capture records exactly what a model
  was trained on (provenance), and lets anyone check that the files did not change.

## Deliverables
- The dataset folder convention, documented in `docs/ARCHITECTURE.md` and the README:

  ```text
  my-captures/
    normal/        *.pcap, *.pcapng   → label "normal"
    portscan/      *.pcap, *.pcapng   → label "portscan"
  ```

  Label names use lowercase letters, digits, `_`, and `-`. `normal` is reserved for benign traffic.
- `src/neural_ids/dataset.py` and the command `neural-ids build-dataset <dir> --out flows.csv
  [--anonymize]`: extracts every capture with T10, adds `label` (the folder name), `is_attack`
  (label is not `normal`), and `capture_id`; writes `flows.csv` and `dataset_manifest.json` (for
  each capture: path relative to `<dir>`, SHA-256, packet and flow counts, label; plus the schema
  version and the extractor settings); prints a summary per label.
- `train` gets `--group-column capture_id`: when given, it splits by group (scikit-learn's
  `GroupShuffleSplit` is fine) and stops with a clear message when a label has too few captures
  to split. The autoencoder mode uses only `normal` flows for training and for the threshold.
- Tests: build a small dataset folder of crafted captures in a temporary folder (2 labels, 3
  captures each) and check labels, capture IDs, and manifest checksums; the group split never
  puts one capture in two splits; clear errors for an empty folder, a bad label name, and a
  file that is not a capture.

## TODO(human)
- The group split: no `capture_id` may appear in more than one split.
- The manifest entry for one capture (like the bundle checksum in T09).

## Acceptance criteria
- `build-dataset` followed by `train` works end to end on crafted captures, in both modes.
- No capture appears in two splits (a test and the `ml-reviewer` subagent confirm).
- All tests pass.

## Out of scope
Labeling rules for mixed captures (time windows and IP addresses, as public datasets need):
phase 2.

## Learning check
1. Why does a random split by flow give better-looking but dishonest numbers here?
2. What does the manifest let another person verify, and what can it not prove?
3. What happens to the autoencoder if a "normal" capture secretly contains an attack?
