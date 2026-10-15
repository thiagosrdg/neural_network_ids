# T08 — PyTorch classifier and cross-check · Report

- Date: 2026-10-15 · Status: done · Learning mode: guided (chosen by the user) · Commit: see `git log` (`feat: add PyTorch MLP classifier, NumPy cross-check, and multi-class metrics`)
## Summary
A PyTorch `MLPClassifier` trains as a binary or a 4-class model. With the same float64 weights, its logits, its autograd
gradients, and `torch.optim.SGD` steps match our NumPy MLP, `backward`, and `SGD`: the T05–T06 backprop is confirmed.
## What was built
- `torch_models.py`: `MLPClassifier(n_features, hidden, n_classes, dropout)` (NumPy-style init), `copy_numpy_weights`
- `train_torch.py`: `FlowDataset`, seeded `DataLoader` (`num_workers=0`), `train_step`, `clone_state`, AdamW, early stopping,
  train-only class weights, `save_model`/`load_model` (`.pt` state_dict + `.json`, `weights_only=True`), curves, CLI
- `metrics.py`: `multiclass_confusion`, `per_class_report`, `macro_f1`, `recall_ceiling_per_class`; `utils.set_seed` seeds
  torch; `train_numpy.compare_on_val` takes extra rows; `docs/notes/softmax.md`; `docs/ARCHITECTURE.md`: T08 section
- Tests: `test_torch_crosscheck.py` (9), `test_train_torch.py` (23), +6 metrics, +1 utils · Dependency added: `torch`
  2.14.1+cpu, from an explicit `pytorch-cpu` uv index (no CUDA libraries)
## My part (TODO(human))
- `train_step` and `copy_numpy_weights`: written by Claude (guided mode, explained line by line in chat)
## Verification (real output)
- Data: SYNTHETIC `{train,val}.npz` (test not opened) · `uv run pytest -q`: 395 passed · ruff: `All checks passed!`, `83 files already formatted`
- Cross-check (float64): logits atol 1e-8 (binary and 4 outputs); gradients rtol 1e-6 for BCE and softmax cross-entropy;
  5 steps of our `SGD(0.9)` against `torch.optim.SGD(momentum=0.9)` agree within 1e-10: all pass.
- Determinism: each of the three commands run twice gave identical output (`diff`).
- `train_torch --target binary` (AdamW lr 1e-3, wd 0.01, batch 64, hidden (64, 32), seed 42): first loss 0.6892 (ln 2 =
  0.6931), stopped at epoch 47, kept 37. Validation, SYNTHETIC (logreg, NumPy MLP, forest rows equal T06):

  | model | accuracy | precision | recall | F1 | TN | FP | FN | TP |
  |---|---|---|---|---|---|---|---|---|
  | logistic regression | 0.917 | 0.790 | 0.835 | 0.812 | 2215 | 143 | 106 | 537 |
  | MLP (NumPy, T06) | 0.960 | 0.945 | 0.862 | 0.902 | 2326 | 32 | 89 | 554 |
  | MLP (PyTorch) | 0.958 | 0.940 | 0.857 | 0.897 | 2323 | 35 | 92 | 551 |
  | random forest (reference) | 0.952 | 0.913 | 0.860 | 0.886 | 2305 | 53 | 90 | 553 |
  | noise ceiling (perfect model) | - | 0.970 | 0.921 | - | | | | |

- `--target multiclass`, without / with `--class-weights` (train weights 0.3181, 3.4478, 3.5136, 3.5456). Both: first loss
  near ln 4 = 1.3863 (1.3904 / 1.3884), stopped at 44, kept 34. Validation, SYNTHETIC; recall ceiling from `flipped_flow_ids`:

  | class (val rows) | recall ceiling | recall: no weights | recall: weights | precision: no w. / w. |
  |---|---|---|---|---|
  | normal (2358) | 0.992 | 0.986 | 0.913 | 0.962 / 0.979 |
  | ssh_bruteforce (218) | 0.917 | 0.890 | 0.917 | 0.965 / 0.885 |
  | syn_scan (213) | 0.939 | 0.873 | 0.944 | 0.912 / 0.634 |
  | udp_flood (212) | 0.906 | 0.802 | 0.906 | 0.950 / 0.738 |
  | macro-F1 · accuracy | | 0.915 · 0.958 | 0.854 · 0.915 | |

- Acceptance criteria: all pass. `ml-reviewer`: no leakage, contract, or pickle problems; fixes applied (weighted-BCE test,
  correct expected first loss, deterministic setting restored after `fit`, no metadata overrides, model-type check).
## Key concepts
- PyTorch automates backward, batching, and updates; same weights → same numbers as our NumPy code.
- Softmax + cross-entropy has gradient p − onehot(y) (binary: sigmoid(z) − y). `state_dict()` shares memory: `clone()` it.
## Decisions and deviations from the spec
- Same init as NumPy (He, last layer × 0.01), not PyTorch's default, so the first loss is near ln K in both frameworks.
- Class weighting is a trade-off. It moves predictions toward the rare attack classes: missed attacks fell from 93 to
  46, but normal flows predicted as an attack rose from 34 to 206. Macro-F1 fell from 0.915 to 0.854. Weighting buys
  attack recall with false alarms, so the choice depends on the alert budget.
- syn_scan's recall with weights (0.944) is above its ceiling (0.939) by 1 flow. The model flags many normal-looking flows
  (precision 0.634), and one of them is a normal flow labeled syn_scan by noise. This is not leakage; the precision drop shows it.
## Problems and open questions · Suggested next step
- Both runs' lowest val loss is at epoch 34 (0.1748, 0.2793): a checked coincidence. Weighted run, epoch 44: macro-F1 0.858 →
  0.895 is real (a seeded replay gave the same values): one epoch moved the normal/syn_scan boundary (normal flows predicted
  syn_scan 112 → 20, syn_scan hits 201 → 186 of 213), and the weighted loss rose (0.2848 → 0.2865) because a missed syn_scan
  weighs about 11× a normal flow (3.5136 / 0.3181). Later: attack/normal view of multi-class predictions; tune dropout. Next: T09.
