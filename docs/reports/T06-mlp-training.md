# T06 — MLP from scratch, part 2: training loop and debugging · Report

- Date: 2026-10-15 · Status: done · Learning mode: guided (chosen by the user) · Commit: not committed yet

## Summary
The NumPy MLP trains with mini-batch SGD with momentum and early stopping on validation loss. It passes both sanity
checks, gives the same output on repeated runs, and is compared with logistic regression and a random forest.

## What was built
- `nn/optim.py`: `SGD(params, lr, momentum)`, PyTorch form `v = μv + g; p -= lr·v`, updates in place
- `nn/layers.py`, `nn/model.py`: `Dense(init_scale=...)`; the last layer starts at He × 0.01 with bias 0
- `train_numpy.py`: `TrainConfig`, `train_epoch`, `fit` (early stopping, best weights restored in place), `--sanity`
  (exit 1 on FAIL), `save_mlp`/`load_mlp`, curves with the best epoch marked, and the validation table
- `models/mlp_numpy.npz` (gitignored): weights, `layer_sizes`, `schema_version`, `input_sha256`. Loading uses
  `allow_pickle=False` and checks the schema version, the input hash, the 29 inputs, and every shape.
- Tests: `test_optim.py` (8), `test_train_numpy.py` (22); `docs/ARCHITECTURE.md`: T06 section. Dependencies added: none
## My part (TODO(human))
- `SGD.step` momentum update and the `train_epoch` body: written by Claude (guided mode, explained line by line in chat)
## Verification (real output)
- Data: SYNTHETIC `{train,val}.npz` (20000 flows; test not loaded) · 328 passed · ruff: `All checks passed!`, `71 files already formatted`
- `--sanity`: first loss 0.6941 (ln 2 = 0.6931); one batch of 64: 0.6906 → 0.013514 (limit 0.02). Over 20 seeds the
  first loss ranges from 0.686 to 0.698 (plain He init: 0.513 to 1.528).
- `uv run python -m neural_ids.train_numpy` (lr 0.05, momentum 0.9, batch 64, hidden (64, 32), seed 42, patience 10):
  stopped after epoch 48, kept epoch 38 (val loss 0.1458). Validation, SYNTHETIC:

  | model | accuracy | precision | recall | F1 | TN | FP | FN | TP |
  |---|---|---|---|---|---|---|---|---|
  | logistic regression | 0.917 | 0.790 | 0.835 | 0.812 | 2215 | 143 | 106 | 537 |
  | MLP (NumPy) | 0.960 | 0.945 | 0.862 | 0.902 | 2326 | 32 | 89 | 554 |
  | random forest (reference) | 0.952 | 0.913 | 0.860 | 0.886 | 2305 | 53 | 90 | 553 |
  | noise ceiling (perfect model) | - | 0.970 | 0.921 | - | | | | |

  The validation split also chose the MLP's stopping epoch, so the MLP's validation numbers are slightly optimistic.
  The test split stays unused until the final comparison.
- Acceptance criteria: all pass (repeated runs: identical output by `diff`; two compared runs saved identical weights).
  `ml-reviewer`: no leakage, contract, or pickle problems. Fixed: reshuffle test, `load_mlp` architecture checks, an
  unmeasured number, `--sanity` exit code, `first_loss` uses `fit`'s init stream.

## Optional experiment: difficulty (validation F1, SYNTHETIC; default data restored, all hashes match)
| difficulty | logreg | MLP | random forest | ceiling recall / precision | MLP kept / ran |
|---|---|---|---|---|---|
| 0 | 0.998 | 1.000 | 1.000 | 1.000 / 1.000 | 100 / 100 (limit) |
| 0.5 | 0.812 | 0.902 | 0.886 | 0.921 / 0.970 | 38 / 48 |
| 1 | 0.628 | 0.787 | 0.766 | 0.824 / 0.960 | 35 / 45 |

When the classes are well separated, a straight line is enough and all models nearly tie. With more overlap,
logistic regression falls fastest (recall 0.493 against the MLP's 0.695 at difficulty 1). The forest stays close to the
MLP, so the gap comes from the data needing a curved boundary, not from something special about the MLP.
## Key concepts
- Momentum: v is a decaying sum of past gradients; steady directions speed up, sign-flipping ones cancel.
- Untrained model: p ≈ 0.5, loss ≈ ln 2. Cannot memorize 64 rows: the bug is in the model or the loop.

## Decisions and deviations from the spec
- Overfit check: 2000 full-batch steps at lr 0.1, limit 0.02. The seed-42 batch holds a flipped label 0.047 from a near
  copy labeled the other way, so the loss stops near 0.0135.
- The T05 note gave 1.4 for the plain-He logit standard deviation; measured here: 0.81 within one model, 1.18 pooled.

## Problems and open questions · Suggested next step
- Later: tune the 0.5 threshold on validation; weight decay; lr decay (the val loss is noisy from epoch to epoch).
- Next: T07 — Autoencoder from scratch: anomaly detection without labels.
