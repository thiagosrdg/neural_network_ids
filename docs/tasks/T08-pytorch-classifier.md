# T08 — PyTorch classifier and cross-check

Phase: 1 (Build) · Depends on: T07

## Goal
Learn PyTorch fundamentals by rebuilding the MLP classifier, extend it from two classes to
several with softmax, and prove that your NumPy backpropagation is correct by comparing it
with PyTorch's automatic differentiation (autograd).

## Why it matters
- ML: frameworks automate gradients, batching, and optimizers — now you know exactly what they
  automate. Softmax with cross-entropy generalizes the binary case.
- Security: analysts need to know what kind of attack it is (triage), not only "attack or not".
  Safe model files: save only the `state_dict` and load it with `weights_only=True`.

## Deliverables
- CPU-only PyTorch installed with uv, following
  https://docs.astral.sh/uv/guides/integration/pytorch/ (an explicit `pytorch-cpu` index).
  Explain the choice and ask before the download.
- `set_seed` also seeds PyTorch.
- `src/neural_ids/torch_models.py`: `MLPClassifier(nn.Module)` with `n_classes` (one output
  for binary, `n_classes` outputs for multi-class) and optional dropout.
- `src/neural_ids/train_torch.py`: a `Dataset` and `DataLoader` over the `.npz` files; a
  training loop with `model.train()`, `model.eval()`, and `torch.no_grad()`;
  `BCEWithLogitsLoss` or `CrossEntropyLoss` (with optional class weights computed from the
  training split); `AdamW`; early stopping; loss curves.
- `tests/test_torch_crosscheck.py`: copy the NumPy MLP weights into the PyTorch model (float64)
  and check that the logits match (atol 1e-8) and that autograd gradients match your `backward`
  (rtol 1e-6).
- `docs/notes/softmax.md`: why the gradient of softmax with cross-entropy is `p − onehot(y)`.
- Results (labeled synthetic): binary and multi-class validation metrics, including recall per
  class and macro-F1.

## TODO(human)
- The PyTorch training step: `zero_grad` → forward → loss → `backward` → `step`.
- The weight-copy function for the cross-check. Watch the shapes: `nn.Linear.weight` is
  `(out, in)`.

## Acceptance criteria
- The cross-check test passes.
- Binary and multi-class training run, and every number comes from a real run.

## Out of scope
The PyTorch autoencoder and the train command (T09).
