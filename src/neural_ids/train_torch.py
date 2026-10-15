"""Train the PyTorch MLP classifier: binary (attack or not) or multi-class (which attack).

Commands:
    uv run python -m neural_ids.train_torch --target binary
    uv run python -m neural_ids.train_torch --target multiclass
    uv run python -m neural_ids.train_torch --target multiclass --class-weights

Reads `data/processed/{train,val}.npz` (never test.npz: the test split is used once, at the end
of an experiment). Saves the best weights as a `state_dict` to `models/mlp_torch_<target>.pt`,
the metadata (layer sizes, class names, schema version, input hash, settings) next to it as
JSON, and the loss curves to `reports/figures/`. Then it prints validation results:
- binary: logistic regression, the NumPy MLP (T06), the PyTorch MLP, a random forest
  (reference), and, for SYNTHETIC data, the label-noise ceiling;
- multi-class: precision, recall, F1 per class with the recall ceiling per class, macro-F1,
  and the confusion matrix.
The validation split also chooses the stopping epoch, so validation numbers are slightly
optimistic.

What PyTorch automates compared with T06:
    T06 (NumPy)                         here (PyTorch)
    minibatch_indices + x[idx]          DataLoader(shuffle=True, generator=...)
    model.backward(loss_fn.backward())  loss.backward()   (autograd)
    SGD.step(model.gradients())         opt.step()        (AdamW)
    snapshot / restore                  clone_state / load_state_dict
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from neural_ids.metrics import (  # noqa: E402
    f1,
    macro_f1,
    multiclass_confusion,
    per_class_report,
    recall_ceiling_per_class,
)
from neural_ids.schema import FEATURE_NAMES, SCHEMA_VERSION, SchemaError  # noqa: E402
from neural_ids.splits import Split, load_split  # noqa: E402
from neural_ids.torch_models import MLPClassifier  # noqa: E402
from neural_ids.train_baselines import read_flipped_ids  # noqa: E402
from neural_ids.train_numpy import (  # noqa: E402
    _ceiling,
    _check_same_source,
    _print_table,
    compare_on_val,
    load_mlp,
)
from neural_ids.utils import set_seed  # noqa: E402

BINARY_CLASS_NAMES = ("normal", "attack")


@dataclass(frozen=True)
class TorchConfig:
    """Settings of one PyTorch training run. Saved in the model's JSON file."""

    lr: float = 1e-3  # AdamW's usual starting point
    weight_decay: float = 0.01  # AdamW default: shrinks weights a little every step
    batch_size: int = 64
    epochs: int = 100  # upper limit; early stopping usually ends sooner
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    class_weights: bool = False  # weight the loss by n / (K · count), fitted on train only
    seed: int = 42
    patience: int = 10  # epochs without a lower validation loss before stopping


@dataclass
class TorchFitResult:
    """A trained model (holding its best weights, in eval mode) and its per-epoch history."""

    model: MLPClassifier
    best_epoch: int  # epoch whose weights the model holds (0 = before training)
    stopped_epoch: int  # last epoch that ran
    stopped_early: bool  # True when patience ran out (even if that was the last epoch)
    # Index 0 is before training; index k is after epoch k. All on the full splits.
    # "val_score" is F1 (binary) or macro-F1 (multi-class).
    history: dict[str, list[float]] = field(default_factory=dict)


# --- Data ----------------------------------------------------------------------------------


class FlowDataset(Dataset):
    """Rows of one split as tensors. The Java analogy: a read-only `List<(x, y)>`.

    A `Dataset` only needs `__len__` and `__getitem__`; the DataLoader asks it for single rows
    and stacks them into batches. Binary labels are float32 with shape (1,) per row, so a batch
    is (B, 1) like the binary logits (BCEWithLogitsLoss needs equal shapes). Multi-class labels
    are int64 class indices with shape () per row, a batch (B,), as CrossEntropyLoss expects.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        """x: (n, d) features; y: (n,) labels (0/1, or class indices for n_classes > 2)."""
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # (n, d)
        if n_classes == 2:
            self.y = torch.from_numpy(y.astype(np.float32).reshape(-1, 1))  # (n, 1)
        else:
            self.y = torch.from_numpy(y.astype(np.int64))  # (n,)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


def make_loader(dataset: FlowDataset, batch_size: int, seed: int) -> DataLoader:
    """Shuffled mini-batches in a reproducible order.

    The DataLoader draws its shuffle order from its own `torch.Generator`, seeded here, so the
    order does not depend on how many other random numbers were drawn before. `num_workers=0`
    loads batches in the main process: no worker processes with their own random state.
    The generator advances every epoch, so each epoch gets a new (but repeatable) order.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Balanced class weights n / (K · count_c), like sklearn's class_weight="balanced".

    Pass TRAINING labels only (CLAUDE.md: fit every preprocessing step on the training split).
    A rare class gets a large weight, so each class contributes the same total weight to the
    loss. O(n) time.

    Raises:
        ValueError: a class has no training rows (its weight would be infinite).
    """
    counts = np.bincount(y, minlength=n_classes)  # (K,)
    if np.any(counts == 0):
        raise ValueError(f"every class needs training rows, counts {counts.tolist()}")
    return torch.tensor(y.size / (n_classes * counts), dtype=torch.float32)  # (K,)


def expected_first_loss(y: np.ndarray, n_classes: int, weights: torch.Tensor | None) -> float:
    """The loss of a model that predicts 1/K for every class (what our init gives at first).

    Each row's loss is then ln K. Unweighted losses and weighted CrossEntropyLoss (which
    divides by the sum of the row weights) average to ln K. BCEWithLogitsLoss with pos_weight
    divides by B instead, so attack rows count pos_weight times: ln 2 · (n0 + pw · n1) / n.
    """
    if n_classes > 2 or weights is None:
        return math.log(n_classes)
    pos_weight = float(weights[1] / weights[0])
    n1 = int(y.sum())
    return math.log(2) * ((y.size - n1) + pos_weight * n1) / y.size


def make_loss(n_classes: int, weights: torch.Tensor | None = None) -> nn.Module:
    """BCEWithLogitsLoss for binary, CrossEntropyLoss for multi-class; both take raw logits.

    With weights: CrossEntropyLoss(weight=w) multiplies each row's loss by w[label] and divides
    by the sum of those weights. For binary, BCEWithLogitsLoss(pos_weight=w1/w0) gives the
    attack rows the same relative weight.
    """
    if n_classes == 2:
        pos_weight = None if weights is None else (weights[1] / weights[0]).reshape(1)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.CrossEntropyLoss(weight=weights)


# --- Training ------------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> float:
    """One update on one batch: zero_grad → forward → loss → backward → step.

    Args:
        xb: (B, d) features.
        yb: (B, 1) float 0/1 for binary, or (B,) int64 class indices for multi-class.

    Returns:
        The batch loss before the update, as a Python float.

    Time O(B · P) for P parameters (forward and backward each cost about the same).
    """
    opt.zero_grad()  # PyTorch ADDS new gradients to p.grad; clear the last batch's first
    logits = model(xb)  # (B, 1) or (B, K); autograd records every operation
    loss = loss_fn(logits, yb)  # 0-dim tensor: the mean loss over the batch
    loss.backward()  # chain rule through the recorded graph: fills p.grad for every p
    opt.step()  # AdamW reads p.grad and updates every p in place
    return loss.item()  # .item(): tensor -> float, detached from the graph


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """A real copy of the model's weights.

    `model.state_dict()` returns the model's own tensors (shared memory, like the NumPy
    `parameters()` in T06): they keep changing while training continues. `.clone()` copies
    the values. O(P) time and memory.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def evaluate_loss(model: nn.Module, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean loss of the current weights on a full split, in eval mode, without autograd."""
    model.eval()  # dropout off
    with torch.no_grad():  # no graph is recorded: faster, less memory
        return loss_fn(model(x), y).item()


def predict_logits(model: nn.Module, x: np.ndarray) -> torch.Tensor:
    """Logits in eval mode. x: (n, d) -> (n, 1) or (n, K)."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))


def predict(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    """(n,) int64: 0/1 for binary (logit >= 0, probability >= 0.5), else the argmax class."""
    logits = predict_logits(model, x)
    if model.n_classes == 2:
        return (logits[:, 0] >= 0).numpy().astype(np.int64)
    return logits.argmax(dim=1).numpy().astype(np.int64)


def _score(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> float:
    """F1 (binary) or macro-F1 (multi-class) at the default decision rule."""
    pred = predict(model, x)
    return f1(y, pred) if model.n_classes == 2 else macro_f1(y, pred, model.n_classes)


def fit(
    config: TorchConfig,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> TorchFitResult:
    """Train an MLPClassifier with AdamW and early stopping on the validation loss.

    Class weights (if enabled) are computed from `y`, the training labels. The validation
    split is never trained on; it only picks the epoch whose weights are kept.
    Time O(epochs · (n + n_val) · P).
    """
    set_seed(config.seed)  # seeds torch: weight init and dropout masks
    # Deterministic kernels only (an error instead of a silently random one). The setting is
    # global, so the previous value is restored at the end and later code is not affected.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _fit(config, x, y, x_val, y_val, n_classes)
    finally:
        torch.use_deterministic_algorithms(previous)


def _fit(
    config: TorchConfig,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> TorchFitResult:
    """The body of `fit`, run with deterministic algorithms on."""
    model = MLPClassifier(x.shape[1], config.hidden, n_classes, config.dropout)
    weights = class_weights(y, n_classes) if config.class_weights else None
    loss_fn = make_loss(n_classes, weights)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_ds = FlowDataset(x, y, n_classes)
    val_ds = FlowDataset(x_val, y_val, n_classes)
    loader = make_loader(train_ds, config.batch_size, config.seed)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_score": []}

    def record() -> float:
        history["train_loss"].append(evaluate_loss(model, loss_fn, train_ds.x, train_ds.y))
        val_loss = evaluate_loss(model, loss_fn, val_ds.x, val_ds.y)
        history["val_loss"].append(val_loss)
        history["val_score"].append(_score(model, x_val, y_val))
        return val_loss

    best_loss, best_epoch, best_state = record(), 0, clone_state(model)
    epoch, waited, stopped_early = 0, 0, False
    for epoch in range(1, config.epochs + 1):
        model.train()  # dropout on (evaluate_loss switched to eval mode)
        for xb, yb in loader:  # xb: (B, d), yb: (B, 1) or (B,)
            train_step(model, loss_fn, opt, xb, yb)
        val_loss = record()
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, clone_state(model)
            waited = 0
        else:
            waited += 1
            if waited >= config.patience:
                stopped_early = True
                break
    model.load_state_dict(best_state)  # copies the kept values back into the model
    model.eval()
    return TorchFitResult(model, best_epoch, epoch, stopped_early, history)


# --- Saving and loading --------------------------------------------------------------------


def save_model(path: Path, model: MLPClassifier, meta: dict) -> None:
    """`state_dict` to `path` (.pt) and the metadata to `path` with a .json suffix.

    The .pt file holds only named tensors, so it loads with `weights_only=True`. Saving the
    whole model object would need pickle, and loading a pickle can run arbitrary code.
    `meta` must hold "class_names" and "input_sha256"; layer sizes, n_classes, dropout, and
    the schema version are added here and cannot be overridden by `meta`.

    Raises:
        ValueError: `meta` holds one of the keys this function sets.
    """
    fixed = {
        "model": "mlp_classifier",
        "layer_sizes": model.layer_sizes,
        "n_classes": model.n_classes,
        "dropout": model.dropout,
        "schema_version": SCHEMA_VERSION,
    }
    clash = fixed.keys() & meta.keys()
    if clash:
        raise ValueError(f"meta must not set {sorted(clash)}: they come from the model and code")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    full = {**fixed, **meta}
    path.with_suffix(".json").write_text(json.dumps(full, indent=2) + "\n")


def load_model(path: Path, input_sha256: str | None = None) -> tuple[MLPClassifier, dict]:
    """Rebuild a model saved by `save_model`; returns (model in eval mode, metadata).

    Raises:
        SchemaError: other schema version, other input CSV (when `input_sha256` is given), or
            an architecture that does not fit the feature contract and the class names.
        RuntimeError: the state_dict keys or shapes do not match (from load_state_dict).
    """
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("model") != "mlp_classifier":
        raise SchemaError(f"{path}: model type {meta.get('model')!r} is not 'mlp_classifier'")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{path}: schema_version {meta.get('schema_version')!r} != "
                          f"code SCHEMA_VERSION {SCHEMA_VERSION!r}")  # fmt: skip
    if input_sha256 is not None and meta.get("input_sha256") != input_sha256:
        raise SchemaError(f"{path}: weights were trained on another input CSV")
    sizes, n_classes = meta["layer_sizes"], meta["n_classes"]
    n_out = 1 if n_classes == 2 else n_classes
    if (
        sizes[0] != len(FEATURE_NAMES)
        or sizes[-1] != n_out
        or len(meta["class_names"]) != (n_classes)
    ):
        raise SchemaError(f"{path}: layer_sizes {sizes} do not fit {len(FEATURE_NAMES)} inputs "
                          f"and classes {meta['class_names']}")  # fmt: skip
    model = MLPClassifier(sizes[0], tuple(sizes[1:-1]), n_classes, meta["dropout"])
    # weights_only=True: only tensors and plain containers are accepted, never code.
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)  # strict: every key present, every shape equal
    return model.eval(), meta


# --- Figures -------------------------------------------------------------------------------


def save_curves(result: TorchFitResult, title: str, score_name: str, path: Path) -> None:
    """Train and validation loss (best epoch marked) and the validation score, per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = result.history
    epochs = range(len(h["train_loss"]))
    best = result.best_epoch
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, h["train_loss"], marker=".", label="train loss")
    ax_loss.plot(epochs, h["val_loss"], marker=".", label="validation loss")
    ax_loss.axvline(best, color="gray", linestyle="--", linewidth=1)
    ax_loss.plot([best], [h["val_loss"][best]], marker="*", markersize=14, color="red",
                 linestyle="none", label=f"best epoch {best} (weights kept)")  # fmt: skip
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("mean loss")
    ax_loss.set_title(f"PyTorch MLP loss ({title})")
    ax_loss.legend()
    ax_score.plot(epochs, h["val_score"], marker=".", color="green")
    ax_score.axvline(best, color="gray", linestyle="--", linewidth=1)
    ax_score.set_xlabel("epoch")
    ax_score.set_ylabel(score_name)
    ax_score.set_title(f"Validation {score_name} ({title})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Reports -------------------------------------------------------------------------------


def _flipped_mask(val: Split, meta_path: Path) -> tuple[np.ndarray | None, str | None]:
    """(n_val,) bool label-noise mask for SYNTHETIC data, or None and the reason. Eval only."""
    if val.source != "SYNTHETIC" or not meta_path.exists():
        return None, "data is not SYNTHETIC or synthetic.json is missing"
    ids = read_flipped_ids(meta_path, val.input_sha256)
    if ids is None:
        return None, f"{meta_path} and the arrays come from different runs"
    return np.isin(val.flow_id, list(ids)), None


def multiclass_report(
    val: Split, pred: np.ndarray, flipped: np.ndarray | None
) -> tuple[dict[str, dict], dict]:
    """Per-class metrics (with the recall ceiling when `flipped` is given) and the summary."""
    k = len(val.class_names)
    rep = per_class_report(val.y_class, pred, k)
    ceiling = (
        recall_ceiling_per_class(val.y_class, flipped, k)
        if flipped is not None
        else np.full(k, np.nan)
    )
    per_class = {
        name: {
            "precision": float(rep["precision"][i]),
            "recall": float(rep["recall"][i]),
            "recall_ceiling": float(ceiling[i]),
            "f1": float(rep["f1"][i]),
            "support": int(rep["support"][i]),
        }
        for i, name in enumerate(val.class_names)
    }
    summary = {
        "macro_f1": float(rep["f1"].mean()),
        "accuracy": float(np.mean(pred == val.y_class)),
        "confusion": multiclass_confusion(val.y_class, pred, k),
    }
    return per_class, summary


def _print_multiclass(
    per_class: dict[str, dict], summary: dict, note: str | None, tag: str, weighted: bool
) -> None:
    print(f"\nValidation, multi-class{tag} · class weights: {'on' if weighted else 'off'}")
    print(f"{'class':<16}{'precision':>10}{'recall':>8}{'ceiling*':>10}{'f1':>7}{'support':>9}")
    for name, m in per_class.items():
        ceil = "n/a" if math.isnan(m["recall_ceiling"]) else f"{m['recall_ceiling']:.3f}"
        print(f"{name:<16}{m['precision']:>10.3f}{m['recall']:>8.3f}{ceil:>10}"
              f"{m['f1']:>7.3f}{m['support']:>9}")  # fmt: skip
    print(f"macro-F1 {summary['macro_f1']:.3f} · accuracy {summary['accuracy']:.3f}")
    if note is None:
        print("* recall ceiling: a perfect model scored against the noisy val labels "
              "(flipped_flow_ids; evaluation only)")  # fmt: skip
    else:
        print(f"* no recall ceiling: {note}")
    names = list(per_class)
    print("confusion matrix (rows = label, columns = prediction):")
    print(" " * 16 + "".join(f"{n[:14]:>16}" for n in names))
    for name, row in zip(names, summary["confusion"], strict=True):
        print(f"{name:<16}" + "".join(f"{v:>16}" for v in row))


# --- Command -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the command; returns the results (used by the tests)."""
    d = TorchConfig()
    parser = argparse.ArgumentParser(description="Train the PyTorch MLP classifier.")
    parser.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--synthetic-meta", type=Path,
                        default=Path("data/processed/synthetic.json"))  # fmt: skip
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--numpy-model", type=Path, default=Path("models/mlp_numpy.npz"),
                        help="T06 NumPy MLP for the binary comparison table")  # fmt: skip
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(d.hidden))
    parser.add_argument("--dropout", type=float, default=d.dropout)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--patience", type=int, default=d.patience)
    args = parser.parse_args(argv)
    config = TorchConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        class_weights=args.class_weights,
        seed=args.seed,
        patience=args.patience,
    )

    train = load_split(args.data_dir / "train.npz")
    val = load_split(args.data_dir / "val.npz")
    _check_same_source(train, val)
    binary = args.target == "binary"
    n_classes = 2 if binary else len(train.class_names)
    class_names = BINARY_CLASS_NAMES if binary else train.class_names
    y, y_val = (train.y_binary, val.y_binary) if binary else (train.y_class, val.y_class)
    tag = " (SYNTHETIC: tests code, not detection)" if val.source == "SYNTHETIC" else ""
    print(f"data source: {val.source} · target: {args.target} · classes: {list(class_names)}")
    counts = np.bincount(y, minlength=n_classes).tolist()
    counts_val = np.bincount(y_val, minlength=n_classes).tolist()
    print(f"train {train.x.shape}, counts {counts} · val {val.x.shape}, counts {counts_val}")
    print(f"config: {asdict(config)}")
    if config.class_weights:
        weights = [round(float(w), 4) for w in class_weights(y, n_classes)]
        print(f"class weights (train split): {weights}")

    # --- Training: train split only; val picks the stopping epoch.
    result = fit(config, train.x, y, val.x, y_val, n_classes)
    h = result.history
    score_name = "F1" if binary else "macro-F1"
    weights_t = class_weights(y, n_classes) if config.class_weights else None
    expected = expected_first_loss(y, n_classes, weights_t)
    print(f"first train loss {h['train_loss'][0]:.4f} (expected for 1/K predictions: "
          f"{expected:.4f}; ln {n_classes} = {math.log(n_classes):.4f})")  # fmt: skip
    print(f"\n{'epoch':>5}{'train_loss':>12}{'val_loss':>10}{'val_' + score_name:>14}")
    for k in range(len(h["train_loss"])):
        mark = "  <- best" if k == result.best_epoch else ""
        print(f"{k:>5}{h['train_loss'][k]:>12.4f}{h['val_loss'][k]:>10.4f}"
              f"{h['val_score'][k]:>14.3f}{mark}")  # fmt: skip
    stop = "early stop" if result.stopped_early else "epoch limit"
    print(f"stopped after epoch {result.stopped_epoch} ({stop}); kept epoch {result.best_epoch}")

    stem = f"mlp_torch_{args.target}" + ("_weighted" if config.class_weights else "")
    model_path = args.models_dir / f"{stem}.pt"
    save_model(model_path, result.model, {
        "class_names": list(class_names), "input_sha256": train.input_sha256,
        "source": train.source, "target": args.target, "config": asdict(config),
        "best_epoch": result.best_epoch, "stopped_epoch": result.stopped_epoch,
    })  # fmt: skip
    figure = args.figures_dir / f"{stem}_loss.png"
    weighted = ", class weights" if config.class_weights else ""
    save_curves(result, f"{args.target}{weighted}, {val.source}", score_name, figure)

    # --- Evaluation on the validation split.
    pred = predict(result.model, val.x)
    out: dict = {"history": h, "best_epoch": result.best_epoch,
                 "stopped_epoch": result.stopped_epoch}  # fmt: skip
    if binary:
        numpy_mlp = None
        if args.numpy_model.exists():
            numpy_mlp = load_mlp(args.numpy_model, input_sha256=train.input_sha256)
        else:
            print(f"(no NumPy MLP row: {args.numpy_model} not found; run neural_ids.train_numpy)")
        models = compare_on_val(train, val, numpy_mlp, config.seed, extra={"mlp_torch": pred})
        ceiling, note = _ceiling(val, args.synthetic_meta)
        _print_table(models, ceiling, note, tag)
        out.update(models=models, ceiling=ceiling)
    else:
        flipped, note = _flipped_mask(val, args.synthetic_meta)
        per_class, summary = multiclass_report(val, pred, flipped)
        _print_multiclass(per_class, summary, note, tag, config.class_weights)
        print("Note: the validation split also chose the stopping epoch, so these numbers are "
              "slightly optimistic.")  # fmt: skip
        out.update(per_class=per_class, summary=summary)
    print(f"\nweights -> {model_path} (+ .json) · figure -> {figure}")
    return out


if __name__ == "__main__":
    main()
