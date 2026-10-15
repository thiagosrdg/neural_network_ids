"""Train the NumPy MLP with mini-batch SGD with momentum and early stopping.

Command:
    uv run python -m neural_ids.train_numpy            # train, save, compare on validation
    uv run python -m neural_ids.train_numpy --sanity   # only the two debugging checks

Reads `data/processed/{train,val}.npz` (never test.npz: the test split is used once, at the end
of an experiment). Saves the best weights to `models/mlp_numpy.npz` (training settings in
`models/mlp_numpy.json`) and the curves to `reports/figures/mlp_numpy_loss.png`, then prints a
validation table: logistic regression, the MLP, a scikit-learn random forest (a reference for
how far a nonlinear model gets on this data), and, for SYNTHETIC data, the label-noise ceiling.

The validation split also chooses the stopping epoch, so the MLP's validation numbers are
slightly optimistic. The test split stays unused until the final comparison.
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from neural_ids.metrics import (  # noqa: E402
    binary_report,
    confusion_matrix,
    f1,
    label_noise_ceiling,
)
from neural_ids.models.batching import minibatch_indices  # noqa: E402
from neural_ids.models.logreg import LogisticRegression  # noqa: E402
from neural_ids.nn.functional import bce_loss, sigmoid  # noqa: E402
from neural_ids.nn.layers import BCEWithLogits  # noqa: E402
from neural_ids.nn.model import MLP  # noqa: E402
from neural_ids.nn.optim import SGD  # noqa: E402
from neural_ids.schema import FEATURE_NAMES, SCHEMA_VERSION, SchemaError  # noqa: E402
from neural_ids.splits import Split, load_split  # noqa: E402
from neural_ids.train_baselines import read_flipped_ids  # noqa: E402
from neural_ids.utils import set_seed  # noqa: E402

SANITY_BATCH = 64  # rows in the overfit-one-batch check
SANITY_FIRST_LOSS_TOL = 0.01  # |first loss − ln 2| must be below this
SANITY_OVERFIT_MAX = 0.02  # "near zero" for the one-batch loss after training


@dataclass(frozen=True)
class TrainConfig:
    """Settings of one MLP training run. Saved next to the weights as JSON."""

    lr: float = 0.05
    momentum: float = 0.9
    batch_size: int = 64
    epochs: int = 100  # upper limit; early stopping usually ends sooner
    hidden: tuple[int, ...] = (64, 32)
    seed: int = 42
    patience: int = 10  # epochs without a lower validation loss before stopping


@dataclass
class FitResult:
    """A trained model (holding its best weights) and its per-epoch history."""

    model: MLP
    best_epoch: int  # epoch whose weights the model holds (0 = before training)
    stopped_epoch: int  # last epoch that ran
    # Index 0 is before training; index k is after epoch k. All on the full splits.
    history: dict[str, list[float]] = field(default_factory=dict)


# --- Training ------------------------------------------------------------------------------


def train_epoch(
    model: MLP,
    loss_fn: BCEWithLogits,
    opt: SGD,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """One pass over (x, y) in shuffled mini-batches: forward → loss → backward → step.

    Args:
        x: (n, d) training features.
        y: (n,) labels in {0, 1}.
        rng: reshuffles the rows; a new order every epoch.

    Returns:
        The mean of the batch losses, weighted by batch size (the last batch may be smaller).
        Each batch loss is measured before that batch's step.

    Time O(n · P) for P parameters; memory O(B · widest layer) for the cached activations.
    """
    total = 0.0
    for idx in minibatch_indices(x.shape[0], batch_size, rng):
        logits = model.forward(x[idx])  # (B, d) -> (B, 1)
        loss = loss_fn.forward(logits, y[idx])  # float, mean over the batch
        model.backward(loss_fn.backward())  # fills every dW, db
        opt.step(model.gradients())  # read fresh: backward stored new arrays
        total += loss * idx.size
    return total / x.shape[0]


def predict_proba(model: MLP, x: np.ndarray) -> np.ndarray:
    """Attack probability. x: (n, d) -> (n,)."""
    return sigmoid(model.forward(x).reshape(-1))  # (n, 1) -> (n,)


def predict(model: MLP, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Labels in {0, 1}. x: (n, d) -> (n,) int64."""
    return (predict_proba(model, x) >= threshold).astype(np.int64)


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(mean BCE, F1 at threshold 0.5) of the current weights on a full split."""
    logits = model.forward(x).reshape(-1)  # (n,)
    pred = (logits >= 0.0).astype(np.int64)  # logit >= 0  <=>  probability >= 0.5
    return bce_loss(logits, y), f1(y, pred)


def snapshot(model: MLP) -> list[np.ndarray]:
    """Copies of every parameter (the model keeps training on its own arrays)."""
    return [p.copy() for p in model.parameters()]


def restore(model: MLP, saved: list[np.ndarray]) -> None:
    """Write saved values back into the model's own arrays, in place."""
    for p, s in zip(model.parameters(), saved, strict=True):
        p[...] = s


def fit(
    config: TrainConfig, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> FitResult:
    """Train an MLP on (x, y) with early stopping on the validation loss.

    The validation split is never trained on; it only picks the epoch whose weights are kept.
    Time O(epochs · (n + n_val) · P).
    """
    init_rng, shuffle_rng = set_seed(config.seed).spawn(2)  # independent streams
    model = MLP([x.shape[1], *config.hidden, 1], init_rng)
    loss_fn = BCEWithLogits()
    opt = SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}

    def record() -> float:
        history["train_loss"].append(evaluate(model, x, y)[0])
        val_loss, val_f1 = evaluate(model, x_val, y_val)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        return val_loss

    best_loss, best_epoch, best_weights = record(), 0, snapshot(model)
    epoch, waited = 0, 0
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, loss_fn, opt, x, y, config.batch_size, shuffle_rng)
        val_loss = record()
        if val_loss < best_loss:
            best_loss, best_epoch, best_weights = val_loss, epoch, snapshot(model)
            waited = 0
        else:
            waited += 1
            if waited >= config.patience:
                break
    restore(model, best_weights)
    return FitResult(model, best_epoch, epoch, history)


# --- Sanity checks -------------------------------------------------------------------------


def first_loss(x: np.ndarray, y: np.ndarray, seed: int, hidden: tuple[int, ...]) -> float:
    """Mean BCE of a freshly initialized MLP on (x, y). Expected: about ln 2."""
    init_rng, _ = set_seed(seed).spawn(2)  # the same init stream as `fit`
    model = MLP([x.shape[1], *hidden, 1], init_rng)
    return evaluate(model, x, y)[0]


def overfit_one_batch(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden: tuple[int, ...],
    steps: int = 2000,
    lr: float = 0.1,
    momentum: float = 0.9,
) -> tuple[float, float]:
    """Train on SANITY_BATCH random rows only, full-batch, and return (loss before, loss after).

    A working loop memorizes 64 rows (the loss goes near 0). If it cannot, the bug is in the
    model or the loop, not in the data. A batch can hold a flipped label next to a near copy
    with the other label; the network then needs many steps to separate the two, which is why
    the default is 2000 steps. Time O(steps · 64 · P).
    """
    rng = set_seed(seed)
    idx = rng.choice(x.shape[0], size=SANITY_BATCH, replace=False)  # (64,)
    xb, yb = x[idx], y[idx]  # (64, d), (64,)
    model = MLP([x.shape[1], *hidden, 1], rng)
    loss_fn = BCEWithLogits()
    opt = SGD(model.parameters(), lr=lr, momentum=momentum)
    before = evaluate(model, xb, yb)[0]
    for _ in range(steps):
        loss_fn.forward(model.forward(xb), yb)
        model.backward(loss_fn.backward())
        opt.step(model.gradients())
    return before, evaluate(model, xb, yb)[0]


# --- Saving and loading --------------------------------------------------------------------


def save_mlp(path: Path, model: MLP, schema_version: str, input_sha256: str) -> None:
    """Weights plus the metadata `load_mlp` checks. Plain arrays only: no pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"param_{i}": p for i, p in enumerate(model.parameters())}  # W1, b1, W2, ...
    np.savez(
        path,
        layer_sizes=np.array(model.layer_sizes, dtype=np.int64),
        schema_version=np.array(schema_version),
        input_sha256=np.array(input_sha256),
        **arrays,
    )


def load_mlp(path: Path, input_sha256: str | None = None) -> MLP:
    """Rebuild an MLP saved by `save_mlp`.

    Args:
        path: the `.npz` file.
        input_sha256: if given, the hash of the flows CSV the caller's arrays came from; it
            must equal the one stored with the weights.

    Raises:
        SchemaError: schema_version differs from the code, input_sha256 differs, or the
            input size does not match the feature contract.
        ValueError: missing or wrongly shaped arrays, or Python objects in the file.
    """
    with np.load(path, allow_pickle=False) as data:
        for key in ("layer_sizes", "schema_version", "input_sha256"):
            if key not in data.files:
                raise ValueError(f"{path}: missing {key!r}")
        version = str(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"{path}: schema_version {version!r} != code SCHEMA_VERSION {SCHEMA_VERSION!r}"
            )
        stored_sha = str(data["input_sha256"])
        if input_sha256 is not None and stored_sha != input_sha256:
            raise SchemaError(f"{path}: weights were trained on another input CSV")
        sizes = data["layer_sizes"]
        if sizes.ndim != 1 or sizes.size < 2:
            raise ValueError(f"{path}: layer_sizes must be a 1-D list of at least 2 sizes")
        if int(sizes[0]) != len(FEATURE_NAMES) or int(sizes[-1]) != 1:
            raise SchemaError(
                f"{path}: layer_sizes {sizes.tolist()} must start with {len(FEATURE_NAMES)} "
                "inputs (the feature contract) and end with 1 output"
            )
        model = MLP([int(n) for n in sizes], np.random.default_rng(0))
        params = model.parameters()
        n_saved = sum(key.startswith("param_") for key in data.files)
        if n_saved != len(params):
            raise ValueError(f"{path}: {n_saved} parameter arrays, model needs {len(params)}")
        for i, p in enumerate(params):
            saved = data[f"param_{i}"]
            if saved.shape != p.shape:
                raise ValueError(f"{path}: param_{i} has shape {saved.shape}, expected {p.shape}")
            p[...] = saved  # in place: the random init is overwritten
    return model


# --- Figures -------------------------------------------------------------------------------


def save_curves(result: FitResult, source: str, path: Path) -> None:
    """Train and validation loss (best epoch marked) and validation F1, per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = result.history
    epochs = range(len(h["train_loss"]))
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, h["train_loss"], marker=".", label="train loss")
    ax_loss.plot(epochs, h["val_loss"], marker=".", label="validation loss")
    best = result.best_epoch
    ax_loss.axvline(best, color="gray", linestyle="--", linewidth=1)
    ax_loss.plot([best], [h["val_loss"][best]], marker="*", markersize=14, color="red",
                 linestyle="none", label=f"best epoch {best} (weights kept)")  # fmt: skip
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("mean binary cross-entropy")
    ax_loss.set_title(f"NumPy MLP loss ({source})")
    ax_loss.legend()
    ax_f1.plot(epochs, h["val_f1"], marker=".", color="green")
    ax_f1.axvline(best, color="gray", linestyle="--", linewidth=1)
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("F1 at threshold 0.5")
    ax_f1.set_title(f"Validation F1 ({source})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Command -------------------------------------------------------------------------------


def _check_same_source(train: Split, val: Split) -> None:
    if (train.class_names, train.source, train.input_sha256) != (
        val.class_names,
        val.source,
        val.input_sha256,
    ):
        raise ValueError(
            "train and val differ in class_names, source, or input CSV; rebuild the arrays"
        )


def run_sanity(train: Split, config: TrainConfig) -> dict[str, float]:
    """Print and return both debugging checks on the training split."""
    loss0 = first_loss(train.x, train.y_binary, config.seed, config.hidden)
    before, after = overfit_one_batch(train.x, train.y_binary, config.seed, config.hidden)
    ok1 = abs(loss0 - math.log(2)) < SANITY_FIRST_LOSS_TOL
    ok2 = after < SANITY_OVERFIT_MAX
    print(f"sanity 1 - first loss {loss0:.4f} (ln 2 = {math.log(2):.4f}, "
          f"tolerance {SANITY_FIRST_LOSS_TOL}): {'PASS' if ok1 else 'FAIL'}")  # fmt: skip
    print(f"sanity 2 - one batch of {SANITY_BATCH}: loss {before:.4f} -> {after:.6f} "
          f"(limit {SANITY_OVERFIT_MAX}): {'PASS' if ok2 else 'FAIL'}")  # fmt: skip
    return {"first_loss": loss0, "overfit_before": before, "overfit_after": after,
            "passed": ok1 and ok2}  # fmt: skip


def compare_on_val(
    train: Split,
    val: Split,
    mlp: MLP | None,
    seed: int,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Fit logistic regression and a random forest on train; score them and the MLP on val.

    Args:
        mlp: the NumPy MLP, or None to leave its row out.
        extra: more models' validation predictions, (n_val,) 0/1 each, shown after the MLP
            (T08 adds the PyTorch MLP here).
    """
    # Same random stream and settings as `train_baselines`, so this row matches T04.
    _, logreg_rng = set_seed(seed).spawn(2)
    logreg = LogisticRegression(train.x.shape[1], lr=0.1, epochs=30, batch_size=64).fit(
        train.x, train.y_binary, val.x, val.y_binary, logreg_rng
    )
    forest = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    forest.fit(train.x, train.y_binary)
    preds = {"logistic_regression": logreg.predict(val.x)}
    if mlp is not None:
        preds["mlp_numpy"] = predict(mlp, val.x)
    preds.update(extra or {})
    preds["random_forest (ref)"] = forest.predict(val.x).astype(np.int64)
    return {
        name: {**binary_report(val.y_binary, p), "confusion": confusion_matrix(val.y_binary, p)}
        for name, p in preds.items()
    }


def _ceiling(val: Split, meta_path: Path) -> tuple[dict | None, str | None]:
    if val.source != "SYNTHETIC" or not meta_path.exists():
        return None, "data is not SYNTHETIC or synthetic.json is missing"
    flipped_ids = read_flipped_ids(meta_path, val.input_sha256)
    if flipped_ids is None:
        return None, f"{meta_path} and the arrays come from different runs"
    flipped = np.isin(val.flow_id, list(flipped_ids))  # (n_val,) bool
    return {**label_noise_ceiling(val.y_binary, flipped), "n_flipped": int(flipped.sum())}, None


def _print_table(models: dict[str, dict], ceiling: dict | None, note: str | None, tag: str) -> None:
    print(f"\nValidation metrics, attack = positive class{tag}")
    print(f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}"
          f"{'TN':>7}{'FP':>6}{'FN':>6}{'TP':>6}")  # fmt: skip
    for name, m in models.items():
        (tn, fp), (fn, tp) = m["confusion"]
        print(f"{name:<22}{m['accuracy']:>10.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>8.3f}{tn:>7}{fp:>6}{fn:>6}{tp:>6}")  # fmt: skip
    if ceiling is not None:
        print(f"{'noise ceiling*':<22}{'-':>10}{ceiling['precision']:>11.3f}"
              f"{ceiling['recall']:>9.3f}{'-':>8}")  # fmt: skip
        print(f"* a perfect model scored against the noisy val labels "
              f"({ceiling['n_flipped']} flipped labels in val; evaluation only)")  # fmt: skip
    else:
        print(f"(no label-noise ceiling: {note})")
    print("Note: the validation split also chose the MLP's stopping epoch, so its validation")
    print(
        "numbers are slightly optimistic. The test split stays unused until the final comparison."
    )


def main(argv: list[str] | None = None) -> dict:
    """Run the command; returns the results (used by the tests)."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the NumPy MLP (SGD with momentum).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--synthetic-meta", type=Path,
                        default=Path("data/processed/synthetic.json"))  # fmt: skip
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--model-out", type=Path, default=Path("models/mlp_numpy.npz"))
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(defaults.hidden))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--sanity", action="store_true", help="run only the two sanity checks")
    args = parser.parse_args(argv)
    config = TrainConfig(
        lr=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden=tuple(args.hidden),
        seed=args.seed,
        patience=args.patience,
    )

    train = load_split(args.data_dir / "train.npz")
    if args.sanity:
        print(f"data source: {train.source} · train {train.x.shape} · seed {config.seed}")
        return {"sanity": run_sanity(train, config)}

    val = load_split(args.data_dir / "val.npz")
    _check_same_source(train, val)
    tag = " (SYNTHETIC: tests code, not detection)" if val.source == "SYNTHETIC" else ""
    print(f"data source: {val.source}")
    print(f"train {train.x.shape}, attack {train.y_binary.mean():.1%} · "
          f"val {val.x.shape}, attack {val.y_binary.mean():.1%}")  # fmt: skip
    print(f"config: {asdict(config)}")

    # --- Training: train split only; val picks the stopping epoch.
    result = fit(config, train.x, train.y_binary, val.x, val.y_binary)
    h = result.history
    print(f"\n{'epoch':>5}{'train_loss':>12}{'val_loss':>10}{'val_f1':>8}")
    for k in range(len(h["train_loss"])):
        mark = "  <- best" if k == result.best_epoch else ""
        print(f"{k:>5}{h['train_loss'][k]:>12.4f}{h['val_loss'][k]:>10.4f}"
              f"{h['val_f1'][k]:>8.3f}{mark}")  # fmt: skip
    stop = "early stop" if result.stopped_epoch < config.epochs else "epoch limit"
    print(f"stopped after epoch {result.stopped_epoch} ({stop}); kept epoch {result.best_epoch}")

    save_mlp(args.model_out, result.model, SCHEMA_VERSION, train.input_sha256)
    settings = {**asdict(config), "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch, "source": train.source}  # fmt: skip
    args.model_out.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n")
    figure = args.figures_dir / "mlp_numpy_loss.png"
    save_curves(result, val.source, figure)

    # --- Evaluation on the validation split.
    models = compare_on_val(train, val, result.model, config.seed)
    ceiling, note = _ceiling(val, args.synthetic_meta)
    _print_table(models, ceiling, note, tag)
    print(f"\nweights -> {args.model_out} · figure -> {figure}")
    return {
        "models": models,
        "ceiling": ceiling,
        "history": h,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
    }


if __name__ == "__main__":
    outcome = main()
    if "sanity" in outcome and not outcome["sanity"]["passed"]:
        raise SystemExit(1)  # non-zero exit, so scripts can use --sanity as a guard
