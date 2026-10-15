"""Train the NumPy autoencoder on normal flows and use it as an anomaly detector.

Command:
    uv run python -m neural_ids.train_autoencoder_numpy             # train, save, score test once
    uv run python -m neural_ids.train_autoencoder_numpy --val-only  # tune: never opens test.npz

"Normal" means y_binary == 0. Those rows include a few attacks that label noise marked as
normal; they stay in, because real "normal" captures are contaminated too, and the label-noise
answer key (`flipped_flow_ids`) is used for evaluation only.

No attack label is used before the final test scoring:
- the inputs are standardized again with statistics of normal training rows only
  (`anomaly.NormalScaler`; the T03 preprocessor saw attack rows too);
- training uses the normal rows of the training split only;
- early stopping uses the reconstruction loss of the normal validation rows;
- the threshold is the 99th percentile of the errors of the normal validation rows.
So tuning (with `--val-only`) looks only at normal data, as it would for a person who has
captured only their own normal traffic. Saves `models/ae_numpy.npz` (weights and threshold),
`models/ae_numpy.json` (settings), `reports/figures/ae_numpy_loss.png`, and, after scoring test,
`reports/figures/ae_numpy_errors.png`.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neural_ids.anomaly import (  # noqa: E402
    NORMAL_CLASS,
    NormalScaler,
    detection_table,
    flag,
    reconstruction_error,
    threshold_from_normal,
)
from neural_ids.models.batching import minibatch_indices  # noqa: E402
from neural_ids.nn.autoencoder import Autoencoder  # noqa: E402
from neural_ids.nn.losses import MSELoss  # noqa: E402
from neural_ids.nn.optim import SGD  # noqa: E402
from neural_ids.schema import FEATURE_NAMES, SCHEMA_VERSION, SchemaError  # noqa: E402
from neural_ids.splits import Split, load_split  # noqa: E402
from neural_ids.train_baselines import read_flipped_ids  # noqa: E402
from neural_ids.train_numpy import _check_same_source, restore, snapshot  # noqa: E402
from neural_ids.utils import set_seed  # noqa: E402


@dataclass(frozen=True)
class AEConfig:
    """Settings of one autoencoder run. Saved next to the weights as JSON.

    lr, epochs, and patience were tuned with `--val-only` on the normal validation loss alone;
    the candidates and their losses are in docs/reports/T07-autoencoder.md.
    """

    lr: float = 0.1
    momentum: float = 0.9
    batch_size: int = 64
    epochs: int = 500  # upper limit; early stopping usually ends sooner
    encoder: tuple[int, ...] = (16, 4)  # widths down to the bottleneck; the decoder mirrors it
    seed: int = 42
    patience: int = 20  # epochs without a lower normal validation loss before stopping
    quantile: float = 0.99  # threshold = this quantile of normal validation errors


def layer_sizes(n_features: int, encoder: tuple[int, ...]) -> list[int]:
    """[d, *encoder, *mirror, d]. For d = 29 and (16, 4): [29, 16, 4, 16, 29]."""
    return [n_features, *encoder, *reversed(encoder[:-1]), n_features]


def normal_rows(split: Split) -> np.ndarray:
    """Features of the flows labeled normal (y_binary == 0). (n, d) -> (n_normal, d).

    Label noise is not cleaned: never use `flipped_flow_ids` here.
    """
    return split.x[split.y_binary == 0]


@dataclass
class AEFitResult:
    """A trained autoencoder (holding its best weights) and its per-epoch history."""

    model: Autoencoder
    best_epoch: int  # 0 = before training
    stopped_epoch: int
    # Index 0 is before training; index k is after epoch k. Both on full normal splits.
    history: dict[str, list[float]] = field(default_factory=dict)


# --- Training ------------------------------------------------------------------------------


def train_epoch(
    model: Autoencoder,
    loss_fn: MSELoss,
    opt: SGD,
    x: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """One pass over x in shuffled mini-batches; the target of each batch is the batch itself.

    x: (n, d) normal training flows. Returns the batch losses' mean, weighted by batch size.
    Time O(n · P) for P parameters.
    """
    total = 0.0
    for idx in minibatch_indices(x.shape[0], batch_size, rng):
        xb = x[idx]  # (B, d)
        loss = loss_fn.forward(model.forward(xb), xb)  # reconstruction vs input
        model.backward(loss_fn.backward())  # fills every dW, db
        opt.step(model.gradients())
        total += loss * idx.size
    return total / x.shape[0]


def mse(model: Autoencoder, x: np.ndarray) -> float:
    """Mean reconstruction loss of the current weights on a full array. x: (n, d)."""
    return MSELoss().forward(model.forward(x), x)


def fit(config: AEConfig, x: np.ndarray, x_val: np.ndarray) -> AEFitResult:
    """Train on normal rows x; early stopping on the loss of normal validation rows x_val.

    Time O(epochs · (n + n_val) · P).
    """
    init_rng, shuffle_rng = set_seed(config.seed).spawn(2)  # independent streams
    model = Autoencoder(layer_sizes(x.shape[1], config.encoder), init_rng)
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def record() -> float:
        history["train_loss"].append(mse(model, x))
        history["val_loss"].append(mse(model, x_val))
        return history["val_loss"][-1]

    best_loss, best_epoch, best_weights = record(), 0, snapshot(model)
    epoch, waited = 0, 0
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, loss_fn, opt, x, config.batch_size, shuffle_rng)
        val_loss = record()
        if val_loss < best_loss:
            best_loss, best_epoch, best_weights = val_loss, epoch, snapshot(model)
            waited = 0
        else:
            waited += 1
            if waited >= config.patience:
                break
    restore(model, best_weights)
    return AEFitResult(model, best_epoch, epoch, history)


def choose_threshold(
    model: Autoencoder, scaler: NormalScaler, val: Split, quantile: float
) -> tuple[float, np.ndarray]:
    """Threshold from the normal validation rows only. Returns (threshold, their errors (n,))."""
    errors = reconstruction_error(model, scaler.transform(normal_rows(val)))  # (n_val_normal,)
    return threshold_from_normal(errors, quantile), errors


def active_units(model: Autoencoder, x: np.ndarray) -> np.ndarray:
    """Share of rows where each bottleneck unit outputs > 0. x: (n, d) -> (bottleneck,).

    A unit with share 0 is dead: its ReLU input is negative for every row, so it passes no
    information and gets no gradient.
    """
    return (model.encode(x) > 0).mean(axis=0)  # (n, k) -> (k,)


# --- Saving and loading --------------------------------------------------------------------


def save_autoencoder(
    path: Path,
    model: Autoencoder,
    scaler: NormalScaler,
    threshold: float,
    quantile: float,
    schema_version: str,
    input_sha256: str,
) -> None:
    """Weights, scaler, threshold, and the metadata `load_autoencoder` checks. No pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"param_{i}": p for i, p in enumerate(model.parameters())}  # W1, b1, W2, ...
    np.savez(
        path,
        layer_sizes=np.array(model.layer_sizes, dtype=np.int64),
        input_mean=scaler.mean,
        input_scale=scaler.scale,
        threshold=np.array(threshold, dtype=np.float64),
        quantile=np.array(quantile, dtype=np.float64),
        schema_version=np.array(schema_version),
        input_sha256=np.array(input_sha256),
        **arrays,
    )


def load_autoencoder(
    path: Path, input_sha256: str | None = None
) -> tuple[Autoencoder, NormalScaler, float]:
    """Rebuild an autoencoder saved by `save_autoencoder`. Returns (model, scaler, threshold).

    Args:
        path: the `.npz` file.
        input_sha256: if given, it must equal the hash stored with the weights.

    Raises:
        SchemaError: other schema_version or input CSV, or sizes outside the feature contract.
        ValueError: missing or wrongly shaped arrays, a bad scaler or threshold, or Python
            objects.
    """
    with np.load(path, allow_pickle=False) as data:
        required = ("layer_sizes", "input_mean", "input_scale", "threshold", "schema_version",
                    "input_sha256")  # fmt: skip
        for key in required:
            if key not in data.files:
                raise ValueError(f"{path}: missing {key!r}")
        version = str(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"{path}: schema_version {version!r} != code SCHEMA_VERSION {SCHEMA_VERSION!r}"
            )
        if input_sha256 is not None and str(data["input_sha256"]) != input_sha256:
            raise SchemaError(f"{path}: weights were trained on another input CSV")
        sizes = data["layer_sizes"]
        if not np.issubdtype(sizes.dtype, np.integer):
            raise ValueError(f"{path}: layer_sizes must be integers, got {sizes.dtype}")
        if sizes.ndim != 1 or int(sizes[0]) != len(FEATURE_NAMES):
            raise SchemaError(
                f"{path}: layer_sizes {sizes.tolist()} must be 1-D and start with "
                f"{len(FEATURE_NAMES)} inputs (the feature contract)"
            )
        model = Autoencoder([int(n) for n in sizes], np.random.default_rng(0))  # checks symmetry
        threshold = data["threshold"]
        if threshold.shape != () or not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"{path}: threshold must be one finite number >= 0")
        mean, scale = data["input_mean"], data["input_scale"]
        d = len(FEATURE_NAMES)
        if mean.shape != (d,) or scale.shape != (d,):
            raise ValueError(f"{path}: input_mean and input_scale must have shape ({d},)")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale)) and np.all(scale > 0)):
            raise ValueError(f"{path}: input_mean must be finite and input_scale finite and > 0")
        params = model.parameters()
        n_saved = sum(key.startswith("param_") for key in data.files)
        if n_saved != len(params):
            raise ValueError(f"{path}: {n_saved} parameter arrays, model needs {len(params)}")
        for i, p in enumerate(params):
            saved = data[f"param_{i}"]
            if saved.shape != p.shape:
                raise ValueError(f"{path}: param_{i} has shape {saved.shape}, expected {p.shape}")
            p[...] = saved  # in place: the random init is overwritten
        return model, NormalScaler(mean=mean, scale=scale), float(threshold)


# --- Figures -------------------------------------------------------------------------------


def save_curves(result: AEFitResult, source: str, path: Path) -> None:
    """Train and validation reconstruction loss per epoch (normal rows), best epoch marked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = result.history
    epochs = range(len(h["train_loss"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, h["train_loss"], marker=".", label="train loss (normal flows)")
    ax.plot(epochs, h["val_loss"], marker=".", label="validation loss (normal flows)")
    best = result.best_epoch
    ax.axvline(best, color="gray", linestyle="--", linewidth=1)
    ax.plot([best], [h["val_loss"][best]], marker="*", markersize=14, color="red",
            linestyle="none", label=f"best epoch {best} (weights kept)")  # fmt: skip
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared reconstruction error")
    ax.set_yscale("log")
    ax.set_title(f"NumPy autoencoder loss ({source})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_error_histogram(
    errors: np.ndarray, split: Split, threshold: float, title: str, path: Path
) -> None:
    """Histogram of reconstruction errors per labeled class (log x axis), threshold marked.

    Each bar is the share of that class's flows in the bin. `density=True` would divide by the
    bin's linear width, which grows along a log axis, and would flatten the high-error bins.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    floor = max(float(errors.min()), 1e-6)
    bins = np.logspace(np.log10(floor), np.log10(float(errors.max())), 60)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(split.class_names):
        rows = split.y_class == i
        n = int(rows.sum())
        ax.hist(np.maximum(errors[rows], floor), bins=bins, histtype="step", linewidth=1.5,
                weights=np.full(n, 1.0 / max(n, 1)), label=f"{name} (n={n})")  # fmt: skip
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("reconstruction error (log scale)")
    ax.set_ylabel("share of the class's flows per bin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Command -------------------------------------------------------------------------------


def _flipped(split: Split, meta_path: Path) -> tuple[np.ndarray | None, str | None]:
    """(n,) bool label-noise mask for SYNTHETIC data from the matching run; evaluation only."""
    if split.source != "SYNTHETIC" or not meta_path.exists():
        return None, "data is not SYNTHETIC or synthetic.json is missing"
    ids = read_flipped_ids(meta_path, split.input_sha256)
    if ids is None:
        return None, f"{meta_path} and the arrays come from different runs"
    return np.isin(split.flow_id, list(ids)), None


def _print_results(t: dict, val_fpr: float, quantile: float, tag: str, note: str | None) -> None:
    """The test results table, with the noise ceiling next to each rate when available."""
    ceiling = t.get("ceiling")
    print(f"\nTest results, threshold from normal validation flows{tag}")
    print(f"{'rate':<36}{'autoencoder':>12}{'ceiling*':>10}{'flows':>7}")

    def fmt(value: float | None, width: int) -> str:
        if value is None:
            return f"{'-':>{width}}"
        return f"{'n/a':>{width}}" if np.isnan(value) else f"{value:>{width}.3f}"

    def row(label: str, value: float, ceil: float | None, n: int) -> None:
        print(f"{label:<36}{fmt(value, 12)}{fmt(ceil, 10)}{n:>7}")

    row("false-positive rate (normal)", t["fpr"], ceiling and ceiling["fpr"], t["n_normal"])
    for name, rate in t["detection"].items():
        row(f"detection rate: {name}", rate, ceiling and ceiling["detection"][name],
            t["n_class"][name])  # fmt: skip
    row("detection rate: all attacks", t["detection_all"], ceiling and ceiling["detection_all"],
        sum(t["n_class"].values()))  # fmt: skip
    print(f"(expected false-positive rate about {1 - quantile:.0%}; on normal validation flows: "
          f"{val_fpr:.3f})")  # fmt: skip
    if ceiling is None:
        print(f"(no label-noise ceiling: {note})")
        return
    print("* a perfect detector scored against the noisy test labels (flipped_flow_ids; "
          "evaluation only)")  # fmt: skip
    print(f"Flagged normal test flows: {t['n_flagged_normal']}, of which "
          f"{t['n_flagged_normal_flipped']} are attacks that label noise marked "
          "as normal.")  # fmt: skip
    print(f"False-positive rate on truly normal test flows: {fmt(t['fpr_true_normal'], 5).strip()} "
          f"({t['n_flagged_normal'] - t['n_flagged_normal_flipped']} of "
          f"{t['n_true_normal']})")  # fmt: skip


def main(argv: list[str] | None = None) -> dict:
    """Run the command; returns the results (used by the tests)."""
    d = AEConfig()
    parser = argparse.ArgumentParser(description="Train the NumPy autoencoder on normal flows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--synthetic-meta", type=Path,
                        default=Path("data/processed/synthetic.json"))  # fmt: skip
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--model-out", type=Path, default=Path("models/ae_numpy.npz"))
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--momentum", type=float, default=d.momentum)
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--encoder", type=int, nargs="+", default=list(d.encoder))
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--patience", type=int, default=d.patience)
    parser.add_argument("--quantile", type=float, default=d.quantile)
    parser.add_argument("--val-only", action="store_true",
                        help="stop after validation (tuning); never opens test.npz")  # fmt: skip
    args = parser.parse_args(argv)
    config = AEConfig(lr=args.lr, momentum=args.momentum, batch_size=args.batch_size,
                      epochs=args.epochs, encoder=tuple(args.encoder), seed=args.seed,
                      patience=args.patience, quantile=args.quantile)  # fmt: skip

    train = load_split(args.data_dir / "train.npz")
    val = load_split(args.data_dir / "val.npz")
    _check_same_source(train, val)
    tag = " (SYNTHETIC: tests code, not detection)" if val.source == "SYNTHETIC" else ""
    scaler = NormalScaler.fit(normal_rows(train))  # normal training rows only
    x_train = scaler.transform(normal_rows(train))  # (n_train_normal, d)
    x_val = scaler.transform(normal_rows(val))  # (n_val_normal, d)
    print(f"data source: {val.source}")
    print(f"normal flows (y_binary == 0): train {x_train.shape[0]} of {train.x.shape[0]}, "
          f"val {x_val.shape[0]} of {val.x.shape[0]}; "
          "attack rows are not used for training")  # fmt: skip
    print(f"config: {asdict(config)} · layers {layer_sizes(x_train.shape[1], config.encoder)}")

    # --- Training: normal training rows only; normal validation rows pick the stopping epoch.
    result = fit(config, x_train, x_val)
    h = result.history
    print(f"\n{'epoch':>5}{'train_mse':>11}{'val_mse':>10}   (normal flows)")
    for k in range(len(h["train_loss"])):
        if k % 10 == 0 or k == result.best_epoch or k == len(h["train_loss"]) - 1:
            mark = "  <- best" if k == result.best_epoch else ""
            print(f"{k:>5}{h['train_loss'][k]:>11.4f}{h['val_loss'][k]:>10.4f}{mark}")
    stop = "early stop" if result.stopped_epoch < config.epochs else "epoch limit"
    print(f"stopped after epoch {result.stopped_epoch} ({stop}); kept epoch {result.best_epoch}")

    model = result.model
    share = active_units(model, x_val)  # (k,)
    print(f"\nbottleneck units active (> 0 for some normal val flows): "
          f"{int((share > 0).sum())} of {share.size}; share of flows per unit: "
          f"{np.round(share, 3).tolist()}")  # fmt: skip

    # --- Threshold: normal validation rows only.
    threshold, val_errors = choose_threshold(model, scaler, val, config.quantile)
    val_fpr = float(flag(val_errors, threshold).mean())
    print(f"threshold ({config.quantile:.0%} quantile of {val_errors.size} normal val errors): "
          f"{threshold:.4f}; normal val flows flagged: {val_fpr:.3f}")  # fmt: skip

    save_autoencoder(args.model_out, model, scaler, threshold, config.quantile, SCHEMA_VERSION,
                     train.input_sha256)  # fmt: skip
    settings = {**asdict(config), "best_epoch": result.best_epoch, "threshold": threshold,
                "stopped_epoch": result.stopped_epoch, "source": train.source}  # fmt: skip
    args.model_out.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n")
    loss_fig = args.figures_dir / "ae_numpy_loss.png"
    save_curves(result, val.source, loss_fig)
    outcome = {"history": h, "best_epoch": result.best_epoch,
               "stopped_epoch": result.stopped_epoch, "threshold": threshold,
               "val_fpr": val_fpr, "active_share": share.tolist()}  # fmt: skip
    if args.val_only:
        print(f"\n--val-only: test split not opened · model -> {args.model_out} · "
              f"figure -> {loss_fig}")  # fmt: skip
        return outcome

    # --- Final scoring of the test split (once, at the end of the experiment).
    test = load_split(args.data_dir / "test.npz")
    _check_same_source(train, test)
    if test.class_names[0] != NORMAL_CLASS:
        raise ValueError(f"expected class 0 to be {NORMAL_CLASS!r}, got {test.class_names}")
    test_errors = reconstruction_error(model, scaler.transform(test.x))  # (n_test,)
    flipped, note = _flipped(test, args.synthetic_meta)
    table = detection_table(flag(test_errors, threshold), test.y_binary, test.y_class,
                            test.class_names, flipped)  # fmt: skip
    _print_results(table, val_fpr, config.quantile, tag, note)
    hist_fig = args.figures_dir / "ae_numpy_errors.png"
    save_error_histogram(test_errors, test, threshold,
                         f"Autoencoder reconstruction error on test flows ({test.source})",
                         hist_fig)  # fmt: skip
    print(f"\nmodel -> {args.model_out} · figures -> {loss_fig}, {hist_fig}")
    return {**outcome, "table": table}


if __name__ == "__main__":
    main()
