"""Train the three baselines on the training split and score them on the validation split.

Command:
    uv run python -m neural_ids.train_baselines
reads `data/processed/{train,val}.npz` (never test.npz: the test split is used once, at the
end of an experiment), prints a validation metrics table with confusion matrices, and saves
`reports/figures/logreg_loss.png` and `reports/figures/perceptron_errors.png`.

When the data is SYNTHETIC, the evaluation step also reads `flipped_flow_ids` from
`synthetic.json` (after training) to print the label-noise ceiling for the validation split.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neural_ids.metrics import binary_report, confusion_matrix, label_noise_ceiling  # noqa: E402
from neural_ids.models.baselines import MajorityClassifier  # noqa: E402
from neural_ids.models.logreg import LogisticRegression  # noqa: E402
from neural_ids.models.perceptron import Perceptron  # noqa: E402
from neural_ids.splits import Split, load_split  # noqa: E402
from neural_ids.utils import set_seed  # noqa: E402


def train_models(
    train: Split, val: Split, seed: int, epochs: int, lr: float, batch_size: int
) -> dict[str, MajorityClassifier | Perceptron | LogisticRegression]:
    """Fit the three models on `train`. `val` is only measured for the curves."""
    perceptron_rng, logreg_rng = set_seed(seed).spawn(2)  # independent streams per model
    d = train.x.shape[1]
    majority = MajorityClassifier().fit(train.x, train.y_binary)
    perceptron = Perceptron(d, lr=lr, epochs=epochs, batch_size=batch_size).fit(
        train.x, train.y_binary, val.x, val.y_binary, perceptron_rng
    )
    logreg = LogisticRegression(d, lr=lr, epochs=epochs, batch_size=batch_size).fit(
        train.x, train.y_binary, val.x, val.y_binary, logreg_rng
    )
    return {"majority": majority, "perceptron": perceptron, "logistic_regression": logreg}


def read_flipped_ids(meta_path: Path, input_sha256: str) -> set[str] | None:
    """`flipped_flow_ids` from synthetic.json: the label-noise answer key (evaluation only).

    Synthetic flow IDs are `synth-0000000`, `synth-0000001`, ... in every run, so IDs alone
    cannot tell whether this json describes the arrays. The CSV hash can: synthetic writes the
    SHA-256 of its CSV (`csv_sha256`), and preprocess stores the hash of its input CSV in each
    .npz (`input_sha256`).

    Args:
        meta_path: synthetic.json written by `neural_ids.synthetic`.
        input_sha256: hash stored in the arrays being evaluated.

    Returns:
        The flipped flow IDs, or None when the hashes differ (or the json has no hash).
    """
    meta = json.loads(meta_path.read_text())
    if meta.get("csv_sha256") != input_sha256:
        return None
    return {str(flow_id) for flow_id in meta["flipped_flow_ids"]}


def save_curves(
    history: dict[str, list[float]], keys: tuple[str, str], ylabel: str, title: str, path: Path
) -> None:
    """Plot the train and validation curves of one model (epoch 0 = before training)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in keys:
        ax.plot(range(len(history[key])), history[key], marker=".", label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    """Run the command; returns the validation results (used by the tests)."""
    parser = argparse.ArgumentParser(description="Train and compare the NumPy baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--synthetic-meta",
        type=Path,
        default=Path("data/processed/synthetic.json"),
        help="provenance file with flipped_flow_ids (used only for SYNTHETIC data)",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    train = load_split(args.data_dir / "train.npz")
    val = load_split(args.data_dir / "val.npz")
    if (train.class_names, train.source, train.input_sha256) != (
        val.class_names,
        val.source,
        val.input_sha256,
    ):
        raise ValueError(
            "train and val differ in class_names, source, or input CSV; rebuild the arrays"
        )
    tag = " (SYNTHETIC: tests code, not detection)" if val.source == "SYNTHETIC" else ""
    print(f"data source: {val.source}")
    print(
        f"train {train.x.shape}, attack {train.y_binary.mean():.1%} · "
        f"val {val.x.shape}, attack {val.y_binary.mean():.1%} · seed {args.seed}, "
        f"epochs {args.epochs}, lr {args.lr}, batch {args.batch_size}"
    )

    # --- Training: train split only (val is measured for the curves, never fitted on).
    models = train_models(train, val, args.seed, args.epochs, args.lr, args.batch_size)

    # --- Evaluation on the validation split.
    results: dict = {"models": {}, "ceiling": None, "ceiling_note": None}
    for name, model in models.items():
        pred = model.predict(val.x)  # (n_val,)
        results["models"][name] = {
            **binary_report(val.y_binary, pred),
            "confusion": confusion_matrix(val.y_binary, pred),
        }
    if val.source != "SYNTHETIC" or not args.synthetic_meta.exists():
        results["ceiling_note"] = "data is not SYNTHETIC or synthetic.json is missing"
    elif (flipped_ids := read_flipped_ids(args.synthetic_meta, val.input_sha256)) is None:
        results["ceiling_note"] = (
            f"{args.synthetic_meta} and the arrays come from different runs "
            "(csv_sha256 != input_sha256); rerun preprocess on the matching CSV"
        )
    else:
        flipped = np.isin(val.flow_id, list(flipped_ids))  # (n_val,) bool
        results["ceiling"] = {
            **label_noise_ceiling(val.y_binary, flipped),
            "n_flipped": int(flipped.sum()),
        }

    print(f"\nValidation metrics, attack = positive class{tag}")
    print(f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}")
    for name, m in results["models"].items():
        print(
            f"{name:<22}{m['accuracy']:>10.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}"
            f"{m['f1']:>8.3f}"
        )
    ceiling = results["ceiling"]
    if ceiling is not None:
        print(
            f"{'noise ceiling*':<22}{'-':>10}{ceiling['precision']:>11.3f}"
            f"{ceiling['recall']:>9.3f}{'-':>8}"
        )
        print(
            f"* a perfect model scored against the noisy val labels "
            f"({ceiling['n_flipped']} flipped labels in val; evaluation only)"
        )
    else:
        print(f"(no label-noise ceiling: {results['ceiling_note']})")

    print("\nConfusion matrices on validation (rows: true normal/attack; columns: predicted)")
    print(f"{'model':<22}{'TN':>7}{'FP':>7}{'FN':>7}{'TP':>7}")
    for name, m in results["models"].items():
        (tn, fp), (fn, tp) = m["confusion"]
        print(f"{name:<22}{tn:>7}{fp:>7}{fn:>7}{tp:>7}")

    logreg_fig = args.figures_dir / "logreg_loss.png"
    perceptron_fig = args.figures_dir / "perceptron_errors.png"
    save_curves(
        models["logistic_regression"].history,
        ("train_loss", "val_loss"),
        "mean binary cross-entropy",
        f"Logistic regression loss ({val.source})",
        logreg_fig,
    )
    save_curves(
        models["perceptron"].history,
        ("train_error", "val_error"),
        "error rate",
        f"Perceptron error rate ({val.source})",
        perceptron_fig,
    )
    history = models["logistic_regression"].history
    print(
        f"\nlogreg loss: train {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}, "
        f"val {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}"
    )
    print(f"figures -> {logreg_fig}, {perceptron_fig}")
    results["history"] = {"logreg": history, "perceptron": models["perceptron"].history}
    return results


if __name__ == "__main__":
    main()
