"""Anomaly scores from an autoencoder: reconstruction error, threshold, and the results table.

A flow's anomaly score is its reconstruction error: the mean over features of
(reconstruction − input)². The autoencoder saw only normal flows during training, so a flow
unlike them comes out badly and gets a high score. A flow is flagged when its score is above a
threshold chosen from normal validation flows only (by default their 99th percentile, so about
1% of normal flows are flagged: the false-positive budget).

The T03 preprocessor is fitted on the whole training split, attacks included. `NormalScaler`
standardizes again with statistics of normal training flows only, so no attack flow shapes
the autoencoder's inputs: the same result a person with only normal captures would get.
"""

from dataclasses import dataclass

import numpy as np

from neural_ids.nn.autoencoder import Autoencoder

NORMAL_CLASS = "normal"
MIN_STD = 1e-6  # below this a column counts as constant in normal traffic and keeps scale 1


@dataclass(frozen=True)
class NormalScaler:
    """Per-feature standardization fitted on normal training flows: (x − mean) / scale."""

    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,), all > 0

    @classmethod
    def fit(cls, x_normal: np.ndarray) -> "NormalScaler":
        """x_normal: (n, d) normal training flows only. Time O(n · d).

        A column that is constant in normal traffic (for example a protocol that never appears
        in it) keeps scale 1 instead of dividing by about 0. Any non-zero value there in a new
        flow still adds to its error, at the size the T03 scaling gave it.
        """
        std = x_normal.std(axis=0)  # (d,)
        return cls(
            mean=x_normal.mean(axis=0).astype(np.float64),
            scale=np.where(std > MIN_STD, std, 1.0).astype(np.float64),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """(n, d) -> (n, d) float64."""
        return (x - self.mean) / self.scale


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error of each flow.

    Args:
        model: a trained autoencoder with output size equal to input size.
        X: (n, d) standardized features.

    Returns:
        (n,) float64, one score per flow, each >= 0. Time O(n · P), memory O(n · d).
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n, d), got {X.shape}")
    recon = model.forward(X)  # (n, d)
    return np.mean((recon - X) ** 2, axis=1)  # (n, d) -> (n,): mean over features


def threshold_from_normal(errors: np.ndarray, quantile: float = 0.99) -> float:
    """Alert threshold: the `quantile` of reconstruction errors of normal flows.

    Pass the errors of normal validation flows only: never training flows (the model has fitted
    them, so their errors are too low) and never attack flows (the threshold must not depend on
    attack labels).

    Args:
        errors: (n,) reconstruction errors of normal flows, n >= 1, all finite.
        quantile: share of normal flows that should stay below the threshold, in (0, 1).

    Returns:
        The threshold as a float. Time O(n): np.quantile uses a partial sort (partition).
    """
    if errors.ndim != 1 or errors.size == 0:
        raise ValueError(f"errors must be a non-empty 1-D array, got shape {errors.shape}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if not np.all(np.isfinite(errors)):
        raise ValueError("errors contain NaN or infinity")
    return float(np.quantile(errors, quantile))


def flag(errors: np.ndarray, threshold: float) -> np.ndarray:
    """(n,) bool: True where the score is strictly above the threshold (an alert)."""
    return errors > threshold


def detection_table(
    flagged: np.ndarray,
    y_binary: np.ndarray,
    y_class: np.ndarray,
    class_names: tuple[str, ...],
    flipped: np.ndarray | None = None,
) -> dict:
    """False-positive rate on normal flows and detection rate per attack class.

    All rates are scored against the labels stored in the split (with label noise).

    Args:
        flagged: (n,) bool alerts.
        y_binary: (n,) 0/1 labels, 1 = attack.
        y_class: (n,) indices into class_names.
        class_names: class names; NORMAL_CLASS is the only normal one.
        flipped: (n,) bool, True where label noise changed the binary label; evaluation only.
            When given, the table also has the noise ceiling (the rates a perfect detector gets
            against the noisy labels) and splits the flagged normal flows into truly normal ones
            and attacks that noise labeled normal.

    Returns:
        {"fpr", "n_normal", "n_flagged_normal", "detection": {class: rate}, "n_class": {...},
         "detection_all"} plus, with `flipped`, {"ceiling": {...}, "n_flagged_normal_flipped",
         "n_true_normal", "fpr_true_normal"}. Time O(n · K) for K classes.
    """
    normal = y_binary == 0  # (n,) bool
    attack_ids = [i for i, name in enumerate(class_names) if name != NORMAL_CLASS]

    def share(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")  # NaN: no rows, printed n/a

    def rates(pred: np.ndarray) -> tuple[float, dict[str, float], float]:
        det = {class_names[i]: share(pred[y_class == i]) for i in attack_ids}
        return share(pred[normal]), det, share(pred[~normal])

    fpr, detection, detection_all = rates(flagged)
    table = {
        "fpr": fpr,
        "n_normal": int(normal.sum()),
        "n_flagged_normal": int((flagged & normal).sum()),
        "detection": detection,
        "n_class": {class_names[i]: int((y_class == i).sum()) for i in attack_ids},
        "detection_all": detection_all,
    }
    if flipped is not None:
        perfect = (
            y_binary.astype(bool) ^ flipped
        )  # (n,) the true class: what a perfect detector flags
        c_fpr, c_det, c_all = rates(perfect)
        table["ceiling"] = {"fpr": c_fpr, "detection": c_det, "detection_all": c_all}
        true_normal = normal & ~flipped  # labeled normal and really normal
        table["n_flagged_normal_flipped"] = int((flagged & normal & flipped).sum())
        table["n_true_normal"] = int(true_normal.sum())
        table["fpr_true_normal"] = share(flagged[true_normal])
    return table
