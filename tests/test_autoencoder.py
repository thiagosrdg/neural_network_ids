"""Tests for T07: MSELoss, the autoencoder, anomaly scores, and the training command."""

from pathlib import Path

import numpy as np
import pytest

from neural_ids import anomaly, preprocess, synthetic, train_autoencoder_numpy
from neural_ids.anomaly import (
    NormalScaler,
    detection_table,
    reconstruction_error,
    threshold_from_normal,
)
from neural_ids.gradcheck import assert_gradients_close, numerical_gradient
from neural_ids.nn.autoencoder import Autoencoder
from neural_ids.nn.layers import Dense, ReLU
from neural_ids.nn.losses import MSELoss
from neural_ids.nn.optim import SGD
from neural_ids.schema import SCHEMA_VERSION, SchemaError
from neural_ids.splits import Split, load_split
from neural_ids.train_autoencoder_numpy import AEConfig

SIZES = [29, 16, 4, 16, 29]
KINK_MARGIN = 1e-3  # same rule as tests/test_nn_gradients.py: ReLU inputs stay away from 0


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """synthetic.csv/json -> preprocess -> {train,val,test}.npz (small, for speed)."""
    root = tmp_path_factory.mktemp("t07")
    csv = root / "synthetic.csv"
    synthetic.main(["--n", "1500", "--seed", "3", "--out", str(csv)])
    preprocess.main(
        ["--input", str(csv), "--out-dir", str(root), "--preprocessor", str(root / "pre.json")]
    )
    return root


@pytest.fixture(scope="module")
def val(data_dir: Path) -> Split:
    return load_split(data_dir / "val.npz")


# --- MSELoss -------------------------------------------------------------------------------


def test_mse_value_and_gradient_formula() -> None:
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    loss = MSELoss()
    assert loss.forward(pred, target) == pytest.approx(np.mean((pred - target) ** 2))
    np.testing.assert_allclose(loss.backward(), 2 * (pred - target) / 15)


def test_mse_rejects_shapes_that_would_broadcast() -> None:
    with pytest.raises(ValueError, match="same shape"):
        MSELoss().forward(np.zeros((4, 3)), np.zeros(3))  # would broadcast to (4, 3)
    with pytest.raises(RuntimeError):
        MSELoss().backward()


def test_mse_gradcheck_float64() -> None:
    rng = np.random.default_rng(1)
    pred, target = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))  # float64
    loss = MSELoss()
    loss.forward(pred, target)
    analytic = loss.backward()
    numeric = numerical_gradient(lambda p: MSELoss().forward(p, target), pred)
    assert_gradients_close(analytic, numeric)


def test_autoencoder_gradcheck_float64() -> None:
    """Backprop through every layer with the MSE loss matches finite differences."""
    rng = np.random.default_rng(2)
    model = Autoencoder([6, 5, 3, 5, 6], rng)
    # Zero biases put a row whose ReLU outputs are all 0 exactly on the next layer's kink
    # (0 · W + 0 = 0), so the check uses random biases.
    for b in model.parameters()[1::2]:
        b[...] = rng.normal(0.0, 0.5, size=b.shape)
    x = rng.normal(size=(8, 6))  # float64
    pre_relu = []
    z = x
    for layer in model.layers:  # collect every Dense output that feeds a ReLU
        z_next = layer.forward(z)
        if isinstance(layer, ReLU):
            pre_relu.append(z)
        z = z_next
    assert min(np.abs(p).min() for p in pre_relu) > KINK_MARGIN, "change the seed"

    loss = MSELoss()
    loss.forward(model.forward(x), x)
    model.backward(loss.backward())
    for p, g in zip(model.parameters(), model.gradients(), strict=True):

        def loss_of(w: np.ndarray, p: np.ndarray = p) -> float:
            saved = p.copy()
            p[...] = w
            value = MSELoss().forward(model.forward(x), x)
            p[...] = saved
            return value

        assert_gradients_close(g.copy(), numerical_gradient(loss_of, p))


# --- Autoencoder ---------------------------------------------------------------------------


def test_autoencoder_structure_and_shapes() -> None:
    model = Autoencoder(SIZES, np.random.default_rng(0))
    kinds = [type(layer) for layer in model.layers]
    assert kinds == [Dense, ReLU, Dense, ReLU, Dense, ReLU, Dense]  # linear output
    assert model.n_parameters == 1121  # docs/ARCHITECTURE.md
    x = np.random.default_rng(1).normal(size=(10, 29))
    assert model.forward(x).shape == (10, 29)
    assert model.bottleneck_size == 4
    code = model.encode(x)
    assert code.shape == (10, 4) and np.all(code >= 0)


@pytest.mark.parametrize("sizes", [[29, 16, 29, 5], [29, 16, 4, 16, 30], [29, 29]])
def test_autoencoder_rejects_non_symmetric_sizes(sizes: list[int]) -> None:
    with pytest.raises(ValueError):
        Autoencoder(sizes, np.random.default_rng(0))


@pytest.mark.parametrize("seed", range(3))
def test_overfits_a_small_batch(val: Split, seed: int) -> None:
    """16 normal rows, 3000 full-batch steps: the loss ends below 5% of its start.

    Measured on the default data: the worst of seeds 0-2 ends at 2.5% of its start. A 4-unit
    bottleneck cannot copy much larger batches exactly, so the batch is small on purpose.
    """
    rng = np.random.default_rng(seed)
    x = train_autoencoder_numpy.normal_rows(val).astype(np.float64)
    xb = x[rng.choice(x.shape[0], size=16, replace=False)]  # (16, 29)
    model = Autoencoder(SIZES, rng)
    loss = MSELoss()
    opt = SGD(model.parameters(), lr=0.05, momentum=0.9)
    first = loss.forward(model.forward(xb), xb)
    for _ in range(3000):
        loss.forward(model.forward(xb), xb)
        model.backward(loss.backward())
        opt.step(model.gradients())
    assert loss.forward(model.forward(xb), xb) < 0.05 * first


# --- Anomaly scores ------------------------------------------------------------------------


def test_normal_scaler_standardizes_and_guards_constant_columns() -> None:
    rng = np.random.default_rng(8)
    x = np.column_stack([rng.normal(3.0, 2.0, size=500), np.full(500, 0.7)])  # col 1 constant
    scaler = NormalScaler.fit(x)
    z = scaler.transform(x)
    np.testing.assert_allclose(z[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(z[:, 0].std(), 1.0)
    assert scaler.scale[1] == 1.0  # no division by about 0
    np.testing.assert_allclose(scaler.transform(np.array([[3.0, 5.7]]))[0, 1], 5.0)


def test_command_scaler_ignores_attack_rows(data_dir: Path, tmp_path: Path) -> None:
    """The saved scaler equals one fitted on the normal training rows alone."""
    _run(data_dir, tmp_path, "--val-only")
    _, saved, _ = train_autoencoder_numpy.load_autoencoder(tmp_path / "ae.npz")
    train = load_split(data_dir / "train.npz")
    expected = NormalScaler.fit(train.x[train.y_binary == 0])
    np.testing.assert_array_equal(saved.mean, expected.mean)
    np.testing.assert_array_equal(saved.scale, expected.scale)


def test_reconstruction_error_is_one_mean_per_flow() -> None:
    rng = np.random.default_rng(3)
    model = Autoencoder(SIZES, rng)
    x = rng.normal(size=(7, 29))
    errors = reconstruction_error(model, x)
    assert errors.shape == (7,)
    np.testing.assert_allclose(errors, ((model.forward(x) - x) ** 2).mean(axis=1))
    with pytest.raises(ValueError):
        reconstruction_error(model, x[0])


def test_threshold_is_the_quantile() -> None:
    errors = np.random.default_rng(4).exponential(size=10_000)
    t = threshold_from_normal(errors)
    assert t == pytest.approx(np.quantile(errors, 0.99))
    assert anomaly.flag(errors, t).mean() == pytest.approx(0.01, abs=1e-3)
    assert threshold_from_normal(errors, 0.5) == pytest.approx(np.median(errors))


@pytest.mark.parametrize(
    ("errors", "q"),
    [
        (np.array([]), 0.99),
        (np.ones((3, 2)), 0.99),
        (np.array([1.0, np.nan]), 0.99),
        (np.ones(5), 1.0),
        (np.ones(5), 0.0),
    ],  # fmt: skip
)
def test_threshold_rejects_bad_input(errors: np.ndarray, q: float) -> None:
    with pytest.raises(ValueError):
        threshold_from_normal(errors, q)


def test_threshold_uses_normal_validation_errors_only(val: Split) -> None:
    """Changing every attack row of val must not move the threshold."""
    model = Autoencoder(SIZES, np.random.default_rng(5))
    scaler = NormalScaler.fit(train_autoencoder_numpy.normal_rows(val))
    t, errors = train_autoencoder_numpy.choose_threshold(model, scaler, val, 0.99)
    assert errors.size == int((val.y_binary == 0).sum())
    x_poisoned = val.x.copy()
    x_poisoned[val.y_binary == 1] = 1e6  # attack rows become extreme
    poisoned = Split(x_poisoned, val.y_binary, val.y_class, val.class_names, val.flow_id,
                     val.source, val.input_sha256)  # fmt: skip
    assert train_autoencoder_numpy.choose_threshold(model, scaler, poisoned, 0.99)[0] == t


def test_command_trains_and_thresholds_on_normal_rows_only(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy on `fit` and `threshold_from_normal`: they receive only y_binary == 0 rows."""
    seen: dict = {}
    real_fit, real_threshold = (
        train_autoencoder_numpy.fit,
        train_autoencoder_numpy.threshold_from_normal,
    )

    def spy_fit(config: AEConfig, x: np.ndarray, x_val: np.ndarray):
        seen["x"], seen["x_val"] = x, x_val
        return real_fit(config, x, x_val)

    def spy_threshold(errors: np.ndarray, quantile: float = 0.99) -> float:
        seen["errors"] = errors
        return real_threshold(errors, quantile)

    monkeypatch.setattr(train_autoencoder_numpy, "fit", spy_fit)
    monkeypatch.setattr(train_autoencoder_numpy, "threshold_from_normal", spy_threshold)
    _run(data_dir, tmp_path, "--val-only")
    train, val = load_split(data_dir / "train.npz"), load_split(data_dir / "val.npz")
    scaler = NormalScaler.fit(train.x[train.y_binary == 0])  # what the command must use
    np.testing.assert_array_equal(seen["x"], scaler.transform(train.x[train.y_binary == 0]))
    np.testing.assert_array_equal(seen["x_val"], scaler.transform(val.x[val.y_binary == 0]))
    assert seen["errors"].size == int((val.y_binary == 0).sum())


def test_detection_table_by_hand() -> None:
    names = ("normal", "scan", "flood")
    y_class = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_binary = (y_class > 0).astype(np.int64)
    flagged = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    flipped = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)  # row 0: attack labeled normal
    t = detection_table(flagged, y_binary, y_class, names, flipped)
    assert t["fpr"] == 0.5 and t["n_flagged_normal"] == 2
    assert t["detection"] == {"scan": 0.5, "flood": 1.0} and t["detection_all"] == 0.75
    assert t["ceiling"] == {"fpr": 0.25, "detection": {"scan": 0.5, "flood": 1.0},
                            "detection_all": 0.75}  # fmt: skip
    assert t["n_flagged_normal_flipped"] == 1
    assert t["n_true_normal"] == 3 and t["fpr_true_normal"] == pytest.approx(1 / 3)
    assert "ceiling" not in detection_table(flagged, y_binary, y_class, names)
    no_flood = detection_table(flagged[:6], y_binary[:6], y_class[:6], names)
    assert np.isnan(no_flood["detection"]["flood"])  # no rows: NaN (printed n/a), no warning


# --- Saving and loading --------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    model = Autoencoder(SIZES, np.random.default_rng(6))
    path = tmp_path / "ae.npz"
    scaler = NormalScaler(mean=np.full(29, 0.5), scale=np.full(29, 2.0))
    train_autoencoder_numpy.save_autoencoder(path, model, scaler, 0.25, 0.99, SCHEMA_VERSION, "abc")
    loaded, loaded_scaler, threshold = train_autoencoder_numpy.load_autoencoder(path, "abc")
    assert threshold == 0.25 and loaded.layer_sizes == SIZES
    np.testing.assert_array_equal(loaded_scaler.mean, scaler.mean)
    np.testing.assert_array_equal(loaded_scaler.scale, scaler.scale)
    x = np.random.default_rng(7).normal(size=(4, 29))
    np.testing.assert_array_equal(loaded.forward(x), model.forward(x))


def test_load_refuses_mismatches(tmp_path: Path) -> None:
    model = Autoencoder(SIZES, np.random.default_rng(6))
    scaler = NormalScaler(mean=np.zeros(29), scale=np.ones(29))

    def save(path: Path, model: Autoencoder, threshold: float, q: float, v: str, h: str) -> None:
        train_autoencoder_numpy.save_autoencoder(path, model, scaler, threshold, q, v, h)

    load = train_autoencoder_numpy.load_autoencoder
    save(tmp_path / "old.npz", model, 0.25, 0.99, "0.9", "abc")
    with pytest.raises(SchemaError):
        load(tmp_path / "old.npz")
    save(tmp_path / "ok.npz", model, 0.25, 0.99, SCHEMA_VERSION, "abc")
    with pytest.raises(SchemaError):
        load(tmp_path / "ok.npz", input_sha256="def")
    save(tmp_path / "nan.npz", model, float("nan"), 0.99, SCHEMA_VERSION, "abc")
    with pytest.raises(ValueError, match="threshold"):
        load(tmp_path / "nan.npz")
    small = Autoencoder([10, 4, 10], np.random.default_rng(0))
    save(tmp_path / "small.npz", small, 0.25, 0.99, SCHEMA_VERSION, "abc")
    with pytest.raises(SchemaError):
        load(tmp_path / "small.npz")
    train_autoencoder_numpy.save_autoencoder(
        tmp_path / "zero_scale.npz", model, NormalScaler(np.zeros(29), np.zeros(29)), 0.25, 0.99,
        SCHEMA_VERSION, "abc",
    )  # fmt: skip
    with pytest.raises(ValueError, match="input_scale"):
        load(tmp_path / "zero_scale.npz")
    np.savez(tmp_path / "pickled.npz", layer_sizes=np.array([object()], dtype=object),
             threshold=0.1, schema_version=SCHEMA_VERSION, input_sha256="abc")  # fmt: skip
    with pytest.raises(ValueError):
        load(tmp_path / "pickled.npz")


# --- Command -------------------------------------------------------------------------------


def _run(data_dir: Path, out: Path, *extra: str) -> dict:
    return train_autoencoder_numpy.main([
        "--data-dir", str(data_dir), "--synthetic-meta", str(data_dir / "synthetic.json"),
        "--figures-dir", str(out / "fig"), "--model-out", str(out / "ae.npz"),
        "--epochs", "5", *extra,
    ])  # fmt: skip


def test_val_only_never_opens_test(data_dir: Path, tmp_path: Path) -> None:
    no_test = tmp_path / "data"
    no_test.mkdir()
    for name in ("train.npz", "val.npz", "synthetic.json"):
        (no_test / name).write_bytes((data_dir / name).read_bytes())
    result = _run(no_test, tmp_path, "--val-only")
    assert "table" not in result
    assert (tmp_path / "fig" / "ae_numpy_loss.png").exists()
    _, _, threshold = train_autoencoder_numpy.load_autoencoder(tmp_path / "ae.npz")
    assert threshold == result["threshold"]


def test_command_scores_test_with_ceiling(data_dir: Path, tmp_path: Path, capsys) -> None:
    result = _run(data_dir, tmp_path)
    out = capsys.readouterr().out
    assert "SYNTHETIC: tests code, not detection" in out
    assert "bottleneck units active" in out and "ceiling*" in out
    assert "attacks that label noise marked as normal" in out
    table = result["table"]
    assert set(table["detection"]) == {"ssh_bruteforce", "syn_scan", "udp_flood"}
    assert "ceiling" in table
    assert (tmp_path / "fig" / "ae_numpy_errors.png").exists()
    assert (tmp_path / "ae.json").exists()


def test_command_is_deterministic(data_dir: Path, tmp_path: Path) -> None:
    a = _run(data_dir, tmp_path / "a", "--val-only")
    b = _run(data_dir, tmp_path / "b", "--val-only")
    assert a["history"] == b["history"] and a["threshold"] == b["threshold"]
