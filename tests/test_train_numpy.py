"""Tests for `neural_ids.train_numpy`: sanity checks, early stopping, saving, and the command."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neural_ids import preprocess, synthetic, train_numpy
from neural_ids.nn.layers import BCEWithLogits
from neural_ids.nn.model import MLP
from neural_ids.nn.optim import SGD
from neural_ids.schema import SCHEMA_VERSION, SchemaError
from neural_ids.splits import Split, load_split
from neural_ids.train_numpy import TrainConfig


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """synthetic.csv/json -> preprocess -> {train,val}.npz. test.npz is deleted on purpose."""
    root = tmp_path_factory.mktemp("t06")
    csv = root / "synthetic.csv"
    synthetic.main(["--n", "1500", "--seed", "3", "--out", str(csv)])
    preprocess.main(
        ["--input", str(csv), "--out-dir", str(root), "--preprocessor", str(root / "pre.json")]
    )
    (root / "test.npz").unlink()  # the command must not need (or open) the test split
    return root


@pytest.fixture(scope="module")
def train(data_dir: Path) -> Split:
    return load_split(data_dir / "train.npz")


@pytest.fixture(scope="module")
def val(data_dir: Path) -> Split:
    return load_split(data_dir / "val.npz")


# --- Sanity checks -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_first_loss_is_within_001_of_ln2(train: Split, seed: int) -> None:
    loss = train_numpy.first_loss(train.x, train.y_binary, seed, (64, 32))
    assert abs(loss - math.log(2)) < 0.01


def test_output_layer_starts_small() -> None:
    model = MLP([29, 64, 32, 1], np.random.default_rng(0))
    w_out, b_out = model.parameters()[-2:]
    assert np.all(b_out == 0.0)
    he_std = math.sqrt(2.0 / 32)
    assert w_out.std() < 0.05 * he_std  # about 0.01 × He; plain He would be about 1 × He


def test_overfits_one_batch_of_64(train: Split) -> None:
    before, after = train_numpy.overfit_one_batch(train.x, train.y_binary, 0, (64, 32))
    assert abs(before - math.log(2)) < 0.01
    assert after < train_numpy.SANITY_OVERFIT_MAX


# --- One epoch and fit ---------------------------------------------------------------------


def test_train_epoch_lowers_the_loss_and_returns_weighted_mean(train: Split) -> None:
    model = MLP([29, 16, 1], np.random.default_rng(0))
    loss_fn = BCEWithLogits()
    opt = SGD(model.parameters(), lr=0.05, momentum=0.9)
    before = train_numpy.evaluate(model, train.x, train.y_binary)[0]
    rng = np.random.default_rng(1)
    running = train_numpy.train_epoch(model, loss_fn, opt, train.x, train.y_binary, 100, rng)
    after = train_numpy.evaluate(model, train.x, train.y_binary)[0]
    assert after < before
    assert 0.0 < running < before  # mean over batches measured while the weights improved


def test_train_epoch_weights_the_last_smaller_batch() -> None:
    # lr tiny: the weights barely move, so the epoch's batch losses are the starting weights'
    # losses. Weighted by batch size, their mean equals the full-data loss L0. A plain mean of
    # the three batch losses gives too much weight to the 2-row batch (off by about 9e-4 here).
    rng = np.random.default_rng(0)
    x = rng.normal(size=(130, 4))  # batches of 64, 64, 2
    y = (x[:, 0] > 0).astype(np.int64)
    model = MLP([4, 8, 1], np.random.default_rng(1))
    l0 = train_numpy.evaluate(model, x, y)[0]
    opt = SGD(model.parameters(), lr=1e-9, momentum=0.0)
    running = train_numpy.train_epoch(model, BCEWithLogits(), opt, x, y, 64, rng)
    assert running == pytest.approx(l0, abs=1e-6)


def test_fit_keeps_best_weights_and_stops_early(train: Split, val: Split) -> None:
    config = TrainConfig(epochs=40, patience=3, hidden=(16,), seed=0)
    result = train_numpy.fit(config, train.x, train.y_binary, val.x, val.y_binary)
    assert result.stopped_epoch < config.epochs  # this setup does stop early
    h = result.history
    assert len(h["train_loss"]) == len(h["val_loss"]) == len(h["val_f1"])
    assert len(h["val_loss"]) == result.stopped_epoch + 1  # index 0 = before training
    assert result.best_epoch == int(np.argmin(h["val_loss"]))
    assert result.stopped_epoch - result.best_epoch == config.patience  # exactly `patience` waited
    # The returned model holds the best epoch's weights, not the last epoch's.
    val_loss, val_f1 = train_numpy.evaluate(result.model, val.x, val.y_binary)
    assert val_loss == pytest.approx(h["val_loss"][result.best_epoch], abs=1e-12)
    assert val_f1 == pytest.approx(h["val_f1"][result.best_epoch], abs=1e-12)


def test_each_epoch_reshuffles_with_the_seeded_generator(
    train: Split, val: Split, monkeypatch: pytest.MonkeyPatch
) -> None:
    orders: list[np.ndarray] = []
    real = train_numpy.minibatch_indices

    def spy(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
        batches = real(n, batch_size, rng)
        orders.append(np.concatenate(batches))  # (n,) row order of this epoch
        return batches

    monkeypatch.setattr(train_numpy, "minibatch_indices", spy)
    config = TrainConfig(epochs=3, patience=10, hidden=(4,), seed=0)
    train_numpy.fit(config, train.x, train.y_binary, val.x, val.y_binary)
    n = train.x.shape[0]
    assert len(orders) == 3
    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(n))  # every row once
        assert not np.array_equal(order, np.arange(n))  # shuffled
    assert not np.array_equal(orders[0], orders[1])  # a new order each epoch,
    assert not np.array_equal(orders[1], orders[2])  # from one generator that moves on


def test_fit_is_deterministic_and_seed_dependent(train: Split, val: Split) -> None:
    config = TrainConfig(epochs=3, patience=5, hidden=(8,), seed=7)
    a = train_numpy.fit(config, train.x, train.y_binary, val.x, val.y_binary)
    b = train_numpy.fit(config, train.x, train.y_binary, val.x, val.y_binary)
    c = train_numpy.fit(replace(config, seed=8), train.x, train.y_binary, val.x, val.y_binary)
    assert a.history == b.history
    assert a.history != c.history


# --- Saving and loading --------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, val: Split) -> None:
    model = MLP([29, 8, 4, 1], np.random.default_rng(0))
    path = tmp_path / "m.npz"
    train_numpy.save_mlp(path, model, SCHEMA_VERSION, "abc")
    loaded = train_numpy.load_mlp(path, input_sha256="abc")
    assert loaded.layer_sizes == [29, 8, 4, 1]
    for p, q in zip(model.parameters(), loaded.parameters(), strict=True):
        np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(model.forward(val.x), loaded.forward(val.x))


def test_load_refuses_other_schema_or_input(tmp_path: Path) -> None:
    model = MLP([29, 4, 1], np.random.default_rng(0))
    old = tmp_path / "old.npz"
    train_numpy.save_mlp(old, model, "0.9", "abc")
    with pytest.raises(SchemaError, match="schema_version"):
        train_numpy.load_mlp(old)
    ok = tmp_path / "ok.npz"
    train_numpy.save_mlp(ok, model, SCHEMA_VERSION, "abc")
    with pytest.raises(SchemaError, match="another input CSV"):
        train_numpy.load_mlp(ok, input_sha256="def")


def test_load_refuses_pickled_objects_and_bad_shapes(tmp_path: Path) -> None:
    meta = {
        "layer_sizes": np.array([29, 1]),
        "schema_version": np.array(SCHEMA_VERSION),
        "input_sha256": np.array("abc"),
    }
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, **meta, param_0=np.array([{"x": 1}], dtype=object), param_1=np.zeros(1))
    with pytest.raises(ValueError):
        train_numpy.load_mlp(pickled)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **meta, param_0=np.zeros((3, 1)), param_1=np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        train_numpy.load_mlp(bad)


@pytest.mark.parametrize(
    ("sizes", "error"),
    [
        ([30, 4, 1], SchemaError),
        ([29, 4, 2], SchemaError),
        ([[29, 1]], ValueError),
        (29, ValueError),
    ],
)
def test_load_refuses_architectures_outside_the_contract(
    tmp_path: Path, sizes: object, error: type[Exception]
) -> None:
    path = tmp_path / "m.npz"
    np.savez(
        path,
        layer_sizes=np.array(sizes),
        schema_version=np.array(SCHEMA_VERSION),
        input_sha256=np.array("abc"),
    )
    with pytest.raises(error):
        train_numpy.load_mlp(path)


# --- Command -------------------------------------------------------------------------------


def _run(data_dir: Path, out: Path) -> dict:
    return train_numpy.main(
        [
            "--data-dir", str(data_dir),
            "--synthetic-meta", str(data_dir / "synthetic.json"),
            "--figures-dir", str(out / "fig"),
            "--model-out", str(out / "mlp.npz"),
            "--epochs", "8", "--patience", "3", "--hidden", "16", "8",
        ]
    )  # fmt: skip


def test_command_saves_files_and_prints_labeled_table(
    data_dir: Path, tmp_path: Path, capsys
) -> None:
    results = _run(data_dir, tmp_path)
    out = capsys.readouterr().out
    assert "SYNTHETIC: tests code, not detection" in out
    assert "noise ceiling" in out
    assert "slightly optimistic" in out and "test split stays unused" in out
    assert (tmp_path / "fig" / "mlp_numpy_loss.png").stat().st_size > 0
    assert (tmp_path / "mlp.json").exists()
    assert set(results["models"]) == {"logistic_regression", "mlp_numpy", "random_forest (ref)"}
    sizes = {int(m["confusion"].sum()) for m in results["models"].values()}
    assert len(sizes) == 1  # every model scored on the same val rows

    meta = np.load(tmp_path / "mlp.npz", allow_pickle=False)
    assert str(meta["schema_version"]) == SCHEMA_VERSION
    assert str(meta["input_sha256"]) == load_split(data_dir / "train.npz").input_sha256
    assert meta["layer_sizes"].tolist() == [29, 16, 8, 1]


def test_command_twice_gives_the_same_numbers(data_dir: Path, tmp_path: Path) -> None:
    a = _run(data_dir, tmp_path / "a")
    b = _run(data_dir, tmp_path / "b")
    assert a["history"] == b["history"]
    assert a["best_epoch"] == b["best_epoch"]
    for name in a["models"]:
        assert {k: v for k, v in a["models"][name].items() if k != "confusion"} == {
            k: v for k, v in b["models"][name].items() if k != "confusion"
        }
        np.testing.assert_array_equal(
            a["models"][name]["confusion"], b["models"][name]["confusion"]
        )
    wa, wb = (np.load(tmp_path / d / "mlp.npz", allow_pickle=False) for d in ("a", "b"))
    assert all(np.array_equal(wa[k], wb[k]) for k in wa.files)


def test_sanity_flag(data_dir: Path, capsys) -> None:
    results = train_numpy.main(["--data-dir", str(data_dir), "--sanity"])
    assert results["sanity"]["passed"]
    assert "PASS" in capsys.readouterr().out
