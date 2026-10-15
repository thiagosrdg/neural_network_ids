"""Tests for `neural_ids.train_torch`: data loading, the training step, early stopping,
determinism, saving, and the command. Synthetic data only: these tests check code, not detection.
"""

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_ids import preprocess, synthetic, train_numpy, train_torch
from neural_ids.schema import SchemaError
from neural_ids.splits import Split, load_split
from neural_ids.torch_models import MLPClassifier
from neural_ids.train_torch import TorchConfig
from neural_ids.utils import set_seed

SMALL = TorchConfig(hidden=(16,), epochs=6, patience=3, batch_size=64)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """synthetic.csv/json -> preprocess -> {train,val}.npz, plus a quick NumPy MLP (T06)."""
    root = tmp_path_factory.mktemp("t08")
    csv = root / "synthetic.csv"
    synthetic.main(["--n", "1500", "--seed", "3", "--out", str(csv)])
    preprocess.main(
        ["--input", str(csv), "--out-dir", str(root), "--preprocessor", str(root / "pre.json")]
    )
    (root / "test.npz").unlink()  # the command must not need (or open) the test split
    train_numpy.main(
        ["--data-dir", str(root), "--synthetic-meta", str(root / "synthetic.json"),
         "--figures-dir", str(root / "fig"), "--model-out", str(root / "mlp_numpy.npz"),
         "--epochs", "3", "--hidden", "8"]
    )  # fmt: skip
    return root


@pytest.fixture(scope="module")
def train(data_dir: Path) -> Split:
    return load_split(data_dir / "train.npz")


@pytest.fixture(scope="module")
def val(data_dir: Path) -> Split:
    return load_split(data_dir / "val.npz")


def _states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- Data ----------------------------------------------------------------------------------


def test_dataset_shapes_and_dtypes(train: Split) -> None:
    binary = train_torch.FlowDataset(train.x, train.y_binary, n_classes=2)
    multi = train_torch.FlowDataset(train.x, train.y_class, n_classes=4)
    assert len(binary) == train.x.shape[0]
    x0, y0 = binary[0]
    assert x0.shape == (29,) and x0.dtype == torch.float32
    assert y0.shape == (1,) and y0.dtype == torch.float32  # matches the (B, 1) binary logits
    _, c0 = multi[0]
    assert c0.shape == () and c0.dtype == torch.int64  # CrossEntropyLoss wants class indices


def test_loader_order_is_seeded(train: Split) -> None:
    ds = train_torch.FlowDataset(train.x, train.y_binary, n_classes=2)

    def first_batch(seed: int) -> torch.Tensor:
        return next(iter(train_torch.make_loader(ds, 32, seed)))[0]

    assert torch.equal(first_batch(1), first_batch(1))
    assert not torch.equal(first_batch(1), first_batch(2))
    assert train_torch.make_loader(ds, 32, 1).num_workers == 0


def test_class_weights_are_balanced_and_fitted_on_given_labels() -> None:
    w = train_torch.class_weights(np.array([0] * 6 + [1] * 2 + [2] * 2), 3)
    # n / (K · count): 10 / (3·6), 10 / (3·2), 10 / (3·2)
    np.testing.assert_allclose(w.numpy(), [10 / 18, 10 / 6, 10 / 6], rtol=1e-6)
    with pytest.raises(ValueError):
        train_torch.class_weights(np.array([0, 0, 1]), 3)  # class 2 has no training rows


def test_weighted_binary_loss_matches_hand_formula() -> None:
    """pos_weight = w1 / w0 = n0 / n1: attack rows' loss counts that many times, mean over B."""
    y = np.array([0, 0, 0, 1])
    w = train_torch.class_weights(y, 2)  # [4/6, 4/2]
    loss_fn = train_torch.make_loss(2, w)
    z = torch.tensor([[0.3], [-1.0], [2.0], [0.5]], dtype=torch.float64)
    t = torch.from_numpy(y.astype(np.float64)).reshape(-1, 1)
    p = torch.sigmoid(z)
    hand = -(3.0 * t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()  # pos_weight = 3
    torch.testing.assert_close(loss_fn(z, t).double(), hand)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("n_classes", "y_attr"), [(2, "y_binary"), (4, "y_class")])
def test_expected_first_loss_matches_a_fresh_model(
    train: Split, n_classes: int, y_attr: str, weighted: bool
) -> None:
    y = getattr(train, y_attr)
    w = train_torch.class_weights(y, n_classes) if weighted else None
    set_seed(0)
    model = MLPClassifier(29, (64, 32), n_classes)
    ds = train_torch.FlowDataset(train.x, y, n_classes)
    loss = train_torch.evaluate_loss(model, train_torch.make_loss(n_classes, w), ds.x, ds.y)
    assert abs(loss - train_torch.expected_first_loss(y, n_classes, w)) < 0.02


# --- The training step ---------------------------------------------------------------------


def test_train_step_matches_manual_gradient_descent() -> None:
    """Two steps of train_step with plain SGD = two manual steps p -= lr · dL/dp.

    This fails if zero_grad is missing (gradients would add up across steps) or if any of
    forward → loss → backward → step is out of order.
    """
    set_seed(0)
    model = MLPClassifier(5, (4,), 3).double()
    manual = copy.deepcopy(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    rng = set_seed(1)
    batches = [
        (torch.from_numpy(rng.normal(size=(8, 5))), torch.from_numpy(rng.integers(0, 3, 8)))
        for _ in range(2)
    ]
    for xb, yb in batches:
        loss = train_torch.train_step(model, loss_fn, opt, xb, yb)
        expected = loss_fn(manual(xb), yb)
        assert loss == pytest.approx(expected.item())  # the loss before this step
        grads = torch.autograd.grad(expected, list(manual.parameters()))
        with torch.no_grad():
            for p, g in zip(manual.parameters(), grads, strict=True):
                p -= 0.5 * g
    for p, q in zip(model.parameters(), manual.parameters(), strict=True):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize(("n_classes", "y_attr"), [(2, "y_binary"), (4, "y_class")])
def test_first_loss_is_near_ln_k(train: Split, n_classes: int, y_attr: str) -> None:
    """He init with a ×0.01 last layer: every class starts near 1/K, so the loss is near ln K."""
    set_seed(0)
    model = MLPClassifier(29, (64, 32), n_classes)
    ds = train_torch.FlowDataset(train.x, getattr(train, y_attr), n_classes)
    loss = train_torch.evaluate_loss(model, train_torch.make_loss(n_classes), ds.x, ds.y)
    assert abs(loss - math.log(n_classes)) < 0.01


# --- Best weights, early stopping, determinism ---------------------------------------------


def test_kept_best_weights_do_not_change_when_training_continues(train: Split) -> None:
    """clone_state makes a real copy; a bare state_dict() shares memory with the model."""
    set_seed(0)
    model = MLPClassifier(29, (16,), 2)
    kept = train_torch.clone_state(model)
    shared = model.state_dict()  # NOT a copy: these tensors are the model's own memory
    reference = {k: v.detach().clone() for k, v in kept.items()}

    ds = train_torch.FlowDataset(train.x, train.y_binary, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = train_torch.make_loss(2)
    for xb, yb in train_torch.make_loader(ds, 64, seed=0):
        train_torch.train_step(model, loss_fn, opt, xb, yb)

    assert _states_equal(kept, reference)  # the copy kept the old values
    assert not _states_equal(shared, reference)  # the shared dict followed the training
    assert _states_equal(shared, model.state_dict())


def test_fit_restores_the_best_epoch_weights(train: Split, val: Split) -> None:
    """After early stopping, the model's validation loss is the best epoch's, not the last's."""
    config = replace(SMALL, lr=0.05, epochs=40, patience=3, hidden=(64,))
    result = train_torch.fit(config, train.x, train.y_binary, val.x, val.y_binary, 2)
    assert result.best_epoch < result.stopped_epoch  # precondition: training went past the best
    ds = train_torch.FlowDataset(val.x, val.y_binary, 2)
    loss = train_torch.evaluate_loss(result.model, train_torch.make_loss(2), ds.x, ds.y)
    assert loss == pytest.approx(result.history["val_loss"][result.best_epoch], rel=1e-6)
    assert result.history["val_loss"][result.best_epoch] == min(result.history["val_loss"])
    assert not result.model.training  # returned in eval mode
    assert result.stopped_early


def test_fit_restores_the_deterministic_setting(train: Split, val: Split) -> None:
    torch.use_deterministic_algorithms(False)
    train_torch.fit(replace(SMALL, epochs=1), train.x, train.y_binary, val.x, val.y_binary, 2)
    assert not torch.are_deterministic_algorithms_enabled()


def test_fit_is_deterministic_and_seed_dependent(train: Split, val: Split) -> None:
    config = replace(SMALL, dropout=0.2)  # dropout masks must repeat too

    def run(seed: int) -> train_torch.TorchFitResult:
        return train_torch.fit(
            replace(config, seed=seed), train.x, train.y_class, val.x, val.y_class, 4
        )

    a, b, c = run(5), run(5), run(6)
    assert a.history == b.history
    assert _states_equal(a.model.state_dict(), b.model.state_dict())
    assert not _states_equal(a.model.state_dict(), c.model.state_dict())


def test_dropout_is_random_only_in_train_mode() -> None:
    set_seed(0)
    model = MLPClassifier(29, (64,), 2, dropout=0.5)
    x = torch.randn(10, 29)
    model.train()
    assert not torch.equal(model(x), model(x))  # a new mask on each call
    model.eval()
    assert torch.equal(model(x), model(x))  # dropout is off: deterministic


# --- Saving and loading --------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path, val: Split) -> None:
    set_seed(0)
    model = MLPClassifier(29, (16, 8), 4, dropout=0.1).eval()
    meta = {"class_names": list(val.class_names), "input_sha256": val.input_sha256}
    path = tmp_path / "m.pt"
    train_torch.save_model(path, model, meta)

    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["layer_sizes"] == [29, 16, 8, 4]
    assert saved["n_classes"] == 4 and saved["dropout"] == 0.1
    assert {"schema_version", "input_sha256", "class_names"} <= saved.keys()
    # The .pt file holds only tensors: it loads with weights_only=True (no arbitrary objects).
    state = torch.load(path, weights_only=True)
    assert set(state) == set(model.state_dict())

    loaded, loaded_meta = train_torch.load_model(path, val.input_sha256)
    assert loaded_meta["class_names"] == list(val.class_names)
    np.testing.assert_array_equal(
        train_torch.predict(loaded, val.x), train_torch.predict(model, val.x)
    )


def test_load_refuses_other_schema_or_input(tmp_path: Path) -> None:
    set_seed(0)
    model = MLPClassifier(29, (8,), 2)
    path = tmp_path / "m.pt"
    train_torch.save_model(path, model, {"class_names": ["normal", "attack"], "input_sha256": "a"})
    with pytest.raises(SchemaError):
        train_torch.load_model(path, input_sha256="b")
    meta = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(json.dumps({**meta, "schema_version": "0.0"}))
    with pytest.raises(SchemaError):
        train_torch.load_model(path)
    path.with_suffix(".json").write_text(json.dumps({**meta, "model": "autoencoder"}))
    with pytest.raises(SchemaError):
        train_torch.load_model(path)


def test_save_refuses_meta_that_overrides_model_keys(tmp_path: Path) -> None:
    set_seed(0)
    model = MLPClassifier(29, (8,), 2)
    with pytest.raises(ValueError):
        train_torch.save_model(tmp_path / "m.pt", model, {"schema_version": "0.0"})


# --- The command ---------------------------------------------------------------------------


def _args(data_dir: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--data-dir", str(data_dir), "--synthetic-meta", str(data_dir / "synthetic.json"),
        "--figures-dir", str(tmp_path / "fig"), "--models-dir", str(tmp_path / "models"),
        "--numpy-model", str(data_dir / "mlp_numpy.npz"),
        "--epochs", "4", "--hidden", "16", *extra,
    ]  # fmt: skip


def test_command_binary_prints_labeled_comparison(data_dir: Path, tmp_path: Path, capsys) -> None:
    out = train_torch.main(_args(data_dir, tmp_path, "--target", "binary"))
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text
    for row in ("logistic_regression", "mlp_numpy", "mlp_torch", "random_forest (ref)"):
        assert row in out["models"]
        assert row in text
    assert "noise ceiling" in text
    assert (tmp_path / "models" / "mlp_torch_binary.pt").exists()
    assert (tmp_path / "models" / "mlp_torch_binary.json").exists()
    assert (tmp_path / "fig" / "mlp_torch_binary_loss.png").exists()


@pytest.mark.parametrize("weighted", [False, True])
def test_command_multiclass_prints_per_class_ceiling(
    data_dir: Path, tmp_path: Path, capsys, weighted: bool
) -> None:
    flags = ["--target", "multiclass"] + (["--class-weights"] if weighted else [])
    out = train_torch.main(_args(data_dir, tmp_path, *flags))
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text and "macro-F1" in text and "ceiling" in text
    assert set(out["per_class"]) == {"normal", "ssh_bruteforce", "syn_scan", "udp_flood"}
    for row in out["per_class"].values():
        assert 0.0 <= row["recall"] <= 1.0 and 0.0 <= row["recall_ceiling"] <= 1.0
    stem = "mlp_torch_multiclass" + ("_weighted" if weighted else "")
    assert (tmp_path / "models" / f"{stem}.pt").exists()
    meta = json.loads((tmp_path / "models" / f"{stem}.json").read_text())
    assert meta["config"]["class_weights"] is weighted


def test_command_twice_prints_identical_output(data_dir: Path, tmp_path: Path, capsys) -> None:
    train_torch.main(_args(data_dir, tmp_path / "a", "--target", "multiclass", "--dropout", "0.1"))
    first = capsys.readouterr().out
    train_torch.main(_args(data_dir, tmp_path / "b", "--target", "multiclass", "--dropout", "0.1"))
    second = capsys.readouterr().out
    assert first.replace(str(tmp_path / "a"), "") == second.replace(str(tmp_path / "b"), "")
