"""Command-level tests for `neural_ids.train_baselines` on a small SYNTHETIC dataset."""

import json
import math
from pathlib import Path

import pytest

from neural_ids import preprocess, synthetic, train_baselines
from neural_ids.splits import load_split
from neural_ids.utils import file_sha256


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """synthetic.csv/json -> preprocess -> {train,val}.npz. test.npz is deleted on purpose."""
    root = tmp_path_factory.mktemp("t04")
    csv = root / "synthetic.csv"
    synthetic.main(["--n", "1500", "--seed", "3", "--out", str(csv)])
    preprocess.main(
        ["--input", str(csv), "--out-dir", str(root), "--preprocessor", str(root / "pre.json")]
    )
    (root / "test.npz").unlink()  # the command must not need (or open) the test split
    return root


def _run(data_dir: Path, figures: Path, epochs: int = 5) -> dict:
    return train_baselines.main(
        [
            "--data-dir", str(data_dir),
            "--synthetic-meta", str(data_dir / "synthetic.json"),
            "--figures-dir", str(figures),
            "--epochs", str(epochs),
        ]
    )  # fmt: skip


def test_command_table_ceiling_and_figures(data_dir: Path, tmp_path: Path, capsys) -> None:
    results = _run(data_dir, tmp_path / "fig")
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "noise ceiling" in out
    assert "TN" in out and "TP" in out
    assert (tmp_path / "fig" / "logreg_loss.png").stat().st_size > 0
    assert (tmp_path / "fig" / "perceptron_errors.png").stat().st_size > 0

    models = results["models"]
    assert models["majority"]["recall"] == 0.0 and models["majority"]["f1"] == 0.0
    # "clearly beats the majority baseline" (which scores 0 on both)
    assert models["logistic_regression"]["recall"] > 0.3
    assert models["logistic_regression"]["f1"] > 0.3
    sizes = {int(m["confusion"].sum()) for m in models.values()}
    assert len(sizes) == 1  # every model scored on the same val rows

    losses = results["history"]["logreg"]["train_loss"]
    assert losses[0] == pytest.approx(math.log(2), abs=1e-6)
    assert losses[-1] < losses[0]


def test_ceiling_counts_only_val_flips(data_dir: Path, tmp_path: Path) -> None:
    results = _run(data_dir, tmp_path / "fig", epochs=1)
    meta = json.loads((data_dir / "synthetic.json").read_text())
    n_flipped = results["ceiling"]["n_flipped"]
    assert 0 < n_flipped < len(meta["flipped_flow_ids"])  # val holds only part of the flips
    assert 0 < results["ceiling"]["recall"] <= 1.0
    assert 0 < results["ceiling"]["precision"] <= 1.0


def test_synthetic_json_read_only_after_training(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real_train, real_read = train_baselines.train_models, train_baselines.read_flipped_ids

    def train_spy(*args, **kwargs):
        calls.append("train")
        return real_train(*args, **kwargs)

    def read_spy(*args, **kwargs):
        calls.append("read_meta")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(train_baselines, "train_models", train_spy)
    monkeypatch.setattr(train_baselines, "read_flipped_ids", read_spy)
    _run(data_dir, tmp_path / "fig", epochs=1)
    assert calls == ["train", "read_meta"]


def test_no_ceiling_without_metadata(data_dir: Path, tmp_path: Path, capsys) -> None:
    results = train_baselines.main(
        [
            "--data-dir", str(data_dir),
            "--synthetic-meta", str(tmp_path / "missing.json"),
            "--figures-dir", str(tmp_path / "fig"),
            "--epochs", "1",
        ]
    )  # fmt: skip
    assert results["ceiling"] is None
    assert "no label-noise ceiling" in capsys.readouterr().out


def _write_meta(data_dir: Path, path: Path, **changes) -> Path:
    meta = json.loads((data_dir / "synthetic.json").read_text())
    meta.update(changes)
    path.write_text(json.dumps(meta))
    return path


def _run_with_meta(data_dir: Path, meta: Path, tmp_path: Path) -> dict:
    return train_baselines.main(
        [
            "--data-dir", str(data_dir),
            "--synthetic-meta", str(meta),
            "--figures-dir", str(tmp_path / "fig"),
            "--epochs", "1",
        ]
    )  # fmt: skip


def test_hashes_match_after_synthetic_then_preprocess(data_dir: Path) -> None:
    meta = json.loads((data_dir / "synthetic.json").read_text())
    val = load_split(data_dir / "val.npz")
    assert meta["csv_sha256"] == file_sha256(data_dir / "synthetic.csv") == val.input_sha256
    assert len(val.input_sha256) == 64


@pytest.mark.parametrize("changes", [{"csv_sha256": "0" * 64}, {"csv_sha256": None}])
def test_hash_mismatch_skips_ceiling(data_dir: Path, tmp_path: Path, capsys, changes: dict) -> None:
    """A synthetic.json from another run (or without a hash) must not produce a ceiling."""
    meta = _write_meta(data_dir, tmp_path / "meta.json", **changes)
    results = _run_with_meta(data_dir, meta, tmp_path)
    assert results["ceiling"] is None
    out = capsys.readouterr().out
    assert "different runs" in out
    assert "noise ceiling*" not in out


def test_regenerated_csv_with_other_seed_is_detected(data_dir: Path, tmp_path: Path) -> None:
    """The real failure case: same flow IDs, different data, stale arrays."""
    other = tmp_path / "synthetic.csv"
    synthetic.main(["--n", "1500", "--seed", "4", "--out", str(other)])
    results = _run_with_meta(data_dir, tmp_path / "synthetic.json", tmp_path)
    assert results["ceiling"] is None
    assert "different runs" in results["ceiling_note"]
