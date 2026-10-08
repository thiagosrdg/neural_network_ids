"""Tests for `neural_ids.preprocess`: split, Standardizer, Preprocessor, targets, and command.

All data here is synthetic: these tests check that preprocessing is correct and leak-free, not
that anything detects real attacks.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from neural_ids import preprocess
from neural_ids.preprocess import (
    LOG_FEATURES,
    NORMAL_CLASS,
    ONE_HOT_FEATURES,
    Preprocessor,
    Standardizer,
    class_names_from,
    count_exact_copies,
    data_source,
    main,
    make_targets,
    split,
)
from neural_ids.schema import (
    FEATURE_NAMES,
    METADATA_COLUMNS,
    OPTIONAL_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
)
from neural_ids.synthetic import make_dataset
from neural_ids.utils import set_seed


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    """1000 SYNTHETIC flows, 30% attacks, with label noise (the default)."""
    return make_dataset(1000, attack_fraction=0.3, seed=0)


@pytest.fixture(scope="module")
def fitted(flows: pd.DataFrame) -> tuple[Preprocessor, pd.DataFrame, pd.DataFrame]:
    """A Preprocessor fitted on the training split, with the train and validation splits."""
    train, val, _ = split(flows, seed=0)
    return Preprocessor().fit(train), train, val


# --- split -------------------------------------------------------------------------------


def test_split_sizes_cover_every_row_once(flows: pd.DataFrame) -> None:
    train, val, test = split(flows, val_fraction=0.15, test_fraction=0.15, seed=1)
    assert len(train) + len(val) + len(test) == len(flows)
    ids = [set(part["flow_id"]) for part in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert ids[0] | ids[1] | ids[2] == set(flows["flow_id"])
    # About 70/15/15; rounding per class moves at most a few rows (6 labels).
    assert abs(len(val) - 150) <= 6 and abs(len(test) - 150) <= 6


def test_split_is_stratified_by_label(flows: pd.DataFrame) -> None:
    train, val, test = split(flows, seed=2)
    totals = flows["label"].value_counts()
    for part, fraction in ((val, 0.15), (test, 0.15), (train, 0.70)):
        counts = part["label"].value_counts().reindex(totals.index, fill_value=0)
        # Per class: round() of the expected count, so off by at most 1 (train: 2).
        assert (abs(counts - totals * fraction) <= 2).all()


def test_split_same_seed_same_rows(flows: pd.DataFrame) -> None:
    first = [part["flow_id"].tolist() for part in split(flows, seed=3)]
    second = [part["flow_id"].tolist() for part in split(flows, seed=3)]
    other = [part["flow_id"].tolist() for part in split(flows, seed=4)]
    assert first == second
    assert first != other


@pytest.mark.parametrize("val, test", [(-0.1, 0.15), (0.15, -0.1), (0.5, 0.5), (0.6, 0.5)])
def test_split_rejects_bad_fractions(flows: pd.DataFrame, val: float, test: float) -> None:
    with pytest.raises(ValueError):
        split(flows, val_fraction=val, test_fraction=test, seed=0)


def test_split_needs_label_column(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="label"):
        split(flows.drop(columns="label"), seed=0)


# --- Standardizer --------------------------------------------------------------------------


def _wide_range_matrix(n: int = 14_000) -> np.ndarray:
    """(n, 5) float64: two very different scales, a zero column, a constant nonzero column."""
    rng = set_seed(0)
    return np.column_stack(
        [
            rng.normal(5.0, 2.0, n),
            rng.lognormal(10.0, 2.0, n),  # heavy tail, values up to about 1e8
            np.zeros(n),  # like urg_count in the synthetic data
            np.full(n, np.log1p(1.0)),  # 0.693...: np.std gives about 1e-16, not 0
            rng.uniform(-1.0, 1.0, n),
        ]
    )


def test_standardizer_matches_sklearn() -> None:
    x = _wide_range_matrix()  # x: (14000, 5)
    ours = Standardizer().fit(x)
    reference = StandardScaler().fit(x)
    np.testing.assert_allclose(ours.mean_, reference.mean_, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ours.scale_, reference.scale_, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ours.transform(x), reference.transform(x), rtol=1e-6, atol=1e-6)


def test_standardizer_constant_columns_get_scale_one() -> None:
    x = _wide_range_matrix()
    s = Standardizer().fit(x)
    assert np.std(x[:, 3]) > 0  # floating-point noise: a naive `std == 0` misses this column
    assert s.scale_[2] == 1.0 and s.scale_[3] == 1.0
    # A value never seen in training moves by its distance to the mean, not by ~1e15.
    new = x[:2].copy()
    new[:, 2] = 1.0  # one URG flag at scoring time
    new[:, 3] = np.log1p(2.0)
    z = s.transform(new)
    np.testing.assert_allclose(z[:, 2], 1.0)
    np.testing.assert_allclose(z[:, 3], np.log1p(2.0) - np.log1p(1.0))


def test_standardizer_train_output_has_mean_0_std_1() -> None:
    x = _wide_range_matrix()
    z = Standardizer().fit(x).transform(x)
    varying = [0, 1, 4]
    np.testing.assert_allclose(z[:, varying].mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z[:, varying].std(axis=0), 1.0, atol=1e-9)


def test_standardizer_uses_only_fit_statistics() -> None:
    rng = set_seed(1)
    train = rng.normal(0.0, 1.0, size=(500, 3))
    shifted = rng.normal(100.0, 10.0, size=(200, 3))  # very different from train
    s = Standardizer().fit(train)
    z = s.transform(shifted)
    np.testing.assert_allclose(z, (shifted - train.mean(axis=0)) / train.std(axis=0))
    assert (z.mean(axis=0) > 50).all()  # not re-centered on the shifted data


def test_standardizer_errors() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Standardizer().transform(np.zeros((2, 3)))
    s = Standardizer().fit(np.ones((4, 3)))
    with pytest.raises(ValueError, match="columns"):
        s.transform(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        Standardizer().fit(np.zeros((0, 3)))


def test_standardizer_dict_round_trip() -> None:
    x = _wide_range_matrix(100)
    s = Standardizer().fit(x)
    copy = Standardizer.from_dict(json.loads(json.dumps(s.to_dict())))
    np.testing.assert_array_equal(copy.transform(x), s.transform(x))


# --- Preprocessor --------------------------------------------------------------------------


def test_log_features_are_exactly_the_non_one_hot_features() -> None:
    assert set(LOG_FEATURES) | set(ONE_HOT_FEATURES) == set(FEATURE_NAMES)
    assert set(LOG_FEATURES).isdisjoint(ONE_HOT_FEATURES)
    assert len(LOG_FEATURES) == 21


def test_preprocessor_output_shape_dtype_finite(fitted) -> None:
    pre, train, val = fitted
    for part in (train, val):
        x = pre.transform(part)
        assert x.shape == (len(part), len(FEATURE_NAMES))
        assert x.dtype == np.float32
        assert np.isfinite(x).all()


def test_preprocessor_keeps_one_hot_and_standardizes_the_rest(fitted) -> None:
    pre, train, _ = fitted
    x = pre.transform(train)  # x: (n_train, 29)
    for i, name in enumerate(FEATURE_NAMES):
        if name in ONE_HOT_FEATURES:
            np.testing.assert_array_equal(x[:, i], train[name].to_numpy(dtype=np.float32))
    i = FEATURE_NAMES.index("bytes_per_s")
    logged = np.log1p(train["bytes_per_s"].to_numpy())
    np.testing.assert_allclose(x[:, i], (logged - logged.mean()) / logged.std(), rtol=1e-5)


def test_preprocessor_statistics_come_from_training_only(flows: pd.DataFrame) -> None:
    # Train and validation differ strongly: benign web flows against UDP floods.
    train = flows[flows["label"] == "web"].reset_index(drop=True)
    val = flows[flows["label"] == "udp_flood"].reset_index(drop=True)
    pre = Preprocessor().fit(train)
    names = list(pre.standardized_features)
    expected = np.log1p(train[names].to_numpy(dtype=np.float64)).mean(axis=0)
    np.testing.assert_allclose(pre.standardizer.mean_, expected)
    expected_std = np.log1p(train[names].to_numpy(dtype=np.float64)).std(axis=0)
    np.testing.assert_allclose(pre.standardizer.scale_, np.where(expected_std > 0, expected_std, 1))
    both = Preprocessor().fit(pd.concat([train, val], ignore_index=True))
    assert not np.allclose(both.standardizer.mean_, pre.standardizer.mean_)
    i = FEATURE_NAMES.index("proto_udp")
    # One-hot, untouched (label noise: a few "udp_flood" rows are flipped non-UDP flows).
    np.testing.assert_array_equal(pre.transform(val)[:, i], val["proto_udp"].to_numpy())
    j = FEATURE_NAMES.index("bwd_packets")
    assert abs(pre.transform(val)[:, j].mean()) > 1.0  # floods get no replies: far from 0


def test_metadata_never_reaches_x(fitted) -> None:
    pre, train, _ = fitted
    assert set(pre.feature_names).isdisjoint(set(METADATA_COLUMNS) | set(OPTIONAL_COLUMNS))
    changed = train.copy()
    changed["flow_id"] = [f"other-{i}" for i in range(len(changed))]
    changed["src_ip"] = "203.0.113.7"
    changed["dst_ip"] = "2001:db8::1"
    changed["src_port"] = 40000
    changed["dst_port"] = 12345
    changed["start_time"] = changed["start_time"] + 1e6
    changed["end_time"] = changed["end_time"] + 1e6
    changed["label"] = "something_else"
    changed["is_attack"] = 1 - changed["is_attack"]
    np.testing.assert_array_equal(pre.transform(changed), pre.transform(train))


def test_transform_validates_the_contract(fitted) -> None:
    pre, train, _ = fitted
    with pytest.raises(SchemaError, match="missing"):
        pre.transform(train.drop(columns="duration_s"))
    with pytest.raises(SchemaError, match="unexpected"):
        pre.transform(train.assign(src_mac="02:00:00:00:00:01"))
    broken = train.copy()
    broken.loc[0, "bytes_per_s"] = np.nan
    with pytest.raises(SchemaError, match="non-finite"):
        pre.transform(broken)


def test_preprocessor_errors_before_fit(flows: pd.DataFrame, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Preprocessor().transform(flows)
    with pytest.raises(RuntimeError, match="fit"):
        Preprocessor().save(tmp_path / "p.json")


def test_json_round_trip(fitted, tmp_path) -> None:
    pre, _, val = fitted
    path = tmp_path / "sub" / "preprocessor.json"
    pre.save(path)
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["feature_names"] == list(FEATURE_NAMES)
    assert data["log1p_features"] == list(LOG_FEATURES)
    assert len(data["mean"]) == len(data["scale"]) == len(data["standardized_features"]) == 21
    loaded = Preprocessor.load(path)
    np.testing.assert_array_equal(loaded.transform(val), pre.transform(val))


def test_loaded_preprocessor_ignores_later_constant_changes(fitted, tmp_path, monkeypatch) -> None:
    pre, _, val = fitted
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    before = pre.transform(val)
    monkeypatch.setattr(preprocess, "LOG_FEATURES", ())  # a later code change
    np.testing.assert_array_equal(Preprocessor.load(path).transform(val), before)


def _edit_json(path, **changes) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_load_refuses_other_schema_version(fitted, tmp_path) -> None:
    pre, _, _ = fitted
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    _edit_json(path, schema_version="0.9")
    with pytest.raises(SchemaError, match="schema version"):
        Preprocessor.load(path)


def test_load_refuses_other_feature_order(fitted, tmp_path) -> None:
    pre, _, _ = fitted
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    _edit_json(path, feature_names=list(reversed(FEATURE_NAMES)))
    with pytest.raises(SchemaError, match="feature order"):
        Preprocessor.load(path)


def test_load_refuses_inconsistent_file(fitted, tmp_path) -> None:
    pre, _, _ = fitted
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    data = json.loads(path.read_text())
    _edit_json(path, scale=data["scale"][:-1])
    with pytest.raises(ValueError):
        Preprocessor.load(path)


# --- targets -------------------------------------------------------------------------------


def _labels(labels: list[str], is_attack: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"label": labels, "is_attack": is_attack})


def test_targets_come_from_is_attack_not_label_names() -> None:
    df = _labels(["web", "dns", "ssh_bruteforce", "web", "ssh_session"], [0, 0, 1, 1, 0])
    names = class_names_from(df)
    assert names == (NORMAL_CLASS, "ssh_bruteforce", "web")
    y_binary, y_class = make_targets(df, names)
    np.testing.assert_array_equal(y_binary, [0, 0, 1, 1, 0])
    np.testing.assert_array_equal(y_class, [0, 0, 1, 2, 0])
    assert y_binary.dtype == np.int64 and y_class.dtype == np.int64


def test_targets_reject_unknown_attack_label() -> None:
    names = class_names_from(_labels(["web", "syn_scan"], [0, 1]))
    with pytest.raises(ValueError, match="udp_flood"):
        make_targets(_labels(["udp_flood"], [1]), names)


def test_targets_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="normal"):
        class_names_from(_labels(["normal"], [1]))  # an attack may not use the reserved name
    with pytest.raises(ValueError, match="is_attack"):
        make_targets(_labels(["web"], [2]), (NORMAL_CLASS,))
    with pytest.raises(ValueError, match="is_attack"):
        class_names_from(pd.DataFrame({"label": ["web"]}))


# --- exact copies --------------------------------------------------------------------------


def test_count_exact_copies(flows: pd.DataFrame) -> None:
    reference = flows.iloc[:100]
    other = flows.iloc[[3, 50, 7]].reset_index(drop=True)
    other.loc[0, "flow_id"] = "new-id"  # metadata does not matter, only features
    other.loc[2, "duration_s"] = 1799.123  # no longer a copy
    assert count_exact_copies(reference, other) == 2
    assert count_exact_copies(reference, other.iloc[2:]) == 0


# --- command -------------------------------------------------------------------------------


def test_command_writes_splits_and_preprocessor(tmp_path, capsys) -> None:
    csv = tmp_path / "flows.csv"
    flows = make_dataset(400, attack_fraction=0.3, seed=5)
    flows.to_csv(csv, index=False)
    model = tmp_path / "models" / "preprocessor.json"
    main(["--input", str(csv), "--out-dir", str(tmp_path / "out"), "--preprocessor", str(model)])

    total, d = 0, set()
    for name in ("train", "val", "test"):
        with np.load(tmp_path / "out" / f"{name}.npz", allow_pickle=False) as data:
            x, y_binary, y_class = data["X"], data["y_binary"], data["y_class"]
            class_names, feature_names = data["class_names"], data["feature_names"]
            schema_version, source = data["schema_version"], data["source"]
        assert x.dtype == np.float32 and x.ndim == 2 and np.isfinite(x).all()
        assert y_binary.shape == y_class.shape == (len(x),)
        assert class_names.dtype.kind == "U" and class_names[0] == NORMAL_CLASS
        assert tuple(feature_names) == FEATURE_NAMES
        assert str(schema_version) == SCHEMA_VERSION and str(source) == "SYNTHETIC"
        assert ((y_class == 0) == (y_binary == 0)).all()
        assert y_class.max() < len(class_names)
        total += len(x)
        d.add(x.shape[1])
    assert total == 400 and d == {len(FEATURE_NAMES)}
    assert Preprocessor.load(model).feature_names == FEATURE_NAMES
    out = capsys.readouterr().out
    assert "train" in out and "exact copies" in out and "SYNTHETIC" in out


def test_command_fits_on_the_training_split_only(tmp_path) -> None:
    flows = make_dataset(400, attack_fraction=0.3, seed=6)
    csv = tmp_path / "flows.csv"
    flows.to_csv(csv, index=False)
    model = tmp_path / "preprocessor.json"
    main(["--input", str(csv), "--out-dir", str(tmp_path), "--preprocessor", str(model)])

    train, _, _ = split(pd.read_csv(csv), seed=42)  # the command's default seed
    saved = json.loads(model.read_text())
    on_train = Preprocessor().fit(train).standardizer
    on_all = Preprocessor().fit(pd.read_csv(csv)).standardizer
    np.testing.assert_allclose(saved["mean"], on_train.mean_)
    np.testing.assert_allclose(saved["scale"], on_train.scale_)
    assert not np.allclose(saved["mean"], on_all.mean_)
    with np.load(tmp_path / "train.npz", allow_pickle=False) as data:
        assert tuple(data["class_names"]) == class_names_from(train)


def test_command_flow_ids_line_up_with_rows(tmp_path) -> None:
    flows = make_dataset(400, attack_fraction=0.3, seed=8)
    csv = tmp_path / "flows.csv"
    flows.to_csv(csv, index=False)
    model = tmp_path / "preprocessor.json"
    main(["--input", str(csv), "--out-dir", str(tmp_path), "--preprocessor", str(model)])

    pre = Preprocessor.load(model)
    by_id = pd.read_csv(csv).set_index("flow_id")
    seen: list[str] = []
    for name in ("train", "val", "test"):
        with np.load(tmp_path / f"{name}.npz", allow_pickle=False) as data:
            x, y_binary, ids = data["X"], data["y_binary"], data["flow_id"]
        assert ids.dtype.kind == "U" and ids.shape == (len(x),)
        rows = by_id.loc[ids].reset_index()  # the CSV rows in the order the arrays claim
        np.testing.assert_array_equal(pre.transform(rows), x)
        np.testing.assert_array_equal(rows["is_attack"].to_numpy(), y_binary)
        seen.extend(ids.tolist())
    assert sorted(seen) == sorted(flows["flow_id"])  # every flow exactly once


def test_data_source_detects_synthetic(flows: pd.DataFrame) -> None:
    assert data_source(flows) == "SYNTHETIC"
    assert data_source(flows.assign(flow_id="capture-1")) == "unspecified"
