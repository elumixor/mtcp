import pytest
import torch

from src.data import Data


@pytest.fixture
def data_mock():
    features = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, -1, 9, -1],
        [11, -12, 13, 14, 15],
        [16, 17, 18, -999, 20],
    ], dtype=torch.float32)

    labels = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    weights = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    mask = torch.tensor([
        [True, True, False, True, True],
        [False, True, False, True, True],
        [True, True, False, True, True],
        [True, True, True, True, True],
    ])

    class_names = ["class0", "class1", "class2"]
    feature_names = ["feature0", "feature1", "feature2", "feature3", "feature[4]"]

    return Data(features, mask, labels, weights, class_names, feature_names)


@pytest.mark.skip
def test_uncut_number_of_samples_match(data_uncut: Data):
    assert data_uncut.process("ttH").n_samples == 834970
    assert data_uncut.process("ttH").n_samples_weighted == pytest.approx(509.3203, 0.001)

    assert data_uncut.process("ttW").n_samples == 581089
    assert data_uncut.process("ttW").n_samples_weighted == pytest.approx(1631.0088, 0.001)

    assert data_uncut.process("ttW_EW").n_samples == 15314
    assert data_uncut.process("ttW_EW").n_samples_weighted == pytest.approx(118.7130, 0.001)

    assert data_uncut.process("ttZ").n_samples == 1585762
    assert data_uncut.process("ttZ").n_samples_weighted == pytest.approx(2135.2892, 0.001)

    assert data_uncut.process("tt").n_samples == 299808
    assert data_uncut.process("tt").n_samples_weighted == pytest.approx(32626.9943, 0.001)

    assert data_uncut.process("VV").n_samples == 3829589
    assert data_uncut.process("VV").n_samples_weighted == pytest.approx(7635.6123, 0.001)

    assert data_uncut.process("tZ").n_samples == 40946
    assert data_uncut.process("tZ").n_samples_weighted == pytest.approx(373.1321, 0.001)

    assert data_uncut.process("WtZ").n_samples == 45931
    assert data_uncut.process("WtZ").n_samples_weighted == pytest.approx(245.5160, 0.001)

    assert data_uncut.process("tW").n_samples == 4493
    assert data_uncut.process("tW").n_samples_weighted == pytest.approx(398.7966, 0.001)

    assert data_uncut.process("threeTop").n_samples == 22980
    assert data_uncut.process("threeTop").n_samples_weighted == pytest.approx(7.8570, 0.001)

    assert data_uncut.process("fourTop").n_samples == 22746
    assert data_uncut.process("fourTop").n_samples_weighted == pytest.approx(44.0129, 0.001)

    assert data_uncut.process("ggVV").n_samples == 792331
    assert data_uncut.process("ggVV").n_samples_weighted == pytest.approx(366.8159, 0.001)

    assert data_uncut.process("VVV").n_samples == 151324
    assert data_uncut.process("VVV").n_samples_weighted == pytest.approx(60.3720, 0.001)

    assert data_uncut.process("VH").n_samples == 255
    assert data_uncut.process("VH").n_samples_weighted == pytest.approx(101.0673, 0.001)

    assert data_uncut.process("ttWW").n_samples == 5944
    assert data_uncut.process("ttWW").n_samples_weighted == pytest.approx(49.2243, 0.001)

    assert data_uncut.process("tHjb").n_samples == 445666
    assert data_uncut.process("tHjb").n_samples_weighted == pytest.approx(20.8376, 0.001)

    assert data_uncut.process("tWH").n_samples == 30587
    assert data_uncut.process("tWH").n_samples_weighted == pytest.approx(15.5663, 0.001)

    # Total
    assert data_uncut.n_samples == 8709735
    assert data_uncut.n_samples_weighted == pytest.approx(46340.1358, 0.001)


@pytest.mark.skip
def test_n_samples_same_with_batch_iteration(data_cut: Data):
    n_samples_per_class = [(data_cut.labels == i).sum() for i in range(data_cut.n_classes)]
    n_samples_per_class_batched = [0 for i in range(len(data_cut))]

    for batch in data_cut.batches():
        for i in range(data_cut.n_classes):
            n_samples_per_class_batched[i] += (batch.labels == i).sum()

    for i in range(data_cut.n_classes):
        assert n_samples_per_class[i] == n_samples_per_class_batched[i]


@pytest.mark.parametrize("index", [
    [0, 1],
    slice(2)
])
def test_indexing(data_mock: Data, index):
    data = data_mock[index]

    assert data.n_features == 5
    assert data.x_names == ["feature0", "feature1", "feature2", "feature3", "feature[4]"]
    assert data.n_samples == 2
    assert data.n_classes == 3

    assert (data.features == torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, -1, 9, -1],
    ], dtype=torch.float32)).all()
    assert (data.labels == torch.tensor([0, 1], dtype=torch.long)).all()
    assert (data.weights == torch.tensor([1, 2], dtype=torch.float32)).all()
    assert (data.mask == torch.tensor([
        [True, True, False, True, True],
        [False, True, False, True, True],
    ], dtype=torch.bool)).all()


@pytest.mark.parametrize("options", [
    {"indices": [0, 2]},
    {"names": ["feature0", "feature2"]},
])
def test_select_features(data_mock: Data, options: dict):
    data = data_mock.select_features(**options)

    assert data.n_features == 2
    assert data.x_names == ["feature0", "feature2"]
    assert data.n_samples == 4
    assert data.n_classes == 3

    assert (data.features == torch.tensor([
        [1, 3],
        [6, -1],
        [11, 13],
        [16, 18],
    ], dtype=torch.float32)).all()
    assert (data.labels == torch.tensor([0, 1, 0, 2], dtype=torch.long)).all()
    assert (data.weights == torch.tensor([1, 2, 3, 4], dtype=torch.float32)).all()
    assert (data.mask == torch.tensor([
        [True, False],
        [False, False],
        [True, False],
        [True, True],
    ], dtype=torch.bool)).all()


def test_select_features_regex(data_mock: Data):
    data = data_mock.select_features(["feature[0-2]"])

    assert data == data_mock.select_features(["feature0", "feature1", "feature2"])
    assert data != data_mock.select_features(["feature1", "feature2"])

    data = data_mock.select_features(["feature[0-2]", "feature[4]"])

    assert data == data_mock.select_features(["feature0", "feature1", "feature2", "feature[4]"])


def test_drop_features(data_mock: Data):
    data = data_mock.drop_features(["feature[0-2]", "feature[4]"])

    assert data.n_features == 1
    assert data.x_names == ["feature3"]
    assert data.n_samples == 4
    assert data.n_classes == 3

    assert (data.features == torch.tensor([
        [4],
        [9],
        [14],
        [-999],
    ], dtype=torch.float32)).all()

    assert (data.labels == torch.tensor([0, 1, 0, 2], dtype=torch.long)).all()
    assert (data.weights == torch.tensor([1, 2, 3, 4], dtype=torch.float32)).all()
    assert (data.mask == torch.tensor([
        [True],
        [True],
        [True],
        [True],
    ], dtype=torch.bool)).all()


def test_no_copying_of_the_data(data_mock: Data):
    other = data_mock[:2]
    other.features[0, 0] = 100
    assert data_mock.features[0, 0] == 100


def test_mask_invalid_features(data_mock: Data):
    invalid = {
        -1: ["feature[0-2]", "feature[4]"],
        -999: ["feature1", "feature3"]
    }

    masked = mask_invalid(data_mock, invalid)

    assert (masked.mask == torch.tensor([
        [True, True, False, True, True],
        [False, True, False, True, False],
        [True, True, False, True, True],
        [True, True, True, False, True],
    ], dtype=torch.bool)).all()


def test_nonmasked_features(data_mock: Data):
    nonmasked = data_mock.nonmasked_features

    assert (nonmasked[0] == torch.tensor([1, 11, 16], dtype=torch.float32)).all()
    assert (nonmasked[1] == torch.tensor([2, 7, -12, 17], dtype=torch.float32)).all()
    assert (nonmasked[2] == torch.tensor([18], dtype=torch.float32)).all()
    assert (nonmasked[3] == torch.tensor([4, 9, 14, -999], dtype=torch.float32)).all()
    assert (nonmasked[4] == torch.tensor([5, -1, 15, 20], dtype=torch.float32)).all()


def test_mean(data_mock: Data):
    mean = data_mock.mean()

    assert (mean == torch.tensor([
        (1 + 11 + 16) / 3,
        (2 + 7 + -12 + 17) / 4,
        18,
        (4 + 9 + 14 + -999) / 4,
        (5 + -1 + 15 + 20) / 4,
    ], dtype=torch.float32)).all()


def test_std(data_mock: Data):
    std = data_mock.std()

    assert std[0] == torch.tensor([1, 11, 16], dtype=torch.float32).std()
    assert std[1] == torch.tensor([2, 7, -12, 17], dtype=torch.float32).std()
    assert torch.isnan(std[2])
    assert std[3] == torch.tensor([4, 9, 14, -999], dtype=torch.float32).std()
    assert std[4] == torch.tensor([5, -1, 15, 20], dtype=torch.float32).std()


def test_unique(data_mock: Data):
    unique = data_mock.unique()

    assert list(unique.keys()) == ["feature0", "feature1", "feature2", "feature3", "feature[4]"]

    assert set(unique["feature0"].tolist()) == set([1, 11, 16])
    assert set(unique["feature1"].tolist()) == set([2, 7, -12, 17])
    assert set(unique["feature2"].tolist()) == set([18])
    assert set(unique["feature3"].tolist()) == set([4, 9, 14, -999])
    assert set(unique["feature[4]"].tolist()) == set([5, -1, 15, 20])


def test_correlated_removes_correct_indices(data_mock: Data):
    correlated = data_mock.correlated_features(names=True, max_correlation=0)
    assert len(correlated) == len(set(correlated)) == data_mock.n_features - 1

    for c in correlated:
        assert c in data_mock.x_names

    correlated = data_mock.correlated_features(indices=True, max_correlation=0)
    assert len(correlated) == len(set(correlated)) == data_mock.n_features - 1


def test_compute_stats(data_mock: Data):
    features = ["feature[0-2]", "feature[4]"]
    categorical = ["feature0"]

    mean, std, unique, features = compute_stats(data_mock, features, categorical, max_correlation=1)

    assert (
        mean.nan_to_num(nan=100500) ==
        data_mock.select_features(["feature1", "feature2", "feature[4]"]).mean().nan_to_num(nan=100500)
    ).all()
    assert (
        std.nan_to_num(nan=100500) ==
        data_mock.select_features(["feature1", "feature2", "feature[4]"]).std().nan_to_num(nan=100500)
    ).all()

    assert list(unique.keys()) == ["feature0"]

    assert (unique["feature0"] == torch.tensor([1, 11, 16], dtype=torch.float32)).all()

    mean, std, unique, correlated = compute_stats(data_mock, data_mock.x_names, categorical, max_correlation=0)

    assert not mean.isnan().any()
    assert not std.isnan().any()

    for values in unique:
        assert len(values) > 1


def test_split_data_no_cut_sizes(data_mock: Data):
    datasets = split_data(data_mock, trn=0.5, val=0.5)
    assert len(datasets) == 4

    trn_uncut, trn_cut, val, tst = datasets

    assert trn_uncut.n_samples == trn_cut.n_samples == val.n_samples == 2
    assert tst.n_samples == 0

    datasets = split_data(data_mock, trn=1)

    assert len(datasets) == 4

    trn_uncut, trn_cut, val, tst = datasets

    assert trn_uncut.n_samples == trn_cut.n_samples == 4
    assert val.n_samples == tst.n_samples == 0

    datasets = split_data(data_mock, trn=0.5, val=0.25, tst=0.25)

    assert len(datasets) == 4

    trn_uncut, trn_cut, val, tst = datasets

    assert trn_uncut.n_samples == trn_cut.n_samples == 2
    assert val.n_samples == tst.n_samples == 1


@pytest.mark.repeat(10)
def test_split_data_no_cut_no_overlap(data_mock: Data):
    print(data_mock.features)

    datasets = split_data(data_mock, trn=0.5, val=0.25, tst=0.25)

    assert len(datasets) == 4

    trn_uncut, trn_cut, val, tst = datasets

    assert trn_uncut.n_samples == trn_cut.n_samples == 2
    assert val.n_samples == tst.n_samples == 1

    # Check that samples are not overlapping
    assert not (val.features[0] == tst.features[0]).all()

    for sample in trn_cut.features:
        assert not (val.features[0] == sample).all()
        assert not (tst.features[0] == sample).all()


@pytest.mark.repeat(10)
def test_split_data_with_cut(data_mock: Data):
    datasets = split_data(data_mock, trn=1.0, cut_expr="(feature0 > 5) & (feature2 < 15)")

    assert len(datasets) == 4

    trn_uncut, trn_cut, val, tst = datasets

    assert trn_uncut.n_samples == 4
    assert trn_cut.n_samples == 2
    assert val.n_samples == tst.n_samples == 0

    i = 0 if (trn_cut.features[0] == torch.tensor([6, 7, -1, 9, -1], dtype=torch.float32)).all() else 1

    features = trn_cut.features[[i, 1 - i]]
    labels = trn_cut.labels[[i, 1 - i]]
    weights = trn_cut.weights[[i, 1 - i]]
    mask = trn_cut.mask[[i, 1 - i]]

    assert (features == torch.tensor([
        [6, 7, -1, 9, -1],
        [11, -12, 13, 14, 15],
    ], dtype=torch.float32)).all()

    assert (labels == torch.tensor([1, 0], dtype=torch.long)).all()
    assert (weights == torch.tensor([2, 3], dtype=torch.float32)).all()
    assert (mask == torch.tensor([
        [False, True, False, True, True],
        [True, True, False, True, True],
    ])).all()


def test_one_hot_encode(data_mock: Data):
    data = data_mock.select_features(["feature0", "feature1"])
    unique = data.unique()
    data = one_hot_encode(data, unique)

    names = []
    for feature, values in unique.items():
        assert feature not in data.feature_names
        for value in values:
            name = f"{feature}={value}"
            assert name in data.feature_names

            names.append(name)

    assert data.n_features == len(names)

    unique = data.unique()

    assert len(unique) == len(names)

    for values in unique.values():
        assert len(values) == 2
        assert (values == torch.tensor([0, 1], dtype=torch.float32)).all()


def test_normalize(data_mock: Data):
    data = data_mock.normalized()

    mean = data.mean()
    std = data.std()

    assert mean.nan_to_num(nan=100500).allclose(torch.tensor([0, 0, 100500, 0, 0], dtype=torch.float32), atol=1e-5)
    assert std.nan_to_num(nan=100500).allclose(torch.tensor([1, 1, 100500, 1, 1], dtype=torch.float32), atol=1e-5)

    data = data_mock.drop_features(["feature2"])

    mean = data.mean()
    std = data.std()

    assert not mean.isnan().any()
    assert not std.isnan().any()

    data = data_mock.normalized(mean, std, features=["feature[0-1]", "feature3", "feature[4]"])

    assert data.x_names == data_mock.x_names

    print(data.features)

    mean = data.mean()
    std = data.std()

    assert mean[0] == pytest.approx(0, abs=1e-5)
    assert mean[1] == pytest.approx(0, abs=1e-5)
    assert mean[2] == data.features[3, 2]
    assert mean[3] == pytest.approx(0, abs=1e-5)
    assert mean[4] == pytest.approx(0, abs=1e-5)

    assert std[0] == pytest.approx(1, abs=1e-5)
    assert std[1] == pytest.approx(1, abs=1e-5)
    assert std[2].isnan()
    assert std[3] == pytest.approx(1, abs=1e-5)
    assert std[4] == pytest.approx(1, abs=1e-5)
