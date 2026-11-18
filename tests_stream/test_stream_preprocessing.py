"""Tests for incremental NumCompute preprocessing."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    MinMaxScaler,
    NotFittedError,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(120, 3))
    X[::9, 1] = np.nan
    return X


def feed(model, X, size):
    for i in range(0, len(X), size):
        model.partial_fit(X[i:i + size])
    return model


# StandardScaler
@pytest.mark.parametrize("size", [1, 7, 40, 120, 500])
def test_scaler_chunks_match_batch(data, size):
    batch = StandardScaler().fit(data)
    stream = feed(StandardScaler(), data, size)
    np.testing.assert_allclose(stream.mean_, batch.mean_)
    np.testing.assert_allclose(stream.scale_, batch.scale_)
    np.testing.assert_allclose(stream.transform(data),
                               batch.transform(data))


def test_scaler_counts_ignore_nan(data):
    s = feed(StandardScaler(), data, 25)
    expected = (~np.isnan(data)).sum(axis=0)
    np.testing.assert_array_equal(s.n_samples_seen_, expected)


def test_scaler_stable_with_large_offset():
    rng = np.random.default_rng(1)
    X = 1e9 + rng.normal(0.0, 1.0, size=(1000, 1))
    s = feed(StandardScaler(), X, 50)
    assert abs(s.scale_[0] - X.std()) < 1e-6


def test_scaler_zero_variance_chunk_then_spread():
    s = StandardScaler()
    s.partial_fit(np.full((5, 1), 3.0))
    assert s.scale_[0] == 1.0 and s.mean_[0] == 3.0
    s.partial_fit(np.array([[1.0], [5.0]]))
    batch = StandardScaler().fit(
        np.vstack([np.full((5, 1), 3.0), [[1.0], [5.0]]]))
    np.testing.assert_allclose(s.scale_, batch.scale_)


def test_scaler_all_nan_column_in_first_chunk():
    s = StandardScaler()
    s.partial_fit(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert s.mean_[1] == 0.0 and s.scale_[1] == 1.0
    s.partial_fit(np.array([[5.0, 2.0], [7.0, 4.0]]))
    np.testing.assert_allclose(s.mean_, [4.0, 3.0])


def test_scaler_empty_chunk_changes_nothing(data):
    s = StandardScaler().partial_fit(data[:20])
    before = s.mean_.copy()
    assert s.partial_fit(data[:0]) is s
    np.testing.assert_array_equal(s.mean_, before)


def test_scaler_empty_first_chunk_leaves_it_unfitted():
    s = StandardScaler().partial_fit(np.empty((0, 2)))
    with pytest.raises(NotFittedError):
        s.transform(np.ones((1, 2)))


def test_scaler_wrong_width_rejected(data):
    s = StandardScaler().partial_fit(data)
    with pytest.raises(ValueError, match="Expected shape"):
        s.partial_fit(data[:, :2])


def test_fit_forgets_earlier_chunks(data):
    s = feed(StandardScaler(), data, 30)
    s.fit(data[:10])
    np.testing.assert_allclose(s.mean_, StandardScaler().fit(data[:10]).mean_)


def test_scaler_inverse_after_streaming(data):
    s = feed(StandardScaler(), data, 17)
    np.testing.assert_allclose(s.inverse_transform(s.transform(data)), data)


# MinMaxScaler
@pytest.mark.parametrize("size", [1, 13, 120])
def test_minmax_chunks_match_batch(data, size):
    batch = MinMaxScaler().fit(data)
    stream = feed(MinMaxScaler(), data, size)
    np.testing.assert_allclose(stream.data_min_, batch.data_min_)
    np.testing.assert_allclose(stream.data_max_, batch.data_max_)
    np.testing.assert_allclose(stream.transform(data), batch.transform(data))


def test_minmax_range_widens_with_new_extremes():
    s = MinMaxScaler().partial_fit(np.array([[0.0], [10.0]]))
    assert s.transform(np.array([[20.0]]))[0, 0] == 2.0
    s.partial_fit(np.array([[20.0]]))
    assert s.transform(np.array([[20.0]]))[0, 0] == 1.0


def test_minmax_nan_column_then_values():
    s = MinMaxScaler().partial_fit(np.array([[np.nan], [np.nan]]))
    assert s.data_min_[0] == 0.0 and s.data_max_[0] == 1.0
    s.partial_fit(np.array([[2.0], [6.0]]))
    assert s.data_min_[0] == 2.0 and s.data_max_[0] == 6.0


# SimpleImputer
@pytest.mark.parametrize("size", [1, 11, 120])
def test_imputer_mean_chunks_match_batch(data, size):
    batch = SimpleImputer("mean").fit(data)
    stream = feed(SimpleImputer("mean"), data, size)
    np.testing.assert_allclose(stream.statistics_, batch.statistics_)
    np.testing.assert_allclose(stream.transform(data), batch.transform(data))


def test_imputer_median_matches_batch_inside_window(data):
    batch = SimpleImputer("median").fit(data)
    stream = feed(SimpleImputer("median", window=1000), data, 10)
    np.testing.assert_allclose(stream.statistics_, batch.statistics_)


def test_imputer_median_uses_only_recent_rows():
    s = SimpleImputer("median", window=3)
    s.partial_fit(np.array([[100.0], [100.0], [100.0]]))
    s.partial_fit(np.array([[1.0], [2.0], [3.0]]))
    assert s.statistics_[0] == 2.0   # the old 100s fell out of the window


def test_imputer_estimate_updates_with_new_chunk():
    s = SimpleImputer("mean").partial_fit(np.array([[2.0], [4.0]]))
    assert s.transform(np.array([[np.nan]]))[0, 0] == 3.0
    s.partial_fit(np.array([[9.0]]))
    assert s.transform(np.array([[np.nan]]))[0, 0] == 5.0


def test_imputer_constant_and_all_nan_fallback():
    s = SimpleImputer("constant", fill_value=7.0)
    s.partial_fit(np.array([[1.0, np.nan]]))
    assert list(s.statistics_) == [7.0, 7.0]
    m = SimpleImputer("mean", fill_value=-1.0)
    m.partial_fit(np.array([[1.0, np.nan]]))
    assert list(m.statistics_) == [1.0, -1.0]


def test_imputer_does_not_modify_input(data):
    s = feed(SimpleImputer("mean"), data, 20)
    copy = data.copy()
    s.transform(data)
    np.testing.assert_array_equal(data, copy)


@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_imputer_bad_window(bad):
    with pytest.raises(ValueError, match="window"):
        SimpleImputer("median", window=bad)


# OneHotEncoder
def test_encoder_chunks_match_batch():
    X = np.array([["a", "x"], ["b", "y"], ["c", "x"], ["a", "z"]])
    batch = OneHotEncoder().fit(X)
    stream = OneHotEncoder()
    for i in range(0, 4, 1):
        stream.partial_fit(X[i:i + 1])
    for a, b in zip(stream.categories_, batch.categories_):
        np.testing.assert_array_equal(a, b)
    assert stream.n_output_features_ == batch.n_output_features_
    np.testing.assert_array_equal(stream.transform(X), batch.transform(X))


def test_encoder_grows_when_new_category_appears():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    assert enc.n_output_features_ == 2
    with pytest.raises(ValueError, match="Unseen"):
        enc.transform(np.array([["c"]]))
    enc.partial_fit(np.array([["c"]]))
    assert enc.n_output_features_ == 3
    np.testing.assert_array_equal(enc.transform(np.array([["c"]])),
                                  [[0.0, 0.0, 1.0]])


def test_encoder_column_positions_can_move():
    enc = OneHotEncoder().partial_fit(np.array([["b"], ["c"]]))
    assert enc.transform(np.array([["c"]])).tolist() == [[0.0, 1.0]]
    enc.partial_fit(np.array([["a"]]))      # sorts before the others
    assert enc.transform(np.array([["c"]])).tolist() == [[0.0, 0.0, 1.0]]


def test_encoder_numeric_categories():
    enc = OneHotEncoder().partial_fit(np.array([[1], [2]]))
    enc.partial_fit(np.array([[3], [1]]))
    assert enc.n_output_features_ == 3


def test_encoder_longer_strings_in_later_chunk():
    enc = OneHotEncoder().partial_fit(np.array([["ab"]]))
    enc.partial_fit(np.array([["abcdef"]]))
    assert enc.transform(np.array([["abcdef"]])).sum() == 1.0


def test_encoder_ignore_mode_gives_zero_block():
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.partial_fit(np.array([["a"], ["b"]]))
    assert enc.transform(np.array([["z"]])).sum() == 0.0


def test_encoder_empty_chunk_and_unfitted_start():
    enc = OneHotEncoder().partial_fit(np.empty((0, 1), dtype=str))
    with pytest.raises(NotFittedError):
        enc.transform(np.array([["a"]]))
    enc.partial_fit(np.array([["a"]]))
    assert enc.n_output_features_ == 1


def test_encoder_rejects_nan_wrong_width_and_type_mix():
    enc = OneHotEncoder().partial_fit(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="NaN"):
        enc.partial_fit(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError, match="Expected shape"):
        enc.partial_fit(np.array([[1.0]]))
    with pytest.raises(ValueError, match="numeric"):
        enc.partial_fit(np.array([["a", "b"]]))


# Additional numerical and compatibility tests
def test_scaler_matches_original_numcompute(data):
    from numcompute.preprocessing import StandardScaler as Original

    original = Original().fit(data)
    streaming = feed(StandardScaler(), data, 17)
    np.testing.assert_allclose(streaming.mean_, original.mean_, rtol=1e-12)
    np.testing.assert_allclose(streaming.scale_, original.scale_, rtol=1e-12)


def test_minmax_matches_original_numcompute(data):
    from numcompute.preprocessing import MinMaxScaler as Original

    original = Original(feature_range=(-1, 1)).fit(data)
    streaming = feed(MinMaxScaler(feature_range=(-1, 1)), data, 11)
    np.testing.assert_allclose(streaming.transform(data),
                               original.transform(data))


def test_imputer_matches_original_numcompute(data):
    from numcompute.preprocessing import SimpleImputer as Original

    original = Original(strategy="mean").fit(data)
    streaming = feed(SimpleImputer(strategy="mean"), data, 9)
    np.testing.assert_allclose(streaming.statistics_, original.statistics_)


def test_onehot_matches_original_numcompute():
    from numcompute.preprocessing import OneHotEncoder as Original

    X = np.array([["red", "low"], ["blue", "high"], ["green", "low"]])
    original = Original().fit(X)
    streaming = feed(OneHotEncoder(), X, 1)
    np.testing.assert_array_equal(streaming.transform(X),
                                  original.transform(X))


def test_imputer_large_equal_values_do_not_overflow():
    s = SimpleImputer("mean")
    s.partial_fit(np.array([[1e308], [1e308]]))
    s.partial_fit(np.array([[1e308]]))
    assert s.statistics_[0] == 1e308
    assert np.isfinite(s.statistics_[0])


def test_imputer_large_opposite_sign_means_do_not_overflow():
    s = SimpleImputer("mean")
    s.partial_fit(np.array([[-1e308]]))
    s.partial_fit(np.array([[1e308]]))
    np.testing.assert_allclose(s.statistics_, [0.0])


def test_imputer_median_large_values_do_not_overflow():
    s = SimpleImputer("median", window=20)
    s.partial_fit(np.array([[1e308], [1e308]]))
    np.testing.assert_array_equal(s.statistics_, [1e308])
    s.fit(np.array([[1e308], [1.5e308]]))
    np.testing.assert_array_equal(s.statistics_, [1.25e308])


def test_minmax_large_opposite_sign_range():
    s = MinMaxScaler().partial_fit([[-1e308], [1e308]])
    X = np.array([[-1e308], [0.0], [1e308]])
    Z = s.transform(X)
    np.testing.assert_allclose(Z[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(s.inverse_transform(Z), X)


def test_zero_feature_inputs_rejected():
    X = np.empty((2, 0))
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer):
        with pytest.raises(ValueError, match="feature"):
            cls().fit(X)
        with pytest.raises(ValueError, match="feature"):
            cls().partial_fit(X)


def test_inf_features_rejected():
    X = np.array([[1.0], [np.inf]])
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer):
        with pytest.raises(ValueError, match="infinity"):
            cls().partial_fit(X)


def test_noncontiguous_chunks_match_contiguous():
    rng = np.random.default_rng(50)
    X = rng.normal(size=(80, 4))
    X[::11, 2] = np.nan
    view = np.asfortranarray(X)[::2]
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer):
        a = feed(cls(), view, 7)
        b = feed(cls(), view.copy(), 7)
        np.testing.assert_allclose(a.transform(view), b.transform(view))


def test_fit_transform_still_works():
    X = np.array([[2.0], [4.0], [6.0]])
    s = StandardScaler()
    np.testing.assert_allclose(
        s.fit_transform(X), [[-1.22474487139], [0], [1.22474487139]]
    )


def test_encoder_new_category_changes_feature_schema():
    encoder = OneHotEncoder().partial_fit([["b"], ["c"]])
    before = encoder.n_output_features_
    encoder.partial_fit([["a"]])
    assert encoder.n_output_features_ == before + 1
    np.testing.assert_array_equal(encoder.transform([["c"]]), [[0., 0., 1.]])


def test_imputer_mean_does_not_retain_raw_rows():
    s = SimpleImputer("mean")
    s.partial_fit(np.ones((100, 2)))
    assert s._rows.shape == (0, 2)
