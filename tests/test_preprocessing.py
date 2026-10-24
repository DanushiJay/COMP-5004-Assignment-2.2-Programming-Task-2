import numpy as np
import pytest

from numcompute.preprocessing import (
    MinMaxScaler, NotFittedError, OneHotEncoder, SimpleImputer, StandardScaler,
)


def test_standard_scaler_zero_mean_unit_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0)


def test_standard_scaler_zero_variance_column_maps_to_zero():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))
    np.testing.assert_array_equal(Z[:, 0], 0.0)


def test_standard_scaler_nan_ignored_in_fit_and_kept_in_output():
    X = np.array([[1.0], [np.nan], [3.0]])
    sc = StandardScaler().fit(X)
    assert sc.mean_[0] == 2.0
    Z = sc.transform(X)
    assert np.isnan(Z[1, 0]) and Z[0, 0] == -1.0


def test_standard_scaler_all_nan_column_stays_nan_without_warning():
    X = np.array([[1.0, np.nan], [2.0, np.nan]])
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isnan(Z[:, 1])) and np.all(np.isfinite(Z[:, 0]))


def test_standard_scaler_inverse_roundtrip():
    X = np.random.default_rng(0).normal(size=(20, 3))
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X)


def test_scalers_accept_non_contiguous_input():
    X = np.arange(24.0).reshape(4, 6)[:, ::2]
    assert not X.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(
        StandardScaler().fit_transform(X),
        StandardScaler().fit_transform(np.ascontiguousarray(X)))
    assert MinMaxScaler().fit_transform(X).shape == (4, 3)


def test_transform_before_fit_raises_not_fitted():
    transformers = (StandardScaler(), MinMaxScaler(), SimpleImputer(),
                    OneHotEncoder())
    for obj in transformers:
        with pytest.raises(NotFittedError):
            obj.transform(np.zeros((2, 2)))


def test_wrong_feature_count_raises_with_standard_message():
    sc = StandardScaler().fit(np.zeros((3, 2)))
    message = r"Expected shape \(n, 2\), got \(3, 3\)"
    with pytest.raises(ValueError, match=message):
        sc.transform(np.zeros((3, 3)))


def test_bad_input_types_and_shapes():
    with pytest.raises(ValueError, match="2-D"):
        StandardScaler().fit([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        StandardScaler().fit([["a", "b"]])
    with pytest.raises(TypeError):
        StandardScaler().fit([["1", "2"], ["3", "4"]])
    with pytest.raises(ValueError, match="0 samples"):
        StandardScaler().fit(np.empty((0, 3)))


def test_minmax_default_and_custom_range():
    X = np.array([[0.0], [5.0], [10.0]])
    out = MinMaxScaler().fit_transform(X).ravel()
    np.testing.assert_allclose(out, [0, 0.5, 1])
    out = MinMaxScaler((-1, 1)).fit_transform(X).ravel()
    np.testing.assert_allclose(out, [-1, 0, 1])


def test_minmax_constant_column_maps_to_low_and_roundtrips():
    X = np.array([[7.0, 1.0], [7.0, 3.0]])
    sc = MinMaxScaler((2, 4)).fit(X)
    Z = sc.transform(X)
    np.testing.assert_array_equal(Z[:, 0], 2.0)
    np.testing.assert_allclose(sc.inverse_transform(Z), X)


def test_minmax_invalid_range_raises():
    for bad in ((1, 1), (2, 1), 5, (1, 2, 3)):
        with pytest.raises(ValueError):
            MinMaxScaler(bad)


def test_imputer_mean_median_constant():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 8.0], [100.0, np.nan]])
    mean = SimpleImputer("mean").fit(X).statistics_
    median = SimpleImputer("median").fit(X).statistics_
    np.testing.assert_allclose(mean, [104 / 3, 6.0])
    np.testing.assert_allclose(median, [3.0, 6.0])
    out = SimpleImputer("constant", fill_value=-1).fit_transform(X)
    assert out[0, 1] == -1 and out[1, 0] == -1 and not np.isnan(out).any()


def test_imputer_all_nan_column_uses_fill_value_and_does_not_mutate_input():
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    before = X.copy()
    out = SimpleImputer("mean", fill_value=9.0).fit_transform(X)
    np.testing.assert_array_equal(out[:, 0], 9.0)
    np.testing.assert_array_equal(X, before)


def test_imputer_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer("mode")


def test_onehot_basic_and_sorted_categories():
    X = np.array(["b", "a", "c", "a"])
    enc = OneHotEncoder().fit(X)
    np.testing.assert_array_equal(enc.categories_[0], ["a", "b", "c"])
    np.testing.assert_array_equal(
        enc.transform(X), [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])


def test_onehot_multi_feature_blocks_and_row_sums():
    X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
    out = OneHotEncoder().fit_transform(X)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out.sum(axis=1), 2.0)


def test_onehot_column_blocks_stay_in_order_with_twelve_features():
    # Ten or more columns: each column keeps its own block, in order.
    X = np.vstack([np.full(12, "a"), np.full(12, "b")])
    enc = OneHotEncoder().fit(X)
    assert len(enc.categories_) == 12 and enc.n_output_features_ == 24
    out = enc.transform(X)
    np.testing.assert_array_equal(out[0], np.tile([1.0, 0.0], 12))
    np.testing.assert_array_equal(out[1], np.tile([0.0, 1.0], 12))


def test_onehot_single_category():
    out = OneHotEncoder().fit_transform(np.array(["only", "only"]))
    np.testing.assert_array_equal(out, [[1.0], [1.0]])


def test_onehot_unseen_category_error_and_ignore():
    enc = OneHotEncoder().fit(np.array(["a", "b"]))
    with pytest.raises(ValueError, match="Unseen categories"):
        enc.transform(np.array(["c"]))
    ignoring = OneHotEncoder(handle_unknown="ignore").fit(["a", "b"])
    out = ignoring.transform(["c", "a"])
    np.testing.assert_array_equal(out, [[0, 0], [1, 0]])


def test_onehot_unseen_value_sorting_after_all_categories():
    enc = OneHotEncoder().fit(np.array(["a", "b"]))
    with pytest.raises(ValueError):
        enc.transform(np.array(["zzz"]))


def test_onehot_wrong_feature_count_and_numeric_categories():
    enc = OneHotEncoder().fit(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="Expected shape"):
        enc.transform(np.array([[1, 2, 3]]))
    assert enc.transform(np.array([[1, 4]])).sum() == 2.0


def test_all_nan_column_gets_neutral_statistics():
    X = np.array([[1.0, np.nan], [2.0, np.nan]])
    sc = StandardScaler().fit(X)
    assert sc.mean_[1] == 0.0 and sc.scale_[1] == 1.0
    mm = MinMaxScaler().fit(X)
    assert mm.data_min_[1] == 0.0 and mm.data_max_[1] == 1.0
    assert np.isnan(mm.transform(X)[:, 1]).all()


def test_transform_accepts_zero_rows_but_fit_does_not():
    empty = np.empty((0, 2))
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert StandardScaler().fit(train).transform(empty).shape == (0, 2)
    assert MinMaxScaler().fit(train).transform(empty).shape == (0, 2)
    assert SimpleImputer().fit(train).transform(empty).shape == (0, 2)
    enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
    assert enc.transform(np.empty((0, 1), dtype=str)).shape == (0, 2)
    with pytest.raises(ValueError, match="0 samples"):
        OneHotEncoder().fit(np.empty((0, 2), dtype=str))


def test_not_fitted_error_is_a_value_error():
    with pytest.raises(ValueError):
        StandardScaler().transform(np.zeros((1, 1)))


def test_onehot_numeric_categories_sorted_numerically_and_keep_dtype():
    enc = OneHotEncoder().fit(np.array([[10], [2], [2], [1]]))
    assert enc.categories_[0].tolist() == [1, 2, 10]
    assert enc.categories_[0].dtype.kind == "i"
    np.testing.assert_array_equal(enc.transform(np.array([[2], [10]])),
                                  [[0, 1, 0], [0, 0, 1]])


def test_onehot_int_and_float_categories_match_by_value():
    enc = OneHotEncoder().fit(np.array([[1], [2]]))
    np.testing.assert_array_equal(enc.transform(np.array([[1.0]])), [[1, 0]])


def test_onehot_nan_category_rejected_at_fit_and_unseen_at_transform():
    with pytest.raises(ValueError, match="NaN is not a valid category"):
        OneHotEncoder().fit(np.array([[1.0], [np.nan]]))
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array([[1.0], [2.0]]))
    out = enc.transform(np.array([[np.nan]]))
    np.testing.assert_array_equal(out, [[0, 0]])


def test_onehot_unseen_message_names_column_and_caps_values():
    enc = OneHotEncoder().fit(np.array([["a", "x"], ["b", "y"]]))
    with pytest.raises(ValueError, match=r"column\(s\) \[1\]: \['z'\]"):
        enc.transform(np.array([["a", "z"]]))
    many = np.array([[str(i)] for i in range(8)])
    with pytest.raises(ValueError, match=r"\(\+3 more\)"):
        enc.fit(np.array([["a"], ["b"]])).transform(many)


def test_onehot_value_known_in_another_column_is_unseen_here():
    enc = OneHotEncoder().fit(np.array([["a", "x"], ["b", "y"]]))
    with pytest.raises(ValueError, match="Unseen"):
        enc.transform(np.array([["x", "a"]]))


def test_onehot_text_numeric_mismatch_raises():
    enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="fitted on text"):
        enc.transform(np.array([[1]]))
    with pytest.raises(ValueError, match="fitted on numeric"):
        OneHotEncoder().fit(np.array([[1], [2]])).transform(np.array([["a"]]))


def test_onehot_object_array_and_zero_features():
    X = np.array([["a", 1], ["b", 2]], dtype=object)
    assert OneHotEncoder().fit_transform(X).shape == (2, 4)
    with pytest.raises(ValueError, match="0 features"):
        OneHotEncoder().fit(np.empty((3, 0)))


def test_imputer_and_onehot_accept_non_contiguous_input():
    X = np.array([[1.0, 9.0, np.nan, 9.0],
                  [np.nan, 9.0, 4.0, 9.0]])[:, ::2]   # [[1, nan], [nan, 4]]
    assert not X.flags["C_CONTIGUOUS"]
    out = SimpleImputer("mean").fit_transform(X)
    np.testing.assert_array_equal(out, [[1.0, 4.0], [1.0, 4.0]])
    C = np.array([["a", "x", "b", "y"],
                  ["b", "x", "a", "y"]])[:, ::2]      # selects a,b / b,a
    assert not C.flags["C_CONTIGUOUS"]
    encoded = OneHotEncoder().fit_transform(C)
    same = OneHotEncoder().fit_transform(np.ascontiguousarray(C))
    np.testing.assert_array_equal(encoded, same)
    np.testing.assert_array_equal(encoded, [[1, 0, 0, 1], [0, 1, 1, 0]])
