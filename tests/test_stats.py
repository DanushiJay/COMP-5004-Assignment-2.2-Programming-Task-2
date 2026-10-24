"""Tests for numcompute.stats. Expected values are worked out by hand."""

import warnings

import numpy as np
import pytest

from numcompute import stats
from numcompute.stats import RunningStats

NAN = np.nan

# 2 samples x 3 features.
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])

# Column 1 is all NaN; row 1 has a single valid value.
X_NAN = np.array([[1.0, NAN, 3.0], [5.0, NAN, NAN]])


def test_reducers_axis_semantics_on_2d_input():
    assert stats.mean(X) == 4.0  # 24 / 6
    assert stats.mean(X, axis=0).tolist() == [2.5, 3.5, 6.0]  # shape (n_features,)
    assert stats.mean(X, axis=1).tolist() == [2.0, 6.0]  # shape (n_samples,)
    assert stats.median(X, axis=1).tolist() == [2.0, 5.0]
    assert stats.std(X, axis=0).tolist() == [1.5, 1.5, 3.0]
    assert stats.var(X, axis=0, ddof=1).tolist() == [4.5, 4.5, 18.0]
    assert stats.min(X, axis=0).tolist() == [1.0, 2.0, 3.0]
    assert stats.max(X, axis=1).tolist() == [3.0, 9.0]


def test_reducers_on_1d_input_return_scalars():
    data = [1, 2, 3, 4]
    assert stats.mean(data) == 2.5
    assert stats.median(data) == 2.5
    assert stats.var(data) == 1.25  # squared deviations sum to 5, over 4
    assert stats.std(data) == pytest.approx(1.25**0.5)
    assert stats.min(data) == 1.0
    assert stats.max(data) == 4.0


def test_nan_policies():
    # propagate (default): any NaN in a slice makes that result NaN
    np.testing.assert_array_equal(stats.mean(X_NAN, axis=0), [3.0, NAN, NAN])
    # omit: average the values that are present
    assert stats.mean(X_NAN, axis=1, nan_policy="omit").tolist() == [2.0, 5.0]
    assert stats.max(X_NAN, nan_policy="omit") == 5.0
    # raise
    with pytest.raises(ValueError, match="NaN"):
        stats.mean(X_NAN, nan_policy="raise")


def test_all_nan_column_and_row_give_nan_without_warning():
    all_nan_row = np.array([[NAN, NAN], [1.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_column = stats.mean(X_NAN, axis=0, nan_policy="omit")
        by_row = stats.median(all_nan_row, axis=1, nan_policy="omit")
    np.testing.assert_array_equal(by_column, [3.0, NAN, 3.0])
    np.testing.assert_array_equal(by_row, [NAN, 2.0])


def test_reducers_on_non_contiguous_input():
    transposed = X.T  # shape (3, 2)
    assert not transposed.flags.c_contiguous
    assert stats.mean(transposed, axis=0).tolist() == [2.0, 6.0]

    strided = np.array([[1.0, 2.0], [100.0, 100.0], [3.0, 6.0], [100.0, 100.0]])[::2]
    assert not strided.flags.c_contiguous
    assert stats.mean(strided, axis=0).tolist() == [2.0, 4.0]
    assert stats.std(strided, axis=0).tolist() == [1.0, 2.0]


def test_reducers_reject_empty_and_non_numeric_input():
    for reducer in (stats.mean, stats.median, stats.std, stats.min, stats.max):
        with pytest.raises(ValueError, match="non-empty"):
            reducer([])
        with pytest.raises(TypeError, match="numeric"):
            reducer(["a", "b"])


def test_quantile_values_and_shapes():
    assert stats.quantile([1, 2, 3, 4], 0.25) == 1.75  # position 0.75
    assert stats.quantile([1, 2, 3, 4], [0.0, 0.5, 1.0]).tolist() == [1.0, 2.5, 4.0]
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])  # (3, 2)
    assert stats.quantile(data, 0.5, axis=0).tolist() == [2.0, 20.0]
    assert stats.quantile(data, [0.0, 1.0], axis=0).shape == (2, 2)  # (n_q, n_features)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stats.quantile([1, 2, 3, 4], 50)  # q is a fraction, not a percentage


def test_quantile_nan_handling():
    data = np.array([[1.0, NAN], [3.0, NAN], [NAN, NAN]])
    assert np.isnan(stats.quantile(data, 0.5))
    result = stats.quantile(data, 0.5, axis=0, nan_policy="omit")
    np.testing.assert_array_equal(result, [2.0, NAN])


def test_histogram_counts_edges_and_closed_last_bin():
    counts, edges = stats.histogram([0, 1, 1, 2, 4], bins=4)
    assert edges.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]  # shape (n_bins + 1,)
    assert counts.tolist() == [1, 2, 1, 1]  # shape (n_bins,)
    # 5 opens the second bin; 10 is counted because the last bin is closed.
    counts, edges = stats.histogram([0, 5, 10], bins=2)
    assert edges.tolist() == [0.0, 5.0, 10.0]
    assert counts.tolist() == [1, 2]


def test_histogram_density_and_range():
    # 5 values in bins of width 1: density is count / (5 * 1).
    density, _ = stats.histogram([0, 1, 1, 2, 4], bins=4, density=True)
    assert density.tolist() == pytest.approx([0.2, 0.4, 0.2, 0.2])
    # Values outside the given range (5 and 10) are not counted.
    counts, edges = stats.histogram([0, 1, 5, 10], bins=2, range=(0, 4))
    assert edges.tolist() == [0.0, 2.0, 4.0]
    assert counts.tolist() == [2, 0]


def test_histogram_nan_and_empty_input():
    counts, edges = stats.histogram([0.0, NAN, 4.0, NAN], bins=2)
    assert edges.tolist() == [0.0, 2.0, 4.0]
    assert counts.tolist() == [1, 1]  # NaNs are dropped
    counts, _ = stats.histogram([], bins=3)
    assert counts.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="NaN"):
        stats.histogram([1.0, NAN], nan_policy="raise")


def test_running_stats_hand_computed_example():
    rs = RunningStats().update([1.0, 2.0]).update([3.0, 4.0])
    assert rs.count == 4
    assert rs.mean == 2.5
    assert rs.variance() == 1.25  # squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5
    assert rs.variance(ddof=1) == pytest.approx(5 / 3)
    assert rs.std() == pytest.approx(1.25**0.5)


def test_running_stats_matches_batch_var():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(1000, 4)) * [1.0, 10.0, 100.0, 0.1] + [0.0, 1e3, -50.0, 7.0]
    rs = RunningStats()
    for chunk in np.array_split(data, [1, 4, 300, 301, 777]):  # uneven chunks
        rs.update(chunk)
    assert rs.count == 1000
    assert rs.mean.shape == (4,)  # (n_features,)
    np.testing.assert_allclose(rs.mean, np.mean(data, axis=0), rtol=1e-12)
    np.testing.assert_allclose(rs.variance(), np.var(data, axis=0), rtol=1e-11)

    # One sample at a time is the classic Welford recurrence.
    single = RunningStats()
    for value in data[:, 0]:
        single.update(value)
    assert single.variance() == pytest.approx(np.var(data[:, 0]), rel=1e-11)


def test_running_stats_rejects_invalid_use():
    with pytest.raises(ValueError, match="No samples"):
        RunningStats().variance()
    rs = RunningStats().update(np.ones((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        rs.update(np.ones((2, 4)))  # feature count changed
    with pytest.raises(ValueError, match="NaN"):
        rs.update([[1.0, NAN, 2.0]])
