"""Tests for numcompute_stream.stats (StreamStats, StreamingHistogram)."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingHistogram, StreamStats


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(120, 3))
    X[::9, 1] = np.nan
    return X


def feed(model, X, size):
    for i in range(0, len(X), size):
        model.update_stats(X[i:i + size])
    return model


# StreamStats
@pytest.mark.parametrize("size", [1, 7, 40, 120, 500])
def test_moments_match_numpy(data, size):
    s = feed(StreamStats(), data, size)
    np.testing.assert_allclose(s.mean, np.nanmean(data, axis=0))
    np.testing.assert_allclose(s.variance(), np.nanvar(data, axis=0))
    np.testing.assert_allclose(s.std(ddof=1), np.nanstd(data, axis=0,
                                                        ddof=1))
    np.testing.assert_allclose(s.minimum, np.nanmin(data, axis=0))
    np.testing.assert_allclose(s.maximum, np.nanmax(data, axis=0))


def test_counts_ignore_nan(data):
    s = feed(StreamStats(), data, 25)
    np.testing.assert_array_equal(s.count, (~np.isnan(data)).sum(axis=0))
    assert s.n_samples_seen_ == 120


def test_quantile_exact_inside_window(data):
    s = feed(StreamStats(window=1000), data, 10)
    np.testing.assert_allclose(s.quantile(0.3),
                               np.nanquantile(data, 0.3, axis=0))
    out = s.quantile([0.1, 0.9])
    assert out.shape == (2, 3)


def test_quantile_uses_only_recent_rows():
    s = StreamStats(window=3)
    s.update_stats(np.array([[100.0], [100.0], [100.0]]))
    s.update_stats(np.array([[1.0], [2.0], [3.0]]))
    assert s.quantile(0.5)[0] == 2.0
    assert s.mean[0] == 51.0          # the mean still covers every row


def test_stable_with_large_offset():
    rng = np.random.default_rng(1)
    X = 1e9 + rng.normal(0.0, 1.0, size=(1000, 1))
    s = feed(StreamStats(), X, 50)
    assert abs(s.std()[0] - X.std()) < 1e-6


def test_one_dimensional_chunk_is_one_feature():
    s = StreamStats().update_stats([1.0, 2.0, 3.0])
    assert s.n_features_in_ == 1 and s.mean[0] == 2.0


def test_all_nan_column_gives_nan_until_values_arrive():
    s = StreamStats().update_stats(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert np.isnan(s.mean[1]) and np.isnan(s.variance()[1])
    assert np.isnan(s.minimum[1])
    s.update_stats(np.array([[5.0, 2.0], [7.0, 4.0]]))
    np.testing.assert_allclose(s.mean, [4.0, 3.0])


def test_variance_is_nan_when_count_not_above_ddof():
    s = StreamStats().update_stats([[5.0]])
    assert s.variance(ddof=0)[0] == 0.0
    assert np.isnan(s.variance(ddof=1)[0])


def test_zero_variance_chunk_then_spread():
    s = StreamStats().update_stats(np.full((5, 1), 3.0))
    assert s.std()[0] == 0.0
    s.update_stats(np.array([[1.0], [5.0]]))
    full = np.vstack([np.full((5, 1), 3.0), [[1.0], [5.0]]])
    np.testing.assert_allclose(s.variance(), full.var(axis=0))


def test_empty_chunk_changes_nothing(data):
    s = StreamStats().update_stats(data[:20])
    before = s.mean
    assert s.update_stats(data[:0]) is s
    np.testing.assert_array_equal(s.mean, before)


def test_results_before_data_raise():
    with pytest.raises(ValueError, match="No samples"):
        StreamStats().mean
    with pytest.raises(ValueError, match="No samples"):
        StreamStats().quantile(0.5)


def test_wrong_width_inf_and_text_rejected(data):
    s = StreamStats().update_stats(data)
    with pytest.raises(ValueError, match="features"):
        s.update_stats(data[:, :2])
    with pytest.raises(ValueError, match="infinity"):
        StreamStats().update_stats([[1.0], [np.inf]])
    with pytest.raises(TypeError, match="numeric"):
        StreamStats().update_stats([["a"], ["b"]])
    with pytest.raises(ValueError, match="1-D or 2-D"):
        StreamStats().update_stats(np.ones((2, 2, 2)))


def test_nan_policy_raise():
    with pytest.raises(ValueError, match="NaN"):
        StreamStats(nan_policy="raise").update_stats([[np.nan]])
    with pytest.raises(ValueError, match="nan_policy"):
        StreamStats(nan_policy="propagate")


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_bad_window(bad):
    with pytest.raises(ValueError, match="window"):
        StreamStats(window=bad)


def test_reset_forgets_everything(data):
    s = feed(StreamStats(), data, 30)
    assert s.reset() is s
    with pytest.raises(ValueError, match="No samples"):
        s.mean
    s.update_stats(data[:10, :2])       # a new width is fine after reset
    assert s.n_features_in_ == 2


def test_update_alias_and_summary(data):
    s = StreamStats()
    assert s.update(data) is s
    out = s.summary()
    assert set(out) == {"n_samples", "count", "mean", "std", "min", "max"}


def test_non_contiguous_chunks_match(data):
    view = np.asfortranarray(data)[::2]
    a = feed(StreamStats(), view, 7)
    b = feed(StreamStats(), view.copy(), 7)
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.std(), b.std())


# StreamingHistogram
def test_histogram_matches_batch_counts(data):
    x = data[:, 0]
    h = StreamingHistogram(bins=8, range=(0.0, 10.0))
    for i in range(0, 120, 13):
        h.update(x[i:i + 13])
    expected, _ = np.histogram(x, bins=8, range=(0.0, 10.0))
    np.testing.assert_array_equal(h.counts, expected)
    assert h.outside_ == x.size - expected.sum()


def test_histogram_range_taken_from_first_chunk():
    h = StreamingHistogram(bins=4)
    h.update([0.0, 4.0])
    np.testing.assert_allclose(h.edges, [0, 1, 2, 3, 4])
    h.update([10.0, 2.0])
    assert h.outside_ == 1 and h.counts.sum() == 3


def test_histogram_constant_first_chunk():
    h = StreamingHistogram(bins=2).update([3.0, 3.0])
    assert h.counts.sum() == 2


def test_histogram_sliding_window_drops_old_chunks():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0), window=2)
    h.update([0.5, 0.5])      # chunk 1: bin 0
    h.update([1.5])           # chunk 2: bin 1
    h.update([1.5, 1.5])      # chunk 3: bin 1, chunk 1 falls out
    np.testing.assert_array_equal(h.counts, [0, 3])


def test_histogram_ignores_nan_and_empty_chunks():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0))
    h.update([np.nan, 0.5])
    h.update([])
    h.update([np.nan])
    assert h.counts.sum() == 1


def test_histogram_density_integrates_to_one(data):
    h = StreamingHistogram(bins=10, range=(-5.0, 15.0))
    h.update(data[:, 0])
    assert abs((h.density * np.diff(h.edges)).sum() - 1.0) < 1e-12
    assert StreamingHistogram(bins=3).density.sum() == 0.0


def test_histogram_reset_clears_counts():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0)).update([0.5, 1.5])
    h.reset()
    assert h.counts.sum() == 0 and h.outside_ == 0


@pytest.mark.parametrize("kw", [{"bins": 0}, {"range": (1.0, 1.0)},
                                {"range": (0.0, np.inf)},
                                {"window": 0}])
def test_histogram_bad_arguments(kw):
    with pytest.raises(ValueError):
        StreamingHistogram(**kw)


def test_histogram_rejects_inf_and_text():
    with pytest.raises(ValueError, match="infinity"):
        StreamingHistogram().update([np.inf])
    with pytest.raises(TypeError, match="numeric"):
        StreamingHistogram().update(["a"])


def test_mean_of_huge_opposite_values_does_not_overflow():
    s = StreamStats()
    s.update_stats([[-1e308], [-1e308]])
    s.update_stats([[1e308], [1e308]])
    assert s.mean[0] == 0.0


def test_ddof_must_be_a_real_int(data):
    s = StreamStats().update_stats(data)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="ddof"):
            s.variance(ddof=bad)


def test_histogram_reset_forgets_estimated_edges():
    h = StreamingHistogram(bins=2).update([0.0, 10.0])
    h.reset()
    assert h.edges is None
    h.update([100.0, 200.0])
    assert h.edges[0] == 100.0


def test_sample_variance_matches_numpy(data):
    s = feed(StreamStats(), data, 13)
    np.testing.assert_allclose(s.variance(ddof=1),
                               np.nanvar(data, axis=0, ddof=1))


def test_quantile_list_of_q_matches_numpy(data):
    s = feed(StreamStats(window=200), data, 19)
    q = [0.25, 0.5, 0.75]
    np.testing.assert_allclose(s.quantile(q),
                               np.nanquantile(data, q, axis=0))


def test_empty_first_chunk_has_no_effect():
    s = StreamStats().update_stats(np.empty((0, 2)))
    assert s.n_samples_seen_ == 0
    with pytest.raises(ValueError, match="No samples"):
        _ = s.mean


def test_histogram_fixed_range_edges_and_outside_count():
    x = np.array([0.1, 0.3, 0.9, 1.0, 1.1, -1.0])
    h = StreamingHistogram(bins=5, range=(0, 1))
    h.update(x[:2]).update(x[2:])
    counts, edges = np.histogram(x, bins=5, range=(0, 1))
    np.testing.assert_array_equal(h.counts, counts)
    np.testing.assert_array_equal(h.edges, edges)
    assert h.outside_ == 2


def test_histogram_reset_with_explicit_range_keeps_edges():
    h = StreamingHistogram(bins=3, range=(0, 3)).update([0.5, 1.5])
    edges = h.edges.copy()
    h.reset()
    np.testing.assert_array_equal(h.edges, edges)
    assert h.counts.sum() == 0
