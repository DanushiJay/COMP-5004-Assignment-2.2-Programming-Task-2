import numpy as np
import pytest

from numcompute.utils import (
    as_float_array, batch_iter, euclidean_distance, log_softmax, logsumexp,
    manhattan_distance, pairwise_diff, relu, sigmoid, softmax,
)


def test_logsumexp_extreme_values_do_not_overflow():
    assert np.isclose(logsumexp([1000.0, 1000.0]), 1000.0 + np.log(2))
    assert np.isclose(logsumexp([-1000.0, -1000.0]), -1000.0 + np.log(2))


def test_logsumexp_matches_naive_formula_on_small_values():
    x = np.array([0.1, 0.5, -0.3])
    assert np.isclose(logsumexp(x), np.log(np.sum(np.exp(x))))


def test_logsumexp_all_neg_inf_is_neg_inf_not_nan():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf


def test_logsumexp_axis_and_keepdims_shapes():
    x = np.arange(6.0).reshape(2, 3)
    assert logsumexp(x, axis=0).shape == (3,)
    assert logsumexp(x, axis=1).shape == (2,)
    assert logsumexp(x, axis=1, keepdims=True).shape == (2, 1)
    assert np.ndim(logsumexp(x)) == 0


def test_logsumexp_empty_is_neg_inf_and_respects_keepdims():
    assert logsumexp(np.array([])) == -np.inf
    out = logsumexp(np.empty((0, 3)), axis=0, keepdims=True)
    assert out.shape == (1, 3) and np.all(out == -np.inf)


def test_logsumexp_rejects_strings():
    with pytest.raises(TypeError):
        logsumexp(["a", "b"])


def test_as_float_array_rejects_numeric_strings_and_none():
    with pytest.raises(TypeError, match="numeric"):
        as_float_array(["1", "2"])
    with pytest.raises(TypeError):
        as_float_array([1, None])
    out = as_float_array([True, 2, 3.5])
    assert out.dtype == float and out.tolist() == [1.0, 2.0, 3.5]


def test_softmax_rows_sum_to_one_with_extreme_inputs():
    x = np.array([[1e4, 0.0, -1e4], [1e4, 1e4, 1e4]])
    s = softmax(x, axis=1)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s.sum(axis=1), 1.0)
    np.testing.assert_allclose(s[1], 1 / 3)


def test_softmax_shift_invariant():
    x = np.array([0.1, 2.0, -1.0])
    np.testing.assert_allclose(softmax(x), softmax(x + 500.0))


def test_softmax_axis_zero_sums_columns():
    x = np.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(softmax(x, axis=0).sum(axis=0), 1.0)


def test_log_softmax_is_log_of_softmax_and_stable():
    x = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(log_softmax(x), np.log(softmax(x)))
    big = log_softmax(np.array([1e4, 0.0]))
    assert np.all(np.isfinite(big[:1])) and big[0] == 0.0


def test_sigmoid_stable_and_symmetric():
    x = np.array([-1e4, -1.0, 0.0, 1.0, 1e4])
    s = sigmoid(x)
    assert np.all(np.isfinite(s))
    assert s[0] == 0.0 and s[-1] == 1.0 and s[2] == 0.5
    np.testing.assert_allclose(s + sigmoid(-x), 1.0)


def test_relu_propagates_nan():
    out = relu([-1.0, 2.0, np.nan])
    assert out[0] == 0.0 and out[1] == 2.0 and np.isnan(out[2])


def test_pairwise_diff_shape_and_values():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0]])
    d = pairwise_diff(a, b)
    assert d.shape == (2, 1, 2)
    np.testing.assert_array_equal(d[:, 0, :], a)


def test_euclidean_distance_known_values_and_shape():
    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0], [6.0, 8.0], [3.0, 4.0]])
    D = euclidean_distance(A, B)
    assert D.shape == (2, 3)
    np.testing.assert_allclose(D, [[0, 10, 5], [5, 5, 0]])


def test_manhattan_distance_and_1d_inputs():
    D = manhattan_distance([1.0, 2.0], [[0.0, 0.0], [1.0, 5.0]])
    np.testing.assert_allclose(D, [[3.0, 3.0]])


def test_distance_feature_mismatch_and_bad_ndim_raise():
    with pytest.raises(ValueError, match="Feature dimensions differ"):
        euclidean_distance(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        euclidean_distance(np.zeros((2, 2, 2)), np.zeros((2, 2)))


def test_batch_iter_ragged_last_batch_and_views():
    X = np.arange(10).reshape(5, 2)
    batches = list(batch_iter(X, 2))
    assert [b.shape[0] for b in batches] == [2, 2, 1]
    assert np.shares_memory(batches[0], X)
    np.testing.assert_array_equal(np.vstack(batches), X)


def test_batch_iter_size_larger_than_n_and_empty_and_invalid():
    X = np.arange(6).reshape(3, 2)
    assert len(list(batch_iter(X, 100))) == 1
    assert list(batch_iter(np.empty((0, 2)), 4)) == []
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            list(batch_iter(X, bad))


def test_batch_iter_shuffle_is_reproducible_and_covers_all_rows():
    X = np.arange(10).reshape(5, 2)
    a = np.vstack(list(batch_iter(X, 2, shuffle=True, seed=1)))
    b = np.vstack(list(batch_iter(X, 2, shuffle=True, seed=1)))
    np.testing.assert_array_equal(a, b)
    assert sorted(a[:, 0]) == sorted(X[:, 0])
    first = next(batch_iter(X, 2, shuffle=True, seed=1))
    assert not np.shares_memory(first, X)


def test_batch_iter_zero_dim_raises():
    with pytest.raises(ValueError, match="at least one dimension"):
        list(batch_iter(np.float64(3.0), 2))


def test_activations_and_logsumexp_accept_non_contiguous_input():
    base = np.arange(12.0).reshape(3, 4)
    for X in (base.T, base[:, ::2]):  # transposed, and strided columns
        assert not X.flags["C_CONTIGUOUS"]
        same = np.ascontiguousarray(X)
        np.testing.assert_allclose(softmax(X, axis=1),
                                   softmax(same, axis=1))
        np.testing.assert_allclose(logsumexp(X, axis=0),
                                   logsumexp(same, axis=0))
        np.testing.assert_allclose(sigmoid(X), sigmoid(same))


def test_distances_accept_non_contiguous_input():
    A = np.arange(24.0).reshape(4, 6)[:, ::2]   # shape (4, 3), strided
    B = np.arange(18.0).reshape(3, 6)[:, ::2]   # shape (3, 3), strided
    assert not A.flags["C_CONTIGUOUS"] and not B.flags["C_CONTIGUOUS"]
    A2, B2 = np.ascontiguousarray(A), np.ascontiguousarray(B)
    np.testing.assert_allclose(euclidean_distance(A, B),
                               euclidean_distance(A2, B2))
    np.testing.assert_allclose(manhattan_distance(A, B),
                               manhattan_distance(A2, B2))


def test_batch_iter_on_strided_and_reversed_input():
    X = np.arange(20).reshape(5, 4)[::-1, ::2]  # negative and positive step
    batches = list(batch_iter(X, 2))
    assert [b.shape[0] for b in batches] == [2, 2, 1]
    np.testing.assert_array_equal(np.vstack(batches), X)
