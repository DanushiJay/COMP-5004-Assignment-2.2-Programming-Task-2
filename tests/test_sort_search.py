"""Tests for numcompute.sort_search. Expected values are worked out by hand."""

import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)


def test_stable_sort_sorts_and_keeps_equal_elements_in_input_order():
    assert stable_sort([3, 1, 2]).tolist() == [1, 2, 3]
    X = np.array([[3, 1], [1, 2]])
    assert stable_sort(X, axis=0).tolist() == [[1, 1], [3, 2]]  # each column
    # 0.0 and -0.0 compare equal but are distinguishable by their sign bit.
    assert np.signbit(stable_sort([0.0, -0.0])).tolist() == [False, True]
    assert np.signbit(stable_sort([-0.0, 0.0])).tolist() == [True, False]


def test_multi_key_sort_orders_by_primary_then_secondary_key():
    # Columns: (group, score, id). Rows 1 and 3 tie on both sort keys.
    table = np.array([[2, 9, 0], [1, 5, 1], [2, 3, 2], [1, 5, 3], [1, 4, 4]])
    out, order = multi_key_sort(table, [0, 1], return_indices=True)
    assert order.tolist() == [4, 1, 3, 2, 0]  # tied rows keep their input order
    assert out[:, 1].tolist() == [4, 5, 5, 3, 9]
    with pytest.raises(ValueError, match="Key columns"):
        multi_key_sort(table, [3])  # there is no column 3


def test_topk_returns_values_and_indices_best_first():
    values = [5, 1, 9, 3, 9, 2]  # the 9 is duplicated
    top, idx = topk(values, 3)
    assert top.tolist() == [9, 9, 5]
    assert sorted(idx[:2].tolist()) == [2, 4]  # the two 9s, in either order
    assert idx[2] == 0
    top, idx = topk(values, 2, largest=False)
    assert top.tolist() == [1, 2]
    assert idx.tolist() == [1, 5]


def test_topk_extreme_k():
    values = [5, 1, 9, 3, 9, 2]
    top, idx = topk(values, 0)
    assert top.shape == (0,) and idx.shape == (0,)
    everything = topk(values, 6, return_indices=False)
    assert isinstance(everything, np.ndarray)  # values only, not a (values, indices) pair
    assert everything.tolist() == [9, 9, 5, 3, 2, 1]
    with pytest.raises(ValueError, match="<= k <="):
        topk(values, 7)  # k > len


def test_topk_all_equal_values():
    top, idx = topk([7, 7, 7, 7], 2)
    assert top.tolist() == [7, 7]
    assert len(set(idx.tolist())) == 2  # two distinct positions


def test_quickselect_returns_kth_smallest_with_duplicates():
    values = [3, 1, 3, 2, 3]  # sorted: 1 2 3 3 3
    assert [quickselect(values, k) for k in range(5)] == [1, 2, 3, 3, 3]
    assert values == [3, 1, 3, 2, 3]  # input is left untouched


def test_quickselect_all_equal_values():
    assert quickselect([7, 7, 7, 7], 2) == 7
    # A naive two-way partition degrades to O(n^2) here; three-way does not.
    assert quickselect(np.full(100_000, 3.0), 50_000) == 3.0


def test_quickselect_sorted_and_reverse_sorted_input():
    ascending = np.arange(1000)
    assert quickselect(ascending, 0) == 0
    assert quickselect(ascending, 999) == 999
    assert quickselect(ascending[::-1], 250) == 250


def test_quickselect_extreme_k_raises():
    with pytest.raises(ValueError, match="<= k <="):
        quickselect([1, 2, 3, 4], 4)  # valid k is 0..3
    with pytest.raises(ValueError, match="<= k <="):
        quickselect([1, 2, 3, 4], -1)
    with pytest.raises(ValueError):
        quickselect([], 0)


def test_binary_search_found_and_not_found():
    data = [1, 3, 3, 3, 7, 9]
    assert binary_search(data, 3) == (1, True)  # first of the three 3s
    assert binary_search(data, 4) == (4, False)  # belongs between 3 and 7
    assert binary_search(data, 0) == (0, False)  # before everything
    assert binary_search(data, 10) == (6, False)  # after everything


def test_empty_arrays():
    assert stable_sort([]).shape == (0,)
    assert np.shape(multi_key_sort(np.empty((0, 3)), [0])) == (0, 3)
    assert np.shape(topk([], 0, return_indices=False)) == (0,)
    assert binary_search([], 5) == (0, False)


def test_non_contiguous_input():
    view = np.array([7, 0, 2, 0, 9, 0, 4, 0, 1])[::2]  # 7, 2, 9, 4, 1
    assert not view.flags.c_contiguous
    assert stable_sort(view).tolist() == [1, 2, 4, 7, 9]
    top, _ = topk(view, 2)
    assert top.tolist() == [9, 7]
    assert quickselect(view, 2) == 4
    assert binary_search(np.arange(10)[::3], 6) == (2, True)  # 0, 3, 6, 9


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="NaN"):
        topk([1.0, np.nan, 3.0], 1)
    with pytest.raises(ValueError, match="NaN"):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(TypeError, match="integer"):
        topk([5, 1, 9], 2.0)
    with pytest.raises(ValueError, match=r"shape \(n,\)"):
        topk(np.ones((2, 2)), 1)
