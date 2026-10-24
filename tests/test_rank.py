"""Tests for numcompute.rank. Expected values are worked out by hand."""

import numpy as np
import pytest

from numcompute.rank import percentile, rank

MODES = ["linear", "lower", "higher", "midpoint"]


def test_rank_single_tie_among_distinct_values():
    data = [10, 20, 20, 30]
    assert rank(data, "average").tolist() == [1.0, 2.5, 2.5, 4.0]  # (2 + 3) / 2
    assert rank(data, "dense").tolist() == [1, 2, 2, 3]  # no rank skipped
    assert rank(data, "ordinal").tolist() == [1, 2, 3, 4]  # tie broken by order


def test_rank_all_equal_values():
    data = [5, 5, 5, 5]
    assert rank(data, "average").tolist() == [2.5, 2.5, 2.5, 2.5]  # (1+2+3+4) / 4
    assert rank(data, "dense").tolist() == [1, 1, 1, 1]
    assert rank(data, "ordinal").tolist() == [1, 2, 3, 4]


def test_rank_unsorted_input_with_tie():
    data = [30, 10, 20, 10]
    assert rank(data, "average").tolist() == [4.0, 1.5, 3.0, 1.5]
    assert rank(data, "dense").tolist() == [3, 1, 2, 1]
    assert rank(data, "ordinal").tolist() == [4, 1, 3, 2]  # the first 10 ranks 1


def test_rank_empty_array():
    for method in ("average", "dense", "ordinal"):
        assert rank([], method).shape == (0,)


def test_percentile_interpolation_modes():
    # q=40 on four points sits at position 0.4 * 3 = 1.2, between 2 and 3.
    data = [1, 2, 3, 4]
    assert percentile(data, 40, "linear") == pytest.approx(2.2)
    assert percentile(data, 40, "lower") == 2.0
    assert percentile(data, 40, "higher") == 3.0
    assert percentile(data, 40, "midpoint") == 2.5


def test_percentile_q0_is_min_and_q100_is_max():
    for mode in MODES:
        assert percentile([4, 1, 3, 2], 0, mode) == 1.0
        assert percentile([4, 1, 3, 2], 100, mode) == 4.0
    assert percentile([4, 1, 3, 2], [0, 50, 100]).tolist() == [1.0, 2.5, 4.0]


def test_percentile_exact_position_survives_float_rounding():
    # Position is 58 * 50 / 100 = 29 exactly; computing 58 / 100 * 50 instead
    # gives 28.999999999999996, and 'lower' would wrongly return 28.
    for mode in MODES:
        assert percentile(np.arange(51), 58, mode) == 29.0


def test_non_contiguous_input():
    view = np.array([3.0, 0.0, 9.0, 0.0, 1.0])[::2]  # 3, 9, 1
    assert not view.flags.c_contiguous
    assert rank(view).tolist() == [2.0, 3.0, 1.0]
    assert percentile(view, 50) == 3.0


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="NaN"):
        rank([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="method"):
        rank([1, 2, 3], "min")
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        percentile([1, 2, 3], 101)
    with pytest.raises(ValueError, match="interpolation"):
        percentile([1, 2, 3], 50, "nearest")
    with pytest.raises(TypeError, match="numeric"):
        rank(["a", "b"])
