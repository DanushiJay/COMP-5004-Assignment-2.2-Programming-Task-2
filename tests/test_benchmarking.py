"""Tests for numcompute.benchmarking."""

import numpy as np
import pytest

from numcompute import rank, sort_search, stats
from numcompute.benchmarking import (
    _loop_average_rank,
    _loop_binary_search,
    _loop_column_vars,
    compare,
    format_table,
    time_function,
)


def test_time_function_returns_positive_seconds():
    assert time_function(sum, [1, 2, 3], repeat=2, number=3) > 0
    with pytest.raises(ValueError, match="repeat"):
        time_function(sum, [1], repeat=0)


def test_compare_rejects_implementations_that_disagree():
    row = compare("sum", lambda: sum([1, 2]), lambda: 3, repeat=1)
    assert set(row) == {"name", "vectorised", "loop", "speedup"}
    with pytest.raises(ValueError, match="differ"):
        compare("wrong", lambda: 3, lambda: 4, repeat=1)


def test_format_table_has_header_and_one_line_per_row():
    rows = [{"name": "demo", "vectorised": 0.001, "loop": 0.1, "speedup": 100.0}]
    lines = format_table(rows).splitlines()
    assert lines[0].startswith("| Benchmark |")
    assert lines[2] == "| demo | 1.000 | 100.000 | 100x |"


def test_loop_versions_match_numcompute():
    values = [30, 10, 20, 10, 30, 30]  # ties, unsorted
    assert _loop_average_rank(values) == rank.rank(values).tolist()
    X = [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]
    np.testing.assert_allclose(_loop_column_vars(X), stats.var(X, axis=0))
    data, queries = [1, 3, 3, 3, 7, 9], [3, 4, 0, 10]  # first of duplicates, gaps, ends
    index, exists = sort_search.binary_search(data, queries)
    loop_index, loop_exists = _loop_binary_search(data, queries)
    np.testing.assert_array_equal(loop_index, index)
    np.testing.assert_array_equal(loop_exists, exists)
