"""Micro-benchmark harness comparing vectorised NumPy code with Python loops.

Run ``python -m numcompute.benchmarking`` to perform the benchmarks. 
Inputs come from a fixed random seed, so every run times the same data.

Each loop version does the same calculation as its NumPy counterpart with the
same time complexity, using plain Python lists and loops, so the comparison
measures vectorisation vs python loop performance.
"""

import os
import platform
import timeit

import numpy as np

from . import rank, sort_search, stats

__all__ = [
    "time_function",
    "compare",
    "environment_info",
    "format_table",
    "run_all",
    "run_size_sweep",
]


def time_function(func, *args, repeat=5, number=1):
    """Time ``func(*args)`` and return the fastest time per call.

    Parameters
    ----------
    func : callable
    *args
        Passed to ``func`` on every call.
    repeat : int, default 5
        Number of timing rounds.
    number : int, default 1
        Calls per round.

    Returns
    -------
    float
        Seconds per call in the fastest round.

    Shapes
    ------
    Output is a scalar.

    Raises
    ------
    ValueError
        If ``repeat`` or ``number`` is less than 1.

    Notes
    -----
    Runs func repeat * number times; space O(1). The fastest round is reported.
    """
    if repeat < 1 or number < 1:
        raise ValueError(
            f"Expected repeat >= 1 and number >= 1, got {repeat} and {number}"
        )
    timer = timeit.Timer(lambda: func(*args))
    return min(timer.repeat(repeat=repeat, number=number)) / number


def compare(name, vectorised, loop, repeat=5, number=1):
    """Check that two implementations agree, then time both.

    Parameters
    ----------
    name : str
        Label for the results table.
    vectorised, loop : callable
        Zero-argument functions computing the same result.
    repeat, number : int
        Passed to ``time_function``.

    Returns
    -------
    dict
        Keys ``name``, ``vectorised`` and ``loop`` (seconds per call) and
        ``speedup`` (loop time divided by vectorised time).

    Shapes
    ------
    Output is a dict of scalars.

    Raises
    ------
    ValueError
        If the two results differ, since timing wrong code is meaningless.

    Notes
    -----
    Runs each function ``repeat * number + 1`` times.
    """
    expected = np.asarray(vectorised(), dtype=np.float64)
    got = np.asarray(loop(), dtype=np.float64)
    if expected.shape != got.shape or not np.allclose(expected, got):
        raise ValueError(f"{name}: the loop and vectorised results differ")
    t_vec = time_function(vectorised, repeat=repeat, number=number)
    t_loop = time_function(loop, repeat=repeat, number=number)
    return {"name": name, "vectorised": t_vec, "loop": t_loop, "speedup": t_loop / t_vec}


def environment_info():
    """Describe the machine and library versions the timings come from.

    Returns
    -------
    dict of str
        Python and NumPy versions, OS, CPU and number of CPU cores.

    Shapes
    ------
    Output is a dict of strings.

    Notes
    -----
    Time and space O(1).
    """
    return {
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "OS": platform.platform(),
        "CPU": platform.processor() or platform.machine(),
        "CPU cores": str(os.cpu_count()),
    }


def format_table(rows):
    """Format ``compare`` results as a Markdown table.

    Parameters
    ----------
    rows : list of dict
        Results from ``compare``.

    Returns
    -------
    str
        Columns: benchmark, vectorised time, loop time, speedup.

    Shapes
    ------
    Output is one string: two header lines plus one line per row.

    Notes
    -----
    Time and space O(len(rows)).
    """
    lines = [
        "| Benchmark | Vectorised (ms) | Python loop (ms) | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['vectorised'] * 1e3:.3f} | "
            f"{r['loop'] * 1e3:.3f} | {r['speedup']:.0f}x |"
        )
    return "\n".join(lines)


# Python-loop versions of the benchmarked functions. Each takes Python lists.

def _loop_mean(values):
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def _loop_column_means(rows):
    n_rows, n_cols = len(rows), len(rows[0])
    means = []
    for j in range(n_cols):
        total = 0.0
        for i in range(n_rows):
            total += rows[i][j]
        means.append(total / n_rows)
    return means


def _loop_column_vars(rows):
    # Two passes, like np.var: first the column means, then squared deviations.
    n_rows, n_cols = len(rows), len(rows[0])
    means = _loop_column_means(rows)
    variances = []
    for j in range(n_cols):
        total = 0.0
        for i in range(n_rows):
            diff = rows[i][j] - means[j]
            total += diff * diff
        variances.append(total / n_rows)
    return variances


def _loop_merge_sort_indices(values):
    # Bottom-up merge sort of positions. Stable, so equal values keep their
    # input order, like np.argsort(kind="stable").
    order = list(range(len(values)))
    n = len(order)
    width = 1
    while width < n:
        merged = []
        for lo in range(0, n, 2 * width):
            left = order[lo:lo + width]
            right = order[lo + width:lo + 2 * width]
            i = j = 0
            while i < len(left) and j < len(right):
                if values[right[j]] < values[left[i]]:
                    merged.append(right[j])
                    j += 1
                else:
                    merged.append(left[i])
                    i += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
        order = merged
        width *= 2
    return order


def _loop_average_rank(values):
    # Sort, then give each run of tied values the mean of its 1-based positions.
    order = _loop_merge_sort_indices(values)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for pos in range(start, end + 1):
            ranks[order[pos]] = average
        start = end + 1
    return ranks


def _loop_binary_search(sorted_values, queries):
    # One hand-written binary search per query, finding the leftmost position.
    n = len(sorted_values)
    indices, exists = [], []
    for x in queries:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_values[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        indices.append(lo)
        exists.append(lo < n and sorted_values[lo] == x)
    return indices, exists


def run_all(seed=0, repeat=5):
    """Time the four main benchmarks on fixed random data.

    Parameters
    ----------
    seed : int, default 0
        Seed for the input data, so runs are reproducible.
    repeat : int, default 5
        Timing rounds per implementation.

    Returns
    -------
    list of dict
        One ``compare`` result each for ``stats.mean``, ``stats.var``,
        ``rank.rank`` and ``sort_search.binary_search``.

    Shapes
    ------
    Output has 4 entries.

    Notes
    -----
    Time and space O(n) for each benchmark, where n is the number of values.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2000, 50))
    values = rng.integers(0, 500, size=5000)  # many ties
    sorted_values = np.sort(rng.integers(0, 2_000_000, size=1_000_000))
    queries = rng.integers(0, 2_000_000, size=10_000)  # some present, some not
    # Lists for the loop versions are built here, outside the timed code.
    X_list, values_list = X.tolist(), values.tolist()
    sorted_list, queries_list = sorted_values.tolist(), queries.tolist()

    benchmarks = [
        ("stats.mean, (2000, 50), axis=0",
         lambda: stats.mean(X, axis=0),
         lambda: _loop_column_means(X_list)),
        ("stats.var, (2000, 50), axis=0",
         lambda: stats.var(X, axis=0),
         lambda: _loop_column_vars(X_list)),
        ("rank.rank (average), 5000 values",
         lambda: rank.rank(values, "average"),
         lambda: _loop_average_rank(values_list)),
        ("sort_search.binary_search, 10000 queries in 1000000 values",
         lambda: sort_search.binary_search(sorted_values, queries),
         lambda: _loop_binary_search(sorted_list, queries_list)),
    ]
    return [compare(name, vec, loop, repeat=repeat) for name, vec, loop in benchmarks]


def run_size_sweep(sizes=(1_000, 10_000, 100_000, 1_000_000), seed=0, repeat=3):
    """Time ``stats.mean`` on growing inputs to show how the speedup scales.

    Parameters
    ----------
    sizes : sequence of int, default (1_000, 10_000, 100_000, 1_000_000)
        Number of values in each run.
    seed : int, default 0
        Seed for the input data.
    repeat : int, default 3
        Timing rounds per implementation.

    Returns
    -------
    list of dict
        One ``compare`` result per size.

    Shapes
    ------
    Output has ``len(sizes)`` entries.

    Notes
    -----
    Both versions are O(n), so the speedup should be roughly constant with input size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        data = rng.normal(size=n)
        data_list = data.tolist()
        rows.append(
            compare(f"stats.mean, n={n:,}",
                    lambda: stats.mean(data),
                    lambda: _loop_mean(data_list),
                    repeat=repeat)
        )
    return rows


if __name__ == "__main__":
    for key, value in environment_info().items():
        print(f"{key}: {value}")
    print("\n" + format_table(run_all()))
    print("\nHow the speedup grows with input size:\n")
    print(format_table(run_size_sweep()))
