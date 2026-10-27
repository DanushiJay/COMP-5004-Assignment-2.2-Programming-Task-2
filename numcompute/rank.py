"""Ranking with tie handling, and percentiles.

Both functions take 1-D data of shape ``(n,)`` and reject NaN, which has no
defined rank.
"""

import numpy as np

__all__ = ["rank", "percentile"]

_METHODS = ("average", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def _as_1d_numeric(data):
    a = np.asarray(data)
    if a.ndim != 1:
        raise ValueError(f"Expected data with shape (n,), got {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"Expected numeric data, got dtype {a.dtype}")
    if a.dtype.kind == "f" and np.isnan(a).any():
        raise ValueError("data contains NaN, which has no defined rank")
    return a


def rank(data, method="average"):
    """Assign 1-based ranks to the data, smallest value first.

    Parameters
    ----------
    data : array_like, shape (n,)
    method : {'average', 'dense', 'ordinal'}, default 'average'
        How tied values are ranked. For ``[10, 20, 20, 30]``:

        - 'average': ties share the mean of their positions -> ``[1, 2.5, 2.5, 4]``
        - 'dense': ties share a rank and no rank is skipped -> ``[1, 2, 2, 3]``
        - 'ordinal': every value gets a distinct rank, ties broken by input
          order -> ``[1, 2, 3, 4]``

    Returns
    -------
    ndarray, shape (n,)
        float64 for 'average', integer for 'dense' and 'ordinal'. An empty
        input gives an empty output.

    Shapes
    ------
    ``data`` is ``(n,)``; output is ``(n,)``.

    Raises
    ------
    ValueError
        If ``data`` is not 1-D, contains NaN, or ``method`` is unknown.
    TypeError
        If ``data`` is not numeric.

    Notes
    -----
    Time O(n log n) for one stable sort, space O(n).
    """
    if method not in _METHODS:
        raise ValueError(f"Expected method in {_METHODS}, got {method!r}")
    a = _as_1d_numeric(data)
    n = a.size
    if n == 0:
        return np.empty(0, dtype=np.float64 if method == "average" else np.intp)

    order = np.argsort(a, kind="stable")
    if method == "ordinal":
        ranks = np.empty(n, dtype=np.intp)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    s = a[order]
    is_new = np.concatenate(([True], s[1:] != s[:-1]))  # start of each tie group
    dense = np.cumsum(is_new)  # group number, 1..n_groups, in sorted order
    if method == "dense":
        ranks = np.empty(n, dtype=np.intp)
        ranks[order] = dense
        return ranks

    starts = np.flatnonzero(is_new)
    counts = np.diff(np.append(starts, n))
    # A group at sorted positions start .. start+count-1 holds ordinal ranks
    # start+1 .. start+count, whose mean is start + (count + 1) / 2.
    average = starts + (counts + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = average[dense - 1]
    return ranks


def percentile(data, q, interpolation="linear"):
    """Compute the q-th percentile(s) of the data.

    Parameters
    ----------
    data : array_like, shape (n,), n >= 1
    q : float or array_like of float, any shape
        Percentile(s) in ``[0, 100]``. 0 is the minimum, 100 the maximum.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default 'linear'
        What to return when the percentile falls between two sorted values
        ``lo`` and ``hi``, at fraction ``f`` of the way from ``lo`` to ``hi``:

        - 'linear': ``lo + f * (hi - lo)``
        - 'lower': ``lo``
        - 'higher': ``hi``
        - 'midpoint': ``(lo + hi) / 2``

    Returns
    -------
    float or ndarray of float
        The percentile(s).

    Shapes
    ------
    ``data`` is ``(n,)``; output has the same shape as ``q`` (a scalar for a
    scalar ``q``).

    Raises
    ------
    ValueError
        If ``data`` is not 1-D, is empty, contains NaN, if any ``q`` is
        outside ``[0, 100]``, or ``interpolation`` is unknown.
    TypeError
        If ``data`` is not numeric.

    Notes
    -----
    Time O(n log n + m) for n data points and m percentiles, space O(n + m).
    The percentile sits at position ``q * (n - 1) / 100`` in the sorted data.
    For ``[1, 2, 3, 4]`` and ``q=40`` that is 1.2, giving linear 2.2, lower 2,
    higher 3 and midpoint 2.5.
    """
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            f"Expected interpolation in {_INTERPOLATIONS}, got {interpolation!r}"
        )
    a = _as_1d_numeric(data)
    n = a.size
    if n == 0:
        raise ValueError("Expected non-empty data, got shape (0,)")
    q = np.asarray(q, dtype=np.float64)
    if not ((q >= 0) & (q <= 100)).all():
        raise ValueError("Expected every q in [0, 100]")

    a = np.sort(a.astype(np.float64))
    pos = q * (n - 1) / 100.0
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    if interpolation == "lower":
        out = a[lo]
    elif interpolation == "higher":
        out = a[hi]
    elif interpolation == "midpoint":
        out = (a[lo] + a[hi]) / 2.0
    else:
        out = a[lo] + (pos - lo) * (a[hi] - a[lo])
    return out[()]  # 0-d result -> scalar; arrays pass through unchanged
