"""Sorting, top-k via argpartition, quickselect and binary search.

Conventions: 1-D inputs have shape ``(n,)``; 2-D inputs have shape
``(n_samples, n_features)``. Inputs are never modified. NaNs are rejected by
``topk`` and ``quickselect`` because NaN has no defined order.
"""

import numpy as np

__all__ = ["stable_sort", "multi_key_sort", "topk", "quickselect", "binary_search"]


def _as_1d(values, name):
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError(f"Expected {name} with shape (n,), got {a.shape}")
    return a


def _check_no_nan(a, name):
    if a.dtype.kind == "f" and np.isnan(a).any():
        raise ValueError(f"{name} contains NaN, which has no defined order")


def _check_k(k, low, high):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"Expected k to be an integer, got {type(k).__name__}")
    if not low <= k <= high:
        raise ValueError(f"Expected {low} <= k <= {high}, got k={k}")
    return int(k)


def stable_sort(a, axis=-1):
    """Sort an array along an axis, keeping equal elements in input order.

    Parameters
    ----------
    a : array_like, any shape
        Array to be sorted.
    axis : int, default -1
        Axis to sort along. For a 2-D array, ``axis=0`` sorts each column
        (features) and ``axis=1`` sorts each row (samples).
        The default is -1, which sorts along the last axis.

    Returns
    -------
    sorted_array :ndarray, same shape and dtype as ``a``
        A sorted copy. NaNs are placed last.

    Raises
    ------
    ValueError
        If ``axis`` is out of bounds (including any axis on a 0-d input).

    Notes
    -----
    Time O(n log n) per sorted lane, space O(n).
    """
    return np.sort(np.asarray(a), axis=axis, kind="stable")


def multi_key_sort(X, keys, return_indices=False):
    """Sort the rows of a 2-D array by several columns.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Array to be sorted.
    keys : int or sequence of int, shape (n_keys,)
        Column indices, primary key first. Later keys break ties in earlier keys.
        Rows equal on every key keep their input order.
        Supports negative indexing.
    return_indices : bool, default False
        Also return the row permutation.

    Returns
    -------
    X_sorted : ndarray, shape (n_samples, n_features)
        A sorted copy of ``X``.
    order : ndarray of int, shape (n_samples,)
        The row indices that would sort ``X``.
        Only if ``return_indices``; ``X_sorted == X[order]``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, ``keys`` is empty, or a key is out of range.
    TypeError
        If ``keys`` is not integer-valued.

    Notes
    -----
    Time O(n_keys * n log n), space O(n_keys * n).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"Expected X with shape (n_samples, n_features), got {X.shape}"
        )
    keys = np.atleast_1d(np.asarray(keys))
    if keys.size == 0:
        raise ValueError("Expected at least one key column, got none")
    if keys.dtype.kind not in "iu":
        raise TypeError(f"Expected integer key columns, got dtype {keys.dtype}")
    n_features = X.shape[1]
    if ((keys < -n_features) | (keys >= n_features)).any():
        raise ValueError(
            f"Key columns must index into {n_features} features, got {keys.tolist()}"
        )
    # np.lexsort treats the LAST key as primary, that's why this reverses the keys.
    order = np.lexsort(X[:, keys[::-1]].T)
    return (X[order], order) if return_indices else X[order]


def topk(values, k, largest=True, return_indices=True):
    """Return the k largest (or smallest) values, best first.

    Parameters
    ----------
    values : array_like, shape (n,)
    k : int
        Number of elements to select, ``0 <= k <= n``.
    largest : bool, default True
        Select the largest values (descending) or the smallest (ascending).
    return_indices : bool, default True
        Also return the positions of the selected values in ``values``.

    Returns
    -------
    top : ndarray, shape (k,)
    indices : ndarray of int, shape (k,)
        Only if ``return_indices``; ``top == values[indices]``.

    Raises
    ------
    ValueError
        If ``values`` is not 1-D, contains NaN, or ``k`` is outside ``[0, n]``.
    TypeError
        If ``k`` is not an integer.

    Notes
    -----
    Time O(n + k log k): ``np.argpartition`` selects in O(n), then only the k
    selected elements are sorted. Space O(n).
    """
    a = _as_1d(values, "values")
    _check_no_nan(a, "values")
    n = a.size
    k = _check_k(k, 0, n)
    if k == 0:
        idx = np.empty(0, dtype=np.intp)
    elif largest:
        idx = np.argpartition(a, n - k)[n - k:]
        idx = idx[np.argsort(a[idx], kind="stable")[::-1]]
    else:
        idx = np.argpartition(a, k - 1)[:k]
        idx = idx[np.argsort(a[idx], kind="stable")]
    top = a[idx]
    return (top, idx) if return_indices else top


def quickselect(values, k, seed=None):
    """Return the k-th smallest value (0-indexed) without fully sorting.

    Parameters
    ----------
    values : array_like, shape (n,)
    k : int
        Rank of the wanted element in sorted order, ``0 <= k < n``. ``k=0`` is
        the minimum and ``k=n-1`` the maximum.
    seed : int, optional
        Seed for the random pivot choice. The result does not depend on it.

    Returns
    -------
    scalar
        Equal to ``np.sort(values)[k]``.

    Raises
    ------
    ValueError
        If ``values`` is not 1-D, is empty, contains NaN, or ``k`` is outside
        ``[0, n - 1]``.
    TypeError
        If ``k`` is not an integer.

    Notes
    -----
    Expected time O(n), worst case O(n^2) with repeatedly unlucky pivots;
    space O(n).
    """
    a = _as_1d(values, "values")
    _check_no_nan(a, "values")
    k = _check_k(k, 0, a.size - 1)
    rng = np.random.default_rng(seed)
    while True:
        pivot = a[rng.integers(a.size)]
        smaller = a[a < pivot]
        n_equal = np.count_nonzero(a == pivot)
        if k < smaller.size:
            a = smaller
        elif k < smaller.size + n_equal:
            return pivot
        else:
            k -= smaller.size + n_equal
            a = a[a > pivot]


def binary_search(sorted_array, x):
    """Find where ``x`` belongs in a sorted array and whether it is present.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Must already be sorted ascending.
    x : scalar or array_like, any shape
        Value(s) to look for. Each is searched independently.

    Returns
    -------
    index : int, or ndarray of int with shape ``x.shape``
        Leftmost position at which ``x`` could be inserted to keep the array
        sorted, in ``[0, n]``. If ``x`` is present this is its first occurrence.
    exists : bool, or ndarray of bool with shape ``x.shape``
        Whether ``x`` is in the array.

    Raises
    ------
    ValueError
        If ``sorted_array`` is not 1-D.

    Notes
    -----
    Time O(log n) per searched value, space O(1) per searched value. Uses
    ``np.searchsorted`` (a binary search) twice.
    """
    a = _as_1d(sorted_array, "sorted_array")
    index = np.searchsorted(a, x, side="left")
    exists = np.searchsorted(a, x, side="right") > index
    if np.ndim(x) == 0:
        return int(index), bool(exists)
    return index, exists
