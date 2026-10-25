"""Numerically stable math helpers used by the other modules.

as_float_array : turn input into floats, but reject text and None.
logsumexp, log_softmax, softmax : safe even for very large numbers
    (up to about +-1e4), because the largest value is subtracted first.
sigmoid, relu : activation functions. ``sigmoid`` uses ``exp(-|x|)``, so
    it never overflows.
pairwise_diff, euclidean_distance, manhattan_distance : distances between
    every pair of rows, with no Python loop.
batch_iter : hands out data in small batches, one at a time. This is one
    of the few places a loop is allowed.

"""

import numpy as np

__all__ = [
    "as_float_array",
    "logsumexp",
    "log_softmax",
    "softmax",
    "sigmoid",
    "relu",
    "pairwise_diff",
    "euclidean_distance",
    "manhattan_distance",
    "batch_iter",
]


def as_float_array(x, name="x"):
    """Convert input to a float array, refusing non-numeric data.

    Parameters
    ----------
    x : array_like
        Numeric input (bool, int or float). Strings and ``None`` are
        rejected instead of being converted quietly.
    name : str, default "x"
        Name used in error messages.

    Returns
    -------
    ndarray of float64, same shape as ``x``.

    Shapes
    ------
    in: any shape -> out: same shape

    Raises
    ------
    TypeError : if ``x`` has a non-numeric dtype (strings, objects).

    Notes
    -----
    Time O(size); no copy is made if ``x`` is already float64.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "biuf":
        raise TypeError(
            f"{name} must be numeric (bool, int or float), "
            f"got dtype {arr.dtype}"
        )
    return arr.astype(float, copy=False)


def logsumexp(x, axis=None, keepdims=False):
    """Compute ``log(sum(exp(x)))`` without overflow or underflow.

    Calculating this direcly overflows for large numbers. Instead the 
    largest value ``m`` is subtracted first: ``m+ log9sum(exp(x-m)))```
    If every value is ``-inf``, the result is ``-inf`` rather than NaN.

    Parameters
    ----------
    x : array_like
        Numeric input.
    axis : int or None, default None
        Axis to reduce; None reduces over all elements.
    keepdims : bool, default False
        Keep the reduced axis with length 1.

    Returns
    -------
    ndarray or float
        Reduced values.

    Shapes
    ------
    in: (n, d), axis=1 -> out: (n,); axis=0 -> (d,); axis=None -> scalar.
    With ``keepdims=True`` the reduced axis stays with length 1.

    Raises
    ------
    TypeError : if ``x`` is not numeric.

    Notes
    -----
    Time O(size), space O(size) for the shifted temporary. Empty input
    gives ``-inf`` (the log of an empty sum).
    """

    x = as_float_array(x)
    if x.size == 0:
        shape = np.sum(x, axis=axis, keepdims=keepdims)
        return np.full_like(shape, -np.inf, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    total = np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = m_safe + np.log(total)
    if not keepdims:
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        else:
            out = out.reshape(())
    return out[()] if out.ndim == 0 else out


def log_softmax(x, axis=-1):
    """Log of the softmax probabilities along ``axis``, computed safely.

    Parameters
    ----------
    x : array_like
        Numeric logits.
    axis : int, default -1
        Axis alog which the probabilities sum up to 1.

    Returns
    -------
    ndarray, same shape as ``x``.

    Shapes
    ------
    in: (n, d) -> out: (n, d)

    Raises
    ------
    TypeError : if ``x`` is not numeric.

    Notes
    -----
    Time O(size), space O(size). Computed as ``x - logsumexp(x)``.
    """
    x = as_float_array(x)
    return x - logsumexp(x, axis=axis, keepdims=True)


def softmax(x, axis=-1):
    """Turn scores into probabilities that add up to 1 along ``axis``.

    The largest value is subtracted first, so large inputes do not overflow.

    Parameters
    ----------
    x : array_like
        Numeric logits.
    axis : int, default -1
        Axis normalised to sum to 1.

    Returns
    -------
    ndarray, same shape as ``x``; entries in [0, 1].

    Shapes
    ------
    in: (n, d) -> out: (n, d)

    Raises
    ------
    TypeError : if ``x`` is not numeric.

    Notes
    -----
    Time O(size), space O(size). Inputs of +-1e4 do not overflow.
    """
    return np.exp(log_softmax(x, axis=axis))


def sigmoid(x):
    """Numerically stable logistic function ``1 / (1 + exp(-x))``.

    Uses ``exp(-|x|)`` so the exponent is never large and positive, which
    keeps it safe for very large or very small inputs.

    Parameters
    ----------
    x : array_like
        Numeric input.

    Returns
    -------
    ndarray, same shape as ``x``, values in [0, 1].

    Shapes
    ------
    in: any -> out: same

    Raises
    ------
    TypeError : if ``x`` is not numeric.

    Notes
    -----
    Time O(size), space O(size).
    """
    x = as_float_array(x)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def relu(x):
    """Replace negative numbers with 0 ``max(x, 0)``; NaN is propagated.

    Parameters
    ----------
    x : array_like
        Numeric input.

    Returns
    -------
    ndarray, same shape as ``x``.

    Shapes
    ------
    in: any -> out: same

    Raises
    ------
    TypeError : if ``x`` is not numeric.

    Notes
    -----
    Time O(size), space O(size).
    """
    x = as_float_array(x)
    return np.where(np.isnan(x), x, np.maximum(x, 0.0))


def pairwise_diff(a, b):
    """Broadcast differences ``a[i] - b[j]`` for every pair of rows.

    Parameters
    ----------
    a : array_like, shape (n, d) or (d,)
    b : array_like, shape (m, d) or (d,)
        1-D inputs are treated as a single row.

    Returns
    -------
    ndarray, shape (n, m, d).

    Shapes
    ------
    in: (n, d), (m, d) -> out: (n, m, d)

    Raises
    ------
    TypeError : non-numeric input.
    ValueError : inputs not 1-D/2-D, or feature dimensions differ.

    Notes
    -----
    Time O(n*m*d), space O(n*m*d).
    """
    a = np.atleast_2d(as_float_array(a, "a"))
    b = np.atleast_2d(as_float_array(b, "b"))
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected 1-D or 2-D inputs, got {a.ndim}-D and {b.ndim}-D"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: a has {a.shape[1]}, "
            f"b has {b.shape[1]}"
        )
    return a[:, None, :] - b[None, :, :]


def euclidean_distance(a, b):
    """Pairwise Euclidean (L2) distances between rows of ``a`` and ``b``.

    Parameters
    ----------
    a : array_like, shape (n, d) or (d,)
    b : array_like, shape (m, d) or (d,)

    Returns
    -------
    ndarray, shape (n, m); entry ``[i, j] = ||a[i] - b[j]||_2``.

    Shapes
    ------
    in: (n, d), (m, d) -> out: (n, m)

    Raises
    ------
    TypeError : non-numeric input.
    ValueError : inputs not 1-D/2-D, or feature dimensions differ.

    Notes
    -----
    Time O(n*m*d), space O(n*m*d) for the broadcast difference.
    """
    diff = pairwise_diff(a, b)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def manhattan_distance(a, b):
    """Pairwise Manhattan (L1) distances between rows of ``a`` and ``b``.

    Parameters
    ----------
    a : array_like, shape (n, d) or (d,)
    b : array_like, shape (m, d) or (d,)

    Returns
    -------
    ndarray, shape (n, m); entry ``[i, j] = sum_k |a[i,k] - b[j,k]|``.

    Shapes
    ------
    in: (n, d), (m, d) -> out: (n, m)

    Raises
    ------
    TypeError : non-numeric input.
    ValueError : inputs not 1-D/2-D, or feature dimensions differ.

    Notes
    -----
    Time O(n*m*d), space O(n*m*d).
    """
    return np.sum(np.abs(pairwise_diff(a, b)), axis=-1)


def batch_iter(X, batch_size, shuffle=False, seed=None):
    """Yield consecutive mini-batches along axis 0.

    Parameters
    ----------
    X : array_like, shape (n_samples, ...)
        Data to batch.
    batch_size : int
        Positive batch size. The last batch may be smaller; a size larger
        than ``n_samples`` yields a single batch.
    shuffle : bool, default False
        Visit rows in random order. Shuffled batches are copies; unshuffled
        batches are views into ``X``.
    seed : int or None, default None
        Seed for reproducible shuffling (ignored if ``shuffle`` is False).

    Yields
    ------
    ndarray, shape (<=batch_size, ...)

    Shapes
    ------
    in: (n, ...) -> yields ceil(n / batch_size) batches of (b, ...).

    Raises
    ------
    ValueError : if ``X`` is 0-dimensional, or ``batch_size`` is not a
        positive integer. Raised when iteration starts (generator).

    Notes
    -----
    Time O(1) per batch (O(b*d) when shuffled) plus O(n) once for the
    permutation. Space O(n) for the index permutation when shuffling, O(1)
    otherwise. One of the sanctioned loops: it iterates over batches only.
    """
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must have at least one dimension")
    bad_type = isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer))
    if bad_type or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )
    n = X.shape[0]
    idx = np.random.default_rng(seed).permutation(n) if shuffle else None
    for start in range(0, n, batch_size):
        stop = start + batch_size
        yield X[start:stop] if idx is None else X[idx[start:stop]]
