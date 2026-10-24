"""Descriptive and streaming (Welford) statistics, histogram and quantiles.

Shared conventions for the reducers (``mean``, ``median``, ``var``, ``std``,
``min``, ``max``, ``quantile``):

Axis and shapes
    ``X`` has shape ``(n_samples, n_features)``. ``axis=0`` reduces over the
    samples and returns shape ``(n_features,)``; ``axis=1`` reduces over the
    features and returns shape ``(n_samples,)``; ``axis=None`` reduces over
    everything and returns a scalar. A 1-D input of shape ``(n,)`` reduces to
    a scalar with ``axis=None`` or ``axis=0``.

NaN policy (``nan_policy``)
    'propagate' (default): any NaN in a slice makes that slice's result NaN.
    'omit': NaNs are ignored; a slice that is entirely NaN gives NaN.
    'raise': a ``ValueError`` is raised if the input contains any NaN.

Errors
    Empty input raises ``ValueError``; non-numeric input raises ``TypeError``.
"""

import warnings

import numpy as np

__all__ = [
    "mean",
    "median",
    "var",
    "std",
    "min",
    "max",
    "quantile",
    "histogram",
    "RunningStats",
]

_NAN_POLICIES = ("propagate", "omit", "raise")


def _as_float(x):
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"Expected numeric data, got dtype {a.dtype}")
    return a.astype(np.float64, copy=False)


def _reduce(x, axis, nan_policy, func, nanfunc, **kwargs):
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"Expected nan_policy in {_NAN_POLICIES}, got {nan_policy!r}")
    a = _as_float(x)
    if a.size == 0:
        raise ValueError(f"Expected a non-empty array, got shape {a.shape}")
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("Input contains NaN and nan_policy='raise'")
    if nan_policy == "omit":
        # NumPy warns on an all-NaN slice; the NaN it returns is the documented result.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return nanfunc(a, axis=axis, **kwargs)
    return func(a, axis=axis, **kwargs)


def mean(x, axis=None, nan_policy="propagate"):
    """Arithmetic mean along an axis.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n); space O(1) extra, or O(n) with 'omit'.
    """
    return _reduce(x, axis, nan_policy, np.mean, np.nanmean)


def median(x, axis=None, nan_policy="propagate"):
    """Median along an axis (mean of the two middle values for even counts).

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Average time O(n); space O(n).
    """
    return _reduce(x, axis, nan_policy, np.median, np.nanmedian)


def var(x, axis=None, ddof=0, nan_policy="propagate"):
    """Variance along an axis.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    ddof : int, default 0
        Divisor is ``N - ddof``: 0 gives the population value, 1 the unbiased
        sample value.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray
        A slice with ``N <= ddof`` gives NaN.

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n); space O(n).
    """
    return _reduce(x, axis, nan_policy, np.var, np.nanvar, ddof=ddof)


def std(x, axis=None, ddof=0, nan_policy="propagate"):
    """Standard deviation along an axis: the square root of the variance.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    ddof : int, default 0
        Divisor is ``N - ddof``: 0 gives the population value, 1 the unbiased
        sample value.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray
        A slice with ``N <= ddof`` gives NaN.

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n); space O(n).
    """
    return _reduce(x, axis, nan_policy, np.std, np.nanstd, ddof=ddof)


def min(x, axis=None, nan_policy="propagate"):
    """Minimum along an axis.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n); space O(1) extra, or O(n) with 'omit'.
    """
    return _reduce(x, axis, nan_policy, np.min, np.nanmin)


def max(x, axis=None, nan_policy="propagate"):
    """Maximum along an axis.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``. Output is a scalar for
    ``axis=None`` or 1-D input, ``(n_features,)`` for ``axis=0`` and
    ``(n_samples,)`` for ``axis=1``.

    Raises
    ------
    ValueError
        If ``x`` is empty, ``axis`` is out of bounds, ``nan_policy`` is
        unknown, or ``nan_policy='raise'`` and ``x`` contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n); space O(1) extra, or O(n) with 'omit'.
    """
    return _reduce(x, axis, nan_policy, np.max, np.nanmax)


def quantile(x, q, axis=None, nan_policy="propagate"):
    """Quantile(s) along an axis, with linear interpolation.

    Parameters
    ----------
    x : array_like, shape (n,) or (n_samples, n_features)
    q : float or array_like of float, shape () or (n_q,)
        Quantile(s) in ``[0, 1]``, so ``quantile(x, 0.25)`` equals
        ``rank.percentile(x, 25)``.
    axis : {None, 0, 1}, default None
        ``axis=0`` reduces over the samples (one result per feature),
        ``axis=1`` reduces over the features (one result per sample),
        ``None`` reduces over every element.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        'propagate' returns NaN for any slice containing NaN; 'omit' ignores
        NaNs (an all-NaN slice gives NaN); 'raise' raises on any NaN.

    Returns
    -------
    float or ndarray
        One result per value of ``q``.

    Shapes
    ------
    ``x`` is ``(n,)`` or ``(n_samples, n_features)``; ``q`` is a scalar or ``(n_q,)``.
    Scalar ``q``: output is a scalar for ``axis=None``, ``(n_features,)`` for
    ``axis=0`` and ``(n_samples,)`` for ``axis=1``. A ``q`` of shape ``(n_q,)``
    adds a leading axis: ``(n_q,)``, ``(n_q, n_features)`` or ``(n_q, n_samples)``.

    Raises
    ------
    ValueError
        If any ``q`` is outside ``[0, 1]``, ``x`` is empty, ``axis`` is out of
        bounds, ``nan_policy`` is unknown, or ``nan_policy='raise'`` and ``x``
        contains NaN.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n) per quantile; space O(n).
    """
    q = np.asarray(q, dtype=np.float64)
    if not ((q >= 0) & (q <= 1)).all():
        raise ValueError("Expected every q in [0, 1]")
    return _reduce(x, axis, nan_policy, np.quantile, np.nanquantile, q=q)


def histogram(x, bins=10, range=None, density=False, nan_policy="omit"):
    """Count how many values fall into each bin.

    Parameters
    ----------
    x : array_like, any shape
        Flattened before counting.
    bins : int or array_like of float, default 10
        Number of equal-width bins, or the increasing bin edges themselves.
    range : (float, float), optional
        Lower and upper edge of the bins. Defaults to ``(x.min(), x.max())``;
        values outside the range are not counted.
    density : bool, default False
        Return a probability density (integrates to 1) instead of counts.
    nan_policy : {'omit', 'raise'}, default 'omit'
        NaNs belong to no bin, so they are dropped or rejected.

    Returns
    -------
    counts : ndarray, shape (n_bins,)
        Integer counts, or floats if ``density``.
    edges : ndarray of float, shape (n_bins + 1,)
        Bin ``i`` covers ``[edges[i], edges[i+1])``. The last bin is closed on
        both sides, so a value equal to ``edges[-1]`` is counted in it.

    Shapes
    ------
    ``x`` is any shape (flattened); ``counts`` is ``(n_bins,)`` and ``edges`` is
    ``(n_bins + 1,)``.

    Raises
    ------
    ValueError
        If ``nan_policy`` is unknown, ``nan_policy='raise'`` and ``x`` contains
        NaN, ``x`` contains infinities with no ``range`` given, or ``bins`` is
        invalid.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n), space O(n + n_bins). Empty input gives all-zero counts.
    """
    if nan_policy not in ("omit", "raise"):
        raise ValueError(
            f"Expected nan_policy in ('omit', 'raise'), got {nan_policy!r}"
        )
    a = _as_float(x).ravel()
    is_nan = np.isnan(a)
    if is_nan.any():
        if nan_policy == "raise":
            raise ValueError("Input contains NaN and nan_policy='raise'")
        a = a[~is_nan]
    return np.histogram(a, bins=bins, range=range, density=density)


class RunningStats:
    """Single-pass mean and variance using Welford's online algorithm.

    Feed data in chunks with ``update``. Only the count, the running mean and
    the running sum of squared deviations are stored, so memory does not grow
    with the number of samples. Read the results from ``count``, ``mean``,
    ``variance()`` and ``std()``.

    Shapes
    ------
    ``update(x)``: x is ``()``, ``(b,)`` or ``(b, n_features)``.
    ``mean``, ``variance()``, ``std()``: a scalar for scalar samples, else
    ``(n_features,)``.

    Notes
    -----
    Welford's update for one new sample ``x`` is::

        n += 1;  delta = x - mean;  mean += delta / n;  m2 += delta * (x - mean)

    ``update`` uses the batch version of this formula, so a whole chunk is
    handled with NumPy instead of a Python loop. It stays accurate when the
    mean is large compared to the spread, where ``mean(x**2) - mean(x)**2``
    does not.

    Time O(b * n_features) per update of b samples; space O(n_features).

    Examples
    --------
    >>> rs = RunningStats()
    >>> _ = rs.update([1.0, 2.0]).update([3.0, 4.0])
    >>> rs.count, float(rs.mean), float(rs.variance())
    (4, 2.5, 1.25)
    """

    def __init__(self):
        self._n = 0
        self._mean = None
        self._m2 = None

    def update(self, x):
        """Absorb a batch of samples.

        Parameters
        ----------
        x : array_like, shape (), (b,) or (b, n_features)
            Axis 0 is the sample axis: a 1-D array is ``b`` single-value
            samples, a 2-D array is ``b`` samples of ``n_features`` values.

        Returns
        -------
        self : RunningStats
            So calls can be chained.

        Shapes
        ------
        ``x`` is ``()``, ``(b,)`` or ``(b, n_features)``; returns ``self``.

        Raises
        ------
        ValueError
            If ``x`` contains NaN, or its per-sample shape differs from
            earlier batches.
        TypeError
            If ``x`` is not numeric.

        Notes
        -----
        Time O(b * n_features), space O(b * n_features) for the deviations.
        An empty batch is ignored.
        """
        batch = np.atleast_1d(_as_float(x))
        if np.isnan(batch).any():
            raise ValueError("RunningStats.update received NaN; drop or impute it first")
        b = batch.shape[0]
        if b == 0:
            return self
        sample_shape = batch.shape[1:]
        if self._mean is None:
            self._mean = np.zeros(sample_shape)
            self._m2 = np.zeros(sample_shape)
        elif sample_shape != self._mean.shape:
            raise ValueError(
                f"Expected samples of shape {self._mean.shape}, got {sample_shape}"
            )

        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self._mean
        total = self._n + b
        self._m2 += batch_m2 + delta**2 * (self._n * b / total)
        self._mean += delta * (b / total)
        self._n = total
        return self

    def _state(self):
        # Returning the arrays (rather than just checking) lets type checkers
        # see that they are no longer None.
        if self._mean is None or self._m2 is None:
            raise ValueError("No samples seen yet; call update() first")
        return self._mean, self._m2

    @property
    def count(self):
        """Number of samples seen so far.

        Returns
        -------
        int
            0 before any samples are added.

        Shapes
        ------
        Output is a scalar.

        Notes
        -----
        Time and space O(1).
        """
        return self._n

    @property
    def mean(self):
        """Running mean of every sample seen so far.

        Returns
        -------
        float or ndarray
            Same as ``np.mean(all_samples, axis=0)``.

        Shapes
        ------
        Output is a scalar for scalar samples, else ``(n_features,)``.

        Raises
        ------
        ValueError
            If no samples have been seen.

        Notes
        -----
        Time and space O(n_features).
        """
        mean, _ = self._state()
        return mean.copy()[()] #turns a 0-d array into a plain float

    def variance(self, ddof=0):
        """Running variance.

        Parameters
        ----------
        ddof : int, default 0
            Divisor is ``count - ddof`` (0: population, 1: unbiased sample).

        Returns
        -------
        float or ndarray
            Same as ``np.var(all_samples, axis=0, ddof=ddof)``.

        Shapes
        ------
        Output is a scalar for scalar samples, else ``(n_features,)``.

        Raises
        ------
        ValueError
            If no samples have been seen, or ``count <= ddof``.

        Notes
        -----
        Time and space O(n_features).
        """
        _, m2 = self._state()
        if self._n - ddof <= 0:
            raise ValueError(f"Need more than ddof={ddof} samples, have {self._n}")
        return (m2 / (self._n - ddof))[()] #turns a 0-d array into a plain float

    def std(self, ddof=0):
        """Running standard deviation (the square root of ``variance``).

        Parameters
        ----------
        ddof : int, default 0
            Divisor is ``count - ddof`` (0: population, 1: unbiased sample).

        Returns
        -------
        float or ndarray
            Same as ``np.std(all_samples, axis=0, ddof=ddof)``.

        Shapes
        ------
        Output is a scalar for scalar samples, else ``(n_features,)``.

        Raises
        ------
        ValueError
            If no samples have been seen, or ``count <= ddof``.

        Notes
        -----
        Time and space O(n_features).
        """
        return np.sqrt(self.variance(ddof))
