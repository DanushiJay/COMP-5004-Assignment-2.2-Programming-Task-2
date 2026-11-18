"""Running per-feature statistics and fixed-bin streaming histograms."""

from collections import deque

import numpy as np

from numcompute.preprocessing import _nan_stat
from numcompute.stats import histogram as _histogram
from numcompute.stats import quantile as _quantile

from .preprocessing import _chunk_mean

__all__ = ["StreamStats", "StreamingHistogram"]

_NAN_POLICIES = ("omit", "raise")


def _positive_int(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def _as_chunk(X, nan_policy):
    """Chunk as a float array of shape (n, d); a 1-D chunk is one feature."""
    X = np.asarray(X)
    if X.dtype.kind not in "biuf":
        raise TypeError(f"Expected numeric data, got dtype {X.dtype}")
    X = X.astype(float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("Expected a 1-D or 2-D chunk of shape "
                         f"(n_samples, n_features), got shape {X.shape}")
    if X.shape[1] == 0:
        raise ValueError("Chunk must contain at least one feature")
    if np.isinf(X).any():
        raise ValueError("Chunk contains infinity; use finite values or NaN")
    if nan_policy == "raise" and np.isnan(X).any():
        raise ValueError("Chunk contains NaN and nan_policy='raise'")
    return X


class StreamStats:
    """Per-feature running statistics, updated with ``update_stats``.

    Parameters
    ----------
    window : int, default 10000
        How many of the most recent rows are kept for ``quantile``. A
        true running quantile needs all the data, so quantiles are exact
        while the stream is shorter than ``window`` and describe the
        latest ``window`` rows after that. Mean, variance, minimum and
        maximum never use the window: they cover every row seen.
    nan_policy : {"omit", "raise"}, default "omit"

    Attributes
    ----------
    n_samples_seen_ : int
        Rows seen so far (NaN rows included).
    n_features_in_ : int
        Set by the first non-empty chunk.

    Shapes
    ------
    Chunk ``(n, d)`` (or ``(n,)`` for one feature). ``count``, ``mean``,
    ``minimum``, ``maximum``, ``variance()`` and ``std()`` have shape
    ``(d,)``; ``quantile(q)`` has shape ``(d,)`` for a scalar ``q`` and
    ``(n_q, d)`` for a list.

    Notes
    -----
    A feature with no non-NaN value yet has NaN statistics. Time
    O(n*d) per chunk, space O(d + window*d).
    """

    def __init__(self, window=10000, nan_policy="omit"):
        if nan_policy not in _NAN_POLICIES:
            raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, "
                             f"got {nan_policy!r}")
        self.window = _positive_int(window, "window")
        self.nan_policy = nan_policy
        self.reset()

    def reset(self):
        """Forget everything and start again. Returns ``self``."""
        self.n_samples_seen_ = 0
        self._count = None
        self._mean = None
        self._m2 = None
        self._lo = None
        self._hi = None
        self._rows = None
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
        return self

    # Updating statistics
    def update_stats(self, X_chunk):
        """Add one chunk of rows to the running statistics.

        Parameters
        ----------
        X_chunk : array_like, shape (n, d) or (n,)
            May have 0 rows (then nothing changes). After the first
            chunk, ``d`` must stay the same.

        Returns
        -------
        self

        Raises
        ------
        TypeError : non-numeric chunk.
        ValueError : bad shape, a different number of features than
            before, infinity, or NaN when ``nan_policy="raise"``.
        """
        X = _as_chunk(X_chunk, self.nan_policy)
        if hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, "
                                 f"got {X.shape[1]}")
        if X.shape[0] == 0:
            return self
        d = X.shape[1]
        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = d
            self._count = np.zeros(d, dtype=np.int64)
            self._mean = np.zeros(d)
            self._m2 = np.zeros(d)
            self._lo = np.full(d, np.nan)
            self._hi = np.full(d, np.nan)
            self._rows = np.empty((0, d))

        nb = (~np.isnan(X)).sum(axis=0)
        mb = _chunk_mean(X)
        with np.errstate(over="ignore", invalid="ignore"):
            m2b = np.nansum((X - mb) ** 2, axis=0)
        na = self._count
        n = na + nb
        frac = nb / np.maximum(n, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = mb - self._mean
            merged_mean = self._mean + delta * frac
            merged_m2 = self._m2 + m2b + delta ** 2 * na * frac
        overflow = (~np.isfinite(merged_mean)) & (na > 0) & (nb > 0)
        if overflow.any():
            weight_a = na[overflow] / n[overflow]
            weight_b = nb[overflow] / n[overflow]
            merged_mean[overflow] = (
                self._mean[overflow] * weight_a + mb[overflow] * weight_b)
        self._mean = np.where(na == 0, mb, merged_mean)
        self._m2 = np.where(na == 0, m2b, merged_m2)
        self._count = n
        self._lo = np.fmin(self._lo, _nan_stat(np.nanmin, X))
        self._hi = np.fmax(self._hi, _nan_stat(np.nanmax, X))
        self._rows = np.vstack([self._rows, X])[-self.window:]
        self.n_samples_seen_ += X.shape[0]
        return self

    update = update_stats

    # Results
    def _check_seen(self):
        if not hasattr(self, "n_features_in_"):
            raise ValueError("No samples seen yet; call update_stats() "
                             "first")

    @property
    def count(self):
        """Non-NaN values seen per feature, shape (d,)."""
        self._check_seen()
        return self._count.copy()

    @property
    def mean(self):
        """Mean per feature (NaN where nothing was seen), shape (d,)."""
        self._check_seen()
        return np.where(self._count > 0, self._mean, np.nan)

    def variance(self, ddof=0):
        """Variance per feature; NaN where ``count <= ddof``.

        ``ddof=0`` is the population value, ``ddof=1`` the unbiased one.
        """
        self._check_seen()
        if (isinstance(ddof, (bool, np.bool_))
                or not isinstance(ddof, (int, np.integer)) or ddof < 0):
            raise ValueError(f"ddof must be an int >= 0, got {ddof!r}")
        denom = self._count - ddof
        var = np.maximum(self._m2, 0.0) / np.maximum(denom, 1)
        return np.where(denom > 0, var, np.nan)

    def std(self, ddof=0):
        """Standard deviation per feature (square root of ``variance``)."""
        return np.sqrt(self.variance(ddof))

    @property
    def minimum(self):
        """Smallest value per feature (NaN if none yet), shape (d,)."""
        self._check_seen()
        return self._lo.copy()

    @property
    def maximum(self):
        """Largest value per feature (NaN if none yet), shape (d,)."""
        self._check_seen()
        return self._hi.copy()

    def quantile(self, q):
        """Quantile(s) of the retained window, per feature.

        Parameters
        ----------
        q : float or array_like of float in [0, 1]

        Returns
        -------
        ndarray, shape (d,) for scalar ``q``, else (n_q, d)
            Exact while no more than ``window`` rows have been seen.
        """
        self._check_seen()
        return _quantile(self._rows, q, axis=0, nan_policy="omit")

    def summary(self):
        """All the running statistics as a dict of arrays, for logging."""
        return {
            "n_samples": self.n_samples_seen_,
            "count": self.count,
            "mean": self.mean,
            "std": self.std(),
            "min": self.minimum,
            "max": self.maximum,
        }


class StreamingHistogram:
    """Histogram with fixed bins that grows chunk by chunk.

    Parameters
    ----------
    bins : int, default 10
        Number of equal-width bins.
    range : (float, float) or None
        Lower and upper edge. If ``None``, the first non-empty chunk's
        minimum and maximum are used and then fixed, so later values
        outside it are counted in ``outside_``, not in a bin.
    window : int or None, default None
        ``None`` counts every chunk so far. An int keeps only the most
        recent ``window`` chunks (a sliding window).

    Attributes
    ----------
    edges : ndarray, shape (bins + 1,) or None before the first chunk.
    counts : ndarray of int, shape (bins,)
    outside_ : int
        Values (in the counted chunks) that fell outside the edges.

    Notes
    -----
    NaN values are ignored. Infinity is rejected. Time O(n) per chunk.
    """

    def __init__(self, bins=10, range=None, window=None):
        self.bins = _positive_int(bins, "bins")
        if range is not None:
            lo, hi = range
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError("range must be a finite (low, high) pair "
                                 f"with low < high, got {range!r}")
            range = (float(lo), float(hi))
        self.range = range
        self.window = None if window is None else _positive_int(window,
                                                                "window")
        self.edges = (None if range is None
                      else np.linspace(range[0], range[1], self.bins + 1))
        self.reset()

    def reset(self):
        """Clear counts, and re-estimate edges if no range was specified."""
        if self.range is None:
            self.edges = None
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.outside_ = 0
        self._chunks = deque()
        return self

    def update(self, x):
        """Add the values in ``x`` (any shape, flattened). Returns self."""
        a = np.asarray(x)
        if a.dtype.kind not in "biuf":
            raise TypeError(f"Expected numeric data, got dtype {a.dtype}")
        a = a.astype(float).ravel()
        if np.isinf(a).any():
            raise ValueError("x contains infinity")
        a = a[~np.isnan(a)]
        if a.size == 0:
            return self
        if self.edges is None:
            lo, hi = float(a.min()), float(a.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            self.edges = np.linspace(lo, hi, self.bins + 1)
        counts, _ = _histogram(a, bins=self.edges)
        outside = int(a.size - counts.sum())
        self.counts = self.counts + counts
        self.outside_ += outside
        if self.window is not None:
            self._chunks.append((counts, outside))
            if len(self._chunks) > self.window:
                old, old_out = self._chunks.popleft()
                self.counts = self.counts - old
                self.outside_ -= old_out
        return self

    update_stats = update

    @property
    def density(self):
        """Counts as a probability density (integrates to 1), or zeros."""
        total = self.counts.sum()
        if self.edges is None or total == 0:
            return np.zeros(self.bins)
        return self.counts / (total * np.diff(self.edges))
