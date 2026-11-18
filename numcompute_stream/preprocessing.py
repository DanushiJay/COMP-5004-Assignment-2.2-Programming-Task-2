"""Incremental preprocessors built on the original NumCompute classes.

Each class supports fit(X), partial_fit(X_chunk), and transform(X).
Numeric preprocessors learn from non-NaN observations and reject
infinity; the encoder learns new categories as chunks arrive. fit()
resets previous state.
"""

import numpy as np

from numcompute.preprocessing import (
    NotFittedError,
    StandardScaler as _StandardScaler,
    MinMaxScaler as _MinMaxScaler,
    SimpleImputer as _SimpleImputer,
    OneHotEncoder as _OneHotEncoder,
    _check_2d,
    _nan_stat,
)

__all__ = [
    "NotFittedError", "StandardScaler", "MinMaxScaler",
    "SimpleImputer", "OneHotEncoder",
]


def _chunk_mean(X):
    """Per-column means, avoiding overflow from summing large values.

    Empty (all-NaN) columns return zero; the caller tracks valid counts.
    Time O(n*d); temporary space O(n*d).
    """
    valid = ~np.isnan(X)
    count = valid.sum(axis=0)
    cols = np.arange(X.shape[1])
    anchor = X[np.argmax(valid, axis=0), cols]
    anchor = np.where(count > 0, anchor, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.where(valid, X - anchor, 0.0)
        mean = anchor + (delta / np.maximum(count, 1)).sum(axis=0)

    # Divide first when subtracting large values would overflow.
    overflow = (~np.isfinite(delta)).any(axis=0)
    if overflow.any():
        x = X[:, overflow]
        v = valid[:, overflow]
        c = np.maximum(count[overflow], 1)
        mean[overflow] = np.where(v, x / c, 0.0).sum(axis=0)

    return np.where(count > 0, mean, 0.0)


def _stable_nanmedian(X):
    """Per-column median without overflow when averaging two large values.

    All-NaN columns return NaN. Time O(d*n*log(n)); space O(n*d).
    """
    ordered = np.sort(X, axis=0)  # NaNs sort to the end.
    count = (~np.isnan(X)).sum(axis=0)
    col = np.arange(X.shape[1])
    low = ordered[np.maximum((count - 1) // 2, 0), col]
    high = ordered[np.maximum(count // 2, 0), col]
    result = high.copy()
    even = (count > 0) & (count % 2 == 0)
    if even.any():
        a, b = low[even], high[even]
        midpoint = np.empty_like(a)
        same_sign = np.signbit(a) == np.signbit(b)
        gap = (b[same_sign] - a[same_sign]) / 2.0
        midpoint[same_sign] = a[same_sign] + gap
        midpoint[~same_sign] = a[~same_sign] / 2.0 + b[~same_sign] / 2.0
        result[even] = midpoint
    return np.where(count > 0, result, np.nan)


class _Streaming:
    """Chunk validation and fit/partial_fit shared by numeric transformers."""

    def _validated_chunk(self, X, allow_empty):
        X = _check_2d(X, allow_empty=allow_empty)
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if np.isinf(X).any():
            raise ValueError("X contains infinity; use finite values or NaN")
        return X

    def fit(self, X):
        """Reset previous state and fit one dataset of shape (n, d).

        Returns self. Raises TypeError/ValueError for invalid input.
        The update costs O(n*d) apart from the median's sort.
        """
        X = self._validated_chunk(X, allow_empty=False)
        self._reset(X.shape[1])
        self._update(X, from_fit=True)
        return self

    def partial_fit(self, X):
        """Update estimates with a numeric chunk of shape (n, d).

        Empty chunks have no effect. Feature width must remain fixed.
        Returns self; raises TypeError or ValueError for invalid input.
        """
        X = self._validated_chunk(X, allow_empty=True)
        if hasattr(self, "n_features_in_"):
            self._check_features(X)
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "n_features_in_"):
            self._reset(X.shape[1])
        self._update(X, from_fit=False)
        return self


class StandardScaler(_Streaming, _StandardScaler):
    """Streaming z-score scaler, ignoring NaNs in per-feature statistics.

    n_samples_seen_, mean_ and scale_ each have shape (n_features,).
    Each chunk costs O(n*d) time and O(n*d) temporary space.
    """

    def _reset(self, d):
        self.n_features_in_ = d
        self.n_samples_seen_ = np.zeros(d, dtype=np.int64)
        self._mean = np.zeros(d, dtype=float)
        self._m2 = np.zeros(d, dtype=float)
        self._refresh()

    def _refresh(self):
        n = self.n_samples_seen_
        std = np.sqrt(np.maximum(self._m2, 0.0) / np.maximum(n, 1))
        self.mean_ = np.where(n > 0, self._mean, 0.0)
        self.scale_ = np.where((n > 0) & (std > 0), std, 1.0)

    def _update(self, X, from_fit):
        nb = (~np.isnan(X)).sum(axis=0)
        mb = _chunk_mean(X)
        with np.errstate(over="ignore", invalid="ignore"):
            m2b = np.nansum((X - mb) ** 2, axis=0)
        na = self.n_samples_seen_
        n = na + nb
        fraction = nb / np.maximum(n, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = mb - self._mean
            new_mean = self._mean + delta * fraction
            new_m2 = self._m2 + m2b + delta**2 * na * fraction
        self._mean = np.where(na == 0, mb, new_mean)
        self._m2 = np.where(na == 0, m2b, new_m2)
        self.n_samples_seen_ = n
        self._refresh()


class MinMaxScaler(_Streaming, _MinMaxScaler):
    """Incremental min/max scaler with the original feature_range API.

    Per-chunk time O(n*d), state O(d). A new extreme changes the scale,
    so earlier transformed rows may need to be transformed again.
    """

    def _reset(self, d):
        self.n_features_in_ = d
        self._lo = np.full(d, np.nan)
        self._hi = np.full(d, np.nan)
        self._refresh()

    def _refresh(self):
        low, high = self.feature_range
        seen = ~np.isnan(self._lo)
        self.data_min_ = np.where(seen, self._lo, 0.0)
        self.data_max_ = np.where(seen, self._hi, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            span = self.data_max_ - self.data_min_
        self._wide_range = ~np.isfinite(span)
        span = np.where(span == 0, 1.0, span)
        self.scale_ = (high - low) / span
        self.min_ = low - self.data_min_ * self.scale_

    def _update(self, X, from_fit):
        self._lo = np.fmin(self._lo, _nan_stat(np.nanmin, X))
        self._hi = np.fmax(self._hi, _nan_stat(np.nanmax, X))
        self._refresh()

    def transform(self, X):
        """Map X (n, d) to feature_range, preserving NaNs.

        Handles feature ranges whose subtraction exceeds float64 limits.
        Raises NotFittedError/ValueError/TypeError for invalid input.
        Time and output space O(n*d).
        """
        X = self._validated_chunk(X, allow_empty=True)
        self._check_features(X)
        out = X * self.scale_ + self.min_
        if self._wide_range.any():
            mask = self._wide_range
            lo, hi = self.feature_range
            a, b = self.data_min_[mask], self.data_max_[mask]
            t = (X[:, mask] / 2.0 - a / 2.0) / (b / 2.0 - a / 2.0)
            out[:, mask] = lo + (hi - lo) * t
        return out

    def inverse_transform(self, Z):
        """Undo transform for Z with shape (n, d); return shape (n, d)."""
        Z = self._validated_chunk(Z, allow_empty=True)
        self._check_features(Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (Z - self.min_) / self.scale_
        if self._wide_range.any():
            mask = self._wide_range
            lo, hi = self.feature_range
            t = (Z[:, mask] - lo) / (hi - lo)
            a, b = self.data_min_[mask], self.data_max_[mask]
            out[:, mask] = 2.0 * (a / 2.0 + t * (b / 2.0 - a / 2.0))
        return out


class SimpleImputer(_Streaming, _SimpleImputer):
    """Incremental imputer with mean, median-window and constant options.

    Mean is updated without retaining rows. Median retains up to `window`
    recent rows when using partial_fit; fit uses all supplied rows.
    Statistics have shape (n_features,). Mean: O(n*d) per chunk and O(d)
    state; median: O(window*d*log(window)) for recomputing the median.
    """

    def __init__(self, strategy="mean", fill_value=0.0, window=10000):
        super().__init__(strategy, fill_value)
        if (isinstance(window, (bool, np.bool_)) or
                not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError(f"window must be an int >= 1, got {window!r}")
        self.window = int(window)

    def _reset(self, d):
        self.n_features_in_ = d
        self._count = np.zeros(d, dtype=np.int64)
        self._mean = np.zeros(d, dtype=float)
        self._rows = np.empty((0, d))
        self._refresh()

    def _refresh(self):
        fallback = float(self.fill_value)
        d = self.n_features_in_
        if self.strategy == "constant":
            stats = np.full(d, fallback)
        elif self.strategy == "mean":
            stats = np.where(self._count > 0, self._mean, fallback)
        elif self._rows.shape[0]:
            stats = _stable_nanmedian(self._rows)
        else:
            stats = np.full(d, fallback)
        self.statistics_ = np.where(np.isnan(stats), fallback, stats)

    def _update(self, X, from_fit):
        if self.strategy == "mean":
            nb = (~np.isnan(X)).sum(axis=0)
            mb = _chunk_mean(X)
            na = self._count
            fraction = nb / np.maximum(na + nb, 1)
            with np.errstate(over="ignore", invalid="ignore"):
                delta = mb - self._mean
                merged = self._mean + delta * fraction
            # Deal with true finite means of large opposite-sign chunks.
            overflow = ~np.isfinite(merged) & (na > 0) & (nb > 0)
            if overflow.any():
                a = na[overflow] / (na[overflow] + nb[overflow])
                b = nb[overflow] / (na[overflow] + nb[overflow])
                merged[overflow] = self._mean[overflow] * a + mb[overflow] * b
            self._mean = np.where(na > 0, merged, mb)
            self._count = na + nb
        elif self.strategy == "median":
            rows = np.vstack((self._rows, X))
            self._rows = rows if from_fit else rows[-self.window:].copy()
        self._refresh()


class OneHotEncoder(_OneHotEncoder):
    """One-hot encoder that learns additional categories from new chunks.

    The feature count and even existing column positions may change when
    a new category appears. A fitted tree must use a matching encoder
    snapshot or be retrained with the new feature representation.
    """

    def partial_fit(self, X):
        """Add categories from X (n, d), returning self.

        An empty chunk does nothing. Raises ValueError for wrong feature
        width, NaNs, or mixing text/numeric categories. Updating requires
        O(d*m*log(m)) time for m categories and O(m) retained state.
        """
        raw = self._prepare(X)
        if raw.shape[0] == 0:
            if hasattr(self, "n_features_in_"):
                self._check_features(raw)
            return self
        if not hasattr(self, "n_features_in_"):
            return self.fit(raw)
        self._check_features(raw)
        if (raw.dtype.kind in "US") != (self._vocab.dtype.kind in "US"):
            kind = "text" if self._vocab.dtype.kind in "US" else "numeric"
            raise ValueError(f"Encoder was fitted on {kind} categories "
                             f"but got dtype {raw.dtype}")
        if raw.dtype.kind == "f" and np.isnan(raw).any():
            raise ValueError("NaN is not a valid category; "
                             "impute missing values first")
        merged = [np.union1d(c, raw[:, j])
                  for j, c in enumerate(self.categories_)]
        sizes = [len(c) for c in merged]
        flat = np.concatenate(merged)
        vocab = np.unique(flat)
        k = vocab.size
        cols = np.repeat(np.arange(len(merged)), sizes)
        self._vocab = vocab
        self._keys_sorted = np.searchsorted(vocab, flat) + k * cols
        self.categories_ = merged
        self.n_output_features_ = int(flat.size)
        return self
