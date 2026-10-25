"""This module is for preparing data: scalers, an imputer and a one-hot encoder.

Every class works the same way:

    fit(X)            learns from the data and returns the object itself
    transform(X)      returns a new, changed copy of the data
    fit_transform(X)  does both steps in one call

``StandardScaler`` and ``MinMaxScaler`` ignore NaN when learning and leave
NaN as it is in the output. A column that is all NaN gets neutral values
(mean 0 and scale 1, or min 0 and max 1) and stays NaN. To remove NaN,
run ``SimpleImputer`` first.

``SimpleImputer`` fills in the NaN values.

``OneHotEncoder`` does not accept NaN as a category. ``fit`` raises an
error, and at ``transform`` time a NaN is treated as an unseen category.
"""

import warnings
import numpy as np
from .utils import as_float_array

__all__ = [
    "NotFittedError",
    "StandardScaler",
    "MinMaxScaler",
    "SimpleImputer",
    "OneHotEncoder",
]


class NotFittedError(ValueError):
    """Raised when ``transform`` is called before ``fit``."""


def _check_2d(X, name="X", allow_empty=False):
    """Return ``X`` as a float 2-D array, or raise a descriptive error."""
    X = as_float_array(X, name)
    if X.ndim != 2:
        raise ValueError(
            "Expected 2-D array of shape (n_samples, n_features), "
            f"got {X.ndim}-D with shape {X.shape}"
        )
    if X.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{name} has 0 samples; at least 1 is required")
    return X


class _Transformer:
    """Shared plumbing: fitted check, feature-count check, fit_transform."""

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted; "
                "call fit() before transform()"
            )

    def _check_features(self, X):
        self._check_fitted()
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected shape (n, {self.n_features_in_}), got {X.shape}"
            )

    def fit_transform(self, X):
        """Fit to ``X`` then transform it.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, n_out_features)

        Shapes
        ------
        in: (n, d) -> out: (n, d) (``OneHotEncoder``: (n, total_categories))

        Raises
        ------
        See ``fit`` and ``transform``.

        Notes
        -----
        Cost is that of ``fit`` plus ``transform``.
        """
        return self.fit(X).transform(X)


def _nan_stat(func, X):
    """Apply a nan-reducer over axis 0 silencing all-NaN-column warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(X, axis=0)


class StandardScaler(_Transformer):
    """Z-score standardisation: ``(X - mean) / std`` per feature.
    Every column ends up centred on 0 with a standard deviation of 1.

    Attributes
    ----------
    mean_ : ndarray, shape (d,)
        Per-feature mean (0 for an all-NaN column).
    scale_ : ndarray, shape (d,)
        Population std (ddof=0). Zero-variance columns, and all-NaN
        columns, use 1.0 so a constant column maps to 0 instead of
        dividing by zero.
    n_features_in_ : int
    """

    def fit(self, X):
        """Learn the mean and standard deviation of each feature.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        self

        Shapes
        ------
        in: (n, d); ``mean_`` and ``scale_`` have shape (d,)

        Raises
        ------
        TypeError : non-numeric X.
        ValueError : X not 2-D or has no rows.

        Notes
        -----
        Time O(n*d), space O(d).
        """
        X = _check_2d(X)
        all_nan = np.isnan(X).all(axis=0)
        self.mean_ = np.where(all_nan, 0.0, _nan_stat(np.nanmean, X))
        std = np.where(all_nan, 1.0, _nan_stat(np.nanstd, X))
        self.scale_ = np.where(std == 0, 1.0, std)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Standardise ``X`` using the mean and std learned in ``fit``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            May have 0 rows.

        Returns
        -------
        ndarray, shape (n_samples, n_features)

        Shapes
        ------
        in: (n, d) -> out: (n, d)

        Raises
        ------
        NotFittedError : if called before ``fit``.
        TypeError : non-numeric X.
        ValueError : X not 2-D, or feature count differs from fit.

        Notes
        -----
        Time O(n*d), space O(n*d).
        """
        X = _check_2d(X, allow_empty=True)
        self._check_features(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        """Undo ``transform``: ``Z * scale + mean``.

        Parameters
        ----------
        Z : array_like, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, n_features)

        Shapes
        ------
        in: (n, d) -> out: (n, d)

        Raises
        ------
        NotFittedError, TypeError, ValueError : as in ``transform``.

        Notes
        -----
        Time O(n*d), space O(n*d).
        """
        Z = _check_2d(Z, "Z", allow_empty=True)
        self._check_features(Z)
        return Z * self.scale_ + self.mean_


class MinMaxScaler(_Transformer):
    """Rescale each feature to a fixed range, 0 to 1 by default.

    Parameters
    ----------
    feature_range : tuple of (float, float), default (0, 1)
        Target ``(low, high)`` with ``low < high``.

    Attributes
    ----------
    data_min_, data_max_ : ndarray, shape (d,)
        Per-feature extremes (0 and 1 for an all-NaN column).
    scale_, min_ : ndarray, shape (d,)
        ``X_out = X * scale_ + min_``. Constant columns (max == min) map
        to ``low``.
    n_features_in_ : int

    Raises
    ------
    ValueError : (at construction) if ``low >= high`` or not a pair.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        try:
            low, high = feature_range
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            raise ValueError(
                "feature_range must be a (low, high) pair, "
                f"got {feature_range!r}"
            ) from None
        if not low < high:
            raise ValueError(
                f"feature_range must satisfy low < high, got {feature_range}"
            )
        self.feature_range = (low, high)

    def fit(self, X):
        """Learn the smallest and largest value of each feature.
        NaN values are ignored.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        self

        Shapes
        ------
        in: (n, d); fitted arrays have shape (d,)

        Raises
        ------
        TypeError : non-numeric X.
        ValueError : X not 2-D or has no rows.

        Notes
        -----
        Time O(n*d), space O(d).
        """
        X = _check_2d(X)
        low, high = self.feature_range
        all_nan = np.isnan(X).all(axis=0)
        self.data_min_ = np.where(all_nan, 0.0, _nan_stat(np.nanmin, X))
        self.data_max_ = np.where(all_nan, 1.0, _nan_stat(np.nanmax, X))
        span = self.data_max_ - self.data_min_
        span = np.where(span == 0, 1.0, span)
        self.scale_ = (high - low) / span
        self.min_ = low - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Rescale ``X`` into ``feature_range``.
        Values not seen during ``fit`` may fall outside the range.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            May have 0 rows.

        Returns
        -------
        ndarray, shape (n_samples, n_features)

        Shapes
        ------
        in: (n, d) -> out: (n, d)

        Raises
        ------
        NotFittedError : if called before ``fit``.
        TypeError : non-numeric X.
        ValueError : X not 2-D, or feature count differs from fit.

        Notes
        -----
        Time O(n*d), space O(n*d).
        """
        X = _check_2d(X, allow_empty=True)
        self._check_features(X)
        return X * self.scale_ + self.min_

    def inverse_transform(self, Z):
        """Undo ``transform``: ``(Z - min_) / scale_``.

        Parameters
        ----------
        Z : array_like, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, n_features)

        Shapes
        ------
        in: (n, d) -> out: (n, d)

        Raises
        ------
        NotFittedError, TypeError, ValueError : as in ``transform``.

        Notes
        -----
        Time O(n*d), space O(n*d).
        """
        Z = _check_2d(Z, "Z", allow_empty=True)
        self._check_features(Z)
        return (Z - self.min_) / self.scale_


class SimpleImputer(_Transformer):
    """Replace NaN by a per-feature statistic or a constant.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default "mean"
    fill_value : float, default 0.0
        Used for ``"constant"``, and as the fallback for a column that is
        entirely NaN under ``"mean"`` / ``"median"``.

    Attributes
    ----------
    statistics_ : ndarray, shape (d,)
    n_features_in_ : int

    Raises
    ------
    ValueError : (at construction) for an unknown strategy.
    """

    _STRATEGIES = ("mean", "median", "constant")

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._STRATEGIES}, "
                f"got {strategy!r}"
            )
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        """Learn the replacement value for each feature.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        self

        Shapes
        ------
        in: (n, d); ``statistics_`` has shape (d,)

        Raises
        ------
        TypeError : non-numeric X.
        ValueError : X not 2-D or has no rows.

        Notes
        -----
        Time O(n*d) (mean) or O(n*d) on average (median), space O(d).
        """
        X = _check_2d(X)
        d = X.shape[1]
        if self.strategy == "constant":
            stats = np.full(d, float(self.fill_value))
        else:
            func = np.nanmean if self.strategy == "mean" else np.nanmedian
            stats = _nan_stat(func, X)
            fallback = float(self.fill_value)
            stats = np.where(np.isnan(stats), fallback, stats)
        self.statistics_ = stats
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Replace the NaN entries of ``X`` with the learned values.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            May have 0 rows.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            A new array; the input is untouched.

        Shapes
        ------
        in: (n, d) -> out: (n, d)

        Raises
        ------
        NotFittedError : if called before ``fit``.
        TypeError : non-numeric X.
        ValueError : X not 2-D, or feature count differs from fit.

        Notes
        -----
        Time O(n*d), space O(n*d).
        """
        X = _check_2d(X, allow_empty=True)
        self._check_features(X)
        return np.where(np.isnan(X), self.statistics_, X)


class OneHotEncoder(_Transformer):
    """Turn each category into its own 0/1 column (dense float output).

    Categories can be strings or numbers. Within each feature they are sorted
    (numbers numerically, strings alphabetically), and the 0/1 columns of each
    feature are placed side by side in the original column order. Numeric
    categories compare by value, so ``1`` and ``1.0`` are the same
    category. Object arrays are treated as strings. An encoder fitted on
    text rejects numbers and vice versa (``ValueError``).

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default "error"
        At transform time an unseen category raises ``ValueError``
        ("error") or becomes an all-zero block ("ignore").

    Attributes
    ----------
    categories_ : list of ndarray
        Sorted categories of each feature, in the input dtype.
    n_features_in_ : int
    n_output_features_ : int
        ``sum(len(c) for c in categories_)``.

    Raises
    ------
    ValueError : (at construction) for an unknown ``handle_unknown``.

    Notes
    -----
    Vectorised without a column loop: all values are mapped to integer ids
    by one ``np.unique`` over the whole array, then ``id + K * column``
    gives every (column, category) pair a unique integer key.
    """

    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                "handle_unknown must be 'error' or 'ignore', "
                f"got {handle_unknown!r}"
            )
        self.handle_unknown = handle_unknown

    @staticmethod
    def _prepare(X):
        """Return ``X`` as a 2-D array with at least one feature."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(
                "Expected 1-D or 2-D array of categories, "
                f"got {X.ndim}-D with shape {X.shape}"
            )
        if X.shape[1] == 0:
            raise ValueError("X has 0 features; at least 1 is required")
        if X.dtype.kind == "O":
            X = X.astype(str)
        return X

    def fit(self, X):
        """Learn the categories of each feature.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features) or (n_samples,)
            1-D input is treated as a single feature.

        Returns
        -------
        self

        Shapes
        ------
        in: (n, d) -> ``categories_`` is a list of d arrays

        Raises
        ------
        ValueError : X not 1-D/2-D, no rows or no features, or NaN present.

        Notes
        -----
        Time O(n*d*log(n*d)) (two sorts), space O(n*d).
        """
        raw = self._prepare(X)
        n, d = raw.shape
        if n == 0:
            raise ValueError("X has 0 samples; at least 1 is required")
        if raw.dtype.kind == "f" and np.isnan(raw).any():
            raise ValueError(
                "NaN is not a valid category; "
                "impute or drop missing values first"
            )
        vocab, ids = np.unique(raw.ravel(), return_inverse=True)
        k = vocab.size
        keys = ids.reshape(n, d) + k * np.arange(d)[None, :]
        uniq, first = np.unique(keys.ravel(), return_index=True)
        counts = np.bincount(uniq // k, minlength=d)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        self._vocab = vocab
        self._keys_sorted = uniq
        self.categories_ = np.split(raw.ravel()[first], offsets[1:-1])
        self.n_features_in_ = d
        self.n_output_features_ = int(uniq.size)
        return self

    def transform(self, X):
        """Encode ``X`` as one-hot blocks.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features) or (n_samples,)
            May have 0 rows.

        Returns
        -------
        ndarray of float, shape (n_samples, n_output_features_)

        Shapes
        ------
        in: (n, d) -> out: (n, sum of category counts); each block of a
        known value has exactly one 1.

        Raises
        ------
        NotFittedError : if called before ``fit``.
        ValueError : X not 1-D/2-D, wrong feature count, text/number
            mismatch with the fitted categories, or an unseen category
            when ``handle_unknown="error"`` (the message names the
            columns and shows at most 5 values).

        Notes
        -----
        Time O(n*d*log(K)) with K total categories (``searchsorted``),
        space O(n*K) for the dense output.
        """
        raw = self._prepare(X)
        self._check_features(raw)
        text_fit = self._vocab.dtype.kind in "US"
        text_new = raw.dtype.kind in "US"
        if text_fit != text_new:
            kind = "text" if text_fit else "numeric"
            raise ValueError(
                f"Encoder was fitted on {kind} categories "
                f"but got dtype {raw.dtype}"
            )
        n, d = raw.shape
        k = self._vocab.size
        flat = raw.ravel()
        p = np.minimum(np.searchsorted(self._vocab, flat), k - 1)
        in_vocab = (self._vocab[p] == flat).reshape(n, d)
        keys = p.reshape(n, d) + k * np.arange(d)[None, :]
        pos = np.minimum(np.searchsorted(self._keys_sorted, keys),
                         self._keys_sorted.size - 1)
        known = in_vocab & (self._keys_sorted[pos] == keys)
        if self.handle_unknown == "error" and not known.all():
            cols = np.unique(np.nonzero(~known)[1])
            bad = np.unique(raw[~known])
            more = f" (+{bad.size - 5} more)" if bad.size > 5 else ""
            raise ValueError(
                f"Unseen categories in column(s) {cols.tolist()}: "
                f"{bad[:5].tolist()}{more}"
            )
        out = np.zeros((n, self.n_output_features_), dtype=float)
        rows = np.broadcast_to(np.arange(n)[:, None], pos.shape)
        out[rows[known], pos[known]] = 1.0
        return out
