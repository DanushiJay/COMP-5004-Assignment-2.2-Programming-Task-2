"""Streaming classification metrics with cumulative and rolling updates."""

import numpy as np

from numcompute.metrics import _safe_divide
from numcompute.metrics import auc as _auc
from numcompute.metrics import confusion_matrix as _confusion_matrix
from numcompute.metrics import roc_curve as _roc_curve

__all__ = [
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingAUC",
]

_NO_DATA = "No samples seen yet; call update() first"


def _as_pair(y_true, y_pred):
    """Two 1-D arrays of the same length (a length of 0 is allowed)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays, got shapes {y_true.shape} and "
            f"{y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Expected shape {y_true.shape} for y_pred, got "
                         f"{y_pred.shape}")
    return y_true, y_pred


def _window(window):
    if window is None:
        return None
    if (isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer)) or window < 1):
        raise ValueError(f"window must be None or an int >= 1, got "
                         f"{window!r}")
    return int(window)


class StreamingConfusionMatrix:
    """Confusion matrix accumulated over chunks.

    Parameters
    ----------
    labels : array_like or None
        Fixed class labels. If ``None``, labels are collected from the
        stream and the matrix grows (rows and columns stay sorted). With
        fixed labels, an unknown label raises ``ValueError``.
    window : int or None
        ``None`` counts every sample. An int uses only the most recent
        ``window`` samples.

    Attributes
    ----------
    labels_ : ndarray or None
        Sorted labels so far (``None`` before any data).
    n_samples_seen_ : int

    Shapes
    ------
    ``update`` takes two 1-D arrays of equal length. ``result()`` returns
    an int array of shape ``(n_classes, n_classes)``: entry ``[i, j]``
    counts true class ``i`` predicted as ``j``.
    """

    def __init__(self, labels=None, window=None):
        self._fixed = None
        if labels is not None:
            self._fixed = np.unique(np.asarray(labels))
        self.window = _window(window)
        self.reset()

    def reset(self):
        """Forget everything and start again. Returns ``self``."""
        self.labels_ = self._fixed
        k = 0 if self._fixed is None else self._fixed.size
        self._matrix = np.zeros((k, k), dtype=np.int64)
        self.n_samples_seen_ = 0
        self._yt = None
        self._yp = None
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of true and predicted labels. Returns ``self``."""
        yt, yp = _as_pair(y_true_chunk, y_pred_chunk)
        for values in (yt, yp):
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                raise ValueError("Labels must not contain NaN or infinity")
        if yt.size == 0:
            return self
        if self._fixed is None:
            seen = np.union1d(yt, yp)
            new = seen if self.labels_ is None else np.union1d(self.labels_,
                                                               seen)
            if self.labels_ is None or new.size != self.labels_.size:
                grown = np.zeros((new.size, new.size), dtype=np.int64)
                if self.labels_ is not None:
                    idx = np.searchsorted(new, self.labels_)
                    grown[np.ix_(idx, idx)] = self._matrix
                self._matrix = grown
                self.labels_ = new
        self._matrix += _confusion_matrix(yt, yp, labels=self.labels_)
        self.n_samples_seen_ += yt.size
        if self.window is not None:
            if self._yt is None:
                self._yt, self._yp = yt.copy(), yp.copy()
            else:
                self._yt = np.concatenate([self._yt, yt])
                self._yp = np.concatenate([self._yp, yp])
            self._yt = self._yt[-self.window:]
            self._yp = self._yp[-self.window:]
        return self

    def matrix(self):
        """Counts over the whole stream, or over the window if set."""
        if self.labels_ is None:
            raise ValueError(_NO_DATA)
        if self.window is None:
            return self._matrix.copy()
        if self._yt is None:  # fixed labels, nothing recorded yet
            return np.zeros_like(self._matrix)
        return _confusion_matrix(self._yt, self._yp, labels=self.labels_)

    def result(self):
        """The confusion matrix, shape ``(n_classes, n_classes)``."""
        return self.matrix()


class StreamingAccuracy(StreamingConfusionMatrix):
    """Fraction of correct predictions so far (or in the window)."""

    def result(self):
        """Return accuracy, raising an error when no samples were recorded."""
        cm = self.matrix()
        total = cm.sum()
        if total == 0:
            raise ValueError(_NO_DATA)
        return float(np.trace(cm) / total)


class _PrecisionRecallBase(StreamingConfusionMatrix):
    """Shared part of precision, recall and F1."""

    def __init__(self, average="binary", pos_label=1, labels=None,
                 window=None):
        if average not in ("binary", "macro"):
            raise ValueError(f"average must be 'binary' or 'macro', got "
                             f"{average!r}")
        self.average = average
        self.pos_label = pos_label
        super().__init__(labels=labels, window=window)

    def _counts(self):
        """Per-class (tp, fp, fn) for the classes being scored."""
        cm = self.matrix().astype(float)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        if self.average == "macro" and self.window is not None \
                and self._fixed is None:
            active = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
            tp, fp, fn = tp[active], fp[active], fn[active]
        if self.average == "binary":
            pick = self.labels_ == self.pos_label
            if not pick.any():
                zero = np.zeros(1)
                return zero, zero, zero
            tp, fp, fn = tp[pick], fp[pick], fn[pick]
        return tp, fp, fn


class StreamingPrecision(_PrecisionRecallBase):
    """Precision ``TP / (TP + FP)``; 0.0 when nothing is predicted."""

    def result(self):
        tp, fp, _ = self._counts()
        return float(np.mean(_safe_divide(tp, tp + fp)))


class StreamingRecall(_PrecisionRecallBase):
    """Recall ``TP / (TP + FN)``; 0.0 when there are no true positives."""

    def result(self):
        tp, _, fn = self._counts()
        return float(np.mean(_safe_divide(tp, tp + fn)))


class StreamingF1(_PrecisionRecallBase):
    """F1, the harmonic mean of precision and recall."""

    def result(self):
        tp, fp, fn = self._counts()
        p = _safe_divide(tp, tp + fp)
        r = _safe_divide(tp, tp + fn)
        return float(np.mean(_safe_divide(2 * p * r, p + r)))


class StreamingAUC:
    """Area under the ROC curve, accumulated over chunks.

    Without a window, scores are put into ``n_bins`` equal bins and only
    the positive and negative counts per bin are kept, so memory stays
    constant. Binning approximates the AUC: if many different scores fall
    into the same bin, its error can be substantial. Scores outside
    ``score_range`` go in the first or last bin. With a window, the last
    ``window`` samples are kept and the AUC is exact for them.

    Parameters
    ----------
    pos_label : scalar, default 1
    n_bins : int, default 1000
    score_range : (float, float), default (0.0, 1.0)
        Fits probabilities; widen it for other scores.
    window : int or None

    Notes
    -----
    ``result()`` is NaN while only one class has been seen, which is
    common in the first chunks, so a running log does not crash.
    Non-finite scores raise ``ValueError``.
    """

    def __init__(self, pos_label=1, n_bins=1000, score_range=(0.0, 1.0),
                 window=None):
        if (isinstance(n_bins, (bool, np.bool_))
                or not isinstance(n_bins, (int, np.integer)) or n_bins < 2):
            raise ValueError(f"n_bins must be an int >= 2, got {n_bins!r}")
        lo, hi = score_range
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("score_range must be a finite (low, high) "
                             f"pair with low < high, got {score_range!r}")
        self.pos_label = pos_label
        self.n_bins = int(n_bins)
        self.score_range = (float(lo), float(hi))
        self.window = _window(window)
        self.reset()

    def reset(self):
        """Forget everything and start again. Returns ``self``."""
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._is_pos = None
        self._scores = None
        self.n_samples_seen_ = 0
        return self

    def update(self, y_true_chunk, y_score_chunk):
        """Add one chunk of true labels and scores. Returns ``self``."""
        y, s = _as_pair(y_true_chunk, y_score_chunk)
        if y.dtype.kind == "f" and not np.isfinite(y).all():
            raise ValueError("y_true must not contain NaN or infinity")
        s = s.astype(float)
        if not np.isfinite(s).all():
            raise ValueError("y_score must contain only finite values "
                             "(no NaN or infinity)")
        if y.size == 0:
            return self
        pos = y == self.pos_label
        self.n_samples_seen_ += y.size
        if self.window is not None:
            if self._is_pos is None:
                self._is_pos, self._scores = pos.copy(), s.copy()
            else:
                self._is_pos = np.concatenate([self._is_pos, pos])
                self._scores = np.concatenate([self._scores, s])
            self._is_pos = self._is_pos[-self.window:]
            self._scores = self._scores[-self.window:]
            return self
        lo, hi = self.score_range
        clipped = np.clip(s, lo, hi)
        if np.isfinite(hi - lo):
            ratio = (clipped - lo) / (hi - lo)
        else:
            ratio = (clipped / 2.0 - lo / 2.0) / (hi / 2.0 - lo / 2.0)
        bins = np.minimum((ratio * self.n_bins).astype(int), self.n_bins - 1)
        self._pos += np.bincount(bins[pos], minlength=self.n_bins)
        self._neg += np.bincount(bins[~pos], minlength=self.n_bins)
        return self

    def _counts(self):
        if self.n_samples_seen_ == 0:
            raise ValueError(_NO_DATA)
        if self.window is not None:
            return int(self._is_pos.sum()), int((~self._is_pos).sum())
        return int(self._pos.sum()), int(self._neg.sum())

    def roc_curve(self):
        """Return ``(fpr, tpr)`` for the stream so far.

        Raises ``ValueError`` if the positive or negative class is missing.
        """
        n_pos, n_neg = self._counts()
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Need both positive and negative samples to "
                             "build a ROC curve")
        if self.window is not None:
            y = np.where(self._is_pos, 1, 0)
            fpr, tpr, _ = _roc_curve(y, self._scores, pos_label=1)
            return fpr, tpr
        tps = np.cumsum(self._pos[::-1])
        fps = np.cumsum(self._neg[::-1])
        return np.r_[0.0, fps / n_neg], np.r_[0.0, tps / n_pos]

    def result(self):
        """AUC so far, or NaN if one of the classes has not appeared."""
        n_pos, n_neg = self._counts()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        fpr, tpr = self.roc_curve()
        return _auc(fpr, tpr)
