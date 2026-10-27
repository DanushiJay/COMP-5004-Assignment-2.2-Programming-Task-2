"""Classification and regression metrics: accuracy, precision, recall, F1, confusion matrix, MSE, ROC/AUC."""

import numpy as np


def _check_pair(y_true, y_pred):
    """Convert to 1-D arrays and validate that the lengths match."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays, got shapes {y_true.shape} and {y_pred.shape}"
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Expected shape {y_true.shape} for y_pred, got {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Inputs must not be empty")
    return y_true, y_pred


def accuracy(y_true, y_pred):
    """Fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.

    Shapes
    ------
    Both inputs ``(n_samples,)``; output is a scalar.

    Raises
    ------
    ValueError
        If the inputs are not 1-D, are empty, or have different shapes.

    Notes
    -----
    Time O(n), space O(n) for the boolean comparison.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix with true classes on rows and predicted classes on columns.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    labels : array_like, shape (n_classes,), optional
        Sorted class labels defining row/column order. Defaults to the sorted
        union of the labels in ``y_true`` and ``y_pred``.

    Returns
    -------
    ndarray of int, shape (n_classes, n_classes)
        Entry ``[i, j]`` counts samples with true class ``i`` predicted as ``j``.

    Shapes
    ------
    Inputs ``(n_samples,)``; output ``(n_classes, n_classes)``.

    Raises
    ------
    ValueError
        If the inputs are invalid (see `accuracy`) or contain labels not in ``labels``.

    Notes
    -----
    Time O(n log k) for k classes (label lookup via ``searchsorted``), space O(k^2 + n).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    n = labels.shape[0]
    t_idx = np.searchsorted(labels, y_true)
    p_idx = np.searchsorted(labels, y_pred)
    t_idx_c = np.minimum(t_idx, n - 1)
    p_idx_c = np.minimum(p_idx, n - 1)
    if np.any(labels[t_idx_c] != y_true) or np.any(labels[p_idx_c] != y_pred):
        raise ValueError("y_true/y_pred contain labels that are not in `labels`")
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (t_idx, p_idx), 1)
    return cm


def mse(y_true, y_pred):
    """Mean squared error.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth values.
    y_pred : array_like, shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)``.

    Shapes
    ------
    Both inputs ``(n_samples,)``; output is a scalar.

    Raises
    ------
    ValueError
        If the inputs are not 1-D, are empty, or have different shapes.
    TypeError
        If the inputs are not numeric.

    Notes
    -----
    Time O(n), space O(n).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    if not (np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number)):
        raise TypeError("mse requires numeric inputs")
    return float(np.mean((y_true - y_pred) ** 2))


def _prf_counts(y_true, y_pred, pos_label):
    """Return per-class (tp, fp, fn) for the classes in ``pos_label``."""
    pos = np.atleast_1d(pos_label)
    t = y_true[:, None] == pos[None, :]
    p = y_pred[:, None] == pos[None, :]
    tp = np.sum(t & p, axis=0).astype(float)
    fp = np.sum(~t & p, axis=0).astype(float)
    fn = np.sum(t & ~p, axis=0).astype(float)
    return tp, fp, fn


def _safe_divide(num, den):
    """Element-wise ``num / den`` returning 0.0 where ``den == 0``."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _resolve_classes(y_true, y_pred, average, pos_label):
    if average == "binary":
        return pos_label
    if average == "macro":
        return np.unique(np.concatenate([y_true, y_pred]))
    raise ValueError(f"average must be 'binary' or 'macro', got {average!r}")


def precision(y_true, y_pred, average="binary", pos_label=1):
    """Precision: ``TP / (TP + FP)``.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro'}, default 'binary'
        ``'binary'`` scores only ``pos_label``; ``'macro'`` is the unweighted
        mean of the per-class precisions.
    pos_label : scalar, default 1
        Positive class for ``average='binary'``.

    Returns
    -------
    float
        Precision in ``[0, 1]``. Returns 0.0 when nothing is predicted positive
        (zero division) instead of raising.

    Shapes
    ------
    Inputs ``(n_samples,)``; output is a scalar.

    Raises
    ------
    ValueError
        If the inputs are invalid (see `accuracy`) or ``average`` is unknown.

    Notes
    -----
    Time O(n k) for k scored classes, space O(n k).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    classes = _resolve_classes(y_true, y_pred, average, pos_label)
    tp, fp, _ = _prf_counts(y_true, y_pred, classes)
    return float(np.mean(_safe_divide(tp, tp + fp)))


def recall(y_true, y_pred, average="binary", pos_label=1):
    """Recall: ``TP / (TP + FN)``.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro'}, default 'binary'
        ``'binary'`` scores only ``pos_label``; ``'macro'`` is the unweighted
        mean of the per-class recalls.
    pos_label : scalar, default 1
        Positive class for ``average='binary'``.

    Returns
    -------
    float
        Recall in ``[0, 1]``. Returns 0.0 when there are no true positives
        possible (zero division) instead of raising.

    Shapes
    ------
    Inputs ``(n_samples,)``; output is a scalar.

    Raises
    ------
    ValueError
        If the inputs are invalid (see `accuracy`) or ``average`` is unknown.

    Notes
    -----
    Time O(n k) for k scored classes, space O(n k).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    classes = _resolve_classes(y_true, y_pred, average, pos_label)
    tp, _, fn = _prf_counts(y_true, y_pred, classes)
    return float(np.mean(_safe_divide(tp, tp + fn)))


def f1(y_true, y_pred, average="binary", pos_label=1):
    """F1 score: harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro'}, default 'binary'
        ``'binary'`` scores only ``pos_label``; ``'macro'`` averages the
        per-class F1 scores.
    pos_label : scalar, default 1
        Positive class for ``average='binary'``.

    Returns
    -------
    float
        F1 in ``[0, 1]``; 0.0 when precision + recall is 0.

    Shapes
    ------
    Inputs ``(n_samples,)``; output is a scalar.

    Raises
    ------
    ValueError
        If the inputs are invalid (see `accuracy`) or ``average`` is unknown.

    Notes
    -----
    Time O(n k), space O(n k).
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    classes = _resolve_classes(y_true, y_pred, average, pos_label)
    tp, fp, fn = _prf_counts(y_true, y_pred, classes)
    p = _safe_divide(tp, tp + fp)
    r = _safe_divide(tp, tp + fn)
    return float(np.mean(_safe_divide(2 * p * r, p + r)))


def roc_curve(y_true, y_score, pos_label=1):
    """ROC curve for binary classification.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels; ``pos_label`` marks the positive class.
    y_score : array_like, shape (n_samples,)
        Scores where larger means more likely positive.
    pos_label : scalar, default 1
        Label of the positive class.

    Returns
    -------
    fpr : ndarray, shape (n_thresholds,)
        False positive rates, starting at 0.0 and ending at 1.0.
    tpr : ndarray, shape (n_thresholds,)
        True positive rates, starting at 0.0 and ending at 1.0.
    thresholds : ndarray, shape (n_thresholds,)
        Decreasing score thresholds; the first is ``inf`` (nothing predicted positive).
        A sample is predicted positive when ``score >= threshold``.

    Shapes
    ------
    Inputs ``(n_samples,)``; outputs ``(n_unique_scores + 1,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid (see `accuracy`), contain NaN scores, or
        ``y_true`` lacks positive or negative samples.

    Notes
    -----
    Time O(n log n) for the sort, space O(n). Tied scores yield a single
    point, so ties never create artificial curve steps.
    """
    y_true, y_score = _check_pair(y_true, y_score)
    y_score = y_score.astype(float)
    if np.any(np.isnan(y_score)):
        raise ValueError("y_score must not contain NaN")
    pos = y_true == pos_label
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples")

    order = np.argsort(-y_score, kind="stable")
    score_sorted = y_score[order]
    pos_sorted = pos[order]
    tps = np.cumsum(pos_sorted)
    fps = np.cumsum(~pos_sorted)
    # index of the last row in each run of equal scores
    last = np.r_[np.flatnonzero(np.diff(score_sorted)), pos.size - 1]
    tpr = np.r_[0.0, tps[last] / n_pos]
    fpr = np.r_[0.0, fps[last] / n_neg]
    thresholds = np.r_[np.inf, score_sorted[last]]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under a curve using the trapezoidal rule.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        Monotonic (increasing or decreasing) x coordinates, e.g. ``fpr``.
    y : array_like, shape (n_points,)
        y coordinates, e.g. ``tpr``.

    Returns
    -------
    float
        Area under the curve (non-negative for increasing ``x``).

    Shapes
    ------
    Both inputs ``(n_points,)``; output is a scalar.

    Raises
    ------
    ValueError
        If shapes differ, there are fewer than 2 points, or ``x`` is not monotonic.

    Notes
    -----
    Time O(n), space O(n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"Expected matching 1-D shapes, got {x.shape} and {y.shape}")
    if x.size < 2:
        raise ValueError("At least 2 points are required to compute an area")
    dx = np.diff(x)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be monotonic")
    area = np.sum(dx * (y[1:] + y[:-1]) / 2.0)
    return float(abs(area))
