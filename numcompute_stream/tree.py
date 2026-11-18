"""Decision tree classifier for batch and buffered streaming learning.

``fit`` trains a new tree. ``partial_fit`` buffers incoming chunks and
rebuilds the tree when the buffer reaches ``buffer_size``. Predictions
use the most recent fitted tree until the next retraining event.
"""

import numpy as np

from numcompute.preprocessing import NotFittedError

__all__ = ["DecisionTreeClassifier"]

_LEAF = -1


def _impurity(counts, criterion):
    """Return Gini impurity or entropy for each row of class counts.

    ``counts`` has shape (n_nodes, n_classes). Empty rows return zero.
    """
    totals = counts.sum(axis=1, keepdims=True)
    p = counts / np.maximum(totals, 1)
    if criterion == "gini":
        result = 1.0 - (p ** 2).sum(axis=1)
    else:
        safe = np.where(p > 0, p, 1.0)
        result = -(p * np.log2(safe)).sum(axis=1)
    return np.where(totals[:, 0] == 0, 0.0, result)


def _best_split(X, onehot, features, criterion):
    """Find the lowest-impurity split among the candidate features.

    Cumulative class counts let us score thresholds without a loop over
    individual samples. Ties follow feature order, then threshold order.
    Returns (feature, threshold, impurity), or (None, None, inf) if no
    valid split exists.
    """
    n = X.shape[0]
    total = onehot.sum(axis=0)
    n_left = np.arange(1, n)[:, None]
    n_right = n - n_left
    best = (None, None, np.inf)
    for j in features:
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        valid = xs[1:] != xs[:-1]
        if not valid.any():
            continue
        left = np.cumsum(onehot[order], axis=0)[:-1]
        right = total - left
        score = (n_left[:, 0] * _impurity(left, criterion)
                 + n_right[:, 0] * _impurity(right, criterion)) / n
        score[~valid] = np.inf
        i = int(np.argmin(score))
        if score[i] < best[2]:
            lo, hi = xs[i], xs[i + 1]
            # Avoid overflow; if rounding reaches hi, use lo instead.
            thr = lo / 2.0 + hi / 2.0
            if not lo <= thr < hi:
                thr = lo
            best = (int(j), float(thr), float(score[i]))
    return best


class DecisionTreeClassifier:
    """Depth-limited decision tree with buffered retraining.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth of the tree (the root has depth 0).
    min_samples_split : int, default 2
        A node with fewer samples than this becomes a leaf.
    max_features : None, "sqrt" or int, default None
        How many features to consider at each split. ``None`` uses all.
    criterion : {"gini", "entropy"}, default "gini"
    buffer_size : int, default 500
        Number of buffered samples that triggers a retrain in
        ``partial_fit``.
    random_state : int or None
        Seed for choosing features when ``max_features`` is set.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels seen in the last training data.
    n_features_in_ : int
    n_retrains_ : int
        How many times the tree has been trained.
    n_buffered_ : int
        Samples waiting in the buffer.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 criterion="gini", buffer_size=500, random_state=None):
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
            raise ValueError(f"max_depth must be an int >= 0, got "
                             f"{max_depth!r}")
        if (not isinstance(min_samples_split, (int, np.integer))
                or min_samples_split < 2):
            raise ValueError(f"min_samples_split must be an int >= 2, got "
                             f"{min_samples_split!r}")
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got "
                             f"{criterion!r}")
        if not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
            raise ValueError(f"buffer_size must be an int >= 1, got "
                             f"{buffer_size!r}")
        if not (max_features is None or max_features == "sqrt"
                or (isinstance(max_features, (int, np.integer))
                    and max_features >= 1)):
            raise ValueError("max_features must be None, 'sqrt' or an "
                             f"int >= 1, got {max_features!r}")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = int(buffer_size)
        self.random_state = random_state
        self.n_retrains_ = 0
        self._tree = None
        self._buf_X = []
        self._buf_y = []

    # Input validation
    @staticmethod
    def _check_X(X):
        X = np.asarray(X)
        if X.dtype.kind not in "biuf":
            raise TypeError(f"X must be numeric, got dtype {X.dtype}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got "
                             f"shape {X.shape}")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        # Larger integers can become indistinguishable when cast to float64.
        if X.dtype.kind in "iu":
            if (np.any(X > 2**53)
                    or (X.dtype.kind == "i" and np.any(X < -(2**53)))):
                raise ValueError("X has integers outside the exactly "
                                 "representable float64 range")
        X = X.astype(float)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity; impute or remove "
                             "them first (e.g. with SimpleImputer)")
        return X

    @classmethod
    def _check_Xy(cls, X, y):
        X = cls._check_X(X)
        y = np.array(y, copy=True)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has "
                             f"{y.shape[0]}")
        if y.dtype.kind in "f" and not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinity")
        return X, y

    @property
    def n_buffered_(self):
        return int(sum(len(c) for c in self._buf_y))

    # Training
    def fit(self, X, y):
        """Fit a new tree to X (n_samples, n_features) and y (n_samples,).

        Returns self. Raises ValueError for invalid shapes or empty data.
        Pending streaming samples, if any, are left in the buffer.
        Split search costs O(d * n log n) per depth level in the worst case.
        """
        X, y = self._check_Xy(X, y)
        if X.shape[0] == 0:
            raise ValueError("cannot fit on zero samples")
        classes, y_idx = np.unique(y, return_inverse=True)
        onehot = np.eye(len(classes))[y_idx]
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        if self.max_features is None:
            k = d
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(d)))
        else:
            k = min(int(self.max_features), d)

        feat, thr, left, right, value = [], [], [], [], []

        def new_node(rows):
            feat.append(_LEAF)
            thr.append(0.0)
            left.append(_LEAF)
            right.append(_LEAF)
            value.append(onehot[rows].sum(axis=0))
            return len(feat) - 1

        stack = [(new_node(np.arange(X.shape[0])),
                  np.arange(X.shape[0]), 0)]
        while stack:
            node, rows, depth = stack.pop()
            counts = value[node]
            if (depth >= self.max_depth
                    or len(rows) < self.min_samples_split
                    or (counts > 0).sum() <= 1):
                continue
            feats = (np.arange(d) if k == d
                     else np.sort(rng.choice(d, size=k, replace=False)))
            j, t, score = _best_split(X[rows], onehot[rows], feats,
                                      self.criterion)
            parent = _impurity(counts[None, :], self.criterion)[0]
            if j is None or score >= parent - 1e-12:
                continue
            go_left = X[rows, j] <= t
            feat[node], thr[node] = j, t
            left[node] = new_node(rows[go_left])
            right[node] = new_node(rows[~go_left])
            stack.append((left[node], rows[go_left], depth + 1))
            stack.append((right[node], rows[~go_left], depth + 1))

        self._tree = {
            "feature": np.array(feat, dtype=int),
            "threshold": np.array(thr, dtype=float),
            "left": np.array(left, dtype=int),
            "right": np.array(right, dtype=int),
            "value": np.array(value, dtype=float),
        }
        self.classes_ = classes
        self.n_features_in_ = d
        self.n_retrains_ += 1
        return self

    # Buffered training
    def partial_fit(self, X_chunk, y_chunk):
        """Buffer samples and rebuild the tree when the threshold is met.

        X_chunk is 2-D and y_chunk is 1-D. Returns self. An empty chunk
        has no effect. Retraining uses only the current buffer.
        """
        X, y = self._check_Xy(X_chunk, y_chunk)
        if X.shape[0] == 0:
            return self
        d = (self._buf_X[0].shape[1] if self._buf_X
             else getattr(self, "n_features_in_", X.shape[1]))
        if X.shape[1] != d:
            raise ValueError(f"chunk has {X.shape[1]} features, expected "
                             f"{d}")
        self._buf_X.append(X)
        self._buf_y.append(y)
        if self.n_buffered_ >= self.buffer_size:
            self.fit(np.vstack(self._buf_X), np.concatenate(self._buf_y))
            self._buf_X, self._buf_y = [], []
        return self

    # Prediction
    def _leaves(self, X):
        t = self._tree
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_depth + 1):
            f = t["feature"][node]
            active = f != _LEAF
            if not active.any():
                break
            rows = np.flatnonzero(active)
            go_left = X[rows, f[rows]] <= t["threshold"][node[rows]]
            node[rows] = np.where(go_left, t["left"][node[rows]],
                                  t["right"][node[rows]])
        return node

    def _prepare(self, X):
        if self._tree is None:
            raise NotFittedError("tree has not been trained yet; call fit "
                                 "or send at least buffer_size samples "
                                 "through partial_fit")
        X = self._check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the tree was "
                             f"trained with {self.n_features_in_}")
        return X

    def predict_proba(self, X):
        """Return class probabilities (n_samples, n_classes).

        Columns follow classes_. Raises NotFittedError before training.
        """
        X = self._prepare(X)
        v = self._tree["value"][self._leaves(X)]
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Return predicted labels (n_samples,); ties use the first class."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """Return mean accuracy on X and y, or 0.0 for empty inputs."""
        X, y = self._check_Xy(X, y)
        return float((self.predict(X) == y).mean()) if len(y) else 0.0
