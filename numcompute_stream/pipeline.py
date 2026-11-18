"""Incremental Pipeline and FeatureUnion built on NumCompute."""

from copy import deepcopy

import numpy as np

from numcompute.pipeline import Pipeline as _Pipeline
from numcompute.pipeline import FeatureUnion as _FeatureUnion
from numcompute.pipeline import _has
from .tree import DecisionTreeClassifier

__all__ = ["Pipeline", "FeatureUnion"]


def _require_partial_fit(name, step):
    if not _has(step, "partial_fit"):
        raise TypeError(f"Step '{name}' has no partial_fit method")


class Pipeline(_Pipeline):
    """Pipeline that updates transformers and a model on incoming chunks.

    For a buffered DecisionTreeClassifier, raw input rows are stored until
    the tree's buffer_size is reached. They are then transformed together
    using the current preprocessors, and the tree is refitted. A copy of
    those preprocessors is kept for predictions until the next retrain.
    This prevents changes in scaling or one-hot columns from changing the
    meaning of features seen by the previously trained tree.

    Other partial_fit models are updated directly after the transformers.
    """

    def __init__(self, steps):
        super().__init__(steps)
        self._clear_pending()
        self._fitted_transformers = None

    def _clear_pending(self):
        self._raw_X = []
        self._raw_y = []
        self.n_buffered_ = 0

    def fit(self, X, y=None):
        """Fit on one dataset and reset any pending streaming chunks."""
        result = super().fit(X, y)
        self._clear_pending()
        if self._final_is_estimator and isinstance(
                self.steps[-1][1], DecisionTreeClassifier):
            self._fitted_transformers = deepcopy(
                [step for _, step in self.steps[:-1]])
        else:
            self._fitted_transformers = None
        return result

    def partial_fit(self, X, y=None):
        """Learn from one chunk, returning self.

        Transformer steps must support partial_fit. The final estimator
        requires y. Buffered decision trees retrain on the current raw
        buffer using one consistent preprocessing snapshot.
        """
        for name, step in self.steps:
            _require_partial_fit(name, step)

        if self._final_is_estimator and y is None:
            raise ValueError("y is required to update a final estimator")

        model = self.steps[-1][1]
        buffered_tree = self._final_is_estimator and isinstance(
            model, DecisionTreeClassifier)

        if buffered_tree:
            X_arr, y_arr = np.asarray(X), np.asarray(y)
            if X_arr.ndim != 2:
                raise ValueError("X must be a 2-D array")
            if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
                raise ValueError("y must be 1-D and match the number of rows")
            if not len(X_arr):
                return self

        Xt = X
        for _, step in self.steps[:-1]:
            step.partial_fit(Xt)
            Xt = step.transform(Xt)

        if not self._final_is_estimator:
            model.partial_fit(Xt)
            return self

        if not buffered_tree:
            model.partial_fit(Xt, y)
            return self

        self._raw_X.append(X_arr.copy())
        self._raw_y.append(y_arr.copy())
        self.n_buffered_ += len(X_arr)
        if self.n_buffered_ >= model.buffer_size:
            Xt = np.concatenate(self._raw_X, axis=0)
            for _, step in self.steps[:-1]:
                Xt = step.transform(Xt)
            model.fit(Xt, np.concatenate(self._raw_y))
            self._fitted_transformers = deepcopy(
                [step for _, step in self.steps[:-1]])
            self._clear_pending()
        return self

    def _predict_features(self, X):
        Xt = X
        steps = (self._fitted_transformers if self._fitted_transformers
                 is not None else [step for _, step in self.steps[:-1]])
        for step in steps:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X):
        """Predict using the preprocessing associated with the fitted model."""
        model = self.steps[-1][1]
        if not _has(model, "predict"):
            raise TypeError("Final step does not implement predict")
        return model.predict(self._predict_features(X))

    def predict_proba(self, X):
        """Return class probabilities from the final estimator."""
        model = self.steps[-1][1]
        if not _has(model, "predict_proba"):
            raise TypeError("Final step does not implement predict_proba")
        return model.predict_proba(self._predict_features(X))

    def score(self, X, y):
        """Return classification accuracy for matching X and y samples."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        predictions = self.predict(X)
        if len(predictions) != len(y):
            raise ValueError("X and y must have the same number of samples")
        return float(np.mean(predictions == y)) if len(y) else 0.0


class FeatureUnion(_FeatureUnion):
    """FeatureUnion whose transformers support incremental updates."""

    def partial_fit(self, X, y=None):
        """Update all branches on a chunk and return self."""
        for name, step in self.transformers:
            _require_partial_fit(name, step)
        for _, step in self.transformers:
            step.partial_fit(X)
        return self
