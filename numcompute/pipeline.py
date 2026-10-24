"""Transformer/Estimator protocol, Pipeline (Compose) and FeatureUnion."""

import numpy as np


class Transformer:
    """Base class documenting the transformer protocol.

    Subclassing is optional; any object with ``fit`` and ``transform`` is accepted.

    Shapes
    ------
    ``fit(X)``: X is ``(n_samples, n_features)``.
    ``transform(X)``: returns ``(n_samples, n_features_out)``.
    """

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit to ``X`` then transform it; returns ``(n_samples, n_features_out)``."""
        return self.fit(X).transform(X)


class Estimator:
    """Base class documenting the estimator protocol.

    Shapes
    ------
    ``fit(X, y)``: X is ``(n_samples, n_features)``, y is ``(n_samples,)``.
    ``predict(X)``: returns ``(n_samples,)``.
    """

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


def _has(obj, *names):
    return all(callable(getattr(obj, n, None)) for n in names)


def _check_steps(steps, what):
    """Validate a list of ``(name, object)`` pairs and return it as a list."""
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(f"{what} must be a non-empty list of (name, object) pairs")
    names = []
    for item in steps:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)):
            raise TypeError(f"Each {what} entry must be a (name, object) pair, got {item!r}")
        names.append(item[0])
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names must be unique, got {names}")
    return [tuple(s) for s in steps]


class Pipeline:
    """Chain transformers sequentially, optionally ending with an estimator.

    Parameters
    ----------
    steps : list of (str, object)
        Named steps. Every step except the last must implement ``fit`` and
        ``transform``. The last step is either a transformer or an estimator
        (``fit`` and ``predict``).

    Shapes
    ------
    ``X`` is ``(n_samples, n_features)``; ``transform`` returns
    ``(n_samples, n_features_out)`` and ``predict`` returns ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``steps`` is empty or has duplicate names.
    TypeError
        If an entry is not a ``(name, object)`` pair or a step does not
        implement the required protocol.

    Notes
    -----
    Time is the sum of the step costs; space is dominated by the largest
    intermediate array, O(n_samples * max n_features_out).
    """

    def __init__(self, steps):
        self.steps = _check_steps(steps, "steps")
        for name, step in self.steps[:-1]:
            if not _has(step, "fit", "transform"):
                raise TypeError(f"Intermediate step '{name}' must implement fit and transform")
        name, last = self.steps[-1]
        if not (_has(last, "fit", "transform") or _has(last, "fit", "predict")):
            raise TypeError(
                f"Final step '{name}' must implement fit and either transform or predict"
            )

    @property
    def _final_is_estimator(self):
        last = self.steps[-1][1]
        return _has(last, "predict") and not _has(last, "transform")

    def fit(self, X, y=None):
        """Fit every step in order; returns ``self``."""
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt) if _has(step, "fit_transform") else step.fit(Xt).transform(Xt)
        last = self.steps[-1][1]
        if self._final_is_estimator:
            if y is None:
                raise ValueError("y is required to fit a final estimator")
            last.fit(Xt, y)
        else:
            last.fit(Xt)
        return self

    def _transform_intermediate(self, X):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    def transform(self, X):
        """Apply every step's ``transform``; the final step must be a transformer."""
        if self._final_is_estimator:
            raise TypeError("Final step is an estimator and has no transform; use predict")
        return self.steps[-1][1].transform(self._transform_intermediate(X))

    def fit_transform(self, X, y=None):
        """Fit the pipeline then transform ``X``."""
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """Transform through the intermediate steps, then call the final ``predict``."""
        last = self.steps[-1][1]
        if not _has(last, "predict"):
            raise TypeError("Final step does not implement predict")
        return last.predict(self._transform_intermediate(X))


class FeatureUnion:
    """Run several transformers on the same input and concatenate their outputs.

    Parameters
    ----------
    transformers : list of (str, object)
        Named transformers, each implementing ``fit`` and ``transform``.

    Shapes
    ------
    ``X`` is ``(n_samples, n_features)``. Branch ``i`` returns
    ``(n_samples, w_i)`` (a 1-D output is treated as ``(n_samples, 1)``) and
    the union returns ``(n_samples, sum(w_i))``.

    Raises
    ------
    ValueError
        If the list is empty, names repeat, a branch returns a different
        number of rows from the others, or a branch output is not 1-D/2-D.
    TypeError
        If an entry is not a ``(name, object)`` pair or lacks ``fit``/``transform``.

    Notes
    -----
    Time is the sum of the branch costs plus O(n_samples * total width) for the
    concatenation; space is O(n_samples * total width).
    """

    def __init__(self, transformers):
        self.transformers = _check_steps(transformers, "transformers")
        for name, t in self.transformers:
            if not _has(t, "fit", "transform"):
                raise TypeError(f"Transformer '{name}' must implement fit and transform")

    def fit(self, X, y=None):
        """Fit every branch on ``X``; returns ``self``."""
        for _, t in self.transformers:
            t.fit(X)
        return self

    def transform(self, X):
        """Transform ``X`` with each branch and concatenate along ``axis=1``."""
        outputs = []
        for name, t in self.transformers:
            out = np.asarray(t.transform(X))
            if out.ndim == 1:
                out = out[:, None]
            if out.ndim != 2:
                raise ValueError(f"Branch '{name}' must return a 1-D or 2-D array, got {out.shape}")
            outputs.append((name, out))
        n = outputs[0][1].shape[0]
        for name, out in outputs:
            if out.shape[0] != n:
                raise ValueError(
                    f"Branch '{name}' returned {out.shape[0]} rows, expected {n}"
                )
        return np.hstack([out for _, out in outputs])

    def fit_transform(self, X, y=None):
        """Fit every branch then return the concatenated output."""
        return self.fit(X, y).transform(X)
