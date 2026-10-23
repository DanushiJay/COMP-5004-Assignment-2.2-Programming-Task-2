"""Transformer/Estimator protocol, Pipeline (Compose) and FeatureUnion."""


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
