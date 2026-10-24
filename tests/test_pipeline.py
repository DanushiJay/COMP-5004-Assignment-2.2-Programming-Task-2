"""Tests for numcompute.pipeline."""

import numpy as np
import pytest

from numcompute.pipeline import FeatureUnion, Pipeline


class Shift:
    def __init__(self, delta):
        self.delta = delta

    def fit(self, X):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) + self.delta

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Double:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


class DropRows:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:-1]


class MeanEstimator:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_pipeline_two_transformers():
    pipe = Pipeline([("shift", Shift(1)), ("double", Double())])
    np.testing.assert_array_equal(pipe.fit_transform(X), (X + 1) * 2)


def test_pipeline_fit_returns_self_and_fits_steps():
    shift = Shift(0)
    pipe = Pipeline([("shift", shift), ("double", Double())])
    assert pipe.fit(X) is pipe
    assert shift.fitted_


def test_pipeline_with_final_estimator_predict():
    pipe = Pipeline([("shift", Shift(1)), ("model", MeanEstimator())])
    pipe.fit(X, np.array([1.0, 2.0, 6.0]))
    np.testing.assert_array_equal(pipe.predict(X), [3.0, 3.0, 3.0])


def test_pipeline_estimator_requires_y():
    pipe = Pipeline([("model", MeanEstimator())])
    with pytest.raises(ValueError):
        pipe.fit(X)


def test_pipeline_transform_on_estimator_pipeline_raises():
    pipe = Pipeline([("model", MeanEstimator())])
    pipe.fit(X, np.zeros(3))
    with pytest.raises(TypeError):
        pipe.transform(X)


def test_pipeline_invalid_step_raises():
    with pytest.raises(TypeError):
        Pipeline([("bad", object()), ("double", Double())])


def test_pipeline_bad_entry_format_raises():
    with pytest.raises(TypeError):
        Pipeline([Double()])


def test_pipeline_duplicate_names_raises():
    with pytest.raises(ValueError):
        Pipeline([("a", Double()), ("a", Double())])


def test_pipeline_empty_steps_raises():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_non_contiguous_input():
    pipe = Pipeline([("double", Double())])
    np.testing.assert_array_equal(pipe.fit_transform(X.T[::-1].T), X[:, ::-1] * 2)


def test_feature_union_width_is_sum_of_branches():
    union = FeatureUnion([("a", Shift(1)), ("b", Double())])
    out = union.fit_transform(X)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[:, :2], X + 1)
    np.testing.assert_array_equal(out[:, 2:], X * 2)


def test_feature_union_mismatched_rows_raises():
    union = FeatureUnion([("a", Double()), ("b", DropRows())])
    with pytest.raises(ValueError, match="rows"):
        union.fit_transform(X)


def test_feature_union_invalid_transformer_raises():
    with pytest.raises(TypeError):
        FeatureUnion([("a", object())])


def test_feature_union_one_dimensional_branch():
    class Sum:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.sum(X, axis=1)

    out = FeatureUnion([("sum", Sum()), ("d", Double())]).fit_transform(X)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 0], [3.0, 7.0, 11.0])


def test_feature_union_nested_in_pipeline():
    pipe = Pipeline(
        [
            ("shift", Shift(1)),
            ("union", FeatureUnion([("a", Double()), ("b", Shift(10))])),
            ("model", MeanEstimator()),
        ]
    )
    pipe.fit(X, np.array([1.0, 2.0, 3.0]))
    assert pipe.predict(X).shape == (3,)
    union_out = Pipeline(pipe.steps[:2]).fit_transform(X)
    assert union_out.shape == (3, 4)
