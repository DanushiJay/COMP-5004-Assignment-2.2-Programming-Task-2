"""Tests for numcompute_stream.pipeline (partial_fit through the steps)."""

from copy import deepcopy

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def stream():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * [1.0, 10.0, 100.0]
    y = (X[:, 0] + X[:, 1] / 10 > 0).astype(int)
    return X, y


def make():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=100)),
    ])


def feed(pipe, X, y, size):
    for i in range(0, len(X), size):
        pipe.partial_fit(X[i:i + size], y[i:i + size])
    return pipe


def test_partial_fit_returns_self_and_learns(stream):
    X, y = stream
    pipe = make()
    assert pipe.partial_fit(X[:50], y[:50]) is pipe
    feed(pipe, X[50:], y[50:], 50)
    assert pipe.steps[-1][1].n_retrains_ == 3
    assert (pipe.predict(X) == y).mean() > 0.9


def test_matches_doing_the_steps_by_hand(stream):
    X, y = stream
    pipe = feed(make(), X, y, 50)
    scaler = StandardScaler()
    tree = DecisionTreeClassifier(max_depth=4)
    raw, snapshot = [], None
    for i in range(0, 300, 50):
        scaler.partial_fit(X[i:i + 50])
        raw.append(i)
        if len(raw) * 50 >= 100:
            rows = np.concatenate([X[j:j + 50] for j in raw])
            labels = np.concatenate([y[j:j + 50] for j in raw])
            tree.fit(scaler.transform(rows), labels)
            snapshot = deepcopy(scaler)
            raw = []
    np.testing.assert_array_equal(pipe.predict(X),
                                  tree.predict(snapshot.transform(X)))


def test_prediction_keeps_the_preprocessing_the_tree_was_trained_with(
        stream):
    X, y = stream
    pipe = feed(make(), X[:100], y[:100], 50)        # one retrain
    before = pipe.predict(X[:60])
    pipe.partial_fit(X[100:150] * 1000.0, y[100:150])  # shifts the scaler
    assert pipe.n_buffered_ == 50
    np.testing.assert_array_equal(pipe.predict(X[:60]), before)


def test_one_hot_encoder_may_grow_inside_the_buffer():
    cats = np.array([["a"], ["b"], ["a"], ["b"], ["c"], ["c"], ["a"],
                     ["b"]])
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    pipe = Pipeline([("enc", OneHotEncoder()),
                     ("model", DecisionTreeClassifier(max_depth=3,
                                                      buffer_size=8))])
    for i in range(0, 8, 2):
        pipe.partial_fit(cats[i:i + 2], y[i:i + 2])
    np.testing.assert_array_equal(pipe.predict(cats), y)


def test_buffer_count_and_reset_by_fit(stream):
    X, y = stream
    pipe = make().partial_fit(X[:30], y[:30])
    assert pipe.n_buffered_ == 30
    pipe.fit(X, y)
    assert pipe.n_buffered_ == 0


def test_other_models_are_updated_directly(stream):
    X, y = stream
    seen = []

    class Recorder:
        def fit(self, X, y):
            return self

        def partial_fit(self, X, y):
            seen.append((X.shape, y.shape))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    pipe = Pipeline([("scale", StandardScaler()), ("model", Recorder())])
    pipe.partial_fit(X[:40], y[:40])
    assert seen == [((40, 3), (40,))]


def test_score_is_accuracy(stream):
    X, y = stream
    pipe = feed(make(), X, y, 50)
    assert pipe.score(X, y) == float(np.mean(pipe.predict(X) == y))
    assert pipe.score(X[:0], y[:0]) == 0.0


def test_three_step_pipeline_with_missing_values(stream):
    X, y = stream
    X = X.copy()
    X[::11, 1] = np.nan
    pipe = Pipeline([
        ("impute", SimpleImputer("mean")),
        ("scale", MinMaxScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=100)),
    ])
    feed(pipe, X, y, 40)
    assert pipe.predict(X).shape == (300,)


def test_predict_before_first_retrain_raises(stream):
    X, y = stream
    pipe = make().partial_fit(X[:50], y[:50])
    with pytest.raises(NotFittedError):
        pipe.predict(X[:5])


def test_predict_proba_rows_sum_to_one(stream):
    X, y = stream
    pipe = feed(make(), X, y, 50)
    np.testing.assert_allclose(pipe.predict_proba(X).sum(axis=1), 1.0)


def test_predict_proba_needs_support():
    pipe = Pipeline([("scale", StandardScaler())])
    with pytest.raises(TypeError, match="predict_proba"):
        pipe.predict_proba(np.ones((2, 1)))


def test_y_required_for_final_model(stream):
    X, _ = stream
    with pytest.raises(ValueError, match="y is required"):
        make().partial_fit(X[:10])


def test_step_without_partial_fit_is_named():
    class Frozen:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("frozen", Frozen()), ("scale", StandardScaler())])
    with pytest.raises(TypeError, match="'frozen'"):
        pipe.partial_fit(np.ones((2, 1)))


def test_transformer_only_pipeline_streams(stream):
    X, _ = stream
    pipe = Pipeline([("a", SimpleImputer()), ("b", StandardScaler())])
    for i in range(0, 300, 60):
        pipe.partial_fit(X[i:i + 60])
    out = pipe.transform(X)
    assert abs(out.mean()) < 0.1


def test_empty_chunk_is_harmless(stream):
    X, y = stream
    pipe = feed(make(), X[:100], y[:100], 50)
    pipe.partial_fit(X[:0], y[:0])
    assert pipe.steps[-1][1].n_retrains_ == 1


def test_wrong_width_chunk_rejected(stream):
    X, y = stream
    pipe = make().partial_fit(X[:50], y[:50])
    with pytest.raises(ValueError, match="Expected shape"):
        pipe.partial_fit(X[:10, :2], y[:10])


def test_fit_still_works_like_the_original(stream):
    X, y = stream
    pipe = make().fit(X, y)
    assert (pipe.predict(X) == y).mean() > 0.9


def test_feature_union_partial_fit(stream):
    X, _ = stream
    union = FeatureUnion([("std", StandardScaler()),
                          ("mm", MinMaxScaler())])
    for i in range(0, 300, 100):
        union.partial_fit(X[i:i + 100])
    out = union.transform(X)
    assert out.shape == (300, 6)
    batch = FeatureUnion([("std", StandardScaler()),
                          ("mm", MinMaxScaler())]).fit(X).transform(X)
    np.testing.assert_allclose(out, batch)


def test_feature_union_names_step_without_partial_fit():
    class Frozen:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    union = FeatureUnion([("frozen", Frozen())])
    with pytest.raises(TypeError, match="'frozen'"):
        union.partial_fit(np.ones((2, 1)))


def test_scaler_update_does_not_change_old_model_prediction():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=4, max_depth=2)),
    ])
    pipe.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    previous = pipe.predict([[2.5]])
    pipe.partial_fit([[100.0]], [0])
    np.testing.assert_array_equal(pipe.predict([[2.5]]), previous)
    assert pipe.n_buffered_ == 1


def test_encoder_can_expand_before_retraining():
    pipe = Pipeline([
        ("encode", OneHotEncoder()),
        ("tree", DecisionTreeClassifier(buffer_size=2, max_depth=2)),
    ])
    pipe.partial_fit([["b"], ["c"]], [0, 1])
    previous = pipe.predict([["c"]])
    pipe.partial_fit([["a"]], [0])
    np.testing.assert_array_equal(pipe.predict([["c"]]), previous)
    pipe.partial_fit([["c"]], [1])
    assert pipe.steps[-1][1].n_features_in_ == 3
    assert pipe.predict([["a"], ["c"]]).shape == (2,)


def test_pending_buffer_uses_one_scaler_snapshot():
    X = np.array([[0.0], [1.0], [100.0], [101.0]])
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(buffer_size=4, max_depth=2)),
    ])
    pipe.partial_fit(X[:2], y[:2])
    pipe.partial_fit(X[2:], y[2:])
    expected = StandardScaler().fit(X)
    tree = DecisionTreeClassifier(max_depth=2).fit(expected.transform(X), y)
    np.testing.assert_array_equal(pipe.predict(X),
                                  tree.predict(expected.transform(X)))


def test_fit_discards_pending_streaming_data(stream):
    X, y = stream
    pipe = make().partial_fit(X[:30], y[:30])
    assert pipe.n_buffered_ == 30
    pipe.fit(X[:80], y[:80])
    assert pipe.n_buffered_ == 0
    assert pipe.predict(X[:5]).shape == (5,)


def test_score_rejects_wrong_number_of_labels(stream):
    X, y = stream
    pipe = make().fit(X, y)
    with pytest.raises(ValueError, match="same number"):
        pipe.score(X[:2], y[:1])
    with pytest.raises(ValueError, match="same number"):
        pipe.score(X[:2], y[:0])
