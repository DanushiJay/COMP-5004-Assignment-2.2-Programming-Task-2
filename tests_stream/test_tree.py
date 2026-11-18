"""Tests for decision tree fitting, prediction and buffered retraining."""

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# batch fit / predict

def test_single_threshold_by_hand():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [0, 0, 1, 1]
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)

    assert np.isclose(tree._tree["threshold"][0], 2.5)
    np.testing.assert_array_equal(tree.predict([[1.5], [3.5]]), [0, 1])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_learns_separable_data(blobs, criterion):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=8, criterion=criterion).fit(X, y)
    assert tree.score(X, y) > 0.97


def test_fit_returns_self():
    tree = DecisionTreeClassifier()
    result = tree.fit([[1.0], [2.0]], [0, 1])
    assert result is tree


def test_max_depth_zero_is_majority_vote():
    tree = DecisionTreeClassifier(max_depth=0).fit(
        [[1.0], [2.0], [3.0]], [1, 1, 0]
    )
    np.testing.assert_array_equal(tree.predict([[0.0], [9.0]]), [1, 1])


def test_depth_is_respected(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert len(tree._tree["feature"]) <= 7


def test_pure_labels_give_single_leaf():
    tree = DecisionTreeClassifier().fit(
        [[1.0], [2.0], [3.0]], [5, 5, 5]
    )
    assert len(tree._tree["feature"]) == 1
    np.testing.assert_array_equal(tree.predict([[0.0]]), [5])


def test_constant_feature_cannot_split():
    tree = DecisionTreeClassifier().fit(
        [[1.0], [1.0], [1.0]], [0, 1, 0]
    )
    assert len(tree._tree["feature"]) == 1
    np.testing.assert_array_equal(tree.predict([[1.0]]), [0])


def test_tied_values_are_not_separated():
    X = [[1.0], [2.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    assert tree.score(X, y) == 0.75


def test_majority_class_tie_uses_smallest_label():
    tree = DecisionTreeClassifier(max_depth=0).fit(
        [[1.0], [2.0], [3.0], [4.0]], [1, 0, 1, 0]
    )
    np.testing.assert_array_equal(tree.predict([[0.0], [9.0]]), [0, 0])


def test_min_samples_split_prevents_split():
    tree = DecisionTreeClassifier(min_samples_split=5).fit(
        [[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1]
    )
    assert len(tree._tree["feature"]) == 1


def test_string_labels_and_valid_probabilities(blobs):
    X, y = blobs
    labels = np.where(y == 1, "yes", "no")
    tree = DecisionTreeClassifier(max_depth=3).fit(X, labels)

    assert set(tree.predict(X)) <= {"yes", "no"}
    proba = tree.predict_proba(X)
    assert proba.shape == (X.shape[0], len(tree.classes_))
    assert np.all(np.isfinite(proba))
    assert np.all((0.0 <= proba) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_single_class_probability_is_one():
    tree = DecisionTreeClassifier().fit([[0.0], [1.0]], ["yes", "yes"])
    np.testing.assert_allclose(tree.predict_proba([[5.0]]), [[1.0]])


def test_max_features_options(blobs):
    X, y = blobs
    for max_features in (None, "sqrt", 1, 2, X.shape[1]):
        tree = DecisionTreeClassifier(
            max_features=max_features, random_state=3
        ).fit(X, y)
        assert tree.predict(X[:7]).shape == (7,)


def test_same_seed_same_tree(blobs):
    X, y = blobs
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_non_contiguous_input_matches_copy(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    view = np.asfortranarray(X)[::2]
    np.testing.assert_array_equal(
        tree.predict(view), tree.predict(view.copy())
    )


def test_empty_prediction_has_correct_shape():
    tree = DecisionTreeClassifier().fit([[1.0], [2.0]], [0, 1])
    assert tree.predict(np.empty((0, 1))).shape == (0,)
    assert tree.predict_proba(np.empty((0, 1))).shape == (0, 2)


# numerical stability

@pytest.mark.parametrize(
    "left,right",
    [
        (1e308, 1.5e308),
        (-1.5e308, -1e308),
        (-1e308, 1e308),
        (1.0, np.nextafter(1.0, np.inf)),
    ],
)
def test_extreme_or_adjacent_feature_values_split_correctly(left, right):
    # Distinct finite float64 feature values must remain separable.
    tree = DecisionTreeClassifier(max_depth=1).fit(
        [[left], [right]], [0, 1]
    )
    threshold = tree._tree["threshold"][0]
    assert np.isfinite(threshold)
    assert left <= threshold < right
    np.testing.assert_array_equal(tree.predict([[left], [right]]), [0, 1])


# input validation

def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError, match="not been trained"):
        DecisionTreeClassifier().predict([[1.0]])


def test_nan_rejected():
    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeClassifier().fit([[np.nan], [1.0]], [0, 1])


def test_infinity_rejected():
    with pytest.raises(ValueError, match="infinity"):
        DecisionTreeClassifier().fit([[np.inf], [1.0]], [0, 1])


def test_shape_mismatch_errors(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="rows"):
        DecisionTreeClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError, match="2-D"):
        DecisionTreeClassifier().fit(y, y)
    tree = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        tree.predict(X[:, :2])


def test_target_must_be_one_dimensional():
    with pytest.raises(ValueError, match="1-D"):
        DecisionTreeClassifier().fit([[1.0], [2.0]], [[0], [1]])


def test_text_features_rejected():
    with pytest.raises(TypeError, match="numeric"):
        DecisionTreeClassifier().fit([["a"], ["b"]], [0, 1])


def test_fit_zero_samples_rejected():
    with pytest.raises(ValueError, match="zero samples"):
        DecisionTreeClassifier().fit(np.empty((0, 2)), np.empty(0))


@pytest.mark.parametrize(
    "kw",
    [
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"criterion": "mse"},
        {"buffer_size": 0},
        {"max_features": 0},
    ],
)
def test_bad_hyperparameters(kw):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kw)


# buffered retraining

def test_no_retrain_until_buffer_is_full(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    tree.partial_fit(X[:40], y[:40])
    tree.partial_fit(X[40:80], y[40:80])

    assert tree.n_retrains_ == 0
    assert tree.n_buffered_ == 80
    with pytest.raises(NotFittedError):
        tree.predict(X[:1])


def test_retrain_at_threshold_clears_buffer(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    for i in range(0, 120, 40):
        tree.partial_fit(X[i:i + 40], y[i:i + 40])
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_ == 0


def test_predict_uses_last_trained_tree_between_retrains(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    tree.partial_fit(X[:100], y[:100])
    before = tree.predict(X)
    tree.partial_fit(X[100:150], 1 - y[100:150])

    assert tree.n_retrains_ == 1
    assert tree.n_buffered_ == 50
    np.testing.assert_array_equal(tree.predict(X), before)


def test_retrain_uses_only_buffered_data(blobs):
    X, _ = blobs
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=50)
    tree.partial_fit(X[:50], np.zeros(50, dtype=int))
    np.testing.assert_array_equal(tree.classes_, [0])
    tree.partial_fit(X[50:100], np.ones(50, dtype=int))
    np.testing.assert_array_equal(tree.classes_, [1])


def test_partial_fit_matches_fit_on_same_data(blobs):
    X, y = blobs
    streaming = DecisionTreeClassifier(max_depth=4, buffer_size=120)
    for i in range(0, 120, 30):
        streaming.partial_fit(X[i:i + 30], y[i:i + 30])

    batch = DecisionTreeClassifier(max_depth=4).fit(X[:120], y[:120])
    np.testing.assert_array_equal(streaming.predict(X), batch.predict(X))


def test_empty_chunk_is_ignored(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=10)
    assert tree.partial_fit(X[:0], y[:0]) is tree
    assert tree.n_buffered_ == 0
    assert tree.n_retrains_ == 0


def test_chunk_with_wrong_width_rejected(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=500)
    tree.partial_fit(X[:10], y[:10])
    with pytest.raises(ValueError, match="features"):
        tree.partial_fit(X[:10, :2], y[:10])
    assert tree.n_buffered_ == 10


def test_oversized_chunk_triggers_one_retrain(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=50)
    tree.partial_fit(X[:120], y[:120])
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_ == 0


def test_multiple_retraining_events(blobs):
    X, y = blobs
    tree = DecisionTreeClassifier(buffer_size=50, max_depth=3)
    for i in range(0, 200, 25):
        tree.partial_fit(X[i:i + 25], y[i:i + 25])
    assert tree.n_retrains_ == 4
    assert tree.n_buffered_ == 0


def test_buffered_labels_are_copied():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    first_labels = np.array([0, 0])
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit(X[:2], first_labels)

    # Mutating the caller's array must not alter buffered training data.
    first_labels[:] = 1
    tree.partial_fit(X[2:], np.array([1, 1]))
    np.testing.assert_array_equal(tree.classes_, [0, 1])
    np.testing.assert_array_equal(tree.predict(X), [0, 0, 1, 1])
