"""Tests for numcompute_stream.metrics (update / reset / result)."""

import numpy as np
import pytest

from numcompute import metrics as group
from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = np.where(rng.random(200) < 0.7, y_true,
                      rng.integers(0, 3, size=200))
    return y_true, y_pred


@pytest.fixture
def scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    s = np.clip(0.35 * y + rng.normal(0.4, 0.2, size=400), 0.0, 1.0)
    return y, s


def feed(metric, a, b, size):
    for i in range(0, len(a), size):
        metric.update(a[i:i + size], b[i:i + size])
    return metric


# equal to the group's metrics
@pytest.mark.parametrize("size", [1, 17, 200, 999])
def test_confusion_matrix_matches_group(labels, size):
    yt, yp = labels
    m = feed(StreamingConfusionMatrix(), yt, yp, size)
    np.testing.assert_array_equal(m.result(),
                                  group.confusion_matrix(yt, yp))


@pytest.mark.parametrize("size", [1, 17, 200])
def test_accuracy_matches_group(labels, size):
    yt, yp = labels
    got = feed(StreamingAccuracy(), yt, yp, size).result()
    assert abs(got - group.accuracy(yt, yp)) < 1e-12


@pytest.mark.parametrize("average", ["binary", "macro"])
@pytest.mark.parametrize("size", [5, 64])
def test_precision_recall_f1_match_group(labels, average, size):
    yt, yp = labels
    for cls, fn in [(StreamingPrecision, group.precision),
                    (StreamingRecall, group.recall),
                    (StreamingF1, group.f1)]:
        got = feed(cls(average=average, pos_label=2), yt, yp, size).result()
        want = fn(yt, yp, average=average, pos_label=2)
        assert abs(got - want) < 1e-12


def test_string_labels(labels):
    yt, yp = labels
    names = np.array(["cat", "dog", "emu"])
    a, b = names[yt], names[yp]
    m = feed(StreamingConfusionMatrix(), a, b, 30)
    np.testing.assert_array_equal(m.result(), group.confusion_matrix(a, b))
    assert list(m.labels_) == ["cat", "dog", "emu"]


# labels that appear
def test_matrix_grows_when_a_new_label_appears():
    m = StreamingConfusionMatrix()
    m.update([0, 1], [0, 0])
    assert m.result().shape == (2, 2)
    m.update([2, 0], [2, 0])
    cm = m.result()
    assert cm.shape == (3, 3)
    np.testing.assert_array_equal(
        cm, group.confusion_matrix([0, 1, 2, 0], [0, 0, 2, 0]))


def test_new_label_in_the_middle_keeps_old_counts_in_place():
    m = StreamingConfusionMatrix()
    m.update([0, 2], [0, 2])
    m.update([1], [1])
    np.testing.assert_array_equal(m.result(), np.eye(3, dtype=int))


def test_fixed_labels_reject_unknown_label():
    m = StreamingConfusionMatrix(labels=[0, 1])
    m.update([0, 1], [1, 1])
    with pytest.raises(ValueError, match="not in"):
        m.update([2], [0])
    assert m.result().shape == (2, 2)


def test_fixed_labels_give_full_shape_from_the_start():
    m = StreamingConfusionMatrix(labels=[0, 1, 2]).update([0], [0])
    assert m.result().shape == (3, 3)


# rolling window
def test_window_uses_only_recent_samples():
    acc = StreamingAccuracy(window=4)
    acc.update([0, 0, 0, 0], [1, 1, 1, 1])     # all wrong
    assert acc.result() == 0.0
    acc.update([0, 0, 0, 0], [0, 0, 0, 0])     # all right
    assert acc.result() == 1.0
    assert acc.n_samples_seen_ == 8


def test_window_metric_matches_group_on_last_samples(labels):
    yt, yp = labels
    f = feed(StreamingF1(average="macro", window=50), yt, yp, 13)
    want = group.f1(yt[-50:], yp[-50:], average="macro")
    # classes come from the whole stream, so compare on the same classes
    assert abs(f.result() - want) < 1e-12


def test_cumulative_matrix_ignores_the_window(labels):
    yt, yp = labels
    m = feed(StreamingConfusionMatrix(window=10), yt, yp, 20)
    assert m.result().sum() == 10
    assert m._matrix.sum() == 200


# edge cases
def test_result_before_data_raises():
    for m in (StreamingConfusionMatrix(), StreamingAccuracy(),
              StreamingPrecision(), StreamingAUC()):
        with pytest.raises(ValueError, match="No samples"):
            m.result()


def test_empty_chunk_is_ignored(labels):
    yt, yp = labels
    m = StreamingAccuracy().update(yt[:10], yp[:10])
    before = m.result()
    m.update([], [])
    assert m.result() == before


def test_shape_errors():
    m = StreamingAccuracy()
    with pytest.raises(ValueError, match="1-D"):
        m.update([[0, 1]], [[0, 1]])
    with pytest.raises(ValueError, match="Expected shape"):
        m.update([0, 1], [0])


def test_reset_starts_over(labels):
    yt, yp = labels
    m = feed(StreamingAccuracy(), yt, yp, 50)
    assert m.reset() is m
    assert m.n_samples_seen_ == 0
    with pytest.raises(ValueError, match="No samples"):
        m.result()
    m.update([1, 1], [1, 0])
    assert m.result() == 0.5


def test_precision_zero_division_returns_zero():
    p = StreamingPrecision(pos_label=1).update([0, 0, 1], [0, 0, 0])
    assert p.result() == 0.0
    assert StreamingRecall(pos_label=9).update([0], [0]).result() == 0.0


def test_bad_average_and_window():
    with pytest.raises(ValueError, match="average"):
        StreamingPrecision(average="micro")
    with pytest.raises(ValueError, match="window"):
        StreamingAccuracy(window=0)


# AUC
@pytest.mark.parametrize("size", [1, 25, 400])
def test_auc_close_to_group_auc(scores, size):
    y, s = scores
    fpr, tpr, _ = group.roc_curve(y, s)
    want = group.auc(fpr, tpr)
    got = feed(StreamingAUC(n_bins=2000), y, s, size).result()
    assert abs(got - want) < 0.005


def test_auc_exact_when_scores_fall_in_separate_bins():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    a = StreamingAUC(n_bins=100)
    a.update(y[:2], s[:2]).update(y[2:], s[2:])
    assert abs(a.result() - 0.75) < 1e-12


def test_auc_ties_count_half():
    a = StreamingAUC(n_bins=10).update([0, 1], [0.5, 0.5])
    assert abs(a.result() - 0.5) < 1e-12


def test_auc_is_nan_until_both_classes_appear():
    a = StreamingAUC().update([1, 1], [0.9, 0.8])
    assert np.isnan(a.result())
    a.update([0], [0.1])
    assert a.result() == 1.0


def test_auc_clips_scores_outside_range():
    a = StreamingAUC(n_bins=10).update([0, 1], [-5.0, 7.0])
    assert a.result() == 1.0


def test_auc_window_is_exact_for_recent_samples(scores):
    y, s = scores
    a = feed(StreamingAUC(window=100), y, s, 30)
    fpr, tpr, _ = group.roc_curve(y[-100:], s[-100:])
    assert abs(a.result() - group.auc(fpr, tpr)) < 1e-12


def test_auc_roc_curve_shape_and_errors(scores):
    y, s = scores
    a = feed(StreamingAUC(n_bins=50), y, s, 100)
    fpr, tpr = a.roc_curve()
    assert fpr[0] == 0.0 and fpr[-1] == 1.0 and tpr[-1] == 1.0
    with pytest.raises(ValueError, match="both"):
        StreamingAUC().update([1], [0.5]).roc_curve()


def test_auc_rejects_nan_and_bad_arguments():
    with pytest.raises(ValueError, match="NaN"):
        StreamingAUC().update([0, 1], [np.nan, 0.5])
    for kw in ({"n_bins": 1}, {"score_range": (1.0, 1.0)},
               {"window": 0}):
        with pytest.raises(ValueError):
            StreamingAUC(**kw)


def test_auc_other_positive_label_and_reset():
    a = StreamingAUC(pos_label="yes", n_bins=10)
    a.update(["no", "yes"], [0.2, 0.9])
    assert a.result() == 1.0
    a.reset()
    assert a.n_samples_seen_ == 0


def test_windowed_macro_scores_only_classes_in_the_window(labels):
    yt = np.array([0] * 30 + [1] * 30 + [2] * 30)
    yp = np.where(np.arange(90) % 7 == 0, 1, yt)
    f = feed(StreamingF1(average="macro", window=30), yt, yp, 10)
    want = group.f1(yt[-30:], yp[-30:], average="macro")
    assert abs(f.result() - want) < 1e-12


def test_nan_labels_rejected():
    with pytest.raises(ValueError, match="NaN"):
        StreamingAccuracy().update([np.nan, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        StreamingAUC().update([np.nan, 1.0], [0.1, 0.2])


def test_auc_rejects_infinite_scores():
    with pytest.raises(ValueError, match="finite"):
        StreamingAUC().update([0, 1], [0.1, np.inf])


def test_auc_scores_far_outside_range_clip_safely():
    metric = StreamingAUC(n_bins=100)
    metric.update([0, 1], [-1e308, 1e308])
    assert metric.result() == 1.0


def test_auc_same_bin_is_an_approximation():
    y = np.array([0, 1])
    scores = np.array([0.0008, 0.0002])
    result = StreamingAUC(n_bins=100).update(y, scores).result()
    assert result == 0.5                  # both land in the first bin
    fpr, tpr, _ = group.roc_curve(y, scores)
    assert group.auc(fpr, tpr) == 0.0     # the exact value differs


def test_rolling_macro_ignores_labels_outside_window():
    metric = StreamingF1(average="macro", window=2)
    metric.update([2, 2], [2, 2])
    metric.update([0, 1], [0, 1])
    assert metric.result() == 1.0


def test_accuracy_with_fixed_labels_requires_samples():
    metric = StreamingAccuracy(labels=[0, 1])
    with pytest.raises(ValueError, match="No samples"):
        metric.result()
    metric.update([0], [0])
    assert metric.result() == 1.0


def test_fixed_labels_with_window_before_any_data():
    cm = StreamingConfusionMatrix(labels=[0, 1], window=5)
    np.testing.assert_array_equal(cm.result(), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="No samples"):
        StreamingAccuracy(labels=[0, 1], window=5).result()
    empty = StreamingAccuracy(labels=[0, 1], window=5).update([], [])
    with pytest.raises(ValueError, match="No samples"):
        empty.result()
