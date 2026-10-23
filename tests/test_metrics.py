"""Tests for numcompute.metrics."""

import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)


def test_accuracy_all_correct():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy([0, 0, 1], [1, 1, 0]) == 0.0


def test_accuracy_partial():
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75


def test_accuracy_shape_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1, 1], [0, 1])


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix_binary_values():
    cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])


def test_confusion_matrix_multiclass_shape():
    cm = confusion_matrix([0, 1, 2, 2, 1], [0, 2, 2, 1, 1])
    assert cm.shape == (3, 3)
    assert cm.sum() == 5
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 1], [0, 1, 1]])


def test_confusion_matrix_includes_class_only_in_predictions():
    cm = confusion_matrix([0, 0], [0, 1])
    assert cm.shape == (2, 2)


def test_confusion_matrix_unknown_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, 2], labels=[0, 1])


def test_mse_known_value():
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4.0 / 3.0)


def test_mse_perfect_prediction_is_zero():
    assert mse([1.5, 2.5], [1.5, 2.5]) == 0.0


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])


def test_precision_recall_known_values():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)


def test_precision_zero_division_returns_zero():
    assert precision([1, 1, 0], [0, 0, 0]) == 0.0


def test_recall_zero_division_returns_zero():
    assert recall([0, 0, 0], [1, 0, 1]) == 0.0


def test_single_class_all_negative_does_not_raise():
    assert precision([0, 0], [0, 0]) == 0.0
    assert recall([0, 0], [0, 0]) == 0.0
    assert f1([0, 0], [0, 0]) == 0.0


def test_f1_known_value():
    # tp=2, fp=1, fn=1 -> p = r = 2/3 -> f1 = 2/3
    assert f1([1, 1, 1, 0, 0], [1, 1, 0, 1, 0]) == pytest.approx(2 / 3)


def test_f1_zero_when_no_overlap():
    assert f1([1, 1, 0], [0, 0, 1]) == 0.0


def test_f1_perfect():
    assert f1([1, 0, 1], [1, 0, 1]) == 1.0


def test_macro_average_multiclass():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 2]
    # precision per class: 1, 0, 2/3 ; recall per class: 1, 0, 1
    assert precision(y_true, y_pred, average="macro") == pytest.approx(5 / 9)
    assert recall(y_true, y_pred, average="macro") == pytest.approx(2 / 3)


def test_invalid_average_raises():
    with pytest.raises(ValueError):
        precision([0, 1], [0, 1], average="weird")


def test_roc_curve_perfect_separation():
    fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    np.testing.assert_allclose(fpr, [0, 0, 0, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 1, 1, 1])
    assert np.isinf(thr[0])
    assert auc(fpr, tpr) == 1.0


def test_roc_curve_hand_computed_auc():
    # scores .1 .4 .35 .8, labels 0 0 1 1 -> AUC 0.75
    fpr, tpr, _ = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert auc(fpr, tpr) == pytest.approx(0.75)


def test_roc_curve_tied_scores_single_point():
    fpr, tpr, thr = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(fpr, [0, 1])
    np.testing.assert_allclose(tpr, [0, 1])
    assert auc(fpr, tpr) == pytest.approx(0.5)


def test_roc_curve_single_class_raises():
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.2, 0.5, 0.9])


def test_roc_curve_nan_score_raises():
    with pytest.raises(ValueError):
        roc_curve([0, 1], [0.1, np.nan])


def test_auc_unit_square_and_triangle():
    assert auc([0, 1], [1, 1]) == 1.0
    assert auc([0, 1], [0, 1]) == 0.5


def test_auc_non_monotonic_x_raises():
    with pytest.raises(ValueError):
        auc([0, 2, 1], [0, 1, 1])


def test_auc_too_few_points_raises():
    with pytest.raises(ValueError):
        auc([0.0], [1.0])
