"""Tests for the streaming visualisation functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream import visualise  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Keep plotting tests independent of one another."""
    yield
    plt.close("all")


@pytest.fixture
def history():
    return np.linspace(0.5, 0.9, 12)


def test_metric_plot_has_correct_values(history):
    fig = visualise.plot_metric_over_time(
        history, title="Accuracy", ylabel="Score", show=False
    )
    ax = fig.axes[0]
    x, y = ax.lines[0].get_data()
    np.testing.assert_array_equal(x, np.arange(1, 13))
    np.testing.assert_allclose(y, history)
    assert ax.get_title() == "Accuracy"
    assert ax.get_ylabel() == "Score"


def test_metric_plot_accepts_custom_x_and_nan_gaps():
    fig = visualise.plot_metric_over_time(
        [np.nan, 0.5, 0.8], x=[10, 20, 30], show=False
    )
    x, y = fig.axes[0].lines[0].get_data()
    np.testing.assert_array_equal(x, [10, 20, 30])
    assert np.isnan(y[0])


def test_plot_saves_image(history, tmp_path):
    filename = tmp_path / "accuracy.png"
    fig = visualise.plot_metric_over_time(
        history, save_path=filename, show=False
    )
    assert filename.exists() and filename.stat().st_size > 1000
    assert fig is not None
    assert fig.number not in plt.get_fignums()


def test_plot_without_save_remains_open(history):
    fig = visualise.plot_metric_over_time(history, show=False)
    assert fig.number in plt.get_fignums()


def test_show_option_calls_matplotlib(history, monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    visualise.plot_metric_over_time(history, show=True)
    assert calls == [True]


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], ["a", "b"], [1.0, np.inf]])
def test_metric_plot_rejects_invalid_histories(bad):
    with pytest.raises((TypeError, ValueError)):
        visualise.plot_metric_over_time(bad, show=False)


def test_metric_plot_rejects_wrong_x_length(history):
    with pytest.raises(ValueError, match="x has"):
        visualise.plot_metric_over_time(history, x=[1, 2], show=False)


def test_metric_plot_rejects_nan_x():
    with pytest.raises(ValueError, match="finite"):
        visualise.plot_metric_over_time(
            [0.2, 0.5], x=[1.0, np.nan], show=False)


def test_compare_models_draws_two_lines(history):
    fig = visualise.compare_models(
        history, history[::-1], ["Tree A", "Tree B"], show=False
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert [label.get_text() for label in ax.get_legend().get_texts()] == [
        "Tree A", "Tree B"
    ]
    np.testing.assert_allclose(ax.lines[1].get_ydata(), history[::-1])


def test_compare_models_supports_different_lengths():
    fig = visualise.compare_models(
        [0.4, 0.5, 0.6], [0.7, 0.8], ["A", "B"], show=False
    )
    assert [len(line.get_xdata()) for line in fig.axes[0].lines] == [3, 2]


def test_compare_models_requires_two_names(history):
    with pytest.raises(ValueError, match="two names"):
        visualise.compare_models(history, history, ["Only one"], show=False)


def test_compare_models_rejects_empty_history(history):
    with pytest.raises(ValueError, match="metric2"):
        visualise.compare_models(history, [], ["A", "B"], show=False)


def test_predictions_plot_counts_errors_and_confusion_cells():
    true = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    fig = visualise.plot_predictions_vs_ground_truth(true, pred, show=False)
    assert len(fig.axes) == 2
    assert "1 of 4 wrong" in fig.axes[0].get_title()
    matrix_values = sorted(txt.get_text() for txt in fig.axes[1].texts)
    assert matrix_values == ["0", "1", "1", "2"]


def test_predictions_plot_accepts_text_labels():
    fig = visualise.plot_predictions_vs_ground_truth(
        ["cat", "dog", "cat", "emu"],
        ["cat", "cat", "cat", "emu"], show=False
    )
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert labels == ["cat", "dog", "emu"]


def test_predictions_plot_includes_labels_only_in_predictions():
    fig = visualise.plot_predictions_vs_ground_truth(
        [0, 0], [0, 3], show=False
    )
    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert labels == ["0", "3"]


@pytest.mark.parametrize("true,pred", [
    ([0, 1], [0]),
    ([], []),
    ([[0, 1]], [[0, 1]]),
])
def test_predictions_plot_rejects_bad_shapes(true, pred):
    with pytest.raises(ValueError):
        visualise.plot_predictions_vs_ground_truth(true, pred, show=False)


def test_predictions_plot_saves_png(tmp_path):
    filename = tmp_path / "predictions.png"
    visualise.plot_predictions_vs_ground_truth(
        [0, 1, 1, 0], [0, 0, 1, 0], save_path=filename, show=False
    )
    assert filename.exists() and filename.stat().st_size > 1000
