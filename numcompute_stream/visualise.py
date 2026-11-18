"""Reusable plots for streaming model results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]


def _metric_values(values, name="metric_values"):
    """Return a non-empty numeric history. NaN marks missing chunks."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain numeric values")
    result = array.astype(float)
    if np.isinf(result).any():
        raise ValueError(f"{name} must not contain infinity")
    return result


def _finish(fig, save_path, show):
    """Save or display a figure and return it for further use."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)
    return fig


def plot_metric_over_time(metric_values, title="", ylabel="", x=None,
                          save_path=None, show=True):
    """Plot one metric across incoming chunks.

    NaN values appear as gaps. By default, the x-axis starts at chunk 1.
    Returns a matplotlib Figure. Set show=False to suppress display.
    """
    y = _metric_values(metric_values)
    xs = np.arange(1, len(y) + 1) if x is None else _metric_values(x, "x")
    if len(xs) != len(y):
        raise ValueError(
            f"x has {len(xs)} values but metric_values has {len(y)}")
    if not np.isfinite(xs).all():
        raise ValueError("x must contain finite positions")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, y, marker="o", markersize=3, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Chunk" if x is None else "")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels, title="", ylabel="",
                   save_path=None, show=True):
    """Compare two models' metric histories on the same axes."""
    if isinstance(labels, (str, bytes)) or labels is None or len(labels) != 2:
        raise ValueError("labels must contain two names")

    a = _metric_values(metric1, "metric1")
    b = _metric_values(metric2, "metric2")
    fig, ax = plt.subplots(figsize=(7, 4))
    for values, name in ((a, labels[0]), (b, labels[1])):
        ax.plot(np.arange(1, len(values) + 1), values, marker="o",
                markersize=3, linewidth=1.5, label=str(name))
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="",
                                     save_path=None, show=True):
    """Plot true/predicted labels and a confusion matrix for one chunk."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D sequences")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    classes = np.union1d(true, pred)
    true_idx = np.searchsorted(classes, true)
    pred_idx = np.searchsorted(classes, pred)
    wrong = true_idx != pred_idx
    sample_numbers = np.arange(1, len(true) + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]}
    )
    ax1.step(sample_numbers, true_idx, where="mid", label="Actual")
    ax1.scatter(sample_numbers, pred_idx, s=16, label="Predicted", zorder=3)
    ax1.scatter(sample_numbers[wrong], pred_idx[wrong], marker="x", s=40,
                color="red", label="Incorrect", zorder=4)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels([str(c) for c in classes])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Class")
    ax1.set_title(title or f"Predictions ({wrong.sum()} of {len(true)} wrong)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Each observation contributes one count to its true/predicted class pair.
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels([str(c) for c in classes])
    ax2.set_yticklabels([str(c) for c in classes])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion matrix")
    midpoint = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > midpoint else "black")

    fig.tight_layout()
    return _finish(fig, save_path, show)
