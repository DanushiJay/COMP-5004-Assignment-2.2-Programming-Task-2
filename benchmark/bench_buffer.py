"""Compare buffer sizes on a data stream with a changing label rule.

Run from the repository root with ``python -m benchmark.bench_buffer``.
Results are saved in ``benchmark/results``.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import compare_models

BUFFER_SIZES = (25, 50, 100, 200, 400, 800, 1600)
N_SAMPLES = 6000
N_FEATURES = 5
CHUNK_SIZE = 25
ROLLING_WINDOW = 500
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_stream(seed=0):
    """Generate a binary classification stream with a midpoint concept change."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    signal = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=N_SAMPLES)
    y = (signal > 0).astype(int)
    y[N_SAMPLES // 2:] = 1 - y[N_SAMPLES // 2:]
    return X, y


def train_stream(buffer_size, X, y):
    model = DecisionTreeClassifier(
        max_depth=5, buffer_size=buffer_size, random_state=0
    )
    trainer = StreamTrainer(model, window=ROLLING_WINDOW)
    for start in range(0, len(y), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        trainer.process_chunk(X[start:end], y[start:end])
    return trainer


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = make_stream()

    # Compare the same chunks for every model, after the largest buffer fills.
    common_start = (max(BUFFER_SIZES) + CHUNK_SIZE - 1) // CHUNK_SIZE
    change_start = (len(y) // 2) // CHUNK_SIZE
    results = []
    curves = {}

    for buffer_size in BUFFER_SIZES:
        trainer = train_stream(buffer_size, X, y)
        accuracy = trainer.history("accuracy")
        common_accuracy = float(np.mean(accuracy[common_start:]))
        after_change = float(np.mean(accuracy[change_start:]))
        fit_seconds = trainer.summary()["total_fit_seconds"]
        retrains = trainer.fit_log[-1]["n_retrains"]

        results.append((buffer_size, retrains, common_accuracy,
                        after_change, fit_seconds))
        curves[buffer_size] = trainer.history("rolling_accuracy")
        print(f"buffer={buffer_size:4d}  retrains={retrains:3d}  "
              f"accuracy={common_accuracy:.3f}  "
              f"after_change={after_change:.3f}  fit={fit_seconds:.4f}s")

    with (RESULTS_DIR / "buffer_sweep.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["buffer_size", "retrains", "accuracy_common_period",
                         "accuracy_after_change", "total_fit_seconds"])
        writer.writerows(results)

    sizes = [row[0] for row in results]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sizes, [row[2] for row in results], "o-",
                 label="Common period")
    axes[0].plot(sizes, [row[3] for row in results], "s-",
                 label="After concept change")
    axes[0].set_ylabel("Prequential accuracy")
    axes[0].legend()

    axes[1].plot(sizes, [row[4] for row in results], "o-")
    axes[1].set_ylabel("Total fitting time (s)")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Buffer size")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "buffer_sweep.png", dpi=150)
    plt.close(fig)

    compare_models(
        curves[50], curves[400], ["Buffer 50", "Buffer 400"],
        title="Rolling accuracy across a concept change",
        ylabel=f"Accuracy (last {ROLLING_WINDOW} scored samples)",
        save_path=RESULTS_DIR / "buffer_curves.png",
        show=False,
    )
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
