"""Compare Python-loop and vectorised decision-tree operations.

Run from the repository root with ``python -m benchmark.bench_split``.
Results are saved in ``benchmark/results``.
"""

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier, _best_split

SIZES = (200, 1000, 5000, 20000)
N_FEATURES = 5
REPEATS = 3
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def gini(counts, n):
    return 1.0 - sum((count / n) ** 2 for count in counts) if n else 0.0


def best_split_loop(X, y_idx, n_classes):
    """Find the best Gini split, checking thresholds in a Python loop."""
    best = (None, None, float("inf"))
    n, n_features = X.shape

    for feature in range(n_features):
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        labels = y_idx[order]
        left = [0] * n_classes
        right = np.bincount(labels, minlength=n_classes).tolist()

        for i in range(n - 1):
            label = labels[i]
            left[label] += 1
            right[label] -= 1
            if values[i] == values[i + 1]:
                continue

            n_left = i + 1
            score = (n_left * gini(left, n_left)
                     + (n - n_left) * gini(right, n - n_left)) / n
            if score < best[2]:
                threshold = values[i] / 2.0 + values[i + 1] / 2.0
                best = (feature, threshold, score)

    return best


def predict_loop(tree, X):
    """Predict one sample at a time by walking the fitted tree."""
    nodes = tree._tree
    predictions = []
    for row in X:
        node = 0
        while nodes["feature"][node] != -1:
            feature = nodes["feature"][node]
            node = (nodes["left"][node] if row[feature] <= nodes["threshold"][node]
                    else nodes["right"][node])
        predictions.append(int(np.argmax(nodes["value"][node])))
    return tree.classes_[predictions]


def benchmark(func):
    """Return the best elapsed time and a result from repeated calls."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    return min(times), result


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0
    return X, y.astype(int)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for n in SIZES:
        X, y = make_data(n)
        classes, y_idx = np.unique(y, return_inverse=True)
        onehot = np.eye(len(classes))[y_idx]
        features = np.arange(N_FEATURES)

        loop_time, loop_split = benchmark(
            lambda: best_split_loop(X, y_idx, len(classes))
        )
        vector_time, vector_split = benchmark(
            lambda: _best_split(X, onehot, features, "gini")
        )
        assert loop_split[0] == vector_split[0]
        np.testing.assert_allclose(loop_split[1:], vector_split[1:], atol=1e-12)
        results.append(("split", n, loop_time, vector_time))

        tree = DecisionTreeClassifier(max_depth=6).fit(X, y)
        loop_time, loop_pred = benchmark(lambda: predict_loop(tree, X))
        vector_time, vector_pred = benchmark(lambda: tree.predict(X))
        np.testing.assert_array_equal(loop_pred, vector_pred)
        results.append(("predict", n, loop_time, vector_time))

    with (RESULTS_DIR / "split_vs_loop.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "n_samples", "loop_seconds",
                         "vectorised_seconds", "speedup"])
        for task, n, loop_time, vector_time in results:
            writer.writerow([task, n, loop_time, vector_time,
                             loop_time / vector_time])
            print(f"{task:7s} n={n:5d}  "
                  f"loop={loop_time:.5f}s  vector={vector_time:.5f}s  "
                  f"speedup={loop_time / vector_time:.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, ("split", "predict")):
        points = [row for row in results if row[0] == task]
        sizes = [row[1] for row in points]
        ax.plot(sizes, [row[2] for row in points], "o-", label="Python loop")
        ax.plot(sizes, [row[3] for row in points], "o-", label="Vectorised")
        ax.set(xscale="log", yscale="log", xlabel="Samples",
               ylabel="Time (s)", title=task.capitalize())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "split_vs_loop.png", dpi=150)
    plt.close(fig)
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
