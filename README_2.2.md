# numcompute_stream: Assignment 2.2

`numcompute_stream` is the individual Assignment 2.2 extension of the group's NumCompute toolkit (Assignment 2.1). It learns from data one chunk at a time: a decision tree that retrains when its buffer fills, preprocessing that updates incrementally, streaming statistics and metrics, and tools for evaluating and plotting a model's performance.

It builds on the group package `numcompute`, which is **not modified**. For the Assignment 2.1 documentation, see [`README.md`](README.md).

## Setup

Use Python 3.10 or later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The core numerical code uses NumPy. The streaming plots also require Matplotlib. If it is not included in your installed dependencies, run:

```bash
python -m pip install matplotlib
```

## Modules

| File               | What it does                                                                           |
| ------------------ | -------------------------------------------------------------------------------------- |
| `tree.py`          | Decision tree with Gini or entropy splitting and buffered `partial_fit`                |
| `preprocessing.py` | Incremental scaling, missing-value handling and one-hot encoding                       |
| `stats.py`         | Running statistics, quantiles and histograms                                           |
| `metrics.py`       | Accuracy, precision, recall, F1, AUC and confusion matrices, including rolling metrics |
| `pipeline.py`      | Connects preprocessing steps with a streaming model                                    |
| `stream.py`        | Tests and trains each chunk, then records metrics, time and memory                     |
| `visualise.py`     | Plots accuracy over time, model comparisons and predictions                            |

## Example: train a model in chunks

```python
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=100)),
])
trainer = StreamTrainer(pipeline, window=100)

for start in range(0, len(X), 25):
    trainer.process_chunk(X[start:start + 25], y[start:start + 25])

print(trainer.summary())
```

The trainer uses **test-then-train** evaluation: it predicts a chunk before updating the model with that chunk. The earliest chunks cannot be scored until the first tree has been trained.

## Tests

Run the tests from the repository root:

```bash
python -m pytest -q
```

The streaming tests cover normal use and edge cases, including empty chunks, missing values, changing categories, incorrect prediction shapes and predictions before fitting. Each module has a matching test file in `tests/`.

## Benchmarks

Run the two benchmark scripts from the repository root:

```bash
python -m benchmark.bench_buffer
python -m benchmark.bench_split
```

- **`bench_buffer.py`** compares buffer sizes on a synthetic stream whose decision rule changes halfway through. It records accuracy, number of retrains and total fitting time.
- **`bench_split.py`** compares Python-loop and vectorised tree operations and checks that their results agree.

Both scripts save CSV results and plots in `benchmark/results/`. Timings depend on the computer and current workload.

## Demo notebook

[`demo/stream_demo.ipynb`](demo/stream_demo.ipynb) uses the Palmer Penguins dataset to demonstrate the complete workflow: loading data, preprocessing, training in chunks, logging results and plotting performance. It also compares buffer sizes and shuffled versus species-ordered data.

It reads `demo/data/penguins.csv`. Open the notebook from the `demo/` folder and run all cells. Figures are saved in `demo/figures/`.

## Design choices and limitations

- The decision tree retrains on buffered data instead of updating its splits after every sample. It continues using the previous tree between retraining events.
- The pipeline keeps preprocessing consistent with the tree used for prediction. New categories may change the number of encoded columns.
- Some streaming estimates use a recent window or an approximation to avoid storing the whole data stream. For example, cumulative streaming AUC is binned.
- Logged memory is the NumPy array-buffer memory reachable from the model, not the total RAM used by Python.
- The ordered Penguins experiment shows the effect of **class distribution and data order**. The synthetic buffer benchmark separately tests a change in the decision rule (**concept drift**).

## AI acknowledgement

AI tools were used for planning, test suggestions, debugging discussions, and editing the benchmarks.
