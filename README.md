# Code-Cube-COMP-5004-Assignment-2.1-Programming-Task-1

NumCompute — a modular scientific computing toolkit built on plain Python and NumPy.

## Overview

NumCompute is a small library of reusable, vectorised building blocks for machine learning workflows. It does not run as an application; you import the modules and call them from Python.

- `io` and `preprocessing` load CSV data and prepare it (scaling, imputing, encoding).
- `sort_search`, `rank` and `stats` provide sorting, top-k, ranking and descriptive statistics.
- `metrics` scores predictions (accuracy, precision, recall, F1, confusion matrix, MSE, ROC/AUC).
- `optim` estimates gradients and Jacobians with finite differences, and has a backtracking line search.
- `pipeline` chains steps with `Pipeline` and joins parallel branches with `FeatureUnion`.
- `utils` and `benchmarking` hold shared helpers and the loop-vs-vectorised timing harness.

Preprocessing steps follow a common `fit` / `transform` interface and models follow `fit` / `predict`. This lets `Pipeline` run them in order, passing each step's output to the next.

## Installation

NumCompute needs Python 3.10 or later and NumPy. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Modules are used through their namespace, for example `from numcompute import stats`, then `stats.mean(X)`.

### Development setup

```bash
pip install -e ".[dev]"
pytest
```

The only runtime dependency is NumPy; `pytest` is the only development dependency.

## Examples

The examples use the Palmer Penguins file in `demo/data/penguins.csv` and are run from the repository root. The full walkthrough is in [`demo/quickstart.ipynb`](demo/quickstart.ipynb).

### Load, clean and prepare data

```python
import numpy as np
from numcompute import io
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

# Numeric columns become floats (blank fields are NaN), text columns stay strings.
X_num, X_cat = io.load_mixed_csv("demo/data/penguins.csv", numeric_cols=[2, 3, 4, 5],
                                 categorical_cols=[0, 1, 6], skip_header=1)
# X_num.shape == (344, 4), with 8 missing values

pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler())])
Z = pipe.fit_transform(X_num)            # (344, 4), each column mean 0, std 1

X_cat = np.where(X_cat == "", "unknown", X_cat)
H = OneHotEncoder().fit_transform(X_cat) # (344, 9): one 0/1 column per category
```

### Statistics

```python
from numcompute import stats

stats.mean(X_num, axis=0)                     # [nan nan nan nan]  NaN propagates by default
stats.mean(X_num, axis=0, nan_policy="omit")  # [43.92 17.15 200.92 4201.75]
stats.quantile(X_num, [0.25, 0.5, 0.75], axis=0, nan_policy="omit").shape  # (3, 4)
counts, edges = stats.histogram(X_num[:, 2], bins=5)  # NaN is dropped; 6 edges
```

### Sorting, searching and ranking

```python
from numcompute import rank, sort_search

v = np.array([3.0, 9.0, 1.0, 9.0, 4.0])
sort_search.topk(v, 2)                          # (array([9., 9.]), array([3, 1]))
sort_search.quickselect(v, 2, seed=0)           # 4.0, the 3rd smallest
sort_search.binary_search([1, 3, 3, 7], [3, 5]) # (array([1, 3]), array([ True, False]))

rank.rank(v, "average")   # [2.  4.5 1.  4.5 3. ]  tied values share the mean rank
rank.rank(v, "dense")     # [2 4 1 4 3]
rank.rank(v, "ordinal")   # [2 4 1 5 3]
rank.percentile(v, [25, 50], interpolation="midpoint")  # [3. 4.]
```

### Metrics, gradients and stable functions

```python
from numcompute import metrics, optim, utils

metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0])          # 0.75
metrics.confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])  # [[2 0]
                                                      #  [1 1]]

optim.grad(lambda x: x[0] ** 2 + 3 * x[1], np.array([1.0, 2.0]))  # [2. 3.]
optim.jacobian(lambda x: np.array([x[0] * x[1], x[0] + x[1]]),
               np.array([1.0, 2.0]))                               # [[2. 1.]
                                                                   #  [1. 1.]]

utils.softmax(np.array([1000.0, 1001.0, 1002.0]))  # [0.09 0.24 0.67], no overflow
```

## API overview

| Module | Public API |
|---|---|
| `io` | `load_csv`, `iter_csv_chunks`, `read_header`, `split_columns`, `load_mixed_csv`, `DEFAULT_MISSING` |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder` (each with `fit`, `transform`, `fit_transform`; scalers also have `inverse_transform`), `NotFittedError` |
| `sort_search` | `stable_sort`, `multi_key_sort`, `topk`, `quickselect`, `binary_search` |
| `rank` | `rank` (`average`, `dense`, `ordinal`), `percentile` (`linear`, `lower`, `higher`, `midpoint`) |
| `stats` | `mean`, `median`, `var`, `std`, `min`, `max`, `quantile`, `histogram`, `RunningStats` |
| `metrics` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `roc_curve`, `auc` |
| `optim` | `grad`, `jacobian` (`central` or `forward`), `line_search` |
| `pipeline` | `Transformer`, `Estimator`, `Pipeline`, `FeatureUnion` |
| `utils` | `as_float_array`, `logsumexp`, `log_softmax`, `softmax`, `sigmoid`, `relu`, `pairwise_diff`, `euclidean_distance`, `manhattan_distance`, `batch_iter` |
| `benchmarking` | `time_function`, `compare`, `environment_info`, `format_table`, `run_all`, `run_size_sweep` |

Conventions shared across the package:

- **Shapes:** 2-D data is `(n_samples, n_features)`. Every docstring lists parameters, return values, shapes, exceptions and time/space complexity.
- **Axis:** `axis=None` returns a scalar, `axis=0` one value per column `(n_features,)` and `axis=1` one value per row `(n_samples,)`.
- **NaN:** `stats` functions take `nan_policy="propagate" | "omit" | "raise"`. `topk`, `quickselect`, `rank` and `percentile` raise a `ValueError` on NaN because NaN has no defined order; `stable_sort` and `multi_key_sort` place NaN last.
- **Errors:** `ValueError` for invalid values or shapes and `TypeError` for wrong types, with a message stating what was expected and what was received. Calling `transform` before `fit` raises `NotFittedError`, a subclass of `ValueError`.

## Performance

`numcompute.benchmarking` times each vectorised function against a plain Python loop that computes the same result with the same time complexity, so the difference comes from vectorisation. Both results are checked to agree before timing, inputs come from a fixed seed, and each time is the fastest of several rounds.

The table below is the output of the benchmark section in [`demo/quickstart.ipynb`](demo/quickstart.ipynb) (`run_all(repeat=3)`). The same benchmarks can also be run from a terminal, which additionally shows how the speedup grows with input size:

```bash
python -m numcompute.benchmarking
```

Environment: Python 3.15.1, NumPy 2.5.4, macOS 26.5.1 (Apple silicon, arm64), 15 CPU cores.

| Benchmark | Vectorised (ms) | Python loop (ms) | Speedup |
|---|---:|---:|---:|
| stats.mean, (2000, 50), axis=0 | 0.024 | 1.561 | 64x |
| stats.var, (2000, 50), axis=0 | 0.090 | 4.260 | 47x |
| rank.rank (average), 5000 values | 0.177 | 3.203 | 18x |
| sort_search.binary_search, 10000 queries in 1000000 values | 0.223 | 8.572 | 38x |
| metrics.confusion_matrix, 100000 samples, 5 classes | 0.841 | 1.913 | 2x |
| utils.euclidean_distance, (200, 10) vs (200, 10) | 0.472 | 11.477 | 24x |
| utils.softmax, (2000, 10), axis=1 | 0.122 | 1.048 | 9x |

Times depend on the machine and its load, so absolute numbers will differ; the speedups are the comparable result.

## AI acknowledgement

Generative AI assistant was used for:

- breaking the assignment down into tasks and planning the work and commits.
- brainstorming design and test cases.
- discuss and explore the structure of some modules and their corresponding tests.
- generating the NumPy-style docstring format used in the code comments.
