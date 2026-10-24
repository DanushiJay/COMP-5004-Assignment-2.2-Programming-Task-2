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

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The only runtime dependency is NumPy; `pytest` is the only development dependency.

## AI acknowledgement

Generative AI assistant was used for:

- breaking the assignment down into tasks and planning the work and commits.
- brainstorming design and test cases.
- discuss and explore the structure of some modules and their corresponding tests.
- generating the NumPy-style docstring format used in the code comments.
