# Code Cube — Task Breakdown (3-Person Team)

## Team Roles

| Person | Focus Area | Modules Owned |
|---|---|---|
| **Danushi** | Data & Preprocessing | `utils.py`, `io.py`, `preprocessing.py` |
| **Kalana** | Algorithms | `sort_search.py`, `rank.py`, `stats.py`, `benchmarking.py` |
| **Sanduni** | Evaluation & Optimization | `metrics.py`, `optim.py`, `pipeline.py` |

Shared/rotating: `numcompute/__init__.py`, `benchmarking.py`, `README.md`, `demo/quickstart.ipynb`, test review.

---

## Phase 0 — Kickoff (Day 1, ~1–2 hrs, whole team together)

- [ ] Agree on repo structure — must match the required layout in the assignment brief exactly:
      `numcompute/` (10 modules incl. `__init__.py`), `tests/`, `demo/quickstart.ipynb`, `README.md`, `pyproject.toml`
- [ ] Create GitHub repo and scaffold **`numcompute/__init__.py`** with empty module imports (public export
      surface is finalised in Phase 3 — but the file must exist from Day 1 so `pip install -e .` works)
- [ ] Set up `pyproject.toml` (or `setup.py`) — dependency: `numpy` only; dev dependency: `pytest`
- [ ] Agree on **API conventions**:
  - [ ] `X` shape convention: `(n_samples, n_features)`
  - [ ] `axis=0` = samples, `axis=1` = features, stated explicitly in every docstring
  - [ ] Error types: `ValueError` for shape mismatch, `TypeError` for dtype mismatch
  - [ ] Standard error message format, e.g. `f"Expected shape (n, {self.n_features_}), got {X.shape}"`
  - [ ] NaN policy per module (raise vs. propagate vs. auto-handle)
- [ ] Agree on **docstring template** — NumPy-style, and per the brief it must cover **all five**:
      Parameters / Returns / **Shapes** (explicit in/out shapes, not just implied by the axis convention) /
      Raises / Notes with **time and space complexity**
- [ ] **Confirm the library restriction as a hard rule**: plain Python + NumPy only —
      no scikit-learn, pandas, PyTorch, TensorFlow, or **SciPy**. This binds `tests/` too: expected values
      must be **hand-computed literals**, never `scipy.stats.rankdata` / `sklearn.metrics` comparisons
- [ ] **Define `pipeline.py`'s `Transformer`/`Estimator` protocol NOW** (Person C leads, ~15 min):
  ```python
  # Transformer protocol
  fit(self, X) -> self
  transform(self, X) -> X_out
  fit_transform(self, X) -> X_out

  # Estimator protocol
  fit(self, X, y) -> self
  predict(self, X) -> y_pred
  ```
- [ ] Set up shared test file convention: `tests/test_<module>.py`
- [ ] Create a `CONTRIBUTING.md` or top-of-README section capturing all of the above

---

# Phase 1 — Core Implementation (Days 2–3)

Each person implements their own modules against the Phase 0 conventions. Unit tests are written
alongside the code (listed per module below); Phase 2 is for finishing and hardening them.

## Danushi — Data & Preprocessing

### `utils.py` (build first — others depend on it)
- [ ] `logsumexp(x, axis=None)` — max-shifted for numerical stability
- [ ] Distance functions: `euclidean_distance`, `manhattan_distance` (vectorized, support broadcasting)
- [ ] Activation functions (e.g. `sigmoid`, `softmax` — softmax must be max-shifted)
- [ ] `batch_iter(X, batch_size)` — batching helper, no unnecessary copies
- [ ] Top-k helper (if shared logic needed by both `utils` and `sort_search`)
- [ ] Docstrings + complexity notes
- [ ] Tests: broadcasting shapes, extreme values (overflow check for `logsumexp`/`softmax`)

### `io.py`
- [ ] CSV loader using `np.genfromtxt`
- [ ] **Document the `loadtxt` vs `genfromtxt` decision** — the spec permits either; we choose
      `genfromtxt` for native missing-value support. Put a one-line rationale in the module docstring
      and the README so the deviation from the spec's first-listed option reads as deliberate
- [ ] Support custom delimiters (comma, tab, configurable)
- [ ] Missing value handling: skip rows OR fill with placeholder (make this a parameter)
- [ ] Return NumPy arrays with sensible dtype inference
- [ ] **Streaming/chunked reading for large files — REQUIRED.** The required package
      structure defines this module as `io.py # CSV reader (streaming/chunking), dtype handling`
- [ ] **Column typing bridge:** provide a documented way to split a loaded CSV into numeric vs.
      categorical (string) columns, so `OneHotEncoder` has a defined input. `genfromtxt` on a mixed CSV
      yields either an all-string array or a structured array — neither drops straight into a scaler.
      Agree the contract with Person C, since Phase 3's end-to-end `Pipeline` check depends on it
- [ ] Informative errors for malformed CSV / inconsistent column counts
- [ ] Tests: missing values, mixed delimiters, empty file, malformed rows, single-column CSV, chunked read
      matching a full read

### `preprocessing.py`
- [ ] `StandardScaler`: `fit`, `transform`, `fit_transform` — z-score; **guard divide-by-zero when std=0**
- [ ] `MinMaxScaler`: scale to arbitrary `(min, max)` range, default `(0,1)`
- [ ] `OneHotEncoder`: use `np.unique(return_inverse=True)`, no loops; decide + document unseen-category behavior at transform time
- [ ] *(Optional per spec)* `SimpleImputer`: fill NaN with constant or column statistic — note the demo
      notebook names an Imputer explicitly, so treat this as effectively required
- [ ] **Every class fully vectorized — no Python loops anywhere in this module**,(spec: "Fully vectorized using NumPy; avoid Python loops")
- [ ] All classes conform to the Phase 0 Transformer protocol
- [ ] Tests: zero-variance column, all-NaN column, single unique category in OneHotEncoder, non-contiguous input array

---

## Kalana — Algorithms

### `sort_search.py`
- [ ] Stable sort wrapper: `np.sort(..., kind='stable')`
- [ ] Multi-key sort (sort by multiple columns, e.g. via `np.lexsort`)
- [ ] `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
- [ ] `quickselect(values, k)` — manual/recursive implementation for educational purposes (loop/recursion OK here)
- [ ] `binary_search(sorted_array, x)` → returns `(insertion_index, exists: bool)`
- [ ] Tests: empty array, k=0, k=len(array), k > len(array), duplicate values, already-sorted vs. reverse-sorted input, binary search on value not present

### `rank.py`
- [ ] `rank(data, method='average'|'dense'|'ordinal')` — correct tie handling per method
- [ ] `percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')`
- [ ] Tests: all-equal array (max tie stress test), single tie among distinct values, percentile edge cases (q=0, q=100), each interpolation mode against a **hand-computed** example (SciPy comparison is banned)

### `stats.py`
- [ ] Mean, median, std, min, max — axis-aware (`axis=0`/`axis=1`/`None`)
- [ ] NaN-aware versions (`np.nanmean` etc.)
- [ ] **Welford's algorithm for streaming/single-pass variance — REQUIRED, not optional.** The required
      package structure defines this module as `stats.py # Streaming stats (Welford), histogram, quantiles`
- [ ] Histogram function (wrap or extend `np.histogram`, document bin edge behavior)
- [ ] Quantiles with NaN handling
- [ ] Clear docs on output shape per axis (e.g. axis=0 on a `(n,m)` array → shape `(m,)`)
- [ ] Tests: all-NaN row/column, 1D vs 2D input, non-contiguous stride input (e.g. `arr[::2]` or `arr.T`), Welford result matching a batch `np.var` on the same data

---

## Sanduni — Evaluation & Optimization

### `metrics.py`
- [ ] `accuracy(y_true, y_pred)`
- [ ] `precision(y_true, y_pred)`, `recall(y_true, y_pred)` — handle zero-division gracefully (return 0.0, don't raise)
- [ ] `f1(y_true, y_pred)` — depends on precision/recall zero-division handling being correct first
- [ ] `confusion_matrix(y_true, y_pred)`
- [ ] `mse(y_true, y_pred)` (regression)
- [ ] **Bonus**: `roc_curve`, `auc` for binary classification
- [ ] Tests: all-correct predictions, all-wrong predictions, single-class edge case (zero-division trigger), multi-class confusion matrix shape check

### `optim.py`
- [ ] `grad(f, x, h=1e-5, method='central'|'forward')` — finite-difference gradient
- [ ] `jacobian(F, x, ...)` — vector-valued function support
- [ ] *(Optional)* `line_search` — explicitly optional in the brief. **Need to decide as a team
      whether to implement it and record the decision**; if skipped, note it in the README so it reads as a
      conscious omission rather than an oversight
- [ ] Document why `h=1e-5` is a reasonable default and that it's not scale-invariant
- [ ] Tests: gradient of a known function (e.g. `f(x)=x**2` → `grad=2x`) checked against analytical solution, compare central vs. forward accuracy, jacobian shape check for `R^n -> R^m`

### `pipeline.py`
- [ ] Finalize `Transformer`/`Estimator` protocol (already sketched in Phase 0)
- [ ] `Pipeline` / `Compose`: **sequential** chain of `(name, transformer)` steps + optional final estimator
- [ ] `fit`, `transform`, `fit_transform`, `predict` (if final step is an estimator)
- [ ] **`FeatureUnion`: parallel composition** — run N transformers on the same `X` and horizontally
      concatenate their outputs. Required by the package structure
      (`pipeline.py # Transformer/Estimator API, Compose, FeatureUnion`) Needs its own row-count validation (all branches
      must return the same `n_samples`) and its own output-width contract
- [ ] Validate steps at construction time (informative error if a step doesn't implement the right protocol)
- [ ] Tests: pipeline of 2+ transformers, pipeline with a mock estimator at the end, error case (invalid step),
      `FeatureUnion` output width = sum of branch widths, `FeatureUnion` with mismatched row counts raises,
      `FeatureUnion` nested inside a `Pipeline`

---

## Phase 2 — Tests & Hardening (Day 4)

- [ ] Each person finishes the test list for their own modules (see per-module checklists above)
- [ ] **Vectorisation audit**: grep each module for `for`/`while` in core computations;
      the only sanctioned loops are `quickselect`, `batch_iter`, and Welford's streaming update
- [ ] **Banned-import check**: no `scipy`, `sklearn`, `pandas`, `torch`, `tensorflow` anywhere in
      `numcompute/` **or** `tests/`
- [ ] **Docstring completeness sweep**: every public function has Parameters / Returns / Shapes / Raises /
      complexity, per the Phase 0 template
- [ ] Confirm running count of unit tests is on track for ≥ 20

---

## Phase 3 — Integration (Day 5, whole team)

- [ ] Merge all branches into `main`/`dev`
- [ ] Run full `pytest` suite — resolve any cross-module API mismatches
- [ ] **Finalize `numcompute/__init__.py`**: decide and export the public API surface (this is what the
      README's "API overview" documents and what `import numcompute` gives a user)
- [ ] Confirm `Pipeline` works end-to-end with real `preprocessing.py` classes (not just mocks)
- [ ] Confirm `FeatureUnion` works with real transformers (e.g. `StandardScaler` on numeric columns
      unioned with `OneHotEncoder` on categorical columns from the same CSV — this is the payoff of
      Person A's column-typing bridge)
- [ ] Cross-review: each person reads through one other person's module for docstring/complexity completeness

---

## Phase 4 — Demo, Benchmarking, Docs (Day 6, split then merge)

### `demo/quickstart.ipynb` — must cover all six required demo steps
- [ ] 1. Load a CSV **with missing values** via `io.py`
- [ ] 2. Preprocess: `StandardScaler`, `Imputer`, `OneHotEncoder`
- [ ] 3. Demonstrate `sort_search` (top-k, quickselect, binary search) and `rank` (ties, percentiles)
- [ ] 4. Compute stats: mean/variance, histogram, quantiles
- [ ] 5. Estimate gradient/Jacobian via `optim.py` on a scalar **and** a vector function
- [ ] 6. Benchmark section: vectorized vs. Python-loop implementation, with a short timing table

### `benchmarking.py`
- [ ] Micro-bench harness (simple `timeit`-based wrapper is sufficient)
- [ ] At least one comparison: vectorized function vs. explicit Python loop equivalent
- [ ] Output as a reproducible script (not just notebook cells) — include environment notes (Python/NumPy version, OS)

### `README.md`
- [ ] Installation instructions (`pip install -e .`)
- [ ] Quick usage example per module
- [ ] Full API overview (matches the `__init__.py` export surface finalised in Phase 3)
- [ ] Performance table (pulled from `benchmarking.py` output)
- [ ] Record the `genfromtxt` rationale and the `line_search` include/skip decision

---

## Phase 5 — Buffer / Polish (Day 7)

- [ ] Confirm ≥ 20 tests total, covering: empty arrays, all-equal values, duplicates/ties, extreme `k`, NaNs, non-contiguous strides
- [ ] Run full test suite one final time
- [ ] Final banned-import check across the whole repo
- [ ] Proofread README + docstrings for consistency
- [ ] Confirm package installs cleanly in a fresh virtual environment
- [ ] Walk the deliverables checklist in the assignment brief line by line
- [ ] Submit

---

## Test Coverage Checklist (cross-reference against the brief's required edge cases)

| Edge Case | Modules to Cover |
|---|---|
| Empty arrays | `sort_search`, `stats`, `rank` |
| All-equal values | `rank`, `sort_search` (quickselect/topk) |
| Duplicates/ties | `rank`, `sort_search` |
| Extreme `k` (0, len, >len) | `sort_search` (`topk`, `quickselect`) |
| NaNs | `stats`, `preprocessing` (Imputer), `io` |
| Non-contiguous strides | `stats`, `preprocessing`, `sort_search` |
| Zero-division | `metrics` (precision/recall/f1) |
| Shape mismatches | `preprocessing`, `pipeline` (incl. `FeatureUnion` row mismatch), `metrics` |

---

## Deliverables Traceability

| Deliverable | Covered by |
|---|---|
| `numcompute/` with all 10 modules | Phase 0 (`__init__.py`) + Phase 1 (Persons A/B/C) + Phase 4 (`benchmarking.py`) |
| `tests/` ≥ 20 unit tests, edge cases | Per-module test items + Phase 2 + Phase 5 |
| `demo/quickstart.ipynb`, all six steps | Phase 4 |
| `README.md` — install, examples, API, perf table | Phase 4 |
| `pyproject.toml` / `setup.py` | Phase 0, verified Phase 5 |
| Docstrings incl. complexity | Phase 0 template + Phase 2 sweep |
| Reproducible benchmarks + environment notes | Phase 4 (`benchmarking.py`) |
