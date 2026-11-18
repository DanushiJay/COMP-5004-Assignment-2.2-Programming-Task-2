"""Chunk-by-chunk training and evaluation of a streaming model.

``StreamTrainer`` wraps any model with ``partial_fit`` and ``predict``
(the buffered ``DecisionTreeClassifier`` or a streaming ``Pipeline``).
It uses the test-then-train (prequential) scheme: each chunk is scored
with the model as it is *before* seeing that chunk, then the model learns
from the chunk. Every call adds one record to a log, so accuracy, timing
and memory can be plotted afterwards with ``numcompute_stream.visualise``.

    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    for X_chunk, y_chunk in chunks:
        trainer.score_chunk(X_chunk, y_chunk)   # before learning
        trainer.fit_chunk(X_chunk, y_chunk)     # then learn
    trainer.history("cumulative_accuracy")
"""

import time

import numpy as np

from numcompute.preprocessing import NotFittedError

from .metrics import StreamingAccuracy, StreamingF1

__all__ = ["StreamTrainer", "memory_footprint"]


def memory_footprint(obj):
    """Bytes held in NumPy arrays reachable from ``obj``.

    Walks attributes, lists, tuples, dicts and sets once each. Only array
    buffers are counted, which is where a model keeps its data; small
    Python objects are ignored. Views that share memory are counted once.
    """
    seen, owners, total = set(), set(), 0
    stack = [obj]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            base = item
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in owners:  # a view shares its owner's memory
                owners.add(id(base))
                total += base.nbytes
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            stack.extend(vars(item).values())
    return int(total)


def _as_chunk(X, y):
    """Validated ``(X, y)`` arrays: 2-D features, 1-D labels, same rows."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    return X, y


class StreamTrainer:
    """Train and evaluate a streaming model one chunk at a time.

    Parameters
    ----------
    model : object
        Needs ``partial_fit(X, y)`` and ``predict(X)``. ``n_retrains_``
        and ``n_buffered_`` are logged when the model has them.
    window : int or None
        If given, ``rolling_accuracy`` covers only the last ``window``
        scored samples (without it, it equals the cumulative value).
        Cumulative metrics always cover every scored sample.

    Attributes
    ----------
    score_log, fit_log : list of dict
        One record per ``score_chunk`` / ``fit_chunk`` call (see below).
    n_scored_ : int
        Samples scored so far (chunks scored before the model was first
        trained are not counted).

    Notes
    -----
    A ``score_chunk`` record has ``chunk``, ``n_samples``, ``scored``,
    ``accuracy``, ``cumulative_accuracy``, ``rolling_accuracy``,
    ``macro_f1`` and ``score_seconds``. While the model has no trained
    tree yet, ``scored`` is False and the metrics are NaN or carried over
    from earlier chunks. A ``fit_chunk`` record has ``chunk``,
    ``n_samples``, ``fit_seconds``, ``memory_bytes``, ``n_retrains`` and
    ``n_buffered``. Empty chunks are logged with ``n_samples == 0`` and
    do nothing else. Memory is the array bytes held by the model.
    """

    def __init__(self, model, window=None):
        for name in ("partial_fit", "predict"):
            if not callable(getattr(model, name, None)):
                raise TypeError(f"model needs a {name} method")
        self.model = model
        self.window = window
        self.reset()

    def reset(self):
        """Clear the logs and metrics (the model is left as it is)."""
        self._cum_acc = StreamingAccuracy()
        self._roll_acc = StreamingAccuracy(window=self.window)
        self._f1 = StreamingF1(average="macro")
        self.score_log, self.fit_log = [], []
        self.n_scored_ = 0
        self._n_score_calls = 0
        self._n_fit_calls = 0
        return self

    # ------------------------------------------------------------ fit
    def fit_chunk(self, X, y):
        """Let the model learn from one chunk and log the cost.

        Returns the log record (a dict).
        """
        X, y = _as_chunk(X, y)
        record = {"chunk": self._n_fit_calls, "n_samples": int(len(y))}
        self._n_fit_calls += 1
        start = time.perf_counter()
        if len(y):
            self.model.partial_fit(X, y)
        record["fit_seconds"] = time.perf_counter() - start
        record["memory_bytes"] = memory_footprint(self.model)

        # Retraining belongs to the estimator, not its outer pipeline.
        tracked_model = self.model
        if (getattr(self.model, "_final_is_estimator", False)
                and hasattr(self.model, "steps")):
            tracked_model = self.model.steps[-1][1]
        record["n_retrains"] = getattr(tracked_model, "n_retrains_", None)
        record["n_buffered"] = getattr(
            self.model, "n_buffered_",
            getattr(tracked_model, "n_buffered_", None))
        self.fit_log.append(record)
        return record

    # ---------------------------------------------------------- score
    def score_chunk(self, X, y):
        """Score one chunk with the current model, without training on it.

        Returns the log record (a dict). If the model has not been
        trained yet (the tree is still filling its first buffer), the
        chunk is logged with ``scored=False`` and NaN accuracy.
        """
        X, y = _as_chunk(X, y)
        record = {"chunk": self._n_score_calls, "n_samples": int(len(y)),
                  "scored": False, "accuracy": np.nan}
        self._n_score_calls += 1
        start = time.perf_counter()
        pred = None
        if len(y):
            try:
                pred = np.asarray(self.model.predict(X))
            except NotFittedError:
                pred = None
            if pred is not None and pred.shape != y.shape:
                raise ValueError(
                    f"Expected predictions with shape {y.shape}, "
                    f"got {pred.shape}")
        record["score_seconds"] = time.perf_counter() - start
        if pred is not None:
            record["scored"] = True
            record["accuracy"] = float(np.mean(pred == y))
            for metric in (self._cum_acc, self._roll_acc, self._f1):
                metric.update(y, pred)
            self.n_scored_ += len(y)
        record["cumulative_accuracy"] = self._value(self._cum_acc)
        record["rolling_accuracy"] = self._value(self._roll_acc)
        record["macro_f1"] = self._value(self._f1)
        self.score_log.append(record)
        return record

    @staticmethod
    def _value(metric):
        return np.nan if metric.n_samples_seen_ == 0 else metric.result()

    # -------------------------------------------------------- running
    def process_chunk(self, X, y):
        """Test-then-train on one chunk. Returns ``(score, fit)`` records."""
        return self.score_chunk(X, y), self.fit_chunk(X, y)

    def run(self, chunks):
        """Test-then-train over an iterable of ``(X, y)`` chunks.

        Returns ``self`` so the logs can be read straight away.
        """
        for X, y in chunks:
            self.process_chunk(X, y)
        return self

    # -------------------------------------------------------- results
    def history(self, key):
        """One logged value per chunk, as a float array.

        ``key`` is any field of the score or fit records, for example
        ``"cumulative_accuracy"``, ``"fit_seconds"`` or ``"memory_bytes"``.
        """
        for log in (self.score_log, self.fit_log):
            if log and key in log[0]:
                return np.array([r[key] if r[key] is not None else np.nan
                                 for r in log], dtype=float)
        if not (self.score_log or self.fit_log):
            raise ValueError("Nothing logged yet; call score_chunk or "
                             "fit_chunk first")
        raise KeyError(f"Unknown log field {key!r}")

    def summary(self):
        """Final numbers as a dict (NaN for what was never measured)."""
        last = self.score_log[-1] if self.score_log else {}
        return {
            "chunks_scored": len(self.score_log),
            "chunks_fitted": len(self.fit_log),
            "samples_scored": self.n_scored_,
            "cumulative_accuracy": last.get("cumulative_accuracy", np.nan),
            "macro_f1": last.get("macro_f1", np.nan),
            "total_fit_seconds": float(sum(r["fit_seconds"]
                                           for r in self.fit_log)),
            "memory_bytes": (self.fit_log[-1]["memory_bytes"]
                             if self.fit_log else 0),
        }
