"""Tests for numcompute_stream.stream (StreamTrainer, memory_footprint)."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer, memory_footprint
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def stream():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def make(**kw):
    return StreamTrainer(DecisionTreeClassifier(max_depth=4,
                                                buffer_size=100), **kw)


def run(trainer, X, y, size=20):
    for i in range(0, len(X), size):
        trainer.process_chunk(X[i:i + size], y[i:i + size])
    return trainer


def test_first_chunks_are_unscored_until_the_tree_exists(stream):
    X, y = stream
    t = run(make(), X, y)
    scored = [r["scored"] for r in t.score_log]
    assert scored[:5] == [False] * 5       # 100 samples fill the buffer
    assert all(scored[5:])
    assert np.isnan(t.score_log[0]["accuracy"])
    assert np.isnan(t.score_log[0]["cumulative_accuracy"])


def test_scoring_happens_before_learning(stream):
    X, y = stream
    t = make()
    t.fit_chunk(X[:100], y[:100])           # tree trained
    before = t.model.predict(X[100:140])
    rec = t.score_chunk(X[100:140], y[100:140])
    assert rec["accuracy"] == float(np.mean(before == y[100:140]))
    assert t.model.n_buffered_ == 0         # scoring did not train


def test_cumulative_accuracy_matches_manual_count(stream):
    X, y = stream
    t = make()
    right = total = 0
    for i in range(0, 400, 20):
        xs, ys = X[i:i + 20], y[i:i + 20]
        if t.model.n_retrains_:
            right += int((t.model.predict(xs) == ys).sum())
            total += 20
        t.process_chunk(xs, ys)
    assert abs(t.score_log[-1]["cumulative_accuracy"] - right / total) < 1e-12
    assert t.n_scored_ == total


def test_rolling_accuracy_uses_only_the_window():
    class Always:
        def __init__(self):
            self.answer = 1

        def partial_fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), self.answer)

    m = Always()
    t = StreamTrainer(m, window=4)
    X = np.zeros((4, 1))
    t.score_chunk(X, np.zeros(4, dtype=int))        # all wrong
    t.score_chunk(X, np.ones(4, dtype=int))         # all right
    last = t.score_log[-1]
    assert last["rolling_accuracy"] == 1.0
    assert last["cumulative_accuracy"] == 0.5


def test_fit_log_fields_and_retrain_counts(stream):
    X, y = stream
    t = run(make(), X, y)
    rec = t.fit_log[-1]
    assert set(rec) == {"chunk", "n_samples", "fit_seconds",
                        "memory_bytes", "n_retrains", "n_buffered"}
    assert rec["n_retrains"] == 4 and rec["n_buffered"] == 0
    assert rec["fit_seconds"] >= 0 and rec["memory_bytes"] > 0
    assert [r["chunk"] for r in t.fit_log] == list(range(20))


def test_memory_grows_while_the_buffer_fills(stream):
    X, y = stream
    t = make()
    t.fit_chunk(X[:20], y[:20])
    small = t.fit_log[-1]["memory_bytes"]
    t.fit_chunk(X[20:80], y[20:80])
    assert t.fit_log[-1]["memory_bytes"] > small


def test_history_returns_float_arrays(stream):
    X, y = stream
    t = run(make(), X, y)
    acc = t.history("cumulative_accuracy")
    assert acc.shape == (20,) and acc.dtype == float
    assert np.isnan(acc[0]) and 0.0 <= acc[-1] <= 1.0
    assert t.history("memory_bytes").shape == (20,)
    with pytest.raises(KeyError, match="nope"):
        t.history("nope")


def test_history_before_anything_logged_raises():
    with pytest.raises(ValueError, match="Nothing logged"):
        make().history("accuracy")


def test_macro_f1_is_logged(stream):
    X, y = stream
    t = run(make(), X, y)
    assert 0.5 < t.score_log[-1]["macro_f1"] <= 1.0


def test_run_over_a_generator(stream):
    X, y = stream
    chunks = ((X[i:i + 50], y[i:i + 50]) for i in range(0, 400, 50))
    t = make().run(chunks)
    assert len(t.score_log) == len(t.fit_log) == 8


def test_works_with_a_pipeline(stream):
    X, y = stream
    pipe = Pipeline([("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(max_depth=4,
                                                     buffer_size=100))])
    t = run(StreamTrainer(pipe), X, y)
    assert t.summary()["cumulative_accuracy"] > 0.8
    assert t.fit_log[-1]["memory_bytes"] > 0
    assert t.fit_log[-1]["n_retrains"] == 4
    assert t.fit_log[-1]["n_buffered"] == 0


def test_pipeline_retrain_count_comes_from_tree(stream):
    X, y = stream
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=100)
    trainer = StreamTrainer(Pipeline([
        ("scale", StandardScaler()),
        ("tree", tree),
    ]))
    trainer.fit_chunk(X[:60], y[:60])
    assert trainer.fit_log[-1]["n_retrains"] == 0
    assert trainer.fit_log[-1]["n_buffered"] == 60

    trainer.fit_chunk(X[60:100], y[60:100])
    assert trainer.fit_log[-1]["n_retrains"] == 1
    assert trainer.fit_log[-1]["n_buffered"] == 0


def test_bad_prediction_shape_is_rejected():
    class WrongShape:
        def partial_fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros((len(X), 1), dtype=int)

    trainer = StreamTrainer(WrongShape())
    with pytest.raises(ValueError, match="Expected predictions with shape"):
        trainer.score_chunk(np.zeros((3, 2)), np.zeros(3, dtype=int))
    assert trainer.n_scored_ == 0
    assert trainer.score_log == []


def test_wrong_prediction_length_is_rejected():
    class WrongLength:
        def partial_fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(1, dtype=int)

    trainer = StreamTrainer(WrongLength())
    with pytest.raises(ValueError, match="Expected predictions with shape"):
        trainer.score_chunk(np.zeros((3, 2)), np.zeros(3, dtype=int))
    assert trainer.n_scored_ == 0


def test_model_without_n_retrains_logs_nan():
    class Plain:
        def partial_fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    t = StreamTrainer(Plain())
    t.fit_chunk(np.zeros((3, 1)), np.zeros(3, dtype=int))
    assert t.fit_log[0]["n_retrains"] is None
    assert np.isnan(t.history("n_retrains")[0])


def test_empty_chunk_is_logged_and_harmless(stream):
    X, y = stream
    t = run(make(), X[:200], y[:200])
    n = t.n_scored_
    rec = t.score_chunk(X[:0], y[:0])
    t.fit_chunk(X[:0], y[:0])
    assert rec["n_samples"] == 0 and rec["scored"] is False
    assert t.n_scored_ == n
    assert t.fit_log[-1]["n_samples"] == 0


def test_bad_chunks_rejected(stream):
    X, y = stream
    t = make()
    with pytest.raises(ValueError, match="2-D"):
        t.fit_chunk(y[:5], y[:5])
    with pytest.raises(ValueError, match="1-D"):
        t.score_chunk(X[:5], X[:5])
    with pytest.raises(ValueError, match="rows"):
        t.fit_chunk(X[:5], y[:4])


def test_model_needs_partial_fit_and_predict():
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(object())

    class NoPredict:
        def partial_fit(self, X, y):
            return self

    with pytest.raises(TypeError, match="predict"):
        StreamTrainer(NoPredict())


def test_reset_clears_logs_but_keeps_the_model(stream):
    X, y = stream
    t = run(make(), X, y)
    assert t.reset() is t
    assert t.score_log == [] and t.fit_log == [] and t.n_scored_ == 0
    assert t.model.n_retrains_ == 4
    t.score_chunk(X[:20], y[:20])
    assert t.score_log[0]["chunk"] == 0


def test_summary_fields(stream):
    X, y = stream
    s = run(make(), X, y).summary()
    assert s["chunks_scored"] == s["chunks_fitted"] == 20
    assert s["samples_scored"] == 300
    assert s["total_fit_seconds"] > 0 and s["memory_bytes"] > 0
    empty = make().summary()
    assert np.isnan(empty["cumulative_accuracy"])
    assert empty["memory_bytes"] == 0


def test_memory_footprint_counts_arrays_once():
    a = np.zeros(100)                       # 800 bytes
    assert memory_footprint(a) == 800
    assert memory_footprint([a, a, a[:10]]) == 800     # same memory
    assert memory_footprint({"x": a, "y": np.ones(10)}) == 880


def test_memory_footprint_follows_attributes_and_handles_cycles():
    class Node:
        pass

    n = Node()
    n.data = np.zeros(50)
    n.self = n
    n.items = [(np.zeros(5), "text"), 3]
    assert memory_footprint(n) == 50 * 8 + 5 * 8
    assert memory_footprint(None) == 0
