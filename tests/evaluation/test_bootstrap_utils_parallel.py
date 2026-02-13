import numpy as np
import pytest

from pysalient.evaluation._bootstrap_utils_parallel import (
    _bootstrap_worker,
    calculate_bootstrap_ci_parallel,
)


class _DummyPool:
    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, iterable):
        return [func(item) for item in iterable]


def test_parallel_bootstrap_small_deterministic_workload():
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=float)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("pysalient.evaluation._bootstrap_utils_parallel.Pool", _DummyPool)
        lower, upper = calculate_bootstrap_ci_parallel(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=lambda yt, yp: float(np.mean((yp >= 0.5) == yt)),
            n_rounds=20,
            alpha=0.05,
            seed=7,
            n_jobs=2,
        )
    assert 0.0 <= lower <= upper <= 1.0


def test_parallel_worker_distribution():
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    out1 = _bootstrap_worker((0, 3), y_true, y_pred, lambda yt, yp: 0.5, seed=1)
    out2 = _bootstrap_worker((3, 2), y_true, y_pred, lambda yt, yp: 0.5, seed=1)

    assert out1.shape == (3,)
    assert out2.shape == (2,)
    assert np.all(out1 == 0.5)
    assert np.all(out2 == 0.5)


def test_parallel_metric_failure_returns_nan():
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("pysalient.evaluation._bootstrap_utils_parallel.Pool", _DummyPool)
        with pytest.warns(RuntimeWarning, match="bootstrap rounds failed"):
            lower, upper = calculate_bootstrap_ci_parallel(
                y_true=y_true,
                y_pred=y_pred,
                metric_func=lambda yt, yp: (_ for _ in ()).throw(RuntimeError("bad")),
                n_rounds=10,
                alpha=0.05,
                seed=7,
                n_jobs=2,
            )

    assert np.isnan(lower)
    assert np.isnan(upper)
