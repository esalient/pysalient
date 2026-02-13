import numpy as np

from pysalient.evaluation import _evaluation_process_parallel as epp


def test_process_thresholds_parallel_sequential_path():
    probas = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    thresholds = [0.5, 0.7]

    results = epp.process_thresholds_parallel(
        threshold_list=thresholds,
        probas=probas,
        labels=labels,
        calculate_threshold_ci=False,
        threshold_ci_method="bootstrap",
        ci_alpha=0.05,
        bootstrap_rounds=10,
        bootstrap_seed=7,
        verbosity=1,
        n_jobs=1,
        parallel_thresholds=False,
    )

    assert len(results) == 2
    assert results[0]["threshold"] == 0.5
    assert "PPV" in results[0]
    assert "F1_Score_Upper_CI" in results[0]


def test_enable_parallel_evaluation_returns_bool():
    result = epp.enable_parallel_evaluation()
    assert isinstance(result, bool)
