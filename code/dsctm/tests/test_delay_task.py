import numpy as np

from dsctm.data.synthetic import make_delay_dependency


def test_delay_task_is_balanced_grouped_and_has_exact_signal_positions():
    ds = make_delay_dependency(delay=7, n_subjects=5, windows_per_subject=8, T=32, seed=2)
    assert ds.X.shape == (40, 32, 4)
    np.testing.assert_array_equal(np.bincount(ds.y), [20, 20])
    assert np.all(ds.X[:, -1, 1] == 1.0)
    assert np.all(np.abs(ds.X[:, -8, 0]) == 3.0)
    assert np.all(np.abs(ds.X[:, -1, 0]) == 3.0)
