import numpy as np

from dsctm.data.studentlife import _ffill, _load_cache
from dsctm.train.trainer import fit_normalizer


def test_forward_fill_never_uses_a_future_observation():
    x = np.array([np.nan, np.nan, 7.0, np.nan, 9.0, np.nan])
    filled, missing = _ffill(x)
    np.testing.assert_array_equal(filled, [0.0, 0.0, 7.0, 7.0, 9.0, 9.0])
    assert missing == 4 / 6


def test_forward_fill_all_missing_is_zero():
    filled, missing = _ffill(np.array([np.nan, np.nan]))
    np.testing.assert_array_equal(filled, [0.0, 0.0])
    assert missing == 1.0


def test_train_mean_normalizer_ignores_missing_and_padding():
    X = np.array([[[1.0], [np.nan], [99.0]], [[3.0], [5.0], [99.0]]], dtype=np.float32)
    mean, std = fit_normalizer(X, lengths=np.array([2, 2]))
    assert mean[0] == 3.0  # observed valid values are 1,3,5
    assert std[0] > 0


def test_early_v2_cache_filename_recovers_unambiguous_version(tmp_path):
    path = tmp_path / "studentlife_causal_ffill_v2.npz"
    np.savez_compressed(
        path, X=np.zeros((1, 2, 8), np.float32), y=np.array([0]),
        subject_id=np.array(["u00"]), timestamp=np.array([0]),
        missingness=np.array([1.0]),
    )
    ds = _load_cache(path)
    assert ds.version == "studentlife-v2-causal_ffill"


def test_ambiguous_metadata_free_cache_remains_legacy(tmp_path):
    path = tmp_path / "studentlife.npz"
    np.savez_compressed(
        path, X=np.zeros((1, 2, 8), np.float32), y=np.array([0]),
        subject_id=np.array(["u00"]), timestamp=np.array([0]),
        missingness=np.array([1.0]),
    )
    assert _load_cache(path).version == "studentlife-v1-legacy"
