import numpy as np
import pytest

from orgpipe import SimConfig, load_recording, preprocess, save_recording, simulate_recording


def _check_roundtrip(path):
    cfg = SimConfig(fs=20_000.0, duration=5.0, n_units=6, seed=5, stim_interval_s=1.5)
    rec = preprocess(simulate_recording(cfg))
    save_recording(rec, path)
    back = load_recording(path)
    assert back.fs == rec.fs
    assert np.allclose(back.data, rec.data)
    assert np.array_equal(back.artifact_mask, rec.artifact_mask)
    assert np.array_equal(back.stim_times, rec.stim_times)
    assert np.array_equal(back.truth.spike_times, rec.truth.spike_times)
    assert back.meta["config"]["seed"] == 5
    assert back.meta["preprocess"]["reference"] == "median"


def test_npz_roundtrip(tmp_path):
    _check_roundtrip(tmp_path / "rec.npz")


def test_hdf5_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    _check_roundtrip(tmp_path / "rec.h5")


def test_real_style_recording_without_truth(tmp_path):
    rec = simulate_recording(SimConfig(fs=20_000.0, duration=3.0, n_units=4))
    rec.truth = None
    save_recording(rec, tmp_path / "real_like.npz")
    assert load_recording(tmp_path / "real_like.npz").truth is None


def test_unsupported_suffix(tmp_path):
    rec = simulate_recording(SimConfig(fs=20_000.0, duration=2.0, n_units=2))
    with pytest.raises(ValueError):
        save_recording(rec, tmp_path / "rec.csv")
