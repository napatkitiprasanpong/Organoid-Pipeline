import numpy as np
import pytest

from orgpipe import SimConfig, preprocess, simulate_recording
from orgpipe.preprocess import estimate_noise
from orgpipe.recording import Recording
from orgpipe.scoring import score_sorting, score_spike_times
from orgpipe.sorters import OWN_SORTERS, sort_own, sort_spikeinterface
from orgpipe.spikes import detect_spikes, extract_waveforms, sort_spikes


def test_detector_dedups_one_spike_seen_on_two_neighbouring_channels():
    fs = 20_000.0
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (4, 4000)).astype(np.float32)
    xy = np.array([[0, 0], [200, 0], [400, 0], [600, 0]], dtype=float)
    data[0, 1000] -= 40  # large on channel 0 ...
    data[1, 1001] -= 25  # ... same spike, smaller, 1 sample later on the neighbour
    data[3, 1000] -= 40  # a different spike far away at the same time must survive
    rec = Recording(data=data, fs=fs, channel_xy=xy)
    det = detect_spikes(rec)
    assert len(det) == 2
    assert set(det.channels) == {0, 3}
    assert set(det.samples) == {1000}


def test_detector_ignores_artifact_mask():
    fs = 20_000.0
    data = np.zeros((2, 2000), dtype=np.float32) + np.random.default_rng(1).normal(0, 1, (2, 2000)).astype(np.float32)
    data[0, 500] -= 50
    mask = np.zeros(2000, dtype=bool)
    mask[480:520] = True
    rec = Recording(data=data, fs=fs, channel_xy=np.array([[0.0, 0.0], [200.0, 0.0]]), artifact_mask=mask)
    assert len(detect_spikes(rec)) == 0


def test_waveforms_are_aligned_on_the_trough():
    cfg = SimConfig(fs=20_000.0, duration=8.0, n_units=6, seed=3)
    rec = preprocess(simulate_recording(cfg))
    det = detect_spikes(rec)
    wf, ok = extract_waveforms(rec, det)
    assert wf.shape[0] == ok.sum() and wf.shape[1] == 4
    trough_pos = wf[:, 0, :].argmin(axis=1)
    assert np.median(trough_pos) == int(round(0.6e-3 * rec.fs))


def test_detection_precision_on_synthetic_data():
    cfg = SimConfig(fs=20_000.0, duration=15.0, n_units=8, seed=21, stim_interval_s=3.0)
    rec = preprocess(simulate_recording(cfg))
    det = detect_spikes(rec)
    score = score_spike_times(rec, det.times, exclude_mask=rec.artifact_mask)
    assert score["any_precision"] > 0.9  # few false events at a 5 SD threshold
    assert score["any_recall"] > 0.4  # weak units sit below threshold by design


def test_sorter_recovers_well_separated_units():
    # Units on different electrodes, high amplitude, >20 spikes each (the sorter's min_cluster_size).
    cfg = SimConfig(fs=20_000.0, duration=20.0, n_units=4, amp_range_uv=(120.0, 160.0), baseline_rate_hz=3.0, seed=7)
    rec = preprocess(simulate_recording(cfg))
    sd = estimate_noise(rec.data, rec.artifact_mask)
    res = score_sorting(rec, sort_spikes(rec), sd, exclude_mask=rec.artifact_mask)["summary"]
    assert res["any_precision"] > 0.95
    assert res["n_well_detected"] == 4


def test_all_registered_own_sorters_run_and_return_valid_labels():
    cfg = SimConfig(fs=20_000.0, duration=10.0, n_units=5, amp_range_uv=(100.0, 150.0), seed=5)
    rec = preprocess(simulate_recording(cfg))
    for name in OWN_SORTERS:
        s = sort_own(rec, name)
        assert len(s.spike_samples) == len(s.labels)
        assert s.runtime_s is not None and s.runtime_s >= 0
        if s.n_units:
            assert s.labels.max() < len(s.unit_channel)


def test_unknown_sorter_name_is_a_clear_error():
    rec = preprocess(simulate_recording(SimConfig(fs=20_000.0, duration=3.0, n_units=2)))
    with pytest.raises(KeyError):
        sort_own(rec, "not-a-sorter")


def test_spikeinterface_wrapper_reports_missing_install():
    try:
        import spikeinterface  # noqa: F401
        pytest.importorskip("never_installed_module_marker")  # installed: this test does not apply
    except ImportError:
        rec = preprocess(simulate_recording(SimConfig(fs=20_000.0, duration=3.0, n_units=2)))
        with pytest.raises(ImportError):
            sort_spikeinterface(rec, "simple")
