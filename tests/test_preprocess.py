import numpy as np

from orgpipe import SimConfig, simulate_recording, preprocess, snr_change
from orgpipe.preprocess import bandpass, blank_artifacts, common_reference


def power_at(x, fs, f0, half_width=2.0):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return spec[(freqs > f0 - half_width) & (freqs < f0 + half_width)].sum()


def test_bandpass_removes_hum_and_keeps_spike_band():
    fs = 20_000.0
    t = np.arange(int(4 * fs)) / fs
    x = np.vstack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 1000 * t)])
    y = bandpass(x, fs)
    assert power_at(y[0], fs, 50) < 1e-3 * power_at(x[0], fs, 50)
    assert power_at(y[1], fs, 1000) > 0.9 * power_at(x[1], fs, 1000)


def test_common_reference_removes_shared_signal():
    rng = np.random.default_rng(0)
    shared = 100 * np.sin(np.linspace(0, 50, 5000))
    data = rng.normal(0, 1, (16, 5000)) + shared
    out = common_reference(data)
    assert out.std() < 0.05 * data.std() + 1.5  # shared 100 uV swing gone, ~1 uV noise left
    assert np.abs(np.median(out, axis=0)).max() < 1e-4


def test_blank_artifacts_removes_transient():
    fs = 20_000.0
    data = np.zeros((4, 4000), dtype=np.float32)
    data[:, 2000:2040] = 3000.0
    out, mask = blank_artifacts(data, fs, np.array([2000 / fs]))
    assert np.abs(out).max() < 1.0
    assert mask[2000] and not mask[100]


def test_preprocess_improves_snr_with_ground_truth():
    cfg = SimConfig(fs=20_000.0, duration=10.0, n_units=10, seed=3, stim_interval_s=2.0)
    raw = simulate_recording(cfg)
    proc = preprocess(raw)
    res = snr_change(raw, proc)
    assert res["gain"] > 1.5
    assert proc.meta["preprocess"]["low_hz"] == 300.0
    assert proc.artifact_mask is not None and proc.artifact_mask.any()
    # input untouched
    assert raw.artifact_mask is None


def test_preprocessed_trace_has_no_stim_ringing():
    cfg = SimConfig(fs=20_000.0, duration=6.0, n_units=6, seed=4, stim_interval_s=2.0)
    raw = simulate_recording(cfg)
    proc = preprocess(raw)
    assert np.abs(proc.data).max() < 0.1 * np.abs(raw.data).max()
