import numpy as np

from orgpipe import SimConfig, simulate_recording


def small(**kw):
    base = dict(fs=20_000.0, duration=12.0, n_units=10, seed=1)
    base.update(kw)
    return SimConfig(**base)


def test_shapes_and_truth_consistency():
    rec = simulate_recording(small())
    assert rec.data.shape == (32, int(12 * 20_000))
    assert rec.channel_xy.shape == (32, 2)
    tr = rec.truth
    assert len(tr.spike_times) == len(tr.spike_units) > 0
    assert np.all(np.diff(tr.spike_times) >= 0)
    assert tr.spike_times.min() >= 0 and tr.spike_times.max() < rec.duration
    assert tr.unit_channel.shape == (10,)


def test_deterministic_for_seed():
    a = simulate_recording(small(duration=4.0))
    b = simulate_recording(small(duration=4.0))
    assert np.array_equal(a.data, b.data)
    c = simulate_recording(small(duration=4.0, seed=2))
    assert not np.array_equal(a.data, c.data)


def test_refractory_period_respected():
    rec = simulate_recording(small())
    tr = rec.truth
    for u in range(tr.n_units):
        t = tr.spike_times[tr.spike_units == u]
        if len(t) > 1:
            assert np.diff(t).min() >= 0.002 - 1e-9


def test_bursts_raise_firing_rate():
    rec = simulate_recording(small())
    tr = rec.truth
    assert len(tr.burst_windows) >= 1
    in_burst = np.zeros(len(tr.spike_times), dtype=bool)
    burst_time = 0.0
    for t0, t1 in tr.burst_windows:
        in_burst |= (tr.spike_times >= t0) & (tr.spike_times < t1)
        burst_time += t1 - t0
    rate_in = in_burst.sum() / burst_time
    rate_out = (~in_burst).sum() / (rec.duration - burst_time)
    assert rate_in > 5 * rate_out


def test_stim_artifacts_added_only_when_requested():
    off = simulate_recording(small(duration=5.0))
    on = simulate_recording(small(duration=5.0, stim_interval_s=1.0))
    assert len(off.stim_times) == 0
    assert len(on.stim_times) >= 3
    assert np.abs(on.data).max() > 5 * np.abs(off.data).max()
