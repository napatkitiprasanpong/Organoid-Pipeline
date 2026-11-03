"""Synthetic MEA generator with known ground truth.

What it produces (all in microvolts):

* Several simulated neurons placed near electrodes of a regular grid, each
  with its own biphasic spike waveform. Amplitude falls off exponentially with
  distance, so a unit can show up on neighbouring channels too.
* Spontaneous activity plus network bursts (all units fire together for a few
  hundred milliseconds), as seen in organoid and culture recordings.
* Noise: white electrode noise, slow LFP-like fluctuations (independent and
  shared across channels) and 50 Hz mains hum shared across channels.
* Optional stimulation artifacts: large, fast-decaying transients that are
  strongest near the stimulating electrode.

This is a deliberately simple generative model. It is good enough to test
whether a pipeline stage does what it claims; it is not a model of real
organoid biophysics, and real artifacts and drift are messier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from .recording import GroundTruth, Recording


@dataclass
class SimConfig:
    fs: float = 30_000.0
    duration: float = 30.0  # seconds
    n_rows: int = 4
    n_cols: int = 8  # 4 x 8 = 32 channels
    pitch_um: float = 200.0
    n_units: int = 24
    seed: int = 0

    # unit properties
    amp_range_uv: tuple[float, float] = (50.0, 150.0)
    spatial_decay_um: float = 50.0
    unit_jitter_um: float = 25.0  # spread of a unit's position around its electrode
    refractory_s: float = 0.002

    # activity
    baseline_rate_hz: float = 0.5
    in_burst_rate_hz: float = 40.0
    burst_rate_hz: float = 0.3  # mean network bursts per second
    burst_duration_s: tuple[float, float] = (0.3, 0.8)
    join_prob_range: tuple[float, float] = (0.5, 1.0)  # per-unit chance of joining a burst
    onset_latency_s: float = 0.05  # max per-unit delay after burst onset

    # noise
    white_noise_uv: float = 8.0
    lfp_rms_uv: float = 25.0  # independent per channel, below ~200 Hz
    shared_lfp_rms_uv: float = 15.0  # common to all channels
    hum_hz: float = 50.0
    hum_uv: float = 20.0  # common to all channels

    # stimulation (off by default)
    stim_interval_s: float | None = None
    stim_first_s: float = 1.0
    stim_electrode: int = 0
    stim_amp_uv: float = 3000.0
    stim_spatial_decay_um: float = 300.0


def _grid(cfg: SimConfig) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.arange(cfg.n_cols) * cfg.pitch_um, np.arange(cfg.n_rows) * cfg.pitch_um
    )
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def _spike_template(fs: float, width_scale: float) -> tuple[np.ndarray, int]:
    """Biphasic waveform (trough then smaller peak), trough normalised to -1."""
    n = int(round(1.5e-3 * fs))
    trough = int(round(0.5e-3 * fs))
    t = (np.arange(n) - trough) / fs
    s1, s2 = 0.10e-3 * width_scale, 0.25e-3 * width_scale
    w = -np.exp(-(t**2) / (2 * s1**2)) + 0.35 * np.exp(-((t - 0.3e-3 * width_scale) ** 2) / (2 * s2**2))
    return w / -w.min(), trough


def _lowpassed_noise(rng, n: int, fs: float, cutoff_hz: float, rms: float) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n))
    return x * (rms / x.std())


def _make_bursts(rng, cfg: SimConfig) -> np.ndarray:
    bursts, t = [], 1.0
    while t < cfg.duration - 1.0:
        dur = rng.uniform(*cfg.burst_duration_s)
        if t + dur > cfg.duration - 0.5:
            break
        bursts.append((t, t + dur))
        t += dur + 1.0 + rng.exponential(1.0 / cfg.burst_rate_hz)
    return np.array(bursts, dtype=float).reshape(-1, 2)


def _unit_spikes(rng, cfg: SimConfig, bursts: np.ndarray, gain: float, p_join: float) -> np.ndarray:
    """Spike times for one unit from a 1 ms rate model, refractory enforced."""
    dt = 1e-3
    n_bins = int(cfg.duration / dt)
    rate = np.full(n_bins, cfg.baseline_rate_hz)
    for t0, t1 in bursts:
        if rng.random() > p_join:
            continue
        start = t0 + rng.uniform(0, cfg.onset_latency_s)
        i0, i1 = int(start / dt), int(t1 / dt)
        if i1 - i0 < 2:
            continue
        rate[i0:i1] += gain * cfg.in_burst_rate_hz * np.hanning(i1 - i0)
    bins = np.flatnonzero(rng.random(n_bins) < rate * dt)
    times = (bins + rng.random(len(bins))) * dt
    if len(times) > 1:
        keep = np.concatenate([[True], np.diff(times) >= cfg.refractory_s])
        times = times[keep]
    return times


def simulate_recording(cfg: SimConfig | None = None) -> Recording:
    """Generate a synthetic recording. Deterministic for a given ``cfg.seed``."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    xy = _grid(cfg)
    n_ch = len(xy)
    n_samples = int(round(cfg.duration * cfg.fs))
    fs = cfg.fs

    bursts = _make_bursts(rng, cfg)

    # --- units -------------------------------------------------------------
    unit_electrode = rng.integers(0, n_ch, cfg.n_units)
    unit_xy = xy[unit_electrode] + rng.normal(0, cfg.unit_jitter_um, (cfg.n_units, 2))
    unit_amp = rng.uniform(*cfg.amp_range_uv, cfg.n_units)
    dist = np.linalg.norm(unit_xy[:, None, :] - xy[None, :, :], axis=2)  # (units, ch)
    ch_gain = np.exp(-dist / cfg.spatial_decay_um)
    unit_channel = np.argmax(ch_gain, axis=1)

    data = np.zeros((n_ch, n_samples), dtype=np.float32)
    all_times, all_units = [], []
    for u in range(cfg.n_units):
        gain = rng.uniform(0.3, 1.0)
        p_join = rng.uniform(*cfg.join_prob_range)
        times = _unit_spikes(rng, cfg, bursts, gain, p_join)
        template, trough = _spike_template(fs, rng.uniform(0.8, 1.3))
        samples = np.round(times * fs).astype(int)
        start = samples - trough
        ok = (start >= 0) & (start + len(template) <= n_samples)
        samples, start = samples[ok], start[ok]
        for ch in np.flatnonzero(ch_gain[u] > 0.1):
            amp = unit_amp[u] * ch_gain[u, ch]
            # spikes of one unit are >= refractory apart (> template length): no overlaps
            for k, w in enumerate(template):
                data[ch, start + k] += np.float32(amp * w)
        all_times.append(samples / fs)
        all_units.append(np.full(len(samples), u))
    spike_times = np.concatenate(all_times)
    spike_units = np.concatenate(all_units)
    order = np.argsort(spike_times)
    spike_times, spike_units = spike_times[order], spike_units[order]

    # --- noise -------------------------------------------------------------
    t = np.arange(n_samples) / fs
    shared = _lowpassed_noise(rng, n_samples, fs, 200.0, cfg.shared_lfp_rms_uv)
    shared += cfg.hum_uv * np.sin(2 * np.pi * cfg.hum_hz * t + rng.uniform(0, 2 * np.pi))
    for ch in range(n_ch):
        data[ch] += (
            rng.normal(0, cfg.white_noise_uv, n_samples)
            + _lowpassed_noise(rng, n_samples, fs, 200.0, cfg.lfp_rms_uv)
            + shared
        ).astype(np.float32)

    # --- stimulation artifacts ----------------------------------------------
    stim_times = np.zeros(0)
    if cfg.stim_interval_s:
        stim_times = np.arange(cfg.stim_first_s, cfg.duration - 0.1, cfg.stim_interval_s)
        te = np.arange(int(0.015 * fs)) / fs  # 15 ms
        shape = np.exp(-te / 0.5e-3) + 0.1 * np.exp(-te / 3e-3)
        d_stim = np.linalg.norm(xy - xy[cfg.stim_electrode], axis=1)
        ch_art = cfg.stim_amp_uv * np.exp(-d_stim / cfg.stim_spatial_decay_um)
        for st in stim_times:
            i0 = int(round(st * fs))
            n = min(len(te), n_samples - i0)
            data[:, i0 : i0 + n] += (ch_art[:, None] * shape[None, :n]).astype(np.float32)

    truth = GroundTruth(
        spike_times=spike_times,
        spike_units=spike_units,
        unit_channel=unit_channel,
        unit_xy=unit_xy,
        unit_amp_uv=unit_amp,
        burst_windows=bursts,
    )
    return Recording(
        data=data,
        fs=fs,
        channel_xy=xy,
        stim_times=stim_times,
        truth=truth,
        meta={"source": "synthetic", "config": asdict(cfg)},
    )
