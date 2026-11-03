"""Preprocessing for MEA recordings.

Order matters:

1. ``blank_artifacts``: remove stimulation transients from the raw trace first,
   so the filter does not smear them into the following milliseconds.
2. ``common_reference``: subtract the across-channel median at each sample to
   remove noise shared by all electrodes (hum, shared LFP, reference drift).
   Done after blanking because an artifact seen on every channel would
   otherwise leak into the median.
3. ``bandpass``: keep the spike band (default 300 to 6000 Hz).

Blanked samples are recorded in ``Recording.artifact_mask`` so later stages
(and scoring) can exclude them instead of pretending those spikes were seen.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import signal

from .recording import Recording


def bandpass(
    data: np.ndarray, fs: float, low_hz: float = 300.0, high_hz: float = 6000.0, order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth band-pass, applied channel by channel."""
    high_hz = min(high_hz, 0.45 * fs)  # stay safely below Nyquist
    sos = signal.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    out = np.empty(data.shape, dtype=np.float32)
    for ch in range(data.shape[0]):
        out[ch] = signal.sosfiltfilt(sos, data[ch].astype(np.float64))
    return out


def common_reference(data: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract the per-sample median (or mean) across channels."""
    if method == "median":
        ref = np.median(data, axis=0)
    elif method == "mean":
        ref = data.mean(axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return (data - ref[None, :]).astype(np.float32)


def blank_artifacts(
    data: np.ndarray,
    fs: float,
    stim_times: np.ndarray,
    pre_ms: float = 0.5,
    post_ms: float = 8.0,
    method: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray]:
    """Remove stimulation artifacts around each stimulus time.

    ``method='interpolate'`` draws a straight line across the window (best
    before filtering, no step edges); ``'zero'`` sets it to zero.
    Returns (cleaned data, boolean mask of blanked samples).
    """
    out = np.array(data, dtype=np.float32, copy=True)
    n = data.shape[1]
    mask = np.zeros(n, dtype=bool)
    pre, post = int(round(pre_ms * 1e-3 * fs)), int(round(post_ms * 1e-3 * fs))
    for st in np.asarray(stim_times, dtype=float):
        i = int(round(st * fs))
        a, b = max(i - pre, 0), min(i + post, n)
        if b - a < 2:
            continue
        mask[a:b] = True
        if method == "zero":
            out[:, a:b] = 0.0
        elif method == "interpolate":
            lo, hi = out[:, max(a - 1, 0)], out[:, min(b, n - 1)]
            w = np.linspace(0.0, 1.0, b - a)[None, :]
            out[:, a:b] = lo[:, None] * (1 - w) + hi[:, None] * w
        else:
            raise ValueError("method must be 'interpolate' or 'zero'")
    return out, mask


def estimate_noise(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Robust per-channel noise SD (median absolute deviation / 0.6745)."""
    x = data if mask is None else data[:, ~mask]
    med = np.median(x, axis=1, keepdims=True)
    return np.median(np.abs(x - med), axis=1) / 0.6745


def preprocess(
    rec: Recording,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    reference: str | None = "median",
    blank: bool = True,
    pre_ms: float = 0.5,
    post_ms: float = 8.0,
) -> Recording:
    """Return a new Recording with artifacts blanked, referenced and band-passed."""
    data, mask = rec.data, rec.artifact_mask
    if blank and len(rec.stim_times):
        data, new_mask = blank_artifacts(data, rec.fs, rec.stim_times, pre_ms, post_ms)
        mask = new_mask if mask is None else (mask | new_mask)
    if reference:
        data = common_reference(data, reference)
    data = bandpass(data, rec.fs, low_hz, high_hz)
    params = dict(low_hz=low_hz, high_hz=high_hz, reference=reference, blank=blank,
                  pre_ms=pre_ms, post_ms=post_ms)
    return dataclasses.replace(
        rec, data=data, artifact_mask=mask, meta={**rec.meta, "preprocess": params}
    )


def spike_snr(
    rec: Recording,
    exclude_mask: np.ndarray | None = None,
    peak_window_ms: float = 0.3,
    baseline_window_ms: float = 2.0,
) -> float:
    """Median ground-truth spike SNR: (local baseline - trough) / channel noise SD.

    Uses the simulator's known spike times and primary channels. The same
    definition is applied to raw and processed data, so the ratio of the two is
    meaningful even though the absolute value is a simple estimate.
    Spikes inside ``exclude_mask`` are skipped so raw and processed runs can be
    compared on identical spikes.
    """
    if rec.truth is None:
        raise ValueError("spike_snr needs ground truth (synthetic recordings only)")
    tr = rec.truth
    s = np.round(tr.spike_times * rec.fs).astype(int)
    ch = tr.unit_channel[tr.spike_units]
    pw = int(round(peak_window_ms * 1e-3 * rec.fs))
    bw = int(round(baseline_window_ms * 1e-3 * rec.fs))
    ok = (s - bw >= 0) & (s + bw < rec.n_samples)
    if exclude_mask is not None:
        ok &= ~exclude_mask[np.clip(s, 0, rec.n_samples - 1)]
    s, ch = s[ok], ch[ok]
    if len(s) == 0:
        return float("nan")
    noise = estimate_noise(rec.data, exclude_mask)
    peak_idx = s[:, None] + np.arange(-pw, pw + 1)[None, :]
    base_idx = s[:, None] + np.arange(-bw, bw + 1)[None, :]
    trough = rec.data[ch[:, None], peak_idx].min(axis=1)
    baseline = np.median(rec.data[ch[:, None], base_idx], axis=1)
    return float(np.median((baseline - trough) / noise[ch]))


def snr_change(raw: Recording, processed: Recording) -> dict[str, float]:
    """Compare ground-truth spike SNR before and after preprocessing."""
    mask = processed.artifact_mask
    before, after = spike_snr(raw, mask), spike_snr(processed, mask)
    return {"snr_before": before, "snr_after": after, "gain": after / before}
