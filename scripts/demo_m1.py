"""M1 demo: simulate -> preprocess -> before/after figure and SNR numbers.

    python scripts/demo_m1.py [--out docs/figures/m1_before_after.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from orgpipe import SimConfig, preprocess, simulate_recording, snr_change

p = argparse.ArgumentParser()
p.add_argument("--out", default="docs/figures/m1_before_after.png")
p.add_argument("--duration", type=float, default=20.0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

cfg = SimConfig(duration=args.duration, seed=args.seed, stim_interval_s=4.0)
raw = simulate_recording(cfg)
proc = preprocess(raw)
res = snr_change(raw, proc)
print(f"units={raw.truth.n_units}  spikes={len(raw.truth.spike_times)}  "
      f"bursts={len(raw.truth.burst_windows)}  stim={len(raw.stim_times)}")
print(f"ground-truth spike SNR: {res['snr_before']:.2f} -> {res['snr_after']:.2f}  (x{res['gain']:.1f})")
print(f"samples blanked around stimulation: {proc.artifact_mask.mean():.2%}")

# pick the channel with the most ground-truth spikes, zoom on the first stimulus
tr = raw.truth
ch = int(np.bincount(tr.unit_channel[tr.spike_units], minlength=raw.n_channels).argmax())
t0 = raw.stim_times[0] - 0.05
sl = slice(int(t0 * raw.fs), int((t0 + 0.4) * raw.fs))
t = np.arange(raw.n_samples)[sl] / raw.fs
spk = tr.spike_times[(tr.unit_channel[tr.spike_units] == ch) & (tr.spike_times >= t[0]) & (tr.spike_times < t[-1])]

fig, ax = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
ax[0].plot(t, raw.data[ch, sl], lw=0.5, color="0.25")
ax[0].set(title=f"Raw, channel {ch} (stimulus at {raw.stim_times[0]:.2f} s)", ylabel="uV")
ax[1].plot(t, proc.data[ch, sl], lw=0.5, color="tab:blue")
ax[1].vlines(spk, proc.data[ch, sl].max() * 1.05, proc.data[ch, sl].max() * 1.25, color="tab:red", lw=1)
ax[1].set(title="After blanking, median reference, 300-6000 Hz band-pass (red ticks = ground-truth spikes)",
          xlabel="time (s)", ylabel="uV")
for r, label, c in [(raw, "raw", "0.25"), (proc, "preprocessed", "tab:blue")]:
    f, pxx = signal.welch(r.data[ch], r.fs, nperseg=8192)
    ax[2].semilogy(f, pxx, label=label, color=c, lw=0.8)
ax[2].set(xlim=(0, 2000), ylim=(1e-3, 1e3), title="Power spectrum (50 Hz hum and LFP removed)", xlabel="Hz", ylabel="uV^2/Hz")
ax[2].legend()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print("saved", out)
