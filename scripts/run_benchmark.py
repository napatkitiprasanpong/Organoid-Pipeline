"""M2 benchmark: simulate -> preprocess -> sort (own methods, optional SpikeInterface) -> score.

    python scripts/run_benchmark.py                          # own sorters only, 5 seeds
    python scripts/run_benchmark.py --spikeinterface simple  # add a SpikeInterface sorter
    python scripts/run_benchmark.py --seeds 0 1 2 --duration 20

Writes docs/m2-results.md (table), docs/figures/m2_benchmark.png and docs/m2-results.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orgpipe import SimConfig
from orgpipe.benchmark import run_benchmark, summary_table
from orgpipe.sorters import OWN_SORTERS

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--duration", type=float, default=30.0)
p.add_argument("--fs", type=float, default=20_000.0)
p.add_argument("--own", nargs="*", default=list(OWN_SORTERS), help="own sorter names")
p.add_argument("--spikeinterface", nargs="*", default=[], help="SpikeInterface sorter names (must be installed)")
p.add_argument("--outdir", default="docs")
args = p.parse_args()

cfg = SimConfig(fs=args.fs, duration=args.duration, stim_interval_s=4.0)
res = run_benchmark(args.seeds, cfg, own=tuple(args.own), spikeinterface=tuple(args.spikeinterface))
table = summary_table(res)
print("\n" + table)

out = Path(args.outdir)
(out / "figures").mkdir(parents=True, exist_ok=True)
(out / "m2-results.md").write_text(
    f"Seeds {args.seeds}, {args.duration:.0f} s at {args.fs:.0f} Hz, {cfg.n_units} units, 32 channels, "
    f"stimulation every {cfg.stim_interval_s:.0f} s. Mean ± SD over seeds; spike tolerance {res['tol_ms']} ms.\n\n"
    + table + "\n")
(out / "m2-results.json").write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# figure: per-unit accuracy vs nominal SNR, one panel per sorter
sorters = [s for s in dict.fromkeys(u["sorter"] for u in res["units"])]
fig, axes = plt.subplots(1, len(sorters), figsize=(3.4 * len(sorters), 3.4), sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes)
for ax, name in zip(axes, sorters):
    us = [u for u in res["units"] if u["sorter"] == name]
    snr = np.array([u["nominal_snr"] for u in us])
    acc = np.array([u["accuracy"] for u in us])
    ax.scatter(snr, acc, s=10, alpha=0.5)
    ax.axvline(6, color="0.6", ls=":", lw=1)
    ax.axhline(0.8, color="0.6", ls=":", lw=1)
    ax.set(title=name, xlabel="nominal unit SNR", xlim=(0, 25), ylim=(-0.05, 1.05))
axes[0].set_ylabel("unit accuracy vs ground truth")
fig.suptitle(f"M2: unit accuracy vs SNR, pooled over seeds {args.seeds}", fontsize=10)
fig.savefig(out / "figures" / "m2_benchmark.png", dpi=130)
print("saved", out / "m2-results.md", out / "figures" / "m2_benchmark.png")
