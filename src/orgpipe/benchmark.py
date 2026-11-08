"""Spike-sorting benchmark on synthetic recordings with ground truth (M2).

Every sorter sees the *same* preprocessed recording per seed and is scored with
the same code (``orgpipe.scoring``). Results are reported as mean +/- SD over
seeds, plus per-unit rows so accuracy can be plotted against nominal SNR.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from .preprocess import estimate_noise, preprocess
from .scoring import score_sorting, score_spike_times
from .simulate import SimConfig, simulate_recording
from .sorters import OWN_SORTERS, sort_own, sort_spikeinterface
from .spikes import detect_spikes

SUMMARY_COLUMNS = [
    ("n_well_detected", "units acc>=0.8 (of true)"),
    ("n_snr_well_detected", "units acc>=0.8 (of SNR>=6)"),
    ("mean_accuracy_snr", "mean acc (SNR>=6)"),
    ("any_recall", "spike recall"),
    ("any_precision", "spike precision"),
    ("n_sorted_units", "sorted units"),
    ("n_spurious_or_split", "spurious/split"),
    ("runtime_s", "runtime (s)"),
]


def run_benchmark(
    seeds=(0, 1, 2, 3, 4),
    base_cfg: SimConfig | None = None,
    own=tuple(OWN_SORTERS),
    spikeinterface: tuple[str, ...] = (),
    tol_ms: float = 1.0,
    progress=print,
) -> dict:
    """Return ``{"rows": [...summary rows...], "units": [...per-unit rows...], "config": {...}}``."""
    base_cfg = base_cfg or SimConfig(fs=20_000.0, duration=30.0, stim_interval_s=4.0)
    rows, unit_rows = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in seeds:
            raw = simulate_recording(replace(base_cfg, seed=seed))
            proc = preprocess(raw)
            mask = proc.artifact_mask
            sd = estimate_noise(proc.data, mask)
            progress(f"seed {seed}: {raw.truth.n_units} units, {len(raw.truth.spike_times)} spikes")

            det = detect_spikes(proc)
            d = score_spike_times(proc, det.times, tol_ms, mask)
            rows.append({"sorter": "detector only (no sorting)", "seed": seed,
                         "any_recall": d["any_recall"], "any_precision": d["any_precision"]})

            runs = [(n, lambda n=n: sort_own(proc, n)) for n in own]
            runs += [(f"spikeinterface:{n}", lambda n=n: sort_spikeinterface(proc, n)) for n in spikeinterface]
            for name, fn in runs:
                try:
                    sorting = fn()
                    res = score_sorting(proc, sorting, sd, tol_ms, mask)
                except Exception as e:  # keep the benchmark going; record the failure
                    progress(f"  {name}: FAILED ({type(e).__name__}: {e})")
                    rows.append({"sorter": name, "seed": seed, "error": f"{type(e).__name__}: {e}"})
                    continue
                rows.append({"sorter": name, "seed": seed, **res["summary"]})
                unit_rows += [{"sorter": name, "seed": seed, **u} for u in res["units"] if u["n_spikes"] > 0]
                s = res["summary"]
                progress(f"  {name}: well {s['n_well_detected']}/{s['n_true_units']}, "
                         f"acc(SNR>=6) {s['mean_accuracy_snr']:.2f}, recall {s['any_recall']:.2f}, "
                         f"precision {s['any_precision']:.2f}")
    return {"rows": rows, "units": unit_rows,
            "config": {k: v for k, v in vars(base_cfg).items()}, "seeds": list(seeds), "tol_ms": tol_ms}


def _fmt(vals: list[float], key: str) -> str:
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "-"
    if key.startswith("n_"):
        return f"{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f}"
    return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}"


def summary_table(result: dict) -> str:
    """Markdown table: mean ± SD over seeds, one row per sorter."""
    names = list(dict.fromkeys(r["sorter"] for r in result["rows"]))
    head = "| sorter | " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(SUMMARY_COLUMNS) + 1)]
    for n in names:
        rs = [r for r in result["rows"] if r["sorter"] == n and "error" not in r]
        if not rs:
            lines.append(f"| {n} | " + " | ".join("failed" for _ in SUMMARY_COLUMNS) + " |")
            continue
        cells = [_fmt([r.get(k) for r in rs], k) if any(k in r for r in rs) else "-" for k, _ in SUMMARY_COLUMNS]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
