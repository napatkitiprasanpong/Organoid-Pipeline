"""Score detections and sortings against simulator ground truth.

Definitions (tolerance ``tol_ms`` on spike time, one-to-one matching):

* per unit pair (true i, sorted k): ``accuracy = tp / (n_true + n_sorted - tp)``
  (the usual SpikeInterface/SpikeForest agreement score)
* units are matched one-to-one with the Hungarian algorithm on that score
* spike level: ``any_recall`` / ``any_precision`` ignore unit identity and ask
  only whether a spike was found, so detection and sorting errors separate.

Spikes inside ``exclude_mask`` (blanked stimulation artifacts) are removed from
both truth and estimate, because nobody can be scored on samples that were
deliberately thrown away.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .preprocess import estimate_noise
from .recording import Recording


def count_matches(a: np.ndarray, b: np.ndarray, tol_s: float) -> int:
    """One-to-one count of spikes in sorted ``a`` that have a spike in sorted ``b`` within ``tol_s``."""
    if len(a) == 0 or len(b) == 0:
        return 0
    j = np.searchsorted(b, a)
    jl, jr = np.clip(j - 1, 0, len(b) - 1), np.clip(j, 0, len(b) - 1)
    dl, dr = np.abs(a - b[jl]), np.abs(a - b[jr])
    use_r = dr < dl
    jj, d = np.where(use_r, jr, jl), np.where(use_r, dr, dl)
    ok = d <= tol_s
    jj, d = jj[ok], d[ok]
    if len(jj) == 0:
        return 0
    order = np.lexsort((d, jj))
    jj = jj[order]
    return int(np.sum(np.r_[True, jj[1:] != jj[:-1]]))


def nominal_unit_snr(rec: Recording, noise_sd: np.ndarray) -> np.ndarray:
    """Nominal peak SNR of each true unit: simulated amplitude on its primary channel / channel noise SD.

    ``noise_sd`` comes from the *processed* data, so this is the SNR a detector actually faces.
    """
    tr = rec.truth
    decay = rec.meta.get("config", {}).get("spatial_decay_um", 50.0)
    d = np.linalg.norm(tr.unit_xy - rec.channel_xy[tr.unit_channel], axis=1)
    return tr.unit_amp_uv * np.exp(-d / decay) / noise_sd[tr.unit_channel]


def _truth_by_unit(rec: Recording, exclude_mask: np.ndarray | None) -> list[np.ndarray]:
    tr = rec.truth
    t, u = tr.spike_times, tr.spike_units
    if exclude_mask is not None:
        keep = ~exclude_mask[np.clip(np.round(t * rec.fs).astype(int), 0, rec.n_samples - 1)]
        t, u = t[keep], u[keep]
    return [np.sort(t[u == i]) for i in range(tr.n_units)]


def score_spike_times(
    rec: Recording, times: np.ndarray, tol_ms: float = 1.0, exclude_mask: np.ndarray | None = None
) -> dict[str, float]:
    """Unit-agnostic detection score: did we find the spikes, and are the ones we report real?"""
    tol = tol_ms * 1e-3
    truth = np.sort(np.concatenate(_truth_by_unit(rec, exclude_mask)))
    est = np.sort(np.asarray(times, dtype=float))
    if exclude_mask is not None and len(est):
        est = est[~exclude_mask[np.clip(np.round(est * rec.fs).astype(int), 0, rec.n_samples - 1)]]
    tp_t = count_matches(truth, est, tol)
    tp_e = count_matches(est, truth, tol)
    rec_ = tp_t / max(len(truth), 1)
    prec = tp_e / max(len(est), 1)
    return {"any_recall": rec_, "any_precision": prec,
            "any_f1": 2 * rec_ * prec / max(rec_ + prec, 1e-12), "n_true": len(truth), "n_est": len(est)}


def score_sorting(
    rec: Recording,
    sorting,
    noise_sd: np.ndarray,
    tol_ms: float = 1.0,
    exclude_mask: np.ndarray | None = None,
    well_detected: float = 0.8,
    snr_min: float = 6.0,
) -> dict:
    """Score a ``SortingResult`` (anything with ``unit_ids`` / ``unit_spike_times`` / ``labels`` / ``spike_times``).

    Returns ``{"summary": {...}, "units": [per true unit dicts]}``.
    """
    tol = tol_ms * 1e-3
    truth = _truth_by_unit(rec, exclude_mask)
    est_ids = [int(k) for k in sorting.unit_ids]
    est = []
    for k in est_ids:
        t = sorting.unit_spike_times(k)
        if exclude_mask is not None and len(t):
            t = t[~exclude_mask[np.clip(np.round(t * rec.fs).astype(int), 0, rec.n_samples - 1)]]
        est.append(t)
    n_t, n_e = len(truth), len(est)

    tp = np.zeros((n_t, n_e), dtype=int)
    for i in range(n_t):
        for k in range(n_e):
            tp[i, k] = count_matches(truth[i], est[k], tol)
    n_true = np.array([len(t) for t in truth])
    n_est = np.array([len(t) for t in est])
    acc = tp / np.maximum(n_true[:, None] + n_est[None, :] - tp, 1) if n_e else np.zeros((n_t, 0))

    match = {}
    if n_e:
        rows, cols = linear_sum_assignment(-acc)
        match = {int(r): int(c) for r, c in zip(rows, cols) if acc[r, c] > 0}

    snr = nominal_unit_snr(rec, noise_sd)
    units = []
    for i in range(n_t):
        k = match.get(i)
        if k is None:
            p = r = a = 0.0
        else:
            p = tp[i, k] / max(n_est[k], 1)
            r = tp[i, k] / max(n_true[i], 1)
            a = float(acc[i, k])
        units.append({"unit": i, "n_spikes": int(n_true[i]), "nominal_snr": float(snr[i]),
                      "sorted_unit": est_ids[k] if k is not None else None,
                      "precision": float(p), "recall": float(r), "accuracy": a})

    accs = np.array([u["accuracy"] for u in units])
    has_spikes = n_true > 0
    good_snr = (snr >= snr_min) & has_spikes
    matched_ok = {k for i, k in match.items() if acc[i, k] >= 0.5}
    anyscore = score_spike_times(
        rec, sorting.spike_times[sorting.labels >= 0], tol_ms, exclude_mask)
    summary = {
        "n_true_units": int(has_spikes.sum()),
        "n_sorted_units": n_e,
        "n_well_detected": int((accs[has_spikes] >= well_detected).sum()),
        "n_snr_units": int(good_snr.sum()),
        "n_snr_well_detected": int((accs[good_snr] >= well_detected).sum()),
        "mean_accuracy": float(accs[has_spikes].mean()) if has_spikes.any() else float("nan"),
        "mean_accuracy_snr": float(accs[good_snr].mean()) if good_snr.any() else float("nan"),
        "n_spurious_or_split": n_e - len(matched_ok),
        "any_recall": anyscore["any_recall"],
        "any_precision": anyscore["any_precision"],
        "any_f1": anyscore["any_f1"],
        "runtime_s": getattr(sorting, "runtime_s", None),
    }
    return {"summary": summary, "units": units}
