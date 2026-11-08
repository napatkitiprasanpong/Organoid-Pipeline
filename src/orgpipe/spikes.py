"""Spike detection and sorting (milestone M2).

Pipeline, applied to a *preprocessed* recording:

1. ``detect_spikes``: negative threshold crossing at ``threshold_sd`` robust
   noise SDs on every channel, then de-duplication so one physical spike seen
   on several neighbouring electrodes becomes one event on its largest channel.
2. ``extract_waveforms``: a short snippet on the peak channel plus its nearest
   neighbours (spatial context separates units that share an electrode).
3. ``compute_features``: PCA or ICA on the concatenated snippet.
4. ``cluster_features``: HDBSCAN, OPTICS or a BIC-selected Gaussian mixture,
   run separately for the events of each peak channel.

Events inside ``Recording.artifact_mask`` are never reported. Label ``-1``
means "detected but not assigned to any cluster".

Limits (see docs/m2-benchmark.md): no spike-overlap resolution, no merging of
a unit split across neighbouring peak channels, no drift handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .preprocess import estimate_noise
from .recording import Recording


# --------------------------------------------------------------------------- detection
@dataclass
class DetectionResult:
    samples: np.ndarray  # int, sample index of the negative peak
    channels: np.ndarray  # int, channel with the largest peak for each event
    amps: np.ndarray  # peak depth in uV (positive number)
    noise_sd: np.ndarray  # per-channel robust noise SD used for thresholding
    fs: float

    @property
    def times(self) -> np.ndarray:
        return self.samples / self.fs

    def __len__(self) -> int:
        return len(self.samples)


def detect_spikes(
    rec: Recording,
    threshold_sd: float = 5.0,
    dedup_ms: float = 0.5,
    radius_um: float = 300.0,
) -> DetectionResult:
    """Threshold detection with cross-channel de-duplication.

    Events are visited from deepest to shallowest; each accepted event
    suppresses every other event within ``dedup_ms`` on channels closer than
    ``radius_um`` (default covers diagonal neighbours of a 200 um grid).
    """
    x, fs, mask = rec.data, rec.fs, rec.artifact_mask
    sd = estimate_noise(x, mask)
    min_dist = max(1, int(round(0.3e-3 * fs)))
    s_all, c_all, a_all = [], [], []
    for ch in range(rec.n_channels):
        pk, props = find_peaks(-x[ch].astype(np.float64), height=threshold_sd * sd[ch], distance=min_dist)
        if mask is not None:
            keep = ~mask[pk]
            pk, heights = pk[keep], props["peak_heights"][keep]
        else:
            heights = props["peak_heights"]
        s_all.append(pk)
        c_all.append(np.full(len(pk), ch))
        a_all.append(heights)
    s = np.concatenate(s_all)
    c = np.concatenate(c_all)
    a = np.concatenate(a_all)
    if len(s) == 0:
        return DetectionResult(s.astype(int), c.astype(int), a, sd, fs)

    order = np.argsort(s, kind="stable")
    s, c, a = s[order], c[order], a[order]
    w = max(1, int(round(dedup_ms * 1e-3 * fs)))
    d_ch = np.linalg.norm(rec.channel_xy[:, None, :] - rec.channel_xy[None, :, :], axis=2)
    near = d_ch <= radius_um
    suppressed = np.zeros(len(s), dtype=bool)
    keep = np.zeros(len(s), dtype=bool)
    for i in np.argsort(-a, kind="stable"):
        if suppressed[i]:
            continue
        keep[i] = True
        lo, hi = np.searchsorted(s, [s[i] - w, s[i] + w + 1])
        sel = np.arange(lo, hi)
        suppressed[sel[near[c[i], c[sel]]]] = True
    return DetectionResult(s[keep].astype(int), c[keep].astype(int), a[keep], sd, fs)


# --------------------------------------------------------------------------- waveforms
def neighbour_channels(channel_xy: np.ndarray, n_neighbors: int) -> np.ndarray:
    """(n_ch, n_neighbors + 1) channel indices, nearest first (itself first)."""
    d = np.linalg.norm(channel_xy[:, None, :] - channel_xy[None, :, :], axis=2)
    return np.argsort(d, axis=1, kind="stable")[:, : n_neighbors + 1]


def extract_waveforms(
    rec: Recording,
    det: DetectionResult,
    n_neighbors: int = 3,
    pre_ms: float = 0.6,
    post_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (waveforms (n_ok, n_neighbors + 1, n_t), boolean ``ok`` over events).

    Events too close to the recording edges are dropped (``ok`` is False).
    """
    pre, post = int(round(pre_ms * 1e-3 * rec.fs)), int(round(post_ms * 1e-3 * rec.fs))
    ok = (det.samples - pre >= 0) & (det.samples + post < rec.n_samples)
    nbr = neighbour_channels(rec.channel_xy, n_neighbors)
    s, c = det.samples[ok], det.channels[ok]
    idx = s[:, None] + np.arange(-pre, post + 1)[None, :]  # (n, n_t)
    chs = nbr[c]  # (n, n_nbr + 1)
    wf = rec.data[chs[:, :, None], idx[:, None, :]]
    return wf.astype(np.float32), ok


# --------------------------------------------------------------------------- features / clustering
def compute_features(wf: np.ndarray, n_components: int = 5, method: str = "pca", seed: int = 0) -> np.ndarray:
    """PCA or ICA of the flattened snippets. Fewer events than components shrinks the output."""
    from sklearn.decomposition import PCA, FastICA

    x = wf.reshape(len(wf), -1).astype(np.float64)
    k = int(min(n_components, x.shape[0] - 1, x.shape[1]))
    if k < 1:
        return np.zeros((len(x), 1))
    if method == "pca":
        return PCA(n_components=k, random_state=seed).fit_transform(x)
    if method == "ica":
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # fine for clustering features
            return FastICA(n_components=k, whiten="unit-variance", random_state=seed,
                           max_iter=1000, tol=1e-3).fit_transform(x)
    raise ValueError("features must be 'pca' or 'ica'")


def cluster_features(
    feats: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 20,
    gmm_max_k: int = 6,
    seed: int = 0,
    min_samples: int | None = None,
    selection: str = "eom",
) -> np.ndarray:
    """Cluster labels for one group of events (-1 = unassigned)."""
    n = len(feats)
    if n < min_cluster_size:
        return np.full(n, -1)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # allow_single_cluster: most electrodes carry exactly one unit, and HDBSCAN
        # would otherwise call that whole group noise.
        ms = min_samples if min_samples is not None else max(5, min_cluster_size // 4)
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=ms, cluster_selection_method=selection,
                       allow_single_cluster=True, copy=True).fit_predict(feats)
    if method == "optics":
        from sklearn.cluster import OPTICS

        ms = min_samples if min_samples is not None else max(5, min_cluster_size // 2)
        return OPTICS(min_samples=ms, xi=0.05, min_cluster_size=min_cluster_size).fit_predict(feats)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        best, best_bic = None, np.inf
        for k in range(1, min(gmm_max_k, n // min_cluster_size) + 1):
            g = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=2).fit(feats)
            bic = g.bic(feats)
            if bic < best_bic:
                best, best_bic = g, bic
        labels = best.predict(feats)
        for k in np.unique(labels):  # drop clusters that are too small to trust
            if (labels == k).sum() < min_cluster_size:
                labels[labels == k] = -1
        return labels
    raise ValueError("clustering must be 'hdbscan', 'optics' or 'gmm'")


# --------------------------------------------------------------------------- sorting
@dataclass
class SortingResult:
    spike_samples: np.ndarray  # int
    labels: np.ndarray  # int, -1 = unassigned
    fs: float
    unit_channel: np.ndarray  # peak channel of each unit id (index = unit id); -1 if unknown
    params: dict = field(default_factory=dict)
    runtime_s: float | None = None

    @property
    def spike_times(self) -> np.ndarray:
        return self.spike_samples / self.fs

    @property
    def unit_ids(self) -> np.ndarray:
        return np.unique(self.labels[self.labels >= 0])

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    def unit_spike_times(self, unit: int) -> np.ndarray:
        return np.sort(self.spike_times[self.labels == unit])


def sort_spikes(
    rec: Recording,
    threshold_sd: float = 5.0,
    features: str = "pca",
    clustering: str = "gmm",
    n_components: int = 5,
    n_neighbors: int = 3,
    min_cluster_size: int = 20,
    seed: int = 0,
    cluster_kwargs: dict | None = None,
) -> SortingResult:
    """Detect, extract, featurise and cluster (per peak channel). Input should be preprocessed."""
    import time

    t0 = time.perf_counter()
    det = detect_spikes(rec, threshold_sd=threshold_sd)
    wf, ok = extract_waveforms(rec, det, n_neighbors=n_neighbors)
    samples, chans = det.samples[ok], det.channels[ok]
    labels = np.full(len(samples), -1, dtype=int)
    unit_channel: list[int] = []
    for ch in np.unique(chans):
        sel = np.flatnonzero(chans == ch)
        if len(sel) < min_cluster_size:
            continue
        f = compute_features(wf[sel], n_components, features, seed)
        lab = cluster_features(f, clustering, min_cluster_size, seed=seed, **(cluster_kwargs or {}))
        for k in np.unique(lab[lab >= 0]):
            labels[sel[lab == k]] = len(unit_channel)
            unit_channel.append(int(ch))
    params = dict(threshold_sd=threshold_sd, features=features, clustering=clustering,
                  n_components=n_components, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
                  **(cluster_kwargs or {}))
    return SortingResult(samples, labels, rec.fs, np.array(unit_channel, dtype=int), params,
                         runtime_s=time.perf_counter() - t0)
