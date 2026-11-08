"""Sorters that can be benchmarked side by side.

``OWN_SORTERS`` are configurations of :func:`orgpipe.spikes.sort_spikes`.
``sort_spikeinterface`` runs any installed SpikeInterface sorter and returns the
same ``SortingResult``, so both go through the same scoring code.

Note on "np-utils style": ``pca+hdbscan``, ``ica+hdbscan`` and ``pca+optics`` use
the same *kind* of recipe as FinalSpark's np-utils ``SpikeSorting`` (ICA or PCA,
then HDBSCAN or OPTICS), but they are this repository's implementation with
its own detector and parameters. They are **not** np-utils itself, and results
here say nothing about np-utils' own performance.
"""

from __future__ import annotations

import tempfile
import time

import numpy as np

from .recording import Recording
from .spikes import SortingResult, sort_spikes

OWN_SORTERS: dict[str, dict] = {
    "pca+gmm": dict(features="pca", clustering="gmm"),
    "pca+optics": dict(features="pca", clustering="optics"),
    "pca+hdbscan": dict(features="pca", clustering="hdbscan"),
    "ica+hdbscan": dict(features="ica", clustering="hdbscan"),
}


def sort_own(rec: Recording, name: str, **overrides) -> SortingResult:
    if name not in OWN_SORTERS:
        raise KeyError(f"unknown sorter {name!r}; choose from {sorted(OWN_SORTERS)}")
    return sort_spikes(rec, **{**OWN_SORTERS[name], **overrides})


def sort_spikeinterface(
    rec: Recording, sorter_name: str, folder: str | None = None, **sorter_params
) -> SortingResult:
    """Run a SpikeInterface sorter on ``rec`` (needs ``pip install spikeinterface probeinterface`` plus
    whatever the chosen sorter needs).

    ``rec`` should be the same preprocessed recording the own sorters get. Many SpikeInterface sorters
    filter internally by default; pass their own parameters through ``sorter_params`` if you want to
    avoid double filtering (see ``spikeinterface.get_default_sorter_params(sorter_name)``).

    NOT VERIFIED in the environment this was written in (no network, SpikeInterface not installed):
    the API calls follow the documented ``NumpyRecording`` / ``run_sorter`` interface, but expect to
    adjust for your installed SpikeInterface version.
    """
    try:
        import spikeinterface.full as si
        from probeinterface import Probe
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError("sort_spikeinterface needs: pip install spikeinterface probeinterface") from e

    traces = np.ascontiguousarray(rec.data.T.astype(np.float32))  # (n_samples, n_channels)
    recording = si.NumpyRecording([traces], sampling_frequency=rec.fs)
    probe = Probe(ndim=2)
    probe.set_contacts(positions=rec.channel_xy.astype(float), shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(rec.n_channels))
    recording = recording.set_probe(probe)

    out = folder or tempfile.mkdtemp(prefix=f"orgpipe_si_{sorter_name}_")
    t0 = time.perf_counter()
    sorting = si.run_sorter(sorter_name, recording, folder=out, remove_existing_folder=True,
                            verbose=False, **sorter_params)
    runtime = time.perf_counter() - t0

    samples, labels = [], []
    for k, uid in enumerate(sorting.unit_ids):
        s = np.asarray(sorting.get_unit_spike_train(uid), dtype=int)
        samples.append(s)
        labels.append(np.full(len(s), k))
    samples = np.concatenate(samples) if samples else np.zeros(0, dtype=int)
    labels = np.concatenate(labels) if labels else np.zeros(0, dtype=int)
    order = np.argsort(samples, kind="stable")
    return SortingResult(
        samples[order], labels[order], rec.fs, np.full(len(sorting.unit_ids), -1, dtype=int),
        {"spikeinterface_sorter": sorter_name, **sorter_params}, runtime_s=runtime,
    )
