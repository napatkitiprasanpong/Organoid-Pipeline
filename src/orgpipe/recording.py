"""Recording container and file I/O.

A ``Recording`` holds a (n_channels, n_samples) array in microvolts plus the
metadata the rest of the pipeline needs. Synthetic recordings also carry a
``GroundTruth`` so every later stage can be scored. Real recordings (for
example from FinalSpark LiveMEA) have ``truth = None`` and load through the
same interface.

Files: ``.h5`` / ``.hdf5`` (needs h5py) or ``.npz``. Both use the same flat
layout: named arrays plus a JSON ``header`` string holding fs and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class GroundTruth:
    """Known structure of a synthetic recording."""

    spike_times: np.ndarray  # seconds, sorted; time of the waveform trough
    spike_units: np.ndarray  # unit id for each spike
    unit_channel: np.ndarray  # primary (largest-amplitude) channel per unit
    unit_xy: np.ndarray  # (n_units, 2) positions in micrometres
    unit_amp_uv: np.ndarray  # peak negative amplitude per unit, microvolts
    burst_windows: np.ndarray  # (n_bursts, 2) network-burst start/end, seconds

    @property
    def n_units(self) -> int:
        return len(self.unit_channel)


@dataclass
class Recording:
    data: np.ndarray  # (n_channels, n_samples), microvolts
    fs: float  # sampling rate, Hz
    channel_xy: np.ndarray  # (n_channels, 2), micrometres
    stim_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    truth: GroundTruth | None = None
    artifact_mask: np.ndarray | None = None  # bool (n_samples,), True = blanked
    meta: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs


_TRUTH_FIELDS = [f.name for f in fields(GroundTruth)]


def _to_arrays(rec: Recording) -> tuple[dict[str, np.ndarray], str]:
    arrays: dict[str, np.ndarray] = {
        "data": np.asarray(rec.data, dtype=np.float32),
        "channel_xy": np.asarray(rec.channel_xy, dtype=np.float64),
        "stim_times": np.asarray(rec.stim_times, dtype=np.float64),
    }
    if rec.artifact_mask is not None:
        arrays["artifact_mask"] = np.asarray(rec.artifact_mask, dtype=bool)
    if rec.truth is not None:
        for name in _TRUTH_FIELDS:
            arrays[f"truth__{name}"] = np.asarray(getattr(rec.truth, name))
    header = json.dumps({"fs": float(rec.fs), "meta": rec.meta}, default=_json_default)
    return arrays, header


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)} in recording metadata")


def _from_arrays(arrays: dict[str, np.ndarray], header: str) -> Recording:
    head = json.loads(header)
    truth = None
    if all(f"truth__{n}" in arrays for n in _TRUTH_FIELDS):
        truth = GroundTruth(**{n: arrays[f"truth__{n}"] for n in _TRUTH_FIELDS})
    return Recording(
        data=arrays["data"],
        fs=float(head["fs"]),
        channel_xy=arrays["channel_xy"],
        stim_times=arrays["stim_times"],
        truth=truth,
        artifact_mask=arrays.get("artifact_mask"),
        meta=head.get("meta", {}),
    )


def save_recording(rec: Recording, path: str | Path) -> Path:
    """Save to ``.h5``/``.hdf5`` or ``.npz`` (chosen by file suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, header = _to_arrays(rec)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py  # imported lazily so .npz works without it

        with h5py.File(path, "w") as f:
            for name, arr in arrays.items():
                if name == "data":
                    chunk = (rec.n_channels, min(rec.n_samples, 65536))
                    f.create_dataset(name, data=arr, chunks=chunk, compression="gzip")
                else:
                    f.create_dataset(name, data=arr)
            f.attrs["header"] = header
    elif suffix == ".npz":
        np.savez_compressed(path, header=np.array(header), **arrays)
    else:
        raise ValueError(f"Unsupported suffix {suffix!r}; use .h5, .hdf5 or .npz")
    return path


def load_recording(path: str | Path) -> Recording:
    """Load a recording written by :func:`save_recording`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            arrays = {name: f[name][()] for name in f.keys()}
            header = f.attrs["header"]
            if isinstance(header, bytes):
                header = header.decode()
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files if name != "header"}
            header = str(z["header"])
    else:
        raise ValueError(f"Unsupported suffix {suffix!r}; use .h5, .hdf5 or .npz")
    return _from_arrays(arrays, header)
