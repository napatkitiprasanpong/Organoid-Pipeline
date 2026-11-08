import numpy as np

from orgpipe import SimConfig, preprocess, simulate_recording
from orgpipe.preprocess import estimate_noise
from orgpipe.scoring import count_matches, score_sorting, score_spike_times
from orgpipe.spikes import SortingResult


def _truth_sorting(rec, relabel=None, drop=None):
    tr = rec.truth
    samples = np.round(tr.spike_times * rec.fs).astype(int)
    labels = tr.spike_units.copy()
    if relabel is not None:
        labels = relabel(labels)
    if drop is not None:
        keep = ~drop(samples, labels)
        samples, labels = samples[keep], labels[keep]
    return SortingResult(samples, labels, rec.fs, tr.unit_channel.copy())


def _rec():
    raw = simulate_recording(SimConfig(fs=20_000.0, duration=10.0, n_units=8, seed=11))
    return preprocess(raw)


def test_count_matches_is_one_to_one_and_respects_tolerance():
    a = np.array([1.0, 2.0, 3.0])
    assert count_matches(a, a + 0.0004, 0.001) == 3
    assert count_matches(a, a + 0.002, 0.001) == 0
    # two true spikes competing for a single estimated spike: only one match
    assert count_matches(np.array([1.000, 1.0008]), np.array([1.0004]), 0.001) == 1
    assert count_matches(np.zeros(0), a, 0.001) == 0


def test_perfect_sorting_scores_one():
    rec = _rec()
    sd = estimate_noise(rec.data, rec.artifact_mask)
    r = score_sorting(rec, _truth_sorting(rec), sd)["summary"]
    assert r["n_well_detected"] == r["n_true_units"] == 8
    assert r["mean_accuracy"] > 0.999 and r["any_recall"] > 0.999 and r["any_precision"] > 0.999
    assert r["n_spurious_or_split"] == 0


def test_merged_units_and_dropped_spikes_are_penalised():
    rec = _rec()
    sd = estimate_noise(rec.data, rec.artifact_mask)
    merged = score_sorting(rec, _truth_sorting(rec, relabel=lambda l: np.where(l == 1, 0, l)), sd)
    assert merged["summary"]["n_well_detected"] < 8  # unit 0 and 1 share a label: at most one can match well
    half = score_sorting(rec, _truth_sorting(rec, drop=lambda s, l: s % 2 == 0), sd)["summary"]
    assert 0.35 < half["any_recall"] < 0.65 and half["any_precision"] > 0.99


def test_spikes_in_artifact_mask_are_ignored():
    rec = _rec()
    mask = np.zeros(rec.n_samples, dtype=bool)
    mask[: rec.n_samples // 2] = True  # first half of the recording is "blanked"
    sorting = _truth_sorting(rec, drop=lambda s, l: s < rec.n_samples // 2)  # we report nothing there
    r = score_spike_times(rec, sorting.spike_times, exclude_mask=mask)
    assert r["any_recall"] > 0.999  # no penalty for spikes we were not allowed to see
    r2 = score_spike_times(rec, sorting.spike_times)
    assert r2["any_recall"] < 0.9
