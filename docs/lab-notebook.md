# Lab notebook

One entry per week: what I did, what worked, what failed, what is next.

## Week 1
- Scaffolded repo: `src/orgpipe` package, tests, CI, environment file, demo script.
- Wrote the synthetic MEA simulator (units, bursts, noise, hum, stim artifacts, ground truth) and
  preprocessing (artifact blanking, median reference, band-pass, SNR metric).
- First result (seed 0, 20 s, stim every 4 s): ground-truth spike SNR 2.17 -> 8.19 after preprocessing.
- Open: HDF5 round-trip test only runs where h5py is installed (CI will cover it).
- Next: np-docs best-practices page, clone np-utils and LiveMEA, start notebook for phase 1.

## M2 session (spike sorting)
- Wrote `spikes.py` (threshold detector with cross-channel de-duplication, waveform extraction on the peak channel
  plus 3 neighbours, PCA/ICA features, HDBSCAN/OPTICS/GMM clustering per peak channel), `scoring.py` (one-to-one
  spike matching, Hungarian unit matching, accuracy/precision/recall, artifact-mask aware), `sorters.py` (registry
  and an untested SpikeInterface wrapper) and `benchmark.py` + `scripts/run_benchmark.py`.
- First run: HDBSCAN found only 2 units. Cause: it cannot return a single cluster by default and most electrodes
  carry one unit. Fixed with `allow_single_cluster=True`.
- It was still poor (about 60% of detected spikes unassigned on seeds 0-4; 70% on the dev seed). Tested the "too few events" idea with a 120 s recording:
  it got worse (91% unassigned on the dev seed), so that explanation is wrong. Left unresolved; PCA+GMM is now the default.
- Result (seeds 0-4): PCA+GMM sorts 35 of 60 units with SNR>=6 to accuracy >= 0.8. 22 of the 25 that fail share
  an electrode with another unit. Detector alone finds 61% of spikes (half the units sit below 6 SNR).
- Mistake caught: my first "easy case" test failed because units had <20 spikes (below `min_cluster_size`);
  the test was wrong, the sorter was behaving as designed.
- Open: run a SpikeInterface sorter (never run), compare with np-utils, merge units split/merged across
  neighbouring electrodes.

