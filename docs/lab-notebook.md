# Lab notebook

One entry per week: what I did, what worked, what failed, what is next.

## Week 1
- Scaffolded repo: `src/orgpipe` package, tests, CI, environment file, demo script.
- Wrote the synthetic MEA simulator (units, bursts, noise, hum, stim artifacts, ground truth) and
  preprocessing (artifact blanking, median reference, band-pass, SNR metric).
- First result (seed 0, 20 s, stim every 4 s): ground-truth spike SNR 2.17 -> 8.19 after preprocessing.
- Open: HDF5 round-trip test only runs where h5py is installed (CI will cover it).
- Next: np-docs best-practices page, clone np-utils and LiveMEA, start notebook for phase 1.
