
# OrgPipe (Organoid-Pipeline)

Open-source Python pipeline for analysing multi-electrode array (MEA) recordings from brain organoids: spike detection and sorting, burst and connectivity analysis, and reservoir-style decoding.

> **Status: early development.** This repository currently holds the project plan and documentation. The modules below are planned milestones, not finished features. Nothing here has been validated on real recordings yet.

## Why this project

Organoid-intelligence labs produce very large MEA recordings, and spike sorting, burst and connectivity analysis, and decoding are still open problems. Most of that work is software. OrgPipe aims to be a small, reproducible pipeline that other people can run end to end with one command.

Synthetic data with known ground truth is used to score every stage, so accuracy is measured rather than guessed. The same code is then pointed at real recordings.

## Pipeline stages

| # | Stage | What it does | How it is checked |
|---|-------|--------------|-------------------|
| 1 | Simulation | Multi-channel traces with several simulated neurons per electrode, bursting network activity, noise and optional stimulation artifacts | Spike times are known, so later steps can be scored |
| 1b | Real data | Ingest live MEA data recorded with FinalSpark's LiveMEA tool into HDF5 | Real files load through the same interface as synthetic ones |
| 2 | Preprocessing | Band-pass filtering, common-mode noise removal, stimulation artifact handling | Before/after plots and signal-to-noise change |
| 3 | Spike detection and sorting | Threshold detection, feature extraction, clustering, then a SpikeInterface sorter for comparison | Precision, recall and unit-match accuracy against ground truth; comparison with np-utils `SpikeSorting` |
| 4 | Activity analysis | Firing rates, inter-spike intervals, burst and network-burst detection, synchrony, cross-correlation, functional connectivity | Recover the simulated burst structure; then track live recordings over hours or days |
| 5 | Decoding | Binned spike counts, linear baseline, then a reservoir readout (fixed random recurrent layer plus trained linear output) | Cross-validated accuracy with confidence intervals vs chance and the linear baseline |
| 6 | Report | Auto-generated HTML or PDF summary per recording | A new recording runs end to end with one command |

## Planned repository layout

```
Organoid-Pipeline/
  README.md
  environment.yml        pinned dependencies
  data/                  synthetic data and notes on public datasets
  src/orgpipe/
    simulate.py          synthetic MEA generator
    ingest_live.py       wrapper around LiveMEA recordings
    preprocess.py        filtering, artifact handling
    spikes.py            detection, sorting wrappers, metrics
    network.py           bursts, synchrony, connectivity
    decode.py            linear and reservoir decoders
    report.py            summary figures and report
  notebooks/             one notebook per phase
  tests/                 unit tests, including ground-truth checks
  docs/                  lab notebook, report draft
```

## Milestones

| Milestone | Target week | Done when |
|-----------|-------------|-----------|
| M1 Simulator and preprocessing | 9 | Synthetic data generated; filtering and artifact handling documented |
| M2 Spike sorting benchmark | 12 | Own pipeline and one SpikeInterface sorter scored against ground truth, with a comparison table |
| M3 Network analysis module | 15 | Burst and connectivity outputs reproduce simulated structure |
| M4 Decoder | 19 | Reservoir readout beats the linear baseline on at least one task, or the reason it does not is explained |
| M5 Real-data run | 21 | Pipeline runs end to end on live LiveMEA recordings; limits of the feed documented |
| M6 Report and release | 24 | Tagged release, 4-6 page report, README with results |

## Tools

Planned stack: NumPy, SciPy, Matplotlib, scikit-learn, PyTorch, SpikeInterface, Neo, Elephant, NetworkX and Brian2. Exact versions will be pinned in `environment.yml` once code lands.

## Data sources and limitations

- **Synthetic data** is used for spike sorting, because it has ground truth.
- **FinalSpark LiveMEA** is the planned free real-data source. Per FinalSpark's public repository, each chunk is about every second, 4096 samples from each of 32 electrodes, across four arrays. FinalSpark only asks that you mention it as the data source, and the code is provided as-is with no maintenance guarantee. Check the live viewer's terms before relying on it.
- **Caveat (inference, not documented):** roughly 4,000 samples per second is probably too coarse for proper spike-waveform sorting. The live feed is therefore treated as suited to burst, activity and network analysis. Spike sorting stays on synthetic data, or on raw recordings obtained through a platform account.
- Following FinalSpark's best-practices guidance, real-data analyses should check that bursts are present, account for the natural decline in activity, and compare against inactive controls or activity tracked over time.

## Reporting results honestly

Results will report effect sizes and confidence intervals, and keep what is simulated separate from what is measured.

## Ethics

Organoid research involves human-derived neural tissue. This project is computational only. Results are described in measured terms, with no claims about sentience or consciousness that the data do not support.

## Acknowledgements

- [FinalSpark](https://github.com/FinalSpark-np) for the open LiveMEA tool and the np-utils and np-docs repositories.
- Background reading: the organoid intelligence roadmap (Smirnova, Hartung et al., 2023), DishBrain (Kagan et al., 2022), and Brainoware (Cai et al., 2023).

## Author

Napat Kitiprasanpong, Biomedical Engineering, Srinakharinwirot University.
GitHub: [napatkitiprasanpong](https://github.com/napatkitiprasanpong)

## License

Not yet chosen. A license file will be added before the first tagged release.
