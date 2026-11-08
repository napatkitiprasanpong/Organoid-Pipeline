Seeds [0, 1, 2, 3, 4], 30 s at 20000 Hz, 24 units, 32 channels, stimulation every 4 s. Mean ± SD over seeds; spike tolerance 1.0 ms.

| sorter | units acc>=0.8 (of true) | units acc>=0.8 (of SNR>=6) | mean acc (SNR>=6) | spike recall | spike precision | sorted units | spurious/split | runtime (s) |
|---|---|---|---|---|---|---|---|---|
| detector only (no sorting) | - | - | - | 0.61 ± 0.13 | 0.98 ± 0.01 | - | - | - |
| pca+gmm | 7.6 ± 1.1 | 7.0 ± 1.4 | 0.70 ± 0.09 | 0.57 ± 0.13 | 0.99 ± 0.00 | 13.0 ± 1.4 | 2.2 ± 1.3 | 2.34 ± 0.21 |
| pca+optics | 6.8 ± 1.5 | 6.2 ± 1.5 | 0.65 ± 0.09 | 0.55 ± 0.11 | 0.99 ± 0.00 | 12.6 ± 1.3 | 2.4 ± 0.9 | 2.42 ± 0.20 |
| pca+hdbscan | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.31 ± 0.06 | 0.24 ± 0.07 | 1.00 ± 0.00 | 12.4 ± 1.5 | 9.2 ± 3.3 | 1.96 ± 0.11 |
| ica+hdbscan | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.31 ± 0.06 | 0.23 ± 0.05 | 1.00 ± 0.00 | 12.2 ± 1.3 | 9.4 ± 3.0 | 2.61 ± 0.31 |
