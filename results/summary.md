# Results summary

All numbers are percentages on the GenImage test (`val`) folders; mean ± std over seeds where more than one seed was run.

## Trained on `sdv5` — clean test images (seeds: [0, 1, 2])

| Method | biggan | midjourney | wukong | sdv5 | adm | glide | vqdm | Avg ACC | Avg AP |
|---|---|---|---|---|---|---|---|---|---|
| LOTA (baseline, K=1) | 63.77 ± 1.79 | 71.17 ± 1.84 | 69.17 ± 1.74 | 74.03 ± 0.83 | 63.03 ± 2.36 | 60.23 ± 3.76 | 68.00 ± 0.54 | **67.06 ± 1.72** | 75.98 ± 1.49 |
| LOTA + MP inference only (K=4) | 69.53 ± 2.97 | 79.40 ± 3.78 | 80.10 ± 2.00 | 85.07 ± 2.16 | 69.27 ± 5.35 | 60.17 ± 5.01 | 74.53 ± 2.20 | **74.01 ± 3.31** | 84.25 ± 2.27 |
| LOTA-MP (proposed, K=4) | 75.83 ± 0.74 | 90.60 ± 2.21 | 85.87 ± 1.88 | 89.70 ± 1.91 | 77.67 ± 2.07 | 73.47 ± 1.28 | 82.13 ± 2.21 | **82.18 ± 1.05** | 91.71 ± 0.84 |

Δ (proposed − baseline), paired over seeds [0, 1, 2]: ACC +15.12 (per seed: +16.56, +16.81, +12.00), AP +15.73

### Number of fused patches K (trained on `sdv5`, Avg ACC / Avg AP)

| Checkpoint | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| LOTA (trained K=1) | 67.06 ± 1.72 / 75.98 ± 1.49 | 70.86 ± 2.45 / 80.59 ± 1.86 | 74.01 ± 3.31 / 84.25 ± 2.27 | 76.04 ± 4.22 / 86.57 ± 2.74 |
| LOTA-MP (trained K=4) | 72.44 ± 1.05 / 82.69 ± 1.15 | 77.58 ± 0.78 / 87.90 ± 1.02 | 82.18 ± 1.05 / 91.71 ± 0.84 | 85.10 ± 1.15 / 94.37 ± 0.84 |

### Fusion rule at K=4 (trained on `sdv5`, Avg ACC)

| Checkpoint | logit mean (proposed) | probability mean | majority vote |
|---|---|---|---|
| LOTA (trained K=1) | 74.01 ± 3.31 | 73.74 ± 3.19 | 71.24 ± 1.61 |
| LOTA-MP (trained K=4) | 82.18 ± 1.05 | 80.72 ± 1.21 | 76.73 ± 2.49 |

### Robustness to unseen degradations (trained on `sdv5`, Avg ACC)

| Method | clean | blur1 | jpeg90 |
|---|---|---|---|
| LOTA (baseline, K=1) | 67.06 ± 1.72 | 50.25 ± 0.45 | 49.65 ± 0.44 |
| LOTA + MP inference only (K=4) | 74.01 ± 3.31 | 50.05 ± 0.53 | 50.97 ± 0.36 |
| LOTA-MP (proposed, K=4) | 82.18 ± 1.05 | 54.11 ± 0.93 | 50.96 ± 0.21 |

## Trained on `biggan` — clean test images (seeds: [0])

| Method | biggan | midjourney | wukong | sdv5 | adm | glide | vqdm | Avg ACC | Avg AP |
|---|---|---|---|---|---|---|---|---|---|
| LOTA (baseline, K=1) | 70.30 | 51.70 | 55.30 | 58.90 | 66.90 | 67.80 | 65.20 | **62.30** | 68.21 |
| LOTA + MP inference only (K=4) | 75.50 | 52.30 | 60.30 | 61.30 | 74.20 | 71.30 | 70.90 | **66.54** | 73.83 |
| LOTA-MP (proposed, K=4) | 77.40 | 62.00 | 61.80 | 67.20 | 75.90 | 74.40 | 71.30 | **70.00** | 78.49 |

Δ (proposed − baseline), paired over seeds [0]: ACC +7.70 (per seed: +7.70), AP +10.28

### Number of fused patches K (trained on `biggan`, Avg ACC / Avg AP)

| Checkpoint | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| LOTA (trained K=1) | 62.30 / 68.21 | 64.64 / 71.29 | 66.54 / 73.83 | 68.91 / 76.25 |
| LOTA-MP (trained K=4) | 64.00 / 69.99 | 66.89 / 74.25 | 70.00 / 78.49 | 72.54 / 81.63 |

### Fusion rule at K=4 (trained on `biggan`, Avg ACC)

| Checkpoint | logit mean (proposed) | probability mean | majority vote |
|---|---|---|---|
| LOTA (trained K=1) | 66.54 | 66.10 | 64.69 |
| LOTA-MP (trained K=4) | 70.00 | 69.49 | 68.97 |

## Additional ablations (clean test images, Avg ACC / Avg AP)

| Run | Trained on | Evaluated with | Avg ACC | Avg AP |
|---|---|---|---|---|
| `lota_mp_inst` | sdv5 | K=4 | 86.79 | 95.22 |
| `lota_resize_first` | sdv5 | K=1 | 56.49 | 62.07 |

## Efficiency

Device: Tesla T4; batch size 1; median over 100 images. Parameters: 23.51 M for every K (shared backbone).

| K | patch extraction (ms) | classification (ms) | total (ms) |
|---|---|---|---|
| 1 | 1.90 | 8.94 | 10.85 |
| 2 | 1.90 | 8.74 | 10.65 |
| 4 | 1.90 | 8.78 | 10.68 |
| 8 | 1.90 | 11.72 | 13.62 |

## Training time

| Run | minutes per training run (mean) | runs |
|---|---|---|
| `lota` | 3.6 | 4 |
| `lota_mp` | 10.6 | 4 |
| `lota_mp_inst` | 10.7 | 1 |
| `lota_resize_first` | 3.9 | 1 |

