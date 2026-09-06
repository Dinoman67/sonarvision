# SonarVision Multi-Source Dataset v5 Report

## v4 → v5 Changes (post-v4-run autopsy)

- unknown_debris = curated h8_data ONLY (one visual mode, brightness 42-50). NOAA-e4 source folder dropped (dim mode never scored on any test in v3/v4).
- NOMBO dropped as a class; NOMBO-only images excluded entirely (unlabeled mine-like echoes would poison mine training).
- drowning_victim dropped (19 originals; val/test were 3-original near-duplicate lotteries). Kaggle = airplane + wreck.
- h8_data re-split WHOLE-PASS group-aware: the user's train/val folders shared 39/39 target passes (61/72 val positives from train passes) and h8_unseen_test drew 48/48 positives from passes in h8_data — near-duplicate frames of one pass now land in ONE split. h8_unseen_test is NOT used as test.
- MILCO re-split: finer groups (batch 15 vs 50) + per-year 70/15/15 so mine test stops being 76% 2015 while train is 21% 2015.
- Oversampling (airplane 3x, mine 2x) applied AFTER the split, TRAIN-ONLY — noise-aug copies never enter val/test.
- 4 classes: unknown_debris (h8) / airplane (Kaggle) / mine (MILCO) / wreck (Kaggle)

## Overall

- Total samples: 5402
- Usable samples: 5394
- Original samples: 5098
- Augmented samples (train only): 296
- Duplicates removed: 8

## By Source

- KAGGLE: 557
- MILCO: 1297
- NOAA: 3540

## By Split

- test: 734
- train: 3954
- val: 706

## By Class

- airplane: 204
- mine: 431
- unknown: 3781
- unknown_debris: 625
- wreck: 353

## Class × Split

| Class | Train | Val | Test |
|-------|-------|-----|------|
| unknown_debris | 434 | 92 | 99 |
| airplane | 180 | 12 | 12 |
| mine | 352 | 50 | 29 |
| wreck | 247 | 53 | 53 |

## Source × Split

| Source | Train | Val | Test |
|--------|-------|-----|------|
| NOAA | 2508 | 513 | 519 |
| MILCO | 1019 | 128 | 150 |
| KAGGLE | 427 | 65 | 65 |

## Detail (year/mode) × Split

| Detail | Train | Val | Test |
|--------|-------|-----|------|
| 2010 | 267 | 37 | 45 |
| 2015 | 177 | 8 | 15 |
| 2017 | 115 | 1 | 0 |
| 2018 | 434 | 67 | 75 |
| 2021 | 26 | 15 | 15 |
| SSS | 427 | 65 | 65 |
| h8 | 2508 | 513 | 519 |
