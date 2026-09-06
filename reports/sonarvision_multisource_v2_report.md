# SonarVision Multi-Source Dataset v2 Report

## v1 → v2 Changes

- Added NOAA h8 dataset (+3548 images)
- Oversampled drowning_victim 10x, airplane 3x
- Added sonar noise pipeline (Gaussian, speckle, brightness)

## Overall

- Total samples: 6306
- Usable samples: 5626
- Original samples: 5287
- Augmented samples: 339
- Duplicates removed: 680

## By Source

- KAGGLE: 851
- MILCO: 1170
- NOAA: 3605

## By Split

- test: 908
- train: 3809
- val: 909

## By Class

- airplane: 252
- drowning_victim: 190
- mine: 274
- unknown_debris: 4557
- wreck: 353

## Class × Split

| Class | Train | Val | Test |
|-------|-------|-----|------|
| unknown_debris | 3086 | 699 | 772 |
| airplane | 169 | 55 | 28 |
| drowning_victim | 150 | 14 | 26 |
| mine | 148 | 90 | 36 |
| wreck | 256 | 51 | 46 |

## Source × Split

| Source | Train | Val | Test |
|--------|-------|-----|------|
| NOAA | 2421 | 582 | 602 |
| MILCO | 777 | 198 | 195 |
| KAGGLE | 611 | 129 | 111 |
