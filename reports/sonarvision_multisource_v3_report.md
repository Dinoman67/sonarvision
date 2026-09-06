# SonarVision Multi-Source Dataset v3 Report

## v2 → v3 Changes (sensor-isolated classes)

- mine = MILCO ONLY (Kaggle mines dropped — bimodal class, corr 0.26)
- unknown_debris = NOAA + MILCO NOMBO (catch-all, 2 real sensors)
- airplane / drowning_victim / wreck = Kaggle
- Leakage-free grouping + 512x512 normalize kept from v2

## Overall

- Total samples: 6250
- Usable samples: 5570
- Original samples: 5231
- Augmented samples: 339
- Duplicates removed: 680

## By Source

- KAGGLE: 795
- MILCO: 1170
- NOAA: 3605

## By Split

- test: 729
- train: 4058
- val: 783

## By Class

- airplane: 252
- drowning_victim: 190
- mine: 218
- unknown: 3824
- unknown_debris: 733
- wreck: 353

## Class × Split

| Class | Train | Val | Test |
|-------|-------|-----|------|
| unknown_debris | 557 | 98 | 78 |
| airplane | 186 | 39 | 27 |
| drowning_victim | 166 | 17 | 7 |
| mine | 192 | 8 | 18 |
| wreck | 251 | 49 | 53 |

## Source × Split

| Source | Train | Val | Test |
|--------|-------|-----|------|
| NOAA | 2585 | 528 | 492 |
| MILCO | 870 | 150 | 150 |
| KAGGLE | 603 | 105 | 87 |
