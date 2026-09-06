# SonarVision Multi-Source Dataset v4 Report

## v3 → v4 Changes (post-mortem fixes)

- mine = MILCO ONLY (Kaggle mines dropped — bimodal class, corr 0.26)
- NOMBO = own class 5 (nombo_contact) — was mixed into debris, tri-modal failure
- stratified group split: representative val/test per source x mode (2015 mines, h8 debris)
- unknown_debris = NOAA ONLY (e4 + h8) — NOMBO moved to its own class 5
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

- test: 767
- train: 4012
- val: 791

## By Class

- airplane: 252
- drowning_victim: 190
- mine: 218
- nombo_contact: 86
- unknown: 3824
- unknown_debris: 647
- wreck: 353

## Class × Split

| Class | Train | Val | Test |
|-------|-------|-----|------|
| unknown_debris | 471 | 89 | 87 |
| airplane | 178 | 37 | 37 |
| drowning_victim | 139 | 24 | 27 |
| mine | 145 | 24 | 49 |
| wreck | 247 | 53 | 53 |
| nombo_contact | 66 | 5 | 15 |

## Source × Split

| Source | Train | Val | Test |
|--------|-------|-----|------|
| NOAA | 2598 | 507 | 500 |
| MILCO | 850 | 170 | 150 |
| KAGGLE | 564 | 114 | 117 |
