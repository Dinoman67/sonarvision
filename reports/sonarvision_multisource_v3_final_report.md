# SonarVision v3 — FINAL Dataset Verification Report

Generated: final prep run | dataset: `datasets/sonarvision_multisource_v3`

## 1. Dataset size

- Images: 5570  (train 4058 / val 783 / test 729)
- Boxes: 4099
- Background-only images: train 2706 / val 572 / test 546
- Format: 512x512 grayscale PNG, uniform

## 2. Class x Sensor matrix (boxes) — sensor isolation check

| class | NOAA | MILCO | KAGGLE | total |
|---|---|---|---|---|
| unknown_debris | 803 | 231 | 0 | 1034 |
| airplane | 0 | 0 | 265 | 265 |
| drowning_victim | 0 | 0 | 1950 | 1950 |
| mine | 0 | 435 | 0 | 435 |
| wreck | 0 | 0 | 415 | 415 |

> EXPECTED: unknown_debris = NOAA+NOMBO(MILCO), mine = MILCO only, airplane/drowning/wreck = KAGGLE only.

## 3. Per-class boxes per split

| class | train | val | test | total | %train |
|---|---|---|---|---|---|
| unknown_debris | 828 | 116 | 90 | 1034 | 80% |
| airplane | 199 | 39 | 27 | 265 | 75% |
| drowning_victim | 1632 | 150 | 168 | 1950 | 84% |
| mine | 398 | 13 | 24 | 435 | 91% |
| wreck | 302 | 56 | 57 | 415 | 73% |

## 4. Suggested class weights (inverse-sqrt normalized, from train boxes)

| class | train boxes | weight (cls_pw) |
|---|---|---|
| unknown_debris | 828 | 1.404 |
| airplane | 199 | 2.864 |
| drowning_victim | 1632 | 1.000 |
| mine | 398 | 2.025 |
| wreck | 302 | 2.325 |

## 5. Cross-sensor compatibility (measured during prep)

- **MILCO mine vs KAGGLE mine (REMOVED in v3):** mean-crop correlation 0.26, opposite aspect (1.80 vs 0.83), 19x relative area → bimodal class, dropped. Kaggle mine-only images removed: 61.
- **NOAA debris vs NOMBO debris (KEPT):** correlation 0.57, near-identical brightness/contrast (mean 50.9/48.6, std 28.4/31.9), similar small-wide-echo geometry → compatible catch-all class; NOMBO boxes also serve as MILCO-mine in-image negatives.
- Montages: `reports/mine_sensor_compare.png`, `reports/debris_sensor_compare.png`

## 6. Verification battery results

- audit_dataset_definitive.py: PASS (0 missing files, 0 aug-copy leaks, 0 group leaks, 0 degenerate boxes, all 512x512)
- audit_multisource_dataset.py: 0 issues, no cross-split duplicates
- Mock training (smoke_train_v3.py): PASS — custom ESI model (3.18M params) trained 2 epochs on mock subset + validated with per-class AP50 on 5 classes (MX550 GPU)

## 7. Known caveats

- mine val/test coverage is thin (13/24 boxes) because MILCO mine batches are grouped; per-source + per-class eval in the Colab script will show if this needs attention.
- drowning_victim dominates box count (Kaggle 10x oversample); weights above compensate.