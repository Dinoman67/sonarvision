# SIH 2026 Live Demo & Presentation Test Suite
**SonarVision: Autonomous Marine Target & Debris Intelligence System**

This directory contains an audited, 100% verified demo test pack drawn strictly from the **unseen test split** of the multi-source acoustic dataset (`datasets/sonarvision_multisource_v6`).

---

## 🎯 Demo Portfolio Overview

| # | Filename | Object Type | Model Class | Sensor Source | Expected Detection & Confidence | Purpose for Judges |
|---|---|---|---|---|---|---|
| **01** | `01_debris_noaa_003553.png` | **Marine Debris** | `unknown_debris` | NOAA Klein 5000 SSS | 2 targets @ **90.5%**, **87.1%** | Marine pollution / hazard detection |
| **02** | `02_debris_noaa_003621.png` | **Marine Debris** | `unknown_debris` | NOAA Klein 5000 SSS | 1 target @ **90.0%** | Isolated debris highlight & acoustic shadow |
| **03** | `03_mine_milco_000103.png` | **Naval Mine** | `mine` | MILCO Klein 3500 MCM | 5 targets @ up to **89.8%** | Mine Countermeasures (MCM) tactical alert |
| **04** | `04_mine_milco_000111.png` | **Naval Mine** | `mine` | MILCO Klein 3500 MCM | 4 targets @ up to **88.1%** | Sub-pixel mine detection on high noise seabed |
| **05** | `05_wreck_kaggle_000084.png` | **Shipwreck** | `wreck` | High-Res Side-Scan Sonar | 1 target @ **97.0%** | Navigation obstruction & underwater heritage |
| **06** | `06_wreck_kaggle_000265.png` | **Shipwreck** | `wreck` | High-Res Side-Scan Sonar | 1 target @ **95.1%** | Large hull contour and acoustic shadow |
| **07** | `07_airplane_kaggle_000029.png` | **Submerged Aircraft** | `airplane` | High-Res Side-Scan Sonar | 1 target @ **94.6%** | Downed aircraft SAR (Search and Rescue) |
| **08** | `08_airplane_kaggle_000055.png` | **Submerged Aircraft** | `airplane` | High-Res Side-Scan Sonar | 1 target @ **94.2%** | Aircraft wing & empennage acoustic footprint |
| **09** | `09_clean_noaa_seabed.png` | **Clean Seabed** | *(No Targets)* | NOAA Klein 5000 SSS | **0 detections** (Zero False Alarms) | Flat seabed baseline verification |
| **10** | `10_clean_milco_sand_ripples.png`| **Clean Seabed** | *(No Targets)* | MILCO Klein 3500 SSS | **0 detections** (Zero False Alarms) | Sand ripple rejection (no false positives) |
| **11** | `11_clean_milco_deep_sand.png` | **Clean Seabed** | *(No Targets)* | MILCO Klein 3500 SSS | **0 detections** (Zero False Alarms) | Acoustic backscatter noise robustness |

---

## 📑 Reports & Intelligence Exports

Unlike generic bounding-box detectors, the SonarVision report engine classifies each target by its **Object Type** (e.g. `Naval Mine`, `Shipwreck`, `Submerged Aircraft`, `Marine Debris`), calculates geospatial coordinates, and generates full multi-format reports:

* **Executive Summary**: Displays `PRIMARY OBJECT TYPE`, `CLASSIFIED TYPES`, and detection status.
* **Target Detection Inventory**: Includes `Object Type (Class)`, confidence %, pixel bounds, center coordinates, and lat/lon.
* **Technical Interpretation**: Explains acoustic highlight & shadow features detected by the Squeeze-and-Excitation (SE) attention blocks.

Pre-computed sample reports for all 11 images are available under:
`SIH_demo/precomputed_reports/<image_name>/`
* `report.pdf`: Publication-ready Intelligence PDF report with embedded annotation and tables.
* `detections.csv`: Structured tabular export with `object_type`, `class_name`, bounding coordinates, and confidence.
* `results.json`: Full machine-readable API payload.
* `annotated.png`, `colormap.png`, `evidence.png`, `mask.png`: Multi-spectral visual verification panels.

---

## 🎤 Hackathon Pitch Flow (Live Presentation Guide)

1. **Start with Clean Seabed (`09_clean_noaa_seabed.png` or `10_clean_milco_sand_ripples.png`)**:
   * *Pitch point*: "Notice that complex acoustic textures like seafloor sand ripples do NOT trigger false alarms. The model reports 0 detections, proving low operational false-alarm rate."
2. **Demonstrate Marine Debris (`01_debris_noaa_003553.png`)**:
   * *Pitch point*: "Deploying SSS for environmental cleanup — the model isolates 2 marine debris targets at 90.5% confidence."
3. **Demonstrate Defense & Tactical MCM (`03_mine_milco_000103.png`)**:
   * *Pitch point*: "In mine warfare exercises, the detector identifies 5 separate bottom mines with 89.8% peak confidence."
4. **Demonstrate Maritime Safety & Search & Rescue (`05_wreck_kaggle_000084.png` & `07_airplane_kaggle_000029.png`)**:
   * *Pitch point*: "For maritime safety and SAR, large wrecks and submerged aircraft are classified with 94%+ confidence."
5. **Download the PDF Report**:
   * Click **Download PDF Report** in the web app to show the judges the official intelligence report containing classified object types, technical interpretation, and inventory table.
