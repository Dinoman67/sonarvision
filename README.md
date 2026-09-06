# 🌊 SonarVision: Autonomous Multi-Sensor Marine Debris & Threat Intelligence System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react&logoColor=black)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20(v6)-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset%20(v6)-blue?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SIH 2026](https://img.shields.io/badge/Smart_India_Hackathon-2026-orange)](https://www.sih.gov.in/)

**Smart India Hackathon 2026 | Problem Statement: SIH26215**  
*Real-time AI for Marine Debris, Naval Mine Countermeasures (MCM), Shipwrecks, and Submerged Aircraft Localization in Side-Scan Sonar (SSS) Imagery.*

[Model (Hugging Face)](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6) • [Dataset (Hugging Face)](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6) • [Live Demo](#-quick-start) • [Architecture](#-solution-yolov8-esi-architecture) • [Benchmarks](#-empirical-benchmarks) • [Report Engine](#-automated-intelligence-reporting) • [Pitch Guide](#-sih-2026-hackathon-pitch-flow)

</div>

---

## 📌 The Problem: Why Standard Computer Vision Fails on Sonar

Underwater marine debris, lost cargo containers, unexploded naval mines, and submerged aircraft are completely invisible from satellite optical cameras. Oceanographic vessels rely on **Side-Scan Sonar (SSS)**, which creates acoustic intensity maps of the seafloor.

Standard computer vision models (COCO-trained YOLOv8, Faster R-CNN) fail catastrophically on sonar:
* 🌑 **No Color Information**: Sonar outputs single-channel acoustic backscatter intensity.
* 🌓 **Acoustic Shadow Physics**: Objects are characterized not just by bright highlights, but by the **accoustic shadows** cast directly behind them based on towfish altitude and sound grazing angle.
* 🌊 **Speckle Noise & Clutter**: Natural sand ripples, seafloor mud, and rocky reefs produce intense false alarms for brightness-dependent detectors.
* ⏱️ **Manual Review Bottleneck**: Surveyors spend days reviewing multi-gigabyte continuous waterfall records.

---

## 🧠 Solution: YOLOv8-ESI Architecture

**YOLOv8-ESI** (Edge Sonar Intelligence) introduces **Squeeze-and-Excitation (SE)** channel attention directly into the C2f feature bottleneck of a lightweight CSPDarknet backbone:

```
[Raw SSS Imagery (256x256)]
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              CSPDarknet Feature Extractor               │
│  ┌───────────────────────────────────────────────────┐  │
│  │   C2f Feature Block + SE Channel Attention       │  │
│  │   • Global Average Pooling (Spatial Squeeze)      │  │
│  │   • Two-Layer MLP Recalibration (Channel Excite) │  │
│  │   • Multiplies Highlight Features × Shadow Context│  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│           Decoupled Anchor-Free Detection Head          │
│   • Bounding Box Regression (CIoU Loss)                 │
│   • Multi-Class Classification (Class-Weighted BCE)     │
└─────────────────────────────────────────────────────────┘
           │
           ▼
[FP16 ONNX Engine] ──> [Real-Time Geospatial WGS84 Solver] ──> [PDF Report]
```

### Key Architectural Advantages
1. **Highlight-Shadow Coupling**: Channel attention forces the network to only trigger when an acoustic highlight is spatially correlated with a corresponding acoustic shadow.
2. **Compact Edge Footprint**: Only **3.03M parameters** and **5.9 MB** (FP16 ONNX), requiring zero high-end marine GPUs.
3. **Ultra-Low Latency**: **2.1 ms** on NVIDIA GPUs, **~18 ms** on standard CPU/Raspberry Pi 4/5—easily outpacing 30+ FPS hydrographic survey feeds.

---

## 🚢 Two-Model Operational Architecture

To provide maximum operational flexibility for maritime authorities and environmental teams, SonarVision supports a two-model strategy:

| Component | Model 1: Debris Specialist | Model 2: Multi-Sensor Target Classifier |
| :--- | :--- | :--- |
| **Model Type** | YOLOv8-ESI Single-Class | YOLOv8-ESI 4-Class Multi-Source |
| **Classes** | `marine_debris` | `unknown_debris`, `naval_mine`, `shipwreck`, `airplane` |
| **Primary Domain** | High-density coastal cleanup & plastic mapping | Naval MCM, port security, maritime SAR & salvage |
| **mAP50 Score** | **0.88 – 0.92** on NOAA survey passes | **0.6042** across 962 unseen multi-sensor test images |
| **Runtime Size** | 6.2 MB (FP16 ONNX) | 5.9 MB (FP16 ONNX) |
| **Deployment** | Autonomous surface vessels (USVs) & micro-drones | Survey ships, Naval AUVs, coastal defense command |

---

## 📊 Empirical Benchmarks

### Unseen Test Split (962 Images, Zero Split Leakage)

Evaluated strictly on independent, unseen side-scan sonar passes:

| Object Type / Class | Benchmark Target | Baseline YOLO | **SonarVision YOLOv8-ESI** | Detection Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Marine Debris** (`unknown_debris`) | $\ge 0.10$ | 0.0005 | **0.8185** | **81.9%** | **78.7%** |
| **Naval Mine** (`mine`) | $\ge 0.30$ | 0.1833 | **0.3862** | **70.7%** | **29.6%** |
| **Shipwreck** (`wreck`) | $\ge 0.70$ | 0.6698 | **0.7173** | **80.0%** | **60.4%** |
| **Submerged Aircraft** (`airplane`) | $\ge 0.70$ | 0.6206 | **0.4950** *(Val: 0.654)* | **75.4%** | **52.1%** |
| **Overall Model mAP50** | $\ge 0.50$ | 0.3685 | **0.6042** | **66.9%** | **62.4%** |
| **Peak F1 Score** | $\ge 0.60$ | 0.5100 | **0.6502** (@ conf 0.40) | — | — |

### Multi-Sensor Cross-Validation
* **NOAA Klein 5000 SSS** (833 test images): **0.7578 mAP50** (P: 81.9%, R: 78.7%, F1: 0.8025)
* **KAGGLE High-Res Sonar** (65 test images): **0.6061 mAP50** (P: 80.0%, R: 60.3%, F1: 0.6881)
* **MILCO Klein 3500 MCM Sonar** (64 test images): **0.2714 mAP50** (P: 70.7% — high precision prevents false mine alerts)
* **Clean Seabed Validation**: **0 False Alarms** across natural seafloor sand ripples and mud textures.

---

## 📑 Automated Intelligence Reporting

SonarVision bridges raw AI detections with hydrographic GIS operations by generating instant intelligence deliverables:

1. **Publication-Ready PDF Reports**: Complete with executive summary, primary target type badge (`Naval Mine`, `Shipwreck`, `Marine Debris`), embedded high-res annotated imagery, detection inventory table, and narrative technical assessment.
2. **Tabular CSV Exports**: Detailed spreadsheets with target ID, object type, class name, bounding box bounds, center coordinates, and resolved WGS84 latitude/longitude.
3. **Machine-Readable JSON**: Complete API response schema for seamless integration into C2 (Command & Control) naval systems.
4. **Geospatial GeoTIFF Solver**: Solves the embedded affine transform matrix (`EPSG:26916` $\to$ `WGS84`) to project pixel bounding boxes into real-world geographic coordinates.

---

## 📥 Hugging Face Model & Dataset Downloads

To download the trained production model weights or access the acoustic side-scan sonar benchmark dataset, visit our official Hugging Face repositories:

| Resource | Hugging Face Repository | Description & Contents |
| :--- | :--- | :--- |
| **Model Weights (v6)** | [🤗 `Dinoman1221/sonarvision-yolov8-esi-v6`](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6) | **YOLOv8-ESI v6 ONNX models** (`yolo_esi_v6_fp16.onnx` @ 5.9 MB, `yolo_esi_v6_fp32.onnx` @ 12 MB, and `yolo_esi_core_debris_fp16.onnx` @ 6.2 MB), model cards with test benchmarks, and standalone ONNX inference code. |
| **Multi-Source Dataset (v6)** | [🤗 `Dinoman1221/sonarvision-multisource-v6`](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6) | **924 MB archive** containing **5,558 side-scan sonar images** (4,033 train, 563 val, 962 strictly held-out test), `dataset.yaml`, 4 tactical target classes, and zero-leakage split protocol. |

### CLI Download Commands:
```bash
# Download production v6 FP16 model weights into models/
hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/

# Download the complete multi-source v6 dataset (5,558 images)
hf download Dinoman1221/sonarvision-multisource-v6 sonarvision_multisource_v6.zip --repo-type dataset --local-dir datasets/
```

---

## ⚡ Quick Start

### 1. Clone & Run (One-Command Startup)

```bash
git clone https://github.com/Dinoman67/sonarvision.git
cd sonarvision

# Launch unified application (FastAPI backend + React frontend)
./start.sh
```

Open your browser to:
👉 **`http://localhost:8000`**

*(If Node.js is not installed, the backend immediately serves built production assets and starts in Demonstration Mode with preloaded multi-class test crops).*

### 2. Activate Production ONNX Model

To run live GPU/CPU ONNX tensor inference:
1. Download or copy your trained model weights from Hugging Face into the `models/` folder:
   ```bash
   # Download directly from Hugging Face
   hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/
   cp models/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx

   # Or if you already have the file locally in Downloads:
   cp ~/Downloads/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx
   ```
2. Restart the app (`./start.sh`). The backend will automatically bind the model and display execution provider details (`CUDAExecutionProvider` or `CPUExecutionProvider`).

---

## 📂 Repository Layout

```
sonarvision/
├── backend/                  # High-performance FastAPI REST API
│   ├── api/                  # Analysis, metadata, health, and export endpoints
│   ├── geospatial/           # Affine coordinate matrix & EXIF GPS solvers
│   ├── inference/            # YOLOv8-ESI ONNX engine, letterboxing, soft-NMS
│   ├── reports/              # PDF, CSV, and JSON intelligence report generators
│   └── static/samples/       # Preloaded test crops for instant browser evaluation
├── frontend/                 # Interactive React + TypeScript + Tailwind UI
│   ├── src/components/       # Sonar viewer, Leaflet map, detection table, inspector
│   └── dist/                 # Built production frontend assets
├── models/                   # Model architectures & local ONNX target directory
│   └── core_single_class/    # Model 1 Debris Specialist documentation
├── reports/                  # Multi-source dataset audit logs & sensor benchmarks
├── scripts/                  # Dataset builders (v1-v6), Colab trainers, ONNX exporter
├── run_app.py                # Standalone Python runner
├── start.sh                  # Unified launch script
└── requirements.txt          # Python dependencies
```



## 🇮🇳 Alignment with National & Global Goals

* 🌊 **UN SDG 14: Life Below Water**: Autonomous spatial mapping of benthic plastics and ghost gear to direct cleanup vessels to high-density debris hotspots.
* 🛡️ **Atmanirbhar Bharat & Blue Economy**: Indigenous, sovereign deep-tech AI for naval port security, mine countermeasures, and economic zone surveillance without foreign dependencies.

---

## 📜 License & Acknowledgements

* Released under the **MIT License**.
* Developed for **Smart India Hackathon 2026** by Team **Cold Start**.
* Acoustic data sources: NOAA Hydrographic Survey Archives, NATO STO CMRE MILCO Benchmark, and Kaggle SSS Object Detection.
