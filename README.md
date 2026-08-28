# 🌊 SonarVision

**Deep learning for side-scan sonar marine debris detection**

Detect underwater debris in side-scan sonar (SSS) imagery using YOLOv8 variants optimized for sonar data.

## Models

| Model | Params | Size | Architecture |
|-------|--------|------|-------------|
| **YOLOv8n** | 3.01M | 6.2MB | Standard YOLOv8 nano (baseline) |
| **SS-YOLO** | 1.66M | 3.5MB | GhostConv + FastC2f (47% lighter) |
| **YOLOv8-ESI** | 3.18M | 6.3MB | C2f + SE attention (better texture) |

All models export to ONNX and are under 80MB for edge deployment.

## Dataset

**E5 Dataset** — NOAA H11833 SSS Marine Debris with realistic noise augmentation:

- **Train**: 953 images (153 debris + 200 clean BG + 600 noisy BG)
- **Val**: 201 images (90 debris + 111 clean BG)
- **Test**: 183 images (45 debris + 138 clean BG)

Noise types: speckle, nadir artifacts, acoustic shadows, brightness variation, seabed textures.

## Quick Start

### Local Training

```bash
# Generate E5 dataset (with noisy backgrounds)
python scripts/generate_e5_noisy.py --e4 datasets/noaa-debris/e4 --e5 datasets/noaa-debris/e5

# Train all 3 models and compare
python scripts/train_sss_comparison.py --dataset datasets/noaa-debris/e5/data.yaml --epochs 100
```

### Google Colab

1. Upload `datasets/noaa-debris/e5.zip` to Colab
2. Open `scripts/colab_sss_train_and_test.ipynb`
3. Run all cells

## Project Structure

```
sonarvision/
├── datasets/
│   └── noaa-debris/
│       ├── e4/          # Original dataset (clean backgrounds)
│       └── e5/          # Fixed dataset (with noisy backgrounds)
├── models/
│   ├── sss_custom_modules.py   # GhostConv, FastC2f, SEBlock, WaveletConv
│   └── build_sss_models.py     # Build SS-YOLO and YOLOv8-ESI
├── scripts/
│   ├── generate_e5_noisy.py           # SSS noise augmentation
│   ├── train_sss_comparison.py        # Train & compare all models
│   ├── colab_sss_train_and_test.ipynb # Colab notebook
│   └── colab_e4_train.ipynb          # Legacy E4 training
└── README.md
```

## Problem We Solved

**Initial issue**: Training data had clean backgrounds with no sonar noise. Model detected everything as debris on real noisy SSS data.

**Fix**: Added 600 realistic noisy background images with SSS-specific artifacts:
- Speckle noise (coherent imaging artifact)
- Nadir line dropout (sonar geometry)
- Acoustic shadows (bright→dark transitions)
- Brightness/contrast variation (gain drift)
- Sand ripples and rock fields (seabed textures)

## Requirements

```
ultralytics>=8.4.0
torch>=2.0.0
opencv-python
numpy
pillow
matplotlib
pandas
```

## License

MIT

## Acknowledgments

- NOAA for the H11833 side-scan sonar dataset
- Ultralytics for YOLOv8
- SS-YOLO paper: "A Lightweight Deep Learning Model Focused on Side-Scan Sonar Target Detection"
- YOLOv8-ESI paper: "Underwater object detection in side-scan sonar images"
