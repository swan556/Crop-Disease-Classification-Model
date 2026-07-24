# 🌾 Crop Leaf Disease Diagnosis System in PyTorch

A field-robust plant leaf disease classification system built with **PyTorch**, **Torchvision**, and **CUDA acceleration**. Designed to assist small-scale farmers in identifying crop diseases early from smartphone photographs to protect global food security.

---

## 📌 Features & Architecture Highlights

1. **Field-Robust Data Augmentations**:
   - Random resized cropping, 360° flips/rotations, and color jitter (brightness, contrast, hue shifts) to simulate direct sunlight, overcast skies, and unaligned camera angles.
2. **Dual Model Support**:
   - **Custom CNN (`LeafDiseaseCNN`)**: Built from scratch with 4 Conv2D-BatchNorm-ReLU-MaxPool blocks and Adaptive Global Average Pooling to learn spatial feature extraction mechanics.
   - **ResNet Transfer Learning (`ResNetLeafClassifier`)**: Fine-tuned ImageNet pre-trained ResNet18/ResNet50 backbone for high accuracy on domain-specific leaf patterns.
3. **End-to-End ML Pipeline**:
   - Automated dataset fetch & verification (`src/download_data.py`).
   - PyTorch `DataLoader` with dynamic train/val/test splits and CUDA memory pinning (`src/dataset.py`).
   - Training loop with AdamW optimizer, learning rate scheduling (`ReduceLROnPlateau`), and model checkpointing (`src/train.py`).
   - Diagnostic evaluation with Confusion Matrix and per-class classification reports (`src/evaluate.py`).
   - Standalone single-image inference (`src/predict.py`) and interactive Web UI (`app.py`).

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_data.py
```

### 3. Model Architecture Verification (Dry Run)
```bash
python src/models.py
```

### 4. Train Model
Train a fine-tuned ResNet18 model:
```bash
python src/train.py --model_type resnet18 --epochs 20 --batch_size 32 --lr 0.0003
```
Or train a custom CNN from scratch:
```bash
python src/train.py --model_type custom_cnn --epochs 20 --batch_size 32 --lr 0.0003
```

### 5. Evaluate Performance
```bash
python src/evaluate.py --checkpoint checkpoints/best_model_resnet18.pth
```

### 6. Run Single Image Inference
```bash
python src/predict.py --checkpoint checkpoints/best_model_resnet18.pth --image path/to/leaf.jpg
```

### 7. Launch Interactive Web Demo
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser to test leaf photos interactively.

---

## 📂 Project Directory Structure

```
Crop-Disease-Classification/
├── .venv/                      # Virtual environment
├── data/
│   └── PlantVillage/           # Leaf dataset (organized by class subfolders)
├── src/
│   ├── __init__.py
│   ├── download_data.py        # Dataset downloader & validator
│   ├── dataset.py              # PyTorch Dataset, DataLoaders, & Field Augmentations
│   ├── models.py               # Custom CNN & ResNet Transfer Learning architectures
│   ├── train.py                # Training & validation loop with checkpointing
│   ├── evaluate.py             # Diagnostic evaluation & confusion matrix generation
│   └── predict.py              # Single-image inference engine & treatment advice
├── app.py                      # Interactive Web App UI (Streamlit)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 🧪 License
Distributed under the MIT License.
