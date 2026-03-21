# 🔥 Fire Detection using YOLOv5

A computer vision project that fine-tunes **YOLOv5s** on a custom fire dataset to detect fire in images and video. Includes real-time webcam inference and CNN feature map visualization for model interpretability.

---

## 📌 Overview

This project:
- Fine-tunes YOLOv5s using transfer learning on a custom fire image dataset
- Detects fire in validation images and video files with bounding box overlays
- Visualizes intermediate CNN feature maps to understand model behavior
- Supports real-time live fire detection from a webcam feed

---

## 🗂️ Repository Structure

```
fire-detection/
│
├── train__1_.ipynb                  # Main training, evaluation, and visualization notebook
├── from_ultralytics_import_YOLO.py  # Real-time webcam inference script
│
├── fire.yaml                        # Dataset configuration (paths + class names)
├── input.mp4                        # Sample video for fire detection inference
│
├── weights/
│   ├── best.pt                      # Best YOLOv5 checkpoint (trained model)
│   └── yolov5su.pt                  # YOLOv5su pretrained base weights
│
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Pipeline

```
Fire Image Dataset (Google Drive)
        ↓
YOLOv5s — Transfer Learning (yolov5s.pt base)
  --img 640  --batch 16  --epochs 3+
        ↓
Detect fire in:
  ├── Validation images  (runs/detect/)
  ├── Video file         (input.mp4)
  └── Live webcam        (source=0)
        ↓
Evaluate: Loss Curves, mAP, Precision, Recall
        ↓
Visualize: CNN Feature Maps (--visualize flag)
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Detection Model | YOLOv5s (Ultralytics) |
| Inference API | Ultralytics YOLO |
| Computer Vision | OpenCV |
| Visualization | Matplotlib, animation |
| Platform | Google Colab (GPU) + Google Drive |

---

## 🏷️ Dataset Configuration (`fire.yaml`)

```yaml
path: /content/drive/MyDrive/fire/fire   # dataset root on Google Drive
train: train/images
val: val/images

nc: 1             # number of classes
names: ['fire']   # class names
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Weights | yolov5s.pt |
| Image Size | 640×640 |
| Batch Size | 16 |
| Epochs | 3 (prototype — increase for production) |
| Data Loader Workers | 1 |
| Config File | fire_config.yaml |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Gurnoor007/fire-detection.git
cd fire-detection
```

### 2. Clone YOLOv5 and install dependencies
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 3. Mount dataset (Google Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```
Place your dataset at `/content/drive/MyDrive/fire/fire/` with `train/images` and `val/images` subfolders.

### 4. Train the model
```bash
python train.py --img 640 --batch 16 --epochs 50 \
  --data ../fire.yaml --weights yolov5s.pt --workers 1
```

### 5. Run inference on images
```bash
python detect.py --weights runs/train/exp/weights/best.pt \
  --img 640 --conf 0.25 --source path/to/val/images/
```

### 6. Run inference on video
```bash
python detect.py --weights runs/train/exp/weights/best.pt \
  --img 640 --conf 0.25 --source input.mp4
```

### 7. Real-time webcam inference
```bash
python from_ultralytics_import_YOLO.py
```

---

## 🔬 Feature Map Visualization

To visualize what the model detects at each CNN layer:
```bash
python detect.py --weights best.pt --img 640 --conf 0.25 \
  --source your_image.jpg --visualize
```
Feature maps are saved to `runs/detect/<exp>/<image_name>/`. The `stage0_Conv_features.png` shows low-level activations (edges, brightness, fire-like textures).

---

## 📊 Evaluation

After training, evaluate using the saved `results.csv`:
```python
from utils.plots import plot_results
plot_results('runs/train/exp/results.csv')
```
Metrics tracked per epoch: box loss, objectness loss, class loss, precision, recall, mAP@0.5.

---

## 🎥 Real-Time Inference (`from_ultralytics_import_YOLO.py`)

```python
from ultralytics import YOLO
YOLO('best.pt')(source=0, show=True)
```
Set `source=0` for webcam or pass any image/video path. Requires `best.pt` in the working directory.

---

## 🙋 Author

**Gurnoor Singh Mander**
B.Tech CSE, Lovely Professional University (2022–2026)
- GitHub: [@Gurnoor007](https://github.com/Gurnoor007)
- Email: gurnoorgaby@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
