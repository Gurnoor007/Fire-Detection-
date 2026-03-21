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

## 🙋 Author

**Gurnoor Singh Mander**
- GitHub: [@Gurnoor007](https://github.com/Gurnoor007)
- Email: gurnoorgaby@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
