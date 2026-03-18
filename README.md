<div align="center">

# 🔍 SceneSolver

### AI-Powered Forensic Video Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)

**🎯 98% binary accuracy · 88% F1 multi-class** on the UCF-Crime benchmark — runs on a 4GB GTX 1650

**[🌐 Try the Live Demo](https://scenesolver-ai.onrender.com)** — no account needed

![SceneSolver Demo](assets/demo.gif)

</div>

---

## Overview

SceneSolver ingests a video and runs it through a **5-model AI pipeline** with an early-exit architecture. A binary CLIP model acts as a gatekeeper — BLIP and BART only activate when crime is actually detected, saving significant compute on normal frames.

Trained and evaluated on the **UCF-Crime dataset** across **5 crime classes:** Fighting, Shooting, Explosion, Robbery, and Shoplifting. Outputs a downloadable PDF forensic incident report.

---

## Pipeline

```
Video Input
    │
    ▼
[1] Binary CLIP         ──→  Normal frame? → Skip (saves compute)
    │ Crime detected
    ▼
[2] Multi-class CLIP    ──→  Crime type: Fighting / Theft / Explosion / Shooting ...
    │
    ▼
[3] BLIP Captioner      ──→  "A person throws a punch near the storefront entrance"
    │
    ▼
[4] YOLOv8 + ByteTrack  ──→  Tracked objects: person ×5, backpack ×1 ...
    │
    ▼
[5] BART Summarizer     ──→  Forensic incident summary paragraph
    │
    ▼
PDF Report + Crime Clip Export
```

### System Architecture

![System Architecture](assets/SYSTEM%20ARCHITECHURE.jpg)

---

## Results

### Training — Multi-class Classifier (5 Crime Types)

![Training Curves](assets/5%20clss%20clssifier%20loss.png)

### Evaluation Metrics

|  | Binary Classifier | Multi-class Classifier |
|--|:-----------------:|:----------------------:|
| **Metrics** | ![Binary Metrics](assets/Binary_Classifier_-Crime_vs._Normal-_metrics.png) | ![Multi-class Metrics](assets/Multi-class_Classifier_-5_Crime_Types-_metrics.png) |
| **Confusion Matrix** | ![Binary CM](assets/Binary_Classifier_-Crime_vs._Normal-_confusion_matrix.png) | ![Multi-class CM](assets/Multi-class_Classifier_-5_Crime_Types-_confusion_matrix.png) |

---

## Features

| Feature | Description |
|---|---|
| 🎯 **5-class crime detection** | Fighting, Shooting, Explosion, Robbery, Shoplifting |
| 🚀 **Early-exit architecture** | Binary CLIP gates expensive models — BLIP/BART skip normal frames entirely |
| 📡 **Live stream & RTSP support** | Real-time analysis via webcam or IP camera (DirectShow backend) |
| 🎬 **Pre-event clip extraction** | 5-second rolling buffer captures context leading up to the crime |
| 🔍 **ByteTrack object tracking** | Persistent person IDs across the full video duration |
| 📄 **PDF forensic reports** | One-click downloadable report with verdict, tracked objects, and AI summary |
| 🔐 **Auth + history** | Session-based login, hashed passwords, per-user analysis history in MongoDB |

---

## Optimization

The full pipeline is ~6–8 GB of raw model weights. After optimization, it runs entirely on a **4 GB GTX 1650**:

| Technique | Impact |
|---|---|
| **Classifier head extraction** | 16 MB saved weights vs 500 MB full model — 70% VRAM reduction |
| **FP16 safetensors (BLIP)** | Model size halved with no quality loss |
| **BitsAndBytes 8-bit (GPU)** | BLIP loaded in 8-bit — 4× smaller in VRAM |
| **Dynamic INT8 quantization (CPU)** | BART + classifiers auto-quantized when no GPU is detected |
| **OpenCV motion pre-filter** | Static/zero-motion frames skipped before hitting the GPU |
| **Batch processing (size 8)** | Full GPU pipeline utilization — not naive frame-by-frame |
| **Lazy BART loading** | Summarizer loaded on first request only — faster startup |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask + Gunicorn |
| Database | MongoDB Atlas |
| Deep learning | PyTorch 2.3, Hugging Face Transformers |
| Vision models | OpenAI CLIP ViT-B/32, Salesforce BLIP, YOLOv8n |
| Language model | Facebook BART-large-CNN |
| Object tracking | ByteTrack (via Ultralytics) |
| PDF generation | ReportLab |
| Video processing | OpenCV, Pillow |

---

## Local Setup

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- NVIDIA GPU recommended (CPU supported, but slower)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/WrishG/scenesolver-ai.git
cd scenesolver-ai
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv && venv\Scripts\activate

# macOS/Linux
python3 -m venv venv && source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download model weights**

Download the trained weights from Hugging Face and place them in the `models/` folder:
```bash
mkdir models
# Required: auto_head_multi.pth, binary_head.pth, blip_finetuned.safetensors, yolov8n.pt
```

> Model weights are not committed to this repo due to file size. [Contact me](#contact) for access.

**5. Configure environment variables**
```bash
cp .env.example .env
# Edit .env and fill in SECRET_KEY and MONGO_URI
```

**6. Run**
```bash
flask run
# → http://127.0.0.1:5000
```

---

## Demo Deployment

A lightweight demo version (`app_demo.py`) with pre-computed results is deployed on Render's free tier using only Flask + Gunicorn (~50 MB RAM).

```bash
# Run demo locally
python app_demo.py
```

---

## Contact

**Ashish** — [mnvashish@gmail.com](mailto:mnvashish@gmail.com)
