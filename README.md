<div align="center">

# 🔍 SceneSolver
### AI-Powered Forensic Video Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)

**[🌐 Live Demo](https://scenesolver-ai.onrender.com)** — no account needed

![SceneSolver Demo](assets/demo.gif)

</div>

---

## What It Does

SceneSolver ingests a video and runs it through a **5-model AI pipeline** to automatically detect crime, identify objects, generate forensic captions, and produce a downloadable PDF incident report — in under 2 minutes on CPU.

Trained and evaluated on the **UCF-Crime dataset** (8 crime classes).

---

## 5-Stage Analysis Pipeline

```
Video Input
    │
    ▼
[1] Binary CLIP         ──→  Normal frame? → Skip (saves compute)
    │ Crime frame
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

---

## Key Features

- 🎯 **8-class crime classification** — Fighting, Theft, Shooting, Explosion, Arson, Abuse, Burglary, Robbery
- 📡 **Live stream support** — analyze RTSP/webcam feeds in real time
- 🎬 **Crime clip extraction** — auto-saves short clips around detected crime frames
- 📄 **PDF report export** — one-click downloadable forensic report with verdict, objects, and summary
- 🔐 **User authentication** — session-based login with hashed passwords and MongoDB
- 📊 **Analysis history** — all past analyses saved per user

---

## Optimization Highlights

The full pipeline is ~6-8GB of models. To make it deployable:

| Technique | What it does |
|---|---|
| **Classifier head extraction** | Saves only the 16MB trained head, not the full 500MB CLIP model |
| **FP16 safetensors** | BLIP model halved in size with no quality loss |
| **BitsAndBytes 8-bit (GPU)** | BLIP loaded in 8-bit on GPU — 4× smaller, faster |
| **Dynamic INT8 quantization (CPU)** | BART + classifiers auto-quantized on CPU-only servers |
| **Batch frame processing** | Frames processed in batches — exponentially faster than one-by-one |
| **Lazy BART loading** | Summarizer only loaded on first request |

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
- NVIDIA GPU recommended (runs on CPU too, slower)

### 1. Clone
```bash
git clone https://github.com/WrishG/scenesolver-ai.git
cd scenesolver-ai
```

### 2. Virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv && source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
Download the trained weights from Hugging Face and place them in the `models/` folder:
```bash
mkdir models
# Download: auto_head_multi.pth, binary_head.pth, blip_finetuned.safetensors, yolov8n.pt
```
> Model weights are not committed to this repo due to file size. Contact for access.

### 5. Environment variables
```bash
cp .env.example .env
# Fill in SECRET_KEY and MONGO_URI in .env
```

### 6. Run
```bash
flask run
# → http://127.0.0.1:5000
```

---

## Live Demo Deployment

A demo-only version (`app_demo.py`) with pre-computed results is deployed on Render free tier using only Flask + Gunicorn (~50MB RAM).

```bash
# Run demo locally
python app_demo.py
```

---

## Contact

**Wrish** — [Wrishg@gmail.com](mailto:Wrishg@gmail.com)

🔗 [GitHub Repo](https://github.com/WrishG/scenesolver-ai)
