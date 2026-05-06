# 🍅 Tomato AI Diagnostics

A transfer-learning application that detects tomato leaf diseases from photos or a live webcam feed, built with TensorFlow/Keras and Streamlit.

---

## Features

- **Static image inference** — upload a leaf photo and get an instant diagnosis with confidence score and treatment recommendation
- **Live webcam inference** — real-time disease detection overlaid directly on the video stream (updates every ~20 frames)
- **Training dashboard** — two-stage transfer learning with live Plotly learning curves that update every epoch
- **CLI pipeline** — train, evaluate, predict, and convert models from the terminal
- **TFLite export** — convert trained models for edge deployment (Raspberry Pi 5)

---

## Disease Classes

| Class | Folder Name |
|---|---|
| Bacterial Spot | `Tomato___Bacterial_spot` |
| Early Blight | `Tomato___Early_blight` |
| Healthy | `Tomato___healthy` |
| Late Blight | `Tomato___Late_blight` |
| Leaf Mold | `Tomato___Leaf_Mold` |
| Septoria Leaf Spot | `Tomato___Septoria_leaf_spot` |
| Spider Mites | `Tomato___Spider_mites Two-spotted_spider_mite` |
| Target Spot | `Tomato___Target_Spot` |
| Tomato Mosaic Virus | `Tomato___Tomato_mosaic_virus` |
| Yellow Leaf Curl Virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |

---

## Project Structure

```
├── app.py                    # Streamlit entry point
├── main.py                   # CLI entry point
├── pipeline.py               # Core ML: train, predict, evaluate, convert
├── config.py                 # Disease classes, page config, CSS
├── streamlit_callback.py     # Keras callback → real-time Streamlit charts
├── Makefile                  # Task automation
├── components/
│   ├── main_ui.py            # Inference + Training Dashboard tabs
│   └── sidebar.py            # Confidence, disease filter, ROI, system stats
├── src/
│   ├── data/dataset.py       # Dataset splitting (80/10/10)
│   ├── inference/predictor.py
│   ├── training/trainer.py
│   └── utils/
│       ├── recommendations.py  # Disease → treatment text
│       ├── predictions.py
│       ├── training_utils.py
│       └── helpers.py
├── data/
│   ├── raw/raw/tomato/       # Original dataset (class subfolders)
│   └── splits/               # train / val / test splits
├── models/
│   ├── trained/              # Saved Keras models
│   └── tflite/               # TFLite exports
├── outputs/                  # Learning curve HTML files
├── logs/
├── tests/
├── requirements-frontend.txt
├── requirements-backend.txt
└── requirements-dev.txt
```

---

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install all dependencies
make setup

# Or manually
python -m venv venv
venv\Scripts\pip install -r requirements-frontend.txt
venv\Scripts\pip install -r requirements-backend.txt
```

### 2. Prepare dataset

Download the [Tomato Leaf Disease dataset from Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) and extract to `data/raw/raw/tomato/`.

```bash
# Split into train / val / test (80/10/10)
python -m src.data.dataset --split
```

### 3. Run the app

```bash
make run
# or
venv\Scripts\python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Training

### Via the UI (recommended)

Open the **Training Dashboard** tab, set your parameters, and click **🚀 Start Training**.  
Live accuracy and loss curves update every epoch. Final curves are saved to `outputs/`.

### Via the CLI

```bash
# Full two-stage pipeline (feature extraction → fine-tuning)
make train

# Or with custom parameters
make train EPOCHS_FE=15 EPOCHS_FT=20 BATCH_SIZE=32

# Single stage
python main.py train --freeze-base --epochs 10 --learning-rate 0.001

# Evaluate on test set
python main.py evaluate --model-path models/trained/latest.keras --test-dir data/splits/test

# Run inference on an image
python main.py predict --model-path models/trained/latest.keras --image-path leaf.jpg
```

### Two-stage transfer learning

| Stage | Base model | Learning rate | Purpose |
|---|---|---|---|
| 1 — Feature extraction | Frozen | 0.001 | Train classification head only |
| 2 — Fine-tuning | Unfrozen | 0.0001 | Adapt entire network to tomato leaves |

---

## Edge Deployment (Raspberry Pi 5)

```bash
# Convert to TFLite with quantization
make convert-int8

# Benchmark
make benchmark

# Copy to Pi
scp models/tflite/*.tflite pi@<ip>:~/tomato-app/
```

Expected Pi 5 performance (MobileNetV3Small, INT8):
- Inference: ~80–120ms/image
- Model size: ~2.5 MB
- RAM: < 500 MB

---

## Live Webcam

Select **📷 Live Webcam** in the Inference tab. The browser will request camera permission — click **Allow**.

> **Note:** Camera access requires HTTPS on non-localhost URLs. Use [ngrok](https://ngrok.com) or configure SSL if accessing over a network.

---

## Requirements

| File | Contents |
|---|---|
| `requirements-frontend.txt` | streamlit, plotly, pillow, pandas, streamlit-webrtc, av |
| `requirements-backend.txt` | tensorflow, keras, opencv-python-headless, scikit-learn, psutil, numpy |
| `requirements-dev.txt` | All of the above + pytest, ruff |
