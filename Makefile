# =============================================================================
# 🍅 Unified Makefile: Tomato Disease Classification (v2.1)
# Frontend: Streamlit | Backend: TensorFlow/Keras | Target: Cross-Platform + Pi 5
# Usage: make <target> [VAR=value]
# Example: make train EPOCHS_FE=15 MODEL=efficientnetb0 DEVICE=cpu
# =============================================================================

.PHONY: help setup install install-dev install-frontend install-backend \
	venv structure data split-data feature-extraction fine-tune train \
	evaluate predict convert convert-int8 benchmark frontend frontend-dev \
	test lint format check clean dist-clean pi5-quickstart deploy \
	run dev build logs notebook

# -----------------------------------------------------------------------------
# ⚙️ OS Detection & Environment Configuration
# -----------------------------------------------------------------------------
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
PROJECT_ROOT := $(shell pwd)

ifeq ($(OS),Windows_NT)
    # Use backslashes for CMD compatibility
    VENV_BIN := venv\Scripts
    PYTHON := $(VENV_BIN)\python.exe
    PIP := $(VENV_BIN)\pip.exe
    STREAMLIT := $(VENV_BIN)\streamlit.exe
    RM := rmdir /s /q
    SHELL := cmd.exe
else
    VENV_BIN := venv/bin
    PYTHON := $(VENV_BIN)/python
    PIP := $(VENV_BIN)/pip
    STREAMLIT := $(VENV_BIN)/streamlit
    RM := rm -rf
    SHELL := /bin/bash
endif


# -----------------------------------------------------------------------------
# 📊 Training Variables (Override via CLI)
# -----------------------------------------------------------------------------
MODEL        ?= mobilenetv3small
INPUT_SHAPE  ?= 224 224 3
BATCH_SIZE   ?= 16
DEVICE       ?= cpu
OUT_DIR      ?= models/trained/$(TIMESTAMP)

# Stage 1: Feature Extraction (frozen base model)
EPOCHS_FE    ?= 10
LR_FE        ?= 0.001

# Stage 2: Fine-Tuning (unfrozen base model)
EPOCHS_FT    ?= 20
LR_FT        ?= 0.0001

# -----------------------------------------------------------------------------
# 🎯 Primary Entry Points
# -----------------------------------------------------------------------------
help: ## 📖 Show this help message with categorized targets
	@echo "🍅 Tomato Disease Classification - Makefile v2.1"
	@echo "================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup           # Full project initialization"
	@echo "  make frontend        # Launch Streamlit UI"
	@echo "  make train           # Run 2-stage training pipeline"
	@echo "  make pi5-quickstart  # Raspberry Pi 5 optimized setup"
	@echo ""
	@echo "📦 Installation:"
	@grep -E '^install[-a-z]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🧠 Training:"
	@grep -E '^(feature-extraction|fine-tune|train|evaluate|predict):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🎨 Frontend:"
	@grep -E '^frontend[-a-z]*:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📱 Deployment:"
	@grep -E '^(convert|benchmark|deploy|pi5-quickstart):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🔧 Development:"
	@grep -E '^(test|lint|format|check|clean|dist-clean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "💡 Tips:"
	@echo "  • Override vars: make train EPOCHS_FE=15 BATCH_SIZE=32"
	@echo "  • Pi 5 optimization: INPUT_SHAPE='160 160 3' BATCH_SIZE=8"
	@echo "  • View logs: make logs"

# -----------------------------------------------------------------------------
# 📦 Project Setup & Installation
# -----------------------------------------------------------------------------
setup: structure venv install ## 🚀 Full project setup (structure + venv + deps)
	@echo "✅ Setup complete!"
	@echo "👉 Next steps:"
	@echo "   • Run UI: make frontend"
	@echo "   • Train model: make train"
	@echo "   • Download data: make data"

venv: ## 🟢 Create virtual environment if it doesn't exist
ifeq ($(OS),Windows_NT)
	@if not exist "venv" (python -m venv venv && echo ✅ Venv created.)
else
	@if [ ! -d "venv" ]; then python3 -m venv venv && echo "✅Venv created."; 
endif


install: venv ## 📦 Install all production dependencies
	@echo "📦 Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip 
	$(PIP) install -r requirements-frontend.txt 
	$(PIP) install -r requirements-backend.txt
	$(PIP) install matplotlib plotly psutil 
	@echo "✅ Production dependencies installed"

install-frontend: venv ## 🎨 Install frontend dependencies only
	@echo "🎨 Installing frontend dependencies..."
	$(PIP) install -r requirements-frontend.txt $(NULL)
	@echo "✅ Frontend ready"

install-backend: venv ## 🧠 Install backend/ML dependencies only
	@echo "🧠 Installing backend dependencies..."
	$(PIP) install -r requirements-backend.txt $(NULL)
	$(PIP) install matplotlib plotly $(NULL)
	@echo "✅ Backend ready"

install-dev: install ## 🔧 Install + development/QA tools
	@echo "🔧 Installing dev tools..."
	$(PIP) install -r requirements-dev.txt $(NULL)
	$(PIP) install -e . $(NULL)
	@echo "✅ Dev environment ready"

structure: ## 🗂️ Create project directory structure
	@echo "🗂️ Creating project structure..."
	$(PYTHON) setup_project.py
	$(MKDIR) $(call FIX_PATH,models/final) $(NULL)
	$(MKDIR) $(call FIX_PATH,models/tflite) $(NULL)
	$(MKDIR) $(call FIX_PATH,outputs) $(NULL)
	$(MKDIR) $(call FIX_PATH,logs) $(NULL)
	$(MKDIR) $(call FIX_PATH,data/raw) $(NULL)
	$(MKDIR) $(call FIX_PATH,data/splits) $(NULL)
	@echo "✅ Directory structure created"

# -----------------------------------------------------------------------------
# 🧠 ML Pipeline: Two-Stage Transfer Learning
# -----------------------------------------------------------------------------
data: ## 📥 Show dataset download instructions
	@echo "📦 Dataset Setup"
	@echo "================"
	@echo "1. Download from Kaggle:"
	@echo "   https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf"
	@echo ""
	@echo "2. Extract to: data/raw/tomato/"
	@echo ""
	@echo "3. Expected classes (10 total):"
	@echo "   • Tomato___Early_blight"
	@echo "   • Tomato___Late_blight"
	@echo "   • Tomato___Bacterial_spot"
	@echo "   • Tomato___Target_Spot"
	@echo "   • Tomato___Yellow_Leaf_Curl_Virus"
	@echo "   • Tomato___Tomato_mosaic_virus"
	@echo "   • Tomato___Septoria_leaf_spot"
	@echo "   • Tomato___Spider_mites_Two_spotted_spider_mite"
	@echo "   • Tomato___Leaf_Mold"
	@echo "   • Tomato___Healthy"
	@echo ""
	@echo "4. Then run: make split-data"

split: ## ✂️ Create train/validation/test splits (80/10/10)
	@echo "✂️ Splitting dataset (80/10/10)..."
	$(PYTHON) -m src.data.dataset --split
	@echo "✅ Splits created in data/splits/"

feature-extraction: ## 🧊 Stage 1: Train classifier head (base frozen)
	@echo "🧊 Stage 1: Feature Extraction"
	@echo "   Model: $(MODEL) | Input: $(INPUT_SHAPE) | LR: $(LR_FE)"
	@echo "   Output: $(OUT_DIR)/feature_extracted.keras"
	$(MKDIR) $(call FIX_PATH,$(OUT_DIR)) $(NULL)
	$(PYTHON) main.py train \
		--base-model $(MODEL) \
		--input-shape $(INPUT_SHAPE) \
		--freeze-base \
		--learning-rate $(LR_FE) \
		--epochs $(EPOCHS_FE) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--output-path $(OUT_DIR)/feature_extracted.keras

fine-tune: ## 🧪 Stage 2: Unfreeze base + fine-tune with low LR
	@echo "🧪 Stage 2: Fine-Tuning"
	@echo "   Model: $(MODEL) | LR: $(LR_FT) | Epochs: $(EPOCHS_FT)"
	@echo "   Output: $(OUT_DIR)/best_model.keras"
	$(PYTHON) main.py train \
		--load-weights $(OUT_DIR)/feature_extracted.keras \
		--base-model $(MODEL) \
		--input-shape $(INPUT_SHAPE) \
		--unfreeze-base \
		--learning-rate $(LR_FT) \
		--epochs $(EPOCHS_FT) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--output-path $(OUT_DIR)/best_model.keras

train: feature-extraction fine-tune ## 🎓 Full 2-stage training pipeline

evaluate: ## 📊 Evaluate model on test set
	@echo "📊 Evaluating model..."
	$(PYTHON) main.py evaluate \
		--model-path $(OUT_DIR)/best_model.keras \
		--test-dir data/splits/test

predict: ## 🔍 Run inference on sample image
	@echo "🔍 Running prediction..."
	$(PYTHON) main.py predict \
		--model-path $(OUT_DIR)/best_model.keras \
		--image-path mytomato/mytomato.jpg

# -----------------------------------------------------------------------------
# 🎨 Frontend: Streamlit Application
# -----------------------------------------------------------------------------
frontend: ## 🌐 Run Streamlit app (standard mode)
	@echo "🌐 Starting Streamlit frontend..."
	@echo "   URL: http://localhost:8501"
	$(PYTHON) -m streamlit run app.py --server.headless true

frontend-dev: ## 🛠️ Run Streamlit with auto-reload + debug logging
	@echo "🛠️ Starting Streamlit in dev mode..."
	@echo "   Auto-reload: enabled | Logging: debug"
	$(PYTHON) -m streamlit run app.py --server.runOnSave true --server.headless true --logger.level debug

# -----------------------------------------------------------------------------
# 📱 Deployment: TensorFlow Lite & Edge Optimization
# -----------------------------------------------------------------------------
convert: ## 🔄 Convert Keras → TensorFlow Lite (default optimizations)
	@echo "🔄 Converting to TensorFlow Lite..."
	@echo "   Input: $(OUT_DIR)/best_model.keras"
	@echo "   Output: models/tflite/tomato_model_$(TIMESTAMP).tflite"
	$(MKDIR) $(call FIX_PATH,models/tflite) $(NULL)
	$(PYTHON) main.py convert \
		--keras-model $(OUT_DIR)/best_model.keras \
		--output-path models/tflite/tomato_model_$(TIMESTAMP).tflite \
		--optimizations default

convert-int8: ## 🗜️ Convert with INT8 quantization (smaller, faster for edge)
	@echo "🗜️ Converting with INT8 quantization..."
	@echo "   Expected: 4x smaller, 2-4x faster inference"
	$(MKDIR) $(call FIX_PATH,models/tflite) $(NULL)
	$(PYTHON) main.py convert \
		--keras-model $(OUT_DIR)/best_model.keras \
		--output-path models/tflite/tomato_model_int8_$(TIMESTAMP).tflite \
		--optimizations default,int8

benchmark: ## ⚡ Benchmark TFLite model performance
	@echo "⚡ Running benchmark (100 iterations)..."
	$(PYTHON) -m src.inference.benchmark \
		--model-path models/tflite/*.tflite \
		--num-runs 100

pi5-quickstart: structure install-backend ## 🚀 Raspberry Pi 5 optimized quick start
	@echo "🍅 Raspberry Pi 5 Quick Start"
	@echo "================================"
	@echo "✅ Project structure created"
	@echo "✅ Backend dependencies installed"
	@echo ""
	@echo "⚡ Recommended Pi 5 training command:"
	@echo "   make train \\"
	@echo "       MODEL=mobilenetv3small \\"
	@echo "       INPUT_SHAPE='160 160 3' \\"
	@echo "       BATCH_SIZE=8 \\"
	@echo "       EPOCHS_FE=8 EPOCHS_FT=12 \\"
	@echo "       DEVICE=cpu"
	@echo ""
	@echo "📱 For edge deployment:"
	@echo "   make convert-int8"
	@echo "   make benchmark"
	@echo ""
	@echo "💡 Expected Pi 5 performance (MobileNetV3Small, 160px, INT8):"
	@echo "   • Inference: ~80-120ms/image @ 1.8GHz"
	@echo "   • Model size: ~2.5 MB"
	@echo "   • RAM usage: <500 MB"
	@echo "   • Accuracy: ~94-96% (vs 97-98% desktop)"

deploy: convert benchmark ## 📦 Full deployment pipeline
	@echo "🚀 Deployment artifacts ready:"
	@echo "   📁 models/tflite/tomato_model_$(TIMESTAMP).tflite"
	@echo ""
	@echo "📋 Next steps for Raspberry Pi 5:"
	@echo "   1. Copy model to Pi:"
	@echo "      scp models/tflite/*.tflite pi@<ip>:~/tomato-app/"
	@echo "   2. On Pi, install runtime:"
	@echo "      pip install tflite-runtime"
	@echo "   3. Run inference:"
	@echo "      python -m src.inference.tflite_predictor \\"
	@echo "        --model model.tflite --image leaf.jpg"

# -----------------------------------------------------------------------------
# 🔧 Development & Quality Assurance
# -----------------------------------------------------------------------------
test: ## 🧪 Run pytest test suite with coverage
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint: ## 🔍 Run ruff linter (fast, modern replacement for flake8)
	@echo "🔍 Running linter..."
	$(PYTHON) -m ruff check src/ tests/ app.py config.py setup_project.py

format: ## 🎨 Format code with ruff (black-compatible)
	@echo "🎨 Formatting code..."
	$(PYTHON) -m ruff format src/ tests/ app.py config.py setup_project.py
	$(PYTHON) -m ruff check src/ tests/ app.py config.py setup_project.py --fix

check: format lint test ## ✅ Full quality pipeline (format → lint → test)
	@echo "✅ All quality checks passed!"

# -----------------------------------------------------------------------------
# 🧹 Cleanup & Maintenance
# -----------------------------------------------------------------------------
clean: ## 🧹 Light clean: Remove caches and temp files
	@echo "🧹 Cleaning temporary files..."
	$(RM) $(call FIX_PATH,__pycache__) $(NULL)
	find . -type d -name "__pycache__" -exec $(RM) {} $(NULL) \;
	find . -type f -name "*.pyc" -delete
	$(RM) $(call FIX_PATH,.pytest_cache) $(NULL)
	$(RM) $(call FIX_PATH,.ruff_cache) $(NULL)
	$(RM) $(call FIX_PATH,.mypy_cache) $(NULL)
	$(RM) $(call FIX_PATH,.streamlit) $(NULL)
	$(RM) $(call FIX_PATH,outputs/*.png) $(NULL)
	$(RM) $(call FIX_PATH,logs/*.log) $(NULL)
	@echo "✨ Temp files removed"

dist-clean: clean ## 🧨 Heavy clean: Remove venv AND all trained models
	@echo "🧨 Performing distribution clean..."
	$(RM) $(call FIX_PATH,venv) $(NULL)
	$(RM) $(call FIX_PATH,models/final/*) $(NULL)
	$(RM) $(call FIX_PATH,models/tflite/*) $(NULL)
	$(RM) $(call FIX_PATH,*.egg-info) $(NULL)
	@echo "✨ Project reset to initial state"

# -----------------------------------------------------------------------------
# 💡 Convenience Aliases & Helpers
# -----------------------------------------------------------------------------
run: frontend ## Alias: make run = make frontend
dev: frontend-dev ## Alias: make dev = make frontend-dev
build: train convert ## Alias: make build = train + convert for deployment
quick: pi5-quickstart ## Alias: make quick = pi5-quickstart

logs: ## 📋 Tail recent training logs
	@echo "📋 Recent logs (last 50 lines):"
	@tail -n 50 logs/*.log 2>/dev/null || echo "No logs found in logs/"

notebook: ## 📓 Launch Jupyter notebook for experimentation
	@echo "📓 Starting Jupyter notebook..."
	$(PYTHON) -m notebook --notebook-dir=. --no-browser

# -----------------------------------------------------------------------------
# 📋 Required Files Reference (Create these in project root)
# -----------------------------------------------------------------------------
# requirements-frontend.txt:
#   streamlit>=1.30.0
#   pillow>=10.0.0
#   numpy>=1.24.0
#   pandas>=2.0.0
#
# requirements-backend.txt:
#   tensorflow>=2.15.0
#   keras>=2.15.0
#   opencv-python-headless>=4.8.0
#   scikit-learn>=1.3.0
#   matplotlib>=3.7.0
#
# requirements-dev.txt:
#   -r requirements-frontend.txt
#   -r requirements-backend.txt
#   pytest>=8.0.0
#   pytest-cov>=4.1.0
#   ruff>=0.3.0
#   jupyter>=1.0.0