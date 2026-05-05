#!/usr/bin/env python3
"""
setup_project.py
Creates the directory structure and skeleton files for the 
Tomato Disease Classification project. Safe to run multiple times.
"""

import sys
from pathlib import Path


def create_dirs(base: Path, dirs: list[str]) -> None:
    """Create directories and add .gitkeep to track them in Git."""
    for d in dirs:
        dir_path = base / d
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"  📁 Created: {d}/")


def write_file(base: Path, rel_path: str, content: str) -> None:
    """Write file only if it doesn't exist (idempotent)."""
    file_path = base / rel_path
    if file_path.exists():
        print(f"  ⏭️  Skipped (exists): {rel_path}")
        return
    file_path.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"  ✅ Created: {rel_path}")


def main() -> None:
    base = Path(__file__).parent.resolve()
    print(f"🍅 Initializing project structure in: {base}\n")

    # 1️⃣ Directories
    dirs = [
        "data/raw/tomato",
        "data/splits",
        "models/final",
        "models/tflite",
        "outputs",
        "logs",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/utils",
        "tests",
        "components",
    ]
    create_dirs(base, dirs)

    # 2️⃣ Files
    files = {
        "__init__.py": "# Package marker",
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/models/__init__.py": "",
        "src/training/__init__.py": "",
        "src/inference/__init__.py": "",
        "src/utils/__init__.py": "",
        "tests/__init__.py": "",
        "components/__init__.py": "",
        "config.py": '''\
"""Project configuration and constants."""
PAGE_CONFIG = {
    "page_title": "Tomato Disease Classification",
    "page_icon": "🍅",
    "layout": "wide"
}

DISEASE_CLASSES = [
    "Early Blight", "Late Blight", "Bacterial Spot", "Target Spot",
    "Tomato Yellow Leaf Curl", "Tomato Mosaic Virus", "Septoria Leaf Spot",
    "Spider Mites", "Leaf Mold", "Healthy"
]

DEFAULT_MODEL = "mobilenetv3small"
INPUT_SHAPE = (224, 224, 3)
''',
        "app.py": '''\
"""Streamlit frontend for Tomato Disease Classification."""
import streamlit as st
from config import PAGE_CONFIG

st.set_page_config(**PAGE_CONFIG)
st.title("🍅 Tomato Disease Classification")
st.info("Upload a tomato leaf image to begin classification.")
''',
        "main.py": '''\
"""CLI entry point for training, evaluation, and deployment."""
import argparse
import sys
from pathlib import Path

def train(args):
    print(f" Training {args.base_model} (frozen={args.freeze_base}, epochs={args.epochs})")
    # TODO: Implement training pipeline
    # model = build_model(args.base_model, args.input_shape, args.freeze_base)
    # train_loop(model, args.epochs, args.batch_size, args.learning_rate)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).touch()
    print(f"✅ Saved checkpoint: {args.output_path}")

def evaluate(args):
    print(f"📊 Evaluating model: {args.model_path}")
    # TODO: Load model & run test set evaluation
    print("✅ Evaluation complete. Metrics logged.")

def predict(args):
    print(f"🔍 Predicting: {args.image_path}")
    # TODO: Load model, preprocess image, run inference
    print("✅ Prediction: Healthy (0.92)")

def convert(args):
    print(f"🔄 Converting Keras → TFLite: {args.keras_model}")
    # TODO: tf.lite.TFLiteConverter.from_keras_model()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).touch()
    print(f"✅ Saved TFLite model: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tomato Disease CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--base-model", default="mobilenetv3small")
    p_train.add_argument("--input-shape", nargs=3, type=int, default=[224, 224, 3])
    p_train.add_argument("--freeze-base", action="store_true")
    p_train.add_argument("--unfreeze-base", action="store_true")
    p_train.add_argument("--learning-rate", type=float, default=0.001)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--output-path", default="models/final/best_model.keras")
    p_train.add_argument("--load-weights", default=None)
    p_train.set_defaults(func=train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--test-dir", required=True)
    p_eval.set_defaults(func=evaluate)

    # Predict
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--image-path", required=True)
    p_pred.set_defaults(func=predict)

    # Convert
    p_conv = subparsers.add_parser("convert")
    p_conv.add_argument("--keras-model", required=True)
    p_conv.add_argument("--output-path", required=True)
    p_conv.add_argument("--optimizations", default="default")
    p_conv.set_defaults(func=convert)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
''',
        "src/data/dataset.py": '''\
"""Dataset loading, splitting, and preprocessing."""
import argparse
from pathlib import Path
import shutil
import random

def split_dataset(raw_dir: str, split_dir: str, ratios: tuple = (0.8, 0.1, 0.1)):
    """Split images into train/val/test folders."""
    raw = Path(raw_dir)
    splits = Path(split_dir)
    splits.mkdir(parents=True, exist_ok=True)
    
    for class_dir in raw.iterdir():
        if not class_dir.is_dir(): continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        
        for folder, imgs in [
            ("train", images[:train_end]),
            ("val", images[train_end:val_end]),
            ("test", images[val_end:])
        ]:
            target = splits / folder / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, target / img.name)
    print(f"✅ Split complete. Output: {split_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Run dataset splitting")
    args = parser.parse_args()
    if args.split:
        split_dataset("data/raw/tomato", "data/splits")
''',
        "src/training/trainer.py": '''\
"""Model training pipeline (feature extraction + fine-tuning)."""
def build_model(base_name: str, input_shape: tuple, freeze_base: bool = True):
    """Construct transfer learning model."""
    # TODO: Load base model, add classification head
    pass

def train_step(model, dataloader, epochs: int, lr: float):
    """Run training loop."""
    # TODO: Implement training logic
    pass
''',
        "src/inference/predictor.py": '''\
"""Model inference for Keras and TFLite."""
from pathlib import Path
from PIL import Image
import numpy as np

def load_model(path: str):
    """Load Keras or TFLite model."""
    # TODO: Implement loader based on extension
    pass

def predict(model, image_path: str, class_names: list):
    """Run inference and return predictions."""
    # TODO: Preprocess, run model, postprocess
    pass
''',
        "src/utils/helpers.py": '''\
"""Utility functions for logging, config loading, and image processing."""
import logging

def setup_logger(name: str, log_file: str = "logs/app.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)
''',
        "tests/test_smoke.py": '''\
"""Basic smoke tests for project setup."""
def test_imports():
    import config
    import src.data.dataset
    import src.inference.predictor
    assert config.DISEASE_CLASSES is not None

def test_cli_structure():
    from main import main
    # Verify argparse structure exists
    assert callable(main)
''',
        "requirements-frontend.txt": """\
streamlit>=1.30.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
""",
        "requirements-backend.txt": """\
tensorflow>=2.15.0
keras>=2.15.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
""",
        "requirements-dev.txt": """\
-r requirements-frontend.txt
-r requirements-backend.txt
pytest>=8.0.0
pytest-cov>=4.1.0
ruff>=0.3.0
""",
        ".gitignore": """\
# Environment
venv/
.env
*.log

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Models & Data (keep structure, ignore large files)
models/final/*.keras
models/tflite/*.tflite
data/raw/
!data/raw/.gitkeep
data/splits/

# IDE & OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Streamlit
.streamlit/
""",
    }

    for rel_path, content in files.items():
        write_file(base, rel_path, content)

    print("\n✨ Project structure initialized successfully!")
    print("👉 Next steps:")
    print("   1. Download dataset: make data")
    print("   2. Install dependencies: make install")
    print("   3. Run frontend: make frontend")
    print("   4. Train model: make train")


if __name__ == "__main__":
    main()