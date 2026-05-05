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
