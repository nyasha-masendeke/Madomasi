import time
from typing import Dict, List, Tuple
from PIL import Image

def run_inference(image: Image.Image) -> Dict[str, float]:
    """Simulate model inference. Replace with actual model later."""
    time.sleep(1.5)  # Simulate processing delay
    return {
        "Early Blight": 0.85,
        "Late Blight": 0.10,
        "Bacterial Spot": 0.03,
        "Healthy": 0.02
    }

def filter_predictions(
    predictions: Dict[str, float],
    confidence_threshold: float,
    selected_classes: List[str]
) -> List[Tuple[str, float]]:
    """Filter and sort predictions."""
    filtered = {
        k: v for k, v in predictions.items()
        if v * 100 >= confidence_threshold and k in selected_classes
    }
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)
