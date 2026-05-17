import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE, OOD_ENTROPY_THRESHOLD
from src.preprocessing.grabcut import grabcut_bg_remove


def log_inference(
    disease: str,
    confidence: float,
    model_name: str,
    entropy: float | None = None,
    is_leaf: bool | None = None,
) -> None:
    """Append one prediction to outputs/inference_log.jsonl for drift monitoring."""
    log_path = Path("outputs/inference_log.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    entry: dict = {
        "ts":         datetime.now().isoformat(),
        "disease":    disease,
        "confidence": round(float(confidence), 4),
        "model":      model_name,
    }
    if entropy is not None:
        entry["entropy"] = round(float(entropy), 4)
    if is_leaf is not None:
        entry["is_leaf"] = bool(is_leaf)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def predict_image(
    model_path: str,
    image_data,
    confidence_threshold: float = 0.5,
) -> dict:
    """Run inference on a single image with GrabCut background removal.

    Args:
        model_path:           Path to a .keras model file.
        image_data:           File-like object or path readable by PIL.
        confidence_threshold: Minimum probability to mark a prediction as passing.

    Returns dict with keys:
        disease, class_idx, confidence, passes_threshold,
        all_probs, is_leaf, entropy, segmented_png
    """
    model = tf.keras.models.load_model(model_path)

    img = Image.open(image_data).resize(IMAGE_SIZE, Image.BILINEAR).convert("RGB")
    img_arr, _ = grabcut_bg_remove(np.array(img, dtype=np.uint8))

    seg_buf = io.BytesIO()
    Image.fromarray(img_arr).save(seg_buf, format="PNG")
    segmented_png = seg_buf.getvalue()

    arr = np.expand_dims(img_arr.astype(np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    entropy = float(-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))))
    max_entropy = float(np.log(len(preds)))
    norm_entropy = round(entropy / max_entropy, 3)
    is_leaf = norm_entropy < OOD_ENTROPY_THRESHOLD

    def class_name(i: int) -> str:
        return DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else f"Class {i}"

    return {
        "disease":           class_name(class_idx),
        "class_idx":         class_idx,
        "confidence":        confidence,
        "passes_threshold":  confidence >= confidence_threshold,
        "all_probs":         {class_name(i): float(p) for i, p in enumerate(preds)},
        "is_leaf":           is_leaf,
        "entropy":           norm_entropy,
        "segmented_png":     segmented_png,
    }
