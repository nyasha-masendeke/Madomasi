import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE, OOD_ENTROPY_THRESHOLD
from src.preprocessing.grabcut import grabcut_bg_remove


def load_class_names(model_path, fallback_len: int | None = None) -> list[str]:
    """Resolve canonical class names for a saved model.

    Looks for `class_names.json` next to the model file. Falls back to the
    global DISEASE_CLASSES only when its length matches the model's output
    dimension; otherwise raises so a 3-class model isn't silently labelled
    with the 10-class names.
    """
    p = Path(model_path)
    cn_path = p.parent / "class_names.json"
    if cn_path.exists():
        return list(json.loads(cn_path.read_text()))
    if fallback_len is None or fallback_len == len(DISEASE_CLASSES):
        return list(DISEASE_CLASSES)
    raise FileNotFoundError(
        f"Model at {p} has {fallback_len} outputs but global DISEASE_CLASSES "
        f"has {len(DISEASE_CLASSES)}, and no class_names.json was found at "
        f"{cn_path}. Write class_names.json next to the model, or pass "
        "class_names= explicitly."
    )


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
    model_or_path,
    image_data,
    confidence_threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> dict:
    """Run inference on a single image with GrabCut background removal.

    Args:
        model_or_path:        Loaded tf.keras.Model or path string to a .keras file.
        image_data:           File-like object or path readable by PIL.
        confidence_threshold: Minimum probability to mark a prediction as passing.
        class_names:          Optional list of canonical class names matching
                              the model's output order. If omitted, looked up
                              via `load_class_names()` next to the model path.

    Returns dict with keys:
        disease, class_idx, confidence, passes_threshold,
        all_probs, is_leaf, entropy, segmented_png
    """
    is_path = not isinstance(model_or_path, tf.keras.Model)
    model = tf.keras.models.load_model(model_or_path) if is_path else model_or_path

    img = Image.open(image_data).resize(IMAGE_SIZE, Image.BILINEAR).convert("RGB")
    img_arr, _ = grabcut_bg_remove(np.array(img, dtype=np.uint8))

    seg_buf = io.BytesIO()
    Image.fromarray(img_arr).save(seg_buf, format="PNG")
    segmented_png = seg_buf.getvalue()

    arr = np.expand_dims(img_arr.astype(np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]

    if class_names is None:
        class_names = (
            load_class_names(model_or_path, fallback_len=len(preds))
            if is_path else list(DISEASE_CLASSES)
        )

    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    entropy = float(-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))))
    max_entropy = float(np.log(len(preds)))
    norm_entropy = round(entropy / max_entropy, 3)
    is_leaf = norm_entropy < OOD_ENTROPY_THRESHOLD

    def class_name(i: int) -> str:
        return class_names[i] if i < len(class_names) else f"Class {i}"

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
