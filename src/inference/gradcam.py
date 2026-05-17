import io

import numpy as np
from PIL import Image
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE


def compute_gradcam(
    model_path: str,
    img_bytes: bytes,
    class_idx: int | None = None,
) -> dict:
    """Grad-CAM: gradient-weighted class activation heatmap.

    Splits the model into backbone + head sub-models, watches the backbone
    feature map with GradientTape, and overlays the resulting heatmap on the
    original image.

    Returns dict with keys:
        overlay_png, original_png, class_name, class_idx, confidence, layer_name
    """
    import cv2

    model = tf.keras.models.load_model(model_path)

    sub_models = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if len(sub_models) < 2:
        raise ValueError("Expected backbone + head sub-models; got unexpected architecture.")
    backbone, head = sub_models[0], sub_models[1]

    original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized  = original_img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = backbone(arr, training=False)
        tape.watch(conv_outputs)
        predictions  = head(conv_outputs, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads   = tape.gradient(loss, conv_outputs)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.einsum("hwc,c->hw", conv_outputs[0], pooled)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    h, w = IMAGE_SIZE
    heatmap_up  = cv2.resize(heatmap, (w, h))
    colored     = cv2.applyColorMap(np.uint8(255 * heatmap_up), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    orig_arr = np.array(img_resized, dtype=np.float32)
    overlay  = np.uint8(0.45 * colored_rgb.astype(np.float32) + 0.55 * orig_arr)

    def _to_png(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    cname = DISEASE_CLASSES[class_idx] if class_idx < len(DISEASE_CLASSES) else f"Class {class_idx}"
    return {
        "overlay_png":  _to_png(overlay),
        "original_png": _to_png(np.array(img_resized)),
        "class_name":   cname,
        "class_idx":    int(class_idx),
        "confidence":   float(predictions[0, class_idx]),
        "layer_name":   backbone.name,
    }
