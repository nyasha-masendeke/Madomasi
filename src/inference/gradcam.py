import io

import numpy as np
from PIL import Image
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE


def compute_gradcam(
    model_or_path,
    img_bytes: bytes,
    class_idx: int | None = None,
) -> dict:
    """Grad-CAM: gradient-weighted class activation heatmap.

    Builds a preproc_backbone from the outer model's input (which includes the
    inline preprocess_input call) to the first sub-model's output, so gradients
    are computed on correctly-preprocessed activations.

    Returns dict with keys:
        overlay_png, original_png, class_name, class_idx, confidence, layer_name
    """
    import cv2

    model = (
        model_or_path if isinstance(model_or_path, tf.keras.Model)
        else tf.keras.models.load_model(model_or_path)
    )

    sub_models = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if len(sub_models) < 2:
        raise ValueError(
            "Grad-CAM requires a model with a separate head sub-model. "
            "Use train_head() (Stage 2) — models from train_model() or "
            "train_two_stage() have inline heads and are not supported."
        )
    # sub_models[0] = MobileNetV3Small base, sub_models[1] = head
    # Build from outer model.input so preprocessing is included in the forward pass.
    preproc_backbone = tf.keras.Model(
        inputs=model.input,
        outputs=sub_models[0].output,
        name="preproc_backbone",
    )
    head = sub_models[1]

    original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized  = original_img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = preproc_backbone(arr, training=False)   # arr is raw [0,255] — preprocessing inside
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
        "layer_name":   sub_models[0].name,
    }
