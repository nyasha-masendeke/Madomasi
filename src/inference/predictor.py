"""Model inference for Keras and TFLite."""
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf


def load_model(path: str):
    path = Path(path)
    if path.suffix in (".h5", ".keras"):
        return tf.keras.models.load_model(path)
    elif path.suffix == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        return interpreter
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}. Use .h5, .keras, or .tflite")


def predict(model, image_path: str, class_names: list):
    """Run inference and return top-5 predictions."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    if isinstance(model, tf.lite.Interpreter):
        # TFLite: check whether the model expects [-1,1] or [0,255] and preprocess accordingly
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        # MobileNetV3 TFLite expects the same preprocess_input range as Keras
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
        model.set_tensor(input_details[0]["index"], arr)
        model.invoke()
        predictions = model.get_tensor(output_details[0]["index"])
    else:
        # Keras model — use MobileNetV3 preprocessing (maps [0,255] -> [-1,1])
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
        predictions = model.predict(arr, verbose=0)

    top_indices = np.argsort(predictions[0])[::-1][:5]
    return [
        {
            "class": class_names[i] if i < len(class_names) else f"Class_{i}",
            "confidence": float(predictions[0][i]),
        }
        for i in top_indices
    ]
