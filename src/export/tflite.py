import tensorflow as tf
from pathlib import Path


def convert_model(keras_model_path: str, output_path: str) -> str:
    """Convert a Keras model to INT8-quantised TFLite for Raspberry Pi deployment."""
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(tflite_model)
    return output_path
