import tensorflow as tf
from pathlib import Path


def convert_model(
    keras_model_path: str,
    output_path: str,
    representative_data_dir: str = "data/splits/val",
) -> str:
    """Convert a Keras model to INT8-quantised TFLite for Raspberry Pi deployment.

    If representative_data_dir exists, feeds 200 validation images to calibrate
    quantisation ranges. Does NOT set inference_input_type / inference_output_type
    — those flags force INT8 I/O which breaks the float32 inference pipeline.
    """
    from config import IMAGE_SIZE

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    val_path = Path(representative_data_dir)
    if val_path.exists():
        ds = tf.keras.utils.image_dataset_from_directory(
            str(val_path), image_size=IMAGE_SIZE, batch_size=1,
            label_mode=None, shuffle=True, seed=42,
        ).take(200)

        def _rep():
            for images in ds:
                yield [tf.cast(images, tf.float32)]

        converter.representative_dataset = _rep

    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(tflite_model)
    return output_path
