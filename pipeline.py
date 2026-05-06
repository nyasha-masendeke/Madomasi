import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)


def load_datasets(data_dir: str, batch_size: int, val_split: float = 0.2):
    """Load train/val splits from a directory of class-named subdirectories."""
    kwargs = dict(
        validation_split=val_split,
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="training", **kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **kwargs
    )
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)


def build_model(num_classes: int, freeze_base: bool = True) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base(x, training=not freeze_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def train_model(
    base_model="mobilenetv3small",
    epochs=10,
    batch_size=16,
    lr=0.001,
    freeze_base=True,
    data_dir="data/train",
    output_path="models/best.keras",
    callbacks=None,
):
    train_ds, val_ds = load_datasets(data_dir, batch_size)
    num_classes = len(train_ds.class_names)

    model = build_model(num_classes, freeze_base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks or [],
    )
    model.save(output_path)
    return model


def predict_image(model_path: str, image_data, confidence_threshold: float = 0.5):
    model = tf.keras.models.load_model(model_path)
    num_classes = model.output_shape[-1]

    img = Image.open(image_data).resize(IMAGE_SIZE).convert("RGB")
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    from config import DISEASE_CLASSES

    def class_name(i):
        return DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else f"Class {i}"

    return {
        "disease": class_name(class_idx),
        "confidence": confidence,
        "passes_threshold": confidence >= confidence_threshold,
        "all_probs": {class_name(i): float(p) for i, p in enumerate(preds)},
    }


def evaluate_model(model_path: str, test_dir: str, batch_size: int = 16):
    model = tf.keras.models.load_model(model_path)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)

    loss, accuracy = model.evaluate(test_ds)
    return {"loss": loss, "accuracy": accuracy}


def convert_model(keras_model_path: str, output_path: str):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(tflite_model)
    return output_path
