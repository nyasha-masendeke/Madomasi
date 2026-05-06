import tensorflow as tf
from pathlib import Path
from datetime import datetime
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
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    # Re-attach class_names lost by prefetch
    train_ds.class_names = class_names
    val_ds.class_names = class_names
    return train_ds, val_ds


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
    data_dir="data/splits/train",
    output_path="models/trained/latest.keras",
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

    # Timestamped save path so runs never overwrite each other
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"run_{ts}_single.keras"

    default_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(save_path), monitor="val_accuracy", save_best_only=True,
            mode="max", verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
        ),
    ]

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=(callbacks or []) + default_callbacks,
        verbose=0,
    )
    return model, str(save_path)


def train_two_stage(
    batch_size=16,
    fe_epochs=10,
    fe_lr=0.001,
    ft_epochs=10,
    ft_lr=0.0001,
    data_dir="data/splits/train",
    output_path="models/trained/latest.keras",
    fe_callbacks=None,
    ft_callbacks=None,
):
    """
    Two-stage transfer learning pipeline.

    Stage 1 — Feature Extraction:
      Base model frozen. Only the classification head is trained at a
      higher learning rate. Fast convergence, establishes a good baseline.

    Stage 2 — Fine-Tuning:
      Base model unfrozen. The entire network is trained end-to-end at a
      much lower learning rate to adapt ImageNet features to tomato leaves.

    Returns a dict with keys:
      stage1_path  — best Stage 1 checkpoint (feature extraction)
      stage2_path  — best Stage 2 checkpoint (fine-tuned, use this for inference)
      model        — the final in-memory Keras model
    """
    train_ds, val_ds = load_datasets(data_dir, batch_size)
    num_classes = len(train_ds.class_names)

    # One timestamp per run — both stages share it so they group together on disk
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    fe_path = save_dir / f"run_{ts}_stage1_fe.keras"
    ft_path = save_dir / f"run_{ts}_stage2_ft.keras"

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
    )

    # ── Stage 1: Feature Extraction ──────────────────────────────────────
    model = build_model(num_classes, freeze_base=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fe_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=fe_epochs,
        validation_data=val_ds,
        callbacks=(fe_callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(fe_path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=0
            ),
            early_stop,
        ],
        verbose=0,
    )

    # ── Stage 2: Fine-Tuning ─────────────────────────────────────────────
    # Reload the best Stage 1 weights before unfreezing
    model = tf.keras.models.load_model(str(fe_path))
    model.layers[1].trainable = True   # base model is layer index 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ft_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=ft_epochs,
        validation_data=val_ds,
        callbacks=(ft_callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(ft_path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=0
            ),
            early_stop,
        ],
        verbose=0,
    )

    return {
        "stage1_path": str(fe_path),
        "stage2_path": str(ft_path),
        "model": model,
    }


def predict_image(model_path: str, image_data, confidence_threshold: float = 0.5):
    model = tf.keras.models.load_model(model_path)

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

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return {"loss": loss, "accuracy": accuracy}


def convert_model(keras_model_path: str, output_path: str):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(tflite_model)
    return output_path
