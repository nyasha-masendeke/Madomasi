import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE, EARLY_STOPPING_PATIENCE, DROPOUT_RATE
from src.training.data import load_datasets, build_model, next_training_output_dir


# ---------------------------------------------------------------------------
# Feature extraction (Stage 1)
# ---------------------------------------------------------------------------

def extract_features(
    data_dir: str,
    output_dir: str = "data/features",
    batch_size: int = 32,
    progress_fn=None,
) -> dict:
    """Run every image through the frozen MobileNetV3Small base and cache features.

    Nothing is trained here — the base model is used purely as a fixed extractor,
    following the Keras transfer-learning guide (stage 1 of 3).

    Returns:
        features_dir, num_samples, num_classes, feature_shape, class_names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    ds_raw      = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=False,
    )
    class_names = ds_raw.class_names
    ds          = ds_raw.prefetch(tf.data.AUTOTUNE)

    cardinality  = tf.data.experimental.cardinality(ds).numpy()
    total_batches = int(cardinality) if cardinality >= 0 else None

    all_features, all_labels = [], []
    for i, (images, labels) in enumerate(ds):
        x     = tf.keras.applications.mobilenet_v3.preprocess_input(images)
        feats = base(x, training=False)
        all_features.append(feats.numpy())
        all_labels.append(labels.numpy())
        if progress_fn:
            progress_fn(i + 1, total_batches or (i + 1))

    features_arr = np.concatenate(all_features, axis=0)
    labels_arr   = np.concatenate(all_labels,   axis=0)

    np.save(output_dir / "features.npy", features_arr)
    np.save(output_dir / "labels.npy",   labels_arr)

    meta = {
        "class_names":   class_names,
        "feature_shape": list(features_arr.shape[1:]),
        "num_samples":   int(len(labels_arr)),
        "num_classes":   len(class_names),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta))

    return {
        "features_dir":  str(output_dir),
        "num_samples":   int(len(labels_arr)),
        "num_classes":   len(class_names),
        "feature_shape": tuple(features_arr.shape[1:]),
        "class_names":   class_names,
    }


# ---------------------------------------------------------------------------
# Head training (Stage 2)
# ---------------------------------------------------------------------------

def train_head(
    features_dir: str = "data/features",
    epochs: int = 10,
    lr: float = 0.001,
    val_split: float = 0.2,
    output_path: str = "models/trained/latest.keras",
    callbacks=None,
    class_weights: dict | None = None,
) -> dict:
    """Train a classification head on top of the cached feature maps.

    The base model is not involved — only head weights are updated.
    After training, assembles and saves a full inference model (image → base → head).

    Returns:
        head_path, full_model_path, run_dir, history
    """
    features_dir = Path(features_dir)
    features = np.load(features_dir / "features.npy")
    labels   = np.load(features_dir / "labels.npy")
    meta     = json.loads((features_dir / "meta.json").read_text())

    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(features))
    features = features[perm]
    labels   = labels[perm]

    num_classes   = meta["num_classes"]
    feature_shape = tuple(meta["feature_shape"])

    inputs  = tf.keras.Input(shape=feature_shape)
    x       = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    head    = tf.keras.Model(inputs, outputs)

    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "head.keras"

    keras_hist = head.fit(
        features, labels,
        epochs=epochs,
        validation_split=val_split,
        shuffle=True,
        class_weight=class_weights,
        callbacks=(callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(head_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=0,
            ),
        ],
        verbose=0,
    )

    history_data = _pack_history(keras_hist)
    cb_dir  = _callback_output_dir(callbacks)
    out_dir = cb_dir if cb_dir else next_training_output_dir()
    (out_dir / "history_head.json").write_text(json.dumps(history_data))
    (run_dir  / "history_head.json").write_text(json.dumps(history_data))

    # Assemble full model: image → preprocess → frozen base → head
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    best_head = tf.keras.models.load_model(str(head_path))

    img_in = tf.keras.Input((*IMAGE_SIZE, 3))
    x      = tf.keras.applications.mobilenet_v3.preprocess_input(img_in)
    x      = base(x, training=False)
    x      = best_head(x)
    full   = tf.keras.Model(img_in, x)

    full_path = run_dir / "head_full.keras"
    full.save(str(full_path))

    class_names = list(meta.get("class_names") or [])
    if class_names:
        (run_dir / "class_names.json").write_text(json.dumps(class_names))

    return {
        "head_path":       str(head_path),
        "full_model_path": str(full_path),
        "run_dir":         str(run_dir),
        "history":         history_data,
    }


# ---------------------------------------------------------------------------
# Fine-tuning (Stage 3)
# ---------------------------------------------------------------------------

def fine_tune_model(
    model_path: str,
    data_dir: str = "data/splits/train",
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 0.00001,
    unfreeze_layers: int = 20,
    output_path: str = "models/trained/latest.keras",
    callbacks=None,
    class_weights: dict | None = None,
) -> dict:
    """Fine-tune the full model end-to-end at a very low learning rate.

    Unfreezes the last `unfreeze_layers` layers of the MobileNetV3Small base so
    ImageNet features adapt to tomato-leaf patterns without catastrophic forgetting.
    Uses the held-out val/ split when available, falling back to an 80/20 split.

    Returns:
        model, path, history
    """
    if unfreeze_layers <= 0:
        raise ValueError("unfreeze_layers must be a positive integer")

    model = tf.keras.models.load_model(model_path)
    base_layer = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base_layer is None:
        raise ValueError("No sub-model found in model.layers — cannot unfreeze base.")
    for layer in base_layer.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_layer.layers[-unfreeze_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    val_dir = Path(data_dir).parent / "val"
    if val_dir.exists():
        raw_train = tf.keras.utils.image_dataset_from_directory(
            data_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
            label_mode="int", shuffle=True, seed=42,
        )
        class_names = raw_train.class_names   # save before prefetch — PrefetchDataset drops this attr
        train_ds = raw_train.prefetch(tf.data.AUTOTUNE)
        train_ds.class_names = class_names

        raw_val = tf.keras.utils.image_dataset_from_directory(
            str(val_dir), image_size=IMAGE_SIZE, batch_size=batch_size,
            label_mode="int", shuffle=False,
        )
        if list(raw_val.class_names) != list(class_names):
            train_ds, val_ds = load_datasets(data_dir, batch_size)
        else:
            val_ds = raw_val.prefetch(tf.data.AUTOTUNE)
            val_ds.class_names = class_names
    else:
        train_ds, val_ds = load_datasets(data_dir, batch_size)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomBrightness(0.15),
    ], name="augmentation")
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    run_dir = Path(model_path).parent
    ft_path = run_dir / "fine_tuned.keras"

    ft_class_names = list(getattr(train_ds, "class_names", []) or [])
    if ft_class_names:
        (run_dir / "class_names.json").write_text(json.dumps(ft_class_names))

    keras_hist = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=(callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(ft_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=0,
            ),
        ],
        verbose=0,
    )

    history_data = _pack_history(keras_hist)
    cb_dir  = _callback_output_dir(callbacks)
    out_dir = cb_dir if cb_dir else next_training_output_dir()
    (out_dir / "history_finetune.json").write_text(json.dumps(history_data))
    (run_dir  / "history_finetune.json").write_text(json.dumps(history_data))

    return {"model": model, "path": str(ft_path), "history": history_data}


# ---------------------------------------------------------------------------
# Legacy single/two-stage entry points (kept for CLI compatibility)
# ---------------------------------------------------------------------------

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
    """Single-stage training (head only or full). Kept for CLI / notebook use."""
    train_ds, val_ds = load_datasets(data_dir, batch_size)
    num_classes = len(train_ds.class_names)
    model = build_model(num_classes, freeze_base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "single_stage.keras"

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=(callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(save_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=0,
            ),
        ],
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
    class_weights: dict | None = None,
):
    """Two-stage transfer learning: feature extraction then fine-tuning."""
    train_ds, val_ds = load_datasets(data_dir, batch_size)
    num_classes = len(train_ds.class_names)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fe_path = run_dir / "stage1_fe.keras"
    ft_path = run_dir / "stage2_ft.keras"

    early_stop_fe = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=0,
    )
    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=0,
    )

    model = build_model(num_classes, freeze_base=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fe_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=fe_epochs, validation_data=val_ds,
        class_weight=class_weights,
        callbacks=(fe_callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(fe_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            early_stop_fe,
        ],
        verbose=0,
    )

    model = tf.keras.models.load_model(str(fe_path))
    base_layer = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
    if base_layer is None:
        raise ValueError("No sub-model found in model.layers — cannot unfreeze base.")
    base_layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ft_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=ft_epochs, validation_data=val_ds,
        class_weight=class_weights,
        callbacks=(ft_callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(ft_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            early_stop_ft,
        ],
        verbose=0,
    )

    return {
        "stage1_path": str(fe_path),
        "stage2_path": str(ft_path),
        "model":       model,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pack_history(keras_hist) -> dict:
    return {
        "epoch":        list(range(1, len(keras_hist.history["accuracy"]) + 1)),
        "accuracy":     [float(v) for v in keras_hist.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in keras_hist.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in keras_hist.history.get("loss", [])],
        "val_loss":     [float(v) for v in keras_hist.history.get("val_loss", [])],
    }


def _callback_output_dir(callbacks) -> Path | None:
    for cb in (callbacks or []):
        d = getattr(cb, "output_dir", None)
        if d:
            return d
    return None
