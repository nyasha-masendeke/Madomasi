import json
import random
import shutil
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from config import DISEASE_CLASSES

IMAGE_SIZE = (224, 224)


def _next_training_output_dir() -> Path:
    """Return the next unused outputs/TrainingN directory and create it.

    outputs/Training1  →  first training run
    outputs/Training2  →  second training run
    ...
    """
    base = Path("outputs")
    base.mkdir(exist_ok=True)
    existing = [
        int(p.name[8:])
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith("Training") and p.name[8:].isdigit()
    ]
    n = max(existing, default=0) + 1
    d = base / f"Training{n}"
    d.mkdir(exist_ok=True)
    return d


def log_inference(disease: str, confidence: float, model_name: str) -> None:
    """Append one prediction to outputs/inference_log.jsonl for drift monitoring."""
    log_path = Path("outputs/inference_log.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts":         datetime.now().isoformat(),
            "disease":    disease,
            "confidence": round(float(confidence), 4),
            "model":      model_name,
        }) + "\n")


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

    # One folder per run so single-stage runs match the two-stage layout
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "single_stage.keras"

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

    # One folder per run — stages are grouped inside it
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fe_path = run_dir / "stage1_fe.keras"
    ft_path = run_dir / "stage2_ft.keras"

    # Each stage gets its own EarlyStopping instance — they must not share state
    early_stop_fe = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
    )
    early_stop_ft = tf.keras.callbacks.EarlyStopping(
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
            early_stop_fe,
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
            early_stop_ft,
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

    img = Image.open(image_data).resize(IMAGE_SIZE, Image.BILINEAR).convert("RGB")
    # Raw [0, 255] float32 — preprocess_input is baked into the model graph (build_model)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    # Entropy-based OOD guard: if the probability mass is spread too evenly
    # across all classes the input is likely not a tomato leaf.
    # Note: this catches *uncertain* predictions only — a model that is
    # confidently wrong (e.g. a face predicted at 98%) will still pass.
    entropy = float(-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))))
    max_entropy = float(np.log(len(preds)))          # ln(num_classes)
    norm_entropy = round(entropy / max_entropy, 3)    # 0 = certain, 1 = uniform
    is_leaf = norm_entropy < 0.75

    def class_name(i):
        return DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else f"Class {i}"

    return {
        "disease": class_name(class_idx),
        "confidence": confidence,
        "passes_threshold": confidence >= confidence_threshold,
        "all_probs": {class_name(i): float(p) for i, p in enumerate(preds)},
        "is_leaf": is_leaf,
        "entropy": norm_entropy,
    }


def evaluate_model(model_path: str, test_dir: str, batch_size: int = 16):
    model = tf.keras.models.load_model(model_path)
    # Compile is needed when the model was assembled (e.g. head_full) but not
    # compiled before saving — compile here with the same settings used for fine-tuning.
    # Always compile — harmless if already compiled, required for models
    # saved without an optimizer (e.g. head_full assembled in train_head).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
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


# =============================================================================
# DATASET SPLITTING
# =============================================================================

def split_dataset(
    source_dir: str,
    output_dir: str = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    progress_fn=None,
) -> dict:
    """
    Copy images from a flat class-named directory structure into train/val/test splits.

    source_dir layout expected:
        source_dir/
            ClassName_A/img1.jpg ...
            ClassName_B/img1.jpg ...

    Output:
        output_dir/train/ClassName_A/ ...
        output_dir/val/ClassName_A/   ...
        output_dir/test/ClassName_A/  ...

    Files are copied — the source dataset is never modified.

    Returns a dict with:
      train_dir, val_dir, test_dir  — absolute paths to each split
      counts                        — {split: total_images}
      num_classes                   — number of class folders found
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    source  = Path(source_dir)
    output  = Path(output_dir)
    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits  = ["train", "val", "test"]
    counts  = {s: 0 for s in splits}

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    random.seed(seed)

    for idx, class_dir in enumerate(class_dirs):
        images = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in exts])
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        buckets = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, imgs in buckets.items():
            dest = output / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)
            counts[split] += len(imgs)

        if progress_fn:
            progress_fn(idx + 1, len(class_dirs), class_dir.name)

    return {
        "output_dir": str(output),
        "train_dir":  str(output / "train"),
        "val_dir":    str(output / "val"),
        "test_dir":   str(output / "test"),
        "counts":     counts,
        "num_classes": len(class_dirs),
    }


# =============================================================================
# THREE-STAGE TRANSFER LEARNING  (mirrors Keras transfer learning guide)
# https://keras.io/guides/transfer_learning/
# =============================================================================

def extract_features(
    data_dir: str,
    output_dir: str = "data/features",
    batch_size: int = 32,
    progress_fn=None,
) -> dict:
    """
    Stage 1 — Feature Extraction.

    Run every image through the frozen MobileNetV3Small base and cache the
    resulting feature maps to disk as NumPy arrays.  Nothing is trained here;
    the base model is used purely as a fixed feature extractor, exactly as
    described in the Keras transfer-learning guide.

    Returns a dict with:
      features_dir  — directory containing features.npy, labels.npy, meta.json
      num_samples   — total images processed
      num_classes   — number of leaf-disease classes
      feature_shape — spatial shape of one feature map, e.g. (7, 7, 576)
      class_names   — ordered list of class folder names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=IMAGE_SIZE, batch_size=batch_size,
        label_mode="int", shuffle=False,
    )
    class_names = ds_raw.class_names          # capture before prefetch strips it
    ds = ds_raw.prefetch(tf.data.AUTOTUNE)

    cardinality = tf.data.experimental.cardinality(ds).numpy()
    total_batches = int(cardinality) if cardinality >= 0 else None

    all_features, all_labels = [], []
    for i, (images, labels) in enumerate(ds):
        x = tf.keras.applications.mobilenet_v3.preprocess_input(images)
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


def train_head(
    features_dir: str = "data/features",
    epochs: int = 10,
    lr: float = 0.001,
    val_split: float = 0.2,
    output_path: str = "models/trained/latest.keras",
    callbacks=None,
) -> dict:
    """
    Stage 2 — Train the Classification Head.

    Load the cached feature maps from extract_features() and train a small
    head (GlobalAveragePooling2D → Dropout → Dense/softmax) on top of them.
    The base model is not involved at all — only the head weights are updated.

    After training, a full inference model (image input → preprocess → frozen
    base → trained head) is assembled and saved so Stage 3 can load it for
    fine-tuning.

    Returns a dict with:
      head_path       — best checkpoint for the head-only model
      full_model_path — combined base + head model ready for fine-tuning
      run_dir         — timestamped folder containing both checkpoints
    """
    features_dir = Path(features_dir)
    features = np.load(features_dir / "features.npy")
    labels   = np.load(features_dir / "labels.npy")
    meta     = json.loads((features_dir / "meta.json").read_text())

    # Shuffle before validation split: features are extracted in alphabetical
    # class order, so without shuffling Keras's validation_split would take only
    # the last class(es), causing 0% validation accuracy on unseen classes.
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(features))
    features = features[perm]
    labels   = labels[perm]

    num_classes   = meta["num_classes"]
    feature_shape = tuple(meta["feature_shape"])

    # Head: takes raw base spatial output → pool → classify
    inputs  = tf.keras.Input(shape=feature_shape)
    x       = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    head    = tf.keras.Model(inputs, outputs)

    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_path).parent / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "head.keras"

    keras_hist = head.fit(
        features, labels,
        epochs=epochs,
        validation_split=val_split,
        shuffle=True,
        callbacks=(callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(head_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3,
                restore_best_weights=True, verbose=0,
            ),
        ],
        verbose=0,
    )

    # Persist history so the UI can show learning curves even after CLI runs.
    # Saved to outputs/TrainingN/ so the dashboard can discover it by scanning
    # that folder in sequential order.
    history_data = {
        "epoch":        list(range(1, len(keras_hist.history["accuracy"]) + 1)),
        "accuracy":     [float(v) for v in keras_hist.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in keras_hist.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in keras_hist.history.get("loss", [])],
        "val_loss":     [float(v) for v in keras_hist.history.get("val_loss", [])],
    }
    # Reuse the callback's TrainingN dir if it was created during training
    # (UI runs), otherwise create a fresh one (CLI runs).
    cb_dir = next(
        (getattr(cb, "output_dir", None) for cb in (callbacks or []) if getattr(cb, "output_dir", None)),
        None,
    )
    out_dir = cb_dir if cb_dir else _next_training_output_dir()
    (out_dir / "history_head.json").write_text(json.dumps(history_data))
    # Also keep a copy alongside the model checkpoint for reference
    (run_dir / "history_head.json").write_text(json.dumps(history_data))

    # Assemble full model for fine-tuning: image → preprocess → frozen base → head
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

    return {
        "head_path":       str(head_path),
        "full_model_path": str(full_path),
        "run_dir":         str(run_dir),
        "history":         history_data,
    }


def fine_tune_model(
    model_path: str,
    data_dir: str = "data/splits/train",
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 0.00001,
    output_path: str = "models/trained/latest.keras",
    callbacks=None,
) -> dict:
    """
    Stage 3 — Fine-Tuning.

    Load the full model produced by train_head(), unfreeze the base model,
    and train the entire network end-to-end at a very low learning rate so
    the ImageNet features adapt to tomato-leaf patterns without catastrophic
    forgetting — exactly as recommended in the Keras transfer-learning guide.

    Returns a dict with:
      model — final in-memory Keras model
      path  — best fine-tuned checkpoint
    """
    model = tf.keras.models.load_model(model_path)

    # Layer index 1 is the MobileNetV3Small base (same layout as build_model)
    model.layers[1].trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, val_ds = load_datasets(data_dir, batch_size)

    run_dir = Path(model_path).parent   # keep checkpoints in the same run folder
    ft_path = run_dir / "fine_tuned.keras"

    keras_hist = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=(callbacks or []) + [
            tf.keras.callbacks.ModelCheckpoint(
                str(ft_path), monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3,
                restore_best_weights=True, verbose=0,
            ),
        ],
        verbose=0,
    )

    # Persist history so the UI can show learning curves even after CLI runs.
    # Saved to outputs/TrainingN/ so the dashboard can discover it in order.
    history_data = {
        "epoch":        list(range(1, len(keras_hist.history["accuracy"]) + 1)),
        "accuracy":     [float(v) for v in keras_hist.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in keras_hist.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in keras_hist.history.get("loss", [])],
        "val_loss":     [float(v) for v in keras_hist.history.get("val_loss", [])],
    }
    cb_dir = next(
        (getattr(cb, "output_dir", None) for cb in (callbacks or []) if getattr(cb, "output_dir", None)),
        None,
    )
    out_dir = cb_dir if cb_dir else _next_training_output_dir()
    (out_dir / "history_finetune.json").write_text(json.dumps(history_data))
    # Also keep a copy alongside the model checkpoint
    (run_dir / "history_finetune.json").write_text(json.dumps(history_data))

    return {"model": model, "path": str(ft_path), "history": history_data}
