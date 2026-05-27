import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE, DROPOUT_RATE


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def next_training_output_dir() -> Path:
    """Return the next unused outputs/TrainingN directory and create it."""
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(data_dir: str, batch_size: int, val_split: float = 0.2):
    """Load train/val splits from a class-named directory tree."""
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
    val_ds   = val_ds.prefetch(autotune)
    train_ds.class_names = class_names
    val_ds.class_names   = class_names
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(num_classes: int, freeze_base: bool = True) -> tf.keras.Model:
    """MobileNetV3Small backbone with a GAP + BatchNorm + Dense(256) + Dropout + Dense head."""
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = not freeze_base

    inputs  = tf.keras.Input((*IMAGE_SIZE, 3))
    x       = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x       = base(x, training=not freeze_base)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir: str,
    output_dir: str = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    progress_fn=None,
) -> dict:
    """Copy images from a flat class-directory tree into train/val/test splits.

    Source layout:
        source_dir/ClassName_A/img1.jpg ...

    Output layout:
        output_dir/train/ClassName_A/ ...
        output_dir/val/ClassName_A/   ...
        output_dir/test/ClassName_A/  ...

    Source files are copied — never modified.

    Returns:
        train_dir, val_dir, test_dir, counts, num_classes
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    source = Path(source_dir)
    output = Path(output_dir)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits = ["train", "val", "test"]
    counts = {s: 0 for s in splits}

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
        "output_dir":  str(output),
        "train_dir":   str(output / "train"),
        "val_dir":     str(output / "val"),
        "test_dir":    str(output / "test"),
        "counts":      counts,
        "num_classes": len(class_dirs),
    }


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------

def check_class_balance(data_dir: str) -> dict:
    """Count images per class and compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight as _ccw

    IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    data_path = Path(data_dir)
    classes   = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    counts    = {
        cls: len([f for f in (data_path / cls).iterdir() if f.suffix.lower() in IMG_EXTS])
        for cls in classes
    }
    total = sum(counts.values())
    if total == 0:
        return {"classes": classes, "counts": counts, "total": 0,
                "weights": {}, "imbalance_ratio": 0.0, "is_imbalanced": False}

    y     = np.array([i for i, cls in enumerate(classes) for _ in range(counts[cls])])
    raw_w = _ccw("balanced", classes=np.unique(y), y=y)
    weights = {cls: float(w) for cls, w in zip(classes, raw_w)}

    max_c = max(counts.values())
    min_c = min(v for v in counts.values() if v > 0)
    ratio = round(max_c / min_c, 2)
    return {
        "classes":        classes,
        "counts":         counts,
        "total":          total,
        "weights":        weights,
        "imbalance_ratio": ratio,
        "is_imbalanced":  ratio > 3.0,
    }
