import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import DISEASE_CLASSES, IMAGE_SIZE


def evaluate_model(model_or_path, test_dir: str, batch_size: int = 16) -> dict:
    """Evaluate a trained model on a held-out test set.

    Args:
        model_or_path: Loaded tf.keras.Model or path string to a .keras file.

    Returns:
        loss, accuracy, confusion_matrix, class_names, report (per-class metrics)
    """
    from sklearn.metrics import confusion_matrix, classification_report

    model = (
        model_or_path if isinstance(model_or_path, tf.keras.Model)
        else tf.keras.models.load_model(model_or_path)
    )
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

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    class_names_raw = sorted([p.name for p in Path(test_dir).iterdir() if p.is_dir()])
    class_names = [
        DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else n
        for i, n in enumerate(class_names_raw)
    ]

    cm     = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "loss":             loss,
        "accuracy":         accuracy,
        "confusion_matrix": cm,
        "class_names":      class_names,
        "report":           report,
    }


def compute_tsne(
    features_dir: str,
    n_samples: int = 1500,
    perplexity: int = 30,
) -> dict:
    """Run t-SNE on cached MobileNetV3 feature vectors and return 2-D coordinates.

    Returns:
        x, y (2-D coords), labels, class_names, n_samples, n_total
    """
    from sklearn.manifold import TSNE

    feat_path   = Path(features_dir) / "features.npy"
    labels_path = Path(features_dir) / "labels.npy"
    meta_path   = Path(features_dir) / "meta.json"

    if not feat_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"features.npy / labels.npy not found in {features_dir}")

    features    = np.load(feat_path)
    labels      = np.load(labels_path).astype(int)
    class_names = DISEASE_CLASSES
    if meta_path.exists():
        meta        = json.loads(meta_path.read_text())
        class_names = meta.get("class_names", DISEASE_CLASSES)

    n = len(features)
    if n > n_samples:
        rng      = np.random.default_rng(42)
        idx      = rng.choice(n, n_samples, replace=False)
        features = features[idx]
        labels   = labels[idx]

    perp   = min(perplexity, max(5, len(features) // 3))
    coords = TSNE(
        n_components=2, perplexity=perp, random_state=42,
        n_iter=1000, learning_rate="auto", init="pca",
    ).fit_transform(features)

    return {
        "x":          coords[:, 0].tolist(),
        "y":          coords[:, 1].tolist(),
        "labels":     labels.tolist(),
        "class_names": class_names,
        "n_samples":  len(features),
        "n_total":    n,
    }
