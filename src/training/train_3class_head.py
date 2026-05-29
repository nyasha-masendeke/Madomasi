"""Retrain just the head on 3 classes: Early Blight, Late Blight, Healthy.

Reuses the cached MobileNetV3Small features in data/features/ — backbone is
not re-run. Filters samples to the 3 target classes, remaps labels to 0..2,
writes a new feature cache, and calls the existing train_head.

Outputs:
  data/features_3class/{features.npy,labels.npy,meta.json}  — filtered cache
  models/trained_3class/run_<ts>/{head.keras,head_full.keras,history_head.json}
  outputs/Training<N>/history_head.json                     — for Excel export

Usage:
  python -m src.training.train_3class_head
  python -m src.training.train_3class_head --epochs 20 --lr 0.0005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.training.data import next_training_output_dir
from src.training.features import train_head
from src.training.resource_callback import ResourceLogger
from src.export.metrics_to_excel import write_workbook, DEFAULT_OUT

TARGET_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

SRC_FEATURES = Path("data/features")
DST_FEATURES = Path("data/features_3class")
MODEL_ROOT   = Path("models/trained_3class")


def build_filtered_cache() -> dict:
    features = np.load(SRC_FEATURES / "features.npy")
    labels   = np.load(SRC_FEATURES / "labels.npy")
    meta     = json.loads((SRC_FEATURES / "meta.json").read_text())

    src_names = meta["class_names"]
    missing = [c for c in TARGET_CLASSES if c not in src_names]
    if missing:
        raise SystemExit(f"Classes not found in source cache: {missing}")

    src_indices = [src_names.index(c) for c in TARGET_CLASSES]
    keep = np.isin(labels, src_indices)
    sub_features = features[keep]
    sub_labels_old = labels[keep]

    remap = {old: new for new, old in enumerate(src_indices)}
    sub_labels = np.array([remap[int(l)] for l in sub_labels_old], dtype=labels.dtype)

    DST_FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(DST_FEATURES / "features.npy", sub_features)
    np.save(DST_FEATURES / "labels.npy",   sub_labels)

    new_meta = {
        "class_names":   TARGET_CLASSES,
        "feature_shape": meta["feature_shape"],
        "num_samples":   int(len(sub_labels)),
        "num_classes":   len(TARGET_CLASSES),
        "derived_from":  str(SRC_FEATURES),
    }
    (DST_FEATURES / "meta.json").write_text(json.dumps(new_meta))

    counts = {TARGET_CLASSES[i]: int((sub_labels == i).sum()) for i in range(len(TARGET_CLASSES))}
    return {"num_samples": int(len(sub_labels)), "per_class": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr",     type=float, default=0.001)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--skip-excel", action="store_true")
    args = ap.parse_args()

    print(">> Filtering cached features to 3 classes …")
    info = build_filtered_cache()
    print(f"   kept {info['num_samples']} samples")
    for k, v in info["per_class"].items():
        print(f"     {v:5d}  {k}")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    training_dir = next_training_output_dir()
    res_logger = ResourceLogger(output_dir=training_dir, stage="head")
    print(f">> Training head … (metrics + resources -> {training_dir})")
    result = train_head(
        features_dir=str(DST_FEATURES),
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        output_path=str(MODEL_ROOT / "latest.keras"),
        callbacks=[res_logger],
    )
    print(f"   head:       {result['head_path']}")
    print(f"   full model: {result['full_model_path']}")
    print(f"   run dir:    {result['run_dir']}")

    hist = result["history"]
    if "val_accuracy" in hist and hist["val_accuracy"]:
        best = max(hist["val_accuracy"])
        print(f"   best val_accuracy: {best:.4f}")

    if not args.skip_excel:
        print(">> Updating Excel workbook …")
        info = write_workbook(DEFAULT_OUT)
        print(f"   wrote {info['path']}  ({info['runs']} histories, {info['sheets']} sheets)")


if __name__ == "__main__":
    main()
