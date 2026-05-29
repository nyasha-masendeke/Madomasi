"""End-to-end fine-tune the 3-class head trained by train_3class_head.py.

Unfreezes the last N layers of MobileNetV3Small and trains the full model
end-to-end at a very low learning rate against data/splits_3class/.

Outputs:
  outputs/Training<N>/history_finetune.json   — per-epoch metrics for Excel
  outputs/Training<N>/resources_finetune.json — per-epoch CPU/RAM/temp
  <head_run_dir>/fine_tuned.keras             — best-val_accuracy checkpoint

Usage:
  python -m src.training.finetune_3class
  python -m src.training.finetune_3class --epochs 5 --unfreeze-layers 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.training.data import next_training_output_dir
from src.training.features import fine_tune_model
from src.training.resource_callback import ResourceLogger
from src.export.metrics_to_excel import write_workbook, DEFAULT_OUT

MODEL_ROOT     = Path("models/trained_3class")
SPLIT_ROOT     = Path("data/splits_3class")


def _latest_head_full() -> Path:
    runs = sorted(MODEL_ROOT.glob("run_*"), key=lambda p: p.name, reverse=True)
    for r in runs:
        m = r / "head_full.keras"
        if m.exists():
            return m
    raise SystemExit(
        f"No head_full.keras found under {MODEL_ROOT}/run_*. "
        "Run `python -m src.training.train_3class_head` first."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr",     type=float, default=1e-5)
    ap.add_argument("--batch-size",       type=int, default=16)
    ap.add_argument("--unfreeze-layers",  type=int, default=20)
    ap.add_argument("--skip-excel",       action="store_true")
    args = ap.parse_args()

    head_full = _latest_head_full()
    train_dir = SPLIT_ROOT / "train"
    if not train_dir.exists():
        raise SystemExit(f"Missing split dir {train_dir} — symlink the 3 class folders into it first.")

    training_dir = next_training_output_dir()
    res_logger = ResourceLogger(output_dir=training_dir, stage="finetune")

    print(f">> Loading head model: {head_full}")
    print(f">> Train dir:          {train_dir}")
    print(f">> Logging to:         {training_dir}")
    print(f">> Fine-tuning ({args.epochs} epochs, lr={args.lr}, unfreeze last {args.unfreeze_layers} layers) …")

    result = fine_tune_model(
        model_path=str(head_full),
        data_dir=str(train_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        output_path=str(MODEL_ROOT / "latest.keras"),
        callbacks=[res_logger],
    )
    print(f"   fine-tuned model:  {result['path']}")

    hist = result["history"]
    if "val_accuracy" in hist and hist["val_accuracy"]:
        print(f"   best val_accuracy: {max(hist['val_accuracy']):.4f}")

    if not args.skip_excel:
        print(">> Updating Excel workbook …")
        info = write_workbook(DEFAULT_OUT)
        print(f"   wrote {info['path']}  ({info['runs']} histories, {info['sheets']} sheets)")


if __name__ == "__main__":
    main()
