"""Continue fine-tuning from the latest fine_tuned.keras and append epochs
to the existing Training<N>/history_finetune.json + resources_finetune.json,
so the metrics_to_excel sheet shows one continuous run.

Usage:
  python -m src.training.finetune_3class_continue --epochs 5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.training.data import next_training_output_dir
from src.training.features import fine_tune_model
from src.training.resource_callback import ResourceLogger
from src.export.metrics_to_excel import write_workbook, DEFAULT_OUT

MODEL_ROOT = Path("models/trained_3class")
SPLIT_ROOT = Path("data/splits_3class")
OUTPUTS    = Path("outputs")


def _latest_fine_tuned() -> Path:
    runs = sorted(MODEL_ROOT.glob("run_*"), key=lambda p: p.name, reverse=True)
    for r in runs:
        m = r / "fine_tuned.keras"
        if m.exists():
            return m
    raise SystemExit("No fine_tuned.keras found — run finetune_3class.py first.")


def _latest_training_dir_with(filename: str) -> Path | None:
    candidates = [
        p for p in OUTPUTS.glob("Training*")
        if p.is_dir() and p.name[8:].isdigit() and (p / filename).exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name[8:]))


def _append_with_offset(existing: dict, new: dict, epoch_offset: int) -> dict:
    """Concatenate per-epoch lists from `new` onto `existing`; shift the new epoch column by offset."""
    n_new = len(new.get("epoch", []))
    out = dict(existing)
    for k, v in new.items():
        if k == "epoch":
            out[k] = list(existing.get(k, [])) + [int(e) + epoch_offset for e in v]
        elif isinstance(v, list) and len(v) == n_new:
            out[k] = list(existing.get(k, [])) + list(v)
        else:
            out.setdefault(k, v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr",     type=float, default=1e-5)
    ap.add_argument("--batch-size",       type=int, default=16)
    ap.add_argument("--unfreeze-layers",  type=int, default=20)
    args = ap.parse_args()

    fine_tuned = _latest_fine_tuned()
    prior_dir  = _latest_training_dir_with("history_finetune.json")
    if prior_dir is None:
        raise SystemExit("No prior history_finetune.json found to append to.")

    prior_hist_path = prior_dir / "history_finetune.json"
    prior_res_path  = prior_dir / "resources_finetune.json"
    prior_history = json.loads(prior_hist_path.read_text())
    prior_resources = json.loads(prior_res_path.read_text()) if prior_res_path.exists() else None
    prior_epochs = len(prior_history.get("epoch", []))

    train_dir   = SPLIT_ROOT / "train"
    scratch_dir = next_training_output_dir()
    res_logger  = ResourceLogger(output_dir=scratch_dir, stage="finetune")

    print(f">> Continuing from:    {fine_tuned}")
    print(f">> Prior run dir:      {prior_dir} ({prior_epochs} epochs)")
    print(f">> Scratch output dir: {scratch_dir}")
    print(f">> Will append {args.epochs} epochs (numbered {prior_epochs+1}..{prior_epochs+args.epochs}).")

    result = fine_tune_model(
        model_path=str(fine_tuned),
        data_dir=str(train_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        unfreeze_layers=args.unfreeze_layers,
        output_path=str(MODEL_ROOT / "latest.keras"),
        callbacks=[res_logger],
    )
    new_history = result["history"]

    new_res_path = scratch_dir / "resources_finetune.json"
    new_resources = json.loads(new_res_path.read_text()) if new_res_path.exists() else None

    merged_history = _append_with_offset(prior_history, new_history, epoch_offset=prior_epochs)
    prior_hist_path.write_text(json.dumps(merged_history))

    if prior_resources and new_resources:
        merged_res = _append_with_offset(prior_resources, new_resources, epoch_offset=prior_epochs)
        prior_res_path.write_text(json.dumps(merged_res))

    shutil.rmtree(scratch_dir, ignore_errors=True)

    total = len(merged_history.get("epoch", []))
    if merged_history.get("val_accuracy"):
        best = max(merged_history["val_accuracy"])
        best_ep = merged_history["epoch"][merged_history["val_accuracy"].index(best)]
        best_vl = min(merged_history["val_loss"])
        print(f">> Merged: {total} epochs, best val_acc {best:.4f} @ epoch {best_ep}, best val_loss {best_vl:.4f}")

    print(">> Updating Excel workbook…")
    info = write_workbook(DEFAULT_OUT)
    print(f"   wrote {info['path']}  ({info['runs']} histories, {info['sheets']} sheets)")


if __name__ == "__main__":
    main()
