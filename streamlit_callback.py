import json
from pathlib import Path
import tensorflow as tf
import streamlit as st

from src.training.data import next_training_output_dir as _next_training_output_dir


class StreamlitTrainCallback(tf.keras.callbacks.Callback):
    """Keras callback that writes training progress to st.session_state["_train_state"].

    Training runs in a background thread; a @st.fragment in training_tab polls
    this state every 2 s so the UI stays live without blocking the server thread.
    """

    def __init__(self, stage_label: str = "", epoch_offset: int = 0):
        super().__init__()
        self.stage_label  = stage_label
        self.epoch_offset = epoch_offset
        self.history = {
            "epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [],
        }
        self.output_dir: Path | None = None

    def on_train_begin(self, logs=None):
        self.output_dir = _next_training_output_dir()
        st.session_state["_train_state"] = {
            "progress": 0.0,
            "text":     f"[{self.stage_label}] Starting…",
            "history":  {"epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []},
            "logs":     {},
            "active":   True,
            "error":    None,
        }

    def on_epoch_end(self, epoch, logs=None):
        logs  = logs or {}
        total = self.params.get("epochs", 1)
        display_epoch = epoch + 1 + self.epoch_offset

        self.history["epoch"].append(display_epoch)
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            self.history[key].append(logs.get(key))

        if self.output_dir:
            label = self.stage_label.replace(" ", "_").lower() or "training"
            (self.output_dir / f"history_{label}.json").write_text(json.dumps(self.history))

        prefix = f"[{self.stage_label}] " if self.stage_label else ""
        st.session_state["_train_state"] = {
            "progress": (epoch + 1) / total,
            "text":     f"{prefix}Epoch {epoch + 1} / {total}",
            "history":  {k: list(v) for k, v in self.history.items()},
            "logs":     {k: logs.get(k) for k in ("accuracy", "val_accuracy", "loss", "val_loss")},
            "active":   True,
            "error":    None,
        }
