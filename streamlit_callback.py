import tensorflow as tf
import streamlit as st


class StreamlitTrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, metrics_placeholder):
        super().__init__()
        self.progress = progress_bar
        self.metrics = metrics_placeholder
        self.history = {
            "epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = self.params.get("epochs", 1)

        self.history["epoch"].append(epoch + 1)
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            self.history[key].append(logs.get(key))

        self.progress.progress(
            (epoch + 1) / total,
            text=f"Epoch {epoch + 1}/{total}"
        )

        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy")
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss")

        val_line = ""
        if val_acc is not None and val_loss is not None:
            val_line = f" | Val Acc: `{val_acc:.3f}` | Val Loss: `{val_loss:.3f}`"

        self.metrics.markdown(
            f"**Epoch {epoch + 1}/{total}** — "
            f"Acc: `{acc:.3f}` | Loss: `{loss:.3f}`{val_line}"
        )
