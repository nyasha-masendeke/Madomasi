import tensorflow as tf
import streamlit as st

class StreamlitTrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, metrics_placeholder):
        self.progress = progress_bar
        self.metrics = metrics_placeholder
        self.history = {"epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoch"].append(epoch + 1)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        # Update UI
        self.progress.progress((epoch + 1) / self.params["epochs"])
        acc = logs.get("accuracy", 0)
        loss = logs.get("loss", 0)
        self.metrics.markdown(f"🔹 **Epoch {epoch+1}/{self.params['epochs']}** | Acc: `{acc:.3f}` | Loss: `{loss:.3f}`")
        st.rerun()  # Force UI update without full script rerun
