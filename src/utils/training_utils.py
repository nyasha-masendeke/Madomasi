import streamlit as st
import tensorflow as tf
import pandas as pd
import psutil

class DashboardCallback(tf.keras.callbacks.Callback):
    """Syncs Keras training metrics with Streamlit's session state."""
    def __init__(self, metrics_placeholder, chart_placeholder):
        super().__init__()
        self.metrics_placeholder = metrics_placeholder
        self.chart_placeholder = chart_placeholder
        if 'history' not in st.session_state:
            st.session_state.history = pd.DataFrame(columns=['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Update DataFrame
        new_data = pd.DataFrame([{
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True)

        # Update Visuals
        with self.metrics_placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("Current Accuracy", f"{logs.get('accuracy', 0):.2%}")
            c2.metric("Validation Loss", f"{logs.get('val_loss', 0):.4f}")

        with self.chart_placeholder.container():
            st.line_chart(st.session_state.history.set_index('epoch')[['accuracy', 'val_accuracy']])

def get_sys_stats():
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent
    }
