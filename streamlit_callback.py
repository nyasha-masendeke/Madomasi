import json
from pathlib import Path
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _next_training_output_dir() -> Path:
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


class StreamlitTrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, metrics_placeholder, chart_placeholder,
                 stage_label: str = "", epoch_offset: int = 0):
        super().__init__()
        self.progress = progress_bar
        self.metrics = metrics_placeholder
        self.chart = chart_placeholder
        self.stage_label = stage_label
        self.epoch_offset = epoch_offset   # shift epoch numbers so stages display continuously
        self.history = {
            "epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []
        }
        self.output_dir: Path | None = None  # set in on_train_begin

    def on_train_begin(self, logs=None):
        """Claim an outputs/TrainingN slot at the very start of training."""
        self.output_dir = _next_training_output_dir()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = self.params.get("epochs", 1)
        display_epoch = epoch + 1 + self.epoch_offset

        # Record metrics using display epoch so stages are continuous on the chart
        self.history["epoch"].append(display_epoch)
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            self.history[key].append(logs.get(key))

        # ── Save history to disk after every epoch ────────────────────────
        # This way curves survive interruptions and are visible on the
        # Training Dashboard even after a page reload or CLI run.
        if self.output_dir:
            label = self.stage_label.replace(" ", "_").lower() or "training"
            history_path = self.output_dir / f"history_{label}.json"
            history_path.write_text(json.dumps(self.history))

        # Progress bar
        prefix = f"[{self.stage_label}] " if self.stage_label else ""
        self.progress.progress(
            (epoch + 1) / total,
            text=f"{prefix}Epoch {epoch + 1} / {total}"
        )

        # Metric summary row
        acc = logs.get("accuracy", 0)
        loss = logs.get("loss", 0)
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        cols = self.metrics.columns(4)
        cols[0].metric("Train Acc", f"{acc:.3f}")
        cols[1].metric("Train Loss", f"{loss:.4f}")
        cols[2].metric("Val Acc", f"{val_acc:.3f}" if val_acc is not None else "—")
        cols[3].metric("Val Loss", f"{val_loss:.4f}" if val_loss is not None else "—")

        # Live learning curves
        self._render_curves()

    def _render_curves(self):
        epochs = self.history["epoch"]
        if not epochs:
            return

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Accuracy", "Loss"),
            horizontal_spacing=0.12,
        )

        # Accuracy
        fig.add_trace(
            go.Scatter(
                x=epochs, y=self.history["accuracy"],
                mode="lines+markers", name="Train Acc",
                line=dict(color="#2E7D32", width=2.5),
                marker=dict(size=6),
            ),
            row=1, col=1,
        )
        val_acc = [v for v in self.history["val_accuracy"] if v is not None]
        if val_acc:
            fig.add_trace(
                go.Scatter(
                    x=epochs[:len(val_acc)], y=val_acc,
                    mode="lines+markers", name="Val Acc",
                    line=dict(color="#FF6F00", width=2.5, dash="dash"),
                    marker=dict(size=6),
                ),
                row=1, col=1,
            )

        # Loss
        fig.add_trace(
            go.Scatter(
                x=epochs, y=self.history["loss"],
                mode="lines+markers", name="Train Loss",
                line=dict(color="#C62828", width=2.5),
                marker=dict(size=6),
            ),
            row=1, col=2,
        )
        val_loss = [v for v in self.history["val_loss"] if v is not None]
        if val_loss:
            fig.add_trace(
                go.Scatter(
                    x=epochs[:len(val_loss)], y=val_loss,
                    mode="lines+markers", name="Val Loss",
                    line=dict(color="#1565C0", width=2.5, dash="dash"),
                    marker=dict(size=6),
                ),
                row=1, col=2,
            )

        fig.update_layout(
            height=380,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig.update_xaxes(title_text="Epoch", gridcolor="#EEEEEE", dtick=1)
        fig.update_yaxes(title_text="Score", gridcolor="#EEEEEE", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Loss", gridcolor="#EEEEEE", row=1, col=2)

        self.chart.plotly_chart(fig, use_container_width=True)
