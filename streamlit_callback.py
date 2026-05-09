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
    def __init__(
        self,
        progress_bar,
        metrics_placeholder,
        chart_placeholder,
        stage_label: str = "",
        epoch_offset: int = 0,
        curve_save_interval: int = 10,
    ):
        super().__init__()
        self.progress = progress_bar
        self.metrics = metrics_placeholder
        self.chart = chart_placeholder
        self.stage_label = stage_label
        self.epoch_offset = epoch_offset
        self.curve_save_interval = curve_save_interval
        self.history = {
            "epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []
        }
        self.output_dir: Path | None = None
        # Tracks which epoch snapshots have been saved so we can report them
        self.saved_snapshots: list[str] = []

    def on_train_begin(self, logs=None):
        """Claim an outputs/TrainingN slot at the very start of training."""
        self.output_dir = _next_training_output_dir()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = self.params.get("epochs", 1)
        display_epoch = epoch + 1 + self.epoch_offset

        self.history["epoch"].append(display_epoch)
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            self.history[key].append(logs.get(key))

        # Save rolling history to the run's TrainingN dir after every epoch
        if self.output_dir:
            label = self.stage_label.replace(" ", "_").lower() or "training"
            (self.output_dir / f"history_{label}.json").write_text(json.dumps(self.history))

        # Snapshot curves at every curve_save_interval epochs
        if display_epoch % self.curve_save_interval == 0:
            self._save_curve_snapshot(display_epoch)

        # Progress bar
        prefix = f"[{self.stage_label}] " if self.stage_label else ""
        self.progress.progress(
            (epoch + 1) / total,
            text=f"{prefix}Epoch {epoch + 1} / {total}",
        )

        # Metric summary row
        acc      = logs.get("accuracy", 0)
        loss     = logs.get("loss", 0)
        val_acc  = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        cols = self.metrics.columns(4)
        cols[0].metric("Train Acc",  f"{acc:.3f}")
        cols[1].metric("Train Loss", f"{loss:.4f}")
        cols[2].metric("Val Acc",    f"{val_acc:.3f}"  if val_acc  is not None else "—")
        cols[3].metric("Val Loss",   f"{val_loss:.4f}" if val_loss is not None else "—")

        # Live learning curves in the dashboard
        self.chart.plotly_chart(self._build_figure(), use_container_width=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Shared palette + dark theme — matches plot_learning_curves in main_ui.py
    _C = {
        "train_acc":  "#22C55E",
        "val_acc":    "#F97316",
        "train_loss": "#EF4444",
        "val_loss":   "#60A5FA",
    }
    _BG   = "#0F172A"
    _PLOT = "#1E293B"
    _GRID = "#334155"
    _FONT = "#F1F5F9"

    def _build_figure(self) -> go.Figure:
        """Build and return the current learning-curve Plotly figure."""
        epochs = self.history["epoch"]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Accuracy — {self.stage_label}",
                f"Loss — {self.stage_label}",
            ),
            horizontal_spacing=0.14,
        )

        fig.add_trace(go.Scatter(
            x=epochs, y=self.history["accuracy"],
            mode="lines+markers", name="Train Acc",
            line=dict(color=self._C["train_acc"], width=3),
            marker=dict(size=7),
        ), row=1, col=1)

        val_acc = [v for v in self.history["val_accuracy"] if v is not None]
        if val_acc:
            fig.add_trace(go.Scatter(
                x=epochs[:len(val_acc)], y=val_acc,
                mode="lines+markers", name="Val Acc",
                line=dict(color=self._C["val_acc"], width=3, dash="dash"),
                marker=dict(size=7),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=epochs, y=self.history["loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color=self._C["train_loss"], width=3),
            marker=dict(size=7),
        ), row=1, col=2)

        val_loss = [v for v in self.history["val_loss"] if v is not None]
        if val_loss:
            fig.add_trace(go.Scatter(
                x=epochs[:len(val_loss)], y=val_loss,
                mode="lines+markers", name="Val Loss",
                line=dict(color=self._C["val_loss"], width=3, dash="dash"),
                marker=dict(size=7),
            ), row=1, col=2)

        fig.update_layout(
            height=380,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.06,
                        xanchor="center", x=0.5, font=dict(size=12, color=self._FONT)),
            margin=dict(l=30, r=20, t=60, b=30),
            paper_bgcolor=self._BG,
            plot_bgcolor=self._PLOT,
            font=dict(color=self._FONT),
        )
        fig.update_xaxes(title_text="Epoch", gridcolor=self._GRID,
                         zeroline=False, dtick=1, color=self._FONT)
        fig.update_yaxes(title_text="Accuracy", gridcolor=self._GRID, zeroline=False,
                         range=[0, 1], color=self._FONT, row=1, col=1)
        fig.update_yaxes(title_text="Loss", gridcolor=self._GRID,
                         zeroline=False, color=self._FONT, row=1, col=2)

        return fig

    def _save_curve_snapshot(self, epoch: int) -> None:
        """Persist history JSON + interactive HTML to outputs/training_curves/training_epoch_{epoch}/."""
        snap_dir = Path("outputs") / "training_curves" / f"training_epoch_{epoch}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        (snap_dir / "history.json").write_text(json.dumps(self.history))

        fig = self._build_figure()
        fig.update_layout(title_text=f"Training curves — epoch {epoch}")
        fig.write_html(str(snap_dir / "curves.html"))

        self.saved_snapshots.append(str(snap_dir))
