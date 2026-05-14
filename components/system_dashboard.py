"""System resource dashboard tab — CPU, RAM, disk, and usage history."""
import json
import platform
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

# Start tracemalloc once at module load so all subsequent allocations are tracked.
# Idempotent — safe to call even if another import already started it.
if not tracemalloc.is_tracing():
    tracemalloc.start(10)


def _read_temp() -> float | None:
    """Read CPU/SoC temperature. Tries Pi thermal zone first, then psutil sensors."""
    # Raspberry Pi (and most ARM Linux boards)
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return round(int(raw) / 1000, 1)
    except Exception:
        pass
    # x86 Linux / macOS via psutil
    try:
        if hasattr(psutil, "sensors_temperatures"):
            for _, entries in (psutil.sensors_temperatures() or {}).items():
                for entry in entries:
                    label = entry.label.lower()
                    if not label or "cpu" in label or "core" in label or "temp" in label:
                        return round(entry.current, 1)
    except Exception:
        pass
    return None


def _get_stats() -> dict:
    vm   = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")

    # interval=1.0 blocks for 1 s to get an accurate system-wide reading.
    # percpu=True lets us report the busiest core separately so a single
    # heavy subprocess doesn't get hidden by averaging across all cores.
    per_cpu = psutil.cpu_percent(interval=1.0, percpu=True)
    cpu_avg = sum(per_cpu) / len(per_cpu)
    cpu_max = max(per_cpu)

    stats = {
        "cpu": cpu_avg,
        "cpu_max_core": cpu_max,
        "ram": vm.percent,
        "ram_used_gb": vm.used / 1e9,
        "ram_total_gb": vm.total / 1e9,
        "disk": disk.percent,
        "disk_used_gb": disk.used / 1e9,
        "disk_total_gb": disk.total / 1e9,
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "os": platform.system(),
        "os_version": platform.version()[:40],
        "python": platform.python_version(),
        "machine": platform.machine(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "temp": _read_temp(),
    }

    # GPU detection via TensorFlow (no extra dependency)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        stats["gpu_count"] = len(gpus)
        stats["gpu_names"] = [g.name.split(":")[-1] for g in gpus]
    except Exception:
        stats["gpu_count"] = 0
        stats["gpu_names"] = []

    return stats


_RESOURCE_LOG = Path("outputs/resource_log.jsonl")


def _persist_sample(sample: dict) -> None:
    """Append one resource sample to outputs/resource_log.jsonl."""
    try:
        _RESOURCE_LOG.parent.mkdir(exist_ok=True)
        with open(_RESOURCE_LOG, "a") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception:
        pass


def _load_resource_history(n: int = 60) -> list:
    """Read the last N entries from resource_log.jsonl."""
    if not _RESOURCE_LOG.exists():
        return []
    try:
        lines = _RESOURCE_LOG.read_text().splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]
    except Exception:
        return []


def record_resource_sample(activity: str = "") -> None:
    """Sample system resources, persist to disk, and update session history."""
    if "resource_history" not in st.session_state:
        st.session_state.resource_history = _load_resource_history()
    stats = _get_stats()
    sample = {
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "timestamp": stats["timestamp"],
        "cpu":       stats["cpu"],
        "ram":       stats["ram"],
        "temp":      stats.get("temp"),
        "activity":  activity,
    }
    _persist_sample(sample)
    st.session_state.resource_history.append(sample)
    st.session_state.resource_history = st.session_state.resource_history[-60:]
    if activity:
        st.session_state.last_activity = activity
        st.session_state.last_activity_time = stats["timestamp"]


def _temp_gauge(temp: float) -> go.Figure:
    """Gauge for temperature in °C with Pi-specific thresholds (throttles at 80°C)."""
    color = "#E63946" if temp > 80 else "#F4A261" if temp > 65 else "#2D9E6B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp,
        number={"valueformat": ".1f", "suffix": "°C",
                "font": {"size": 20, "color": "#16213E", "family": "Inter"}},
        title={"text": "Temperature", "font": {"size": 13, "color": "#6B7280", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0",
                     "tickfont": {"size": 9, "color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#F8FAFC",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  65], "color": "#F0FFF4"},
                {"range": [65, 80], "color": "#FFFBEB"},
                {"range": [80, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": "#E63946", "width": 2},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _gauge(value: float, title: str, color: str, suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".0f", "suffix": suffix,
                "font": {"size": 20, "color": "#16213E", "family": "Inter"}},
        title={"text": title, "font": {"size": 13, "color": "#6B7280", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0",
                     "tickfont": {"size": 9, "color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#F8FAFC",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60],  "color": "#F0FFF4"},
                {"range": [60, 80], "color": "#FFFBEB"},
                {"range": [80, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": "#E63946", "width": 2},
                "thickness": 0.75,
                "value": 85,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _history_chart(history: list) -> go.Figure:
    times = [h["timestamp"] for h in history]
    cpu   = [h["cpu"] for h in history]
    ram   = [h["ram"] for h in history]
    temps = [h.get("temp") for h in history]
    has_temp = any(t is not None for t in temps)

    inference_times = [h["timestamp"] for h in history if h.get("activity") == "Inference"]
    training_times  = [h["timestamp"] for h in history if h.get("activity") == "Training"]

    n_cols = 3 if has_temp else 2
    subplot_titles = ["CPU Usage Over Time", "RAM Usage Over Time"]
    if has_temp:
        subplot_titles.append("Temperature Over Time")

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=times, y=cpu, mode="lines", fill="tozeroy",
        name="CPU %",
        line=dict(color="#E63946", width=2),
        fillcolor="rgba(230,57,70,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=times, y=ram, mode="lines", fill="tozeroy",
        name="RAM %",
        line=dict(color="#6366F1", width=2),
        fillcolor="rgba(99,102,241,0.08)",
    ), row=1, col=2)

    if has_temp:
        temp_vals = [t if t is not None else None for t in temps]
        fig.add_trace(go.Scatter(
            x=times, y=temp_vals, mode="lines", fill="tozeroy",
            name="Temp °C",
            line=dict(color="#F4A261", width=2),
            fillcolor="rgba(244,162,97,0.08)",
            connectgaps=True,
        ), row=1, col=3)
        fig.add_hline(y=80, line_dash="dash", line_color="#E63946",
                      annotation_text="Throttle 80°C",
                      annotation_font_size=9,
                      row=1, col=3)
        fig.update_yaxes(range=[0, 100], gridcolor="#F1F5F9", zeroline=False, row=1, col=3)

    # Inference markers
    if inference_times:
        inf_t   = [t for t in inference_times if t in times]
        inf_cpu = [cpu[times.index(t)] for t in inf_t]
        inf_ram = [ram[times.index(t)] for t in inf_t]
        fig.add_trace(go.Scatter(
            x=inf_t, y=inf_cpu, mode="markers", name="Inference",
            marker=dict(color="#F59E0B", size=9, symbol="diamond",
                        line=dict(color="white", width=1.5)),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inf_t, y=inf_ram, mode="markers", name="Inference",
            marker=dict(color="#F59E0B", size=9, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            showlegend=False,
        ), row=1, col=2)

    # Training markers
    if training_times:
        trn_t   = [t for t in training_times if t in times]
        trn_cpu = [cpu[times.index(t)] for t in trn_t]
        trn_ram = [ram[times.index(t)] for t in trn_t]
        fig.add_trace(go.Scatter(
            x=trn_t, y=trn_cpu, mode="markers", name="Training",
            marker=dict(color="#8B5CF6", size=9, symbol="star",
                        line=dict(color="white", width=1.5)),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trn_t, y=trn_ram, mode="markers", name="Training",
            marker=dict(color="#8B5CF6", size=9, symbol="star",
                        line=dict(color="white", width=1.5)),
            showlegend=False,
        ), row=1, col=2)

    fig.update_yaxes(range=[0, 100], gridcolor="#F1F5F9", zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[0, 100], gridcolor="#F1F5F9", zeroline=False, row=1, col=2)
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False, tickangle=-30,
                     tickfont=dict(size=9))
    fig.update_layout(
        height=260,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
                    font=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _render_confidence_drift(df: pd.DataFrame, baseline_df: pd.DataFrame, recent_df: pd.DataFrame) -> None:
    n = len(df)
    df = df.copy()
    window = min(10, n)
    df["rolling"] = df["confidence"].rolling(window, min_periods=1).mean()

    low_zones: list[tuple] = []
    in_zone = False
    zone_start = None
    for _, row in df.iterrows():
        if row["rolling"] < 0.5 and not in_zone:
            in_zone = True
            zone_start = row["ts"]
        elif row["rolling"] >= 0.5 and in_zone:
            in_zone = False
            low_zones.append((zone_start, row["ts"]))
    if in_zone and zone_start is not None:
        low_zones.append((zone_start, df["ts"].iloc[-1]))

    fig = go.Figure()
    for z0, z1 in low_zones:
        fig.add_vrect(x0=z0, x1=z1, fillcolor="#FEB2B2", opacity=0.25, line_width=0)

    baseline_mean = baseline_df["confidence"].mean()
    fig.add_hline(y=baseline_mean, line_dash="dot", line_color="#718096",
                  annotation_text=f"Baseline {baseline_mean:.1%}",
                  annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F4A261",
                  annotation_text="50% threshold", annotation_position="bottom right")

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["confidence"], mode="markers", name="Prediction",
        marker=dict(color="#CBD5E0", size=5, opacity=0.7),
        hovertemplate="%{x|%H:%M:%S} · %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["rolling"], mode="lines", name=f"Rolling avg ({window})",
        line=dict(color="#E63946", width=2.5),
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    if len(recent_df) >= 3:
        r_mean = recent_df["confidence"].mean()
        fig.add_trace(go.Scatter(
            x=recent_df["ts"], y=[r_mean] * len(recent_df),
            mode="lines", name=f"Recent avg {r_mean:.1%}",
            line=dict(color="#F4A261", width=2, dash="dash"),
        ))

    fig.update_layout(
        height=280, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        xaxis=dict(title="Time"),
    )
    st.plotly_chart(fig, width="stretch", key="drift_conf_chart")
    if low_zones:
        st.caption(f"Shaded regions: {len(low_zones)} low-confidence zone(s) where rolling avg < 50%.")


def _render_class_shift(df: pd.DataFrame, baseline_df: pd.DataFrame, recent_df: pd.DataFrame) -> None:
    from config import DISEASE_DISPLAY
    all_diseases = sorted(df["disease"].unique())
    display_names = [DISEASE_DISPLAY.get(d, d) for d in all_diseases]

    b_counts = baseline_df["disease"].value_counts()
    r_counts = recent_df["disease"].value_counts()
    b_pct = [b_counts.get(d, 0) / max(len(baseline_df), 1) for d in all_diseases]
    r_pct = [r_counts.get(d, 0) / max(len(recent_df), 1) for d in all_diseases]
    deltas = [r - b for r, b in zip(r_pct, b_pct)]

    for d, delta in zip(all_diseases, deltas):
        if abs(delta) > 0.15:
            direction = "spike ↑" if delta > 0 else "drop ↓"
            st.warning(f"**{DISEASE_DISPLAY.get(d, d)}** {direction} {delta:+.0%} vs baseline", icon="⚠️")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline", x=display_names, y=b_pct,
        marker_color="#90CDF4",
        text=[f"{v:.0%}" for v in b_pct], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=f"Recent (last {len(recent_df)})", x=display_names, y=r_pct,
        marker_color="#F4A261",
        text=[f"{v:.0%}" for v in r_pct], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", height=320, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", title="Proportion"),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, width="stretch", key="drift_class_chart")

    with st.expander("Delta table"):
        delta_df = pd.DataFrame({
            "Disease":    display_names,
            "Baseline %": [f"{v:.1%}" for v in b_pct],
            "Recent %":   [f"{v:.1%}" for v in r_pct],
            "Change":     [f"{v:+.1%}" for v in deltas],
        })
        st.dataframe(delta_df, hide_index=True, width="stretch")


def _render_ood_entropy(df: pd.DataFrame) -> None:
    has_entropy = "entropy" in df.columns and df["entropy"].notna().any()
    has_ood = "is_leaf" in df.columns and df["is_leaf"].notna().any()

    if not has_entropy and not has_ood:
        st.info("No entropy/OOD data yet — run new predictions to populate.", icon="📊")
        return

    df = df.copy()
    df["ood"] = (~df["is_leaf"].fillna(True).astype(bool)).astype(int)
    window = min(10, len(df))
    df["rolling_ood"] = df["ood"].rolling(window, min_periods=1).mean()

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Prediction entropy** — 0 = certain, 1 = uniform (OOD)")
        if has_entropy:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["entropy"], mode="lines+markers", name="Entropy",
                line=dict(color="#9F7AEA", width=2), marker=dict(size=5),
                hovertemplate="%{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0.75, line_dash="dash", line_color="#E53E3E",
                          annotation_text="OOD threshold (0.75)",
                          annotation_position="top right")
            fig.update_layout(
                height=250, template="plotly_white",
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1], title="Entropy"),
                xaxis=dict(title="Time"),
            )
            st.plotly_chart(fig, width="stretch", key="drift_entropy_chart")
        else:
            st.info("No entropy data in log yet.")

    with col2:
        st.caption(f"**Rolling OOD rate** — % non-leaf inputs in last {window} predictions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["rolling_ood"], mode="lines", name="OOD rate",
            line=dict(color="#E53E3E", width=2),
            fill="tozeroy", fillcolor="rgba(229,62,62,0.1)",
            hovertemplate="%{y:.0%}<extra></extra>",
        ))
        fig.add_hline(y=0.25, line_dash="dash", line_color="#D69E2E",
                      annotation_text="25% alert", annotation_position="top right")
        fig.update_layout(
            height=250, template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=".0%", range=[0, 1], title="OOD rate"),
            xaxis=dict(title="Time"),
        )
        st.plotly_chart(fig, width="stretch", key="drift_ood_chart")

    ood_count = int(df["ood"].sum())
    if ood_count > 0:
        st.info(f"{ood_count} OOD prediction(s) detected ({ood_count / len(df):.0%} of total).", icon="🔍")


def _render_time_patterns(df: pd.DataFrame) -> None:
    hourly = (
        df.groupby("hour")
        .agg(count=("confidence", "count"), avg_conf=("confidence", "mean"))
        .reindex(range(24), fill_value=0)
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["count"], name="Predictions",
        marker_color="#90CDF4",
        hovertemplate="Hour %{x}:00 — %{y} predictions<extra></extra>",
    ), secondary_y=False)

    mask = hourly["count"] > 0
    fig.add_trace(go.Scatter(
        x=hourly[mask]["hour"], y=hourly[mask]["avg_conf"],
        mode="lines+markers", name="Avg confidence",
        line=dict(color="#E63946", width=2), marker=dict(size=6),
        hovertemplate="Hour %{x}:00 — %{y:.1%}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        height=280, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Hour of day", dtick=2),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Predictions", secondary_y=False)
    fig.update_yaxes(title_text="Avg Confidence", tickformat=".0%", range=[0, 1], secondary_y=True)
    st.plotly_chart(fig, width="stretch", key="drift_time_chart")


@st.fragment(run_every="30s")
def _drift_panel() -> None:
    """Full inference drift dashboard — PSI, class shift, OOD/entropy, time patterns."""
    from config import DISEASE_DISPLAY

    st.markdown('<div class="section-label">Inference Drift Dashboard</div>', unsafe_allow_html=True)

    log_path = Path("outputs/inference_log.jsonl")
    if not log_path.exists():
        st.info("No inference log yet — run predictions to populate drift metrics.", icon="📊")
        return

    try:
        df = pd.read_json(log_path, lines=True)
    except Exception:
        st.warning("Inference log exists but could not be read.")
        return

    if len(df) < 5:
        st.info(f"{len(df)} prediction(s) logged — need at least 5 for analysis.", icon="📊")
        return

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour

    # Back-compat: old log entries won't have entropy / is_leaf
    if "entropy" not in df.columns:
        df["entropy"] = float("nan")
    if "is_leaf" not in df.columns:
        df["is_leaf"] = True

    n = len(df)
    half = max(1, n // 2)
    recent_n = min(20, half)
    baseline_df = df.iloc[:half]
    recent_df   = df.iloc[n - recent_n:]

    avg_conf     = df["confidence"].mean()
    baseline_conf = baseline_df["confidence"].mean()
    recent_conf  = recent_df["confidence"].mean()
    conf_delta   = recent_conf - baseline_conf

    ood_mask     = ~df["is_leaf"].fillna(True).astype(bool)
    ood_rate     = ood_mask.mean()
    recent_ood   = (~recent_df["is_leaf"].fillna(True).astype(bool)).mean()

    top_disease  = df["disease"].value_counts().idxmax()
    top_display  = DISEASE_DISPLAY.get(top_disease, top_disease)

    # Population Stability Index on confidence distribution
    def _psi(base_s: pd.Series, curr_s: pd.Series, bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        b = np.histogram(base_s.dropna(), bins=edges)[0].astype(float) + 1e-6
        c = np.histogram(curr_s.dropna(), bins=edges)[0].astype(float) + 1e-6
        b /= b.sum(); c /= c.sum()
        return float(np.sum((c - b) * np.log(c / b)))

    psi = _psi(baseline_df["confidence"], recent_df["confidence"]) if n >= 20 else None

    if psi is None:
        drift_label = "Insufficient data"
    elif psi < 0.1:
        drift_label = "Stable"
    elif psi < 0.25:
        drift_label = "Moderate drift"
    else:
        drift_label = "Significant drift"

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Predictions", n)
    c2.metric("Avg Confidence",  f"{avg_conf:.1%}",
              delta=f"{conf_delta:+.1%}" if n >= 10 else None)
    c3.metric("OOD Rate",        f"{ood_rate:.1%}",
              delta=f"{recent_ood - ood_rate:+.1%}" if n >= 10 else None,
              delta_color="inverse")
    c4.metric("Top Disease",     top_display)
    c5.metric("PSI Drift",       drift_label,
              delta=f"{psi:.3f}" if psi is not None else None,
              delta_color="off")

    # ── Alert banner ──────────────────────────────────────────────────────────
    if psi is not None:
        if psi >= 0.25:
            st.error(
                f"PSI = {psi:.3f} — significant distribution shift detected. "
                "Review recent inputs and consider retraining.", icon="🔴"
            )
        elif psi >= 0.1:
            st.warning(f"PSI = {psi:.3f} — moderate drift detected. Monitor closely.", icon="⚠️")
        elif recent_ood > 0.25:
            st.warning(
                f"OOD rate is {recent_ood:.0%} in recent predictions — "
                "many non-leaf inputs are being submitted.", icon="⚠️"
            )
        else:
            st.success(f"PSI = {psi:.3f} — model input distribution is stable.", icon="✅")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confidence Drift", "Class Distribution", "OOD / Entropy", "Time Patterns"
    ])
    with tab1:
        _render_confidence_drift(df, baseline_df, recent_df)
    with tab2:
        _render_class_shift(df, baseline_df, recent_df)
    with tab3:
        _render_ood_entropy(df)
    with tab4:
        _render_time_patterns(df)

    # ── Footer controls ───────────────────────────────────────────────────────
    st.divider()
    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        if st.button("Clear log", type="secondary", key="clear_inf_log"):
            log_path.unlink(missing_ok=True)
            st.rerun()
    with col_info:
        last_ts = df["ts"].max().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"`{log_path}` · {n} entries · last updated {last_ts}")


@st.fragment(run_every="5s")
def _live_metrics() -> None:
    """Gauges + history — reruns every 5 s independently of the main page.

    Using @st.fragment avoids the old time.sleep(5); st.rerun() pattern that
    blocked the entire Streamlit server thread and froze the UI.
    """
    auto_refresh = st.session_state.get("_sys_auto_refresh", True)
    if not auto_refresh:
        return

    stats = _get_stats()          # blocks ~1 s for accurate CPU reading

    # Seed from disk on first load so history survives restarts
    if "resource_history" not in st.session_state:
        st.session_state.resource_history = _load_resource_history()

    sample = {
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "timestamp": stats["timestamp"],
        "cpu":       stats["cpu"],
        "ram":       stats["ram"],
        "temp":      stats.get("temp"),
        "activity":  "",
    }
    _persist_sample(sample)
    st.session_state.resource_history.append(sample)
    st.session_state.resource_history = st.session_state.resource_history[-60:]

    # ── Activity Status Banner ─────────────────────────────────────────
    is_training   = st.session_state.get("training_active", False)
    last_activity = st.session_state.get("last_activity", "")
    last_time     = st.session_state.get("last_activity_time", "")

    if is_training:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#EDE9FE,#DDD6FE);
                        border:1.5px solid #8B5CF6;border-radius:10px;
                        padding:0.65rem 1.1rem;display:flex;align-items:center;
                        gap:10px;margin-bottom:1rem;">
                <span style="font-size:1.1rem;">⚡</span>
                <span style="color:#5B21B6;font-weight:600;font-size:0.88rem;">
                    Training in progress — resource data is being recorded
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif last_activity:
        activity_color = "#F59E0B" if last_activity == "Inference" else "#8B5CF6"
        bg_color = "#FFFBEB" if last_activity == "Inference" else "#EDE9FE"
        border_color = "#F59E0B" if last_activity == "Inference" else "#8B5CF6"
        icon = "🔍" if last_activity == "Inference" else "🧠"
        st.markdown(
            f"""
            <div style="background:{bg_color};border:1.5px solid {border_color};
                        border-radius:10px;padding:0.65rem 1.1rem;
                        display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
                <span style="font-size:1.1rem;">{icon}</span>
                <span style="color:{activity_color};font-weight:600;font-size:0.88rem;">
                    Last activity: {last_activity} at {last_time}
                    &nbsp;·&nbsp; markers shown on history chart below
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Gauges ────────────────────────────────────────────────────────
    temp = stats.get("temp")
    g1, g2, g3, g4 = st.columns(4)

    cpu_color  = "#E63946" if stats["cpu"]  > 80 else "#F4A261" if stats["cpu"]  > 60 else "#2D9E6B"
    ram_color  = "#E63946" if stats["ram"]  > 80 else "#F4A261" if stats["ram"]  > 60 else "#6366F1"
    disk_color = "#E63946" if stats["disk"] > 85 else "#F4A261" if stats["disk"] > 70 else "#0EA5E9"

    with g1:
        st.markdown('<div class="section-label">CPU</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["cpu"], "CPU Usage", cpu_color), width="stretch", key="g_cpu")
        max_core = stats.get("cpu_max_core", stats["cpu"])
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["cpu_count"]} physical &nbsp;·&nbsp; {stats["cpu_count_logical"]} logical cores'
            f'<br>Busiest core: <b style="color:#16213E">{max_core:.0f}%</b></p>',
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown('<div class="section-label">Memory</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["ram"], "RAM Usage", ram_color), width="stretch", key="g_ram")
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["ram_used_gb"]:.1f} GB used of {stats["ram_total_gb"]:.1f} GB</p>',
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown('<div class="section-label">Disk</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["disk"], "Disk Usage", disk_color), width="stretch", key="g_disk")
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["disk_used_gb"]:.1f} GB used of {stats["disk_total_gb"]:.1f} GB</p>',
            unsafe_allow_html=True,
        )
    with g4:
        st.markdown('<div class="section-label">Temperature</div>', unsafe_allow_html=True)
        if temp is not None:
            st.plotly_chart(_temp_gauge(temp), width="stretch", key="g_temp")
            throttle_margin = 80 - temp
            st.markdown(
                f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
                f'SoC &nbsp;·&nbsp; throttle at 80°C'
                f'<br>Headroom: <b style="color:#16213E">{throttle_margin:.0f}°C</b></p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="height:220px;display:flex;align-items:center;justify-content:center;'
                'color:#CBD5E0;font-size:0.8rem;text-align:center;">'
                '🌡️<br>No sensor<br>detected</div>',
                unsafe_allow_html=True,
            )

    # Alerts
    if stats["cpu"] > 85:
        st.error("High CPU usage — this may slow down training.", icon="⚠️")
    if stats["ram"] > 85:
        st.error("High memory usage — consider closing other applications.", icon="⚠️")
    elif stats["ram"] > 75:
        st.warning("Memory usage above 75%.", icon="⚠️")
    if temp is not None and temp > 80:
        st.error(f"Temperature critical: {temp}°C — Pi is throttling! Check airflow.", icon="🌡️")
    elif temp is not None and temp > 70:
        st.warning(f"Temperature elevated: {temp}°C — approaching throttle limit.", icon="🌡️")

    st.divider()

    # ── Usage History Chart ────────────────────────────────────────────
    st.markdown('<div class="section-label">Usage History (last 60 samples)</div>', unsafe_allow_html=True)
    history = st.session_state.resource_history
    if len(history) >= 2:
        st.plotly_chart(_history_chart(history), width="stretch", key="history_chart")
    else:
        st.markdown(
            """
            <div style="background:white;border-radius:14px;padding:2.5rem;
                        text-align:center;border:1px solid #EEF0F2;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
                <div style="color:#94A3B8;font-size:0.85rem;">
                    History builds up as you use the app.
                    Run inference or start training to see activity markers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()


def render_dashboard():
    # ── Header ────────────────────────────────────────────────────────
    col_title, col_auto, col_refresh = st.columns([3, 1, 1])
    col_title.markdown(
        """
        <div style="margin-bottom:0.25rem;">
            <span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:1.2px;color:#94A3B8;">Live Monitoring</span>
            <h3 style="margin:0;color:#16213E;font-size:1.3rem;font-weight:700;">
                System Resource Usage
            </h3>
            <p style="margin:0;color:#6B7280;font-size:0.85rem;">
                Tracks CPU, memory and disk load during training and inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    auto_refresh = col_auto.checkbox(
        "Auto-refresh", value=True,
        help="Polls every 5 seconds without blocking the UI.",
    )
    # Store in session_state so the fragment can read the latest value
    st.session_state["_sys_auto_refresh"] = auto_refresh

    if col_refresh.button("Refresh", type="secondary", width="stretch"):
        st.rerun()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Live metrics fragment — reruns every 5 s independently (no blocking sleep)
    _live_metrics()

    st.divider()

    # ── System Info (static — doesn't need live refresh) ───────────────
    st.markdown('<div class="section-label">System Information</div>', unsafe_allow_html=True)
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log  = psutil.cpu_count(logical=True)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Operating System", platform.system())
    i2.metric("Python Version",   platform.python_version())
    i3.metric("Architecture",     platform.machine())
    i4.metric("CPU Cores", f"{cpu_phys} / {cpu_log}",
              delta="physical / logical", delta_color="off")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    try:
        gpus      = tf.config.list_physical_devices("GPU")
        gpu_count = len(gpus)
        gpu_names = [g.name.split(":")[-1] for g in gpus]
    except Exception:
        gpu_count, gpu_names = 0, []
    gpu_color  = "#2D9E6B" if gpu_count > 0 else "#94A3B8"
    gpu_label  = ", ".join(gpu_names) if gpu_count else "None detected"
    gpu_status = "Training will use GPU" if gpu_count else "Training will use CPU"
    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:0.75rem 1.25rem;'
        f'display:inline-flex;align-items:center;gap:12px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #EEF0F2;">'
        f'<span style="font-size:1.2rem;">🖥️</span>'
        f'<span style="color:#6B7280;font-size:0.85rem;">GPU ({gpu_count} detected):</span>'
        f'<span style="color:{gpu_color};font-weight:700;font-size:0.9rem;">{gpu_label}</span>'
        f'<span style="color:#94A3B8;font-size:0.78rem;margin-left:4px;">— {gpu_status}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    _drift_panel()
