"""System resource dashboard tab — CPU, RAM, disk, and usage history."""
import platform
import time
from datetime import datetime

import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots


def _get_stats() -> dict:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    stats = {
        "cpu": psutil.cpu_percent(interval=0.3),
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
    }
    try:
        if hasattr(psutil, "sensors_temperatures"):
            for _, entries in (psutil.sensors_temperatures() or {}).items():
                for entry in entries:
                    if "cpu" in entry.label.lower() or "core" in entry.label.lower():
                        stats["temp"] = round(entry.current, 1)
                        break
    except Exception:
        pass
    return stats


def _gauge(value: float, title: str, color: str, suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": "#16213E", "family": "Inter"}},
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
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _history_chart(history: list) -> go.Figure:
    times = [h["timestamp"] for h in history]
    cpu   = [h["cpu"] for h in history]
    ram   = [h["ram"] for h in history]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CPU Usage Over Time", "RAM Usage Over Time"),
        horizontal_spacing=0.1,
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

    fig.update_yaxes(range=[0, 100], gridcolor="#F1F5F9", zeroline=False)
    fig.update_xaxes(gridcolor="#F1F5F9", zeroline=False, tickangle=-30,
                     tickfont=dict(size=9))
    fig.update_layout(
        height=240,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def render_dashboard():
    stats = _get_stats()

    # Append to history (keep last 60 samples)
    if "resource_history" not in st.session_state:
        st.session_state.resource_history = []
    st.session_state.resource_history.append({
        "timestamp": stats["timestamp"],
        "cpu": stats["cpu"],
        "ram": stats["ram"],
    })
    st.session_state.resource_history = st.session_state.resource_history[-60:]

    # ── Header ────────────────────────────────────────────────────────
    col_title, col_refresh = st.columns([4, 1])
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
                Refresh to update.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if col_refresh.button("Refresh", type="secondary", use_container_width=True):
        st.rerun()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Gauges ────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)

    cpu_color  = "#E63946" if stats["cpu"]  > 80 else "#F4A261" if stats["cpu"]  > 60 else "#2D9E6B"
    ram_color  = "#E63946" if stats["ram"]  > 80 else "#F4A261" if stats["ram"]  > 60 else "#6366F1"
    disk_color = "#E63946" if stats["disk"] > 85 else "#F4A261" if stats["disk"] > 70 else "#0EA5E9"

    with g1:
        st.markdown('<div class="section-label">CPU</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["cpu"], "CPU Usage", cpu_color), use_container_width=True, key="g_cpu")
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["cpu_count"]} physical &nbsp;·&nbsp; {stats["cpu_count_logical"]} logical cores</p>',
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown('<div class="section-label">Memory</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["ram"], "RAM Usage", ram_color), use_container_width=True, key="g_ram")
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["ram_used_gb"]:.1f} GB used of {stats["ram_total_gb"]:.1f} GB</p>',
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown('<div class="section-label">Disk</div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(stats["disk"], "Disk Usage", disk_color), use_container_width=True, key="g_disk")
        st.markdown(
            f'<p style="text-align:center;color:#6B7280;font-size:0.78rem;margin-top:-12px;">'
            f'{stats["disk_used_gb"]:.1f} GB used of {stats["disk_total_gb"]:.1f} GB</p>',
            unsafe_allow_html=True,
        )

    # Temperature
    if stats.get("temp"):
        temp = stats["temp"]
        temp_color = "#E63946" if temp > 80 else "#F4A261" if temp > 65 else "#2D9E6B"
        st.markdown(
            f'<div style="background:white;border-radius:10px;padding:0.75rem 1.25rem;'
            f'display:inline-flex;align-items:center;gap:10px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #EEF0F2;">'
            f'<span style="font-size:1.2rem;">🌡️</span>'
            f'<span style="color:#6B7280;font-size:0.85rem;">CPU Temperature:</span>'
            f'<span style="color:{temp_color};font-weight:700;font-size:1rem;">{temp}°C</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Alerts
    if stats["cpu"] > 85:
        st.error("High CPU usage — this may slow down training.", icon="⚠️")
    if stats["ram"] > 85:
        st.error("High memory usage — consider closing other applications.", icon="⚠️")
    elif stats["ram"] > 75:
        st.warning("Memory usage above 75%.", icon="⚠️")

    st.divider()

    # ── Usage History Chart ────────────────────────────────────────────
    st.markdown('<div class="section-label">Usage History (last 60 samples)</div>', unsafe_allow_html=True)
    history = st.session_state.resource_history
    if len(history) >= 2:
        st.plotly_chart(_history_chart(history), use_container_width=True, key="history_chart")
    else:
        st.markdown(
            """
            <div style="background:white;border-radius:14px;padding:2.5rem;
                        text-align:center;border:1px solid #EEF0F2;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
                <div style="color:#94A3B8;font-size:0.85rem;">
                    History builds up as you refresh or use the app.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── System Info ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">System Information</div>', unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Operating System", stats["os"])
    i2.metric("Python Version", stats["python"])
    i3.metric("Architecture", stats["machine"])
    i4.metric("CPU Cores", f'{stats["cpu_count"]} / {stats["cpu_count_logical"]}',
              delta="physical / logical", delta_color="off")
