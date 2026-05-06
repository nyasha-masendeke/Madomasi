import platform
import psutil
import streamlit as st
from config import DISEASE_CLASSES, DISEASE_DISPLAY


def _sys_stats() -> dict:
    stats = {"cpu": psutil.cpu_percent(interval=0.3), "ram": psutil.virtual_memory().percent}
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            for name, entries in (temps or {}).items():
                for entry in entries:
                    if "cpu" in name.lower() or "core" in entry.label.lower():
                        stats["temp"] = round(entry.current, 1)
                        break
    except Exception:
        pass
    return stats


def _cpu_color(pct: float) -> str:
    if pct > 85:
        return "#E63946"
    if pct > 60:
        return "#F4A261"
    return "#2D9E6B"


def render_sidebar() -> dict:
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding: 1.2rem 0 1rem 0; border-bottom: 1px solid #1E3A5F; margin-bottom: 1.2rem;">
                <div style="font-size:1.5rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.5px;">
                    🍅 Tomato AI
                </div>
                <div style="font-size:0.75rem; color:#64748B; font-weight:500; margin-top:2px;">
                    DISEASE DIAGNOSTICS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Confidence Threshold ───────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;color:#64748B;margin-bottom:6px;">Confidence Threshold</p>',
            unsafe_allow_html=True,
        )
        confidence = st.slider(
            "confidence_slider",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0,
            label_visibility="collapsed",
            help="Minimum probability required to report a detection",
        )
        col_l, col_r = st.columns(2)
        col_l.markdown(
            f'<p style="color:#94A3B8;font-size:0.78rem;">Threshold</p>'
            f'<p style="color:#F8FAFC;font-weight:700;font-size:1.1rem;margin-top:-8px;">{confidence:.0f}%</p>',
            unsafe_allow_html=True,
        )
        col_r.markdown(
            f'<p style="color:#94A3B8;font-size:0.78rem;">Sensitivity</p>'
            f'<p style="color:#F8FAFC;font-weight:700;font-size:1.1rem;margin-top:-8px;">'
            f'{"High" if confidence < 40 else "Medium" if confidence < 70 else "Low"}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color:#1E3A5F;margin:1rem 0;'>", unsafe_allow_html=True)

        # ── Disease Filter ─────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;color:#64748B;margin-bottom:6px;">Disease Filter</p>',
            unsafe_allow_html=True,
        )
        selected_classes = st.multiselect(
            "disease_filter",
            options=DISEASE_CLASSES,
            default=DISEASE_CLASSES,
            format_func=lambda x: DISEASE_DISPLAY.get(x, x),
            label_visibility="collapsed",
        )

        if not selected_classes:
            st.warning("No classes selected — all detections will be hidden.")
        else:
            healthy = [c for c in selected_classes if "healthy" in c.lower()]
            diseased = [c for c in selected_classes if "healthy" not in c.lower()]
            pills_html = ""
            for c in healthy:
                pills_html += (
                    f'<span style="background:#065F46;color:#D1FAE5;padding:3px 10px;'
                    f'border-radius:12px;font-size:0.72rem;font-weight:600;'
                    f'display:inline-block;margin:2px;">{DISEASE_DISPLAY.get(c, c)}</span>'
                )
            for c in diseased:
                pills_html += (
                    f'<span style="background:#7F1D1D;color:#FEE2E2;padding:3px 10px;'
                    f'border-radius:12px;font-size:0.72rem;font-weight:600;'
                    f'display:inline-block;margin:2px;">{DISEASE_DISPLAY.get(c, c)}</span>'
                )
            st.markdown(
                f'<div style="margin-top:6px;line-height:2;">{pills_html}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#64748B;font-size:0.75rem;margin-top:6px;">'
                f'{len(selected_classes)} of {len(DISEASE_CLASSES)} classes active</p>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#1E3A5F;margin:1rem 0;'>", unsafe_allow_html=True)

        # ── ROI ────────────────────────────────────────────────────────
        x_min, y_min, x_max, y_max = 0.1, 0.1, 0.9, 0.9
        roi_pct = (x_max - x_min) * (y_max - y_min) * 100
        with st.expander("Region of Interest (ROI)", expanded=False):
            st.markdown(
                '<p style="color:#94A3B8;font-size:0.78rem;margin-bottom:8px;">'
                'Crop analysis to a specific leaf area</p>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            x_min = c1.slider("X Min", 0.0, 1.0, 0.1, 0.01, key="roi_x_min")
            y_min = c1.slider("Y Min", 0.0, 1.0, 0.1, 0.01, key="roi_y_min")
            x_max = c2.slider("X Max", 0.0, 1.0, 0.9, 0.01, key="roi_x_max")
            y_max = c2.slider("Y Max", 0.0, 1.0, 0.9, 0.01, key="roi_y_max")
            if x_min >= x_max or y_min >= y_max:
                st.error("Min values must be less than Max values.")
            else:
                roi_pct = (x_max - x_min) * (y_max - y_min) * 100
                st.markdown(
                    f'<p style="color:#94A3B8;font-size:0.78rem;">Coverage: '
                    f'<strong style="color:#F8FAFC;">{roi_pct:.1f}%</strong></p>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='border-color:#1E3A5F;margin:1rem 0;'>", unsafe_allow_html=True)

        # ── System Health ──────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;color:#64748B;margin-bottom:10px;">System Health</p>',
            unsafe_allow_html=True,
        )
        stats = _sys_stats()

        cpu_color = _cpu_color(stats["cpu"])
        ram_color = _cpu_color(stats["ram"])

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            f'<span style="color:#94A3B8;font-size:0.78rem;">CPU</span>'
            f'<span style="color:{cpu_color};font-size:0.78rem;font-weight:600;">{stats["cpu"]:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(stats["cpu"] / 100)

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;margin-top:8px;">'
            f'<span style="color:#94A3B8;font-size:0.78rem;">RAM</span>'
            f'<span style="color:{ram_color};font-size:0.78rem;font-weight:600;">{stats["ram"]:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(stats["ram"] / 100)

        if stats.get("temp"):
            temp_color = "#E63946" if stats["temp"] > 80 else "#F4A261" if stats["temp"] > 65 else "#2D9E6B"
            st.markdown(
                f'<p style="color:#94A3B8;font-size:0.78rem;margin-top:8px;">'
                f'Temp: <strong style="color:{temp_color};">{stats["temp"]}°C</strong></p>',
                unsafe_allow_html=True,
            )

        if stats["ram"] > 90:
            st.error("Critical: High memory usage")
        elif stats["ram"] > 75:
            st.warning("Memory usage above 75%")

        st.markdown(
            f'<p style="color:#3D5166;font-size:0.72rem;margin-top:8px;">'
            f'{platform.system()} &nbsp;·&nbsp; {psutil.cpu_count()} cores</p>',
            unsafe_allow_html=True,
        )

    return {
        "confidence": confidence,
        "selected_classes": selected_classes,
        "roi": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }
