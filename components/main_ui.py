import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

from components.system_dashboard import render_dashboard, record_resource_sample
import config as _config_module
from config import DISEASE_CLASSES, DISEASE_DISPLAY, DISEASE_SEVERITY

PROJECT_ROOT = Path(_config_module.__file__).parent

from pipeline import (
    predict_image, evaluate_model,
    extract_features, train_head, fine_tune_model, split_dataset, log_inference,
)
from src.utils.recommendations import get_recommendation
from streamlit_callback import StreamlitTrainCallback


# =============================================================================
# DATA HELPERS
# =============================================================================

def _load_latest_history() -> dict:
    import json as _json
    # Resolve to E:\Madomasi\outputs regardless of which worktree the server runs from
    main_outputs = Path(__file__).parents[4] / "outputs"
    base = main_outputs if main_outputs.exists() else Path("outputs")
    if not base.exists():
        return {}
    dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("Training") and p.name[8:].isdigit()],
        key=lambda p: int(p.name[8:]), reverse=True,
    )
    for td in dirs:
        files = sorted(td.glob("history_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            try:
                return _json.loads(files[0].read_text())
            except Exception:
                continue
    return {}


def _find_models() -> list:
    base = Path("models/trained")
    if not base.exists():
        return []
    return sorted([str(p) for p in base.rglob("*.keras")],
                  key=lambda p: Path(p).stat().st_mtime, reverse=True)


def _model_picker(label: str, key: str, default: str = "models/trained/latest.keras") -> str:
    available = _find_models()
    CUSTOM = "Custom path..."
    if available:
        choice = st.selectbox(label, options=available + [CUSTOM],
                              format_func=lambda x: Path(x).name if x != CUSTOM else CUSTOM,
                              key=f"{key}_select")
        if choice == CUSTOM:
            return st.text_input("Custom model path", "", key=f"{key}_custom",
                                 placeholder="models/trained/run_xxx/stage2_ft.keras")
        return choice
    detected = _detect_dir([default, "models/trained/latest.keras"], fallback="")
    return st.text_input(label, detected, placeholder="models/trained/run_xxx/stage2_ft.keras", key=key)


def _detect_dir(candidates: list, fallback: str = "") -> str:
    for p in candidates:
        if p and Path(p).exists():
            return p
    return fallback


def _dir_input(label: str, candidates: list, fallback: str, key: str, help: str = "") -> str:
    default = _detect_dir(candidates, fallback)
    value = st.text_input(label, value=default, key=key, help=help)
    if default and Path(default).exists():
        st.caption(f"Auto-detected: `{default}`")
    elif not value:
        st.caption("Not found — enter path manually.")
    return value


def _model_info(path: str) -> str:
    """Return a one-line HTML string with model file metadata."""
    if not path or not Path(path).exists():
        return ""
    p = Path(path)
    size_kb = p.stat().st_size / 1024
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d %Y")
    size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    return (f'<span style="font-size:0.72rem;color:#374151;">'
            f'&#128196; {p.name} &nbsp;·&nbsp; {size_str} &nbsp;·&nbsp; {mtime}</span>')


# =============================================================================
# UI HELPERS
# =============================================================================

def _html(html: str):
    """Render HTML via st.markdown, collapsing whitespace so the markdown
    parser never sees a blank line inside an HTML block (which would restart
    paragraph mode and escape inner tags)."""
    import re
    compact = re.sub(r'\n\s*\n', '\n', html).strip()
    st.markdown(compact, unsafe_allow_html=True)


def _model_status_html(path: str) -> str:
    exists = bool(path) and Path(path).exists()
    dot = "&#9679;"
    if exists:
        return f'<span class="model-status-ok">{dot} Model Ready</span>'
    return f'<span class="model-status-missing">{dot} Model Not Found</span>'


def _hero():
    scans = st.session_state.get("scans", 0)
    diseases = st.session_state.get("diseases", 0)
    healthy = scans - diseases
    healthy_rate = f"{healthy / scans * 100:.0f}%" if scans else "—"
    _html(f"""
    <div style="
        background:linear-gradient(135deg,#0F1623 0%,#1A0A0C 50%,#0D0408 100%);
        border-radius:20px; padding:2.5rem 3rem 2rem;
        margin-bottom:1.75rem; position:relative; overflow:hidden;
        border:1px solid rgba(230,57,70,0.12);
        box-shadow:0 0 80px rgba(230,57,70,0.06),0 1px 0 rgba(255,255,255,0.04) inset;
    ">
        <div style="position:absolute;top:-80px;right:-60px;width:320px;height:320px;
                    background:radial-gradient(circle,rgba(230,57,70,0.12) 0%,transparent 70%);pointer-events:none;"></div>
        <div style="position:absolute;bottom:-60px;right:200px;width:200px;height:200px;
                    background:radial-gradient(circle,rgba(99,102,241,0.07) 0%,transparent 70%);pointer-events:none;"></div>

        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1.5rem;">
            <div>
                <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.6rem;">
                    <span style="font-size:0.65rem;font-weight:700;letter-spacing:2.5px;
                                 color:rgba(230,57,70,0.7);text-transform:uppercase;">
                        AI-Powered Plant Health
                    </span>
                </div>
                <div style="font-size:2.4rem;font-weight:900;color:#FFFFFF;
                            letter-spacing:-1.5px;line-height:1.1;margin-bottom:0.5rem;">
                    Madomasi
                </div>
                <div style="font-size:0.9rem;color:rgba(255,255,255,0.4);font-weight:400;">
                    MobileNetV3 transfer learning &nbsp;·&nbsp; 10 disease classes &nbsp;·&nbsp; Real-time inference
                </div>
            </div>

            <div style="display:flex;gap:2.5rem;flex-wrap:wrap;padding-top:0.25rem;">
                <div>
                    <div style="font-size:2rem;font-weight:900;color:#FFFFFF;letter-spacing:-1px;">{scans}</div>
                    <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);font-weight:600;
                                text-transform:uppercase;letter-spacing:1px;margin-top:2px;">Leaves Scanned</div>
                </div>
                <div>
                    <div style="font-size:2rem;font-weight:900;color:#F87171;letter-spacing:-1px;">{diseases}</div>
                    <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);font-weight:600;
                                text-transform:uppercase;letter-spacing:1px;margin-top:2px;">Diseases Found</div>
                </div>
                <div>
                    <div style="font-size:2rem;font-weight:900;color:#34D399;letter-spacing:-1px;">{healthy_rate}</div>
                    <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);font-weight:600;
                                text-transform:uppercase;letter-spacing:1px;margin-top:2px;">Healthy Rate</div>
                </div>
            </div>
        </div>

        <div style="margin-top:1.5rem;height:1px;background:linear-gradient(90deg,rgba(230,57,70,0.3),rgba(99,102,241,0.2),transparent);"></div>
        <div style="margin-top:0.75rem;font-size:0.72rem;color:rgba(255,255,255,0.2);">
            Session started {datetime.now().strftime("%b %d, %Y %H:%M")} &nbsp;·&nbsp; MobileNetV3Small backbone
        </div>
    </div>
    """)


def _prob_bars_html(all_probs: dict, top_disease: str) -> str:
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:8]
    rows = []
    for cls, prob in sorted_probs:
        label = DISEASE_DISPLAY.get(cls, cls)
        is_active = cls == top_disease
        is_healthy = "healthy" in cls.lower()
        fill_class = "prob-fill-h" if is_healthy else ("prob-fill-d" if is_active else "prob-fill-n")
        label_class = "prob-label prob-label-active" if is_active else "prob-label"
        w = f"{prob * 100:.1f}%"
        rows.append(
            f'<div class="prob-row">'
            f'  <div class="{label_class}" title="{label}">{label}</div>'
            f'  <div class="prob-track"><div class="{fill_class}" style="width:{w}"></div></div>'
            f'  <div class="prob-pct">{prob:.1%}</div>'
            f'</div>'
        )
    return (
        '<div class="prob-section">'
        '<div class="prob-section-title">Class Probabilities</div>'
        + "".join(rows) +
        '</div>'
    )


def _diagnosis_card(result: dict):
    disease = result["disease"]
    display = DISEASE_DISPLAY.get(disease, disease)
    conf = result["confidence"]
    passes = result["passes_threshold"]
    is_healthy = "healthy" in disease.lower()
    severity_label, severity_class = DISEASE_SEVERITY.get(disease, ("Unknown", "moderate"))

    card_class = "result-card-healthy" if is_healthy else "result-card-diseased"
    conf_class = "conf-big-healthy" if is_healthy else ("conf-big-diseased" if passes else "conf-big-low")
    icon = "✓" if is_healthy else "⚠"
    conf_bar_color = "#34D399" if is_healthy else ("#F87171" if passes else "#FBBF24")
    status_text = "Plant is healthy — no treatment required." if is_healthy else "Disease detected — see recommendations below."

    _html(f"""
    <div class="result-card {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.25rem;">
            <div>
                <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:2px;color:#374151;margin-bottom:0.4rem;">Diagnosis Result</div>
                <div style="font-size:1.55rem;font-weight:800;color:#F1F5F9;line-height:1.15;">
                    {icon} {display}
                </div>
                <div style="font-size:0.8rem;color:#4B5563;margin-top:0.3rem;">{status_text}</div>
            </div>
            <span class="badge-{severity_class}">{severity_label}</span>
        </div>

        <div style="display:flex;gap:2rem;align-items:flex-end;margin-bottom:1rem;">
            <div>
                <div class="conf-big {conf_class}">{conf:.0%}</div>
                <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:1.2px;color:#374151;margin-top:3px;">Confidence</div>
            </div>
            <div style="flex:1;padding-bottom:0.4rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#374151;margin-bottom:5px;">
                    <span>Threshold</span>
                    <span style="color:{"#34D399" if passes else "#FBBF24"};font-weight:700;">
                        {"Pass ✓" if passes else "Below ✗"}
                    </span>
                </div>
                <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;">
                    <div style="width:{conf * 100:.1f}%;height:100%;
                                background:{conf_bar_color};border-radius:6px;
                                transition:width 0.8s cubic-bezier(0.4,0,0.2,1);"></div>
                </div>
            </div>
        </div>

        {_prob_bars_html(result["all_probs"], disease)}
    </div>
    """)

    if not passes:
        st.warning("⚠ Confidence is below your threshold — try a clearer, well-lit image.", icon=None)

    with st.expander("Treatment & Recommendations", expanded=not is_healthy):
        rec = get_recommendation(disease)
        st.markdown(
            f'<div class="treatment-card">'
            f'<div class="treatment-label">Recommended Action</div>'
            f'{rec}'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# CHARTS
# =============================================================================

def plot_learning_curves(history: dict):
    if not history or not history.get("epoch"):
        return
    df = pd.DataFrame(history).dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"], mode="lines+markers",
                             name="Train", line=dict(color="#34D399", width=2.5),
                             marker=dict(size=5)), row=1, col=1)
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers",
                                 name="Val", line=dict(color="#F4A261", width=2.5, dash="dash"),
                                 marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#F87171", width=2.5),
                             marker=dict(size=5), showlegend=False), row=1, col=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers",
                                 name="Val Loss", line=dict(color="#818CF8", width=2.5, dash="dash"),
                                 marker=dict(size=5), showlegend=False), row=1, col=2)
    fig.update_layout(
        height=340, template="plotly_dark", hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#94A3B8"),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, key="learning_curves")


def plot_inference_charts(log: list):
    if not log:
        return
    df = pd.DataFrame(log)
    df["Confidence_f"] = df["Confidence"].str.rstrip("%").astype(float) / 100
    df["Index"] = range(1, len(df) + 1)
    df["IsDisease"] = ~df["Disease"].str.lower().str.contains("healthy")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Confidence per Scan", "Disease vs Healthy"),
        horizontal_spacing=0.14,
    )
    colors = ["#F87171" if d else "#34D399" for d in df["IsDisease"]]
    fig.add_trace(go.Bar(x=df["Index"], y=df["Confidence_f"], name="Confidence",
                         marker_color=colors, hovertemplate="%{y:.1%}<extra></extra>"), row=1, col=1)

    disease_count = df["IsDisease"].sum()
    healthy_count = len(df) - disease_count
    fig.add_trace(go.Pie(
        labels=["Disease", "Healthy"],
        values=[disease_count, healthy_count],
        marker=dict(colors=["#F87171", "#34D399"],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        hole=0.55,
        textinfo="percent+label",
        textfont=dict(color="#94A3B8", size=11),
        hovertemplate="%{label}: %{value} scans<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        height=320, template="plotly_dark", showlegend=False,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#94A3B8"),
    )
    fig.update_yaxes(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    fig.update_xaxes(title_text="Scan #", gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, key="inference_charts")


# =============================================================================
# INFERENCE HELPERS
# =============================================================================

@st.cache_resource
def load_cached_model(path: str):
    return tf.keras.models.load_model(path)




def _run_inference(img_bytes_or_file, model_path: str, confidence: float) -> dict | None:
    """Run prediction and update session state. Returns result dict or None on error."""
    from io import BytesIO
    try:
        load_cached_model(model_path)
        src = BytesIO(img_bytes_or_file) if isinstance(img_bytes_or_file, bytes) else img_bytes_or_file
        result = predict_image(model_path, src, confidence / 100)
        record_resource_sample("Inference")
        log_inference(result["disease"], result["confidence"], Path(model_path).name)
        st.session_state.scans = st.session_state.get("scans", 0) + 1
        if "healthy" not in result["disease"].lower():
            st.session_state.diseases = st.session_state.get("diseases", 0) + 1
        st.session_state.setdefault("diagnosis_log", []).append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Disease": DISEASE_DISPLAY.get(result["disease"], result["disease"]),
            "Confidence": f"{result['confidence']:.1%}",
            "Status": "Pass" if result.get("passes_threshold") else "Low",
            "Model": Path(model_path).name,
        })
        return result
    except FileNotFoundError:
        st.error("Model file not found. Train a model or enter a valid path.")
    except Exception as e:
        st.error(f"Inference failed: {e}")
    return None


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def _save_learning_curves(history: dict, model_path: str = "", output_dir: Path | None = None) -> str:
    import json as _json
    from streamlit_callback import _next_training_output_dir
    out_dir = output_dir if output_dir else _next_training_output_dir()
    stem = Path(model_path).stem if model_path else "run"
    json_path = out_dir / f"history_{stem}.json"
    json_path.write_text(_json.dumps(history))
    html_path = out_dir / f"curves_{stem}.html"
    df = pd.DataFrame(history).dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"], mode="lines+markers",
                             name="Train Acc", line=dict(color="#34D399", width=2.5)), row=1, col=1)
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers",
                                 name="Val Acc", line=dict(color="#F4A261", width=2.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#F87171", width=2.5)), row=1, col=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers",
                                 name="Val Loss", line=dict(color="#818CF8", width=2.5, dash="dash")), row=1, col=2)
    fig.update_layout(height=420, template="plotly_white", hovermode="x unified",
                      margin=dict(l=30, r=20, t=60, b=30))
    fig.write_html(str(html_path))
    return str(html_path)


# =============================================================================
# MAIN APP
# =============================================================================

def _clear_stale_dir_keys():
    dir_keys = {
        "shared_data_dir":  ("data",),
        "shared_model_base":("models",),
        "raw_source":       ("data",),
        "split_output":     ("data",),
        "feat_out":         ("data",),
        "eval_test_dir":    ("data",),
    }
    for key, prefixes in dir_keys.items():
        val = st.session_state.get(key, "")
        if val and not any(val.startswith(p) for p in prefixes):
            del st.session_state[key]


def run_app():
    os.chdir(PROJECT_ROOT)
    _clear_stale_dir_keys()
    _hero()

    tab_inference, tab_training, tab_system = st.tabs([
        "  Inference  ",
        "  Training  ",
        "  System  ",
    ])

    # =========================================================================
    # INFERENCE TAB
    # =========================================================================
    with tab_inference:

        # ── Settings bar (always visible, no expander) ────────────────────
        with st.container():
            s1, s2 = st.columns([1, 2])
            with s1:
                confidence = st.slider(
                    "Confidence threshold (%)", 0.0, 100.0, 50.0, 1.0,
                    help="Minimum probability required to accept a prediction",
                )
            with s2:
                selected_classes = st.multiselect(
                    "Disease filter",
                    options=DISEASE_CLASSES, default=DISEASE_CLASSES,
                    format_func=lambda x: DISEASE_DISPLAY.get(x, x),
                )
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown('<div class="section-label">Upload Leaf Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "upload", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if uploaded:
                img_bytes = uploaded.read()
                uploaded.seek(0)
                size_kb = len(img_bytes) / 1024
                st.image(uploaded, use_container_width=True)
                st.markdown(
                    f'<div class="img-meta">'
                    f'<span>&#128190; {uploaded.name}</span>'
                    f'<span>{size_kb:.0f} KB</span>'
                    f'<span>{uploaded.type}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if size_kb < 5:
                    st.warning("Image is very small — prediction quality may be low.")
            else:
                st.session_state.pop("last_static_result", None)
                _html("""
                <div class="empty-state">
                    <span class="empty-state-icon">🌿</span>
                    <div class="empty-state-title">No image selected</div>
                    <div class="empty-state-sub">JPG or PNG · up to 200 MB</div>
                </div>""")

        with col_result:
            st.markdown('<div class="section-label">Model & Diagnosis</div>', unsafe_allow_html=True)
            model_path = _model_picker("Model", key="img_model")
            c_status, c_info = st.columns([1, 2])
            c_status.markdown(_model_status_html(model_path), unsafe_allow_html=True)
            info_html = _model_info(model_path)
            if info_html:
                c_info.markdown(info_html, unsafe_allow_html=True)

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            run_btn = st.button(
                "Run Diagnosis", type="primary",
                use_container_width=True, disabled=uploaded is None,
            )

            if run_btn and uploaded:
                if not model_path or not Path(model_path).exists():
                    st.error("No model found. Train a model first or enter a valid path.")
                else:
                    with st.spinner("Analysing leaf…"):
                        uploaded.seek(0)
                        result = _run_inference(uploaded, model_path, confidence)
                        if result:
                            st.session_state["last_static_result"] = result
                            st.rerun()

            last_result = st.session_state.get("last_static_result")
            if last_result and uploaded is not None:
                _diagnosis_card(last_result)
            elif uploaded is not None:
                _html("""
                <div class="empty-state" style="margin-top:1rem;">
                    <span class="empty-state-icon">🔬</span>
                    <div class="empty-state-title">Ready to analyse</div>
                    <div class="empty-state-sub">Click Run Diagnosis to classify this leaf</div>
                </div>""")

        # ── Inference Analytics ───────────────────────────────────────────
        log = st.session_state.get("diagnosis_log", [])
        if log:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.divider()

            h1, h2 = st.columns([3, 1])
            h1.subheader("Session Analytics")

            with h2:
                log_df = pd.DataFrame(log[::-1])
                csv_bytes = log_df.to_csv(index=False).encode()
                st.download_button(
                    "Export CSV", csv_bytes,
                    file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", type="secondary", use_container_width=True,
                )

            plot_inference_charts(log)

            with st.expander(f"Session Log — {len(log)} scans"):
                st.dataframe(log_df, use_container_width=True, hide_index=True)
                if st.button("Clear Session", type="secondary"):
                    st.session_state.diagnosis_log = []
                    st.session_state.pop("last_static_result", None)
                    st.session_state.scans = 0
                    st.session_state.diseases = 0
                    st.rerun()

    # =========================================================================
    # TRAINING TAB
    # =========================================================================
    with tab_training:
        if "shared_data_dir" not in st.session_state:
            st.session_state["shared_data_dir"] = _detect_dir([
                st.session_state.get("split_train_dir", ""),
                "data/splits/train", "data/raw/raw/tomato", "data/raw",
            ])
        if "shared_model_base" not in st.session_state:
            st.session_state["shared_model_base"] = _detect_dir(["models/trained"], fallback="")
        if "features_dir" not in st.session_state:
            detected = _detect_dir(["data/features"], fallback="")
            if detected:
                st.session_state["features_dir"] = detected
        if "head_full_path" not in st.session_state:
            for c in ["models/trained/head_full.keras", "models/trained/latest.keras"]:
                if Path(c).exists():
                    st.session_state["head_full_path"] = c
                    break

        # ── Shared config bar ─────────────────────────────────────────────
        cfg1, cfg2, cfg3 = st.columns([2, 2, 1])
        data_dir = cfg1.text_input(
            "Dataset directory", placeholder="data/splits/train", key="shared_data_dir",
        )
        model_save_base = cfg2.text_input(
            "Model output", placeholder="models/trained", key="shared_model_base",
        )
        batch_size = cfg3.select_slider(
            "Batch size", options=[4, 8, 16, 32, 64], value=16,
        )

        col_left, col_right = st.columns([2, 3], gap="large")

        # ── Left: pipeline stages ─────────────────────────────────────────
        # (rendered below after col_right is defined)

        # ── Right: live training output ───────────────────────────────────
        with col_right:
            st.subheader("Live Training Curves")
            progress_bar        = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            if not st.session_state.get("training_active"):
                hist = _load_latest_history() or st.session_state.get("train_history")
                if hist and hist.get("epoch"):
                    st.session_state["train_history"] = hist

                    def _last(seq):
                        vals = [v for v in (seq or []) if v is not None]
                        return vals[-1] if vals else None

                    with metrics_placeholder:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Epoch",     hist["epoch"][-1])
                        m2.metric("Train Acc", f"{_last(hist.get('accuracy')):.3f}"    if _last(hist.get('accuracy'))    is not None else "—")
                        m3.metric("Val Acc",   f"{_last(hist.get('val_accuracy')):.3f}" if _last(hist.get('val_accuracy')) is not None else "—")
                        m4.metric("Val Loss",  f"{_last(hist.get('val_loss')):.4f}"     if _last(hist.get('val_loss'))     is not None else "—")

                    with chart_placeholder:
                        plot_learning_curves(hist)
                else:
                    with chart_placeholder:
                        st.markdown(
                            '<div class="empty-state">'
                            '<span class="empty-state-icon">📈</span>'
                            '<div class="empty-state-title">No training history yet</div>'
                            '<div class="empty-state-sub">Run a training stage — live curves appear here</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

        # ── Left: pipeline stages ────────────────────────────────────────
        with col_left:
            split_done = bool(st.session_state.get("split_train_dir"))
            feat_done  = bool(st.session_state.get("features_dir"))
            head_done  = bool(st.session_state.get("head_full_path"))
            ft_done    = bool(st.session_state.get("fine_tuned_path"))

            # ── Stage 0 — Split
            with st.expander(
                f"{'✅' if split_done else '○'} Stage 0 — Split Dataset",
                expanded=not split_done,
            ):
                if "raw_source" not in st.session_state:
                    st.session_state["raw_source"] = _detect_dir([
                        "data/raw/raw/tomato", "data/raw/tomato", "data/raw", "dataset",
                    ])
                if "split_output" not in st.session_state:
                    st.session_state["split_output"] = _detect_dir(["data/splits"], fallback="")
                raw_source   = st.text_input("Raw dataset dir", placeholder="path/to/raw", key="raw_source")
                split_output = st.text_input("Splits output dir", placeholder="data/splits", key="split_output")
                r1, r2, r3 = st.columns(3)
                train_ratio = r1.number_input("Train %", 0, 100, 70, step=5, key="train_ratio") / 100
                val_ratio   = r2.number_input("Val %",   0, 100, 20, step=5, key="val_ratio")   / 100
                test_ratio  = r3.number_input("Test %",  0, 100, 10, step=5, key="test_ratio")  / 100
                ratio_ok = abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
                if not ratio_ok:
                    st.warning("Train + Val + Test must sum to 100%.")
                btn_split = st.button(
                    "Split Dataset", type="primary", use_container_width=True,
                    disabled=not ratio_ok or st.session_state.get("training_active", False),
                    key="btn_split",
                )
                if split_done:
                    st.caption(f"Train: `{st.session_state.split_train_dir}`")

            # ── Stage 1 — Feature Extraction
            with st.expander(
                f"{'✅' if feat_done else '○'} Stage 1 — Feature Extraction",
                expanded=split_done and not feat_done,
            ):
                _src_opts = list(dict.fromkeys([
                    p for p in [
                        st.session_state.get("split_train_dir", ""),
                        "data/splits/train", "data/raw/raw/tomato", data_dir,
                    ] if p and Path(p).exists()
                ]))
                _CUSTOM_SRC = "Custom path..."
                if _src_opts:
                    _choice = st.selectbox(
                        "Source data", options=_src_opts + [_CUSTOM_SRC],
                        format_func=lambda x: x if x == _CUSTOM_SRC else Path(x).as_posix(),
                        key="feat_src_select",
                    )
                    feat_src_dir = (
                        st.text_input("Custom source dir", placeholder="data/splits/train", key="feat_src_custom")
                        if _choice == _CUSTOM_SRC else _choice
                    )
                else:
                    feat_src_dir = st.text_input("Source data dir", placeholder="data/splits/train", key="feat_src_custom")

                if "feat_out" not in st.session_state:
                    st.session_state["feat_out"] = _detect_dir(["data/features"], fallback="")
                feat_out  = st.text_input("Features output dir", placeholder="data/features", key="feat_out")
                feat_batch = st.select_slider("Batch size", options=[4, 8, 16, 32, 64], value=batch_size, key="feat_batch_size")
                btn_extract = st.button(
                    "Extract Features", type="primary", use_container_width=True,
                    disabled=st.session_state.get("training_active", False), key="btn_extract",
                )
                if feat_done:
                    st.caption(f"Features: `{st.session_state.features_dir}`")

            # ── Stage 2 — Train Head
            with st.expander(
                f"{'✅' if head_done else '○'} Stage 2 — Train Head",
                expanded=feat_done and not head_done,
            ):
                head_epochs = st.slider("Epochs", 1, 50, 10, key="head_epochs")
                head_lr     = st.select_slider("Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001, key="head_lr")
                btn_head = st.button(
                    "Train Head", type="primary", use_container_width=True,
                    disabled=not feat_done or st.session_state.get("training_active", False),
                    key="btn_head",
                )
                if not feat_done:
                    st.caption("Complete Feature Extraction first.")
                elif head_done:
                    st.caption(f"Model: `{st.session_state.head_full_path}`")

            # ── Stage 3 — Fine-Tuning
            with st.expander(
                f"{'✅' if ft_done else '○'} Stage 3 — Fine-Tuning",
                expanded=head_done and not ft_done,
            ):
                ft_epochs = st.slider("Epochs", 1, 50, 10, key="ft_epochs")
                ft_lr     = st.select_slider("Learning rate", [0.000001, 0.00001, 0.0001], value=0.00001, key="ft_lr")
                btn_ft = st.button(
                    "Fine-Tune", type="primary", use_container_width=True,
                    disabled=not head_done or st.session_state.get("training_active", False),
                    key="btn_ft",
                )
                if not head_done:
                    st.caption("Train the Head first.")
                elif ft_done:
                    st.caption(f"Model: `{st.session_state.fine_tuned_path}`")

        # ── Stage execution ───────────────────────────────────────────────
        if btn_split:
            if not raw_source:
                st.error("Enter the raw dataset directory before splitting.")
            elif not Path(raw_source).exists():
                st.error(f"Directory not found: `{raw_source}`")
            else:
                st.session_state.update({
                    "training_active": True, "_active_stage": "split",
                    "_raw_source": raw_source, "_split_output": split_output or "data/splits",
                    "_train_ratio": train_ratio, "_val_ratio": val_ratio, "_test_ratio": test_ratio,
                })
                st.rerun()

        if btn_extract:
            _ext_data = feat_src_dir or st.session_state.get("split_train_dir") or data_dir
            if not _ext_data:
                st.error("Select a source data directory.")
            elif not Path(_ext_data).exists():
                st.error(f"Directory not found: `{_ext_data}`")
            else:
                st.session_state.update({
                    "training_active": True, "_active_stage": "extract",
                    "_data_dir": _ext_data, "_feat_out": feat_out or "data/features",
                    "_batch_size": feat_batch,
                })
                st.rerun()

        if btn_head:
            _feat_dir = st.session_state.get("features_dir")
            if not _feat_dir or not Path(_feat_dir).exists():
                st.error("Features directory not found. Run Feature Extraction first.")
            else:
                st.session_state.update({
                    "training_active": True, "_active_stage": "head",
                    "_head_epochs": head_epochs, "_head_lr": head_lr,
                    "_model_save_base": model_save_base or "models/trained/latest.keras",
                })
                st.rerun()

        if btn_ft:
            _head_path = st.session_state.get("head_full_path")
            _ft_data   = data_dir or st.session_state.get("split_train_dir", "")
            if not _head_path or not Path(_head_path).exists():
                st.error("Head model not found. Complete Stage 2 first.")
            elif not _ft_data or not Path(_ft_data).exists():
                st.error("Enter a valid dataset directory for fine-tuning.")
            else:
                st.session_state.update({
                    "training_active": True, "_active_stage": "finetune",
                    "_ft_epochs": ft_epochs, "_ft_lr": ft_lr,
                    "_batch_size": batch_size, "_data_dir": _ft_data,
                })
                st.rerun()

        # ── Active training ───────────────────────────────────────────────
        if st.session_state.get("training_active", False):
            stage = st.session_state.get("_active_stage", "")
            record_resource_sample("Training")
            progress_bar        = st.progress(0, text="Initialising…")
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            try:
                if stage == "split":
                    def _split_prog(current, total, class_name):
                        progress_bar.progress(current / max(total, 1),
                                              text=f"Splitting {current}/{total}: {class_name}")
                    with st.spinner("Splitting dataset into train / val / test…"):
                        result = split_dataset(
                            source_dir  = st.session_state["_raw_source"],
                            output_dir  = st.session_state["_split_output"],
                            train_ratio = st.session_state["_train_ratio"],
                            val_ratio   = st.session_state["_val_ratio"],
                            test_ratio  = st.session_state["_test_ratio"],
                            progress_fn = _split_prog,
                        )
                    st.session_state.split_train_dir = result["train_dir"]
                    st.session_state.split_val_dir   = result["val_dir"]
                    st.session_state.split_test_dir  = result["test_dir"]
                    st.session_state.pop("shared_data_dir", None)
                    progress_bar.progress(1.0, text="Split complete")
                    c = result["counts"]
                    st.success(
                        f"{result['num_classes']} classes — "
                        f"train: {c['train']} · val: {c['val']} · test: {c['test']} images"
                    )

                elif stage == "extract":
                    def _ext_prog(current, total):
                        progress_bar.progress(min(current / max(total, 1), 1.0),
                                              text=f"Extracting batch {current}/{total}")
                    with st.spinner("Extracting MobileNetV3 features…"):
                        result = extract_features(
                            data_dir    = st.session_state["_data_dir"],
                            output_dir  = st.session_state["_feat_out"],
                            batch_size  = st.session_state["_batch_size"],
                            progress_fn = _ext_prog,
                        )
                    st.session_state.features_dir = result["features_dir"]
                    progress_bar.progress(1.0, text="Extraction complete")
                    st.success(
                        f"{result['num_samples']} samples · {result['num_classes']} classes · "
                        f"shape {result['feature_shape']} → `{result['features_dir']}`"
                    )

                elif stage == "head":
                    cb = StreamlitTrainCallback(
                        progress_bar, metrics_placeholder, chart_placeholder, stage_label="Train Head",
                    )
                    with st.spinner("Training classification head…"):
                        result = train_head(
                            features_dir = st.session_state["features_dir"],
                            epochs       = st.session_state["_head_epochs"],
                            lr           = st.session_state["_head_lr"],
                            output_path  = st.session_state["_model_save_base"],
                            callbacks    = [cb],
                        )
                    st.session_state.head_full_path = result["full_model_path"]
                    st.session_state.train_history  = cb.history
                    record_resource_sample("Training")
                    _save_learning_curves(cb.history, result["full_model_path"], output_dir=cb.output_dir)
                    st.success(f"Head trained → `{result['full_model_path']}`")

                elif stage == "finetune":
                    cb = StreamlitTrainCallback(
                        progress_bar, metrics_placeholder, chart_placeholder, stage_label="Fine-Tuning",
                    )
                    with st.spinner("Fine-tuning end-to-end…"):
                        result = fine_tune_model(
                            model_path = st.session_state["head_full_path"],
                            data_dir   = st.session_state["_data_dir"],
                            batch_size = st.session_state["_batch_size"],
                            epochs     = st.session_state["_ft_epochs"],
                            lr         = st.session_state["_ft_lr"],
                            callbacks  = [cb],
                        )
                    st.session_state.fine_tuned_path = result["path"]
                    st.session_state.train_history   = cb.history
                    record_resource_sample("Training")
                    _save_learning_curves(cb.history, result["path"], output_dir=cb.output_dir)
                    st.success(f"Fine-tuned model → `{result['path']}`")

            except Exception as e:
                st.error(f"Stage failed: {e}")
            finally:
                st.session_state.training_active = False
                st.rerun()

        # ── Evaluation ────────────────────────────────────────────────────
        eval_done = bool(st.session_state.get("eval_result"))
        with st.expander(f"{'✅' if eval_done else '○'} Model Evaluation", expanded=False):
            ev_c1, ev_c2 = st.columns(2)
            with ev_c1:
                eval_model_path = _model_picker("Model to evaluate", key="eval_model")
                if "eval_test_dir" not in st.session_state:
                    st.session_state["eval_test_dir"] = _detect_dir([
                        st.session_state.get("split_test_dir", ""),
                        "data/splits/test", "data/splits/val",
                    ])
                eval_test_dir = st.text_input("Test directory", placeholder="data/splits/test", key="eval_test_dir")
                eval_btn = st.button("Evaluate", type="primary", use_container_width=True)

            with ev_c2:
                if eval_btn:
                    if not Path(eval_model_path).exists():
                        st.error("Model file not found.")
                    elif not Path(eval_test_dir).exists():
                        st.error("Test directory not found.")
                    else:
                        with st.spinner("Evaluating…"):
                            try:
                                ev = evaluate_model(eval_model_path, eval_test_dir)
                                st.session_state.eval_result     = ev
                                st.session_state.eval_model_name = Path(eval_model_path).name
                            except Exception as e:
                                st.error(f"Evaluation failed: {e}")

                if st.session_state.get("eval_result"):
                    ev  = st.session_state.eval_result
                    acc = ev["accuracy"]
                    m1, m2 = st.columns(2)
                    m1.metric("Test Accuracy", f"{acc:.2%}")
                    m2.metric("Test Loss", f"{ev['loss']:.4f}")
                    verdict = (
                        "Excellent — ready for deployment." if acc >= 0.90 else
                        "Good — consider more fine-tuning." if acc >= 0.75 else
                        "Needs improvement — increase epochs or data."
                    )
                    st.caption(verdict)

        # ── TFLite Export ─────────────────────────────────────────────────
        tfl_done = bool(st.session_state.get("tfl_exported"))
        with st.expander(f"{'✅' if tfl_done else '○'} Export for Raspberry Pi (TFLite)", expanded=False):
            tfl_c1, tfl_c2 = st.columns(2)
            with tfl_c1:
                tfl_model = _model_picker("Model to export", key="tfl_model")
                tfl_out   = st.text_input("Output path", value="models/trained/model.tflite", key="tfl_out_path")
                tfl_btn   = st.button("Export TFLite (INT8)", type="primary", use_container_width=True)
            with tfl_c2:
                if tfl_btn:
                    if not bool(tfl_model) or not Path(tfl_model).exists():
                        st.error("Model file not found.")
                    else:
                        with st.spinner("Converting to TFLite with INT8 quantisation…"):
                            try:
                                from pipeline import convert_model
                                out = convert_model(tfl_model, tfl_out)
                                size_mb = Path(out).stat().st_size / 1e6
                                st.session_state.tfl_exported = True
                                st.success(f"Exported → `{out}` ({size_mb:.1f} MB)")
                            except Exception as e:
                                st.error(f"Export failed: {e}")

    # =========================================================================
    # SYSTEM DASHBOARD TAB
    # =========================================================================
    with tab_system:
        render_dashboard()
