import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

from components.system_dashboard import render_dashboard, record_resource_sample
from components.webcam_selector import webcam_selector
import config as _config_module
from config import DISEASE_CLASSES, DISEASE_DISPLAY

# Project root is the directory that contains config.py — resolves correctly
# even when Streamlit's CWD differs from the project root (e.g. git worktrees).
PROJECT_ROOT = Path(_config_module.__file__).parent
from pipeline import (
    predict_image, train_model, train_two_stage, evaluate_model,
    extract_features, train_head, fine_tune_model, split_dataset, log_inference,
)
from src.utils.recommendations import get_recommendation
from streamlit_callback import StreamlitTrainCallback



def _load_latest_history() -> dict:
    """Load training history from the most recent outputs/TrainingN/ folder.

    Scans outputs/Training1/, outputs/Training2/, … and returns the history
    from the highest-numbered directory that contains a history_*.json file.
    Falls back to an empty dict when none exists.
    """
    import json as _json
    base = Path("outputs")
    if not base.exists():
        return {}
    # Collect all TrainingN dirs sorted by N descending
    training_dirs = sorted(
        [p for p in base.iterdir()
         if p.is_dir() and p.name.startswith("Training") and p.name[8:].isdigit()],
        key=lambda p: int(p.name[8:]),
        reverse=True,
    )
    for td in training_dirs:
        files = sorted(td.glob("history_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            try:
                return _json.loads(files[0].read_text())
            except Exception:
                continue
    return {}


def _find_models() -> list:
    """Return .keras files under models/trained/, newest first."""
    base = Path("models/trained")
    if not base.exists():
        return []
    return sorted(
        [str(p) for p in base.rglob("*.keras")],
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )


def _model_picker(label: str, key: str, default: str = "models/trained/latest.keras") -> str:
    """Selectbox of available models; falls back to a text input.

    When models exist on disk they are listed newest-first.
    When none are found the field is left blank if the default path does not
    exist, so the user is never silently pointed at a missing file.
    """
    available = _find_models()
    CUSTOM = "Custom path..."
    if available:
        choice = st.selectbox(
            label,
            options=available + [CUSTOM],
            format_func=lambda x: Path(x).name if x != CUSTOM else CUSTOM,
            key=f"{key}_select",
        )
        if choice == CUSTOM:
            detected = _detect_dir([default], fallback="")
            return st.text_input("Custom model path", detected, key=f"{key}_custom",
                                 placeholder="models/trained/run_xxx/stage2_ft.keras")
        return choice

    # No models found — auto-detect a usable default, else leave blank
    detected = _detect_dir([default, "models/trained/latest.keras"], fallback="")
    value = st.text_input(label, detected,
                          placeholder="models/trained/run_xxx/stage2_ft.keras", key=key)
    if detected:
        st.caption(f"Auto-detected: `{detected}`")
    else:
        st.caption("No trained model found — enter path manually or train a model first.")
    return value


def _detect_dir(candidates: list, fallback: str = "") -> str:
    """Return the first existing path from candidates, else fallback.

    CWD is guaranteed to be PROJECT_ROOT (set in run_app), so relative paths
    resolve correctly.  Returns the original candidate string (not absolute) so
    field values stay short and readable.
    """
    for p in candidates:
        if p and Path(p).exists():
            return p
    return fallback


def _dir_input(label: str, candidates: list, fallback: str, key: str, help: str = "") -> str:
    """Text input pre-filled with the first detected directory, blank if none found."""
    default = _detect_dir(candidates, fallback)
    found   = default != fallback or (fallback and Path(fallback).exists())
    value   = st.text_input(label, value=default, key=key, help=help)
    if default and Path(default).exists():
        st.caption(f"Auto-detected: `{default}`")
    elif not value:
        st.caption("Directory not found — enter path manually.")
    return value




# =============================================================================
# HELPERS
# =============================================================================

def _model_status_html(path: str) -> str:
    exists = bool(path) and Path(path).exists()
    if exists:
        return '<span class="model-status-ok">&#9679; Model Ready</span>'
    return '<span class="model-status-missing">&#9679; Model Not Found</span>'


def _hero():
    scans = st.session_state.get("scans", 0)
    diseases = st.session_state.get("diseases", 0)
    healthy = scans - diseases
    healthy_rate = f"{(healthy / scans * 100):.0f}%" if scans else "—"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #C1121F 0%, #E63946 45%, #16213E 100%);
            border-radius: 18px;
            padding: 2.2rem 2.8rem;
            margin-bottom: 1.8rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position:absolute;top:-60px;right:-40px;
                width:260px;height:260px;
                background:rgba(255,255,255,0.04);
                border-radius:50%;
            "></div>
            <div style="
                position:absolute;bottom:-80px;right:80px;
                width:180px;height:180px;
                background:rgba(255,255,255,0.03);
                border-radius:50%;
            "></div>
            <div style="font-size:0.72rem;font-weight:700;letter-spacing:2px;
                        color:rgba(255,255,255,0.55);margin-bottom:0.5rem;
                        text-transform:uppercase;">
                AI-Powered Plant Health
            </div>
            <div style="font-size:2.1rem;font-weight:800;color:#FFFFFF;
                        letter-spacing:-0.5px;line-height:1.15;margin-bottom:0.4rem;">
                Tomato Disease Diagnostics
            </div>
            <div style="font-size:0.95rem;color:rgba(255,255,255,0.7);
                        font-weight:400;margin-bottom:1.8rem;">
                MobileNetV3 transfer learning &nbsp;·&nbsp; 10 disease classes &nbsp;·&nbsp; Real-time webcam inference
            </div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;">
                <div>
                    <div style="font-size:1.7rem;font-weight:800;color:#FFFFFF;">{scans}</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.55);font-weight:500;
                                text-transform:uppercase;letter-spacing:0.5px;">Leaves Scanned</div>
                </div>
                <div>
                    <div style="font-size:1.7rem;font-weight:800;color:#FCA5A5;">{diseases}</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.55);font-weight:500;
                                text-transform:uppercase;letter-spacing:0.5px;">Diseases Found</div>
                </div>
                <div>
                    <div style="font-size:1.7rem;font-weight:800;color:#6EE7B7;">{healthy_rate}</div>
                    <div style="font-size:0.75rem;color:rgba(255,255,255,0.55);font-weight:500;
                                text-transform:uppercase;letter-spacing:0.5px;">Healthy Rate</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _diagnosis_card(result: dict):
    disease = result["disease"]
    display = DISEASE_DISPLAY.get(disease, disease)
    conf = result["confidence"]
    passes = result["passes_threshold"]
    is_healthy = "healthy" in disease.lower()

    card_class = "diagnosis-card-healthy" if is_healthy else "diagnosis-card-diseased"
    icon = "✅" if is_healthy else "⚠️"
    status = "Healthy Plant" if is_healthy else "Disease Detected"
    title_color = "#065F46" if is_healthy else "#991B1B"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:1px;color:#6B7280;margin-bottom:0.4rem;">Diagnosis</div>
            <div class="diagnosis-title" style="color:{title_color};">{icon} {display}</div>
            <div class="diagnosis-sub">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{conf:.1%}")
    c2.metric("Threshold", "Pass" if passes else "Below", delta=None)

    if not passes:
        st.warning("Confidence below threshold — results may be unreliable. Try a clearer image.")

    with st.expander("Treatment Recommendation", expanded=is_healthy is False):
        st.markdown(get_recommendation(disease))

    with st.expander("All Class Probabilities"):
        prob_df = (
            pd.DataFrame(result["all_probs"].items(), columns=["Disease", "Probability"])
            .assign(Disease=lambda d: d["Disease"].map(lambda x: DISEASE_DISPLAY.get(x, x)))
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        prob_df["Probability"] = prob_df["Probability"].map("{:.2%}".format)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def plot_learning_curves(history: dict):
    if not history or not history.get("epoch"):
        return
    df = pd.DataFrame(history).dropna()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy", "Loss"),
        shared_xaxes=True,
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"], mode="lines+markers",
                             name="Train", line=dict(color="#2D9E6B", width=2.5)), row=1, col=1)
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers",
                                 name="Val", line=dict(color="#F4A261", width=2.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#E63946", width=2.5), showlegend=False), row=1, col=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers",
                                 name="Val Loss", line=dict(color="#6366F1", width=2.5, dash="dash"), showlegend=False), row=1, col=2)
    fig.update_layout(
        height=340, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, key="learning_curves")


@st.cache_resource
def load_cached_model(path: str):
    return tf.keras.models.load_model(path)


@st.cache_data(ttl=30, show_spinner=False)
def _enumerate_opencv_cameras():
    """Return list of (index, name) for cameras openable via OpenCV/DirectShow."""
    import subprocess
    # Get names via PowerShell
    try:
        res = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_PnPEntity | Where-Object {$_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image'} | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=5
        )
        names = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    except Exception:
        names = []

    cameras = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                label = names[idx] if idx < len(names) else f"Camera {idx}"
                cameras.append((idx, label))
    return cameras


def _run_cam_inference(img_bytes, model_path, confidence):
    """Shared inference runner for both webcam modes."""
    from io import BytesIO
    with st.spinner("Analysing leaf…"):
        try:
            load_cached_model(model_path)
            result = predict_image(model_path, BytesIO(img_bytes), confidence / 100)
            record_resource_sample("Inference")
            log_inference(result["disease"], result["confidence"], Path(model_path).name)
            st.session_state["_cam_last_result"] = result
            st.session_state.scans = st.session_state.get("scans", 0) + 1
            if "healthy" not in result["disease"].lower():
                st.session_state.diseases = st.session_state.get("diseases", 0) + 1
            st.session_state.setdefault("diagnosis_log", []).append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Disease": DISEASE_DISPLAY.get(result["disease"], result["disease"]),
                "Confidence": f"{result['confidence']:.1%}",
                "Status": "Pass" if result.get("passes_threshold") else "Low confidence",
                "Model": Path(model_path).name,
            })
        except Exception as e:
            st.error(f"Inference error: {e}")


def _draw_leaf_bbox(frame_bgr):
    """
    Detect the primary leaf/subject in a BGR frame and draw a bounding box.

    Strategy:
      1. HSV colour segmentation for green / yellow-green tones.
      2. Morphological cleanup to remove noise.
      3. Largest contour → bounding rect with padding.
      4. Fallback: centre 80 % crop if no large-enough contour is found.

    Returns:
        annotated_rgb  – uint8 RGB array with the box drawn on it
        crop_bgr       – BGR crop of the detected region (for inference)
        bbox           – (x, y, w, h) of the drawn box
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Green + yellow-green range covers healthy leaves and many diseased tones
    mask_green  = cv2.inRange(hsv, np.array([25, 30, 30]),  np.array([90, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([10, 40, 40]),  np.array([25, 255, 255]))
    mask = mask_green | mask_yellow

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_found = False
    bbox = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > w * h * 0.02:   # ≥ 2 % of frame area
            x, y, bw, bh = cv2.boundingRect(largest)
            pad = 20
            x  = max(0, x - pad)
            y  = max(0, y - pad)
            x2 = min(w, x + bw + 2 * pad)
            y2 = min(h, y + bh + 2 * pad)
            bbox = (x, y, x2 - x, y2 - y)
            leaf_found = True

    if bbox is None:                                   # fallback: centre 80 %
        mx, my = w // 10, h // 10
        bbox = (mx, my, w - 2 * mx, h - 2 * my)

    bx, by, bw, bh = bbox
    crop_bgr   = frame_bgr[by : by + bh, bx : bx + bw]
    annotated  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    colour = (0, 230, 80)                             # green box
    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), colour, 2)
    label = "Leaf detected" if leaf_found else "Region of interest"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(annotated, (bx, by - th - 8), (bx + tw + 8, by), colour, -1)
    cv2.putText(annotated, label, (bx + 4, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    return annotated, crop_bgr, bbox


@st.cache_resource
def _get_camera(device_idx: int):
    """
    Open and warm up a DirectShow camera, kept alive across reruns via
    st.cache_resource so the sensor stays warm and frames are never black.
    """
    import time as _time
    cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _time.sleep(0.6)
    for _ in range(12):          # drain early black frames
        cap.read()
    return cap


def _render_opencv_camera(model_path, confidence):
    """Render the server-side OpenCV camera capture UI with real-time preview."""
    cameras = _enumerate_opencv_cameras()
    if not cameras:
        st.error("No cameras accessible via OpenCV. Try the Browser Webcam mode.")
        return

    options = {label: idx for idx, label in cameras}
    chosen_name = st.selectbox("Camera", list(options.keys()), key="opencv_cam_select")
    chosen_idx  = options[chosen_name]

    if "opencv_streaming" not in st.session_state:
        st.session_state.opencv_streaming = False

    # Placeholder for the live preview image
    preview_ph = st.empty()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        toggle_label = "⏹ Stop Preview" if st.session_state.opencv_streaming else "▶ Live Preview"
        if st.button(toggle_label, key="opencv_preview_toggle", use_container_width=True):
            st.session_state.opencv_streaming = not st.session_state.opencv_streaming
            st.rerun()
    with col2:
        capture_clicked = st.button(
            "📷 Capture & Analyze", key="opencv_capture_btn",
            type="primary", use_container_width=True,
        )

    # ── Read one frame whenever we're streaming or the user clicked capture ──
    if st.session_state.opencv_streaming or capture_clicked:
        import time as _time
        cap = _get_camera(chosen_idx)
        ret, frame = cap.read()

        # If the frame is black (sensor still waking), try a few more reads
        if ret and frame is not None and frame.mean() <= 2.0:
            deadline = _time.time() + 1.5
            while _time.time() < deadline:
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 2.0:
                    break
                _time.sleep(0.05)

        if ret and frame is not None and frame.mean() > 2.0:
            annotated_rgb, crop_bgr, _ = _draw_leaf_bbox(frame)
            preview_ph.image(annotated_rgb, use_container_width=True)

            if capture_clicked:
                _, buf = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                _run_cam_inference(buf.tobytes(), model_path, confidence)
        elif capture_clicked:
            st.error("Failed to read a valid frame. Try toggling Live Preview first.")

    # ── Trigger the next frame after a short delay ───────────────────────────
    if st.session_state.opencv_streaming:
        import time as _time
        _time.sleep(1 / 15)   # ~15 fps
        st.rerun()


def _save_learning_curves(history: dict, model_path: str = "", output_dir: Path | None = None) -> str:
    """Save learning curves HTML + history JSON to outputs/TrainingN/.

    Returns the path to the saved HTML file.
    """
    import json as _json
    from streamlit_callback import _next_training_output_dir
    # Reuse the callback's dir (already created during training) or make a new one
    out_dir = output_dir if output_dir else _next_training_output_dir()
    stem = Path(model_path).stem if model_path else "run"

    # JSON — for dashboard reload
    json_path = out_dir / f"history_{stem}.json"
    json_path.write_text(_json.dumps(history))

    # HTML — interactive chart
    html_path = out_dir / f"curves_{stem}.html"
    df = pd.DataFrame(history).dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"], mode="lines+markers",
                             name="Train Acc", line=dict(color="#2D9E6B", width=2.5)), row=1, col=1)
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers",
                                 name="Val Acc", line=dict(color="#F4A261", width=2.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#E63946", width=2.5)), row=1, col=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers",
                                 name="Val Loss", line=dict(color="#6366F1", width=2.5, dash="dash")), row=1, col=2)
    fig.update_layout(height=420, template="plotly_white", hovermode="x unified",
                      margin=dict(l=30, r=20, t=60, b=30))
    fig.write_html(str(html_path))
    return str(html_path)


# =============================================================================
# MAIN APP
# =============================================================================

def _clear_stale_dir_keys():
    """Remove session-state entries for directory inputs that hold non-directory values.

    This fires once per session on the first render and silently discards any
    value that was stored from an older code state (e.g. a model file path
    stored in a dataset-directory key).
    """
    dir_keys = {
        "shared_data_dir": ("data",),
        "shared_model_base": ("models",),
        "raw_source": ("data",),
        "split_output": ("data",),
        "feat_out": ("data",),
        "eval_test_dir": ("data",),
    }
    for key, prefixes in dir_keys.items():
        val = st.session_state.get(key, "")
        if val and not any(val.startswith(p) for p in prefixes):
            del st.session_state[key]


def run_app():
    # Pin the process CWD to the project root so all relative paths work
    # regardless of where Streamlit was launched from (e.g. a git worktree).
    os.chdir(PROJECT_ROOT)
    _clear_stale_dir_keys()
    _hero()

    tab_inference, tab_training, tab_system = st.tabs([
        "  Inference  ",
        "  Training Dashboard  ",
        "  System Dashboard  ",
    ])

    # =========================================================================
    # INFERENCE TAB
    # =========================================================================
    with tab_inference:
        # ── Inline settings bar ───────────────────────────────────────
        with st.expander("Detection Settings", expanded=False):
            s1, s2 = st.columns([1, 2])
            confidence = s1.slider(
                "Confidence threshold (%)", 0.0, 100.0, 50.0, 1.0,
                help="Minimum probability required to report a detection",
            )
            selected_classes = s2.multiselect(
                "Disease filter",
                options=DISEASE_CLASSES,
                default=DISEASE_CLASSES,
                format_func=lambda x: DISEASE_DISPLAY.get(x, x),
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        mode = st.radio(
            "input_mode", ["Static Image", "Live Webcam"],
            horizontal=True, label_visibility="collapsed",
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ----- Static Image -----
        if mode == "Static Image":
            col_upload, col_result = st.columns([1, 1], gap="large")

            with col_upload:
                st.markdown(
                    '<div class="section-label">Upload Leaf Image</div>',
                    unsafe_allow_html=True,
                )
                uploaded = st.file_uploader(
                    "upload", type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                )
                if uploaded:
                    st.image(uploaded, use_container_width=True, caption="Uploaded image")

            with col_result:
                st.markdown(
                    '<div class="section-label">Model & Diagnosis</div>',
                    unsafe_allow_html=True,
                )
                model_path = _model_picker("Model", key="img_model")
                st.markdown(_model_status_html(model_path), unsafe_allow_html=True)
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                run_btn = st.button(
                    "Run Diagnosis", type="primary",
                    use_container_width=True,
                    disabled=uploaded is None,
                )

                if uploaded is None:
                    # Clear stale result when the user removes the image
                    st.session_state.pop("last_static_result", None)
                    st.markdown(
                        """
                        <div style="background:#F8FAFC;border:2px dashed #E2E8F0;
                                    border-radius:12px;padding:2.5rem 1.5rem;
                                    text-align:center;margin-top:1rem;">
                            <div style="font-size:2rem;margin-bottom:0.5rem;">🌿</div>
                            <div style="color:#94A3B8;font-size:0.88rem;font-weight:500;">
                                Upload a leaf image to begin diagnosis
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if run_btn and uploaded:
                    with st.spinner("Analysing leaf..."):
                        try:
                            load_cached_model(model_path)
                            result = predict_image(
                                model_path, uploaded, confidence / 100
                            )
                            record_resource_sample("Inference")
                            log_inference(
                                result["disease"],
                                result["confidence"],
                                Path(model_path).name,
                            )
                            st.session_state.scans = st.session_state.get("scans", 0) + 1
                            if "healthy" not in result["disease"].lower():
                                st.session_state.diseases = st.session_state.get("diseases", 0) + 1
                            # Log to session history
                            if "diagnosis_log" not in st.session_state:
                                st.session_state.diagnosis_log = []
                            st.session_state.diagnosis_log.append({
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Disease": DISEASE_DISPLAY.get(result["disease"], result["disease"]),
                                "Confidence": f"{result['confidence']:.1%}",
                                "Status": "Pass" if result["passes_threshold"] else "Low confidence",
                                "Model": Path(model_path).name,
                            })
                            # Store result so it survives the rerun below
                            st.session_state["last_static_result"] = result
                            st.rerun()
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

                # Render the diagnosis card persistently — survives reruns
                # because it reads from session_state, not from the button callback.
                last_result = st.session_state.get("last_static_result")
                if last_result and uploaded is not None:
                    _diagnosis_card(last_result)
                elif uploaded is not None and not st.session_state.get("last_static_result"):
                    st.markdown(
                        """
                        <div style="background:#F8FAFC;border:2px dashed #E2E8F0;
                                    border-radius:12px;padding:2.5rem 1.5rem;
                                    text-align:center;margin-top:1rem;">
                            <div style="font-size:2rem;margin-bottom:0.5rem;">🔬</div>
                            <div style="color:#94A3B8;font-size:0.88rem;font-weight:500;">
                                Click <b>Run Diagnosis</b> to analyse this leaf
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ----- Live Webcam -----
        else:
            col_cam, col_live = st.columns([3, 2], gap="large")

            with col_cam:
                st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
                st.info("Select your camera, point at a tomato leaf, then click **📷 Take Photo** to diagnose.", icon="📷")
                model_path_cam = _model_picker("Model", key="cam_model")

                cam_source = st.radio(
                    "cam_source", ["Browser Webcam", "System Camera (OpenCV)"],
                    horizontal=True, label_visibility="collapsed",
                )

                # ── Browser webcam ───────────────────────────────────────────────────
                if cam_source == "Browser Webcam":
                    frame_data = webcam_selector(height=440, key="webcam_capture")
                    if frame_data is not None:
                        import base64
                        raw = base64.b64decode(frame_data.split(",")[1])
                        nparr = np.frombuffer(raw, np.uint8)
                        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        annotated_rgb, crop_bgr, _ = _draw_leaf_bbox(frame_bgr)
                        st.image(annotated_rgb, caption="Captured frame",
                                 use_container_width=True)
                        _, buf = cv2.imencode('.jpg', crop_bgr,
                                             [cv2.IMWRITE_JPEG_QUALITY, 92])
                        _run_cam_inference(buf.tobytes(), model_path_cam, confidence)

                # ── Server-side OpenCV capture ───────────────────────────────────────
                else:
                    _render_opencv_camera(model_path_cam, confidence)

            with col_live:
                st.markdown('<div class="section-label">Last Prediction</div>', unsafe_allow_html=True)
                r = st.session_state.get("_cam_last_result")
                if r:
                    display = DISEASE_DISPLAY.get(r["disease"], r["disease"])
                    is_healthy = "healthy" in r["disease"].lower()
                    card_class = "diagnosis-card-healthy" if is_healthy else "diagnosis-card-diseased"
                    icon = "✅" if is_healthy else "⚠️"
                    title_color = "#065F46" if is_healthy else "#991B1B"
                    st.markdown(
                        f"""
                        <div class="{card_class}" style="margin-bottom:1rem;">
                            <div class="diagnosis-title" style="color:{title_color};">
                                {icon} {display}
                            </div>
                            <div class="diagnosis-sub">{r['confidence']:.1%} confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if not r.get("passes_threshold", True):
                        st.caption("Below threshold — move closer or improve lighting.")
                    with st.expander("Treatment", expanded=not is_healthy):
                        st.markdown(get_recommendation(r["disease"]))
                else:
                    st.markdown(
                        """
                        <div style="background:#F8FAFC;border:2px dashed #E2E8F0;
                                    border-radius:12px;padding:3rem 1.5rem;text-align:center;">
                            <div style="font-size:2rem;margin-bottom:0.5rem;">📷</div>
                            <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;">
                                Capture a frame to see predictions
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Diagnosis History ─────────────────────────────────────────
        if st.session_state.get("diagnosis_log"):
            st.divider()
            st.markdown(
                '<div class="section-label">Session Diagnosis History</div>',
                unsafe_allow_html=True,
            )
            log_df = pd.DataFrame(st.session_state.diagnosis_log[::-1])
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            if st.button("Clear history", type="secondary"):
                st.session_state.diagnosis_log = []
                st.rerun()

    # =========================================================================
    # TRAINING TAB  — Three-stage Keras transfer-learning pipeline
    # https://keras.io/guides/transfer_learning/
    # =========================================================================
    with tab_training:

        def _stage_badge(done: bool) -> str:
            if done:
                return ('<span style="background:#D1FAE5;color:#065F46;padding:3px 10px;'
                        'border-radius:20px;font-size:0.75rem;font-weight:600;">Done</span>')
            return ('<span style="background:#F1F5F9;color:#94A3B8;padding:3px 10px;'
                    'border-radius:20px;font-size:0.75rem;font-weight:600;">Pending</span>')

        # ── Shared config ─────────────────────────────────────────────────
        # Initialise session-state defaults once so Streamlit's widget state
        # takes over from the second render onwards without resetting user edits.
        if "shared_data_dir" not in st.session_state:
            st.session_state["shared_data_dir"] = _detect_dir([
                st.session_state.get("split_train_dir", ""),
                "data/splits/train",
                "data/raw/raw/tomato",
                "data/raw",
            ])
        if "shared_model_base" not in st.session_state:
            st.session_state["shared_model_base"] = _detect_dir(
                ["models/trained"], fallback=""
            )
        # Auto-detect completed stages from disk so buttons aren't
        # always disabled on fresh page loads after prior training runs.
        if "features_dir" not in st.session_state:
            detected = _detect_dir(["data/features"], fallback="")
            if detected:
                st.session_state["features_dir"] = detected
        if "head_full_path" not in st.session_state:
            for candidate in ["models/trained/head_full.keras", "models/trained/latest.keras"]:
                if Path(candidate).exists():
                    st.session_state["head_full_path"] = candidate
                    break

        st.markdown('<div class="section-label">Shared Settings</div>', unsafe_allow_html=True)
        sh1, sh2, sh3 = st.columns(3)
        data_dir = sh1.text_input(
            "Dataset directory",
            placeholder="Enter dataset directory path",
            key="shared_data_dir",
        )
        model_save_base = sh2.text_input(
            "Model output base",
            placeholder="Enter model output path",
            key="shared_model_base",
        )
        batch_size = sh3.select_slider(
            "Batch size", options=[4, 8, 16, 32, 64], value=16,
            help="Larger batches are faster but use more RAM.",
        )
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        col_cfg, col_viz = st.columns([1, 2], gap="large")

        # ── Left: three stage cards ───────────────────────────────────────
        with col_cfg:

            # ── Stage 0: Split Dataset ────────────────────────────────────
            split_done = bool(st.session_state.get("split_train_dir"))
            with st.expander(
                f"Stage 0 — Split Dataset  {'✅' if split_done else '⬜'}",
                expanded=not split_done,
            ):
                if "raw_source" not in st.session_state:
                    st.session_state["raw_source"] = _detect_dir([
                        "data/raw/raw/tomato", "data/raw/tomato", "data/raw", "dataset",
                    ])
                if "split_output" not in st.session_state:
                    st.session_state["split_output"] = _detect_dir(["data/splits"], fallback="")
                s0a, s0b = st.columns(2)
                raw_source = s0a.text_input(
                    "Raw dataset dir",
                    placeholder="path/to/raw",
                    help="Folder with one sub-folder per disease class.",
                    key="raw_source",
                )
                split_output = s0b.text_input(
                    "Splits output dir",
                    placeholder="data/splits",
                    key="split_output",
                )
                r1, r2, r3 = st.columns(3)
                train_ratio = r1.number_input("Train %", 0, 100, 70, step=5, key="train_ratio") / 100
                val_ratio   = r2.number_input("Val %",   0, 100, 20, step=5, key="val_ratio")   / 100
                test_ratio  = r3.number_input("Test %",  0, 100, 10, step=5, key="test_ratio")  / 100

                ratio_ok = abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
                if not ratio_ok:
                    st.warning("Train + Val + Test must add up to 100%.")

                btn_split = st.button(
                    "Split Dataset", type="primary", use_container_width=True,
                    disabled=(not ratio_ok) or st.session_state.get("training_active", False),
                    key="btn_split",
                    help="Divide raw images into train / val / test folders.",
                )

                if split_done:
                    st.caption(f"✓ Train dir: `{st.session_state.split_train_dir}`")
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            # ── Stage 1: Feature Extraction ──────────────────────────────
            feat_done = bool(st.session_state.get("features_dir"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <p class="stage-title">Stage 1 — Feature Extraction</p>
                        {_stage_badge(feat_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "feat_out" not in st.session_state:
                st.session_state["feat_out"] = _detect_dir(["data/features"], fallback="")
            feat_out = st.text_input(
                "Features output dir",
                placeholder="data/features",
                help="Run the frozen MobileNetV3Small base and cache feature maps to disk. No weights updated.",
                key="feat_out",
            )
            btn_extract = st.button(
                "Extract Features", type="primary", use_container_width=True,
                disabled=st.session_state.get("training_active", False),
                key="btn_extract",
            )

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            # ── Stage 2: Train Head ───────────────────────────────────────
            head_done = bool(st.session_state.get("head_full_path"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <p class="stage-title">Stage 2 — Train Head</p>
                        {_stage_badge(head_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            h1, h2 = st.columns(2)
            head_epochs = h1.slider("Epochs", 1, 50, 10, key="head_epochs")
            head_lr     = h2.select_slider(
                "Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001, key="head_lr",
            )
            btn_head = st.button(
                "Train Head", type="primary", use_container_width=True,
                disabled=(not feat_done) or st.session_state.get("training_active", False),
                key="btn_head",
                help="Train the classification head on cached features. Base model stays frozen.",
            )
            if not feat_done:
                st.caption("⚠ Run Feature Extraction first.")

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            # ── Stage 3: Fine-Tuning ──────────────────────────────────────
            ft_done = bool(st.session_state.get("fine_tuned_path"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <p class="stage-title">Stage 3 — Fine-Tuning</p>
                        {_stage_badge(ft_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            f1, f2 = st.columns(2)
            ft_epochs = f1.slider("Epochs", 1, 50, 10, key="ft_epochs")
            ft_lr     = f2.select_slider(
                "Learning rate", [0.000001, 0.00001, 0.0001], value=0.00001, key="ft_lr",
            )
            btn_ft = st.button(
                "Fine-Tune", type="primary", use_container_width=True,
                disabled=(not head_done) or st.session_state.get("training_active", False),
                key="btn_ft",
                help="Unfreeze the base model and train end-to-end at a very low learning rate.",
            )
            if not head_done:
                st.caption("⚠ Train the Head first.")

            # ── Saved paths summary ───────────────────────────────────────
            any_saved = any([
                st.session_state.get("features_dir"),
                st.session_state.get("head_full_path"),
                st.session_state.get("fine_tuned_path"),
            ])
            if any_saved:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Saved Artefacts</div>', unsafe_allow_html=True)
                for label, key in [
                    ("Features dir",  "features_dir"),
                    ("Head model",    "head_full_path"),
                    ("Fine-tuned",    "fine_tuned_path"),
                ]:
                    val = st.session_state.get(key)
                    if val:
                        st.markdown(
                            f'<p style="font-size:0.78rem;color:#6B7280;margin:4px 0 0 0;">{label}</p>'
                            f'<code style="font-size:0.72rem;">{val}</code>',
                            unsafe_allow_html=True,
                        )

        # ── Right: live output ────────────────────────────────────────────
        with col_viz:
            st.markdown('<div class="section-label">Live Training Output</div>', unsafe_allow_html=True)
            # Keep these as invisible empties when idle — the training block below
            # replaces them with real widgets once a stage starts.
            progress_bar        = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            if not st.session_state.get("training_active"):
                hist = st.session_state.get("train_history")
                if not hist:
                    # Try to load from disk (CLI runs or page refreshes)
                    hist = _load_latest_history()
                    if hist and hist.get("epoch"):
                        st.session_state["train_history"] = hist

                if hist and hist.get("epoch"):
                    with chart_placeholder:
                        plot_learning_curves(hist)
                else:
                    chart_placeholder.markdown(
                        """
                        <div style="background:white;border-radius:14px;padding:1.5rem 2rem;
                                    text-align:center;border:1px solid #EEF0F2;
                                    box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                            <div style="font-size:2rem;margin-bottom:0.4rem;">📈</div>
                            <div style="font-weight:600;color:#16213E;margin-bottom:0.25rem;font-size:0.95rem;">
                                No training data yet
                            </div>
                            <div style="color:#94A3B8;font-size:0.82rem;">
                                Run the stages on the left — curves appear here in real time.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Stage execution ───────────────────────────────────────────────
        if btn_split:
            if not raw_source:
                st.error("Enter the raw dataset directory path before splitting.")
            elif not Path(raw_source).exists():
                st.error(f"Raw dataset directory not found: `{raw_source}`")
            else:
                st.session_state.training_active = True
                st.session_state._active_stage   = "split"
                st.session_state._raw_source     = raw_source
                st.session_state._split_output   = split_output or "data/splits"
                st.session_state._train_ratio    = train_ratio
                st.session_state._val_ratio      = val_ratio
                st.session_state._test_ratio     = test_ratio
                st.rerun()

        if btn_extract:
            _ext_data = st.session_state.get("split_train_dir") or data_dir
            if not _ext_data:
                st.error("Enter the dataset directory path before extracting features.")
            elif not Path(_ext_data).exists():
                st.error(f"Dataset directory not found: `{_ext_data}`")
            else:
                st.session_state.training_active = True
                st.session_state._active_stage   = "extract"
                st.session_state._data_dir       = _ext_data
                st.session_state._feat_out       = feat_out or "data/features"
                st.session_state._batch_size     = batch_size
                st.rerun()

        if btn_head:
            _feat_dir = st.session_state.get("features_dir")
            if not _feat_dir or not Path(_feat_dir).exists():
                st.error("Features directory not found. Run Feature Extraction first.")
            else:
                st.session_state.training_active  = True
                st.session_state._active_stage    = "head"
                st.session_state._head_epochs     = head_epochs
                st.session_state._head_lr         = head_lr
                st.session_state._model_save_base = model_save_base or "models/trained/latest.keras"
                st.rerun()

        if btn_ft:
            _head_path = st.session_state.get("head_full_path")
            _ft_data   = data_dir or st.session_state.get("split_train_dir", "")
            if not _head_path or not Path(_head_path).exists():
                st.error("Head model not found. Train the Head first.")
            elif not _ft_data or not Path(_ft_data).exists():
                st.error("Enter a valid dataset directory for fine-tuning.")
            else:
                st.session_state.training_active = True
                st.session_state._active_stage   = "finetune"
                st.session_state._ft_epochs      = ft_epochs
                st.session_state._ft_lr          = ft_lr
                st.session_state._batch_size     = batch_size
                st.session_state._data_dir       = _ft_data
                st.rerun()

        if st.session_state.get("training_active", False):
            stage = st.session_state.get("_active_stage", "")
            record_resource_sample("Training")

            progress_bar        = st.progress(0, text="Initialising...")
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            try:
                # ── Split Dataset ─────────────────────────────────────────
                if stage == "split":
                    def _split_progress(current, total, class_name):
                        progress_bar.progress(
                            current / max(total, 1),
                            text=f"Splitting class {current}/{total}: {class_name}",
                        )

                    with st.spinner("Splitting dataset into train / val / test..."):
                        result = split_dataset(
                            source_dir=st.session_state["_raw_source"],
                            output_dir=st.session_state["_split_output"],
                            train_ratio=st.session_state["_train_ratio"],
                            val_ratio=st.session_state["_val_ratio"],
                            test_ratio=st.session_state["_test_ratio"],
                            progress_fn=_split_progress,
                        )
                    st.session_state.split_train_dir = result["train_dir"]
                    st.session_state.split_val_dir   = result["val_dir"]
                    st.session_state.split_test_dir  = result["test_dir"]
                    # Clear widget so shared data_dir field picks up new train dir
                    st.session_state.pop("shared_data_dir", None)
                    progress_bar.progress(1.0, text="Dataset split complete")
                    c = result["counts"]
                    st.success(
                        f"{result['num_classes']} classes split — "
                        f"train: {c['train']} · val: {c['val']} · test: {c['test']} images"
                    )

                # ── Extract Features ──────────────────────────────────────
                elif stage == "extract":
                    def _ext_progress(current, total):
                        progress_bar.progress(
                            min(current / max(total, 1), 1.0),
                            text=f"Extracting batch {current} / {total}",
                        )

                    with st.spinner("Extracting features from dataset..."):
                        result = extract_features(
                            data_dir=st.session_state["_data_dir"],
                            output_dir=st.session_state["_feat_out"],
                            batch_size=st.session_state["_batch_size"],
                            progress_fn=_ext_progress,
                        )
                    st.session_state.features_dir = result["features_dir"]
                    progress_bar.progress(1.0, text="Feature extraction complete")
                    st.success(
                        f"Extracted {result['num_samples']} samples · "
                        f"{result['num_classes']} classes · "
                        f"feature shape {result['feature_shape']} → `{result['features_dir']}`"
                    )

                # ── Train Head ────────────────────────────────────────────
                elif stage == "head":
                    cb = StreamlitTrainCallback(
                        progress_bar, metrics_placeholder, chart_placeholder,
                        stage_label="Train Head",
                    )
                    with st.spinner("Training classification head..."):
                        result = train_head(
                            features_dir=st.session_state["features_dir"],
                            epochs=st.session_state["_head_epochs"],
                            lr=st.session_state["_head_lr"],
                            output_path=st.session_state["_model_save_base"],
                            callbacks=[cb],
                        )
                    st.session_state.head_full_path   = result["full_model_path"]
                    st.session_state.train_history    = cb.history
                    record_resource_sample("Training")
                    curves_path = _save_learning_curves(cb.history, result["full_model_path"], output_dir=cb.output_dir)
                    st.success(
                        f"Head trained · full model → `{result['full_model_path']}`  "
                        f"· curves → `{curves_path}`"
                    )

                # ── Fine-Tune ─────────────────────────────────────────────
                elif stage == "finetune":
                    cb = StreamlitTrainCallback(
                        progress_bar, metrics_placeholder, chart_placeholder,
                        stage_label="Fine-Tuning",
                    )
                    with st.spinner("Fine-tuning full model end-to-end..."):
                        result = fine_tune_model(
                            model_path=st.session_state["head_full_path"],
                            data_dir=st.session_state["_data_dir"],
                            batch_size=st.session_state["_batch_size"],
                            epochs=st.session_state["_ft_epochs"],
                            lr=st.session_state["_ft_lr"],
                            callbacks=[cb],
                        )
                    st.session_state.fine_tuned_path = result["path"]
                    st.session_state.train_history   = cb.history
                    record_resource_sample("Training")
                    curves_path = _save_learning_curves(cb.history, result["path"], output_dir=cb.output_dir)
                    st.success(
                        f"Fine-tuned model → `{result['path']}`  "
                        f"· curves → `{curves_path}`"
                    )

            except Exception as e:
                st.error(f"Stage failed: {e}")
            finally:
                st.session_state.training_active = False
                st.rerun()

        _final_hist = st.session_state.get("train_history") or _load_latest_history()
        if _final_hist and _final_hist.get("epoch"):
            st.divider()
            st.markdown('<div class="section-label">Final Learning Curves</div>', unsafe_allow_html=True)
            plot_learning_curves(_final_hist)

        # ── Model Evaluation ─────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">Model Evaluation</div>', unsafe_allow_html=True)
        ev_c1, ev_c2 = st.columns([1, 1], gap="large")
        with ev_c1:
            eval_model_path = _model_picker("Model to evaluate", key="eval_model")
            if "eval_test_dir" not in st.session_state:
                st.session_state["eval_test_dir"] = _detect_dir([
                    st.session_state.get("split_test_dir", ""),
                    "data/splits/test",
                    "data/splits/val",
                ])
            eval_test_dir = st.text_input(
                "Test directory",
                placeholder="Enter test directory path",
                help="Folder of class-named subdirectories, separate from training data.",
                key="eval_test_dir",
            )
            eval_btn = st.button("Evaluate Model", type="primary", use_container_width=True)
        with ev_c2:
            if eval_btn:
                if not Path(eval_model_path).exists():
                    st.error("Model file not found — select a valid path.")
                elif not Path(eval_test_dir).exists():
                    st.error("Test directory not found — check the path.")
                else:
                    with st.spinner("Evaluating on test set..."):
                        try:
                            ev_result = evaluate_model(eval_model_path, eval_test_dir)
                            st.session_state.eval_result     = ev_result
                            st.session_state.eval_model_name = Path(eval_model_path).name
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")

            if st.session_state.get("eval_result"):
                ev  = st.session_state.eval_result
                acc = ev["accuracy"]
                st.markdown(
                    f'<div class="section-label">Results — {st.session_state.eval_model_name}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Test Accuracy", f"{acc:.2%}")
                m2.metric("Test Loss",     f"{ev['loss']:.4f}")
                color   = "#2D9E6B" if acc >= 0.9 else "#F4A261" if acc >= 0.75 else "#E63946"
                verdict = "Excellent — ready for deployment." if acc >= 0.9 else \
                          "Good — consider more fine-tuning." if acc >= 0.75 else \
                          "Needs improvement — try more epochs."
                st.markdown(
                    f'<div style="margin-top:0.75rem;padding:0.6rem 1rem;background:#F8FAFC;'
                    f'border-radius:8px;border-left:4px solid {color};">'
                    f'<span style="color:{color};font-weight:600;font-size:0.88rem;">{verdict}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background:#F8FAFC;border:2px dashed #E2E8F0;
                                border-radius:12px;padding:2.5rem 1.5rem;text-align:center;">
                        <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
                        <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;">
                            Select a model and test directory, then click Evaluate.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── TFLite Export ─────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">Export for Edge Deployment</div>', unsafe_allow_html=True)
        tfl_c1, tfl_c2 = st.columns([1, 1], gap="large")
        with tfl_c1:
            tfl_model = _model_picker("Model to export", key="tfl_model")
            tfl_out   = st.text_input(
                "TFLite output path",
                value="models/trained/model.tflite",
                key="tfl_out_path",
            )
            tfl_btn = st.button("Export TFLite", type="primary", use_container_width=True)
        with tfl_c2:
            if tfl_btn:
                if not bool(tfl_model) or not Path(tfl_model).exists():
                    st.error("Model file not found — select a valid path.")
                else:
                    with st.spinner("Converting to TFLite with INT8 quantisation..."):
                        try:
                            from pipeline import convert_model
                            out = convert_model(tfl_model, tfl_out)
                            size_mb = Path(out).stat().st_size / 1e6
                            st.success(
                                f"Exported → `{out}`  ({size_mb:.1f} MB)  "
                                f"· Ready for Raspberry Pi / mobile deployment"
                            )
                        except Exception as e:
                            st.error(f"Export failed: {e}")
            else:
                st.markdown(
                    """
                    <div style="background:#F8FAFC;border:2px dashed #E2E8F0;
                                border-radius:12px;padding:2.5rem 1.5rem;text-align:center;">
                        <div style="font-size:2rem;margin-bottom:0.5rem;">📱</div>
                        <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;">
                            Convert a trained model to TFLite for fast inference<br>
                            on Raspberry Pi or mobile devices.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # SYSTEM DASHBOARD TAB
    # =========================================================================
    with tab_system:
        render_dashboard()
