import os
import threading
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from components.system_dashboard import render_dashboard, record_resource_sample
from config import DISEASE_CLASSES, DISEASE_DISPLAY
from pipeline import (
    predict_image, train_model, train_two_stage, evaluate_model,
    extract_features, train_head, fine_tune_model,
)
from src.utils.recommendations import get_recommendation
from streamlit_callback import StreamlitTrainCallback

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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
    """Selectbox of available models with a custom-path fallback."""
    available = _find_models()
    CUSTOM = "Custom path..."
    if available:
        choice = st.selectbox(
            label,
            options=[CUSTOM] + available,
            format_func=lambda x: Path(x).name if x != CUSTOM else CUSTOM,
            key=f"{key}_select",
        )
        if choice == CUSTOM:
            return st.text_input("Custom model path", default, key=f"{key}_custom")
        return choice
    return st.text_input(label, default, placeholder="models/trained/run_xxx/stage2_ft.keras", key=key)


# =============================================================================
# WEBCAM PREDICTOR
# =============================================================================

class FramePredictor:
    INFERENCE_EVERY = 20

    def __init__(self, model_path: str, conf_threshold: float):
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_result = None
        self._load_error = None

    def _load(self):
        if self._model is None and self._load_error is None:
            try:
                self._model = tf.keras.models.load_model(self._model_path)
            except Exception as e:
                self._load_error = str(e)

    def _infer(self, bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        # Raw [0, 255] — preprocess_input is baked into the model graph
        arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
        preds = self._model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        name = DISEASE_CLASSES[idx] if idx < len(DISEASE_CLASSES) else f"Class {idx}"
        return {"disease": name, "confidence": conf, "passes": conf >= self._conf_threshold}

    def _draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        disease = result["disease"]
        conf = result["confidence"]
        passes = result["passes"]
        if "healthy" in disease.lower():
            color = (0, 180, 0)
        elif passes:
            color = (0, 0, 210)
        else:
            color = (0, 140, 255)
        cv2.rectangle(frame, (0, 0), (w, 62), (15, 15, 30), -1)
        cv2.rectangle(frame, (0, 0), (w, 62), color, 3)
        display = DISEASE_DISPLAY.get(disease, disease)
        label = f"{display}  {conf:.1%}"
        cv2.putText(frame, label, (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2, cv2.LINE_AA)
        if not passes:
            cv2.putText(frame, "Below confidence threshold", (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)
        return frame

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1
        if self._frame_count % self.INFERENCE_EVERY == 1:
            with self._lock:
                self._load()
                if self._model is not None:
                    try:
                        self._last_result = self._infer(img)
                    except Exception as e:
                        self._load_error = f"Inference error: {e}"
        if self._load_error:
            cv2.putText(img, f"Model error: {self._load_error[:50]}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
        elif self._last_result:
            img = self._draw(img, self._last_result)
        else:
            cv2.putText(img, "Loading model...", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (200, 200, 200), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================================================================
# HELPERS
# =============================================================================

def _model_status_html(path: str) -> str:
    exists = Path(path).exists()
    if exists:
        return (
            '<span class="model-status-ok">&#9679; Model Ready</span>'
        )
    return (
        '<span class="model-status-missing">&#9679; Model Not Found</span>'
    )


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


def _save_learning_curves(history: dict) -> str:
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/learning_curves_{ts}.html"
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
    fig.write_html(path)
    return path


# =============================================================================
# MAIN APP
# =============================================================================

def run_app():
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
                            _diagnosis_card(result)
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
                            st.rerun()
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

        # ----- Live Webcam -----
        else:
            col_cam, col_live = st.columns([3, 2], gap="large")

            with col_cam:
                st.markdown(
                    '<div class="section-label">Live Camera Feed</div>',
                    unsafe_allow_html=True,
                )
                st.info("Point your webcam at a tomato leaf. Diagnosis updates every ~20 frames.", icon="📷")
                st.warning(
                    "Click **Allow** when your browser asks for camera permission. "
                    "If denied, click the camera icon in your address bar → Allow → refresh.",
                    icon="🔒",
                )
                model_path_cam = _model_picker("Model", key="cam_model")
                if st.session_state.get("_cam_model_loaded") != model_path_cam:
                    st.session_state.frame_predictor = FramePredictor(
                        model_path_cam, confidence / 100
                    )
                    st.session_state._cam_model_loaded = model_path_cam
                predictor = st.session_state.frame_predictor
                webrtc_streamer(
                    key="tomato-webcam",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIG,
                    video_frame_callback=predictor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

            with col_live:
                st.markdown('<div class="section-label">Last Prediction</div>', unsafe_allow_html=True)
                if predictor._last_result:
                    r = predictor._last_result
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
                    if not r["passes"]:
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
                                Start the camera to see live predictions
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
        st.markdown('<div class="section-label">Shared Settings</div>', unsafe_allow_html=True)
        sh1, sh2, sh3 = st.columns(3)
        data_dir        = sh1.text_input("Dataset directory", "data/splits/train")
        model_save_base = sh2.text_input("Model output base", "models/trained/latest.keras")
        batch_size      = sh3.select_slider(
            "Batch size", options=[4, 8, 16, 32, 64], value=16,
            help="Larger batches are faster but use more RAM.",
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        col_cfg, col_viz = st.columns([2, 3], gap="large")

        # ── Left: three stage cards ───────────────────────────────────────
        with col_cfg:

            # ── Stage 1: Feature Extraction ──────────────────────────────
            feat_done = bool(st.session_state.get("features_dir"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <p class="stage-title">Stage 1 — Feature Extraction</p>
                            <p class="stage-desc">
                                Run the frozen MobileNetV3Small base over the dataset and cache
                                the output feature maps to disk.  No weights are updated.
                            </p>
                        </div>
                        {_stage_badge(feat_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            feat_out = st.text_input("Features output directory", "data/features", key="feat_out")
            btn_extract = st.button(
                "Extract Features", type="primary", use_container_width=True,
                disabled=st.session_state.get("training_active", False),
                key="btn_extract",
            )

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            # ── Stage 2: Train Head ───────────────────────────────────────
            head_done = bool(st.session_state.get("head_full_path"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <p class="stage-title">Stage 2 — Train Head</p>
                            <p class="stage-desc">
                                Train only the classification head (GAP → Dropout → Dense)
                                on the cached features.  Base model stays frozen.
                            </p>
                        </div>
                        {_stage_badge(head_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            head_epochs = st.slider("Epochs", 1, 50, 10, key="head_epochs")
            head_lr     = st.select_slider(
                "Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001, key="head_lr",
            )
            btn_head = st.button(
                "Train Head", type="primary", use_container_width=True,
                disabled=(not feat_done) or st.session_state.get("training_active", False),
                key="btn_head",
            )
            if not feat_done:
                st.caption("Run Feature Extraction first to enable this.")

            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

            # ── Stage 3: Fine-Tuning ──────────────────────────────────────
            ft_done = bool(st.session_state.get("fine_tuned_path"))
            st.markdown(
                f"""
                <div class="stage-header">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <p class="stage-title">Stage 3 — Fine-Tuning</p>
                            <p class="stage-desc">
                                Unfreeze the base model and train the entire network
                                end-to-end at a very low learning rate.
                            </p>
                        </div>
                        {_stage_badge(ft_done)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ft_epochs = st.slider("Epochs", 1, 50, 10, key="ft_epochs")
            ft_lr     = st.select_slider(
                "Learning rate", [0.000001, 0.00001, 0.0001], value=0.00001, key="ft_lr",
            )
            btn_ft = st.button(
                "Fine-Tune", type="primary", use_container_width=True,
                disabled=(not head_done) or st.session_state.get("training_active", False),
                key="btn_ft",
            )
            if not head_done:
                st.caption("Train the Head first to enable this.")

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
            progress_bar        = st.progress(0, text="Waiting to start...")
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            if not st.session_state.get("training_active") and not st.session_state.get("train_history"):
                chart_placeholder.markdown(
                    """
                    <div style="background:white;border-radius:14px;padding:4rem 2rem;
                                text-align:center;border:1px solid #EEF0F2;
                                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                        <div style="font-size:2.5rem;margin-bottom:0.75rem;">📈</div>
                        <div style="font-weight:600;color:#16213E;margin-bottom:0.4rem;">
                            No training data yet
                        </div>
                        <div style="color:#94A3B8;font-size:0.85rem;">
                            Follow the three stages on the left.<br>
                            Accuracy and loss curves appear here in real time.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Stage execution ───────────────────────────────────────────────
        if btn_extract:
            st.session_state.training_active = True
            st.session_state._active_stage   = "extract"
            st.session_state._feat_out       = feat_out
            st.session_state._batch_size     = batch_size
            st.rerun()

        if btn_head:
            st.session_state.training_active  = True
            st.session_state._active_stage    = "head"
            st.session_state._head_epochs     = head_epochs
            st.session_state._head_lr         = head_lr
            st.session_state._model_save_base = model_save_base
            st.rerun()

        if btn_ft:
            st.session_state.training_active  = True
            st.session_state._active_stage    = "finetune"
            st.session_state._ft_epochs       = ft_epochs
            st.session_state._ft_lr           = ft_lr
            st.session_state._batch_size      = batch_size
            st.session_state._data_dir        = data_dir
            st.rerun()

        if st.session_state.get("training_active", False):
            stage = st.session_state.get("_active_stage", "")
            record_resource_sample("Training")

            progress_bar        = st.progress(0, text="Initialising...")
            metrics_placeholder = st.empty()
            chart_placeholder   = st.empty()

            try:
                # ── Extract Features ──────────────────────────────────────
                if stage == "extract":
                    def _ext_progress(current, total):
                        progress_bar.progress(
                            min(current / max(total, 1), 1.0),
                            text=f"Extracting batch {current} / {total}",
                        )

                    with st.spinner("Extracting features from dataset..."):
                        result = extract_features(
                            data_dir=st.session_state.get("_feat_out", "data/features"),
                            output_dir=st.session_state.get("_feat_out", "data/features"),
                            batch_size=st.session_state.get("_batch_size", 32),
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
                            features_dir=st.session_state.get("features_dir", "data/features"),
                            epochs=st.session_state.get("_head_epochs", 10),
                            lr=st.session_state.get("_head_lr", 0.001),
                            output_path=st.session_state.get("_model_save_base",
                                                              "models/trained/latest.keras"),
                            callbacks=[cb],
                        )
                    st.session_state.head_full_path   = result["full_model_path"]
                    st.session_state.train_history    = cb.history
                    record_resource_sample("Training")
                    curves_path = _save_learning_curves(cb.history)
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
                            model_path=st.session_state.get("head_full_path", ""),
                            data_dir=st.session_state.get("_data_dir", "data/splits/train"),
                            batch_size=st.session_state.get("_batch_size", 16),
                            epochs=st.session_state.get("_ft_epochs", 10),
                            lr=st.session_state.get("_ft_lr", 0.00001),
                            callbacks=[cb],
                        )
                    st.session_state.fine_tuned_path = result["path"]
                    st.session_state.train_history   = cb.history
                    record_resource_sample("Training")
                    curves_path = _save_learning_curves(cb.history)
                    st.success(
                        f"Fine-tuned model → `{result['path']}`  "
                        f"· curves → `{curves_path}`"
                    )

            except Exception as e:
                st.error(f"Stage failed: {e}")
            finally:
                st.session_state.training_active = False
                st.rerun()

        if st.session_state.get("train_history", {}).get("epoch"):
            st.divider()
            st.markdown('<div class="section-label">Final Learning Curves</div>', unsafe_allow_html=True)
            plot_learning_curves(st.session_state.train_history)

        # ── Model Evaluation ─────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">Model Evaluation</div>', unsafe_allow_html=True)
        ev_c1, ev_c2 = st.columns([1, 1], gap="large")
        with ev_c1:
            eval_model_path = _model_picker("Model to evaluate", key="eval_model")
            eval_test_dir   = st.text_input(
                "Test directory", "data/splits/test",
                help="Folder of class-named subdirectories, separate from training data.",
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

    # =========================================================================
    # SYSTEM DASHBOARD TAB
    # =========================================================================
    with tab_system:
        render_dashboard()
