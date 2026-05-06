import threading
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
from plotly.subplots import make_subplots
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from components import sidebar
from config import DISEASE_CLASSES, DISEASE_DISPLAY
from pipeline import predict_image, train_model, train_two_stage
from src.utils.recommendations import get_recommendation
from streamlit_callback import StreamlitTrainCallback

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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
        arr = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
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
                    except Exception:
                        pass
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
    import os
    from datetime import datetime
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
    params = sidebar.render_sidebar()

    _hero()

    tab_inference, tab_training = st.tabs(["  Inference  ", "  Training Dashboard  "])

    # =========================================================================
    # INFERENCE TAB
    # =========================================================================
    with tab_inference:
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
                model_path = st.text_input(
                    "Model path", "models/trained/latest.keras",
                    placeholder="models/trained/run_xxx/stage2_ft.keras",
                )
                st.markdown(
                    _model_status_html(model_path), unsafe_allow_html=True
                )
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

                run_btn = st.button(
                    "Run Diagnosis", type="primary",
                    use_container_width=True,
                    disabled=uploaded is None,
                )

                if uploaded is None:
                    st.markdown(
                        """
                        <div style="
                            background:#F8FAFC;border:2px dashed #E2E8F0;
                            border-radius:12px;padding:2.5rem 1.5rem;
                            text-align:center;margin-top:1rem;
                        ">
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
                            result = predict_image(
                                model_path, uploaded, params["confidence"] / 100
                            )
                            _diagnosis_card(result)
                            st.session_state.scans = st.session_state.get("scans", 0) + 1
                            if "healthy" not in result["disease"].lower():
                                st.session_state.diseases = st.session_state.get("diseases", 0) + 1
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
                st.info(
                    "Point your webcam at a tomato leaf. "
                    "Diagnosis updates every ~20 frames.",
                    icon="📷",
                )
                st.warning(
                    "Click **Allow** when your browser asks for camera permission. "
                    "If denied, click the camera icon in your address bar and set it to Allow, then refresh.",
                    icon="🔒",
                )
                model_path_cam = st.text_input(
                    "Model path", "models/trained/latest.keras", key="cam_model_path"
                )
                if st.session_state.get("_cam_model_loaded") != model_path_cam:
                    st.session_state.frame_predictor = FramePredictor(
                        model_path_cam, params["confidence"] / 100
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
                st.markdown(
                    '<div class="section-label">Last Prediction</div>',
                    unsafe_allow_html=True,
                )
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
                        <div style="
                            background:#F8FAFC;border:2px dashed #E2E8F0;
                            border-radius:12px;padding:3rem 1.5rem;
                            text-align:center;
                        ">
                            <div style="font-size:2rem;margin-bottom:0.5rem;">📷</div>
                            <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;">
                                Start the camera to see live predictions
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # =========================================================================
    # TRAINING TAB
    # =========================================================================
    with tab_training:
        col_cfg, col_viz = st.columns([2, 3], gap="large")

        with col_cfg:
            st.markdown(
                '<div class="section-label">Training Configuration</div>',
                unsafe_allow_html=True,
            )

            two_stage = st.toggle(
                "Two-Stage Training",
                value=True,
                help="Stage 1: freeze base (feature extraction). Stage 2: unfreeze (fine-tuning).",
            )
            data_dir = st.text_input("Dataset directory", "data/splits/train")
            model_save_path = st.text_input("Save to", "models/trained/latest.keras")

            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

            if two_stage:
                st.markdown(
                    """
                    <div class="stage-header">
                        <p class="stage-title">Stage 1 — Feature Extraction</p>
                        <p class="stage-desc">Base frozen &nbsp;·&nbsp; trains classification head only</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fe_epochs = st.slider("Epochs", 1, 50, 10, key="fe_epochs")
                fe_lr = st.select_slider(
                    "Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001, key="fe_lr"
                )

                st.markdown(
                    """
                    <div class="stage-header" style="margin-top:0.75rem;">
                        <p class="stage-title">Stage 2 — Fine-Tuning</p>
                        <p class="stage-desc">Base unfrozen &nbsp;·&nbsp; end-to-end training</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                ft_epochs = st.slider("Epochs", 1, 50, 10, key="ft_epochs")
                ft_lr = st.select_slider(
                    "Learning rate", [0.00001, 0.0001, 0.001], value=0.0001, key="ft_lr"
                )
            else:
                fe_epochs = st.slider("Epochs", 1, 50, 10)
                fe_lr = st.select_slider(
                    "Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001
                )
                freeze = st.toggle("Freeze Base Model", value=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            btn_train = st.button(
                "Start Training",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.get("training_active", False),
            )

            # Saved model paths from last run
            if st.session_state.get("last_stage2_path"):
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-label">Saved Models</div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.get("last_stage1_path"):
                    st.markdown(
                        f'<p style="font-size:0.78rem;color:#6B7280;margin:0;">Stage 1</p>'
                        f'<code style="font-size:0.72rem;">{st.session_state.last_stage1_path}</code>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<p style="font-size:0.78rem;color:#6B7280;margin:4px 0 0 0;">Stage 2 (use for inference)</p>'
                    f'<code style="font-size:0.72rem;">{st.session_state.last_stage2_path}</code>',
                    unsafe_allow_html=True,
                )

        with col_viz:
            st.markdown(
                '<div class="section-label">Live Training Metrics</div>',
                unsafe_allow_html=True,
            )
            progress_bar = st.progress(0, text="Waiting to start...")
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()

            if not st.session_state.get("training_active") and not st.session_state.get("train_history"):
                chart_placeholder.markdown(
                    """
                    <div style="
                        background:white;border-radius:14px;
                        padding:4rem 2rem;text-align:center;
                        border:1px solid #EEF0F2;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);
                    ">
                        <div style="font-size:2.5rem;margin-bottom:0.75rem;">📈</div>
                        <div style="font-weight:600;color:#16213E;margin-bottom:0.4rem;">
                            No training data yet
                        </div>
                        <div style="color:#94A3B8;font-size:0.85rem;">
                            Configure parameters and click Start Training.<br>
                            Accuracy and loss curves will appear here in real time.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if btn_train:
            st.session_state.training_active = True
            st.session_state.training_two_stage = two_stage
            st.rerun()

        if st.session_state.get("training_active", False):
            progress_bar = st.progress(0, text="Initialising...")
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            cb = StreamlitTrainCallback(progress_bar, metrics_placeholder, chart_placeholder)

            with st.spinner("Training in progress..."):
                try:
                    if st.session_state.get("training_two_stage"):
                        result = train_two_stage(
                            fe_epochs=fe_epochs,
                            fe_lr=fe_lr,
                            ft_epochs=ft_epochs,
                            ft_lr=ft_lr,
                            data_dir=data_dir,
                            output_path=model_save_path,
                            fe_callbacks=[cb],
                            ft_callbacks=[cb],
                        )
                        st.session_state.last_stage1_path = result["stage1_path"]
                        st.session_state.last_stage2_path = result["stage2_path"]
                    else:
                        _, saved_path = train_model(
                            epochs=fe_epochs,
                            lr=fe_lr,
                            freeze_base=freeze,
                            data_dir=data_dir,
                            output_path=model_save_path,
                            callbacks=[cb],
                        )
                        st.session_state.last_stage1_path = None
                        st.session_state.last_stage2_path = saved_path

                    st.session_state.train_history = cb.history
                    curves_path = _save_learning_curves(cb.history)
                    st.success(f"Training complete. Curves saved to `{curves_path}`")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                finally:
                    st.session_state.training_active = False
                    st.rerun()

        if st.session_state.get("train_history", {}).get("epoch"):
            st.divider()
            st.markdown(
                '<div class="section-label">Final Learning Curves</div>',
                unsafe_allow_html=True,
            )
            plot_learning_curves(st.session_state.train_history)
