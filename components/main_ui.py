import threading

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
from config import DISEASE_CLASSES
from pipeline import predict_image, train_model
from src.utils.recommendations import get_recommendation
from streamlit_callback import StreamlitTrainCallback

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# =============================================================================
# WEBCAM PREDICTOR
# =============================================================================

class FramePredictor:
    """
    Callable for streamlit_webrtc's video_frame_callback.
    Loads the model once, runs inference every N frames, and draws
    the result overlay directly on the video frame.
    Runs in a background thread — all model access is lock-guarded.
    """

    INFERENCE_EVERY = 20  # frames between predictions (~every ~0.67s at 30fps)

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

        if disease == "Healthy":
            color = (0, 180, 0)       # green
        elif passes:
            color = (0, 0, 210)       # red
        else:
            color = (0, 140, 255)     # orange — below threshold

        # Dark header bar
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, 0), (w, 60), color, 3)

        label = f"{disease}  {conf:.1%}"
        cv2.putText(frame, label, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

        if not passes:
            cv2.putText(frame, "Below confidence threshold", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =============================================================================
# LEARNING CURVES
# =============================================================================

def plot_learning_curves(history: dict):
    if not history or not history.get("epoch"):
        st.warning("No training data available.")
        return

    df = pd.DataFrame(history).dropna()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy vs Epochs", "Loss vs Epochs"),
        shared_xaxes=True,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["accuracy"], mode="lines+markers",
                   name="Train Acc", line=dict(color="#2E7D32", width=2.5)),
        row=1, col=1,
    )
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(
            go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers",
                       name="Val Acc", line=dict(color="#FF6F00", width=2.5, dash="dash")),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["loss"], mode="lines+markers",
                   name="Train Loss", line=dict(color="#C62828", width=2.5)),
        row=1, col=2,
    )
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(
            go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers",
                       name="Val Loss", line=dict(color="#1565C0", width=2.5, dash="dash")),
            row=1, col=2,
        )

    fig.update_layout(
        height=420, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Score", gridcolor="#EEEEEE", row=1, col=1)
    fig.update_yaxes(title_text="Loss", gridcolor="#EEEEEE", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True, key="learning_curves")


@st.cache_resource
def load_cached_model(path: str):
    return tf.keras.models.load_model(path)


def _save_learning_curves(history: dict):
    """Save the final learning curve figure to outputs/ as an HTML file."""
    import os
    from datetime import datetime
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/learning_curves_{timestamp}.html"

    df = pd.DataFrame(history).dropna()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy vs Epochs", "Loss vs Epochs"),
        horizontal_spacing=0.12,
    )
    epochs = df["epoch"]
    fig.add_trace(go.Scatter(x=epochs, y=df["accuracy"], mode="lines+markers",
                             name="Train Acc", line=dict(color="#2E7D32", width=2.5)), row=1, col=1)
    if "val_accuracy" in df and df["val_accuracy"].notna().any():
        fig.add_trace(go.Scatter(x=epochs, y=df["val_accuracy"], mode="lines+markers",
                                 name="Val Acc", line=dict(color="#FF6F00", width=2.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=df["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#C62828", width=2.5)), row=1, col=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        fig.add_trace(go.Scatter(x=epochs, y=df["val_loss"], mode="lines+markers",
                                 name="Val Loss", line=dict(color="#1565C0", width=2.5, dash="dash")), row=1, col=2)

    fig.update_layout(height=420, template="plotly_white", hovermode="x unified",
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Epoch", dtick=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.write_html(path)
    return path


# =============================================================================
# MAIN APP
# =============================================================================

def run_app():
    params = sidebar.render_sidebar()
    st.title("🍅 Tomato AI Diagnostics")
    st.markdown("---")

    tab_inference, tab_training = st.tabs(["🔍 Inference", "📊 Training Dashboard"])

    # ===== INFERENCE TAB =====
    with tab_inference:
        mode = st.radio(
            "Input Source", ["📤 Static Image", "📷 Live Webcam"], horizontal=True
        )

        # ----- Static image -----
        if "Static" in mode:
            uploaded = st.file_uploader(
                "Upload leaf photo", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded:
                col_img, col_res = st.columns([2, 1])
                with col_img:
                    st.image(uploaded, use_container_width=True, caption="Uploaded Image")
                with col_res:
                    st.subheader("🔎 Diagnostic Result")
                    model_path = st.text_input("Model Path", "models/final/best_model.keras")
                    if st.button("Run Inference", type="primary"):
                        with st.spinner("Analyzing leaf..."):
                            try:
                                result = predict_image(
                                    model_path, uploaded, params["confidence"] / 100
                                )
                                status = (
                                    "✅ Healthy"
                                    if result["disease"] == "Healthy"
                                    else "⚠️ Diseased"
                                )
                                st.markdown(f"### {status}: `{result['disease']}`")
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                                st.progress(result["confidence"])
                                if not result["passes_threshold"]:
                                    st.warning("Confidence below threshold — review manually.")
                                with st.expander("💊 Recommended Treatment"):
                                    st.write(get_recommendation(result["disease"]))
                                with st.expander("📊 All Class Probabilities"):
                                    prob_df = (
                                        pd.DataFrame(
                                            result["all_probs"].items(),
                                            columns=["Disease", "Probability"],
                                        )
                                        .sort_values("Probability", ascending=False)
                                        .reset_index(drop=True)
                                    )
                                    st.dataframe(prob_df, use_container_width=True)
                                st.session_state.scans = st.session_state.get("scans", 0) + 1
                                if result["disease"] != "Healthy":
                                    st.session_state.diseases = (
                                        st.session_state.get("diseases", 0) + 1
                                    )
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")

        # ----- Live webcam -----
        else:
            st.info(
                "Point your **webcam** at a tomato leaf. "
                "The model runs every 20 frames and draws the diagnosis directly on the video."
            )

            model_path_cam = st.text_input(
                "Model Path", "models/final/best_model.keras", key="cam_model_path"
            )
            conf_cam = params["confidence"] / 100

            # Rebuild predictor only when model path changes
            stored_path = st.session_state.get("_cam_model_loaded")
            if stored_path != model_path_cam:
                st.session_state.frame_predictor = FramePredictor(model_path_cam, conf_cam)
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

            # Show last prediction below the video as text
            if predictor._last_result:
                r = predictor._last_result
                col_d, col_c = st.columns(2)
                col_d.metric("Last Diagnosis", r["disease"])
                col_c.metric("Confidence", f"{r['confidence']:.1%}")
                if not r["passes"]:
                    st.warning("Confidence below threshold — move camera closer or improve lighting.")
                with st.expander("💊 Recommended Treatment"):
                    st.write(get_recommendation(r["disease"]))

    # ===== TRAINING TAB =====
    with tab_training:
        st.subheader("🎯 Model Training Dashboard")
        col_cfg, col_viz = st.columns([1, 2])

        with col_cfg:
            st.write("**⚙️ Configuration**")
            epochs = st.slider("Epochs", 1, 50, 10)
            lr = st.select_slider(
                "Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001
            )
            freeze = st.toggle("Freeze Base Model", value=True)
            data_dir = st.text_input("Dataset Directory", "data/splits/train")
            model_path = st.text_input("Save Path", "models/trained/latest.keras")
            btn_train = st.button(
                "🚀 Start Training", type="primary", use_container_width=True,
                disabled=st.session_state.get("training_active", False)
            )

        with col_viz:
            st.write("**📈 Real-time Learning Curves**")
            progress_bar = st.progress(0, text="Waiting to start...")
            metrics_text = st.empty()
            chart_placeholder = st.empty()

        if btn_train:
            st.session_state.training_active = True
            st.rerun()

        if st.session_state.get("training_active", False):
            progress_bar = st.progress(0, text="Initializing training...")
            metrics_text = st.empty()
            chart_placeholder = st.empty()
            cb = StreamlitTrainCallback(progress_bar, metrics_text, chart_placeholder)
            with st.spinner("🔄 Training in progress..."):
                try:
                    train_model(
                        epochs=epochs,
                        lr=lr,
                        freeze_base=freeze,
                        data_dir=data_dir,
                        output_path=model_path,
                        callbacks=[cb],
                    )
                    st.session_state.train_history = cb.history
                    _save_learning_curves(cb.history)
                    st.success("✅ Training Complete!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                finally:
                    st.session_state.training_active = False
                    st.rerun()

        if st.session_state.get("train_history", {}).get("epoch"):
            st.divider()
            st.subheader("📊 Final Learning Curves")
            plot_learning_curves(st.session_state.train_history)

    # ===== FOOTER =====
    st.divider()
    st.subheader("📊 Session Statistics")
    c1, c2, c3 = st.columns(3)
    scans = st.session_state.get("scans", 0)
    diseases = st.session_state.get("diseases", 0)
    healthy_rate = f"{((scans - diseases) / scans * 100):.0f}%" if scans else "N/A"
    c1.metric("Leaves Scanned", scans)
    c2.metric("Diseases Found", diseases)
    c3.metric("Healthy Rate", healthy_rate)
