import streamlit as st
import cv2
import av
import psutil
import platform
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from config import DISEASE_CLASSES
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.utils.training_utils import DashboardCallback, get_sys_stats
from . import sidebar
from pipeline import train_model, predict_image
from streamlit_callback import StreamlitTrainCallback
import tensorflow as tf
# =============================================================================
#  LEARNING CURVES PLOTTER
# =============================================================================
def plot_learning_curves(history: dict):
    """Generate interactive learning curves after training completes."""
    df = pd.DataFrame(history)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Accuracy vs Epochs", " Loss vs Epochs"),
        shared_xaxes=True,
        horizontal_spacing=0.1
    )
    
    # Accuracy traces
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['accuracy'], mode='lines+markers', 
                   name='Train Acc', line=dict(color='#2E7D32', width=2.5)),
        row=1, col=1
    )
    if 'val_accuracy' in df:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_accuracy'], mode='lines+markers', 
                       name='Val Acc', line=dict(color='#FF6F00', width=2.5, dash='dash')),
            row=1, col=1
        )
        
    # Loss traces
    fig.add_trace(
        go.Scatter(x=df['epoch'], y=df['loss'], mode='lines+markers', 
                   name='Train Loss', line=dict(color='#C62828', width=2.5)),
        row=1, col=2
    )
    if 'val_loss' in df:
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_loss'], mode='lines+markers', 
                       name='Val Loss', line=dict(color='#1565C0', width=2.5, dash='dash')),
            row=1, col=2
        )
        
    # Layout styling
    fig.update_layout(
        height=420,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Score", gridcolor="#EEEEEE", row=1, col=1)
    fig.update_yaxes(title_text="Loss", gridcolor="#EEEEEE", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True, key="learning_curves")

# =============================================================================
# 🔹 MAIN APP ENTRY (Focus on Training Dashboard Integration)
# =============================================================================
def run_app():
    params = sidebar.render_sidebar()
    st.title("🍅 Tomato AI Diagnostics")
    st.markdown("---")
    #render_resource_monitor()
    
    tab_inference, tab_training = st.tabs(["🔍 Inference", "📊 Training Dashboard"])
    
    # ================= INFERENCE TAB =================
    with tab_inference:
        mode = st.radio("Input Source", ["📤 Static Image", " Live Camera Feed"], horizontal=True)
        
        if "Static" in mode:
            uploaded = st.file_uploader("Upload leaf photo", type=['jpg','jpeg','png'], label_visibility="collapsed")
            if uploaded:
                col_img, col_res = st.columns([2, 1])
                
                with col_img:
                    st.image(uploaded, use_container_width=True, caption="📍 Uploaded Image")
                
                with col_res:
                    st.subheader("🔎 Diagnostic Result")
                    
                    # 🔌 Replace with your actual saved model path
                    MODEL_PATH = st.text_input("Model Path", "models/final/best_model.keras")
                    
                    if st.button("Run Inference", type="primary"):
                        with st.spinner("Analyzing leaf..."):
                            try:
                                result = predict_image(MODEL_PATH, uploaded, params["conf"]/100)
                                
                                status = "✅ Healthy" if result["disease"] == "Healthy" else "️ Diseased"
                                st.markdown(f"### {status}: `{result['disease']}`")
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                                st.progress(result["confidence"])
                                
                                if not result["passes_threshold"]:
                                    st.warning("Confidence below threshold. Review manually.")
                                    
                                with st.expander("💊 Recommended Treatment"):
                                    treatments = {
                                        "Early Blight": "Remove infected leaves. Apply copper fungicide.",
                                        "Late Blight": "Improve ventilation. Apply systemic fungicide.",
                                        "Healthy": "Maintain current care routine. Monitor weekly."
                                    }
                                    st.write(treatments.get(result["disease"], "Consult agronomist."))
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
        else:
            st.info("🎥 Live camera mode requires WebRTC integration with frame-by-frame prediction.")

    # ================= TRAINING TAB =================
    with tab_training:
        st.subheader("🎯 Model Training Dashboard")
        col_cfg, col_viz = st.columns([1, 2])
        
        with col_cfg:
            st.write("**⚙️ Configuration**")
            epochs = st.slider("Epochs", 1, 50, 10)
            lr = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
            freeze = st.toggle("Freeze Base Model", value=True)
            model_path = st.text_input("Save Path", "models/trained/latest.keras")
            
            btn_train = st.button("🚀 Start Training", type="primary", use_container_width=True)
            
        with col_viz:
            st.write("**📈 Real-time Analytics**")
            progress_bar = st.progress(0, text="Starting training...")
            metrics_text = st.empty()
            
        # Initialize callback before training
        cb = StreamlitTrainCallback(progress_bar, metrics_text)
        
        if btn_train:
            #Guard against double clicking
            if st.session_state.get("training") == True:
                st.warning('⏳ Training already in progress. Please wait.')
            else :
                st.session_state.training_active = True
                st.session_state.train_history = {'epoch': [],'accuracy': [], 'val_loss': []}

                with st.spinner('🔄 Training in progress...'):
                    try:
                        trained_model = train_model(
                            epochs=epochs,
                            lr=lr,
                            freeze_base=freeze,
                            output_path=model_path,
                            callbacks=[cb]
                        )
                        st.success("✅ Training Complete!")
                        st.session_state["training"] = False
                        
                        # Save history for learning curves
                        st.session_state["train_history"] = cb.history
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.session_state["training"] = False
                    finally:
                        st.session_state.training_active = False
                        progress_bar.empty()        

        # 📊 Learning Curves (Post-Training)
        # 📊 Post-Training Learning Curves (Safe rendering)
    if "train_history" in st.session_state and st.session_state.train_history["epoch"]:
        st.divider()
        st.subheader("📊 Final Learning Curves")
        plot_learning_curves(st.session_state.train_history)  # Your existing Plotly function

    # ... [keep footer stats] ...

    # ================= FOOTER =================
    st.divider()
    st.subheader("📊 Session Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Leaves Scanned", "12", "+2")
    c2.metric("Diseases Found", "3", "-1")
    c3.metric("Healthy Rate", "75%")

if __name__ == "__main__":
    run_app()
