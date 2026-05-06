import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components import sidebar
from pipeline import train_model, predict_image
from streamlit_callback import StreamlitTrainCallback
from src.utils.recommendations import get_recommendation


def plot_learning_curves(history: dict):
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
        height=420,
        template="plotly_white",
        hovermode="x unified",
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


def run_app():
    params = sidebar.render_sidebar()
    st.title("🍅 Tomato AI Diagnostics")
    st.markdown("---")

    tab_inference, tab_training = st.tabs(["🔍 Inference", "📊 Training Dashboard"])

    # ===== INFERENCE TAB =====
    with tab_inference:
        mode = st.radio(
            "Input Source", ["📤 Static Image", "🎥 Live Camera Feed"], horizontal=True
        )

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
                                    model_path,
                                    uploaded,
                                    params["confidence"] / 100,
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

                                # Track session stats
                                st.session_state.scans = st.session_state.get("scans", 0) + 1
                                if result["disease"] != "Healthy":
                                    st.session_state.diseases = (
                                        st.session_state.get("diseases", 0) + 1
                                    )

                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
        else:
            st.info("🎥 Live camera mode requires WebRTC integration with frame-by-frame prediction.")

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
            data_dir = st.text_input("Dataset Directory", "data/train")
            model_path = st.text_input("Save Path", "models/trained/latest.keras")
            btn_train = st.button("🚀 Start Training", type="primary", use_container_width=True)

        with col_viz:
            st.write("**📈 Real-time Metrics**")
            progress_bar = st.progress(0, text="Waiting to start...")
            metrics_text = st.empty()

        if btn_train:
            if st.session_state.get("training"):
                st.warning("⏳ Training already in progress. Please wait.")
            else:
                st.session_state.training = True
                cb = StreamlitTrainCallback(progress_bar, metrics_text)
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
                    st.success("✅ Training complete!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                finally:
                    st.session_state.training = False
                    progress_bar.empty()

    # Post-training learning curves (rendered outside the tab so they persist on rerun)
    history = st.session_state.get("train_history", {})
    if history.get("epoch"):
        st.divider()
        st.subheader("📊 Learning Curves")
        plot_learning_curves(history)

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
