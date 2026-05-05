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
    


# 🔹 Cache model loader (prevents reloading on every UI interaction)
@st.cache_resource
def load_cached_model(path: str):
    return tf.keras.models.load_model(path)

# ... [keep your existing sidebar & resource monitor code] ...

def run_app():
    params = sidebar.render_sidebar()
    st.title("🍅 Tomato AI Diagnostics")
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
            progress_bar = st.progress(0,0, text="Starting training...")
            metrics_text = st.empty()
            chartz_box = st.empty()
            
        if btn_train:
            #Guard against double clicking
            if st.session_state.get("training") == True:
                st.warning('⏳ Training already in progress. Please wait.')
            else :
                st.session_state.training_active = True
                st.session_state.train_history = {'epoch': [],'accuracy': [], 'val_los': []}

                with st.spinner('🔄 Training in progress...'):
                    try:
                        for e in range(epochs):
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
                        
                        # Update UI safely
                        progress_bar.progress((e + 1) / epochs, text=f"Epoch {e+1}/{epochs}")
                        metrics_box.markdown(f"🔹 **Acc:** `{acc:.3f}` | **Loss:** `{loss:.3f}`")
                        
                        # Store for post-training curves
                        st.session_state.train_history["epoch"].append(e + 1)
                        st.session_state.train_history["accuracy"].append(acc)
                        st.session_state.train_history["val_loss"].append(loss)
                        
                    st.success("✅ Training Complete!")
                except Exception as ex:
                    st.error(f"Training failed: {ex}")
                finally:
                    st.session_state.training_active = False
                    progress_bar.empty()
            # Initialize callback
           ''' cb = StreamlitTrainCallback(progress_bar, metrics_text)'''        

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
'''import streamlit as st
import cv2
import av
import psutil
import platform
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from config import DISEASE_CLASSES
import time
from src.utils.training_utils import DashboardCallback, get_sys_stats

# =============================================================================
# 🔹 AUTO-UPDATING SYSTEM MONITOR (Runs independently every 2 seconds)
# =============================================================================
@st.fragment(run_every=2)
def render_resource_monitor():
    """Live system health monitor that updates without full page reruns."""
    stats = get_sys_stats()
    with st.sidebar.expander(" System Pulse", expanded=True):
        st.caption("CPU Usage")
        st.progress(stats['cpu'] / 100)
        st.caption("RAM Usage")
        st.progress(stats['ram'] / 100)
        
        if stats['ram'] > 90:
            st.warning("⚠️ Critical Memory Usage!")
        elif stats['ram'] > 75:
            st.warning("⚠️ High Memory Usage")
            
        if stats.get('temp'):
            st.caption(f"🌡️ CPU Temp: {stats['temp']}°C")
            if stats['temp'] > 80:
                st.error("⚠️ Overheating Warning!")

# =============================================================================
#  SIDEBAR DETECTION CONTROLS
# =============================================================================
def render_sidebar() -> dict:
    """Render detection settings and return parameters."""
    with st.sidebar:
        st.header("🔍 Detection Settings", divider=True)
        
        # Confidence Threshold
        conf = st.slider(
            "Confidence Threshold", 0.0, 100.0, 50.0, 1.0,
            help="Minimum probability to report a disease."
        )
        st.write(f"**Current:** {conf:.1f}%")
        
        # Disease Classes
        classes = st.multiselect(
            "Active Disease Filters", DISEASE_CLASSES,
            default=DISEASE_CLASSES[:3]
        )
        if not classes:
            st.warning("⚠️ No classes selected!")
            
        st.divider()
        
        # ROI Adjustment
        st.header(" ROI Adjustment", divider=True)
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.slider("X Min (Left)", 0.0, 1.0, 0.1, 0.01)
            y_min = st.slider("Y Min (Top)", 0.0, 1.0, 0.1, 0.01)
        with col2:
            x_max = st.slider("X Max (Right)", 0.0, 1.0, 0.9, 0.01)
            y_max = st.slider("Y Max (Bottom)", 0.0, 1.0, 0.9, 0.01)
            
        if x_min >= x_max or y_min >= y_max:
            st.error("⚠️ Invalid ROI: Min must be < Max")
            
    return {
        "conf": conf,
        "classes": classes,
        "roi": (x_min, y_min, x_max, y_max)
    }

# =============================================================================
#  VISUALIZATION HELPERS
# =============================================================================
def draw_roi_overlay(image, roi):
    """Draw ROI rectangle on image for preview."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    x1, y1, x2, y2 = roi
    draw.rectangle([x1*w, y1*h, x2*w, y2*h], outline="#FF4B4B", width=4)
    return image

# =============================================================================
# 🔹 MAIN APP ENTRY
# =============================================================================
def run_app():
    st.set_page_config(page_title="Tomato AI Diagnostics", layout="wide")
    st.title(" Tomato AI Diagnostics")
    
    #  Initialize auto-updating monitor
    render_resource_monitor()
    
    # ⚙️ Load sidebar parameters
    params = render_sidebar()
    
    # 📑 Main Workspace Tabs
    tab_inference, tab_training = st.tabs([" Inference", "📊 Training Dashboard"])
    
    # ================= INFERENCE TAB =================
    with tab_inference:
        mode = st.radio("Input Source", ["📤 Static Image", "🎥 Live Camera Feed"], horizontal=True)
        
        if "Static" in mode:
            uploaded = st.file_uploader("Upload leaf photo", type=['jpg','jpeg','png'], label_visibility="collapsed")
            if uploaded:
                col_img, col_res = st.columns([2, 1])
                img = Image.open(uploaded).convert("RGB")
                
                with col_img:
                    st.image(draw_roi_overlay(img, params["roi"]), use_container_width=True, caption="📍 Analysis Region")
                
                with col_res:
                    st.subheader("Diagnostic Result")
                    # 🔌 TODO: Replace with actual model.predict()
                    pred, conf_val = "Early Blight", 88.5
                    st.markdown(f"### :red[{pred}]")
                    st.metric("Confidence", f"{conf_val}%")
                    st.progress(conf_val/100)
                    
                    with st.expander("💊 Recommended Treatment"):
                        st.write("• Remove infected leaves immediately")
                        st.write("• Apply copper-based fungicide")
                        st.write("• Improve air circulation & reduce humidity")
        else:
            st.info("📷 Camera feed active. Grant browser permissions.")
            webrtc_streamer(
                key="tomato-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                video_frame_callback=lambda f: av.VideoFrame.from_ndarray(
                    cv2.rectangle(
                        f.to_ndarray(format="bgr24"),
                        (int(params["roi"][0]*640), int(params["roi"][1]*480)),
                        (int(params["roi"][2]*640), int(params["roi"][3]*480)),
                        (75, 167, 40), 3
                    ), format="bgr24"),
                media_stream_constraints={"video": True, "audio": False}
            )
            
    # ================= TRAINING TAB =================
    with tab_training:
        st.subheader("🎯 Model Training Dashboard")
        col_cfg, col_viz = st.columns([1, 3])
        
        with col_cfg:
            st.write("**️ Configuration**")
            epochs = st.slider("Epochs", 1, 50, 10)
            btn_train = st.button("🚀 Start Training", type="primary", use_container_width=True)
            
        with col_viz:
            st.write("**📈 Real-time Analytics**")
            m_place = st.empty()
            c_place = st.empty()
            
        if btn_train:
            cb = DashboardCallback(m_place, c_place)
            with st.spinner("Training model... (Simulated)"):
                for e in range(epochs):
                    time.sleep(0.5)  # Simulate epoch
                    cb.on_epoch_end(e, logs={
                        'accuracy': 0.5 + (e/20),
                        'val_loss': 1.0 - (e/20)
                    })
            st.success("✅ Training Complete!")
            
    # ================= FOOTER STATS =================
    st.divider()
    st.subheader("📊 Session Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Leaves Scanned", "12", "+2")
    c2.metric("Diseases Found", "3", "-1")
    c3.metric("Healthy Rate", "75%")

if __name__ == "__main__":
    run_app()
    '''
'''import streamlit as st
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

# ... (keep render_resource_monitor and render_sidebar as they are) ...

def draw_roi_overlay(image, roi):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle([roi[0]*w, roi[1]*h, roi[2]*w, roi[3]*h], outline="#FF4B4B", width=5)
    return image

class EnhancedDashboardCallback:
    """Enhanced callback with metrics history for interactive charts."""
    
    def __init__(self, chart_placeholder, metrics_placeholder):
        self.chart_placeholder = chart_placeholder
        self.metrics_placeholder = metrics_placeholder
        self.history = {
            'epoch': [],
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': [],
            'precision': [],
            'recall': []
        }
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Store metrics
        self.history['epoch'].append(epoch + 1)
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['precision'].append(logs.get('precision', 0))
        self.history['recall'].append(logs.get('recall', 0))
        
        # Update visualizations
        self._update_charts()
        self._update_metrics(logs)
    
    def _update_charts(self):
        """Create and display interactive Plotly charts."""
        df = pd.DataFrame(self.history)
        
        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Over Time', 'Loss Over Time', 
                          'Precision & Recall', 'Training vs Validation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['accuracy'], 
                      mode='lines+markers', name='Train Acc',
                      line=dict(color='#2E7D32', width=3),
                      marker=dict(size=6, color='#2E7D32')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_accuracy'], 
                      mode='lines+markers', name='Val Acc',
                      line=dict(color='#FF6F00', width=3, dash='dash'),
                      marker=dict(size=6, color='#FF6F00')),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['loss'], 
                      mode='lines+markers', name='Train Loss',
                      line=dict(color='#C62828', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_loss'], 
                      mode='lines+markers', name='Val Loss',
                      line=dict(color='#1565C0', width=3, dash='dash')),
            row=1, col=2
        )
        
        # Precision & Recall
        if any(self.history['precision']):
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['precision'], 
                          mode='lines+markers', name='Precision',
                          line=dict(color='#7B1FA2', width=3)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['recall'], 
                          mode='lines+markers', name='Recall',
                          line=dict(color='#00897B', width=3)),
                row=2, col=1
            )
        
        # Combined view
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['accuracy'], 
                      mode='lines', name='Accuracy',
                      line=dict(color='#2E7D32', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_loss'], 
                      mode='lines', name='Val Loss',
                      line=dict(color='#1565C0', width=2)),
            row=2, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        # Display chart
        self.chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    def _update_metrics(self, logs):
        """Update real-time metrics display."""
        elapsed = time.time() - self.start_time
        current_epoch = len(self.history['epoch'])
        
        # Calculate improvements
        if len(self.history['accuracy']) > 1:
            acc_improvement = self.history['accuracy'][-1] - self.history['accuracy'][0]
            loss_change = self.history['loss'][0] - self.history['loss'][-1]
        else:
            acc_improvement = 0
            loss_change = 0
        
        metrics_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3 style='margin: 0 0 15px 0;'>📊 Training Progress</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 12px; opacity: 0.9;'>Current Epoch</div>
                    <div style='font-size: 24px; font-weight: bold;'>{current_epoch}</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 12px; opacity: 0.9;'>Accuracy</div>
                    <div style='font-size: 24px; font-weight: bold;'>{logs.get('accuracy', 0)*100:.1f}%</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 12px; opacity: 0.9;'>Val Accuracy</div>
                    <div style='font-size: 24px; font-weight: bold;'>{logs.get('val_accuracy', 0)*100:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
            <div style='background: #f0f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #2E7D32;'>
                <div style='font-size: 12px; color: #666;'>Training Loss</div>
                <div style='font-size: 20px; font-weight: bold; color: #C62828;'>{logs.get('loss', 0):.4f}</div>
                <div style='font-size: 11px; color: #2E7D32;'>↓ {loss_change:.4f} improvement</div>
            </div>
            <div style='background: #f0f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #1565C0;'>
                <div style='font-size: 12px; color: #666;'>Validation Loss</div>
                <div style='font-size: 20px; font-weight: bold; color: #1565C0;'>{logs.get('val_loss', 0):.4f}</div>
            </div>
            <div style='background: #f0f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #7B1FA2;'>
                <div style='font-size: 12px; color: #666;'>Precision</div>
                <div style='font-size: 20px; font-weight: bold; color: #7B1FA2;'>{logs.get('precision', 0)*100:.1f}%</div>
            </div>
            <div style='background: #f0f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #00897B;'>
                <div style='font-size: 12px; color: #666;'>Recall</div>
                <div style='font-size: 20px; font-weight: bold; color: #00897B;'>{logs.get('recall', 0)*100:.1f}%</div>
            </div>
        </div>
        
        <div style='margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px;'>
            <div style='font-size: 12px; color: #856404;'>
                ⏱️ Elapsed Time: {elapsed:.1f}s | 
                Avg Epoch: {elapsed/max(current_epoch,1):.2f}s
            </div>
        </div>
        """
        
        self.metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)


def run_app():
    params = sidebar.render_sidebar()
    st.title("🍅 Tomato AI Diagnostics")
    st.markdown("---")
    render_resource_monitor()
    
    tab1, tab2 = st.tabs(['🔍 Inference', '📊 Training Dashboard'])
    
    with tab1:
        # Mode Selection with visual icons
        mode = st.radio("Select Input Source", ["📤 Static Image", "🎥 Live Camera Feed"], horizontal=True)

        if "Static" in mode:
            uploaded = st.file_uploader("Upload leaf photo", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            if uploaded:
                col1, col2 = st.columns([2, 1])
                img = Image.open(uploaded).convert("RGB")
                
                with col1:
                    st.image(draw_roi_overlay(img, params["roi"]), use_container_width=True, caption="Analysis Region")
                
                with col2:
                    st.subheader("Diagnostic Result")
                    # Mock result - Replace with your model.predict() logic
                    top_pred = "Early Blight"
                    conf_val = 88.5
                    
                    status_color = "red" if top_pred != "Healthy" else "green"
                    st.markdown(f"### :{status_color}[{top_pred}]")
                    st.metric("Confidence", f"{conf_val}%")
                    st.progress(conf_val/100)
                    
                    with st.expander("💊 Recommended Treatment"):
                        st.write("1. Remove infected leaves immediately.")
                        st.write("2. Apply copper-based fungicide.")
                        st.write("3. Ensure better air circulation.")

        else:
            st.info("Ensure your browser has camera permissions enabled.")
            webrtc_streamer(
                key="tomato-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun://google.com"]}]},
                video_frame_callback=lambda f: av.VideoFrame.from_ndarray(
                    cv2.rectangle(f.to_ndarray(format="bgr24"), 
                    (int(params["roi"][0]*640), int(params["roi"][1]*480)), 
                    (int(params["roi"][2]*640), int(params["roi"][3]*480)), (75, 167, 40), 3), format="bgr24"),
                media_stream_constraints={"video": True, "audio": False}
            )

    with tab2:
        st.markdown("### 🎯 Model Training Dashboard")
        st.markdown("Monitor training metrics in real-time with interactive visualizations")
        st.divider()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("⚙️ Configuration")
            epochs = st.slider("Number of Epochs", 1, 50, 10)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=1)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            
            st.divider()
            btn = st.button("🚀 Start Training", type="primary", use_container_width=True)
            
            # Training info
            with st.expander("ℹ️ Training Info"):
                st.write("""
                - **Dataset**: Tomato Leaf Diseases
                - **Classes**: 10 disease types + Healthy
                - **Validation Split**: 20%
                - **Early Stopping**: Enabled (patience=5)
                """)
        
        with col2:
            st.subheader("📈 Real-time Analytics")
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

        if btn:
            # Initialize enhanced callback
            cb = EnhancedDashboardCallback(chart_placeholder, metrics_placeholder)
            
            with st.spinner("Training model... This may take a few minutes."):
                # Simulated training - Replace with actual model.fit()
                for e in range(epochs):
                    time.sleep(0.5)  # Simulate epoch work
                    
                    # Simulate realistic training metrics
                    import random
                    epoch_logs = {
                        'accuracy': 0.6 + (e * 0.03) + random.uniform(-0.02, 0.02),
                        'val_accuracy': 0.58 + (e * 0.028) + random.uniform(-0.03, 0.02),
                        'loss': max(0.3, 1.5 - (e * 0.08) + random.uniform(-0.05, 0.05)),
                        'val_loss': max(0.35, 1.6 - (e * 0.07) + random.uniform(-0.05, 0.05)),
                        'precision': 0.65 + (e * 0.025) + random.uniform(-0.02, 0.02),
                        'recall': 0.62 + (e * 0.027) + random.uniform(-0.02, 0.02)
                    }
                    
                    # Clamp values between 0 and 1
                    for key in epoch_logs:
                        epoch_logs[key] = min(1.0, max(0.0, epoch_logs[key]))
                    
                    cb.on_epoch_end(e, logs=epoch_logs)
                    
                    # Check for early stopping (simulated)
                    if len(cb.history['val_loss']) > 5:
                        if cb.history['val_loss'][-1] > cb.history['val_loss'][-5]:
                            st.warning("⚠️ Early stopping triggered at epoch " + str(e+1))
                            break
            
            st.success("✅ Training Complete!")
            
            # Final summary
            final_metrics = {
                'Final Accuracy': f"{cb.history['accuracy'][-1]*100:.2f}%",
                'Final Val Accuracy': f"{cb.history['val_accuracy'][-1]*100:.2f}%",
                'Best Epoch': str(cb.history['val_accuracy'].index(max(cb.history['val_accuracy'])) + 1),
                'Total Time': f"{time.time() - cb.start_time:.1f}s"
            }
            
            st.metric_container = st.columns(4)
            for i, (metric, value) in enumerate(final_metrics.items()):
                st.metric_container[i].metric(metric, value)

    st.divider()
    # Session Statistics
    st.subheader("📊 Session Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Leaves Scanned", "12", "+2")
    c2.metric("Diseases Found", "3", "-1")
    c3.metric("Healthy Rate", "75%")


if __name__ == "__main__":
    run_app()
'''
