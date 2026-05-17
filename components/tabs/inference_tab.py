"""Inference tab — webcam/upload input, model selection, diagnosis card, Grad-CAM."""
import hashlib
import io
from datetime import datetime
from pathlib import Path

import streamlit as st

from components.ui_helpers import html, model_picker, model_info_html, model_status_html
from components.webcam_selector import webcam_selector
from components.system_dashboard import record_resource_sample
from config import DISEASE_CLASSES, DISEASE_DISPLAY, DISEASE_SEVERITY
from pipeline import predict_image, log_inference, compute_gradcam
from src.utils.recommendations import get_recommendation


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(img_bytes_or_file, model_path: str, confidence: float) -> dict | None:
    """Run prediction and update session counters. Returns result dict or None."""
    try:
        from io import BytesIO
        src    = BytesIO(img_bytes_or_file) if isinstance(img_bytes_or_file, bytes) else img_bytes_or_file
        result = predict_image(model_path, src, confidence / 100)
        try:
            st.session_state["last_img_bytes"]  = (
                img_bytes_or_file if isinstance(img_bytes_or_file, bytes)
                else img_bytes_or_file.getvalue()
            )
            st.session_state["last_model_path"] = model_path
        except Exception:
            pass
        st.session_state.pop("gradcam_result", None)
        record_resource_sample("Inference")
        log_inference(
            result["disease"], result["confidence"], Path(model_path).name,
            entropy=result.get("entropy"), is_leaf=result.get("is_leaf"),
        )
        st.session_state.scans = st.session_state.get("scans", 0) + 1
        if "healthy" not in result["disease"].lower():
            st.session_state.diseases = st.session_state.get("diseases", 0) + 1
        st.session_state.setdefault("diagnosis_log", []).append({
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Disease":    DISEASE_DISPLAY.get(result["disease"], result["disease"]),
            "Confidence": f"{result['confidence']:.1%}",
            "Status":     "Pass" if result.get("passes_threshold") else "Low",
            "Model":      Path(model_path).name,
        })
        return result
    except FileNotFoundError:
        st.error("Model file not found. Train a model or enter a valid path.")
    except Exception as e:
        st.error(f"Inference failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Diagnosis card
# ---------------------------------------------------------------------------

def _prob_bars_html(all_probs: dict, top_disease: str) -> str:
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:8]
    rows = []
    for cls, prob in sorted_probs:
        label      = DISEASE_DISPLAY.get(cls, cls)
        is_active  = cls == top_disease
        is_healthy = "healthy" in cls.lower()
        fill_class  = "prob-fill-h" if is_healthy else ("prob-fill-d" if is_active else "prob-fill-n")
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
        + "".join(rows)
        + '</div>'
    )


def diagnosis_card(result: dict) -> None:
    if not result.get("is_leaf", True):
        norm_entropy = result.get("entropy", 1.0)
        html(f"""
        <div class="result-card" style="border-color:rgba(245,158,11,0.35);
             background:linear-gradient(145deg,rgba(245,158,11,0.08) 0%,rgba(8,12,20,0.9) 100%);
             box-shadow:0 0 40px rgba(245,158,11,0.07);">
            <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:2px;color:#78350F;margin-bottom:0.5rem;">Input Detection</div>
            <div style="font-size:1.45rem;font-weight:800;color:#FBBF24;line-height:1.2;
                        margin-bottom:0.4rem;">
                ⚠ Unrecognised Input
            </div>
            <div style="font-size:0.82rem;color:#92400E;margin-bottom:1rem;">
                The model's confidence is spread too evenly across all 10 classes
                (entropy {norm_entropy:.0%} of max). This image is likely not a tomato leaf.
            </div>
            <div style="font-size:0.72rem;color:#78350F;background:rgba(245,158,11,0.07);
                        border:1px solid rgba(245,158,11,0.15);border-radius:8px;padding:0.6rem 0.9rem;">
                💡 For best results, photograph a single tomato leaf in good lighting.
            </div>
        </div>
        """)
        return

    disease       = result["disease"]
    display       = DISEASE_DISPLAY.get(disease, disease)
    conf          = result["confidence"]
    passes        = result["passes_threshold"]
    is_healthy    = "healthy" in disease.lower()
    severity_label, severity_class = DISEASE_SEVERITY.get(disease, ("Unknown", "moderate"))

    card_class     = "result-card-healthy" if is_healthy else "result-card-diseased"
    conf_class     = "conf-big-healthy" if is_healthy else ("conf-big-diseased" if passes else "conf-big-low")
    icon           = "✓" if is_healthy else "⚠"
    conf_bar_color = "#34D399" if is_healthy else ("#F87171" if passes else "#FBBF24")
    status_text    = "Plant is healthy — no treatment required." if is_healthy else "Disease detected — see recommendations below."

    html(f"""
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
        <div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid rgba(255,255,255,0.05);
                    display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:1.5px;color:#374151;">Prediction Entropy</span>
            <span style="font-size:0.72rem;color:{'#34D399' if result.get('entropy',0) < 0.35 else '#FBBF24' if result.get('entropy',0) < 0.6 else '#F87171'};font-weight:700;">
                {result.get('entropy', 0):.0%}
            </span>
            <span style="font-size:0.68rem;color:#374151;">
                {'· Low — model is certain' if result.get('entropy',0) < 0.35 else '· Moderate — some uncertainty' if result.get('entropy',0) < 0.6 else '· High — verify with a clearer image'}
            </span>
        </div>
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


# ---------------------------------------------------------------------------
# Grad-CAM panel
# ---------------------------------------------------------------------------

def render_gradcam(gradcam: dict) -> None:
    dname = DISEASE_DISPLAY.get(gradcam["class_name"], gradcam["class_name"])
    st.markdown("#### Grad-CAM — What the model is looking at")
    st.caption(
        f"Target class: **{dname}** · confidence {gradcam['confidence']:.1%} · "
        f"layer `{gradcam['layer_name']}`"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(gradcam["original_png"], caption="Original image", use_container_width=True)
    with col2:
        st.image(gradcam["overlay_png"],  caption="Grad-CAM overlay", use_container_width=True)
    st.caption(
        "🔴 Red/warm = high activation (model focuses here)  ·  "
        "🔵 Blue/cool = low activation. "
        "Activations should concentrate on leaf lesions — if they land on soil or background "
        "the model may be using spurious correlations."
    )


# ---------------------------------------------------------------------------
# Analytics charts
# ---------------------------------------------------------------------------

def plot_inference_charts(log: list) -> None:
    import pandas as pd
    import plotly.graph_objects as go

    if not log:
        return
    df = pd.DataFrame(log)
    df["Confidence_f"] = df["Confidence"].str.rstrip("%").astype(float) / 100
    df["Index"]        = range(1, len(df) + 1)
    df["IsDisease"]    = ~df["Disease"].str.lower().str.contains("healthy")

    disease_count = int(df["IsDisease"].sum())
    healthy_count = int(len(df) - disease_count)
    avg_conf      = df["Confidence_f"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scans",    len(df))
    m2.metric("Diseases Found", disease_count)
    m3.metric("Healthy",        healthy_count)
    m4.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        colors  = ["#F87171" if d else "#34D399" for d in df["IsDisease"]]
        fig_bar = go.Figure(go.Bar(
            x=df["Index"], y=df["Confidence_f"],
            marker_color=colors,
            hovertemplate="Scan %{x}<br>Confidence: %{y:.1%}<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text="Confidence per Scan", font=dict(size=13, color="#64748B"), x=0),
            height=300, template="plotly_dark", showlegend=False,
            margin=dict(l=20, r=10, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#94A3B8", size=12),
            yaxis=dict(tickformat=".0%", gridcolor="rgba(255,255,255,0.05)",
                       title=dict(text="Confidence", font=dict(size=12))),
            xaxis=dict(title=dict(text="Scan #", font=dict(size=12)),
                       gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_bar, width="stretch", key="bar_chart")

    with c2:
        fig_pie = go.Figure(go.Pie(
            labels=["Disease", "Healthy"],
            values=[disease_count, healthy_count],
            marker=dict(colors=["#F87171", "#34D399"],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            hole=0.6, textinfo="percent+label",
            textfont=dict(size=13, color="#E2E8F0"),
            hovertemplate="%{label}: %{value} scans<extra></extra>",
            insidetextorientation="radial",
        ))
        fig_pie.update_layout(
            title=dict(text="Disease vs Healthy", font=dict(size=13, color="#64748B"), x=0),
            height=300, template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5, font=dict(size=12)),
        )
        st.plotly_chart(fig_pie, width="stretch", key="pie_chart")


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_cached_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def render(confidence: float, selected_classes: list) -> None:
    """Render the full Inference tab."""
    input_mode = st.radio(
        "Input mode", ["📁 Upload Image", "📷 Live Camera"],
        horizontal=True, label_visibility="collapsed",
        key="input_mode",
    )
    if st.session_state.get("_last_input_mode") != input_mode:
        st.session_state.pop("last_static_result", None)
        st.session_state["_last_input_mode"] = input_mode

    col_upload, col_result = st.columns([1, 1], gap="large")
    source = None

    with col_upload:
        if input_mode == "📷 Live Camera":
            st.markdown('<div class="section-label">Live Camera</div>', unsafe_allow_html=True)
            raw = webcam_selector(key="wc")
            if raw:
                import base64 as _b64
                img_bytes = _b64.b64decode(raw.split(",", 1)[1])
                size_kb   = len(img_bytes) / 1024
                st.caption(f"📷 Captured &nbsp;·&nbsp; {size_kb:.0f} KB")
                source = io.BytesIO(img_bytes)
            else:
                st.session_state.pop("last_static_result", None)
        else:
            st.markdown('<div class="section-label">Upload Leaf Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png"],
                                        label_visibility="collapsed")
            if uploaded:
                img_bytes = uploaded.read()
                uploaded.seek(0)
                size_kb = len(img_bytes) / 1024
                st.image(uploaded, width="stretch")
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
                source = uploaded
            else:
                st.session_state.pop("last_static_result", None)
                html("""
                <div class="empty-state">
                    <span class="empty-state-icon">🌿</span>
                    <div class="empty-state-title">No image selected</div>
                    <div class="empty-state-sub">JPG or PNG · up to 200 MB</div>
                </div>""")

    with col_result:
        st.markdown('<div class="section-label">Model & Diagnosis</div>', unsafe_allow_html=True)
        model_path = model_picker("Model", key="img_model")
        c_status, c_info = st.columns([1, 2])
        c_status.markdown(model_status_html(model_path), unsafe_allow_html=True)
        info_html = model_info_html(model_path)
        if info_html:
            c_info.markdown(info_html, unsafe_allow_html=True)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        if input_mode == "📷 Live Camera" and source is not None:
            img_hash = hashlib.md5(source.getvalue()).hexdigest()
            source.seek(0)
            if st.session_state.get("_cam_auto_hash") != img_hash:
                st.session_state["_cam_auto_hash"] = img_hash
                if not model_path or not Path(model_path).exists():
                    st.error("No model found. Train a model first or enter a valid path.")
                else:
                    with st.spinner("Analysing leaf…"):
                        result = run_inference(source, model_path, confidence)
                        if result:
                            st.session_state["last_static_result"] = result
                            st.rerun()

        btn_label = "📷 Re-diagnose" if input_mode == "📷 Live Camera" else "Run Diagnosis"
        run_btn   = st.button(btn_label, type="primary", width="stretch", disabled=source is None)

        if run_btn and source is not None:
            if not model_path or not Path(model_path).exists():
                st.error("No model found. Train a model first or enter a valid path.")
            else:
                with st.spinner("Analysing leaf…"):
                    if hasattr(source, "seek"):
                        source.seek(0)
                    result = run_inference(source, model_path, confidence)
                    if result:
                        st.session_state["last_static_result"] = result
                        st.rerun()

        last_result = st.session_state.get("last_static_result")
        if last_result and source is not None:
            diagnosis_card(last_result)

            seg_png = last_result.get("segmented_png")
            if seg_png:
                with st.expander("GrabCut Segmentation", expanded=True):
                    st.image(seg_png, caption="Leaf after background removal", use_container_width=True)
                    st.caption("Grey pixels were classified as background by GrabCut and excluded from inference.")

            _mpath = st.session_state.get("last_model_path", "")
            if last_result.get("is_leaf", True) and _mpath.endswith(".keras"):
                if st.button("🔍 Explain with Grad-CAM", key="btn_gradcam", type="secondary"):
                    _ibytes = st.session_state.get("last_img_bytes")
                    if _ibytes:
                        with st.spinner("Computing Grad-CAM gradients…"):
                            try:
                                st.session_state["gradcam_result"] = compute_gradcam(
                                    _mpath, _ibytes,
                                    class_idx=last_result.get("class_idx"),
                                )
                            except Exception as _e:
                                st.error(f"Grad-CAM failed: {_e}")
            if st.session_state.get("gradcam_result"):
                render_gradcam(st.session_state["gradcam_result"])

        elif source is not None:
            html("""
            <div class="empty-state" style="margin-top:1rem;">
                <span class="empty-state-icon">🔬</span>
                <div class="empty-state-title">Ready to analyse</div>
                <div class="empty-state-sub">Click the button above to classify this leaf</div>
            </div>""")

    # Session analytics
    log = st.session_state.get("diagnosis_log", [])
    if log:
        import pandas as pd
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.divider()
        h1, h2 = st.columns([3, 1])
        h1.subheader("Session Analytics")
        with h2:
            log_df    = pd.DataFrame(log[::-1])
            csv_bytes = log_df.to_csv(index=False).encode()
            st.download_button(
                "Export CSV", csv_bytes,
                file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", type="secondary", width="stretch",
            )
        plot_inference_charts(log)
        with st.expander(f"Session Log — {len(log)} scans"):
            st.dataframe(log_df, width="stretch", hide_index=True)
            if st.button("Clear Session", type="secondary"):
                st.session_state.diagnosis_log = []
                st.session_state.pop("last_static_result", None)
                st.session_state.scans    = 0
                st.session_state.diseases = 0
                st.rerun()
