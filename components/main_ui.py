"""Main UI orchestrator — renders hero banner and delegates each tab to its module."""
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

import config as _config_module
from config import DISEASE_CLASSES, DISEASE_DISPLAY
from components.system_dashboard import render_dashboard
from components.tabs import inference_tab, training_tab
from components.ui_helpers import html as _html

PROJECT_ROOT = Path(_config_module.__file__).parent


def _hero() -> None:
    scans       = st.session_state.get("scans", 0)
    diseases    = st.session_state.get("diseases", 0)
    healthy     = scans - diseases
    healthy_rate = f"{healthy / scans * 100:.0f}%" if scans else "—"

    import re
    def _html(raw: str) -> None:
        st.markdown(re.sub(r'\n\s*\n', '\n', raw).strip(), unsafe_allow_html=True)

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


def _clear_stale_dir_keys() -> None:
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


def run_app() -> None:
    os.chdir(PROJECT_ROOT)
    _clear_stale_dir_keys()
    _hero()

    tab_inference, tab_training, tab_system = st.tabs([
        "  Inference  ",
        "  Training  ",
        "  System  ",
    ])

    with tab_inference:
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
        inference_tab.render(confidence, selected_classes)

    with tab_training:
        training_tab.render()

    with tab_system:
        render_dashboard()
