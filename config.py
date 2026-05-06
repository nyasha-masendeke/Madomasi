"""Project configuration, constants, and modular CSS."""
import streamlit as st

PAGE_CONFIG = {
    "page_title": "Tomato AI | Deepstack UI",
    "page_icon": "🍅",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

DISEASE_CLASSES = [
    "Early Blight", "Late Blight", "Bacterial Spot", "Target Spot",
    "Tomato Yellow Leaf Curl", "Healthy"
]

CSS_MODULES = {
    "base": """
        .stApp { background-color: #46499e; }
        [data-testid="stHeader"] { background: rgba(1,04,90,0); }
    """,
    "sidebar": """
        section[data-testid="stSidebar"] {
            background-color: #415461;
            border-right: 1px solid #e0e0e0;
        }
        .stSlider > div > div > div > div { background-color: #ff4b4b; }
    """,
    "results": """
        .result-block {
            background-color: #1e1e1e;
            color: #4ec9b0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Courier New', monospace;
            border-left: 5px solid #ff4b4b;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            white-space: pre-wrap;
        }
        .filtered-count {
            font-weight: 700;
            color: #212529;
            font-size: 1.1rem;
            margin-top: 10px;
        }
    """
}



def inject_css():
    full_css = "\n".join(CSS_MODULES.values())
    st.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)
