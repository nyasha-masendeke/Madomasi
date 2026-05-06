"""Project configuration, constants, and modular CSS."""
import streamlit as st

PAGE_CONFIG = {
    "page_title": "Tomato AI | Deepstack UI",
    "page_icon": "🍅",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Must match folder names in data/raw/raw/tomato/ sorted alphabetically —
# Keras image_dataset_from_directory assigns class indices in this order.
DISEASE_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
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
