"""Project configuration, constants, and design system."""
import streamlit as st

PAGE_CONFIG = {
    "page_title": "Tomato AI Diagnostics",
    "page_icon": "🍅",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
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

# Human-readable display names mapped from folder names
DISEASE_DISPLAY = {
    "Tomato___Bacterial_spot":                       "Bacterial Spot",
    "Tomato___Early_blight":                         "Early Blight",
    "Tomato___healthy":                              "Healthy",
    "Tomato___Late_blight":                          "Late Blight",
    "Tomato___Leaf_Mold":                            "Leaf Mold",
    "Tomato___Septoria_leaf_spot":                   "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot":                          "Target Spot",
    "Tomato___Tomato_mosaic_virus":                  "Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":        "Yellow Leaf Curl",
}

CSS = """
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: #F0F2F6;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none !important; }
.main .block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #16213E !important;
    border-right: none;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] caption {
    color: #CBD5E0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 1.4rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
    color: #64748B !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1E3A5F !important;
}
section[data-testid="stSidebar"] .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #E63946, #F4A261) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #1E3A5F !important;
    border: 1px solid #2D4E7A !important;
    border-radius: 8px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 24px;
    color: #6C757D !important;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E63946, #C1121F) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(230,57,70,0.3);
}

/* ── Primary Button ───────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E63946, #C1121F) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1.8rem !important;
    box-shadow: 0 4px 14px rgba(230,57,70,0.35) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(230,57,70,0.45) !important;
}
.stButton > button[kind="secondary"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1.5px solid #E2E8F0 !important;
    color: #475569 !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #EEF0F2;
}
[data-testid="stMetricValue"] {
    color: #16213E !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: #6C757D !important; font-size: 0.82rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Progress Bar ─────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #E63946, #F4A261) !important;
    border-radius: 6px !important;
}

/* ── Text Inputs ──────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 0.75rem !important;
    background: white !important;
}
.stTextInput > div > div > input:focus {
    border-color: #E63946 !important;
    box-shadow: 0 0 0 3px rgba(230,57,70,0.1) !important;
}

/* ── Select / Multiselect ─────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: white !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: #E63946 !important;
}

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: white;
    border-radius: 10px !important;
    border: 1px solid #EEF0F2 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── File Uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E0 !important;
    border-radius: 12px !important;
    background: white !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #E63946 !important;
}

/* ── Alerts ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Dataframe ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #EEF0F2 !important;
}

/* ── Toggle ───────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] span[aria-checked="true"],
[role="switch"][aria-checked="true"] {
    background-color: #E63946 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F0F2F6; }
::-webkit-scrollbar-thumb { background: #CBD5E0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── Custom Component Classes ─────────────────────────────────────────── */
.section-card {
    background: white;
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #EEF0F2;
    margin-bottom: 1.25rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}
.diagnosis-card-healthy {
    background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 1.5px solid #6EE7B7;
    border-radius: 14px;
    padding: 1.5rem;
}
.diagnosis-card-diseased {
    background: linear-gradient(135deg, #FFF7ED 0%, #FEE2E2 100%);
    border: 1.5px solid #FCA5A5;
    border-radius: 14px;
    padding: 1.5rem;
}
.diagnosis-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}
.diagnosis-sub {
    font-size: 0.82rem;
    color: #6B7280;
    margin: 0;
}
.model-status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #D1FAE5;
    color: #065F46;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.model-status-missing {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FEE2E2;
    color: #991B1B;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.stage-header {
    background: linear-gradient(135deg, #F8FAFC, #EEF2FF);
    border: 1px solid #E0E7FF;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.stage-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #3730A3;
    margin: 0;
}
.stage-desc {
    font-size: 0.75rem;
    color: #6B7280;
    margin: 2px 0 0 0;
}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
