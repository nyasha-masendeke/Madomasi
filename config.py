"""Project configuration, constants, and design system."""
import streamlit as st

PAGE_CONFIG = {
    "page_title": "Madomasi — Tomato AI",
    "page_icon": "🍅",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

DISEASE_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

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

# (label, badge-class)
DISEASE_SEVERITY = {
    "Tomato___Bacterial_spot":                       ("High",     "critical"),
    "Tomato___Early_blight":                         ("Moderate", "moderate"),
    "Tomato___healthy":                              ("Healthy",  "healthy"),
    "Tomato___Late_blight":                          ("Critical", "critical"),
    "Tomato___Leaf_Mold":                            ("Moderate", "moderate"),
    "Tomato___Septoria_leaf_spot":                   ("Moderate", "moderate"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("High",     "critical"),
    "Tomato___Target_Spot":                          ("Moderate", "moderate"),
    "Tomato___Tomato_mosaic_virus":                  ("High",     "critical"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":        ("Critical", "critical"),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Keyframes ─────────────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(14px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
  from { opacity:0; } to { opacity:1; }
}
@keyframes pulse {
  0%,100% { opacity:1; } 50% { opacity:0.3; }
}
@keyframes shimmer {
  0%   { background-position: -400% center; }
  100% { background-position:  400% center; }
}
@keyframes fillBar {
  from { width:0 !important; }
}
@keyframes glow {
  0%,100% { box-shadow: 0 0 20px rgba(230,57,70,0.3); }
  50%      { box-shadow: 0 0 40px rgba(230,57,70,0.6); }
}

/* ── Hide chrome ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"],
[data-testid="collapseSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"] { display:none !important; }

/* ── Base ───────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    color:#E2E8F0;
}
.stApp { background:#080C14; }
[data-testid="stHeader"] { background:transparent; }
.main .block-container {
    padding: 0 2.5rem 4rem;
    max-width: 1440px;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:5px; gap:4px;
    margin-bottom:1.5rem; backdrop-filter:blur(10px);
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px; font-weight:600; font-size:0.875rem;
    padding:9px 28px; color:#4B5563 !important;
    background:transparent; border:none; transition:all 0.2s;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#E63946 0%,#C1121F 100%) !important;
    color:#fff !important;
    box-shadow:0 0 24px rgba(230,57,70,0.45),0 2px 8px rgba(230,57,70,0.3);
}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#E63946 0%,#C1121F 100%) !important;
    border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:0.92rem !important;
    padding:0.65rem 1.8rem !important; color:#fff !important;
    box-shadow:0 0 20px rgba(230,57,70,0.35),0 4px 12px rgba(230,57,70,0.25) !important;
    transition:all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 0 36px rgba(230,57,70,0.55),0 6px 20px rgba(230,57,70,0.4) !important;
}
.stButton > button[kind="primary"]:active { transform:translateY(0) !important; }
.stButton > button[kind="secondary"] {
    border-radius:10px !important; font-weight:600 !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    color:#94A3B8 !important; background:rgba(255,255,255,0.04) !important;
    transition:all 0.2s !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color:rgba(230,57,70,0.35) !important; color:#E63946 !important;
}

/* ── Metric cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background:rgba(255,255,255,0.035); border-radius:14px;
    padding:1.1rem 1.3rem; border:1px solid rgba(255,255,255,0.07);
    backdrop-filter:blur(12px); transition:border-color 0.2s,transform 0.2s;
}
[data-testid="stMetric"]:hover { border-color:rgba(230,57,70,0.25); transform:translateY(-1px); }
[data-testid="stMetricValue"]  { color:#F1F5F9 !important; font-weight:800 !important; font-size:1.7rem !important; }
[data-testid="stMetricLabel"]  { color:#64748B !important; font-size:0.78rem !important; font-weight:600 !important; text-transform:uppercase; letter-spacing:0.8px; }
[data-testid="stMetricDelta"]  { font-size:0.78rem !important; }

/* ── Progress bar ───────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#C1121F,#E63946,#FF6B6B,#E63946,#C1121F) !important;
    background-size:300% auto !important;
    animation:shimmer 2.5s linear infinite !important;
    border-radius:6px !important;
}
.stProgress > div > div > div { background:rgba(255,255,255,0.05) !important; border-radius:6px !important; }

/* ── Spinner ────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p,[data-testid="stSpinner"] span,
[data-testid="stSpinner"] div { color:#94A3B8 !important; }
[data-testid="stSpinner"] svg { stroke:#E63946 !important; }

/* ── Text inputs ────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius:8px !important; border:1px solid rgba(255,255,255,0.09) !important;
    font-size:0.875rem !important; padding:0.5rem 0.75rem !important;
    background:rgba(255,255,255,0.04) !important; color:#E2E8F0 !important; transition:all 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color:#E63946 !important; box-shadow:0 0 0 3px rgba(230,57,70,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color:#374151 !important; }

/* ── Number input ───────────────────────────────────────────────────── */
.stNumberInput > div > div > input {
    border-radius:8px !important; border:1px solid rgba(255,255,255,0.09) !important;
    background:rgba(255,255,255,0.04) !important; color:#E2E8F0 !important;
}

/* ── Select / Multiselect ───────────────────────────────────────────── */
.stSelectbox > div > div,.stMultiSelect > div > div {
    border-radius:8px !important; border:1px solid rgba(255,255,255,0.09) !important;
    background:rgba(255,255,255,0.04) !important; color:#E2E8F0 !important;
}
[data-baseweb="select"] span,[data-baseweb="input"] input { color:#E2E8F0 !important; }
[data-baseweb="popover"] {
    background:#0F1623 !important; border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:12px !important; box-shadow:0 8px 32px rgba(0,0,0,0.5) !important;
}
[data-baseweb="menu"] { background:#0F1623 !important; border-radius:12px !important; }
[data-baseweb="option"] { color:#CBD5E0 !important; }
[data-baseweb="option"]:hover { background:rgba(230,57,70,0.12) !important; color:#fff !important; }

/* ── Sliders ────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div { background:#E63946 !important; }
.stSlider > div > div > div { background:rgba(255,255,255,0.08) !important; }

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:rgba(255,255,255,0.025) !important;
    border-radius:12px !important; border:1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stExpander"] summary { color:#CBD5E0 !important; font-weight:600; font-size:0.88rem; }
[data-testid="stExpander"] summary svg { fill:#4B5563 !important; }
[data-testid="stExpander"] details[open] > summary { color:#E63946 !important; }
[data-testid="stExpander"] details[open] > summary svg { fill:#E63946 !important; }

/* ── File uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border:2px dashed rgba(255,255,255,0.09) !important;
    border-radius:16px !important; background:rgba(255,255,255,0.02) !important; transition:all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color:rgba(230,57,70,0.4) !important; background:rgba(230,57,70,0.03) !important;
}

/* ── Radio ──────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label,[data-testid="stRadio"] label p,
[data-testid="stRadio"] div[role="radiogroup"] label span p {
    color:#CBD5E0 !important; font-weight:500 !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius:10px !important; }
[data-testid="stAlert"][kind="info"] { background:rgba(99,102,241,0.08) !important; border-color:rgba(99,102,241,0.25) !important; }

/* ── Dataframe ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius:12px !important; overflow:hidden; border:1px solid rgba(255,255,255,0.07) !important;
}

/* ── Widget labels ──────────────────────────────────────────────────── */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label {
    color:#64748B !important; font-weight:600 !important; font-size:0.78rem !important;
}

/* ── Toggle ─────────────────────────────────────────────────────────── */
[role="switch"][aria-checked="true"] { background-color:#E63946 !important; }

/* ── Divider ────────────────────────────────────────────────────────── */
hr { border-color:rgba(255,255,255,0.06) !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#E63946; }

/* ── Container borders ──────────────────────────────────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] > div {
    background:rgba(255,255,255,0.025) !important;
    border:1px solid rgba(255,255,255,0.07) !important; border-radius:14px !important;
}

/* ── Typography ─────────────────────────────────────────────────────── */
h1,h2,h3 { color:#F1F5F9 !important; font-weight:800 !important; letter-spacing:-0.3px; }
h2 { font-size:1.35rem !important; }
h3 { font-size:1.05rem !important; }
p,label,caption,small { color:#94A3B8; }
code { background:rgba(230,57,70,0.1) !important; color:#FCA5A5 !important; border-radius:4px !important; padding:2px 6px !important; }

/* ══════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════════════════════════════ */

/* ── Section label ──────────────────────────────────────────────────── */
.section-label {
    font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.8px; color:#374151; margin-bottom:0.5rem;
}

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    background:rgba(255,255,255,0.018); border:2px dashed rgba(255,255,255,0.07);
    border-radius:18px; padding:3.5rem 2rem; text-align:center;
    animation:fadeIn 0.4s ease;
}
.empty-state-icon { font-size:3rem; display:block; margin-bottom:0.75rem; opacity:0.45; }
.empty-state-title { font-size:1rem; font-weight:700; color:#374151; margin-bottom:0.25rem; }
.empty-state-sub   { font-size:0.8rem; color:#1F2937; }

/* ── Result card ────────────────────────────────────────────────────── */
.result-card {
    border-radius:18px; padding:1.75rem; animation:fadeInUp 0.35s ease;
    border:1px solid rgba(255,255,255,0.07); background:rgba(255,255,255,0.03);
}
.result-card-diseased {
    border-color:rgba(230,57,70,0.3);
    background:linear-gradient(145deg,rgba(230,57,70,0.09) 0%,rgba(8,12,20,0.9) 100%);
    box-shadow:0 0 50px rgba(230,57,70,0.08),inset 0 1px 0 rgba(230,57,70,0.15);
}
.result-card-healthy {
    border-color:rgba(16,185,129,0.3);
    background:linear-gradient(145deg,rgba(16,185,129,0.09) 0%,rgba(8,12,20,0.9) 100%);
    box-shadow:0 0 50px rgba(16,185,129,0.08),inset 0 1px 0 rgba(16,185,129,0.15);
}

/* ── Diagnosis cards (legacy compat) ────────────────────────────────── */
.diagnosis-card-healthy {
    background:linear-gradient(135deg,rgba(16,185,129,0.1) 0%,rgba(5,150,105,0.06) 100%);
    border:1px solid rgba(16,185,129,0.25); border-radius:14px; padding:1.5rem;
    box-shadow:0 0 30px rgba(16,185,129,0.07);
}
.diagnosis-card-diseased {
    background:linear-gradient(135deg,rgba(230,57,70,0.1) 0%,rgba(193,18,31,0.06) 100%);
    border:1px solid rgba(230,57,70,0.25); border-radius:14px; padding:1.5rem;
    box-shadow:0 0 30px rgba(230,57,70,0.07);
}
.diagnosis-title { font-size:1.35rem; font-weight:800; margin:0 0 0.2rem; }
.diagnosis-sub   { font-size:0.82rem; color:#4B5563; margin:0; }

/* ── Severity badges ────────────────────────────────────────────────── */
.badge-critical { display:inline-block; background:rgba(239,68,68,0.15); color:#F87171; border:1px solid rgba(239,68,68,0.3); padding:3px 11px; border-radius:20px; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.9px; }
.badge-moderate { display:inline-block; background:rgba(245,158,11,0.15); color:#FBBF24; border:1px solid rgba(245,158,11,0.3); padding:3px 11px; border-radius:20px; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.9px; }
.badge-healthy  { display:inline-block; background:rgba(16,185,129,0.15); color:#34D399;  border:1px solid rgba(16,185,129,0.3); padding:3px 11px; border-radius:20px; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.9px; }

/* ── Big confidence number ──────────────────────────────────────────── */
.conf-big          { font-size:3.8rem; font-weight:900; letter-spacing:-4px; line-height:1; }
.conf-big-healthy  { color:#34D399; }
.conf-big-diseased { color:#F87171; }
.conf-big-low      { color:#FBBF24; }

/* ── Probability bars ───────────────────────────────────────────────── */
.prob-section { margin-top:1.25rem; }
.prob-section-title { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; color:#374151; margin-bottom:0.75rem; }
.prob-row   { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.55rem; }
.prob-label { font-size:0.75rem; color:#64748B; width:120px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prob-label-active { color:#F1F5F9 !important; font-weight:700; }
.prob-track { flex:1; height:5px; background:rgba(255,255,255,0.05); border-radius:5px; overflow:hidden; }
.prob-fill-h { height:100%; border-radius:5px; background:linear-gradient(90deg,#059669,#34D399); animation:fillBar 0.7s cubic-bezier(0.4,0,0.2,1); }
.prob-fill-d { height:100%; border-radius:5px; background:linear-gradient(90deg,#C1121F,#F87171); animation:fillBar 0.7s cubic-bezier(0.4,0,0.2,1); }
.prob-fill-n { height:100%; border-radius:5px; background:linear-gradient(90deg,#3730A3,#818CF8); animation:fillBar 0.7s cubic-bezier(0.4,0,0.2,1); }
.prob-pct   { font-size:0.72rem; font-weight:700; color:#6B7280; width:38px; text-align:right; }

/* ── Model status ───────────────────────────────────────────────────── */
.model-status-ok {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(16,185,129,0.12); color:#34D399;
    padding:4px 12px; border-radius:20px; font-size:0.72rem; font-weight:700;
    border:1px solid rgba(16,185,129,0.2);
}
.model-status-missing {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(230,57,70,0.12); color:#F87171;
    padding:4px 12px; border-radius:20px; font-size:0.72rem; font-weight:700;
    border:1px solid rgba(230,57,70,0.2);
}

/* ── Section card ───────────────────────────────────────────────────── */
.section-card {
    background:rgba(255,255,255,0.025); border-radius:16px;
    padding:1.5rem; border:1px solid rgba(255,255,255,0.06); margin-bottom:1rem;
}

/* ── Stage header ───────────────────────────────────────────────────── */
.stage-header {
    background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.14);
    border-radius:10px; padding:0.45rem 0.75rem; margin-bottom:0.5rem;
}
.stage-title { font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.9px; color:#818CF8; margin:0; }

/* ── Live dot ───────────────────────────────────────────────────────── */
.live-dot {
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:#E63946; animation:pulse 1.5s ease-in-out infinite;
    margin-right:5px; vertical-align:middle;
}

/* ── Treatment card ─────────────────────────────────────────────────── */
.treatment-card {
    background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.14);
    border-radius:12px; padding:1rem 1.25rem; margin-top:0.75rem;
}
.treatment-label { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#78350F; margin-bottom:0.4rem; }

/* ── Config bar ─────────────────────────────────────────────────────── */
.config-bar {
    background:rgba(255,255,255,0.025); border-radius:14px;
    padding:1rem 1.25rem; border:1px solid rgba(255,255,255,0.06);
    margin-bottom:1.25rem; display:flex; gap:1rem; align-items:flex-end; flex-wrap:wrap;
}

/* ── Image preview card ─────────────────────────────────────────────── */
.img-preview-card {
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; overflow:hidden; margin-top:0.75rem;
}
.img-meta { padding:0.6rem 0.85rem; font-size:0.72rem; color:#4B5563; display:flex; gap:1rem; }

/* ── Session stats bar ──────────────────────────────────────────────── */
.stats-chip {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:4px 12px; font-size:0.75rem; color:#94A3B8;
}
.stats-chip strong { color:#E2E8F0; font-weight:700; }
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
