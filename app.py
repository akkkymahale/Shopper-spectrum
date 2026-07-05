import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import pipeline

# --------------------------------------------------------------------------
# 1. PAGE SETUP & DEFINITIVE SHOPPER SPECTRUM CANVASES (ULTRA-DARK THEME)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum | Retail Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep obsidian dark dashboard canvas style mapping with upgraded professional fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;700;800&family=Montserrat:wght=700;800&display=swap');

    /* Global Application Canvas Base */
    .stApp {
        background-color: #06040A !important;
        color: #F1EFF7 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Force consistent color and professional typography across headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', 'Inter', sans-serif !important;
        color: #FFFFFF !important;
    }

    /* Target main body paragraphs and markdown cleanly without breaking structural buttons */
    .stApp [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
        color: #F1EFF7 !important;
    }
    
    /* Left Sidebar Panel container styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0813 !important;
        border-right: 1px solid #1C172E !important;
    }
    
    /* Fixed Sidebar styling: selectively overrides text targets without breaking native 'keyboard_double' layout icons */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] .stRadio label {
        color: #B3AECE !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Shopper Spectrum High-Gloss Core Feature Cards */
    div[data-testid="metric-container"] {
        background: #110E1C !important;
        border: 1px solid #1F1936 !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        letter-spacing: -1px;
        font-family: 'Montserrat', sans-serif !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8D87A4 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700 !important;
    }

    /* Interactive Functional Element Modules */
    .feature-card {
        background: #110E1C;
        border: 1px solid #1F1936;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Live Stats Sidebar Widget block */
    .live-stats-box {
        margin-top: 1.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #1C172E;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #6C6684;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: monospace;
    }

    /* Developer layout profile widgets */
    .dev-profile-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 1.5rem;
        padding: 0.8rem;
        background: #110E1C;
        border: 1px solid #1F1936;
        border-radius: 12px;
    }
    .dev-details {
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 2. MODELS & ARTIFACTS ORCHESTRATION PIPELINE
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
    st.title("System Initialization")
    st.write("Constructing analytical cluster matrices and compiling local workspace assets...")
    if st.button("Initialize Platform Engines", type="primary", key="init_platform_engine_btn"):
        with st.spinner("Downloading source registries and training model layers natively..."):
            pipeline.main()
            st.rerun()
    st.stop()

@st.cache_resource(show_spinner=False)
def load_workspace_data():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f: scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "rb") as f: kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "cluster_labels.pkl"), "rb") as f: cluster_map = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "similarity_matrix
