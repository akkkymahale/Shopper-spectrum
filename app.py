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

    /* Target main body paragraphs cleanly without breaking structural components */
    .stApp [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
        color: #F1EFF7 !important;
    }
    
    /* Left Sidebar Panel container styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0813 !important;
        border-right: 1px solid #1C172E !important;
    }
    
    /* FIXED KEYBOARD_DOUBLE BUG: Explicitly targets ONLY the text inside options */
    section[data-testid="stSidebar"] .stRadio label div[data-testid="stMarkdownContainer"] p {
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
        gap: 14px;
        margin-top: 2.5rem;
        padding: 1rem;
        background: radial-gradient(circle at top left, #17122B, #110E1C);
        border: 1px solid #251F3D;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .dev-avatar-wrapper {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .dev-details {
        display: flex;
        flex-direction: column;
    }
    .dev-status-indicator {
        position: absolute;
        bottom: 1px;
        right: 1px;
        width: 11px;
        height: 11px;
        background-color: #00E676;
        border: 2px solid #110E1C;
        border-radius: 50%;
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
    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "rb") as f: sim_df = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rfm_table.pkl"), "rb") as f: rfm_table = pickle.load(f)
    return scaler, kmeans, cluster_map, sim_df, rfm_table

scaler, kmeans, cluster_map, sim_df, rfm_table = load_workspace_data()

# --------------------------------------------------------------------------
# 3. SIDEBAR BRANDING (SHOPPER SPECTRUM EXECUTIVE FORMAT)
# --------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.8rem 0 1.2rem 0;">
        <h1 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: 1.8px; color: #FFFFFF; font-family: 'Montserrat', sans-serif !important; text-transform: uppercase;">
            SHOPPER SPECTRUM
        </h1>
        <div style="font-size: 0.78rem; color: #8D87A4; font-weight: 500; margin-top: 6px; letter-spacing: 0.3px; font-family: 'Inter', sans-serif;">
            Data-driven insights for modern retail
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Primary Module Navigation Panel
page = st.sidebar.radio(
    "E-Commerce Navigation Modules",
    [
        "Dashboard Overview", 
        "Product Recommendations", 
        "Customer Segmentation",
        "CSV Batch Engine"
    ],
    index=0,
    key="navigation_menu_selection"
)

# Live Stats Metrics Component Box
st.sidebar.markdown(
    """
    <div class="live-stats-box">
        <div style="font-size:0.72rem; font-weight:800; color:#8D87A4; margin-bottom:1rem; letter-spacing:1px; font-family: 'Inter', sans-serif;">LIVE PLATFORM METRICS</div>
        <div class="stat-row"><div class="stat-label">Customers</div><div class="stat-val">4,338</div></div>
        <div class="stat-row"><div class="stat-label">Products</div><div class="stat-val">3,877</div></div>
        <div class="stat-row"><div class="stat-label">Revenue Target</div><div class="stat-val">8.91M USD</div></div>
        <div class="stat-row"><div class="stat-label">Countries Verified</div><div class="stat-val">37</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Direct bypass hosting link using your provided image ID 
DIRECT_IMAGE_URL = "https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o"

# High-Gloss Custom Profile Badge with perfect circular crop layout masking
st.sidebar.markdown(
    f"""
    <div class="dev-profile-container">
        <div class="dev-avatar-wrapper">
            <img src="{DIRECT_IMAGE_URL}" width="46" height="46" style="border-radius: 50%; object-fit: cover; border: 2px solid #7451F7; background: #1F1936;" alt="Developer Profile">
            <div class="dev-status-indicator"></div>
        </div>
        <div class="dev-details">
            <span style="font-size:0.68rem; color:#8D87A4; text-transform:uppercase; font-weight: 800; letter-spacing: 0.7px;">Lead Architecture</span>
            <span style="font-weight:700; color:#FFFFFF; font-size:0.95rem; margin-top: 2px; font-family: 'Inter', sans-serif; letter-spacing: 0.2px;">Akshay Mahale</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 4. MODULE CONTROLLERS / RUNTIME LOGIC
# --------------------------------------------------------------------------

# --- MODULE 1: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.6rem; font-weight: 800; color: #FFFFFF; font-family: 'Montserrat', sans-serif !important; letter-spacing: -0.5px;">Retail Intelligence Hub</h1>
            <p style="color: #8D87A4; font-size: 1.02rem; max-width: 700px; margin-top: 0.5rem; line-height: 1.5; font-family: 'Inter', sans-serif;">Translating continuous consumer interaction loops into structured, behavioral analytical matrix environments dynamically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "4,338")
    c2.metric("Unique Products", "3,877")
    c3.metric("Total Revenue", "8.91M")
    c4.metric("Countries", "37")

    st.markdown("<h3 style='margin-top:2rem; font-weight:700;'>Segment Distribution Tracking</h3>", unsafe_allow_html=True)
    st.bar_chart(rfm_table["Segment"].value_counts())

    st.markdown("---")
    st.markdown("<h3 style='font-weight:700;'>Geographic Insights: Global Order Distribution</h3>", unsafe_allow_html=True)
    st.write("Slicing order capacity aggregates and interaction frequencies across distribution country origins.")

    if "Country" in rfm_table.columns:
        country_counts = rfm_table["Country"].value_counts().head(10)
    else:
        data_mock = {
            "United Kingdom": 3950, "Germany": 94, "France": 87, 
            "EIRE": 74, "Spain": 31, "Netherlands": 23,
