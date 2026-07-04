"""
Shopper Spectrum: Customer Segmentation & Product Recommendations
Streamlit web app - Production Build for Akshay Mahale.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Direct native import for execution environment safety
import pipeline

# --------------------------------------------------------------------------
# Page config & Complete Visual Match Stylesheet (Electric Sky-Blue Theme)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum | Akshay Mahale",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep Custom Theme Overrides to Match the Target Interface Exactly
st.markdown(
    """
    <style>
    /* Absolute Dark Mode Canvas */
    .stApp {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
    }
    
    /* Global Typography Reset to White/Silver */
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {
        color: #f0f6fc !important;
    }
    
    /* Premium Translucent Sidebar Design */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* Target Dashboard KPI Cards Match */
    div[data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.7) !important;
        border: 1px solid #30363d !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-family: monospace;
        font-weight: bold !important;
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        text-transform: uppercase;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px;
    }

    /* Product Recommendation & Output Cards */
    .product-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        background: #161b22;
        border: 1px solid #30363d;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .product-card strong {
        color: #58a6ff !important;
        font-size: 1.1rem;
    }
    
    /* Segment Result Overlay Card */
    .segment-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    /* Badges & Accents */
    .rank-badge {
        background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
        color: #ffffff !important;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.85rem;
        margin-right: 0.6rem;
        font-weight: bold;
    }

    /* Custom Developer Profile Badge Widget (Bottom Left Sidebar) */
    .dev-profile {
        background: #21262d;
        border: 1px solid #30363d;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    .dev-name {
        color: #58a6ff !important;
        font-weight: bold;
        font-size: 1rem;
    }
    
    /* Clean Input Overrides to stay dark-friendly */
    input, div[data-baseweb="input"], div[data-baseweb="number-input"] {
        background-color: #0d1117 !important;
        color: #f0f6fc !important;
        border-color: #30363d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SEGMENT_COLORS = {
    "High-Value": "#2ecc71",
    "Regular": "#3498db",
    "Occasional": "#f39c12",
    "At-Risk": "#e74c3c",
}

SEGMENT_DESCRIPTIONS = {
    "High-Value": "Recent, frequent, big-spending customers — your best customers. Prioritize loyalty perks and early access.",
    "Regular": "Steady purchasers with moderate frequency and spend. Good candidates for cross-sell campaigns.",
    "Occasional": "Infrequent, lower-spend shoppers. Consider targeted promotions to increase engagement.",
    "At-Risk": "Haven't purchased in a long time. Prioritize for win-back/retention campaigns.",
}

# --------------------------------------------------------------------------
# PIPELINE AUTOMATION SYSTEM
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

artifacts_exist = os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))

# Navigation Header & Branding
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.caption("Customer Analytics & Recommendations Dashboard")

if not artifacts_exist:
    st.title("📦 First-Time Setup Required")
    st.write("The machine learning models need to be built and trained before the dashboard can load.")
    
    if st.button("🚀 Start Data Download & Model Training", type="primary"):
        with st.spinner("Downloading dataset and training models natively..."):
            try:
                pipeline.main()
                st.success("Training complete! Loading dashboard space...")
                st.button("🔄 Load Dashboard")
            except Exception as e:
                st.error("Training failed! Environment trace log:")
                st.exception(e)
    st.stop()

# --------------------------------------------------------------------------
# Core Data Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "cluster_labels.pkl"), "rb") as f:
        cluster_map = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "rb") as f:
        sim_df = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "product_lookup.pkl"), "rb") as f:
        product_lookup = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rfm_table.pkl"), "rb") as f:
        rfm_table = pickle.load(f)
    return scaler, kmeans, cluster_map, sim_df, product_lookup, rfm_table

scaler, kmeans, cluster_map, sim_df, product_lookup, rfm_table = load_artifacts()

def predict_segment(recency, frequency, monetary, scaler, kmeans, cluster_map):
    X = np.array([[np.log1p(max(recency, 0)), np.log1p(max(frequency, 0)), np.log1p(max(monetary, 0))]])
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]
    return cluster_map.get(cluster, f"Cluster {cluster}")

def get_recommendations(product_name, sim_df, top_n=5):
    all_products = sim_df.index
    exact = [p for p in all_products if p.lower() == product_name.lower()]
    if exact:
        match = exact[0]
    else:
        contains = [p for p in all_products if product_name.lower() in p.lower()]
        if not contains:
            return None, []
        match = contains[0]

    scores = sim_df[match].drop(labels=[match], errors="ignore").sort_values(ascending=False)
    top = scores.head(top_n)
    return match, list(top.items())

# --------------------------------------------------------------------------
# Navigation Panel
# --------------------------------------------------------------------------
page = st.sidebar.radio(
    "Choose a module",
    ["🏠 Overview", "🎯 Product Recommendations", "👥 Customer Segmentation"],
)

# Custom Personal Branding Component Injection
st.sidebar.markdown(
    """
    <div class="dev-profile">
        <div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase;">Developer Profile</div>
        <div class="dev-name">Akshay Mahale</div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-top: 4px;">Cloud & DevOps Engineer</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Module 1: Overview
# --------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛒 Shopper Spectrum Dashboard")
    st.subheader("Customer Analytics Platform")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers analyzed", f"{rfm_table.shape[0]:,}")
    col2.metric("Products in catalog", f"{sim_df.shape[0]:,}")
    col3.metric("Customer segments", f"{rfm_table['Segment'].nunique()}")

    st.markdown("### Segment distribution")
    seg_counts = rfm_table["Segment"].value_counts()
    st.bar_chart(seg_counts)

    st.markdown("### Architecture Mechanics")
    st.markdown(
        """
