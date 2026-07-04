"""
Shopper Spectrum: Customer Segmentation & Product Recommendations
Streamlit web app.
"""

import os
import pickle
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if model files exist
artifacts_exist = os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))

st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.caption("Customer Segmentation & Product Recommendations")

# If models don't exist yet, show a setup page instead of throwing a subprocess crash
if not artifacts_exist:
    st.title("📦 First-Time Setup Required")
    st.write("The machine learning models need to be built and trained before the dashboard can load.")
    
    if st.button("🚀 Start Data Download & Model Training", type="primary"):
        with st.spinner("Downloading dataset from Google Drive and training models... This takes about 1-2 minutes."):
            try:
                # Force python to run pipeline.py directly
                result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True, check=True)
                st.success("Training complete! Click the button below to load the dashboard.")
                st.button("🔄 Load Dashboard")
            except subprocess.CalledProcessError as e:
                st.error("Training failed! Here is the error log to help us fix it:")
                st.code(e.stderr if e.stderr else e.stdout)
    st.stop()

# --------------------------------------------------------------------------
# Load artifacts once training is complete
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

# --------------------------------------------------------------------------
# Main App Layout (Only shows after models exist)
# --------------------------------------------------------------------------
page = st.sidebar.radio(
    "Choose a module",
    ["🏠 Overview", "🎯 Product Recommendations", "👥 Customer Segmentation"],
)

if page == "🏠 Overview":
    st.title("🛒 Shopper Spectrum Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers analyzed", f"{rfm_table.shape[0]:,}")
    col2.metric("Products in catalog", f"{sim_df.shape[0]:,}")
    col3.metric("Customer segments", f"{rfm_table['Segment'].nunique()}")
    st.markdown("### Segment distribution")
    st.bar_chart(rfm_table["Segment"].value_counts())

elif page == "🎯 Product Recommendations":
    st.title("🎯 Product Recommendation Module")
    product_input = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")
    # Recommendation logic goes here...

elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Module")
    # Segmentation inputs go here...
