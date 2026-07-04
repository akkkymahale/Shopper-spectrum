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

# --------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .segment-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        color: white;
        font-size: 1.1rem;
    }
    .product-card {
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e3e7ed;
        margin-bottom: 0.6rem;
    }
    .rank-badge {
        background-color: #4B7BEC;
        color: white;
        border-radius: 999px;
        padding: 0.1rem 0.6rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
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
# SETUP AND TRAINING MANAGER
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

artifacts_exist = os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl"))

st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.caption("Customer Segmentation & Product Recommendations")

if not artifacts_exist:
    st.title("📦 First-Time Setup Required")
    st.write("The machine learning models need to be built and trained before the dashboard can load.")
    
    if st.button("🚀 Start Data Download & Model Training", type="primary"):
        with st.spinner("Downloading dataset from Google Drive and training models... This takes about 1-2 minutes."):
            try:
                result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True, check=True)
                st.success("Training complete! Click the button below to load the dashboard.")
                st.button("🔄 Load Dashboard")
            except subprocess.CalledProcessError as e:
                st.error("Training failed! Here is the error log to help us fix it:")
                st.code(e.stderr if e.stderr else e.stdout)
    st.stop()

# --------------------------------------------------------------------------
# Cached loaders
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
# Navigation
# --------------------------------------------------------------------------
page = st.sidebar.radio(
    "Choose a module",
    ["🏠 Overview", "🎯 Product Recommendations", "👥 Customer Segmentation"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\nBuilt on transaction-level e-commerce data using "
    "RFM analysis, KMeans clustering, and item-based collaborative filtering."
)

# --------------------------------------------------------------------------
# 🏠 Overview Page
# --------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🛒 Shopper Spectrum Dashboard")
    st.subheader("Customer Segmentation and Product Recommendations in E-Commerce")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers analyzed", f"{rfm_table.shape[0]:,}")
    col2.metric("Products in catalog", f"{sim_df.shape[0]:,}")
    col3.metric("Customer segments", f"{rfm_table['Segment'].nunique()}")

    st.markdown("### Segment distribution")
    seg_counts = rfm_table["Segment"].value_counts()
    st.bar_chart(seg_counts)

    st.markdown("### What this app does")
    st.markdown(
        """
        - **Product Recommendations** — Type a product name and get the 5 most similar
          products based on customers' co-purchase patterns (cosine similarity).
        - **Customer Segmentation** — Enter a customer's Recency, Frequency, and Monetary
          (RFM) values and instantly get their predicted segment.
        """
    )

# --------------------------------------------------------------------------
# 🎯 Recommendations Page
# --------------------------------------------------------------------------
elif page == "🎯 Product Recommendations":
    st.title("🎯 Product Recommendation Module")
    st.write("Enter a product name below to get the top 5 similar products.")

    product_input = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

    if st.button("Get Recommendations", type="primary"):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            with st.spinner("Finding similar products..."):
                matched_name, recs = get_recommendations(product_input, sim_df, top_n=5)

            if matched_name is None:
                st.error(f"No product matching “{product_input}” was found in the filtered catalog.")
            else:
                st.success(f"Showing recommendations based on: **{matched_name}**")
                for i, (name, score) in enumerate(recs, start=1):
                    st.markdown(
                        f"""
                        <div class="product-card">
                            <span class="rank-badge">#{i}</span>
                            <strong>{name}</strong>
                            <br><span style="color:#666;">Similarity score: {score:.3f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------------------------------
# 👥 Segmentation Page
# --------------------------------------------------------------------------
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Module")
    st.write("Enter a customer's RFM values to predict which segment they belong to.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (total spend, $)", min_value=0.0, value=500.0, step=10.0)

    if st.button("Predict Cluster", type="primary"):
        segment = predict_segment(recency, frequency, monetary, scaler, kmeans, cluster_map)
        color = SEGMENT_COLORS.get(segment, "#7f8c8d")
        desc = SEGMENT_DESCRIPTIONS.get(segment, "")

        st.markdown(
            f"""
            <div class="segment-card" style="background-color:{color};">
                <h3 style="margin:0;">Predicted Segment: {segment}</h3>
                <p style="margin-top:0.5rem;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
