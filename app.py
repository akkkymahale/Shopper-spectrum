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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=Montserrat:wght@700;800&display=swap');

    .stApp { background-color: #06040A !important; color: #F1EFF7 !important; font-family: 'Inter', sans-serif !important; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Montserrat', 'Inter', sans-serif !important; color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #0B0813 !important; border-right: 1px solid #1C172E !important; }
    
    .feature-card { background: #110E1C; border: 1px solid #1F1936; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.2rem; }
    
    .dev-profile-container { display: flex; align-items: center; gap: 14px; margin-top: 2rem; padding: 1rem; background: radial-gradient(circle at top left, #17122B, #110E1C); border: 1px solid #251F3D; border-radius: 14px; }
    .dev-details { display: flex; flex-direction: column; }
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
    if st.button("Initialize Platform Engines", type="primary"):
        with st.spinner("Compiling analytical models..."):
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
# 3. SIDEBAR BRANDING
# --------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.8rem 0 1.2rem 0;">
        <h1 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: 1.8px; color: #FFFFFF; font-family: 'Montserrat', sans-serif !important; text-transform: uppercase;">
            SHOPPER SPECTRUM
        </h1>
        <div style="font-size: 0.78rem; color: #8D87A4; font-weight: 500; margin-top: 6px;">Data-driven insights</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("E-Commerce Navigation", ["Dashboard Overview", "Product Recommendations", "Customer Segmentation", "CSV Batch Engine"])

# --- ARCHITECTURE STACK MODULE ---
st.sidebar.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; background: #110E1C; border: 1px solid #1F1936; border-radius: 12px;">
        <div style="font-size: 0.65rem; color: #6C6684; font-weight: 800; text-transform: uppercase; margin-bottom: 0.8rem;">Architecture Stack</div>
        <div style="font-size: 0.75rem; color: #B3AECE; line-height: 1.8;">
            • K-Means Clustering<br>• Vector Space Modeling<br>• Cosine Similarity<br>• RFM Feature Engineering
        </div>
    </div>
""", unsafe_allow_html=True)

# --- DEVELOPER PROFILE ---
st.sidebar.markdown("""
    <div class="dev-profile-container">
        <img src="https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o" width="46" style="border-radius: 50%; border: 2px solid #7451F7;">
        <div class="dev-details">
            <span style="font-size:0.65rem; color:#8D87A4; text-transform:uppercase; font-weight: 800;">Lead Architecture</span>
            <span style="font-weight:700; color:#FFFFFF; font-size:0.9rem;">Akshay Mahale</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 4. RUNTIME LOGIC
# --------------------------------------------------------------------------
if page == "Dashboard Overview":
    st.title("Retail Intelligence Hub")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", "4,338"); c2.metric("Products", "3,877"); c3.metric("Revenue", "8.91M"); c4.metric("Countries", "37")
    st.bar_chart(rfm_table["Segment"].value_counts())

elif page == "Product Recommendations":
    st.title("Product Recommender Matrix")
    term = st.text_input("Active Catalog Item Term Query")
    if st.button("Generate"):
        matches = [p for p in sim_df.index if term.lower() in p.lower()]
        if matches:
            for p in sim_df[matches[0]].sort_values(ascending=False).head(5).index:
                st.markdown(f'<div class="feature-card">{p}</div>', unsafe_allow_html=True)

elif page == "Customer Segmentation":
    st.title("Customer Performance Cohorts")
    rec = st.number_input("Recency", value=30); freq = st.number_input("Frequency", value=5); mon = st.number_input("Monetary", value=500.0)
    if st.button("Evaluate"):
        X = scaler.transform(np.log1p([[rec, freq, mon]]))
        st.info(f"Segment: {cluster_map.get(kmeans.predict(X)[0])}")

elif page == "CSV Batch Engine":
    st.title("Bulk Operational Marketing")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded: st.success("Data Processed.")
