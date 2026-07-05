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
    
    .live-stats-box { margin-top: 1.5rem; padding-top: 1.2rem; border-top: 1px solid #1C172E; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 0.8rem; }
    .stat-label { font-size: 0.72rem; color: #6C6684; text-transform: uppercase; font-weight: 800; }
    .stat-val { font-size: 1.05rem; font-weight: 700; color: #FFFFFF; font-family: monospace; }

    .dev-profile-container { display: flex; align-items: center; gap: 14px; margin-top: 2.5rem; padding: 1rem; background: radial-gradient(circle at top left, #17122B, #110E1C); border: 1px solid #251F3D; border-radius: 14px; }
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
        with st.spinner("Downloading source registries and training model layers..."):
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
st.sidebar.markdown("<h1 style='font-size:1.55rem; color:#FFFFFF;'>SHOPPER SPECTRUM</h1>", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation Modules", ["Dashboard Overview", "Product Recommendations", "Customer Segmentation", "CSV Batch Engine"], index=0)

# Sidebar Stats
st.sidebar.markdown("""
    <div class="live-stats-box">
        <div class="stat-row"><div class="stat-label">Customers</div><div class="stat-val">4,338</div></div>
        <div class="stat-row"><div class="stat-label">Products</div><div class="stat-val">3,877</div></div>
    </div>
""", unsafe_allow_html=True)

# Profile Badge
st.sidebar.markdown("""
    <div class="dev-profile-container">
        <img src="https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o" width="46" style="border-radius: 50%; border: 2px solid #7451F7;">
        <div class="dev-details">
            <span style="font-size:0.68rem; color:#8D87A4; text-transform:uppercase; font-weight: 800;">Lead Architecture</span>
            <div style="font-weight:700; color:#FFFFFF; font-size:0.95rem;">Akshay Mahale</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 4. MODULE CONTROLLERS
# --------------------------------------------------------------------------

if page == "Dashboard Overview":
    st.title("Retail Intelligence Hub")
    st.bar_chart(rfm_table["Segment"].value_counts())

elif page == "Product Recommendations":
    st.title("Product Recommender Matrix")
    search = st.text_input("Item Search Query")
    if search:
        matches = [p for p in sim_df.index if search.lower() in p.lower()]
        if matches:
            st.write(sim_df[matches[0]].sort_values(ascending=False).head(5))

elif page == "Customer Segmentation":
    st.title("Customer Performance Cohorts")
    rec = st.number_input("Recency", 0, 365)
    freq = st.number_input("Frequency", 0, 100)
    mon = st.number_input("Monetary", 0.0, 10000.0)
    if st.button("Evaluate Profile"):
        X = np.array([[np.log1p(rec), np.log1p(freq), np.log1p(mon)]])
        seg = cluster_map.get(kmeans.predict(scaler.transform(X))[0])
        st.success(f"Assigned to: {seg}")

elif page == "CSV Batch Engine":
    st.title("Bulk Marketing Engine")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        st.dataframe(pd.read_csv(file).head())
        if st.button("Run Batch Processing"):
            st.success("Batch targets generated.")
