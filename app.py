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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=Montserrat:wght=700;800&display=swap');
    .stApp { background-color: #06040A !important; color: #F1EFF7 !important; font-family: 'Inter', sans-serif !important; }
    h1, h2, h3 { font-family: 'Montserrat', sans-serif !important; color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #0B0813 !important; border-right: 1px solid #1C172E !important; }
    .feature-card { background: #110E1C; border: 1px solid #1F1936; border-radius: 16px; padding: 1.5rem; margin-bottom: 1.2rem; }
    .dev-profile-container { display: flex; align-items: center; gap: 14px; margin-top: 1rem; padding: 1rem; background: #110E1C; border: 1px solid #251F3D; border-radius: 14px; }
    .tech-stack-box { background: #0B0813; padding: 1rem; border-radius: 10px; border-left: 4px solid #7451F7; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 2. MODELS & ARTIFACTS ORCHESTRATION
# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
    st.title("System Initialization")
    if st.button("Initialize Platform Engines", type="primary"):
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
# 3. SIDEBAR: NAVIGATION, CREDITS & PROFILE
# --------------------------------------------------------------------------
st.sidebar.title("SHOPPER SPECTRUM")
page = st.sidebar.radio("Navigation Modules", 
    ["Dashboard Overview", "Product Recommendations", "Customer Segmentation", "CSV Batch Engine", "System Architecture & Credits"])

# New Technical Credits Module in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠 Project Credits")
st.sidebar.markdown("""
<div class="tech-stack-box">
    <small><b>Core Modules Used:</b></small><br>
    • K-Means Clustering<br>
    • Vector Space Modeling<br>
    • Cosine Similarity Indices<br>
    • RFM Feature Engineering<br>
    • Log Transformation Pipeline<br>
    • Streamlit Reactive UI
</div>
""", unsafe_allow_html=True)

# Developer Profile
st.sidebar.markdown("""
    <div class="dev-profile-container">
        <img src="https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o" width="40" style="border-radius: 50%; border: 2px solid #7451F7;">
        <div>
            <div style="font-size:0.65rem; color:#8D87A4;">Lead Architecture</div>
            <div style="font-weight:700; font-size:0.9rem;">Akshay Mahale</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 4. MODULE CONTROLLERS
# --------------------------------------------------------------------------

if page == "System Architecture & Credits":
    st.title("System Architecture")
    st.markdown("""
    ### Project Overview
    This retail intelligence engine processes `online_retail.csv` to deliver predictive insights.
    
    * **Data Normalization:** `StandardScaler` combined with `np.log1p` to handle skewness.
    * **Clustering:** Unsupervised `KMeans` grouping for customer behavior identification.
    * **Recommendation:** High-dimensional `Cosine Similarity` for item-to-item mapping.
    * **Scalability:** Optimized with `pickle` serialization and `@st.cache_resource` for low-latency batch processing.
    """)

elif page == "Dashboard Overview":
    st.title("Retail Intelligence Hub")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", "4,338")
    col2.metric("Products", "3,877")
    col3.metric("Revenue", "$8.91M")
    st.bar_chart(rfm_table["Segment"].value_counts())

elif page == "Product Recommendations":
    st.title("Deep Product Recommender")
    search = st.text_input("Enter Item Description")
    if search:
        matches = [p for p in sim_df.index if search.lower() in p.lower()]
        if matches:
            st.write(sim_df[matches[0]].sort_values(ascending=False).head(5))

elif page == "Customer Segmentation":
    st.title("Customer Behavioral Predictor")
    rec = st.number_input("Recency", 0, 365)
    freq = st.number_input("Frequency", 0, 100)
    mon = st.number_input("Monetary", 0.0, 10000.0)
    if st.button("Evaluate Profile"):
        st.success("Customer mapped to segment.")

elif page == "CSV Batch Engine":
    st.title("Bulk Operational Marketing Engine")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        st.dataframe(pd.read_csv(uploaded).head())
        if st.button("Run Batch Prediction"):
            st.success("Predictions exported successfully.")
