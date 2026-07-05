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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep obsidian dark dashboard canvas style mapping
st.markdown(
    """
    <style>
    /* Global Application Canvas Base */
    .stApp {
        background-color: #06040A !important;
        color: #F1EFF7 !important;
    }
    
    /* Force consistent color across headings and markdown elements */
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown {
        color: #F1EFF7 !important;
    }
    
    /* Left Sidebar Panel container styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0813 !important;
        border-right: 1px solid #1C172E !important;
    }
    section[data-testid="stSidebar"] * {
        color: #B3AECE !important;
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
        font-weight: 700 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8D87A4 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
        margin-bottom: 0.8rem;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6C6684;
        text-transform: uppercase;
        font-weight: bold;
    }
    .stat-val {
        font-size: 1.1rem;
        font-weight: bold;
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
    .dev-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #7451F7;
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
    st.title("⚡ System Initialization")
    st.write("Constructing analytical cluster matrices and compiling local workspace assets...")
    if st.button("Initialize Platform Engines", type="primary"):
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
        <h1 style="margin: 0; font-size: 1.65rem; font-weight: 800; letter-spacing: 1.5px; color: #FFFFFF; font-family: 'Helvetica Neue', Arial, sans-serif; text-transform: uppercase;">
            SHOPPER SPECTRUM
        </h1>
        <div style="font-size: 0.8rem; color: #8D87A4; font-weight: 500; margin-top: 6px; letter-spacing: 0.3px;">
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
        "📊 Dashboard Overview", 
        "🎯 Product Recommendations", 
        "👥 Customer Segmentation",
        "📂 CSV Batch Engine"
    ],
    index=0
)

# Live Stats Metrics Component Box
st.sidebar.markdown(
    """
    <div class="live-stats-box">
        <div style="font-size:0.75rem; font-weight:bold; color:#8D87A4; margin-bottom:1rem; letter-spacing:1px;">LIVE PLATFORM METRICS</div>
        <div class="stat-row"><div class="stat-label">Customers</div><div class="stat-val">4,338</div></div>
        <div class="stat-row"><div class="stat-label">Products</div><div class="stat-val">3,877</div></div>
        <div class="stat-row"><div class="stat-label">Revenue Target</div><div class="stat-val">8.91M USD</div></div>
        <div class="stat-row"><div class="stat-label">Countries Verified</div><div class="stat-val">37</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Profile link setup
dev_photo_url = "https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o"

st.sidebar.markdown(
    f"""
    <div class="dev-profile-container">
        <img src="{dev_photo_url}" class="dev-avatar" alt="Akshay Mahale Profile">
        <div class="dev-details">
            <span style="font-size:0.7rem; color:#6C6684; text-transform:uppercase; font-weight: bold; letter-spacing: 0.5px;">Lead Architecture</span>
            <span style="font-weight:bold; color:#7451F7; font-size:0.95rem; margin-top: 1px;">Akshay Mahale</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 4. MODULE CONTROLLERS / RUNTIME LOGIC
# --------------------------------------------------------------------------

# --- MODULE 1: DASHBOARD OVERVIEW ---
if page == "📊 Dashboard Overview":
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">Retail Intelligence Hub</h1>
            <p style="color: #8D87A4; font-size: 1.05rem; max-width: 700px; margin-top: 0.4rem;">Translating continuous consumer interaction loops into structured, behavioral analytical matrix environments dynamically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "4,338")
    c2.metric("Unique Products", "3,877")
    c3.metric("Total Revenue", "8.91M")
    c4.metric("Countries", "37")

    st.markdown("### 📈 Segment Distribution Tracking")
    st.bar_chart(rfm_table["Segment"].value_counts())

    # --- GEOGRAPHIC REGIONAL INSIGHTS GRAPH ---
    st.markdown("---")
    st.markdown("### 🌍 Geographic Insights: Global Order Distribution")
    st.write("Slicing order capacity aggregates and interaction frequencies across distribution country origins.")

    if "Country" in rfm_table.columns:
        country_counts = rfm_table["Country"].value_counts().head(10)
    else:
        country_counts = pd.Series({
            "United Kingdom": 3950,
            "Germany": 94,
            "France": 87,
            "EIRE": 74,
            "Spain": 31,
            "Netherlands": 23,
            "Belgium": 21,
            "Switzerland": 21,
            "Portugal": 19,
            "Australia": 9
        })
        
    st.bar_chart(country_counts)

# --- MODULE 2: PRODUCT RECOMMENDATIONS ---
elif page == "🎯 Product Recommendations":
    st.title("🎯 Deep Product Recommender Matrix")
    st.write("Perform search index matching across catalog feature arrays via high-dimensional Cosine Similarity.")

    search_term = st.text_input("Active Catalog Item Term Query", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")
    top_n = st.slider("Select quantity of target recommendations to fetch", min_value=1, max_value=10, value=5)

    if st.button("Generate Recommendations", type="primary"):
        if search_term.strip():
            all_prods = sim_df.index
            matches = [p for p in all_prods if search_term.lower() in p.lower()]
            
            if not matches:
                st.error("No active catalog items matched your key phrase input query.")
            else:
                target_key = matches[0]
                st.info(f"Target Selection Vector Context mapped to: **{target_key}**")
                
                scores = sim_df[target_key].drop(labels=[target_key], errors="ignore").sort_values(ascending=False).head(top_n)
                
                for idx, (prod_name, value) in enumerate(scores.items(), 1):
                    html_card = f"""
                    <div class="feature-card" style="border-left: 4px solid #7451F7;">
                        <div style="font-weight: bold; color: #FFFFFF; font-size:1.05rem;">#{idx} {prod_name}</div>
                        <div style="color: #8D87A4; font-size: 0.85rem; margin-top: 4px;">Cosine Proximity Weight Metric: <span style="color:#FFF; font-family:monospace;">{value:.4f}</span></div>
                    </div>
                    """
                    st.markdown(html_card, unsafe_allow_html=True)

# --- MODULE 3: CUSTOMER SEGMENTATION ---
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Performance Cohorts")
    st.write("Pass active interaction arrays to evaluate cluster alignments instantly.")

    col1, col2, col3 = st.columns(3)
    with col1: rec = st.number_input("Recency Value (Days from Last Interaction)", min_value=0, value=30)
    with col2: freq = st.number_input("Frequency Value (Accumulated Order Total)", min_value=0, value=5)
    with col3: mon = st.number_input("Monetary Value (Gross Order Margin Sum, $)", min_value=0.0, value=500.0)

    if st.button("Evaluate Metrics Profile", type="primary"):
        X_input = np.array([[np.log1p(max(rec, 0)), np.log1p(max(freq, 0)), np.log1p(max(mon, 0))]])
        scaled_features = scaler.transform(X_input)
        cluster_id = kmeans.predict(scaled_features)[0]
        assigned_segment = cluster_map
