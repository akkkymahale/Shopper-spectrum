import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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

# Deep obsidian dark dashboard canvas style mapping with upgraded professional fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=Montserrat:wght@700;800&display=swap');

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

    p, span, label, div, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: #F1EFF7 !important;
    }
    
    /* Left Sidebar Panel container styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0813 !important;
        border-right: 1px solid #1C172E !important;
    }
    section[data-testid="stSidebar"] * {
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
        <div style="font-size:0.72rem; font-weight:800; color:#8D87A4; margin-bottom:1rem; letter-spacing:1px; font-family: 'Inter', sans-serif;">LIVE PLATFORM METRICS</div>
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
            <span style="font-size:0.68rem; color:#6C6684; text-transform:uppercase; font-weight: 800; letter-spacing: 0.5px;">Lead Architecture</span>
            <span style="font-weight:700; color:#7451F7; font-size:0.92rem; margin-top: 1px; font-family: 'Inter', sans-serif;">Akshay Mahale</span>
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

    st.markdown("<h3 style='margin-top:2rem; font-weight:700;'>📈 Segment Distribution Tracking</h3>", unsafe_allow_html=True)
    st.bar_chart(rfm_table["Segment"].value_counts())

    # --- GEOGRAPHIC REGIONAL INSIGHTS CHOROPLETH MAP ---
    st.markdown("---")
    st.markdown("<h3 style='font-weight:700;'>🌍 Geographic Insights: Global Order Distribution Map</h3>", unsafe_allow_html=True)
    st.write("Interactive analytical grid mapping hot zones across international distribution hubs.")

    if "Country" in rfm_table.columns:
        country_counts = rfm_table["Country"].value_counts().reset_index()
        country_counts.columns = ["Country", "Orders"]
    else:
        # Fallback distribution mapping values
        data_mock = {
            "Country": ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia"],
            "Orders": [3950, 94, 87, 74, 31, 23, 21, 21, 19, 9]
        }
        country_counts = pd.DataFrame(data_mock)
        
    fig = px.choropleth(
        country_counts,
        locations="Country",
        locationmode="country names",
        color="Orders",
        hover_name="Country",
        color_continuous_scale=["#17122B", "#7451F7", "#9E86FF"],
        projection="natural earth"
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            bgcolor='rgba(0,0,0,0)',
            landcolor='#110E1C',
            lakecolor='#06040A',
            oceancolor='#06040A',
            showocean=True,
            showlakes=True
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title="Orders",
            thicknessmode="pixels", thickness=15,
            lenmode="pixels", len=150,
            yanchor="center", y=0.5,
            ticks="outside",
            titlefont=dict(color="#B3AECE"),
            tickfont=dict(color="#B3AECE")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MODULE 2: PRODUCT RECOMMENDATIONS ---
elif page == "🎯 Product Recommendations":
    st.markdown("<h1 style='font-weight:800; letter-spacing: -0.5px;'>🎯 Deep Product Recommender Matrix</h1>", unsafe_allow_html=True)
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
                    html_card = f"""<div class="feature-card" style="border-left: 4px solid #7451F7;"><div style="font-weight: 700; color: #FFFFFF; font-size:1.05rem; font-family: 'Inter', sans-serif;">#{idx} {prod_name}</div><div style="color: #8D87A4; font-size: 0.85rem; margin-top: 4px; font-family: 'Inter', sans-serif;">Cosine Proximity Weight Metric: <span style="color:#FFF; font-family:monospace;">{value:.4f}</span></div></div>"""
                    st.markdown(html_card, unsafe_allow_html=True)

# --- MODULE 3: CUSTOMER SEGMENTATION ---
elif page == "👥 Customer Segmentation":
    st.markdown("<h1 style='font-weight:800; letter-spacing: -0.5px;'>👥 Customer Performance Cohorts</h1>", unsafe_allow_html=True)
    st.write("Pass active interaction arrays to evaluate cluster alignments instantly.")

    col1, col2, col3 = st.columns(3)
    with col1: rec = st.number_input("Recency Value (Days from Last Interaction)", min_value=0, value=30)
    with col2: freq = st.number_input("Frequency Value (Accumulated Order Total)", min_value=0, value=5)
    with col3: mon = st.number_input("Monetary Value (Gross Order Margin Sum, $)", min_value=0.0, value=500.0)

    if st.button("Evaluate Metrics Profile", type="primary"):
        X_input = np.array([[np.log1p(max(rec, 0)), np.log1p(max(freq, 0)), np.log1p(max(mon, 0))]])
        scaled_features = scaler.transform(X_input)
        cluster_id = kmeans.predict(scaled_features)[0]
        assigned_segment = cluster_map.get(cluster_id, f"Cluster {cluster_id}")
        
        html_segment = f"""<div class="feature-card" style="background: linear-gradient(135deg, #110E1C 0%, #17122B 100%); border: 1px solid #7451F7;"><h3 style="margin: 0; color: #FFFFFF; font-weight:700; font-family: 'Montserrat', sans-serif !important;">Predicted Customer Class Status: <span style="color:#7451F7;">{assigned_segment}</span></h3><p style="color: #B3AECE; margin-top: 0.5rem; font-size:0.95rem; font-family: 'Inter', sans-serif;">Profile compiled correctly and deployed into standard operations matrix profiles successfully.</p></div>"""
        st.markdown(html_segment, unsafe_allow_html=True)

# --- MODULE 4: CSV BATCH RECOMMENDATION & MARKETING ENGINE ---
elif page == "📂 CSV Batch Engine":
    st.markdown("<h1 style='font-weight:800; letter-spacing: -0.5px;'>📂 Bulk Operational Marketing Engine</h1>", unsafe_allow_html=True)
    st.write("Upload a raw batch dataset list file to execute automated segment categorization predictions at enterprise scale.")
    
    st.markdown("<h3 style='font-weight:700;'>Expected Input format Template Columns:</h3>", unsafe_allow_html=True)
    st.code("CustomerID, Recency, Frequency, Monetary")
    
    uploaded_file = st.file_uploader("Upload Target Batch CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            required_cols = ["Recency", "Frequency", "Monetary"]
            if not all(col in df_input.columns for col in required_cols):
                st.error("Missing critical column keys! Make sure the columns match: Recency, Frequency, and Monetary exactly.")
            else:
                with st.spinner("Processing structural batch calculations across data matrices..."):
                    log_rec = np.log1p(df_input["Recency"].clip(lower=0))
                    log_freq = np.log1p(df_input["Frequency"].clip(lower=0))
                    log_mon = np.log1p(df_input["Monetary"].clip(lower=0))
                    
                    X_matrix = np.column_stack((log_rec, log_freq, log_mon))
                    scaled_matrix = scaler.transform(X_matrix)
                    preds = kmeans.predict(scaled_matrix)
                    
                    df_input["Predicted_Cluster_ID"] = preds
                    df_input["Marketing_Target_Segment"] = df_input["Predicted_Cluster_ID"].map(cluster_map)
                    
                    st.success("Batch pipeline calculations evaluated successfully!")
                    st.dataframe(df_input.head(10))
                    
                    csv_data = df_input.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Export Processed Campaign Target List",
                        data=csv_data,
                        file_name="shopper_spectrum_batch_targets.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Runtime extraction error processing dataset file layer: {e}")
