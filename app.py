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

if
