# --------------------------------------------------------------------------
# 3. SIDEBAR BRANDING & REFINED LAYOUT
# --------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 0.8rem 0 1.2rem 0;">
        <h1 style="margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: 1.8px; color: #FFFFFF; font-family: 'Montserrat', sans-serif !important; text-transform: uppercase;">
            SHOPPER SPECTRUM
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation
page = st.sidebar.radio("E-Commerce Navigation", ["Dashboard Overview", "Product Recommendations", "Customer Segmentation", "CSV Batch Engine"])

# Spacing + Architecture Module
st.sidebar.markdown("<div style='margin-top: 3.5rem;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="padding: 1.2rem; background: #110E1C; border: 1px solid #1F1936; border-radius: 16px;">
        <div style="font-size: 0.65rem; color: #8D87A4; font-weight: 800; text-transform: uppercase; margin-bottom: 0.8rem; letter-spacing: 1px;">Architecture Stack</div>
        <div style="font-size: 0.75rem; color: #B3AECE; line-height: 1.8; font-family: 'Inter', sans-serif;">
            • K-Means Clustering<br>
            • Vector Space Modeling<br>
            • Cosine Similarity<br>
            • RFM Feature Engineering
        </div>
    </div>
""", unsafe_allow_html=True)

# Developer Profile
st.sidebar.markdown("""
    <div class="dev-profile-container">
        <img src="https://lh3.googleusercontent.com/d/1wGYtf22gb7TGrSl2tJRJFnulwCVSRk3o" width="46" style="border-radius: 50%; border: 2px solid #7451F7;">
        <div class="dev-details">
            <span style="font-size:0.65rem; color:#8D87A4; text-transform:uppercase; font-weight: 800;">Lead Architecture</span>
            <span style="font-weight:700; color:#FFFFFF; font-size:0.9rem;">Akshay Mahale</span>
        </div>
    </div>
""", unsafe_allow_html=True)
