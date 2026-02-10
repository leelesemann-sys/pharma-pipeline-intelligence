"""
Pharma Pipeline Intelligence Dashboard
Main entry point - Landing page with KPI overview and navigation.
"""

import streamlit as st

# ─────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Pipeline Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important;}
    h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important;}

    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
        color: #2563eb !important;
    }

    /* Dataframe Styling */
    .stDataFrame {border-radius: 8px; overflow: hidden;}

    /* Links */
    a {color: #2563eb !important; text-decoration: none !important;}
    a:hover {text-decoration: underline !important;}

    /* Divider */
    hr {border-color: #e2e8f0 !important; margin: 1.5rem 0 !important;}

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Pharma Pipeline Intelligence")
    st.markdown("**Diabetes & Obesity**")
    st.markdown("Competitive Landscape Dashboard")
    st.divider()
    st.markdown("#### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/1_Pipeline_Overview.py", label="Pipeline Overview", icon="📊")
    st.page_link("pages/2_Competitive_Landscape.py", label="Competitive Landscape", icon="🗺️")
    st.page_link("pages/3_Trial_Analytics.py", label="Trial Analytics", icon="📈")
    st.page_link("pages/4_Drug_Deep_Dive.py", label="Drug Deep Dive", icon="💊")
    st.page_link("pages/5_Market_Intelligence.py", label="Market Intelligence", icon="💰")
    st.page_link("pages/6_Safety_Profile.py", label="Safety Profile", icon="🛡️")
    st.page_link("pages/7_Patent_LOE.py", label="Patent & LOE", icon="📋")
    st.page_link("pages/8_ML_Predictions.py", label="ML Predictions", icon="🤖")
    st.divider()
    st.page_link("pages/0_How_It_Works.py", label="How It Works", icon="📖")
    st.divider()
    st.markdown(
        "<div style='color: #94a3b8; font-size: 0.75rem;'>"
        "Data: ClinicalTrials.gov, ChEMBL, openFDA<br>"
        "Updated: Feb 2026"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown("# 🔬 Pharma Pipeline Intelligence")
st.markdown("### Diabetes & Obesity Competitive Landscape Dashboard")
st.markdown("")

# Load KPIs
from utils.queries import get_kpis
with st.spinner("Loading dashboard data..."):
    kpis = get_kpis()

if not kpis.empty:
    row = kpis.iloc[0]

    # KPI Row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Drugs Tracked", f"{int(row['total_drugs']):,}")
    col2.metric("Clinical Trials", f"{int(row['total_trials']):,}")
    col3.metric("MoA Classes", f"{int(row['moa_classes']):,}")
    col4.metric("Indications", f"{int(row['total_indications']):,}")
    col5.metric("Companies", f"{int(row['total_companies']):,}")

    st.markdown("")

    # KPI Row 2
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Trials", f"{int(row['active_trials']):,}")
    col2.metric("Terminated Trials", f"{int(row['terminated_trials']):,}")
    col3.metric("Stale Trials", f"{int(row['stale_trials']):,}")
    col4.metric("Trials with Results", f"{int(row['trials_with_results']):,}")

st.divider()

# Quick Navigation Cards
st.markdown("### Explore the Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">📊 Pipeline Overview</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            All 43 drugs with filterable table, phase distribution, and approval status.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">🗺️ Competitive Landscape</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            MoA x Indication heatmap, trend analysis, hot & cold zones.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">📈 Trial Analytics</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            Status monitoring, trial starts over time, termination analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">💊 Drug Deep Dive</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            Deep analysis per drug: trials, approvals, competitive positioning.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Navigation Cards Row 2 (new tabs)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">💰 Market Intelligence</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            UK prescriptions, US Medicare spending, cross-market comparison.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">🛡️ Safety Profile</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            FDA FAERS adverse events, class signals, drug-to-drug comparison.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">📋 Patent & LOE</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            Orange/Purple Book patents, LOE timeline, exclusivity calendar.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #9d174d 0%, #ec4899 100%);
                color: white; padding: 20px; border-radius: 12px; height: 160px;">
        <h4 style="color: white !important; margin-bottom: 8px;">🤖 ML Predictions</h4>
        <p style="font-size: 0.85rem; opacity: 0.9;">
            Phase transition PoS, XGBoost feature importance, drug ranking.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 20px;'>"
    "Built with Streamlit + Plotly | Data: ClinicalTrials.gov, ChEMBL, openFDA, NHS, CMS, FDA FAERS | "
    "Azure SQL Backend"
    "</div>",
    unsafe_allow_html=True,
)
