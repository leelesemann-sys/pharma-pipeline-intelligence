"""
Tab 1: Pipeline Overview
All 43 drugs as interactive, filterable table with phase distribution chart.
"""

import streamlit as st
import pandas as pd
from utils.queries import get_pipeline_overview, get_drugs_by_phase, get_filter_options, get_kpis
from utils.charts import pipeline_phase_bar, PRIMARY

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Overview | Pharma Pipeline Intelligence",
    page_icon="📊",
    layout="wide",
)

# Custom CSS (same as app.py)
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important; font-size: 0.85rem !important;
        font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;
    }
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Pharma Pipeline Intelligence")
    st.divider()
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/1_Pipeline_Overview.py", label="Pipeline Overview", icon="📊")
    st.page_link("pages/2_Competitive_Landscape.py", label="Competitive Landscape", icon="🗺️")
    st.page_link("pages/3_Trial_Analytics.py", label="Trial Analytics", icon="📈")
    st.page_link("pages/4_Drug_Deep_Dive.py", label="Drug Deep Dive", icon="💊")
    st.page_link("pages/5_Market_Intelligence.py", label="Market Intelligence", icon="💰")
    st.page_link("pages/6_Safety_Profile.py", label="Safety Profile", icon="🛡️")
    st.page_link("pages/7_Patent_LOE.py", label="Patent & LOE", icon="📋")
    st.divider()
    st.page_link("pages/0_How_It_Works.py", label="How It Works", icon="📖")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 📊 Pipeline Overview")
st.markdown("All tracked drugs with development status, trial activity, and approval information.")
st.markdown("")

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
with st.spinner("Loading pipeline data..."):
    kpis = get_kpis()
    pipeline_df = get_pipeline_overview()
    phase_df = get_drugs_by_phase()
    filter_opts = get_filter_options()

if not kpis.empty:
    row = kpis.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Drugs Tracked", f"{int(row['total_drugs']):,}")
    col2.metric("Clinical Trials", f"{int(row['total_trials']):,}")
    col3.metric("MoA Classes", f"{int(row['moa_classes']):,}")
    col4.metric("Indications", f"{int(row['total_indications']):,}")
    col5.metric("Active Trials", f"{int(row['active_trials']):,}")

st.divider()

# ─────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────
st.markdown("### Filters")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    selected_moa = st.multiselect(
        "MoA Class",
        options=filter_opts["moa_classes"],
        default=[],
        placeholder="All MoA Classes",
    )

with fcol2:
    selected_indication = st.multiselect(
        "Indication",
        options=filter_opts["indications"],
        default=[],
        placeholder="All Indications",
    )

with fcol3:
    selected_phase = st.multiselect(
        "Highest Phase",
        options=filter_opts["phases"],
        default=[],
        placeholder="All Phases",
    )

with fcol4:
    selected_sponsor = st.multiselect(
        "Sponsor Type",
        options=filter_opts["sponsor_types"],
        default=[],
        placeholder="All Sponsor Types",
    )

# Apply filters
filtered_df = pipeline_df.copy()

if selected_moa:
    filtered_df = filtered_df[filtered_df["moa_class"].isin(selected_moa)]
if selected_indication:
    # Filter drugs that have at least one of the selected indications
    filtered_df = filtered_df[filtered_df["indications"].apply(
        lambda x: any(ind in str(x) for ind in selected_indication) if pd.notna(x) else False
    )]
if selected_phase:
    filtered_df = filtered_df[filtered_df["highest_phase"].isin(selected_phase)]
if selected_sponsor:
    filtered_df = filtered_df[filtered_df["company_type"].isin(selected_sponsor)]

st.markdown(f"**Showing {len(filtered_df)} of {len(pipeline_df)} drugs**")
st.markdown("")

# ─────────────────────────────────────────────
# Pipeline Table
# ─────────────────────────────────────────────
st.markdown("### Drug Pipeline Table")

# Prepare display columns
display_df = filtered_df[[
    "drug_name", "moa_class", "highest_phase", "modality",
    "top_sponsor", "company_type", "indications",
    "approved_indications", "total_trials", "active_trials",
    "phase3_trials", "has_fda_approval"
]].copy()

# Format FDA approval column
display_df["has_fda_approval"] = display_df["has_fda_approval"].map({1: "Yes", 0: "No"})

# Rename columns for display
display_df.columns = [
    "Drug", "MoA Class", "Highest Phase", "Modality",
    "Top Sponsor", "Sponsor Type", "All Indications",
    "Approved Indications", "Total Trials", "Active Trials",
    "Phase 3 Trials", "FDA Approved"
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Drug": st.column_config.TextColumn("Drug", width="medium"),
        "MoA Class": st.column_config.TextColumn("MoA Class", width="medium"),
        "Highest Phase": st.column_config.TextColumn("Phase", width="small"),
        "Total Trials": st.column_config.NumberColumn("Total Trials", format="%d"),
        "Active Trials": st.column_config.NumberColumn("Active Trials", format="%d"),
        "Phase 3 Trials": st.column_config.NumberColumn("P3 Trials", format="%d"),
    },
)

# CSV Download
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Pipeline Data (CSV)",
    csv,
    "pharma_pipeline_overview.csv",
    "text/csv",
    key="pipeline_csv",
)

st.divider()

# ─────────────────────────────────────────────
# Pipeline Phase Chart
# ─────────────────────────────────────────────
st.markdown("### Drugs by Development Phase")

col_chart, col_space = st.columns([3, 1])

with col_chart:
    if not phase_df.empty:
        fig = pipeline_phase_bar(phase_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No phase data available.")
