"""
Tab 3: Trial Analytics
Deep analysis of the trial landscape: status monitoring, stale detection, sponsor analysis.
"""

import streamlit as st
import pandas as pd
from utils.queries import (
    get_kpis,
    get_trial_status_breakdown,
    get_trial_phase_breakdown,
    get_trial_starts_over_time,
    get_top_sponsors,
    get_termination_rate_by_moa,
    get_termination_rate_by_indication,
    get_termination_rate_by_phase,
    get_stale_trials,
    get_filter_options,
)
from utils.charts import (
    donut_chart,
    stacked_area_chart,
    horizontal_bar_chart,
    STATUS_COLORS,
    PRIMARY,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trial Analytics | Pharma Pipeline Intelligence",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important;}
    h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important; font-size: 0.85rem !important;
        font-weight: 500 !important; text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700 !important;
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
st.markdown("# 📈 Trial Analytics")
st.markdown("Status monitoring, trial activity trends, termination analysis, and stale trial detection.")
st.markdown("")

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
with st.spinner("Loading trial analytics..."):
    kpis = get_kpis()
    status_df = get_trial_status_breakdown()
    phase_breakdown_df = get_trial_phase_breakdown()
    starts_df = get_trial_starts_over_time()
    sponsors_df = get_top_sponsors()
    term_moa_df = get_termination_rate_by_moa()
    term_ind_df = get_termination_rate_by_indication()
    term_phase_df = get_termination_rate_by_phase()

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
if not kpis.empty:
    row = kpis.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trials", f"{int(row['total_trials']):,}")
    col2.metric("Recruiting", f"{int(row['active_trials']):,}")
    col3.metric("Terminated", f"{int(row['terminated_trials']):,}")
    col4.metric("Stale Trials", f"{int(row['stale_trials']):,}")
    col5.metric("With Results", f"{int(row['trials_with_results']):,}")

st.divider()

# ─────────────────────────────────────────────
# Status & Phase Donuts
# ─────────────────────────────────────────────
col_status, col_phase = st.columns(2)

with col_status:
    st.markdown("### Trial Status Breakdown")
    if not status_df.empty:
        fig = donut_chart(
            status_df,
            names_col="overall_status",
            values_col="trial_count",
            title="Trials by Status",
            color_map=STATUS_COLORS,
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

with col_phase:
    st.markdown("### Trial Phase Breakdown")
    if not phase_breakdown_df.empty:
        fig = donut_chart(
            phase_breakdown_df,
            names_col="phase",
            values_col="trial_count",
            title="Trials by Phase",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# Trial Starts Over Time
# ─────────────────────────────────────────────
st.markdown("### Trial Starts Over Time")

if not starts_df.empty:
    fig = stacked_area_chart(
        starts_df,
        x_col="start_year",
        y_col="trial_count",
        color_col="phase",
        title="Interventional Trial Starts by Phase (2000-2026)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# Top Sponsors & Termination Analysis
# ─────────────────────────────────────────────
col_sponsors, col_term = st.columns(2)

with col_sponsors:
    st.markdown("### Top 20 Sponsors")
    if not sponsors_df.empty:
        # Truncate long sponsor names
        sponsors_display = sponsors_df.copy()
        sponsors_display["lead_sponsor_name"] = sponsors_display["lead_sponsor_name"].apply(
            lambda x: x[:40] + "..." if len(str(x)) > 40 else x
        )
        fig = horizontal_bar_chart(
            sponsors_display,
            x_col="trial_count",
            y_col="lead_sponsor_name",
            color_col="company_type",
            title="Top Sponsors by Trial Count",
            height=600,
            color_map={
                "big_pharma": "#2563eb",
                "biotech": "#059669",
                "academic": "#7c3aed",
                "government": "#d97706",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

with col_term:
    st.markdown("### Termination Analysis")

    term_view = st.selectbox(
        "View by",
        options=["MoA Class", "Indication", "Phase"],
        index=0,
    )

    if term_view == "MoA Class" and not term_moa_df.empty:
        term_data = term_moa_df.copy()
        term_data["label"] = term_data.apply(
            lambda r: f"{r['termination_rate_pct']:.1f}%", axis=1
        )
        fig = horizontal_bar_chart(
            term_data,
            x_col="termination_rate_pct",
            y_col="moa_class",
            title="Termination Rate by MoA Class",
            height=400,
            text_col="label",
        )
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(fig, use_container_width=True)

    elif term_view == "Indication" and not term_ind_df.empty:
        term_data = term_ind_df.copy()
        term_data["label"] = term_data.apply(
            lambda r: f"{r['termination_rate_pct']:.1f}%", axis=1
        )
        fig = horizontal_bar_chart(
            term_data,
            x_col="termination_rate_pct",
            y_col="indication",
            title="Termination Rate by Indication",
            height=500,
            text_col="label",
        )
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(fig, use_container_width=True)

    elif term_view == "Phase" and not term_phase_df.empty:
        term_data = term_phase_df.copy()
        term_data["label"] = term_data.apply(
            lambda r: f"{r['termination_rate_pct']:.1f}%", axis=1
        )
        fig = horizontal_bar_chart(
            term_data,
            x_col="termination_rate_pct",
            y_col="phase",
            title="Termination Rate by Phase",
            height=400,
            text_col="label",
        )
        fig.update_traces(marker_color=PRIMARY)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# Stale Trials Monitor
# ─────────────────────────────────────────────
st.markdown("### ⚠️ Stale Trials Monitor")
st.markdown(
    "Trials with `last_update > 12 months` AND `completion_date` in the past. "
    "These may require follow-up or status clarification."
)

with st.spinner("Loading stale trials..."):
    stale_df = get_stale_trials()

if not stale_df.empty:
    st.markdown(f"**{len(stale_df)} stale trials detected**")

    # Filters for stale trials
    scol1, scol2 = st.columns(2)
    with scol1:
        stale_phase_filter = st.multiselect(
            "Filter by Phase",
            options=stale_df["phase"].dropna().unique().tolist(),
            default=[],
            key="stale_phase",
            placeholder="All Phases",
        )
    with scol2:
        stale_status_filter = st.multiselect(
            "Filter by Status",
            options=stale_df["overall_status"].dropna().unique().tolist(),
            default=[],
            key="stale_status",
            placeholder="All Statuses",
        )

    filtered_stale = stale_df.copy()
    if stale_phase_filter:
        filtered_stale = filtered_stale[filtered_stale["phase"].isin(stale_phase_filter)]
    if stale_status_filter:
        filtered_stale = filtered_stale[filtered_stale["overall_status"].isin(stale_status_filter)]

    # Add ClinicalTrials.gov links
    filtered_stale["CT.gov Link"] = filtered_stale["nct_id"].apply(
        lambda x: f"https://clinicaltrials.gov/study/{x}"
    )

    display_stale = filtered_stale[[
        "nct_id", "title", "phase", "overall_status", "lead_sponsor_name",
        "enrollment", "days_since_update", "indications", "CT.gov Link"
    ]].copy()

    display_stale.columns = [
        "NCT ID", "Title", "Phase", "Status", "Sponsor",
        "Enrollment", "Days Since Update", "Indications", "CT.gov"
    ]

    st.dataframe(
        display_stale,
        use_container_width=True,
        height=400,
        column_config={
            "CT.gov": st.column_config.LinkColumn("CT.gov", display_text="View"),
            "Days Since Update": st.column_config.NumberColumn("Days Stale", format="%d"),
            "Enrollment": st.column_config.NumberColumn("Enrollment", format="%d"),
        },
    )

    # CSV Download
    csv = filtered_stale.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Stale Trials (CSV)",
        csv,
        "stale_trials.csv",
        "text/csv",
        key="stale_csv",
    )
else:
    st.success("No stale trials detected.")
