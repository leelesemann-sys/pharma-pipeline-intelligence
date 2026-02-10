"""
Tab 2: Competitive Landscape
THE core feature - MoA x Indication heatmap, trend analysis, hot & cold zones.
"""

import streamlit as st
import pandas as pd
from utils.queries import get_heatmap_data, get_trial_starts_trend, get_hot_cold_zones
from utils.charts import heatmap_chart, trend_line_chart, LAYOUT_DEFAULTS, PRIMARY, DANGER, SUCCESS

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Competitive Landscape | Pharma Pipeline Intelligence",
    page_icon="🗺️",
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
    .hot-zone {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1px solid #fca5a5; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    }
    .cold-zone {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    }
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
st.markdown("# 🗺️ Competitive Landscape")
st.markdown("Identify crowded and white-space areas across MoA classes and indications.")
st.markdown("")

# ─────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
    metric_choice = st.selectbox(
        "Heatmap Metric",
        options=["trial_count", "active_trial_count", "drug_count", "recent_trial_starts"],
        format_func=lambda x: {
            "trial_count": "Total Trials",
            "active_trial_count": "Active Trials",
            "drug_count": "Drug Count",
            "recent_trial_starts": "Recent Trial Starts (2y)",
        }[x],
        index=0,
    )

with ctrl2:
    phase_filter = st.multiselect(
        "Phase Filter",
        options=["Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4"],
        default=[],
        placeholder="All Phases",
    )

with ctrl3:
    show_active_only = st.checkbox("Active Trials Only", value=False)

st.divider()

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
with st.spinner("Loading competitive landscape data..."):
    heatmap_df = get_heatmap_data(phase_filter=phase_filter if phase_filter else None)
    trend_df = get_trial_starts_trend()
    hot_cold_df = get_hot_cold_zones()

# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────
st.markdown("### MoA Class x Indication Heatmap")

if not heatmap_df.empty:
    metric_label = {
        "trial_count": "Total Trials",
        "active_trial_count": "Active Trials",
        "drug_count": "Drug Count",
        "recent_trial_starts": "Recent Trial Starts (2y)",
    }[metric_choice]

    fig = heatmap_chart(
        heatmap_df,
        x_col="indication",
        y_col="moa_class",
        z_col=metric_choice,
        title=f"Competitive Density: {metric_label} by MoA x Indication",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No heatmap data available for the selected filters.")

st.divider()

# ─────────────────────────────────────────────
# Trend Chart
# ─────────────────────────────────────────────
st.markdown("### Trial Starts per Year by MoA Class")

if not trend_df.empty:
    fig = trend_line_chart(
        trend_df,
        x_col="start_year",
        y_col="trial_starts",
        color_col="moa_class",
        title="Interventional Trial Starts by MoA Class (2010-2026)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trend data available.")

st.divider()

# ─────────────────────────────────────────────
# Hot & Cold Zones
# ─────────────────────────────────────────────
st.markdown("### Hot & Cold Zones")
st.markdown("Comparing trial activity in the last 2 years vs. the 2 years before.")

col_hot, col_cold = st.columns(2)

with col_hot:
    st.markdown("#### 🔥 Hot Zones")
    st.markdown("*Strongest growth in trial starts*")

    if not hot_cold_df.empty:
        hot = hot_cold_df.dropna(subset=["growth_ratio"]).nlargest(5, "growth_ratio")
        for _, row in hot.iterrows():
            ratio_str = f"{row['growth_ratio']:.1f}x" if pd.notna(row['growth_ratio']) else "New"
            st.markdown(
                f'<div class="hot-zone">'
                f'<strong>{row["moa_class"]}</strong> x <strong>{row["indication"]}</strong><br>'
                f'<span style="color: #dc2626; font-weight: 600;">{ratio_str} growth</span> | '
                f'Recent: {int(row["recent_2y"])} trials | Prior: {int(row["prior_2y"])} trials'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No hot zone data available.")

with col_cold:
    st.markdown("#### ❄️ Cold Zones")
    st.markdown("*Declining or stagnant areas*")

    if not hot_cold_df.empty:
        cold = hot_cold_df.dropna(subset=["growth_ratio"]).nsmallest(5, "growth_ratio")
        for _, row in cold.iterrows():
            ratio_str = f"{row['growth_ratio']:.1f}x"
            recent = int(row["recent_2y"])
            prior = int(row["prior_2y"])
            change = recent - prior
            st.markdown(
                f'<div class="cold-zone">'
                f'<strong>{row["moa_class"]}</strong> x <strong>{row["indication"]}</strong><br>'
                f'<span style="color: #2563eb; font-weight: 600;">{ratio_str} ratio</span> | '
                f'Recent: {recent} trials | Prior: {prior} trials ({change:+d})'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No cold zone data available.")
