"""
Tab 5: Market Intelligence
UK and US market data: prescriptions, spending, pricing trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.queries import (
    get_uk_prescriptions_trend, get_uk_latest_month, get_uk_moa_aggregation,
    get_us_spending_trend, get_us_spending_2023_top, get_us_spending_growth,
    get_market_kpis, get_drug_list,
)
from utils.charts import _layout, MOA_COLORS, PRIMARY, SECONDARY

st.set_page_config(page_title="Market Intelligence | Pharma Pipeline", page_icon="📊", layout="wide")

st.markdown("""<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important;}
    h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important;}
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
</style>""", unsafe_allow_html=True)

# Sidebar
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

st.markdown("# 💰 Market Intelligence")
st.markdown("UK prescriptions and US Medicare spending data for diabetes & obesity drugs.")

# KPIs
kpis = get_market_kpis()
if not kpis.empty:
    k = kpis.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    us_total = k.get("us_total_2023", 0)
    c1.metric("US Total Spending (2023)", f"${us_total/1e9:.1f}B" if us_total else "N/A")
    c2.metric("UK Drugs Tracked", f"{int(k.get('uk_drug_count', 0))}/43")
    c3.metric("US Drugs Tracked", f"{int(k.get('us_drug_count', 0))}/43")
    uk_top = k.get("uk_top_items", 0)
    c4.metric("UK Top Rx Items/Mo", f"{int(uk_top):,}" if uk_top else "N/A")

st.divider()

# Sub-tabs
tab_uk, tab_us, tab_compare = st.tabs(["UK Market 🇬🇧", "US Market 🇺🇸", "Cross-Market 🌍"])

# ── UK Market ──
with tab_uk:
    uk_trend = get_uk_prescriptions_trend()
    uk_latest = get_uk_latest_month()
    uk_moa = get_uk_moa_aggregation()

    if uk_trend.empty:
        st.info("No UK prescription data available.")
    else:
        # Filters
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        moa_options = sorted(uk_trend["moa_class"].dropna().unique())
        with col_f1:
            sel_moa_uk = st.multiselect("MoA Class", moa_options, default=[], key="uk_moa")
        with col_f2:
            drug_options = sorted(uk_trend["inn"].unique())
            sel_drugs_uk = st.multiselect("Drugs", drug_options, default=[], key="uk_drugs")
        with col_f3:
            metric_uk = st.radio("Metric", ["Items", "Cost (GBP)", "Cost/Unit"], key="uk_metric", horizontal=True)

        # Filter data
        df = uk_trend.copy()
        if sel_moa_uk:
            df = df[df["moa_class"].isin(sel_moa_uk)]
        if sel_drugs_uk:
            df = df[df["inn"].isin(sel_drugs_uk)]

        y_col = {"Items": "items", "Cost (GBP)": "actual_cost", "Cost/Unit": "cost_per_unit_gbp"}[metric_uk]
        y_label = {"Items": "Prescription Items", "Cost (GBP)": "Cost (GBP)", "Cost/Unit": "Cost per Unit (GBP)"}[metric_uk]

        # Trend chart
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(df, x="date", y=y_col, color="inn", title=f"UK {y_label} Trend",
                              color_discrete_map=MOA_COLORS, markers=False)
                fig.update_layout(**_layout(), height=450, xaxis_title="", yaxis_title=y_label,
                                  legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font_size=10))
                fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # MoA class comparison bar
                if not uk_moa.empty:
                    fig2 = px.bar(uk_moa, x="total_items", y="moa_class", orientation="h",
                                  title="UK Prescriptions by MoA (Latest Month)", text="total_items",
                                  color="total_cost", color_continuous_scale="Blues")
                    fig2.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_size=10)
                    fig2.update_layout(**_layout(), height=450, xaxis_title="Items", yaxis_title="",
                                       yaxis=dict(autorange="reversed"), coloraxis_colorbar=dict(title="Cost (GBP)"))
                    fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig2, use_container_width=True)

        # Latest month table
        if not uk_latest.empty:
            st.markdown("### Latest Month Data")
            display = uk_latest.copy()
            display.columns = ["Drug", "MoA Class", "Items", "Cost (GBP)", "Quantity", "Cost/Unit (GBP)", "Date"]
            st.dataframe(display, use_container_width=True, height=400,
                         column_config={"Items": st.column_config.NumberColumn(format="%d"),
                                        "Cost (GBP)": st.column_config.NumberColumn(format="£%.2f"),
                                        "Cost/Unit (GBP)": st.column_config.NumberColumn(format="£%.4f")})
            csv = uk_latest.to_csv(index=False).encode("utf-8")
            st.download_button("Download UK Data (CSV)", csv, "uk_prescriptions.csv", "text/csv", key="uk_csv")

# ── US Market ──
with tab_us:
    us_trend = get_us_spending_trend()
    us_top = get_us_spending_2023_top()
    us_growth_raw = get_us_spending_growth()

    if us_trend.empty:
        st.info("No US Medicare spending data available.")
    else:
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            sel_drugs_us = st.multiselect("Drugs", sorted(us_trend["inn"].unique()), default=[], key="us_drugs")
        with col_f2:
            log_scale = st.toggle("Log Scale", value=False, key="us_log")

        df_us = us_trend.copy()
        if sel_drugs_us:
            df_us = df_us[df_us["inn"].isin(sel_drugs_us)]

        col1, col2 = st.columns(2)
        with col1:
            # Aggregate by inn+year for trend
            agg = df_us.groupby(["inn", "moa_class", "year"]).agg(
                total_spending=("total_spending", "sum"),
                total_claims=("total_claims", "sum")).reset_index()
            fig = px.line(agg, x="year", y="total_spending", color="inn",
                          title="US Medicare Spending Trend (2019-2023)", markers=True)
            if log_scale:
                fig.update_yaxes(type="log")
            fig.update_layout(**_layout(), height=450, xaxis_title="Year", yaxis_title="Total Spending ($)",
                              legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font_size=10))
            fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", dtick=1)
            fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Growth bar
            if not us_growth_raw.empty:
                gdf = us_growth_raw.copy()
                gdf["growth_pct"] = ((gdf["spend_2023"] - gdf["spend_2019"]) / gdf["spend_2019"].replace(0, float("nan"))) * 100
                gdf = gdf.dropna(subset=["growth_pct"]).sort_values("growth_pct", ascending=True)
                gdf["color"] = gdf["growth_pct"].apply(lambda x: "#2E7D32" if x > 0 else "#C62828")
                fig2 = go.Figure(go.Bar(y=gdf["inn"], x=gdf["growth_pct"], orientation="h",
                                        marker_color=gdf["color"],
                                        text=gdf["growth_pct"].apply(lambda x: f"{x:+.0f}%"),
                                        textposition="outside", textfont_size=10))
                fig2.update_layout(**_layout(), height=450, title="US Spending Growth 2019-2023 (%)",
                                   xaxis_title="Growth (%)", yaxis_title="")
                fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig2, use_container_width=True)

        # Top 2023 table
        if not us_top.empty:
            st.markdown("### Top Drugs by 2023 Spending")
            display = us_top.head(20).copy()
            display.columns = ["Drug", "MoA", "Brand", "Spending ($)", "Claims", "Beneficiaries", "Cost/Unit ($)"]
            st.dataframe(display, use_container_width=True, height=400,
                         column_config={"Spending ($)": st.column_config.NumberColumn(format="$%.2f"),
                                        "Claims": st.column_config.NumberColumn(format="%d"),
                                        "Cost/Unit ($)": st.column_config.NumberColumn(format="$%.4f")})
            csv = us_top.to_csv(index=False).encode("utf-8")
            st.download_button("Download US Data (CSV)", csv, "us_spending.csv", "text/csv", key="us_csv")

# ── Cross-Market ──
with tab_compare:
    drug_list = get_drug_list()
    if drug_list.empty:
        st.info("No drug data available.")
    else:
        sel_drug = st.selectbox("Select Drug for Comparison", drug_list["drug_name"].tolist(), key="cross_drug")
        sel_id = drug_list[drug_list["drug_name"] == sel_drug]["drug_id"].iloc[0]

        from utils.queries import get_drug_uk_trend, get_drug_us_spending
        uk_d = get_drug_uk_trend(sel_id)
        us_d = get_drug_us_spending(sel_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### UK Prescriptions")
            if not uk_d.empty:
                fig = px.line(uk_d, x="date", y="items", title=f"{sel_drug} - UK Items/Month", markers=True,
                              color_discrete_sequence=["#1B5E20"])
                fig.update_layout(**_layout(), height=350, xaxis_title="", yaxis_title="Items")
                fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No UK data for {sel_drug}.")

        with col2:
            st.markdown("### US Medicare Spending")
            if not us_d.empty:
                agg = us_d.groupby("year").agg(total_spending=("total_spending", "sum")).reset_index()
                fig = px.bar(agg, x="year", y="total_spending", title=f"{sel_drug} - US Spending/Year",
                             text="total_spending", color_discrete_sequence=["#1565C0"])
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont_size=10)
                fig.update_layout(**_layout(), height=350, xaxis_title="Year", yaxis_title="Spending ($)")
                fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", dtick=1)
                fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No US data for {sel_drug}.")
