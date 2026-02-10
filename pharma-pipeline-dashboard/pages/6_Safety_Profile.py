"""
Tab 6: Safety Profile
FDA FAERS adverse-event data: heatmap, temporal trends, class signals, drug comparator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.queries import (
    get_ae_heatmap_data, get_ae_class_signals, get_ae_trends,
    get_ae_trends_by_moa, get_safety_kpis, get_drug_list,
    get_drug_ae_data, get_drug_ae_trend,
)
from utils.charts import _layout, MOA_COLORS, PRIMARY, SECONDARY, HEATMAP_COLORS, DANGER

st.set_page_config(page_title="Safety Profile | Pharma Pipeline", page_icon="🛡️", layout="wide")

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

st.markdown("# 🛡️ Safety Profile")
st.markdown("FDA FAERS adverse-event analysis: heatmaps, trends, class-level signals, and drug comparisons.")

# KPIs
kpis = get_safety_kpis()
if not kpis.empty:
    k = kpis.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Drugs with FAERS Data", f"{int(k.get('drugs_with_faers', 0))}/43")
    top_drug = k.get("top_drug_by_reports", "N/A")
    c2.metric("Most Reported Drug", top_drug)
    top_total = k.get("top_drug_total", 0)
    c3.metric("Top Drug Total Reports", f"{int(top_total):,}" if top_total else "N/A")

st.divider()

# Sub-tabs
tab_heatmap, tab_trends, tab_class, tab_compare = st.tabs([
    "AE Heatmap 🔥", "Temporal Trends 📈", "Class Signals 🏷️", "Drug Comparator ⚖️"
])

# -- AE Heatmap --
with tab_heatmap:
    ae_data = get_ae_heatmap_data()

    if ae_data.empty:
        st.info("No adverse event data available.")
    else:
        # Filters
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        moa_options = sorted(ae_data["moa_class"].dropna().unique())
        with col_f1:
            sel_moa = st.multiselect("MoA Class", moa_options, default=[], key="ae_moa")
        with col_f2:
            top_n = st.slider("Top N AEs per Drug", 5, 30, 10, key="ae_topn")
        with col_f3:
            color_by = st.radio("Color by", ["Total Count", "Serious Ratio"], key="ae_color", horizontal=True)

        # Filter and limit AEs
        df = ae_data.copy()
        if sel_moa:
            df = df[df["moa_class"].isin(sel_moa)]
        df = df[df["rank_in_drug"] <= top_n]

        if not df.empty:
            # Build heatmap pivot
            z_col = "total_count" if color_by == "Total Count" else "serious_ratio"
            pivot = df.pivot_table(index="inn", columns="event_term", values=z_col, fill_value=0)

            if color_by == "Total Count":
                colorscale = [
                    [0.0, "#f0f9ff"], [0.15, "#bae6fd"], [0.35, "#38bdf8"],
                    [0.55, "#2563eb"], [0.75, "#1e40af"], [0.9, "#7c3aed"], [1.0, "#dc2626"],
                ]
                colorbar_title = "Reports"
            else:
                colorscale = [
                    [0.0, "#f0fdf4"], [0.2, "#86efac"], [0.4, "#fbbf24"],
                    [0.6, "#f97316"], [0.8, "#ef4444"], [1.0, "#991b1b"],
                ]
                colorbar_title = "Serious %"

            # Custom hover text
            hover_df = df.pivot_table(index="inn", columns="event_term",
                                       values=["total_count", "serious_count"], fill_value=0)

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=colorscale,
                hovertemplate="<b>%{y}</b> x <b>%{x}</b><br>Value: %{z:.2f}<extra></extra>",
                showscale=True,
                colorbar=dict(title=colorbar_title, thickness=15),
            ))
            fig.update_layout(
                **_layout(margin=dict(l=150, r=40, t=50, b=200)),
                title="Drug x Adverse Event Heatmap (FAERS)",
                height=max(500, len(pivot.index) * 30 + 250),
                xaxis=dict(tickangle=-45, tickfont_size=10, side="bottom"),
                yaxis=dict(tickfont_size=11, autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.markdown("### Top AEs by Total Reports")
            summary = df.groupby("event_term").agg(
                total=("total_count", "sum"),
                serious=("serious_count", "sum"),
                drugs=("inn", "nunique")
            ).sort_values("total", ascending=False).head(20).reset_index()
            summary["serious_pct"] = (summary["serious"] / summary["total"].replace(0, np.nan) * 100).round(1)
            summary.columns = ["Adverse Event", "Total Reports", "Serious Reports", "Drugs Reporting", "Serious %"]
            st.dataframe(summary, use_container_width=True, height=400,
                         column_config={
                             "Total Reports": st.column_config.NumberColumn(format="%d"),
                             "Serious Reports": st.column_config.NumberColumn(format="%d"),
                             "Serious %": st.column_config.NumberColumn(format="%.1f%%"),
                         })

# -- Temporal Trends --
with tab_trends:
    ae_trends = get_ae_trends()
    ae_trends_moa = get_ae_trends_by_moa()

    if ae_trends.empty:
        st.info("No temporal trend data available.")
    else:
        # View selector
        view = st.radio("View", ["By Drug", "By MoA Class"], key="trend_view", horizontal=True)

        if view == "By Drug":
            col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
            with col_f1:
                drug_opts = sorted(ae_trends["inn"].unique())
                sel_drugs_trend = st.multiselect("Drugs", drug_opts, default=[], key="trend_drugs")
            with col_f2:
                moa_opts = sorted(ae_trends["moa_class"].dropna().unique())
                sel_moa_trend = st.multiselect("MoA Class", moa_opts, default=[], key="trend_moa")
            with col_f3:
                metric = st.radio("Metric", ["Total", "Serious", "Serious %"], key="trend_metric", horizontal=True)

            df = ae_trends.copy()
            if sel_drugs_trend:
                df = df[df["inn"].isin(sel_drugs_trend)]
            if sel_moa_trend:
                df = df[df["moa_class"].isin(sel_moa_trend)]

            if not df.empty:
                y_map = {"Total": "total_reports", "Serious": "serious_reports", "Serious %": "serious_ratio"}
                y_col = y_map[metric]
                y_label = {"Total": "Total Reports", "Serious": "Serious Reports", "Serious %": "Serious Ratio"}[metric]

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(df, x="quarter_date", y=y_col, color="inn",
                                  title=f"FAERS {y_label} Trend by Drug",
                                  color_discrete_map=MOA_COLORS, markers=True)
                    fig.update_layout(**_layout(), height=450, xaxis_title="Quarter",
                                      yaxis_title=y_label,
                                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font_size=10))
                    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Stacked area total reports
                    fig2 = px.area(df, x="quarter_date", y="total_reports", color="inn",
                                   title="Cumulative FAERS Reports by Drug",
                                   color_discrete_map=MOA_COLORS)
                    fig2.update_layout(**_layout(), height=450, xaxis_title="Quarter",
                                       yaxis_title="Total Reports",
                                       legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font_size=10))
                    fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig2.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data matching the selected filters.")

        else:  # By MoA Class
            if not ae_trends_moa.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(ae_trends_moa, x="quarter_date", y="class_total_reports",
                                  color="moa_class", title="FAERS Reports by MoA Class",
                                  color_discrete_map=MOA_COLORS, markers=True)
                    fig.update_layout(**_layout(), height=450, xaxis_title="Quarter",
                                      yaxis_title="Total Reports",
                                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font_size=10))
                    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.line(ae_trends_moa, x="quarter_date", y="class_serious_reports",
                                   color="moa_class", title="Serious Reports by MoA Class",
                                   color_discrete_map=MOA_COLORS, markers=True)
                    fig2.update_layout(**_layout(), height=450, xaxis_title="Quarter",
                                       yaxis_title="Serious Reports",
                                       legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font_size=10))
                    fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig2.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No MoA trend data available.")

# -- Class Signals --
with tab_class:
    class_signals = get_ae_class_signals()

    if class_signals.empty:
        st.info("No class-level signal data available.")
    else:
        moa_opts = sorted(class_signals["moa_class"].dropna().unique())
        sel_moa_class = st.selectbox("MoA Class", moa_opts, key="class_moa")

        df = class_signals[class_signals["moa_class"] == sel_moa_class].copy()
        df["serious_pct"] = (df["class_serious"] / df["class_total"].replace(0, np.nan) * 100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            top_aes = df.sort_values("class_total", ascending=True).tail(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_aes["event_term"], x=top_aes["class_total"], orientation="h",
                name="Total", marker_color=PRIMARY,
                text=top_aes["class_total"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside", textfont_size=10,
            ))
            fig.add_trace(go.Bar(
                y=top_aes["event_term"], x=top_aes["class_serious"], orientation="h",
                name="Serious", marker_color=DANGER,
                text=top_aes["class_serious"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside", textfont_size=10,
            ))
            fig.update_layout(**_layout(), height=500, barmode="overlay",
                              title=f"Top AEs: {sel_moa_class}", xaxis_title="Reports",
                              legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
            fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Serious ratio bubble chart
            top20 = df.sort_values("class_total", ascending=False).head(20)
            fig2 = px.scatter(top20, x="class_total", y="serious_pct", size="drugs_reporting",
                              text="event_term", title=f"AE Profile: {sel_moa_class}",
                              size_max=40, color="serious_pct",
                              color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"])
            fig2.update_traces(textposition="top center", textfont_size=9)
            fig2.update_layout(**_layout(), height=500,
                               xaxis_title="Total Reports (class-wide)",
                               yaxis_title="Serious %",
                               coloraxis_colorbar=dict(title="Serious %"))
            fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
            fig2.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.markdown(f"### All Class-Level AEs: {sel_moa_class}")
        display = df.sort_values("class_total", ascending=False).copy()
        display.columns = ["MoA Class", "Adverse Event", "Total Reports", "Serious Reports",
                           "Drugs Reporting", "Serious %"]
        st.dataframe(display, use_container_width=True, height=400,
                     column_config={
                         "Total Reports": st.column_config.NumberColumn(format="%d"),
                         "Serious Reports": st.column_config.NumberColumn(format="%d"),
                         "Serious %": st.column_config.NumberColumn(format="%.1f%%"),
                     })

# -- Drug Comparator --
with tab_compare:
    drug_list = get_drug_list()
    if drug_list.empty:
        st.info("No drug data available.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            drug_a_name = st.selectbox("Drug A", drug_list["drug_name"].tolist(), key="cmp_a")
        with col_s2:
            drug_b_name = st.selectbox("Drug B", drug_list["drug_name"].tolist(), index=min(1, len(drug_list)-1), key="cmp_b")

        drug_a_id = drug_list[drug_list["drug_name"] == drug_a_name]["drug_id"].iloc[0]
        drug_b_id = drug_list[drug_list["drug_name"] == drug_b_name]["drug_id"].iloc[0]

        ae_a = get_drug_ae_data(drug_a_id)
        ae_b = get_drug_ae_data(drug_b_id)

        if ae_a.empty and ae_b.empty:
            st.info("No FAERS data available for either drug.")
        else:
            # Butterfly chart: top shared AEs
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {drug_a_name} vs {drug_b_name}")

                # Merge on event_term for butterfly
                a = ae_a.head(20).copy().rename(columns={"total_count": "count_a", "serious_ratio": "sr_a"})
                b = ae_b.head(20).copy().rename(columns={"total_count": "count_b", "serious_ratio": "sr_b"})

                merged = pd.merge(a[["event_term", "count_a"]], b[["event_term", "count_b"]],
                                  on="event_term", how="outer").fillna(0)
                merged = merged.sort_values("count_a", ascending=True).tail(15)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=merged["event_term"], x=-merged["count_a"], orientation="h",
                    name=drug_a_name, marker_color=PRIMARY,
                    text=merged["count_a"].apply(lambda x: f"{int(x):,}"),
                    textposition="outside", textfont_size=9,
                ))
                fig.add_trace(go.Bar(
                    y=merged["event_term"], x=merged["count_b"], orientation="h",
                    name=drug_b_name, marker_color=DANGER,
                    text=merged["count_b"].apply(lambda x: f"{int(x):,}"),
                    textposition="outside", textfont_size=9,
                ))
                fig.update_layout(**_layout(), height=500, barmode="overlay",
                                  title="Butterfly Comparison (Top AEs)",
                                  xaxis_title="Reports", yaxis_title="",
                                  legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
                max_val = max(merged["count_a"].max(), merged["count_b"].max()) * 1.3
                fig.update_xaxes(range=[-max_val, max_val], showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Temporal Comparison")
                trend_a = get_drug_ae_trend(drug_a_id)
                trend_b = get_drug_ae_trend(drug_b_id)

                if not trend_a.empty or not trend_b.empty:
                    fig2 = go.Figure()
                    if not trend_a.empty:
                        fig2.add_trace(go.Scatter(
                            x=trend_a["quarter_date"], y=trend_a["total_reports"],
                            mode="lines+markers", name=drug_a_name,
                            line=dict(color=PRIMARY, width=2.5), marker=dict(size=6),
                        ))
                    if not trend_b.empty:
                        fig2.add_trace(go.Scatter(
                            x=trend_b["quarter_date"], y=trend_b["total_reports"],
                            mode="lines+markers", name=drug_b_name,
                            line=dict(color=DANGER, width=2.5), marker=dict(size=6),
                        ))
                    fig2.update_layout(**_layout(), height=500, title="FAERS Reports Over Time",
                                       xaxis_title="Quarter", yaxis_title="Total Reports",
                                       legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
                    fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig2.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No temporal trend data for these drugs.")

            # Side-by-side tables
            st.markdown("### Detailed AE Comparison")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"**{drug_a_name}**")
                if not ae_a.empty:
                    disp_a = ae_a.head(20).copy()
                    disp_a["serious_ratio"] = (disp_a["serious_ratio"] * 100).round(1)
                    disp_a.columns = ["AE", "Total", "Serious", "Non-Serious", "Serious %"]
                    st.dataframe(disp_a, use_container_width=True, height=400,
                                 column_config={
                                     "Total": st.column_config.NumberColumn(format="%d"),
                                     "Serious": st.column_config.NumberColumn(format="%d"),
                                     "Serious %": st.column_config.NumberColumn(format="%.1f%%"),
                                 })
                else:
                    st.info(f"No FAERS data for {drug_a_name}.")
            with col_t2:
                st.markdown(f"**{drug_b_name}**")
                if not ae_b.empty:
                    disp_b = ae_b.head(20).copy()
                    disp_b["serious_ratio"] = (disp_b["serious_ratio"] * 100).round(1)
                    disp_b.columns = ["AE", "Total", "Serious", "Non-Serious", "Serious %"]
                    st.dataframe(disp_b, use_container_width=True, height=400,
                                 column_config={
                                     "Total": st.column_config.NumberColumn(format="%d"),
                                     "Serious": st.column_config.NumberColumn(format="%d"),
                                     "Serious %": st.column_config.NumberColumn(format="%.1f%%"),
                                 })
                else:
                    st.info(f"No FAERS data for {drug_b_name}.")
