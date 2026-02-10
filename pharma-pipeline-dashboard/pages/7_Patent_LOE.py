"""
Tab 7: Patent & LOE (Loss of Exclusivity)
Orange Book / Purple Book patent data, LOE timeline, exclusivity calendar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.queries import (
    get_loe_calendar, get_patent_type_distribution, get_exclusivity_distribution,
    get_loe_kpis,
)
from utils.charts import _layout, MOA_COLORS, PRIMARY, SECONDARY, SUCCESS, WARNING, DANGER

st.set_page_config(page_title="Patent & LOE | Pharma Pipeline", page_icon="📋", layout="wide")

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

st.markdown("# 📋 Patent & Loss of Exclusivity")
st.markdown("Orange Book / Purple Book patent landscape, LOE calendar, and exclusivity analysis.")

# KPIs
kpis = get_loe_kpis()
if not kpis.empty:
    k = kpis.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drugs with LOE Data", f"{int(k.get('drugs_with_loe', 0))}/43")
    c2.metric("Future LOE", int(k.get("future_loe", 0)))
    c3.metric("Past LOE (Generics)", int(k.get("past_loe", 0)))
    next_drug = k.get("next_loe_drug", "N/A")
    next_years = k.get("next_loe_years", None)
    c4.metric("Next LOE", f"{next_drug} ({next_years:.1f}y)" if next_years else next_drug)

st.divider()

# Sub-tabs
tab_timeline, tab_calendar, tab_patents, tab_excl = st.tabs([
    "LOE Timeline 📅", "LOE Calendar 📋", "Patent Analysis 🔬", "Exclusivity Codes 🏷️"
])

# -- LOE Timeline (Gantt-style) --
with tab_timeline:
    loe_data = get_loe_calendar()

    if loe_data.empty:
        st.info("No LOE data available.")
    else:
        # Filter
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            moa_opts = sorted(loe_data["moa_class"].dropna().unique())
            sel_moa = st.multiselect("MoA Class", moa_opts, default=[], key="loe_moa")
        with col_f2:
            status_opts = sorted(loe_data["loe_status"].dropna().unique())
            sel_status = st.multiselect("LOE Status", status_opts, default=[], key="loe_status")

        df = loe_data.copy()
        if sel_moa:
            df = df[df["moa_class"].isin(sel_moa)]
        if sel_status:
            df = df[df["loe_status"].isin(sel_status)]

        if not df.empty:
            # Status colors
            status_colors = {
                "Past LOE": "#64748b",
                "Imminent (<2y)": "#ef4444",
                "Medium (2-5y)": "#f59e0b",
                "Protected (>5y)": "#22c55e",
            }

            # Gantt-style horizontal bar from today to LOE date
            today = pd.Timestamp.now()
            df_sorted = df.sort_values("effective_loe_date", ascending=True).copy()
            df_sorted["effective_loe_date"] = pd.to_datetime(df_sorted["effective_loe_date"])
            df_sorted["years_display"] = df_sorted["years_until_loe"].apply(
                lambda x: f"{x:.1f}y" if pd.notna(x) else "N/A"
            )

            fig = go.Figure()

            for _, row in df_sorted.iterrows():
                loe_date = row["effective_loe_date"]
                if pd.isna(loe_date):
                    continue  # skip rows without a valid LOE date
                status = row["loe_status"]
                color = status_colors.get(status, "#94a3b8")
                label = f"{row['inn']} ({row.get('trade_name', '')})"
                years_str = row["years_display"]
                patents = int(row.get("patent_count", 0))

                if status == "Past LOE":
                    bar_start = loe_date
                    bar_end = today
                else:
                    bar_start = today
                    bar_end = loe_date

                fig.add_trace(go.Bar(
                    y=[label], x=[(bar_end - bar_start).days / 365.25],
                    orientation="h", marker_color=color, showlegend=False,
                    text=f"{years_str} | {patents} patents",
                    textposition="outside", textfont_size=9,
                    hovertemplate=(
                        f"<b>{row['inn']}</b><br>"
                        f"Trade Name: {row.get('trade_name', 'N/A')}<br>"
                        f"LOE Date: {loe_date.strftime('%Y-%m-%d')}<br>"
                        f"Status: {status}<br>"
                        f"Patents: {patents}<br>"
                        f"MoA: {row['moa_class']}<extra></extra>"
                    ),
                ))

            # Add legend entries manually
            for status, color in status_colors.items():
                fig.add_trace(go.Bar(
                    y=[None], x=[None], marker_color=color, name=status, showlegend=True,
                ))

            fig.update_layout(
                **_layout(margin=dict(l=250, r=80, t=50, b=40)),
                title="LOE Timeline (Years from Today)",
                height=max(500, len(df_sorted) * 28 + 150),
                xaxis_title="Years",
                yaxis=dict(autorange="reversed", tickfont_size=10),
                barmode="stack",
                legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", font_size=11),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data matching the selected filters.")

# -- LOE Calendar (Table with conditional formatting) --
with tab_calendar:
    loe_data = get_loe_calendar()

    if loe_data.empty:
        st.info("No LOE data available.")
    else:
        st.markdown("### LOE Calendar")
        st.markdown("Color coding: 🔴 Imminent (<2y) | 🟠 Medium (2-5y) | 🟢 Protected (>5y) | ⚪ Past LOE")

        # Build display table
        display = loe_data.copy()
        display["patent_types"] = display.apply(
            lambda r: ", ".join(filter(None, [
                "Substance" if r.get("has_substance_patent") else None,
                "Use" if r.get("has_use_patent") else None,
                "Product" if r.get("has_product_patent") else None,
            ])) or "-",
            axis=1
        )
        display["years_display"] = display["years_until_loe"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

        # Select columns for display
        table = display[[
            "inn", "moa_class", "trade_name", "effective_loe_date", "years_display",
            "loe_status", "patent_count", "patent_types", "exclusivity_codes"
        ]].copy()
        table.columns = [
            "Drug", "MoA Class", "Trade Name", "LOE Date", "Years to LOE",
            "Status", "Patents", "Patent Types", "Exclusivity Codes"
        ]

        # Status-based styling via column_config
        st.dataframe(
            table,
            use_container_width=True,
            height=600,
            column_config={
                "Patents": st.column_config.NumberColumn(format="%d"),
                "LOE Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            },
        )

        # Summary by status
        st.markdown("### Summary by LOE Status")
        summary = display.groupby("loe_status").agg(
            drugs=("inn", "count"),
            avg_patents=("patent_count", "mean"),
        ).reset_index()
        summary["avg_patents"] = summary["avg_patents"].round(1)
        summary.columns = ["LOE Status", "Drug Count", "Avg Patents"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(summary, use_container_width=True)
        with col2:
            # Status pie chart
            status_colors_map = {
                "Past LOE": "#64748b",
                "Imminent (<2y)": "#ef4444",
                "Medium (2-5y)": "#f59e0b",
                "Protected (>5y)": "#22c55e",
            }
            fig = go.Figure(data=[go.Pie(
                labels=summary["LOE Status"],
                values=summary["Drug Count"],
                hole=0.55,
                marker=dict(colors=[status_colors_map.get(s, "#94a3b8") for s in summary["LOE Status"]]),
                textinfo="label+value",
                textposition="outside",
                textfont_size=11,
            )])
            fig.update_layout(**_layout(margin=dict(l=20, r=20, t=40, b=20)),
                              title="Drugs by LOE Status", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # CSV download
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download LOE Calendar (CSV)", csv, "loe_calendar.csv", "text/csv", key="loe_csv")

# -- Patent Analysis --
with tab_patents:
    patent_dist = get_patent_type_distribution()
    loe_data = get_loe_calendar()

    if patent_dist.empty:
        st.info("No patent data available.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Patent type donut
            fig = go.Figure(data=[go.Pie(
                labels=patent_dist["patent_type"],
                values=patent_dist["patent_count"],
                hole=0.55,
                textinfo="label+percent",
                textposition="outside",
                textfont_size=10,
                hovertemplate="<b>%{label}</b><br>Patents: %{value:,}<br>Drugs: %{customdata}<extra></extra>",
                customdata=patent_dist["drug_count"],
            )])
            fig.update_layout(**_layout(margin=dict(l=20, r=20, t=50, b=20)),
                              title="Patent Type Distribution", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Patent count per MoA (from LOE data)
            if not loe_data.empty:
                moa_patents = loe_data.groupby("moa_class").agg(
                    total_patents=("patent_count", "sum"),
                    avg_patents=("patent_count", "mean"),
                    drug_count=("inn", "count"),
                ).sort_values("total_patents", ascending=True).reset_index()

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    y=moa_patents["moa_class"], x=moa_patents["total_patents"],
                    orientation="h", marker_color=PRIMARY,
                    text=moa_patents["total_patents"].apply(lambda x: f"{int(x)}"),
                    textposition="outside", textfont_size=10,
                    hovertemplate="<b>%{y}</b><br>Total Patents: %{x}<br>Drugs: %{customdata}<extra></extra>",
                    customdata=moa_patents["drug_count"],
                ))
                fig2.update_layout(**_layout(), height=400, title="Total Patents by MoA Class",
                                   xaxis_title="Total Patents", yaxis_title="")
                fig2.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                st.plotly_chart(fig2, use_container_width=True)

        # Patent landscape table
        if not loe_data.empty:
            st.markdown("### Patent Landscape per Drug")
            patent_table = loe_data[[
                "inn", "moa_class", "trade_name", "patent_count",
                "has_substance_patent", "has_use_patent", "has_product_patent",
                "latest_patent_expiry", "latest_exclusivity_expiry",
            ]].copy()
            patent_table["has_substance_patent"] = patent_table["has_substance_patent"].astype(bool)
            patent_table["has_use_patent"] = patent_table["has_use_patent"].astype(bool)
            patent_table["has_product_patent"] = patent_table["has_product_patent"].astype(bool)
            patent_table.columns = [
                "Drug", "MoA Class", "Trade Name", "Patents",
                "Substance", "Use", "Product",
                "Latest Patent Expiry", "Latest Exclusivity",
            ]
            patent_table = patent_table.sort_values("Patents", ascending=False)
            st.dataframe(patent_table, use_container_width=True, height=500,
                         column_config={
                             "Patents": st.column_config.NumberColumn(format="%d"),
                             "Substance": st.column_config.CheckboxColumn(),
                             "Use": st.column_config.CheckboxColumn(),
                             "Product": st.column_config.CheckboxColumn(),
                             "Latest Patent Expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
                             "Latest Exclusivity": st.column_config.DateColumn(format="YYYY-MM-DD"),
                         })

# -- Exclusivity Codes --
with tab_excl:
    excl_dist = get_exclusivity_distribution()

    if excl_dist.empty:
        st.info("No exclusivity data available.")
    else:
        # Exclusivity code descriptions
        excl_desc = {
            "NCE": "New Chemical Entity (5 years)",
            "NCE-1": "New Chemical Entity (5y) - First Applicant",
            "ODE": "Orphan Drug Exclusivity (7 years)",
            "BPCIA-12": "Biologics 12-Year Reference Product Exclusivity",
            "BPCIA-4": "Biologics 4-Year Data Exclusivity",
            "NP": "New Product (3 years)",
            "I-459": "Patent Infringement (30-month stay)",
        }

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(excl_dist, x="drug_count", y="exclusivity_code", orientation="h",
                         title="Drugs per Exclusivity Code", text="drug_count",
                         color_discrete_sequence=[PRIMARY])
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_layout(**_layout(), height=400, xaxis_title="Drug Count", yaxis_title="",
                              yaxis=dict(autorange="reversed"))
            fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Exclusivity Code Reference")
            for _, row in excl_dist.iterrows():
                code = row["exclusivity_code"]
                desc = excl_desc.get(code, "FDA Exclusivity Code")
                drugs = row.get("drugs", "")
                st.markdown(
                    f"**`{code}`** - {desc}  \n"
                    f"*Drugs: {drugs}*"
                )
                st.markdown("---")

        # Full table
        st.markdown("### All Exclusivity Records")
        display = excl_dist.copy()
        display.columns = ["Exclusivity Code", "Drug Count", "Drugs"]
        st.dataframe(display, use_container_width=True)
