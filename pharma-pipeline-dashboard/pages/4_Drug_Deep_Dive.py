"""
Tab 4: Drug Deep Dive
Single-drug view with all available data: indications, trials, FDA approvals, competitive position.
"""

import streamlit as st
import pandas as pd
from utils.queries import (
    get_drug_list,
    get_drug_detail,
    get_drug_indications,
    get_drug_trials,
    get_drug_approvals,
    get_competitive_position,
    get_drug_uk_trend,
    get_drug_us_spending,
    get_drug_ae_data,
    get_drug_ae_trend,
    get_drug_loe_summary,
    get_drug_patents,
)
from utils.charts import (
    trial_timeline_scatter,
    competitive_comparison_bar,
    STATUS_COLORS,
    PRIMARY,
    SECONDARY,
    DANGER,
    SUCCESS,
    WARNING,
    _layout,
)
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Deep Dive | Pharma Pipeline Intelligence",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important;}
    h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important;}
    section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    .drug-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .drug-card h2 { margin-bottom: 12px !important; }
    .drug-card .detail-row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 10px;
    }
    .drug-card .detail-item {
        display: flex;
        flex-direction: column;
    }
    .drug-card .detail-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .drug-card .detail-value {
        font-size: 1rem;
        font-weight: 500;
        color: #0f172a;
        margin-top: 2px;
    }
    .indication-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .status-dot-approved {
        width: 10px; height: 10px; border-radius: 50%;
        background-color: #22c55e; display: inline-block;
    }
    .status-dot-investigational {
        width: 10px; height: 10px; border-radius: 50%;
        border: 2px solid #2563eb; display: inline-block;
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
# Header & Drug Selector
# ─────────────────────────────────────────────
st.markdown("# 💊 Drug Deep Dive")
st.markdown("In-depth view of a single drug: development status, clinical trials, approvals, and competitive position.")
st.markdown("")

# Load drug list
drug_list = get_drug_list()

if drug_list.empty:
    st.warning("No drugs found in the database.")
    st.stop()

# Drug selector
drug_options = drug_list.apply(
    lambda r: f"{r['drug_name']} ({r['moa_class']})" if pd.notna(r['moa_class']) else r['drug_name'],
    axis=1,
).tolist()

drug_ids = drug_list["drug_id"].tolist()

selected_idx = st.selectbox(
    "Select Drug",
    options=range(len(drug_options)),
    format_func=lambda i: drug_options[i],
    index=0,
)

selected_drug_id = drug_ids[selected_idx]

st.divider()

# ─────────────────────────────────────────────
# Load Drug Data
# ─────────────────────────────────────────────
with st.spinner("Loading drug details..."):
    detail_df = get_drug_detail(selected_drug_id)
    indications_df = get_drug_indications(selected_drug_id)
    trials_df = get_drug_trials(selected_drug_id)
    approvals_df = get_drug_approvals(selected_drug_id)
    competitive_df = get_competitive_position(selected_drug_id)

# ─────────────────────────────────────────────
# Drug Card
# ─────────────────────────────────────────────
drug_name = "Unknown"  # default in case detail query fails
if not detail_df.empty:
    drug = detail_df.iloc[0]
    drug_name = drug.get("inn", "Unknown")
    moa = drug.get("moa_class", "N/A")
    modality = drug.get("modality", "N/A")
    originator = drug.get("originator_company", "N/A")
    highest_phase = drug.get("highest_phase", "N/A")
    chembl_id = drug.get("chembl_id", "N/A")
    atc_code = drug.get("atc_code", "N/A")

    # Get approved/investigational indications
    approved_inds = indications_df[indications_df["status"] == "approved"]["name"].tolist() if not indications_df.empty else []
    invest_inds = indications_df[indications_df["status"] != "approved"]["name"].tolist() if not indications_df.empty else []

    approved_str = ", ".join(approved_inds) if approved_inds else "None"
    invest_str = ", ".join(invest_inds) if invest_inds else "None"

    st.markdown(f"""
    <div class="drug-card">
        <h2 style="color: #0f172a !important; font-size: 1.5rem !important;">{drug_name}</h2>
        <div class="detail-row">
            <div class="detail-item">
                <span class="detail-label">MoA Class</span>
                <span class="detail-value">{moa}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Modality</span>
                <span class="detail-value">{modality}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Originator</span>
                <span class="detail-value">{originator}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Highest Phase</span>
                <span class="detail-value">{highest_phase}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">ChEMBL ID</span>
                <span class="detail-value">{chembl_id}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">ATC Code</span>
                <span class="detail-value">{atc_code}</span>
            </div>
        </div>
        <div class="detail-row" style="margin-top: 16px;">
            <div class="detail-item">
                <span class="detail-label">Approved Indications</span>
                <span class="detail-value" style="color: #22c55e;">{approved_str}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Investigational</span>
                <span class="detail-value" style="color: #2563eb;">{invest_str}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Indication Status & Trial Timeline
# ─────────────────────────────────────────────
col_ind, col_timeline = st.columns([1, 2])

with col_ind:
    st.markdown("### Indication Status")
    if not indications_df.empty:
        for _, ind_row in indications_df.iterrows():
            status = ind_row["status"]
            phase = ind_row.get("phase", "")
            name = ind_row["name"]
            if status == "approved":
                dot_class = "status-dot-approved"
                label = "Approved"
            else:
                dot_class = "status-dot-investigational"
                label = f"Phase {phase}" if pd.notna(phase) and phase else "Investigational"
            st.markdown(
                f'<div class="indication-status">'
                f'<span class="{dot_class}"></span>'
                f'<span style="font-weight: 500;">{name}</span>'
                f'<span style="color: #64748b; font-size: 0.85rem; margin-left: auto;">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No indication data available.")

with col_timeline:
    st.markdown("### Trial Timeline")
    if not trials_df.empty and "start_date" in trials_df.columns:
        timeline_data = trials_df.dropna(subset=["start_date"]).copy()
        if not timeline_data.empty:
            fig = trial_timeline_scatter(
                timeline_data,
                title=f"Clinical Trial Timeline: {drug_name}",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trial start dates available for timeline.")
    else:
        st.info("No trial data available.")

st.divider()

# ─────────────────────────────────────────────
# Trial Table
# ─────────────────────────────────────────────
st.markdown("### Clinical Trials")

if not trials_df.empty:
    st.markdown(f"**{len(trials_df)} trials linked to {drug_name}**")

    # Add CT.gov links
    trials_display = trials_df.copy()
    trials_display["CT.gov"] = trials_display["nct_id"].apply(
        lambda x: f"https://clinicaltrials.gov/study/{x}"
    )

    display_cols = [
        "nct_id", "title", "phase", "overall_status",
        "lead_sponsor_name", "enrollment", "start_date",
        "completion_date", "has_results", "indications", "CT.gov"
    ]
    available_cols = [c for c in display_cols if c in trials_display.columns]
    trials_show = trials_display[available_cols].copy()

    # Rename for display
    col_rename = {
        "nct_id": "NCT ID",
        "title": "Title",
        "phase": "Phase",
        "overall_status": "Status",
        "lead_sponsor_name": "Sponsor",
        "enrollment": "Enrollment",
        "start_date": "Start Date",
        "completion_date": "Completion Date",
        "has_results": "Has Results",
        "indications": "Indications",
        "CT.gov": "CT.gov",
    }
    trials_show = trials_show.rename(columns=col_rename)

    st.dataframe(
        trials_show,
        use_container_width=True,
        height=400,
        column_config={
            "CT.gov": st.column_config.LinkColumn("CT.gov", display_text="View"),
            "Enrollment": st.column_config.NumberColumn("Enrollment", format="%d"),
            "Has Results": st.column_config.CheckboxColumn("Results"),
        },
    )

    # CSV Download
    csv = trials_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {drug_name} Trials (CSV)",
        csv,
        f"{drug_name.lower().replace(' ', '_')}_trials.csv",
        "text/csv",
        key="drug_trials_csv",
    )
else:
    st.info(f"No clinical trials linked to {drug_name}.")

st.divider()

# ─────────────────────────────────────────────
# FDA Approvals
# ─────────────────────────────────────────────
st.markdown("### FDA Approvals")

if not approvals_df.empty:
    st.markdown(f"**{len(approvals_df)} FDA approval records**")

    approval_display = approvals_df.copy()
    # Select relevant columns if they exist
    display_cols = ["application_number", "approval_date", "submission_type",
                    "brand_name", "dosage_form", "route", "sponsor_name"]
    available_cols = [c for c in display_cols if c in approval_display.columns]

    if available_cols:
        st.dataframe(
            approval_display[available_cols],
            use_container_width=True,
            height=300,
        )
    else:
        st.dataframe(approval_display, use_container_width=True, height=300)
else:
    st.info(f"No FDA approval records for {drug_name}.")

st.divider()

# ─────────────────────────────────────────────
# Competitive Position
# ─────────────────────────────────────────────
st.markdown("### Competitive Position")
st.markdown(f"*Other drugs in the same MoA class ({moa})*")

if not competitive_df.empty:
    col_table, col_chart = st.columns([1, 1])

    with col_table:
        comp_display = competitive_df.copy()
        comp_display.columns = ["Drug", "Highest Phase", "Total Trials", "Active Trials"]
        st.dataframe(comp_display, use_container_width=True, height=350)

    with col_chart:
        fig = competitive_comparison_bar(
            competitive_df,
            drug_name=drug_name,
            title=f"Trial Activity: {moa} Drugs",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No other drugs found in the {moa} class.")

st.divider()

# -------------------------------------------------
# Market Data
# -------------------------------------------------
st.markdown("### Market Data")

with st.spinner("Loading market data..."):
    uk_trend = get_drug_uk_trend(selected_drug_id)
    us_spend = get_drug_us_spending(selected_drug_id)

col_uk, col_us = st.columns(2)

with col_uk:
    st.markdown("**UK Prescriptions (Monthly)**")
    if not uk_trend.empty:
        fig_uk = px.line(uk_trend, x="date", y="items", markers=True,
                         color_discrete_sequence=["#1B5E20"],
                         title=f"{drug_name} - UK Prescription Items")
        fig_uk.update_layout(**_layout(), height=300, xaxis_title="", yaxis_title="Items")
        fig_uk.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        fig_uk.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
        st.plotly_chart(fig_uk, use_container_width=True)
    else:
        st.info(f"No UK prescription data for {drug_name}.")

with col_us:
    st.markdown("**US Medicare Spending (Yearly)**")
    if not us_spend.empty:
        agg = us_spend.groupby("year").agg(total_spending=("total_spending", "sum")).reset_index()
        fig_us = px.bar(agg, x="year", y="total_spending", text="total_spending",
                        color_discrete_sequence=["#1565C0"],
                        title=f"{drug_name} - US Medicare Spending")
        fig_us.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont_size=10)
        fig_us.update_layout(**_layout(), height=300, xaxis_title="Year", yaxis_title="Spending ($)")
        fig_us.update_xaxes(showgrid=True, gridcolor="#e2e8f0", dtick=1)
        fig_us.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
        st.plotly_chart(fig_us, use_container_width=True)
    else:
        st.info(f"No US spending data for {drug_name}.")

st.divider()

# -------------------------------------------------
# Safety Profile
# -------------------------------------------------
st.markdown("### Safety Profile (FAERS)")

with st.spinner("Loading safety data..."):
    ae_data = get_drug_ae_data(selected_drug_id)
    ae_trend = get_drug_ae_trend(selected_drug_id)

col_ae1, col_ae2 = st.columns(2)

with col_ae1:
    if not ae_data.empty:
        top10 = ae_data.head(10).sort_values("total_count", ascending=True)
        fig_ae = go.Figure()
        fig_ae.add_trace(go.Bar(
            y=top10["event_term"], x=top10["total_count"], orientation="h",
            name="Total", marker_color=PRIMARY,
            text=top10["total_count"].apply(lambda x: f"{int(x):,}"),
            textposition="outside", textfont_size=10,
        ))
        fig_ae.add_trace(go.Bar(
            y=top10["event_term"], x=top10["serious_count"], orientation="h",
            name="Serious", marker_color=DANGER,
            text=top10["serious_count"].apply(lambda x: f"{int(x):,}"),
            textposition="outside", textfont_size=10,
        ))
        fig_ae.update_layout(**_layout(), height=350, barmode="overlay",
                             title=f"Top 10 Adverse Events: {drug_name}",
                             xaxis_title="Reports",
                             legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        fig_ae.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        st.plotly_chart(fig_ae, use_container_width=True)
    else:
        st.info(f"No FAERS data for {drug_name}.")

with col_ae2:
    if not ae_trend.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=ae_trend["quarter_date"], y=ae_trend["total_reports"],
            mode="lines+markers", name="Total",
            line=dict(color=PRIMARY, width=2.5), marker=dict(size=6),
        ))
        fig_trend.add_trace(go.Scatter(
            x=ae_trend["quarter_date"], y=ae_trend["serious_reports"],
            mode="lines+markers", name="Serious",
            line=dict(color=DANGER, width=2), marker=dict(size=5),
        ))
        fig_trend.update_layout(**_layout(), height=350,
                                title=f"FAERS Trend: {drug_name}",
                                xaxis_title="Quarter", yaxis_title="Reports",
                                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        fig_trend.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        fig_trend.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info(f"No FAERS trend data for {drug_name}.")

st.divider()

# -------------------------------------------------
# Patent & LOE
# -------------------------------------------------
st.markdown("### Patent & Loss of Exclusivity")

with st.spinner("Loading patent data..."):
    loe_summary = get_drug_loe_summary(selected_drug_id)
    patents = get_drug_patents(selected_drug_id)

col_loe, col_pat = st.columns(2)

with col_loe:
    if not loe_summary.empty:
        loe = loe_summary.iloc[0]
        trade = loe.get("trade_name", "N/A") or "N/A"
        loe_date = loe.get("effective_loe_date", "N/A")
        years = loe.get("years_until_loe", None)
        pat_count = int(loe.get("patent_count", 0))
        has_sub = bool(loe.get("has_substance_patent", 0))
        has_use = bool(loe.get("has_use_patent", 0))
        has_prod = bool(loe.get("has_product_patent", 0))
        excl_codes = loe.get("exclusivity_codes", "-") or "-"

        # Determine status color
        if years is not None:
            if years < 0:
                status_color = "#64748b"
                status_text = "Past LOE"
            elif years < 2:
                status_color = "#ef4444"
                status_text = "Imminent"
            elif years < 5:
                status_color = "#f59e0b"
                status_text = "Medium-term"
            else:
                status_color = "#22c55e"
                status_text = "Protected"
            years_str = f"{years:.1f} years"
        else:
            status_color = "#94a3b8"
            status_text = "Unknown"
            years_str = "N/A"

        patent_types = ", ".join(filter(None, [
            "Substance" if has_sub else None,
            "Use" if has_use else None,
            "Product" if has_prod else None,
        ])) or "None"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); border: 1px solid #e2e8f0;
                    border-left: 4px solid {status_color}; border-radius: 8px; padding: 20px; margin-bottom: 12px;">
            <div style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 600;">
                LOE Status: <span style="color: {status_color};">{status_text}</span>
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #0f172a; margin: 8px 0;">
                {loe_date}
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px;">
                <div><span style="font-size: 0.75rem; color: #64748b;">Years to LOE</span><br>
                    <span style="font-weight: 600;">{years_str}</span></div>
                <div><span style="font-size: 0.75rem; color: #64748b;">Trade Name</span><br>
                    <span style="font-weight: 600;">{trade}</span></div>
                <div><span style="font-size: 0.75rem; color: #64748b;">Patents</span><br>
                    <span style="font-weight: 600;">{pat_count}</span></div>
                <div><span style="font-size: 0.75rem; color: #64748b;">Patent Types</span><br>
                    <span style="font-weight: 600;">{patent_types}</span></div>
                <div><span style="font-size: 0.75rem; color: #64748b;">Exclusivity</span><br>
                    <span style="font-weight: 600;">{excl_codes}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No LOE data for {drug_name}. (Drug may be in pipeline or not marketed in the US)")

with col_pat:
    if not patents.empty:
        st.markdown("**Patent Portfolio**")
        display_pat = patents.copy()
        display_pat["drug_substance_flag"] = display_pat["drug_substance_flag"].apply(lambda x: x == "Y")
        display_pat["drug_product_flag"] = display_pat["drug_product_flag"].apply(lambda x: x == "Y")
        display_pat.columns = ["Patent No", "Expiry Date", "Substance", "Product", "Use Code", "Delisted"]
        display_pat = display_pat.drop(columns=["Delisted"])
        st.dataframe(display_pat, use_container_width=True, height=350,
                     column_config={
                         "Expiry Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                         "Substance": st.column_config.CheckboxColumn(),
                         "Product": st.column_config.CheckboxColumn(),
                     })
    else:
        st.info(f"No patent data for {drug_name}.")
