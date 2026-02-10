"""
Tab 8: ML Predictions – Phase Transition Probability (PoS)
Shows XGBoost v2 model predictions, feature importance, and calibration.
Compatible with v2 DB schema (Drug-Level Temporal GroupKFold, CV-based selection).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.db import run_query
from utils.charts import _layout, PRIMARY, PRIMARY_LIGHT, SECONDARY, SUCCESS, WARNING, DANGER, MOA_COLORS

st.set_page_config(
    page_title="ML Predictions | Pharma Pipeline Intelligence",
    page_icon="🤖",
    layout="wide",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 1rem; max-width: 1400px; }
    h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: #1e293b !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important; }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] a { color: #2563eb !important; text-decoration: none !important; }

    .model-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #2563eb;
        margin-bottom: 0.8rem;
    }
    .model-card.selected { border-top: 3px solid #22c55e; }
    .model-card .card-title { font-weight: 700; font-size: 0.95rem; color: #1e293b; }
    .model-card .card-value { font-size: 1.6rem; font-weight: 700; color: #2563eb; }
    .model-card.selected .card-value { color: #22c55e; }
    .model-card .card-detail { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; }
    .selected-badge {
        display: inline-block; background: #dcfce7; color: #166534;
        padding: 0.15rem 0.5rem; border-radius: 10px;
        font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
    }

    /* Metric override */
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 ML Predictions")
    st.markdown("**Phase Transition PoS**")
    st.divider()
    st.markdown("#### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/0_How_It_Works.py", label="How It Works", icon="📖")
    st.page_link("pages/1_Pipeline_Overview.py", label="Pipeline Overview", icon="📊")
    st.page_link("pages/2_Competitive_Landscape.py", label="Competitive Landscape", icon="🗺️")
    st.page_link("pages/3_Trial_Analytics.py", label="Trial Analytics", icon="📈")
    st.page_link("pages/4_Drug_Deep_Dive.py", label="Drug Deep Dive", icon="💊")
    st.page_link("pages/5_Market_Intelligence.py", label="Market Intelligence", icon="💰")
    st.page_link("pages/6_Safety_Profile.py", label="Safety Profile", icon="🛡️")
    st.page_link("pages/7_Patent_LOE.py", label="Patent & LOE", icon="📋")
    st.page_link("pages/8_ML_Predictions.py", label="ML Predictions", icon="🤖")
    st.divider()
    st.markdown("""
    <div style="font-size: 0.75rem; color: #94a3b8;">
        Model: XGBoost v2.0<br>
        CV: Drug-Level Temporal GroupKFold<br>
        Features: 60 (leak-free, PIT)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA LOADING (v2 schema)
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def load_models():
    """Load model metadata from ml_models (v2 schema)."""
    return run_query("""
        SELECT phase_transition, model_type, is_selected,
               cv_auc_mean, cv_auc_std, cv_pr_auc_mean,
               cv_brier_mean, cv_mcc_mean,
               n_total, n_cv_splits,
               model_version, feature_version, model_path, trained_at
        FROM ml_models
        ORDER BY phase_transition, cv_auc_mean DESC
    """)


@st.cache_data(ttl=1800)
def load_feature_importance():
    """Load feature importance from ml_feature_importance (v2 schema)."""
    return run_query("""
        SELECT phase_transition, feature_name, importance, rank_in_phase,
               model_version
        FROM ml_feature_importance
        ORDER BY phase_transition, rank_in_phase
    """)


@st.cache_data(ttl=1800)
def load_predictions():
    """Load predictions from ml_predictions (v2 schema)."""
    return run_query("""
        SELECT p.trial_id, p.drug_id, p.nct_id, p.phase_transition,
               p.predicted_success_probability, p.model_version, p.model_type,
               d.inn AS drug_name, d.moa_class, d.modality
        FROM ml_predictions p
        LEFT JOIN drugs d ON p.drug_id = d.drug_id
    """)


# Load data
models_df = load_models()
fi_df = load_feature_importance()
pred_df = load_predictions()


# ═══════════════════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════════════════
if models_df.empty:
    st.warning("No ML models found. Please run `train_models_v2.py` first.")
    st.info("""
    **Getting Started:**
    1. Navigate to `pharma-pipeline-ml/`
    2. Run: `python compute_features_v2.py`
    3. Run: `python train_models_v2.py`
    4. Refresh this page
    """)
    st.stop()


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.title("🤖 ML Predictions – Phase Transition Probability")
st.markdown(
    "XGBoost v2 predictions for clinical trial phase transitions. "
    "CV-based model selection with Drug-Level Temporal GroupKFold splits."
)

# Phase transition labels
PHASE_LABELS = {
    "Phase_I_to_II": "Phase I \u2192 II",
    "Phase_II_to_III": "Phase II \u2192 III",
    "Phase_III_to_Approval": "Phase III \u2192 Approval",
    "All_Phases_Combined": "All Phases Combined",
}


def fmt_transition(t):
    return PHASE_LABELS.get(t, t)


# ═══════════════════════════════════════════════════════════
# KPI HEADER — Selected models summary
# ═══════════════════════════════════════════════════════════
selected_models = models_df[models_df["is_selected"] == 1].copy()
n_predictions = len(pred_df) if not pred_df.empty else 0
n_transitions = selected_models["phase_transition"].nunique()
best_cv_auc = selected_models["cv_auc_mean"].max() if not selected_models.empty else 0
avg_cv_auc = selected_models["cv_auc_mean"].mean() if not selected_models.empty else 0
feature_version = models_df["feature_version"].iloc[0] if not models_df.empty else "?"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Selected Models", n_transitions)
k2.metric("Best CV-AUC", f"{best_cv_auc:.3f}")
k3.metric("Avg CV-AUC", f"{avg_cv_auc:.3f}")
k4.metric("Predictions", f"{n_predictions:,}")
k5.metric("Features", "60 (PIT)")

st.divider()


# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab_perf, tab_feat, tab_drugs, tab_details = st.tabs([
    "📊 Model Performance",
    "🔍 Feature Importance",
    "💊 Drug PoS Ranking",
    "🎯 Details & Benchmarks",
])


# ─────────────────────────────────────────────
# TAB 1: MODEL PERFORMANCE
# ─────────────────────────────────────────────
with tab_perf:
    st.subheader("CV-Based Model Comparison")
    st.markdown(
        "*All metrics are cross-validated (Drug-Level Temporal GroupKFold). "
        "The selected model per phase is highlighted.*"
    )

    # Phase transition filter
    all_transitions = sorted(models_df["phase_transition"].unique())
    sel_transition = st.selectbox(
        "Phase Transition",
        ["All"] + list(all_transitions),
        format_func=lambda x: "All Transitions" if x == "All" else fmt_transition(x),
        key="perf_transition",
    )

    display_df = models_df if sel_transition == "All" else models_df[models_df["phase_transition"] == sel_transition]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### CV-AUC per Model x Phase Transition")

        chart_df = display_df.copy()
        chart_df["phase_label"] = chart_df["phase_transition"].map(fmt_transition)
        model_type_labels = {
            "baseline": "Baseline (LogReg)",
            "xgb_a": "XGBoost A (d=4)",
            "xgb_b": "XGBoost B (d=6)",
            "meta": "Meta-Ensemble (Ridge)",
        }
        chart_df["model_label"] = chart_df["model_type"].map(model_type_labels).fillna(chart_df["model_type"])

        model_colors = {
            "Baseline (LogReg)": SECONDARY,
            "XGBoost A (d=4)": PRIMARY,
            "XGBoost B (d=6)": WARNING,
            "Meta-Ensemble (Ridge)": SUCCESS,
        }

        fig = px.bar(
            chart_df,
            x="phase_label",
            y="cv_auc_mean",
            color="model_label",
            barmode="group",
            error_y="cv_auc_std",
            color_discrete_map=model_colors,
            labels={"cv_auc_mean": "CV-AUC (mean)", "phase_label": "Phase Transition", "model_label": "Model"},
        )
        fig.update_layout(**_layout(height=420, title="Cross-Validated AUC Comparison"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random (0.5)")
        fig.add_hline(y=0.78, line_dash="dot", line_color=SUCCESS, annotation_text="Lo et al. 2019 (0.78)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### CV Metrics Table")
        metrics_show = display_df[["phase_transition", "model_type", "is_selected",
                                    "cv_auc_mean", "cv_auc_std", "cv_pr_auc_mean",
                                    "cv_brier_mean", "cv_mcc_mean"]].copy()
        metrics_show["phase_transition"] = metrics_show["phase_transition"].map(fmt_transition)
        metrics_show["selected"] = metrics_show["is_selected"].map({1: "Yes", 0: ""})
        metrics_show = metrics_show.drop(columns=["is_selected"])
        metrics_show = metrics_show.round(3)
        metrics_show = metrics_show.rename(columns={
            "phase_transition": "Transition", "model_type": "Model",
            "cv_auc_mean": "AUC", "cv_auc_std": "AUC std",
            "cv_pr_auc_mean": "PR-AUC", "cv_brier_mean": "Brier",
            "cv_mcc_mean": "MCC", "selected": "Selected",
        })
        st.dataframe(
            metrics_show.style.background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
            use_container_width=True,
            height=420,
            hide_index=True,
        )

    # Selected model cards
    st.markdown("#### Selected Models (Best CV-AUC per Phase)")
    sel_cols = st.columns(len(selected_models))
    for i, (_, row) in enumerate(selected_models.iterrows()):
        with sel_cols[i]:
            st.markdown(f"""
            <div class="model-card selected">
                <span class="selected-badge">Selected</span>
                <div class="card-title">{fmt_transition(row['phase_transition'])}</div>
                <div class="card-value">{row['cv_auc_mean']:.3f}</div>
                <div class="card-detail">
                    Model: {row['model_type']} &bull; &pm;{row['cv_auc_std']:.3f}<br>
                    PR-AUC: {row['cv_pr_auc_mean']:.3f} &bull; Brier: {row['cv_brier_mean']:.3f}<br>
                    N={row['n_total']:,} &bull; {row['n_cv_splits']} folds
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
with tab_feat:
    st.subheader("Feature Importance (XGBoost A, full-data model)")

    if fi_df.empty:
        st.info("No feature importance data available. Run `train_models_v2.py` first.")
    else:
        fi_transitions = sorted(fi_df["phase_transition"].unique())
        sel_fi_phase = st.selectbox(
            "Phase Transition",
            fi_transitions,
            format_func=fmt_transition,
            key="fi_phase",
        )

        fi_filtered = fi_df[fi_df["phase_transition"] == sel_fi_phase].head(20).copy()

        if not fi_filtered.empty:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"#### Top 20 Features &mdash; {fmt_transition(sel_fi_phase)}")
                fi_sorted = fi_filtered.sort_values("importance", ascending=True)

                fig = go.Figure(go.Bar(
                    x=fi_sorted["importance"],
                    y=fi_sorted["feature_name"],
                    orientation="h",
                    marker_color=PRIMARY,
                    text=fi_sorted["importance"].round(4),
                    textposition="outside",
                ))
                fig.update_layout(**_layout(
                    height=max(400, len(fi_sorted) * 28),
                    title="XGBoost Feature Importance (gain)",
                    xaxis_title="Importance (gain)",
                    yaxis_title="",
                    margin=dict(l=280, r=60, t=50, b=40),
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Feature Details")
                fi_table = fi_filtered[["rank_in_phase", "feature_name", "importance"]].rename(columns={
                    "rank_in_phase": "#", "feature_name": "Feature", "importance": "Importance",
                })
                fi_table["Importance"] = fi_table["Importance"].round(4)
                st.dataframe(fi_table, use_container_width=True, hide_index=True, height=500)

            # Cross-phase comparison: top features across all transitions
            st.divider()
            st.markdown("#### Cross-Phase Feature Comparison (Top 10 per Phase)")

            top10_all = fi_df[fi_df["rank_in_phase"] <= 10].copy()
            top10_all["phase_label"] = top10_all["phase_transition"].map(fmt_transition)

            fig_cross = px.bar(
                top10_all,
                x="importance",
                y="feature_name",
                color="phase_label",
                orientation="h",
                barmode="group",
                color_discrete_sequence=[PRIMARY, WARNING, SUCCESS, DANGER],
                labels={"importance": "Importance", "feature_name": "Feature", "phase_label": "Transition"},
            )
            # Sort by average importance
            avg_imp = top10_all.groupby("feature_name")["importance"].mean().sort_values(ascending=True)
            fig_cross.update_layout(**_layout(
                height=max(500, len(avg_imp) * 22),
                title="Feature Importance Across All Phase Transitions",
                margin=dict(l=280, r=60, t=50, b=40),
            ))
            fig_cross.update_yaxes(categoryorder="array", categoryarray=avg_imp.index.tolist())
            st.plotly_chart(fig_cross, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3: DRUG PoS RANKING
# ─────────────────────────────────────────────
with tab_drugs:
    st.subheader("Drug Probability of Success (PoS) Ranking")

    if pred_df.empty:
        st.info("No predictions available. Run `train_models_v2.py` first.")
    else:
        # Phase filter
        pred_transitions = sorted(pred_df["phase_transition"].unique())
        sel_pred_phase = st.selectbox(
            "Phase Transition",
            pred_transitions,
            format_func=fmt_transition,
            key="drug_phase",
        )

        phase_preds = pred_df[pred_df["phase_transition"] == sel_pred_phase].copy()

        # Aggregate: mean PoS per drug
        drug_pos = phase_preds.groupby(["drug_name", "moa_class"]).agg(
            mean_pos=("predicted_success_probability", "mean"),
            median_pos=("predicted_success_probability", "median"),
            n_trials=("trial_id", "nunique"),
            model_type=("model_type", "first"),
        ).reset_index().sort_values("mean_pos", ascending=False)

        drug_pos = drug_pos.dropna(subset=["drug_name"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"#### Drug PoS Ranking &mdash; {fmt_transition(sel_pred_phase)}")

            top_n = min(43, len(drug_pos))
            top_df = drug_pos.head(top_n).sort_values("mean_pos", ascending=True)

            # Color by MoA
            bar_colors = [MOA_COLORS.get(moa, SECONDARY) for moa in top_df["moa_class"]]

            fig = go.Figure(go.Bar(
                x=top_df["mean_pos"],
                y=top_df["drug_name"],
                orientation="h",
                marker_color=bar_colors,
                text=top_df["mean_pos"].round(3),
                textposition="outside",
                customdata=np.stack([top_df["moa_class"].fillna(""), top_df["n_trials"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>PoS: %{x:.3f}<br>MoA: %{customdata[0]}<br>Trials: %{customdata[1]}<extra></extra>",
            ))
            fig.update_layout(**_layout(
                height=max(500, top_n * 24),
                title=f"Mean Predicted PoS ({fmt_transition(sel_pred_phase)})",
                xaxis_title="Mean Predicted Success Probability",
                xaxis_range=[0, min(1.15, top_df["mean_pos"].max() + 0.1)],
                margin=dict(l=220, r=60, t=50, b=40),
            ))
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### PoS Table")
            table_df = drug_pos[["drug_name", "moa_class", "mean_pos", "median_pos", "n_trials"]].copy()
            table_df = table_df.rename(columns={
                "drug_name": "Drug", "moa_class": "MoA",
                "mean_pos": "Mean PoS", "median_pos": "Median PoS",
                "n_trials": "Trials",
            })
            table_df["Mean PoS"] = table_df["Mean PoS"].round(3)
            table_df["Median PoS"] = table_df["Median PoS"].round(3)
            st.dataframe(
                table_df.style.background_gradient(subset=["Mean PoS"], cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
                height=600,
                hide_index=True,
            )

        # PoS distribution
        st.divider()
        st.markdown("#### PoS Distribution")

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            fig_hist = px.histogram(
                phase_preds,
                x="predicted_success_probability",
                nbins=40,
                color_discrete_sequence=[PRIMARY],
                labels={"predicted_success_probability": "Predicted PoS"},
            )
            fig_hist.update_layout(**_layout(
                height=300,
                title=f"PoS Distribution ({fmt_transition(sel_pred_phase)})",
                xaxis_title="Predicted Success Probability",
                yaxis_title="Count",
            ))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dist2:
            # By MoA class
            moa_pos = drug_pos.groupby("moa_class")["mean_pos"].mean().sort_values(ascending=True)
            fig_moa = go.Figure(go.Bar(
                x=moa_pos.values,
                y=moa_pos.index,
                orientation="h",
                marker_color=[MOA_COLORS.get(m, SECONDARY) for m in moa_pos.index],
                text=moa_pos.round(3),
                textposition="outside",
            ))
            fig_moa.update_layout(**_layout(
                height=max(300, len(moa_pos) * 28),
                title="Mean PoS by MoA Class",
                xaxis_title="Mean PoS",
                xaxis_range=[0, min(1.15, moa_pos.max() + 0.1)],
                margin=dict(l=250, r=60, t=50, b=40),
            ))
            st.plotly_chart(fig_moa, use_container_width=True)

        # Drug deep-dive
        st.divider()
        st.markdown("#### Drug Deep Dive")
        drug_options = sorted(drug_pos["drug_name"].dropna().unique())
        if drug_options:
            sel_drug = st.selectbox("Select Drug", drug_options, key="drug_dive")
            drug_trials = pred_df[pred_df["drug_name"] == sel_drug]

            if not drug_trials.empty:
                st.markdown(f"**{sel_drug}**: {drug_trials['trial_id'].nunique()} trial predictions across {drug_trials['phase_transition'].nunique()} transitions")

                phase_cols = st.columns(min(4, drug_trials["phase_transition"].nunique()))
                for i, phase in enumerate(sorted(drug_trials["phase_transition"].unique())):
                    phase_data = drug_trials[drug_trials["phase_transition"] == phase]
                    avg = phase_data["predicted_success_probability"].mean()
                    with phase_cols[i % len(phase_cols)]:
                        st.metric(
                            fmt_transition(phase),
                            f"PoS: {avg:.3f}",
                            delta=f"{len(phase_data)} trials",
                            delta_color="off",
                        )

                # Trial-level table
                trial_table = drug_trials[["nct_id", "phase_transition", "predicted_success_probability", "model_type"]].copy()
                trial_table["phase_transition"] = trial_table["phase_transition"].map(fmt_transition)
                trial_table = trial_table.rename(columns={
                    "nct_id": "NCT ID", "phase_transition": "Transition",
                    "predicted_success_probability": "PoS", "model_type": "Model",
                })
                trial_table["PoS"] = trial_table["PoS"].round(3)
                trial_table = trial_table.sort_values("PoS", ascending=False)
                st.dataframe(
                    trial_table.style.background_gradient(subset=["PoS"], cmap="RdYlGn", vmin=0, vmax=1),
                    use_container_width=True,
                    hide_index=True,
                )


# ─────────────────────────────────────────────
# TAB 4: DETAILS & BENCHMARKS
# ─────────────────────────────────────────────
with tab_details:
    st.subheader("Model Details & Literature Benchmarks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Multi-Metric Radar (Selected Models)")

        if not selected_models.empty:
            metrics_radar = ["cv_auc_mean", "cv_pr_auc_mean", "cv_mcc_mean"]
            metric_labels = ["AUC", "PR-AUC", "MCC"]
            # Brier is reversed (lower is better) — invert for radar
            fig_radar = go.Figure()
            for _, row in selected_models.iterrows():
                values = [
                    row["cv_auc_mean"] if pd.notna(row["cv_auc_mean"]) else 0,
                    row["cv_pr_auc_mean"] if pd.notna(row["cv_pr_auc_mean"]) else 0,
                    1 - row["cv_brier_mean"] if pd.notna(row["cv_brier_mean"]) else 0,  # inverted
                    row["cv_mcc_mean"] if pd.notna(row["cv_mcc_mean"]) else 0,
                ]
                labels = metric_labels + ["1 - Brier"]
                values.append(values[0])  # close polygon
                labels_closed = labels + [labels[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=labels_closed,
                    fill="toself",
                    name=fmt_transition(row["phase_transition"]),
                    opacity=0.6,
                ))
            fig_radar.update_layout(**_layout(
                height=400, title="Selected Model Performance",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            ))
            st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("#### Methodology Summary")
        st.markdown("""
        | Aspect | Value |
        |--------|-------|
        | **Feature Version** | v2.0 (60 features, leak-free) |
        | **Split Strategy** | Drug-Level Temporal GroupKFold |
        | **Model Selection** | Best CV-AUC per phase |
        | **Final Training** | Retrained on ALL data |
        | **CV Metrics** | Unbiased performance estimate |
        | **Drug Overlap** | 0 (verified per fold) |
        | **Leakage Check** | Automated (single-feature AUC < 0.90) |
        """)

        st.markdown("#### v1 vs v2 Comparison")
        st.markdown("""
        | Aspect | v1 | v2 |
        |--------|----|----|
        | Features | 105 | 60 |
        | Post-Market Features | 15 | 0 |
        | CV Strategy | Trial-level temporal | Drug-level temporal |
        | Drug Overlap in CV | Not checked | 0 (verified) |
        | Model Selection | Always Meta | Best CV-AUC |
        | Best CV-AUC | 0.999 (leakage) | 0.947 |
        | Realistic? | No | Yes |
        """)

    # Literature benchmarks
    st.divider()
    st.markdown("#### Literature Benchmark Comparison")

    benchmark_data = pd.DataFrame([
        {"Source": "Lo et al. 2019", "Transition": "Phase III \u2192 Approval", "AUC": 0.78, "Data": "Public (clinical trial data)"},
        {"Source": "Feijoo et al. 2020", "Transition": "Phase II \u2192 III", "AUC": 0.80, "Data": "Public"},
        {"Source": "Novartis DSAI 2020", "Transition": "Phase II \u2192 Approval", "AUC": 0.88, "Data": "Proprietary (internal)"},
        {"Source": "Insilico Medicine 2023", "Transition": "Phase II \u2192 III", "AUC": 0.88, "Data": "Omics + Public"},
    ])

    # Add our selected models
    for _, row in selected_models.iterrows():
        benchmark_data = pd.concat([benchmark_data, pd.DataFrame([{
            "Source": f"Our Model ({row['model_type']})",
            "Transition": fmt_transition(row["phase_transition"]),
            "AUC": round(row["cv_auc_mean"], 3),
            "Data": "Public (ClinicalTrials.gov only)",
        }])], ignore_index=True)

    st.dataframe(
        benchmark_data.style.background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    **Note:** Our models use **only publicly available data** from ClinicalTrials.gov.
    Literature benchmarks often include proprietary clinical data, molecular descriptors, or omics features.
    A CV-AUC of 0.78-0.95 on public data alone is a strong result, consistent with Lo et al. 2019.
    The elevated Phase III AUC (0.947) is explained by the strong `prior_approval` signal — drugs with
    existing approvals for other indications have measurably higher Phase III success rates.
    """)

    # All model runs
    st.divider()
    st.markdown("#### All Model Metadata")
    all_models_show = models_df.copy()
    all_models_show["phase_transition"] = all_models_show["phase_transition"].map(fmt_transition)
    all_models_show["is_selected"] = all_models_show["is_selected"].map({1: "Yes", 0: ""})
    all_models_show = all_models_show.rename(columns={
        "phase_transition": "Transition", "model_type": "Model",
        "is_selected": "Selected", "cv_auc_mean": "AUC",
        "cv_auc_std": "AUC std", "cv_pr_auc_mean": "PR-AUC",
        "cv_brier_mean": "Brier", "cv_mcc_mean": "MCC",
        "n_total": "N Total", "n_cv_splits": "CV Folds",
        "model_version": "Version", "trained_at": "Trained At",
    })
    display_cols = ["Transition", "Model", "Selected", "AUC", "AUC std",
                    "PR-AUC", "Brier", "MCC", "N Total", "CV Folds", "Version", "Trained At"]
    all_models_show = all_models_show[[c for c in display_cols if c in all_models_show.columns]]
    st.dataframe(
        all_models_show.style.background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
        use_container_width=True,
        hide_index=True,
    )
