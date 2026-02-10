"""
How It Works — Architecture & Methodology Documentation
Explains each component of the Pharma Pipeline Intelligence project.
Teal/Emerald accent theme to distinguish from the blue dashboard pages.
"""

import streamlit as st

st.set_page_config(
    page_title="How It Works | Pharma Pipeline Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS — Teal/Emerald documentation theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 1400px;}
    h1 {font-size: 1.8rem !important; font-weight: 700 !important; color: #0f172a !important;}
    h2 {font-size: 1.3rem !important; font-weight: 600 !important; color: #134e4a !important;}
    h3 {font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important;}

    /* Sidebar — light base with teal accent strip */
    section[data-testid="stSidebar"] {
        background-color: #f0fdfa;
        border-right: 3px solid #0d9488;
    }

    /* Header — teal gradient instead of blue */
    .how-header {
        background: linear-gradient(135deg, #134e4a 0%, #0f766e 40%, #0d9488 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.25);
    }
    .how-header h1 { color: white !important; font-size: 2.2rem !important; margin-bottom: 0.3rem !important; }
    .how-header p { color: #99f6e4 !important; font-size: 1.1rem !important; margin: 0 !important; }

    /* Context badge — tells user they're in documentation */
    .doc-badge {
        display: inline-block;
        background: #ccfbf1; color: #0f766e;
        padding: 0.2rem 0.8rem; border-radius: 20px;
        font-weight: 600; font-size: 0.78rem;
        letter-spacing: 0.04em; text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Tech badges */
    .tech-badge {
        background: #f0fdfa; color: #134e4a;
        padding: 0.25rem 0.7rem; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem;
        display: inline-block; margin: 0.15rem 0.2rem;
    }
    .tech-badge.api { background: #ccfbf1; color: #0f766e; }
    .tech-badge.db { background: #d1fae5; color: #065f46; }
    .tech-badge.frontend { background: #e0e7ff; color: #3730a3; }
    .tech-badge.python { background: #fef3c7; color: #92400e; }

    /* Pipeline steps — teal border */
    .pipeline-step {
        background: #f0fdfa; border-radius: 8px;
        padding: 0.6rem 1rem; margin: 0.3rem 0;
        border-left: 3px solid #0d9488; font-size: 0.9rem;
    }

    /* Architecture grid */
    .arch-grid {
        display: grid; grid-template-columns: 2fr 1fr;
        gap: 1.5rem; margin-bottom: 1rem;
    }
    .arch-box {
        background: white; border-radius: 14px;
        padding: 1.5rem 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-top: 3px solid #0d9488; box-sizing: border-box;
    }

    /* Tables */
    .stMarkdown table {
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stMarkdown th { background: #f0fdfa !important; }

    /* Section headings — teal underline */
    .section-head {
        color: #134e4a; font-size: 1.15rem; font-weight: 700;
        margin-top: 1.8rem; margin-bottom: 0.6rem;
        padding-bottom: 0.4rem; border-bottom: 2px solid #99f6e4;
    }

    /* Intelligence items — teal accent */
    .intel-item {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border-left: 3px solid #0d9488;
        padding: 0.55rem 0.9rem; margin: 0.3rem 0;
        border-radius: 0 6px 6px 0; font-size: 0.9rem;
    }

    /* Challenge items — amber accent (same as dashboard) */
    .challenge-item {
        background: linear-gradient(135deg, #fff7ed, #fffbeb);
        border-left: 3px solid #ea580c;
        padding: 0.55rem 0.9rem; margin: 0.3rem 0;
        border-radius: 0 6px 6px 0; font-size: 0.9rem;
    }

    /* Scale cards — teal top border */
    .scale-card {
        background: white; border-radius: 12px;
        padding: 1.1rem 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 3px solid #0d9488;
        min-height: 110px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .scale-value { font-size: 1.7rem; font-weight: 700; color: #0f766e; }
    .scale-label { font-size: 0.82rem; color: #555; margin-top: 0.25rem; }
    .scale-detail { font-size: 0.72rem; color: #999; margin-top: 0.3rem; }

    /* Back-to-dashboard link */
    .back-link {
        display: inline-flex; align-items: center; gap: 0.4rem;
        color: #0d9488 !important; font-weight: 600; font-size: 0.9rem;
        text-decoration: none !important; margin-bottom: 0.5rem;
    }
    .back-link:hover { color: #0f766e !important; text-decoration: underline !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Sidebar — consistent with dashboard but teal-tinted
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📖 Project Documentation")
    st.markdown("**Pharma Pipeline Intelligence**")
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
    st.markdown(
        "<div style='color: #5eead4; font-size: 0.75rem;'>"
        "You are viewing<br><b style='color: #0d9488;'>Project Documentation</b>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════
st.markdown('<span class="doc-badge">Documentation</span>', unsafe_allow_html=True)
st.markdown("""
<div class="how-header">
    <h1>📖 How It Works</h1>
    <p>Architecture, data engineering, and methodology behind the Pharma Pipeline Intelligence platform</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Section Navigation — tabs in main content area
# ═══════════════════════════════════════════════════════════════
SECTIONS = [
    "🏗️ Architecture Overview",
    "🗄️ Data Engineering",
    "📊 Dashboard & Visualization",
    "🔗 External Data Sources",
    "🤖 ML Predictions",
    "🧠 Key Design Decisions",
]

selected = st.radio(
    "Section",
    SECTIONS,
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown(f"## {selected}")


# ═══════════════════════════════════════════════════════════════
# Section Data
# ═══════════════════════════════════════════════════════════════

SECTION_DATA = {

    # ----------------------------------------------------------
    "data_engineering": {
        "icon": "🗄️",
        "title": "Data Engineering",
        "what_it_does": """
Ingests, normalizes, and links clinical trial data from the **ClinicalTrials.gov API v2** into a structured **Azure SQL** database.

The pipeline handles:
- **32,811 clinical trials** across Diabetes & Obesity indications
- **43 drugs** with INN names, MoA classes, modalities
- **Entity linking**: Drug&rarr;Trial, Trial&rarr;Indication, Company&rarr;Type
- **6,023 sponsor companies** classified into 4 types (academic, biotech, big_pharma, government)
- **13 disease indications** mapped from free-text condition fields
""",
        "how_it_works": [
            ("Step 1", "Query ClinicalTrials.gov API v2 with condition filters (Diabetes, Obesity, etc.) &mdash; paginated at 1,000 studies/page"),
            ("Step 2", "Parse JSON responses: extract NCT ID, title, phase, status, sponsor, enrollment, dates, interventions"),
            ("Step 3", "Bulk INSERT into Azure SQL (7 normalized tables) with duplicate detection via NCT ID"),
            ("Step 4", "Entity Resolution: match drug names (INN) to trial interventions via fuzzy string matching"),
            ("Step 5", "Coverage validation: verify Drug&rarr;Trial linking rate, iterate to improve (76.2% &rarr; 78.8%)"),
        ],
        "intelligence": [
            "<b>MoA Classification (22 classes)</b> &mdash; each of 43 drugs assigned to mechanism of action class (GLP-1 RA, SGLT2 Inhibitor, DPP-4 Inhibitor, etc.) based on pharmacological knowledge",
            "<b>Indication Mapping (13 indications)</b> &mdash; free-text condition fields normalized to standardized indication names (Type 2 Diabetes, Obesity, NASH/MASH, etc.)",
            "<b>Company Type Resolution</b> &mdash; 6,023 unique sponsor names classified into academic, biotech, big_pharma, government via curated lookup tables",
            "<b>Drug-Trial Linking Strategy</b> &mdash; INN-based matching with alias handling (semaglutide = Ozempic = Wegovy = Rybelsus)",
            "<b>Phase Normalization</b> &mdash; ClinicalTrials.gov uses phase1, phase2_phase3 etc. &mdash; mapped to human-readable 'Phase 1', 'Phase 2/3' with numerical ranking for sorting",
            "<b>Stale Trial Detection</b> &mdash; automated flagging of trials not updated in >12 months past their completion date",
        ],
        "metrics": [
            ("Drug&rarr;Trial Coverage", "78.8%", "Primary quality metric: what % of drugs have linked trials"),
            ("Total Trials Loaded", "32,811", "Full ClinicalTrials.gov extract for Diabetes & Obesity"),
            ("Drugs Tracked", "43", "Curated list spanning 22 MoA classes, old and new"),
            ("Normalized Tables", "7", "drugs, trials, companies, indications, drug_trials, trial_indications, approvals"),
            ("Companies Resolved", "6,023 &rarr; 4 types", "Sponsor classification enables competitive analysis"),
            ("Stale Trials Flagged", "automated", "Real-time detection of potentially abandoned studies"),
        ],
        "challenges": [
            ("STRING_AGG(DISTINCT) not supported in Azure SQL", "Azure SQL / T-SQL does not support DISTINCT inside STRING_AGG &mdash; solved with nested subquery pattern: <code>(SELECT STRING_AGG(x.name, ', ') FROM (SELECT DISTINCT ...) x)</code>"),
            ("API pagination at 1,000 studies/page", "ClinicalTrials.gov limits to 1,000 results per request &mdash; implemented offset-based pagination with pageToken handling"),
            ("Drug name normalization (INN vs. Brand)", "Clinical trials reference drugs by INN, brand name, or chemical name &mdash; built alias mapping table for all 43 drugs"),
            ("originator_company_id always NULL", "The drugs table had no company linkage populated &mdash; solved by deriving top sponsor and company type from trial sponsor data via correlated subqueries"),
            ("Phase value format mismatch", "Database stores 'phase3' but UI needs 'Phase 3' &mdash; implemented CASE-based mapping in SQL with numerical ranking for correct sort order"),
        ],
        "poc_vs_prod": [
            ("Manual Python ETL scripts", "Azure Data Factory / Apache Airflow orchestrated pipelines"),
            ("pyodbc direct SQL", "SQLAlchemy ORM with migration management (Alembic)"),
            ("CSV-based drug/indication mapping", "ML-based entity linking (BiomedBERT, SciSpaCy)"),
            ("One-time bulk load", "Incremental daily sync with CDC (Change Data Capture)"),
            ("Manual coverage validation", "Automated data quality monitoring (Great Expectations)"),
            ("Single developer review", "Data steward review process with approval workflows"),
        ],
        "scale": [
            ("32,811", "Clinical Trials", "from ClinicalTrials.gov API v2"),
            ("43", "Drugs Tracked", "across 22 MoA classes"),
            ("78.8%", "Drug&rarr;Trial Coverage", "entity linking accuracy"),
        ],
    },

    # ----------------------------------------------------------
    "dashboard": {
        "icon": "📊",
        "title": "Dashboard & Visualization",
        "what_it_does": """
A **4-tab Streamlit dashboard** with interactive Plotly charts providing competitive intelligence on the Diabetes & Obesity drug pipeline.

The 4 tabs deliver:
- **Pipeline Overview** &mdash; 43 drugs as filterable table with MoA, Phase, Indication, Sponsor filters
- **Competitive Landscape** &mdash; MoA &times; Indication heatmap, trend lines, hot/cold zones
- **Trial Analytics** &mdash; status/phase donuts, stacked area charts, top sponsors, termination rates
- **Drug Deep Dive** &mdash; single-drug card with indication status, trial timeline, FDA approvals, competitive position
""",
        "how_it_works": [
            ("Step 1", "Cached database connection via <code>@st.cache_resource</code> &mdash; single pyodbc connection reused across requests"),
            ("Step 2", "20+ SQL queries in <code>utils/queries.py</code> with <code>@st.cache_data(ttl=3600)</code> &mdash; 1-hour cache"),
            ("Step 3", "Results as Pandas DataFrames &mdash; client-side filtering via multiselect widgets"),
            ("Step 4", "10+ reusable Plotly chart functions in <code>utils/charts.py</code> with consistent color palette"),
            ("Step 5", "Multi-page Streamlit layout with sidebar navigation and custom CSS theming"),
        ],
        "intelligence": [
            "<b>Phase Ranking System</b> &mdash; highest_phase derived from trials via correlated subquery with numerical ranking (phase4=7 &gt; phase3=6 &gt; ... &gt; early_phase1=1)",
            "<b>Sponsor Type Derivation</b> &mdash; since originator_company_id was NULL for all drugs, company_type is derived from the most frequent trial sponsor type per drug",
            "<b>Hot/Cold Zone Detection</b> &mdash; compares trial starts in recent 2 years vs. prior 2 years per MoA &times; Indication combination to identify growing and declining areas",
            "<b>Indication-Aware Filtering</b> &mdash; drugs can have multiple indications (comma-separated) &mdash; filter uses substring matching instead of exact match",
            "<b>Professional Color Palette</b> &mdash; 15 MoA-specific colors + 9 trial status colors for visual consistency across all charts",
            "<b>Stale Trial Monitor</b> &mdash; identifies trials past completion date with no updates &mdash; provides CT.gov links for verification",
        ],
        "metrics": [
            ("Dashboard Tabs", "4", "Pipeline Overview, Competitive Landscape, Trial Analytics, Drug Deep Dive"),
            ("Chart Types", "10+", "Bar, donut, heatmap, scatter, area, line, horizontal bar, competitive comparison"),
            ("Interactive Filters", "4", "MoA Class, Indication, Highest Phase, Sponsor Type &mdash; all multiselect"),
            ("SQL Queries", "~20", "Centralized in queries.py, all cached with 1-hour TTL"),
            ("CSV Export", "per tab", "Download filtered data as CSV directly from the dashboard"),
            ("KPI Metrics", "9", "Home page: drugs, trials, MoA classes, indications, companies, active/terminated/stale/results"),
        ],
        "challenges": [
            ("Plotly margin conflict (TypeError)", "<code>**LAYOUT_DEFAULTS</code> contained margin AND functions passed explicit margin &mdash; solved by splitting into <code>_LAYOUT_BASE</code> (no margin) + <code>_layout()</code> helper"),
            ("Streamlit cache unhashable params", "<code>params=[drug_id]</code> (list) is not hashable for <code>@st.cache_data</code> &mdash; changed all to <code>params=(drug_id,)</code> (tuple)"),
            ("NaN values in Plotly scatter size", "Enrollment can be NULL &mdash; Plotly raises ValueError on NaN in size property &mdash; solved with <code>fillna(0).astype(float)</code>"),
            ("SELECT DISTINCT + ORDER BY in SQL Server", "T-SQL requires ORDER BY items in SELECT list when using DISTINCT &mdash; wrapped in subquery with alias column"),
            ("Secrets fallback chain", "<code>st.secrets.get()</code> throws FileNotFoundError when no secrets.toml exists &mdash; wrapped in try/except with 3-tier fallback: env var &rarr; secrets &rarr; default"),
        ],
        "poc_vs_prod": [
            ("Streamlit (Python, rapid prototyping)", "React/Next.js (TypeScript, production-grade SPA)"),
            ("pyodbc direct connection", "SQLAlchemy ORM + connection pooling"),
            ("@st.cache_data(ttl=3600)", "Redis/Memcached distributed cache"),
            ("Plotly (Python-native charts)", "D3.js / ECharts (full custom interactivity)"),
            ("Client-side filtering (Pandas)", "Server-side filtering (parameterized SQL)"),
            ("Single-server deployment", "Docker + Kubernetes with load balancing"),
        ],
        "scale": [
            ("4 Tabs", "Dashboard Pages", "each with distinct analytical perspective"),
            ("10+", "Chart Types", "professional Plotly visualizations"),
            ("~20", "SQL Queries", "cached, centralized, parameterized"),
        ],
    },

    # ----------------------------------------------------------
    "external_data": {
        "icon": "🔗",
        "title": "External Data Sources",
        "what_it_does": """
Validated **3 external APIs** for market and safety data enrichment, tested against 5 representative drugs across different MoA classes.

The sources provide:
- **NHS OpenPrescribing** &mdash; UK prescription volumes, costs, and trends (monthly, since Dec 2020)
- **FDA FAERS (openFDA)** &mdash; Adverse event reports, serious/non-serious split, temporal trends
- **CMS Medicare Part D** &mdash; US drug spending, claims, beneficiaries (annual, 2019-2023)
""",
        "how_it_works": [
            ("Step 1", "BNF-Code Mapping: <code>GET /bnf_code/?q={inn}</code> maps INN to NHS chemical code &mdash; works for all 5 test drugs"),
            ("Step 2", "NHS Spending: <code>GET /spending/?code={bnf_code}</code> returns monthly items, quantity, actual_cost"),
            ("Step 3", "FAERS AE Counts: <code>GET /drug/event.json?search=generic_name:&quot;{inn}&quot;&amp;count=reactionmeddrapt</code>"),
            ("Step 4", "FAERS Trends: <code>&amp;count=receivedate</code> for temporal analysis of adverse event reporting"),
            ("Step 5", "CMS API: <code>GET /data-api/v1/dataset/{id}/data?filter[Gnrc_Name]={INN}</code> for spending data"),
        ],
        "intelligence": [
            "<b>INN &rarr; BNF-Code Automation</b> &mdash; NHS OpenPrescribing API accepts INN search directly (no manual BNF lookup needed) &mdash; chemical-level code aggregates all formulations",
            "<b>generic_name vs. brand_name Strategy (FAERS)</b> &mdash; generic_name captures 100% of reports; brand_name sum covers only ~66% due to unmapped records &mdash; always use generic_name",
            "<b>Salt-Suffix Handling (CMS)</b> &mdash; CMS uses 'METFORMIN HCL' and 'SITAGLIPTIN PHOSPHATE' with salt suffixes &mdash; requires LIKE matching or suffix-aware lookup",
            "<b>Multiple Brands per Drug</b> &mdash; Semaglutide has 3 brands (Ozempic $9.19B + Rybelsus $1.67B + Wegovy $200K in Medicare) &mdash; aggregation at INN level needed",
            "<b>Serious Event Classification</b> &mdash; FAERS serious:1 filter shows ~48% of semaglutide reports are serious &mdash; different AE profile than non-serious (gastroparesis vs. nausea)",
            "<b>Wide-to-Long Format ETL</b> &mdash; CMS data has year-columns (Tot_Spndng_2019...2023) requiring pivot for time-series analysis",
        ],
        "metrics": [
            ("Drugs Validated", "5/5 per source", "semaglutide, metformin, empagliflozin, sitagliptin, tirzepatide"),
            ("NHS Date Range", "Dec 2020 &ndash; Nov 2025", "61 months of monthly prescription data"),
            ("FAERS Semaglutide Reports", "61,549", "with temporal trend data since 2013"),
            ("Ozempic Medicare Spending", "$9.19B (2023)", "2nd highest drug in Medicare Part D"),
            ("All APIs Free", "$0 cost", "no authentication required for any source"),
            ("CMS Years Available", "2019 &ndash; 2023", "5-year longitudinal spending data"),
        ],
        "challenges": [
            ("FAERS Brand vs. Generic gap (34%)", "Brand-name search (Ozempic+Wegovy+Rybelsus) captures only 40K of 61K reports &mdash; ~21K reports lack brand mapping &mdash; always use generic_name"),
            ("Tirzepatide only available from 2023", "NHS: first prescription data Feb 2023 (1 item!), then exponential growth to 264K items/month by Nov 2025"),
            ("CMS JavaScript-only portal", "data.cms.gov requires JavaScript rendering &mdash; solved by extracting API endpoint via browser DevTools: <code>/data-api/v1/dataset/{uuid}/data</code>"),
            ("Metformin generic naming", "CMS lists as 'METFORMIN HCL' (with salt suffix) &mdash; simple INN search returns empty &mdash; requires LIKE or brand_name filter"),
            ("FAERS rate limiting", "240 requests/min without API key, 120K/day with key &mdash; sufficient for validation but needs throttling for full ingest of 43 drugs"),
        ],
        "poc_vs_prod": [
            ("5 test drugs validated manually", "Automated pipeline for all 43 drugs"),
            ("Markdown validation report", "Live dashboard tabs with real-time API data"),
            ("One-time API calls", "Scheduled monthly refresh (NHS) + quarterly (FAERS, CMS)"),
            ("No database storage", "Dedicated tables: drug_adverse_events, drug_prescriptions_uk, drug_spending_us"),
            ("Manual BNF/CMS name mapping", "Automated mapping table with fuzzy matching fallback"),
        ],
        "scale": [
            ("3", "External APIs", "all free, no authentication"),
            ("5/5", "Test Drugs Found", "per data source"),
            ("$9.19B", "Ozempic Spending", "Medicare Part D 2023"),
        ],
    },

    # ----------------------------------------------------------
    "ml_predictions": {
        "icon": "🤖",
        "title": "ML Predictions",
        "what_it_does": """
Predicts the **Probability of Success (PoS)** for clinical trial phase transitions using XGBoost ensemble models trained on historical ClinicalTrials.gov data.

The pipeline covers:
- **3 phase transitions**: Phase I&rarr;II, Phase II&rarr;III, Phase III&rarr;Approval
- **9,625 trial-drug predictions** with calibrated probabilities
- **60 leak-free features** using strict point-in-time architecture
- **Drug-Level Temporal GroupKFold** cross-validation to prevent data leakage
""",
        "how_it_works": [
            ("Step 1", "Feature engineering (<code>compute_features_v2.py</code>): 60 features from trial metadata + drug history, all point-in-time safe (only data known BEFORE trial start date)"),
            ("Step 2", "Drug-Level Temporal GroupKFold: expanding-window CV with 5 splits &mdash; no drug appears in both train and test within any fold"),
            ("Step 3", "Train 4 model types per phase: LogReg Baseline, XGBoost A (depth=4), XGBoost B (depth=6), Ridge Meta-Learner (stacks A+B)"),
            ("Step 4", "Model selection via mean CV-AUC &mdash; best model per phase selected automatically (avoids test-set selection bias)"),
            ("Step 5", "Final models retrained on ALL data &mdash; CV metrics serve as unbiased performance estimate"),
            ("Step 6", "9,625 predictions written to Azure SQL (<code>ml_predictions</code>) with feature importance (<code>ml_feature_importance</code>)"),
        ],
        "intelligence": [
            "<b>Point-in-Time (PIT) Architecture</b> &mdash; every feature uses only information publicly known BEFORE the trial's start date: <code>WHERE t2.start_date < t.start_date</code>. This prevents temporal leakage where future data predicts past outcomes",
            "<b>Drug History Features</b> &mdash; prior trial count, prior max phase, prior approval status, prior completion rate &mdash; captures a drug's track record as the strongest predictive signal",
            "<b>Prior Approval Signal</b> &mdash; <code>feat_drug_has_prior_approval</code> is the #1 feature for Phase III&rarr;Approval (importance: 0.205). Drugs with existing approvals for other indications have higher success rates &mdash; this is a genuine, non-leaking signal",
            "<b>Automated Leakage Detection</b> &mdash; single-feature AUC check (threshold 0.90) runs before DB insert. Any feature with AUC &gt; 0.90 alone triggers an alert &mdash; prevents accidental target leakage",
            "<b>Drug-Level Grouping</b> &mdash; unlike trial-level splits, drug-level grouping ensures that a drug's Phase I trial in train doesn't leak information about the same drug's Phase II trial in test",
            "<b>Literature Benchmarks</b> &mdash; Lo et al. 2019 achieved AUC 0.78 on Phase III&rarr;Approval with public data. Our CV-AUC of 0.779&ndash;0.947 is consistent with published results",
        ],
        "metrics": [
            ("Phase I&rarr;II CV-AUC", "0.794 &pm; 0.036", "XGBoost A selected, 2,701 samples, 5-fold CV"),
            ("Phase II&rarr;III CV-AUC", "0.779 &pm; 0.065", "XGBoost A selected, 2,255 samples, 5-fold CV"),
            ("Phase III&rarr;Approval CV-AUC", "0.947 &pm; 0.046", "XGBoost A selected, 2,821 samples, 3-fold CV"),
            ("Total Predictions", "9,625", "across all 3 phase transitions"),
            ("Features (leak-free)", "60", "down from 105 in v1 after removing 15 post-market + 15 MoA features"),
            ("Drug Overlap in CV", "0 (verified)", "no drug appears in both train and test of any fold"),
        ],
        "challenges": [
            ("v1 data leakage (AUC 0.999)", "v1 models achieved suspiciously perfect AUC &mdash; root cause: post-market features (approval counts, Medicare spending) available only AFTER the outcome was known. v2 removes all 15 post-market features"),
            ("Meta-Learner degradation", "Ridge meta-learner on 2 nearly-identical XGBoost outputs performs WORSE than individual models (Phase III: 0.847 vs 0.947). Solved by CV-based model selection instead of always using meta"),
            ("Drug-level temporal splits", "Standard temporal splits allow the same drug in train and test &mdash; built custom GroupKFold that groups by drug_id with expanding time windows"),
            ("Small test sets in expanding window", "Phase II&rarr;III had only 68 test samples in some folds &mdash; mitigated by using CV-aggregated metrics (across all folds) as primary performance measure"),
            ("Azure SQL UNIQUEIDENTIFIER joins", "trial_id and drug_id are UUIDs &mdash; feature engineering requires careful JOIN handling and NULL-safe aggregations across 7 tables"),
        ],
        "poc_vs_prod": [
            ("XGBoost + LogReg baseline (4 models)", "Neural network ensemble + Bayesian optimization + AutoML"),
            ("60 features from ClinicalTrials.gov only", "200+ features including molecular descriptors, omics data, patent filings"),
            ("Drug-Level Temporal GroupKFold (5 splits)", "Nested CV with hyperparameter tuning in inner loop"),
            ("Manual feature engineering (Python)", "Automated feature stores (Feast, Tecton)"),
            ("Batch predictions (one-time run)", "Real-time prediction API with model versioning (MLflow Model Registry)"),
            ("Joblib model serialization", "ONNX / TensorRT for production inference"),
        ],
        "scale": [
            ("9,625", "Predictions", "trial-drug pairs scored across 3 transitions"),
            ("0.794", "Phase I&rarr;II AUC", "CV-validated, leak-free"),
            ("60", "Features", "all point-in-time safe"),
        ],
    },

    # ----------------------------------------------------------
    "decisions": {
        "icon": "🧠",
        "title": "Key Design Decisions",
        "what_it_does": """
Documents the strategic choices made during development and the reasoning behind each decision.

Every technical project involves trade-offs. This section explains **why** specific technologies, architectures, and approaches were chosen over alternatives &mdash; demonstrating not just what was built, but the engineering judgment behind it.
""",
        "how_it_works": [
            ("Decision Framework", "Each choice evaluated on: Cost, Speed-to-Deploy, Scalability, Maintainability, Portfolio Impact"),
            ("Trade-off Analysis", "Documented pros/cons for each alternative considered"),
            ("Iteration", "Several decisions were revised during development based on real-world data (e.g., phase mapping, sponsor derivation)"),
        ],
        "intelligence": [
            "<b>Azure SQL Serverless (Gen5, 1 vCore)</b> &mdash; auto-pauses after 1 hour idle, resumes in ~1 min. Cost: ~$5/month vs. $50+ for always-on. Trade-off: first query after pause takes 5-10 sec cold start",
            "<b>ClinicalTrials.gov API v2 (not v1)</b> &mdash; v2 returns structured JSON with nested fields, supports pagination tokens, better filter syntax. v1 returns XML and is being deprecated",
            "<b>Streamlit (not React/Vue)</b> &mdash; 10x faster prototyping for data dashboards, native Python, built-in caching. Trade-off: limited customization, not suitable for complex SPAs. Perfect for portfolio/PoC",
            "<b>Plotly (not Altair/Matplotlib)</b> &mdash; interactive charts with hover, zoom, pan out of the box. Altair is declarative but limited chart types. Matplotlib is static. Plotly balances interactivity + variety",
            "<b>pyodbc (not SQLAlchemy)</b> &mdash; direct ODBC connection, zero ORM overhead, exact SQL control. SQLAlchemy adds abstraction useful for large teams but unnecessary for single-developer PoC",
            "<b>Coverage-First Strategy</b> &mdash; achieved 78.8% Drug&rarr;Trial linking before building any dashboard. Data quality must precede visualization &mdash; a beautiful chart on bad data is worse than no chart",
            "<b>Monorepo Structure</b> &mdash; all code (ETL scripts, dashboard, utils) in one repository. Simplifies deployment, version control, and portfolio presentation vs. microservices",
        ],
        "metrics": [
            ("Azure SQL Cost", "~$5/month", "Serverless auto-pause eliminates idle costs"),
            ("Development Time", "~60 hours", "Phase 0-4: API validation + ETL + Dashboard + Data validation + ML"),
            ("External API Cost", "$0", "ClinicalTrials.gov, openFDA, NHS, CMS all free"),
            ("Lines of Code", "~4,500", "Dashboard (8 tabs) + ML pipeline (feature engineering + training)"),
            ("Data Freshness", "manual refresh", "PoC: on-demand. Prod: would be daily/weekly automated"),
        ],
        "challenges": [
            ("Cold start latency on Azure SQL Serverless", "First query after auto-pause takes 5-10 seconds &mdash; acceptable for portfolio demo, would need connection warming in production"),
            ("Streamlit state management limitations", "No persistent client-side state &mdash; every widget interaction triggers full rerun. Mitigated with aggressive caching (<code>@st.cache_data</code>)"),
            ("pyodbc connection stability", "Connections can drop after Azure SQL auto-pause &mdash; implemented reconnect logic on ProgrammingError with fresh <code>get_connection()</code> call"),
            ("Password with special characters in connection string", "Password contains # character &mdash; requires <code>{braces}</code> in ODBC connection string to escape properly"),
        ],
        "poc_vs_prod": [
            ("Azure SQL Serverless ($5/mo)", "Azure SQL Dedicated + Read Replicas ($200+/mo)"),
            ("Manual ETL scripts", "Azure Data Factory + Event-driven triggers"),
            ("Streamlit Community Cloud", "Azure App Service + Docker + CI/CD"),
            ("pyodbc + raw SQL", "SQLAlchemy + Alembic migrations + ORM"),
            ("Single developer", "Team with data engineer, frontend dev, DevOps"),
            ("Manual testing in browser", "pytest + Playwright E2E + unit tests"),
        ],
        "scale": [
            ("~$5/mo", "Infrastructure Cost", "Azure SQL Serverless auto-pause"),
            ("$0", "API Costs", "all data sources are free and open"),
            ("~60h", "Development Time", "Phase 0 through Phase 4"),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# Render Function — standardized layout for all sections
# ═══════════════════════════════════════════════════════════════

def render_section(key):
    data = SECTION_DATA[key]

    # --- Section 1: What It Does + How It Works ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What It Does")
        st.markdown(data["what_it_does"], unsafe_allow_html=True)
    with col2:
        st.markdown("#### How It Works")
        for label, text in data["how_it_works"]:
            st.markdown(
                f'<div class="pipeline-step"><b>{label}:</b> {text}</div>',
                unsafe_allow_html=True,
            )

    # --- Section 2: Domain Intelligence ---
    st.markdown(
        '<div class="section-head">🧠 Domain Intelligence Built In</div>',
        unsafe_allow_html=True,
    )
    for item in data["intelligence"]:
        st.markdown(
            f'<div class="intel-item">{item}</div>',
            unsafe_allow_html=True,
        )

    # --- Section 3: Key Metrics ---
    st.markdown(
        '<div class="section-head">📏 Key Metrics &amp; Outputs</div>',
        unsafe_allow_html=True,
    )
    table = "| Metric | Value | Context |\n"
    table += "|--------|-------|--------|\n"
    for metric, value, context in data["metrics"]:
        table += f"| {metric} | {value} | {context} |\n"
    st.markdown(table, unsafe_allow_html=True)

    # --- Section 4: Challenges Solved ---
    st.markdown(
        '<div class="section-head">🔧 Technical Challenges Solved</div>',
        unsafe_allow_html=True,
    )
    for title, desc in data["challenges"]:
        st.markdown(
            f'<div class="challenge-item"><b>{title}</b> &mdash; {desc}</div>',
            unsafe_allow_html=True,
        )

    # --- Section 5: PoC vs. Production ---
    st.markdown(
        '<div class="section-head">🔬 Proof of Concept vs. Production Grade</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "*This is a Proof of Concept built to demonstrate end-to-end data engineering and analytics skills. "
        "The comparison shows what was built vs. what a fully funded production solution would use "
        "— demonstrating awareness of the optimal approach.*"
    )
    table = "| Our PoC Approach | Production-Grade Alternative |\n"
    table += "|-----------------|-----------------------------|\n"
    for poc, prod in data["poc_vs_prod"]:
        table += f"| {poc} | {prod} |\n"
    st.markdown(table, unsafe_allow_html=True)

    # --- Section 6: Scale Cards ---
    st.markdown(
        '<div class="section-head">📊 Numbers at a Glance</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, (value, label, detail) in enumerate(data["scale"]):
        with cols[i]:
            st.markdown(f"""
            <div class="scale-card">
                <div class="scale-value">{value}</div>
                <div class="scale-label">{label}</div>
                <div class="scale-detail">{detail}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Page Routing
# ═══════════════════════════════════════════════════════════════

NAV_TO_KEY = {
    SECTIONS[1]: "data_engineering",
    SECTIONS[2]: "dashboard",
    SECTIONS[3]: "external_data",
    SECTIONS[4]: "ml_predictions",
    SECTIONS[5]: "decisions",
}

# --- Architecture Overview ---
if selected == SECTIONS[0]:
    st.markdown("""
    <div class="arch-grid">
        <div class="arch-box">
            <h4>Project Pipeline</h4>
            <p>The project follows a phased approach from data validation to interactive analytics:</p>
            <div class="pipeline-step"><b>Phase 0 &mdash; API Validation:</b> Tested ClinicalTrials.gov API v2 endpoints, pagination, field availability</div>
            <div class="pipeline-step"><b>Phase 1 &mdash; Data Engineering:</b> Azure SQL schema, bulk data load (32,811 trials), entity linking (Drug&rarr;Trial, Trial&rarr;Indication), coverage optimization (78.8%)</div>
            <div class="pipeline-step"><b>Phase 2 &mdash; Dashboard:</b> 4-tab Streamlit dashboard with Plotly charts, interactive filters, CSV export</div>
            <div class="pipeline-step"><b>Phase 3 &mdash; Data Validation:</b> 3 external APIs validated (NHS, FAERS, CMS) for market &amp; safety data enrichment</div>
            <div class="pipeline-step"><b>Phase 4 &mdash; ML Predictions:</b> Leak-free XGBoost models for phase transition probability &mdash; 60 PIT features, Drug-Level Temporal GroupKFold CV, 9,625 predictions</div>
        </div>
        <div class="arch-box">
            <h4>Tech Stack</h4>
            <p>
                <span class="tech-badge db">Azure SQL Serverless</span><br>
                <span class="tech-badge frontend">Streamlit</span>
                <span class="tech-badge frontend">Plotly</span><br>
                <span class="tech-badge api">ClinicalTrials.gov API v2</span><br>
                <span class="tech-badge api">openFDA FAERS API</span><br>
                <span class="tech-badge api">NHS OpenPrescribing API</span><br>
                <span class="tech-badge api">CMS Medicare Part D API</span><br>
                <span class="tech-badge python">Python 3.12</span>
                <span class="tech-badge python">pyodbc</span>
                <span class="tech-badge python">pandas</span><br>
                <span class="tech-badge" style="background: #fce7f3; color: #9d174d;">XGBoost</span>
                <span class="tech-badge" style="background: #fce7f3; color: #9d174d;">scikit-learn</span>
                <span class="tech-badge" style="background: #fce7f3; color: #9d174d;">MLflow</span>
            </p>
            <h4 style="margin-top: 1rem;">Focus Area</h4>
            <p style="font-size: 0.9rem;">
                Diabetes &bull; Obesity &bull; Metabolic Disease<br>
                43 Drugs &bull; 22 MoA Classes &bull; 13 Indications
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Data Model (10 Tables)")
    st.markdown("""
    | Table | Records | Description |
    |-------|---------|-------------|
    | `drugs` | 43 | INN, MoA class, modality, highest phase |
    | `trials` | 32,811 | NCT ID, phase, status, sponsor, enrollment, dates |
    | `companies` | 6,023 | Sponsor names with type classification |
    | `indications` | 13 | Standardized disease indication names |
    | `drug_trials` | ~34K | Many-to-many: which drugs are tested in which trials |
    | `trial_indications` | ~60K | Many-to-many: which trials target which indications |
    | `approvals` | ~200 | FDA approval records with dates and application numbers |
    | `ml_models` | 16 | Model metadata: CV metrics, selection, version (4 models x 4 transitions) |
    | `ml_predictions` | 9,625 | Trial-drug PoS predictions with model version |
    | `ml_feature_importance` | ~240 | XGBoost feature importance per phase transition |
    """)

    st.markdown("#### Dashboard Components")
    st.markdown("""
    | Tab | Key Features | Primary Charts |
    |-----|-------------|---------------|
    | Pipeline Overview | 43 drugs, 4 filters, CSV export | Phase distribution bar chart |
    | Competitive Landscape | MoA &times; Indication matrix | Heatmap, trend lines, hot/cold zones |
    | Trial Analytics | Status breakdown, sponsor ranking | Donuts, stacked area, termination rates |
    | Drug Deep Dive | Single-drug profile, FDA data | Drug card, trial scatter, competitive bar |
    | Market Intelligence | UK prescriptions, US Medicare spending | Trend lines, MoA comparison |
    | Safety Profile | FDA FAERS adverse events | Heatmaps, AE trends, class signals |
    | Patent & LOE | Patent landscape, loss of exclusivity | Timeline, exclusivity calendar |
    | ML Predictions | Phase transition PoS, feature importance | PoS ranking, SHAP bars, calibration |
    """)

    st.markdown("#### Total Project Economics")
    st.markdown("""
    <table>
        <tr><th></th><th>This PoC</th><th>Enterprise Alternative</th></tr>
        <tr><td><b>Infrastructure</b></td><td>~$5/month (Azure SQL Serverless)</td><td>$200+/month (dedicated DB + app server)</td></tr>
        <tr><td><b>Data Sources</b></td><td>$0 (all APIs free)</td><td>$10K+/year (Cortellis, GlobalData, Evaluate)</td></tr>
        <tr><td><b>Development</b></td><td>~60 hours (single developer)</td><td>3-6 months (team of 4-5)</td></tr>
        <tr><td><b>Hosting</b></td><td>Free (Streamlit Community Cloud)</td><td>$100+/month (cloud hosting + CDN)</td></tr>
        <tr><td><b>Total Year 1</b></td><td><b>~$60</b></td><td><b>$50K+</b></td></tr>
    </table>
    """, unsafe_allow_html=True)


# --- Section Pages ---
else:
    section_key = NAV_TO_KEY.get(selected)
    if section_key:
        render_section(section_key)


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem 0;">
    <b>Pharma Pipeline Intelligence</b> &mdash; Competitive Intelligence Dashboard for Diabetes &amp; Obesity<br>
    Powered by Azure SQL &bull; ClinicalTrials.gov &bull; openFDA &bull; NHS OpenPrescribing &bull; CMS Medicare<br>
    <br>
    <span style="font-size: 0.8rem; opacity: 0.7;">
        Built as a portfolio project demonstrating end-to-end data engineering &amp; analytics
    </span>
</div>
""", unsafe_allow_html=True)
