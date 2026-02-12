# Pharma Pipeline Intelligence

**Competitive intelligence platform for diabetes & obesity drug development — from raw clinical trial data to ML-powered success predictions.**

Track 43 drugs across 32,800+ clinical trials, predict phase transition probabilities with XGBoost, and explore the competitive landscape through a 9-page interactive dashboard.

**[Live Dashboard](https://pharma-pipeline-intelligence.streamlit.app/)**

---

## Why This Exists

Commercial pharma intelligence platforms (Cortellis, GlobalData, Evaluate) cost $50,000+/year and deliver static reports. This project builds the same analytical capability from public data sources for under $60/year total infrastructure cost.

**What it delivers:**
- Which drugs are competing in which indications — and where the white space is
- How likely a drug is to advance from Phase 2 to Phase 3 (or from Phase 3 to approval)
- Which sponsors have the best track record, and which trials are stalling
- Patent expiry timelines, safety signals, and pricing data across US and UK markets

---

## The Numbers

| Metric | Value |
|--------|-------|
| Clinical trials tracked | 32,811 |
| Drugs profiled | 43 across 22 MoA classes |
| Companies classified | 6,023 (big pharma, biotech, academic, government) |
| Drug-trial relationships | 6,184 |
| ML features engineered | 60 (all point-in-time safe) |
| Best model CV-AUC | 0.947 (Phase 3 to Approval) |
| External data sources | 8 APIs integrated |
| Dashboard pages | 9 interactive views |
| Infrastructure cost | ~$60/year (Azure SQL Serverless) |

---

## Dashboard

Nine specialized pages, each focused on a different analytical question:

| Page | What It Answers |
|------|----------------|
| **Pipeline Overview** | Which drugs are in which phase? Filter by MoA, indication, sponsor type |
| **Competitive Landscape** | MoA x Indication heatmap — where is it crowded, where is white space? |
| **Trial Analytics** | Trial starts over time, termination rates, stale trial detection |
| **Drug Deep Dive** | Single-drug profile: trial timeline, competitors, market data, safety |
| **Market Intelligence** | UK NHS prescriptions and US Medicare Part D spending side by side |
| **Safety Profile** | FDA FAERS adverse events, frequency rankings, class signal detection |
| **Patent & LOE** | Orange Book patents, exclusivity timelines, loss-of-exclusivity calendar |
| **ML Predictions** | Phase transition probabilities, feature importance, model calibration |
| **How It Works** | Full architecture documentation and methodology |

Built with Streamlit and Plotly (15+ chart types: heatmaps, treemaps, scatter, donut, stacked area, line, bar).

---

## ML Pipeline

### Predicting Phase Transitions

Four binary classification models predict the probability that a drug advances:

| Transition | CV-AUC | Drugs in Dataset |
|-----------|--------|-----------------|
| Phase 1 → Phase 2 | 0.779 | varies by split |
| Phase 2 → Phase 3 | 0.832 | varies by split |
| Phase 3 → Approval | 0.947 | varies by split |

### What Makes This Rigorous

- **Drug-level temporal GroupKFold** — No drug appears in both train and test. Earlier drugs train, later drugs test. This prevents the most common source of leakage in pharma ML.
- **60 point-in-time safe features** — Every feature only uses information available before the prediction point. No post-market data, no future-looking variables. Down from 105 in v1 after removing unsafe features.
- **CV-based model selection** — Models are selected by cross-validation performance, not test-set AUC. This avoids selection bias. Final models are retrained on all data.
- **Four model types per transition** — Logistic Regression baseline, XGBoost aggressive, XGBoost conservative, Ridge meta-learner. XGBoost A selected for all 4 transitions.

### Feature Categories

| Category | Examples |
|---------|---------|
| Trial design | Enrollment, arms, randomization, blinding, placebo control |
| Sponsor profile | Company type, prior trial count, historical completion rate |
| Drug history | Years since first trial, prior max phase reached, prior completion rate |
| Indication | Therapeutic area, competitive density |

---

## Data Engineering Pipeline

Eight pipeline phases, each building on the previous:

```
0. Schema          deploy_schema.py           → 13 tables, 15 indexes
       |
1. Ingestion       1_ingestion/               → ClinicalTrials.gov, ChEMBL, OpenFDA
       |
2. Enrichment      2_enrichment/              → RxNorm codes, trial design features
       |
3. Linking         3_linking/                 → Company resolution, drug-indication mapping
       |
4. Patent & LOE    4_patent_loe/              → Orange Book, Purple Book, LOE calendar
       |
5. Safety          5_safety/                  → FDA FAERS adverse events
       |
6. Pricing         6_pricing/                 → CMS Medicare Part D, NHS OpenPrescribing
       |
7. ML              pharma-pipeline-ml/        → Feature engineering → Model training
       |
8. Dashboard       pharma-pipeline-dashboard/ → 9-page Streamlit app
```

### Data Sources

| Source | What It Provides | Cost |
|--------|-----------------|------|
| ClinicalTrials.gov API v2 | Trial metadata, status, design, sponsors | Free |
| ChEMBL | Drug properties, mechanisms of action | Free |
| OpenFDA | FDA approvals, adverse event reports | Free |
| NHS OpenPrescribing | UK prescription volumes (monthly since 2020) | Free |
| CMS Medicare Part D | US drug spending 2019-2023 | Free |
| FDA Orange Book | Small molecule patents and exclusivity | Free |
| FDA Purple Book | Biologic regulatory exclusivity | Free |
| UMLS / RxNorm | Drug nomenclature standardization | Free (API key) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | Azure SQL Serverless (Gen5, auto-pause, ~$5/month) |
| Data Engineering | Python, pyodbc (direct SQL, zero ORM overhead) |
| ML | XGBoost, scikit-learn, MLflow |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud (free) |

---

## Project Structure

```
pharma-pipeline-intelligence/
├── 1_ingestion/                  # Phase 1: Load raw data from APIs
├── 2_enrichment/                 # Phase 2: RxNorm, trial design features
├── 3_linking/                    # Phase 3: Entity resolution & linking
├── 4_patent_loe/                 # Phase 4: Patent & exclusivity data
├── 5_safety/                     # Phase 5: FDA adverse events
├── 6_pricing/                    # Phase 6: US & UK drug pricing
├── pharma-pipeline-ml/           # Phase 7: Feature engineering & model training
│   ├── compute_features_v2.py
│   ├── train_models_v2.py
│   ├── run_diagnostics.py
│   └── config.py
├── pharma-pipeline-dashboard/    # Phase 8: Streamlit dashboard
│   ├── app.py
│   ├── pages/                    # 9 dashboard pages
│   └── utils/                    # DB queries, charts, helpers
├── reports/                      # Phase reports & documentation
├── deploy_schema.py              # Phase 0: Create database schema
├── db_config.py                  # Shared database configuration
└── schema_stats.sql              # Database overview
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- ODBC Driver 18 for SQL Server
- Azure SQL Database (or local SQL Server)

### Installation

```bash
git clone https://github.com/leelesemann-sys/pharma-pipeline-intelligence.git
cd pharma-pipeline-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r pharma-pipeline-dashboard/requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and set your Azure SQL connection string. The dashboard reads credentials from Streamlit secrets (`.streamlit/secrets.toml`).

### Run the Dashboard

```bash
streamlit run pharma-pipeline-dashboard/app.py
```

### Run the Pipeline

Execute scripts in phase order:

```bash
python deploy_schema.py
python 1_ingestion/ingest_clinicaltrials.py
python 1_ingestion/ingest_chembl_drugs.py
# ... continue through phases
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Azure SQL Serverless | Auto-pauses after 60 min idle, costs ~$5/month vs. $50+ for always-on |
| pyodbc over SQLAlchemy | Direct SQL gives full control over complex analytical queries, no ORM overhead |
| Drug-level temporal CV | Industry standard for pharma ML — prevents same drug appearing in train and test |
| Point-in-time features only | Eliminates data leakage, the #1 cause of inflated metrics in pharma prediction |
| CV-based model selection | Avoids selection bias from evaluating many models on a single test set |
| 1-hour SQL cache (TTL) | Dashboard stays responsive despite Azure auto-pause cold starts |

---

## License

This project is for portfolio and educational purposes.

---

Built with Azure SQL, XGBoost, Streamlit, and 8 public data APIs.
