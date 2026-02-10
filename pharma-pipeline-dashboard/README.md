# Pharma Pipeline Intelligence Dashboard

**Competitive Intelligence Dashboard for Diabetes & Obesity Drug Development**

A professional Streamlit dashboard providing real-time insights into the pharmaceutical pipeline for diabetes and obesity therapeutics. Built with data from ClinicalTrials.gov, ChEMBL, and openFDA.

## Features

### Tab 1: Pipeline Overview
- KPI metrics (43 drugs, 32K+ trials, 9 MoA classes)
- Interactive, filterable drug pipeline table
- Phase distribution visualization
- CSV export

### Tab 2: Competitive Landscape
- **MoA x Indication heatmap** - identify crowded vs. white-space areas
- Trial starts trend by MoA class (2010-2026)
- Hot zones (growing) and cold zones (declining)

### Tab 3: Trial Analytics
- Trial status and phase breakdowns (donut charts)
- Trial starts over time (stacked area chart)
- Top 20 sponsors analysis
- Termination rate analysis by MoA, indication, and phase
- Stale trials monitor with CT.gov links

### Tab 4: Drug Deep Dive
- Detailed drug card with all metadata
- Indication status (approved vs. investigational)
- Trial timeline scatter plot
- Full trial table with links
- FDA approval history
- Competitive position vs. same MoA class drugs

## Tech Stack
- **Frontend:** Streamlit + Plotly
- **Backend:** Azure SQL Database (Serverless Gen5)
- **Data Sources:** ClinicalTrials.gov API v2, ChEMBL, openFDA

## Local Setup

### Prerequisites
- Python 3.10+
- ODBC Driver 18 for SQL Server
- Access to the Azure SQL database

### Installation
```bash
cd pharma-pipeline-dashboard
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### Configuration
The database connection can be configured via:
1. **Environment variable:** `AZURE_SQL_CONN_STR`
2. **Streamlit secrets:** `.streamlit/secrets.toml` (not in Git)
3. **Default:** placeholder (requires one of the above to be set)

## Deployment

### Streamlit Community Cloud
1. Push code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app, select repository
4. Configure secrets (AZURE_SQL_CONN_STR)
5. Deploy

**Note:** Ensure `packages.txt` is included for ODBC driver installation on Linux.

### Azure App Service
```bash
docker build -t pharma-pipeline-dashboard .
docker run -p 8501:8501 -e AZURE_SQL_CONN_STR="..." pharma-pipeline-dashboard
```

### Azure SQL Firewall
For cloud deployment, enable "Allow Azure services and resources to access this server" in the Azure SQL firewall settings.

## Project Structure
```
pharma-pipeline-dashboard/
    app.py                          # Main app with sidebar navigation
    pages/
        1_Pipeline_Overview.py      # Drug pipeline table + charts
        2_Competitive_Landscape.py  # Heatmap + trends + hot/cold zones
        3_Trial_Analytics.py        # Status, sponsors, termination, stale
        4_Drug_Deep_Dive.py         # Single drug detail view
    utils/
        db.py                       # Cached DB connection
        queries.py                  # All SQL queries as functions
        charts.py                   # Reusable Plotly chart components
    .streamlit/
        config.toml                 # Theme configuration
    requirements.txt
    packages.txt                    # System packages for Streamlit Cloud
    Dockerfile                      # For Azure App Service deployment
    README.md
```

## Data Coverage
- **32,811 clinical trials** (78.8% with indication mapping)
- **43 drugs** across 9 MoA classes
- **13 indications** (T2DM, T1DM, Obesity, NASH, NAFLD, etc.)
- **6,023 companies** (big pharma, biotech, academic, government)
- **1,678 FDA approval records**
