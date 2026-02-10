# Pharma Pipeline Intelligence — Statusbericht

**Datum:** 09. Februar 2026
**Projekt:** Competitive Intelligence Dashboard — Diabetes & Obesity Drug Pipeline
**Status:** Phase 3 abgeschlossen + Documentation Page fertig

---

## Was wurde gebaut

Ein vollstaendiges Data-Engineering- und Analytics-Projekt, das klinische Studiendaten aus oeffentlichen APIs in eine Azure SQL-Datenbank laedt, verknuepft und als interaktives Streamlit-Dashboard visualisiert.

**Fokus:** 43 Medikamente im Bereich Diabetes & Adipositas, 22 Wirkmechanismus-Klassen, 13 Indikationen.

---

## Projektphasen

### Phase 0 — API-Validierung
- ClinicalTrials.gov API v2 getestet (Endpunkte, Paginierung, Feldverfuegbarkeit)
- Ergebnis: API liefert strukturiertes JSON, 1.000 Studien pro Seite

### Phase 1 — Data Engineering
- **Azure SQL Serverless** aufgesetzt (West Europe, ~5 EUR/Monat)
- **7 normalisierte Tabellen** erstellt (drugs, trials, companies, indications, drug_trials, trial_indications, approvals)
- **32.811 klinische Studien** geladen aus ClinicalTrials.gov
- **43 Medikamente** mit INN-Namen, MoA-Klassen, Modalitaeten
- **6.023 Sponsoren** klassifiziert in 4 Typen (academic, biotech, big_pharma, government)
- **Entity Linking:** Drug-Trial-Coverage von 76,2% auf **78,8%** optimiert
- **13 Indikationen** aus Freitext-Condition-Feldern normalisiert

### Phase 2 — Dashboard
Streamlit-Dashboard mit 4 interaktiven Seiten:

| Seite | Funktion |
|-------|----------|
| **Pipeline Overview** | 43 Medikamente als filterbare Tabelle, Phase-Distribution-Chart |
| **Competitive Landscape** | MoA x Indication Heatmap, Trend-Linien, Hot/Cold Zones |
| **Trial Analytics** | Status-Donuts, Stacked-Area-Charts, Top-Sponsors, Termination Rates |
| **Drug Deep Dive** | Einzelnes Medikament mit Indikationen, Trial-Timeline, FDA-Approvals |

- 10+ Plotly-Chart-Typen
- 4 interaktive Filter (MoA, Indication, Phase, Sponsor Type)
- CSV-Export pro Tab
- ~20 SQL-Queries, alle gecacht (1h TTL)

### Phase 3 — Externe Datenquellen validiert
3 APIs erfolgreich gegen 5 Test-Medikamente validiert:

| Quelle | Daten | Status |
|--------|-------|--------|
| **NHS OpenPrescribing** | UK-Verschreibungen, monatlich seit Dez 2020 | 5/5 Drugs validiert |
| **FDA FAERS (openFDA)** | Nebenwirkungsberichte, Trends | 5/5 Drugs validiert |
| **CMS Medicare Part D** | US-Arzneimittelausgaben 2019-2023 | 5/5 Drugs validiert |

Alle APIs sind kostenlos und ohne Authentifizierung nutzbar.

### Documentation Page — "How It Works"
- Eigene Streamlit-Seite mit **Teal/Emerald Farbschema** (visuell abgegrenzt vom blauen Dashboard)
- 5 Sektionen: Architecture Overview, Data Engineering, Dashboard & Visualization, External Data Sources, Key Design Decisions
- Pro Sektion: What It Does, How It Works, Domain Intelligence, Key Metrics, Challenges Solved, PoC vs. Production, Numbers at a Glance
- Navigation in allen Seiten verlinkt

---

## Tech Stack

| Komponente | Technologie |
|------------|-------------|
| Datenbank | Azure SQL Serverless (Gen5, 1 vCore) |
| Backend | Python 3.12, pyodbc, pandas, requests |
| Frontend | Streamlit, Plotly |
| APIs | ClinicalTrials.gov v2, openFDA, NHS OpenPrescribing, CMS Medicare |
| Hosting | Streamlit Community Cloud (geplant) |

---

## Projektstruktur

```
pharma_pipeline/
  deploy_schema.py              # DB-Schema erstellen
  ingest_clinicaltrials.py      # ClinicalTrials.gov ETL
  ingest_chembl_drugs.py        # ChEMBL Drug Master laden
  ingest_nachtrag_drugs.py      # Fehlende Drugs nachtraeglich laden
  ingest_openfda_approvals.py   # FDA-Approvals laden
  phase1_*.py                   # Entity Linking Skripte
  coverage_quickwins.py         # Coverage-Optimierung
  *.md                          # Berichte (Loading, Completion, Coverage, Market/Safety)

  pharma-pipeline-dashboard/
    app.py                      # Hauptseite mit KPI-Uebersicht
    pages/
      0_How_It_Works.py         # Architektur- & Methodik-Doku (Teal Theme)
      1_Pipeline_Overview.py    # Medikamenten-Tabelle
      2_Competitive_Landscape.py # Heatmap & Trends
      3_Trial_Analytics.py      # Trial-Analyse
      4_Drug_Deep_Dive.py       # Einzelmedikament-Ansicht
    utils/
      db.py                     # DB-Verbindung
      queries.py                # ~20 SQL-Queries
      charts.py                 # 10+ Plotly-Charts
    requirements.txt
    Dockerfile
    README.md
```

---

## Kosten

| Posten | Kosten |
|--------|--------|
| Azure SQL Serverless | ~5 EUR/Monat (auto-pause) |
| Alle APIs | 0 EUR (kostenlos) |
| Hosting (Streamlit Cloud) | 0 EUR |
| **Gesamt Jahr 1** | **~60 EUR** |

Vergleich Enterprise-Loesung: >50.000 EUR/Jahr (Cortellis, GlobalData, dedizierte Infrastruktur)

---

## Naechste Schritte (optional)

- [ ] Externe API-Daten in die Datenbank laden (NHS, FAERS, CMS Tabellen)
- [ ] Deployment auf Streamlit Community Cloud
- [ ] Git Repository initialisieren
- [ ] README.md fuer GitHub aktualisieren
- [ ] Weitere Indikationen/Medikamente ergaenzen
