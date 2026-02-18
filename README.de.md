# Pharma Pipeline Intelligence

> **Sprache:** [English](README.md) | Deutsch

**Competitive-Intelligence-Plattform für die Arzneimittelentwicklung in Diabetes & Adipositas — von Rohdaten klinischer Studien bis zu ML-gestützten Erfolgsprognosen.**

Verfolgen Sie 43 Wirkstoffe über 32.800+ klinische Studien, prognostizieren Sie Phasenübergangswahrscheinlichkeiten mit XGBoost und erkunden Sie die Wettbewerbslandschaft über ein interaktives Dashboard mit 9 Seiten.

**[Live-Dashboard](https://pharma-pipeline-intelligence.streamlit.app/)**

---

## Warum dieses Projekt

Kommerzielle Pharma-Intelligence-Plattformen (Cortellis, GlobalData, Evaluate) kosten über 50.000 $/Jahr und liefern statische Reports. Dieses Projekt baut dieselbe analytische Fähigkeit aus öffentlichen Datenquellen auf — für unter 60 $/Jahr Gesamtinfrastrukturkosten.

**Was es liefert:**
- Welche Wirkstoffe in welchen Indikationen konkurrieren — und wo Lücken bestehen
- Wie wahrscheinlich es ist, dass ein Wirkstoff von Phase 2 zu Phase 3 vorrückt (oder von Phase 3 zur Zulassung)
- Welche Sponsoren die beste Erfolgsbilanz haben und welche Studien stagnieren
- Patentablauf-Zeitpläne, Sicherheitssignale und Preisdaten aus US- und UK-Märkten

---

## Die Zahlen

| Kennzahl | Wert |
|----------|------|
| Verfolgte klinische Studien | 32.811 |
| Profilierte Wirkstoffe | 43 über 22 MoA-Klassen |
| Klassifizierte Unternehmen | 6.023 (Big Pharma, Biotech, Akademisch, Staatlich) |
| Wirkstoff-Studien-Verknüpfungen | 6.184 |
| Engineered ML-Features | 60 (alle Point-in-Time-sicher) |
| Bester Modell-CV-AUC | 0,947 (Phase 3 zur Zulassung) |
| Externe Datenquellen | 8 integrierte APIs |
| Dashboard-Seiten | 9 interaktive Ansichten |
| Infrastrukturkosten | ~60 $/Jahr (Azure SQL Serverless) |

---

## Dashboard

Neun spezialisierte Seiten, jede auf eine andere analytische Fragestellung ausgerichtet:

| Seite | Welche Frage sie beantwortet |
|-------|------------------------------|
| **Pipeline Overview** | Welche Wirkstoffe befinden sich in welcher Phase? Filter nach MoA, Indikation, Sponsortyp |
| **Competitive Landscape** | MoA x Indikation-Heatmap — wo ist es überfüllt, wo gibt es Lücken? |
| **Trial Analytics** | Studienstarts im Zeitverlauf, Abbruchraten, Erkennung stagnierender Studien |
| **Drug Deep Dive** | Einzelwirkstoff-Profil: Studien-Timeline, Wettbewerber, Marktdaten, Sicherheit |
| **Market Intelligence** | UK-NHS-Verordnungen und US-Medicare-Part-D-Ausgaben im Vergleich |
| **Safety Profile** | FDA FAERS Nebenwirkungen, Häufigkeitsrankings, Signaldetektion auf Klassenebene |
| **Patent & LOE** | Orange-Book-Patente, Exklusivitäts-Timelines, Loss-of-Exclusivity-Kalender |
| **ML Predictions** | Phasenübergangswahrscheinlichkeiten, Feature Importance, Modellkalibrierung |
| **How It Works** | Vollständige Architekturdokumentation und Methodik |

Gebaut mit Streamlit und Plotly (15+ Diagrammtypen: Heatmaps, Treemaps, Scatter, Donut, Stacked Area, Line, Bar).

---

## ML-Pipeline

### Vorhersage von Phasenübergängen

Vier binäre Klassifikationsmodelle prognostizieren die Wahrscheinlichkeit, dass ein Wirkstoff vorrückt:

| Übergang | CV-AUC | Wirkstoffe im Datensatz |
|----------|--------|-------------------------|
| Phase 1 → Phase 2 | 0,779 | variiert nach Split |
| Phase 2 → Phase 3 | 0,832 | variiert nach Split |
| Phase 3 → Zulassung | 0,947 | variiert nach Split |

### Was die Methodik robust macht

- **Drug-level Temporal GroupKFold** — Kein Wirkstoff erscheint sowohl im Trainings- als auch im Testset. Frühere Wirkstoffe trainieren, spätere testen. Dies verhindert die häufigste Quelle von Data Leakage in Pharma-ML.
- **60 Point-in-Time-sichere Features** — Jedes Feature nutzt nur Informationen, die vor dem Vorhersagezeitpunkt verfügbar waren. Keine Post-Market-Daten, keine zukunftsgerichteten Variablen. Reduziert von 105 in v1 nach Entfernung unsicherer Features.
- **CV-basierte Modellauswahl** — Modelle werden nach Cross-Validation-Performance ausgewählt, nicht nach Test-Set-AUC. Dies vermeidet Selection Bias. Finale Modelle werden auf allen Daten neu trainiert.
- **Vier Modelltypen pro Übergang** — Logistic Regression Baseline, XGBoost Aggressiv, XGBoost Konservativ, Ridge Meta-Learner. XGBoost A für alle 4 Übergänge ausgewählt.

### Feature-Kategorien

| Kategorie | Beispiele |
|-----------|-----------|
| Studiendesign | Enrollment, Arme, Randomisierung, Verblindung, Placebokontrolle |
| Sponsorprofil | Unternehmenstyp, bisherige Studienanzahl, historische Abschlussrate |
| Wirkstoffhistorie | Jahre seit erster Studie, bisherige höchste Phase, bisherige Abschlussrate |
| Indikation | Therapeutisches Gebiet, Wettbewerbsdichte |

---

## Data-Engineering-Pipeline

Acht Pipeline-Phasen, jede aufbauend auf der vorherigen:

```
0. Schema          deploy_schema.py           → 13 Tabellen, 15 Indizes
       |
1. Ingestion       1_ingestion/               → ClinicalTrials.gov, ChEMBL, OpenFDA
       |
2. Enrichment      2_enrichment/              → RxNorm-Codes, Studiendesign-Features
       |
3. Linking         3_linking/                 → Unternehmensauflösung, Wirkstoff-Indikation-Mapping
       |
4. Patent & LOE    4_patent_loe/              → Orange Book, Purple Book, LOE-Kalender
       |
5. Safety          5_safety/                  → FDA FAERS Nebenwirkungen
       |
6. Pricing         6_pricing/                 → CMS Medicare Part D, NHS OpenPrescribing
       |
7. ML              pharma-pipeline-ml/        → Feature Engineering → Modelltraining
       |
8. Dashboard       pharma-pipeline-dashboard/ → 9-seitiges Streamlit-Dashboard
```

### Datenquellen

| Quelle | Was sie liefert | Kosten |
|--------|-----------------|--------|
| ClinicalTrials.gov API v2 | Studien-Metadaten, Status, Design, Sponsoren | Kostenlos |
| ChEMBL | Wirkstoffeigenschaften, Wirkmechanismen | Kostenlos |
| OpenFDA | FDA-Zulassungen, Nebenwirkungsberichte | Kostenlos |
| NHS OpenPrescribing | UK-Verordnungsvolumina (monatlich seit 2020) | Kostenlos |
| CMS Medicare Part D | US-Arzneimittelausgaben 2019–2023 | Kostenlos |
| FDA Orange Book | Kleinmolekül-Patente und Exklusivität | Kostenlos |
| FDA Purple Book | Biologika-Regulierungsexklusivität | Kostenlos |
| UMLS / RxNorm | Standardisierung der Wirkstoffnomenklatur | Kostenlos (API Key) |

---

## Tech Stack

| Schicht | Technologie |
|---------|-------------|
| Datenbank | Azure SQL Serverless (Gen5, Auto-Pause, ~5 $/Monat) |
| Data Engineering | Python, pyodbc (direktes SQL, kein ORM-Overhead) |
| ML | XGBoost, scikit-learn, MLflow |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud (kostenlos) |

---

## Projektstruktur

```
pharma-pipeline-intelligence/
├── 1_ingestion/                  # Phase 1: Rohdaten von APIs laden
├── 2_enrichment/                 # Phase 2: RxNorm, Studiendesign-Features
├── 3_linking/                    # Phase 3: Entity Resolution & Linking
├── 4_patent_loe/                 # Phase 4: Patent- & Exklusivitätsdaten
├── 5_safety/                     # Phase 5: FDA-Nebenwirkungen
├── 6_pricing/                    # Phase 6: US- & UK-Arzneimittelpreise
├── pharma-pipeline-ml/           # Phase 7: Feature Engineering & Modelltraining
│   ├── compute_features_v2.py
│   ├── train_models_v2.py
│   ├── run_diagnostics.py
│   └── config.py
├── pharma-pipeline-dashboard/    # Phase 8: Streamlit Dashboard
│   ├── app.py
│   ├── pages/                    # 9 Dashboard-Seiten
│   └── utils/                    # DB-Abfragen, Diagramme, Hilfsfunktionen
├── reports/                      # Phasenberichte & Dokumentation
├── deploy_schema.py              # Phase 0: Datenbankschema erstellen
├── db_config.py                  # Gemeinsame Datenbankkonfiguration
└── schema_stats.sql              # Datenbankübersicht
```

---

## Lokale Einrichtung

### Voraussetzungen

- Python 3.11+
- ODBC Driver 18 for SQL Server
- Azure SQL Database (oder lokaler SQL Server)

### Installation

```bash
git clone https://github.com/leelesemann-sys/pharma-pipeline-intelligence.git
cd pharma-pipeline-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r pharma-pipeline-dashboard/requirements.txt
```

### Konfiguration

Kopieren Sie `.env.example` nach `.env` und tragen Sie Ihren Azure SQL Connection String ein. Das Dashboard liest die Zugangsdaten aus Streamlit Secrets (`.streamlit/secrets.toml`).

### Dashboard starten

```bash
streamlit run pharma-pipeline-dashboard/app.py
```

### Pipeline ausführen

Skripte in Phasenreihenfolge ausführen:

```bash
python deploy_schema.py
python 1_ingestion/ingest_clinicaltrials.py
python 1_ingestion/ingest_chembl_drugs.py
# ... weitere Phasen fortsetzen
```

---

## Zentrale Designentscheidungen

| Entscheidung | Begründung |
|-------------|------------|
| Azure SQL Serverless | Pausiert automatisch nach 60 Min. Inaktivität, kostet ~5 $/Monat statt 50+ $ für Always-On |
| pyodbc statt SQLAlchemy | Direktes SQL gibt volle Kontrolle über komplexe analytische Abfragen, kein ORM-Overhead |
| Drug-level Temporal CV | Industriestandard für Pharma-ML — verhindert, dass derselbe Wirkstoff in Train und Test erscheint |
| Nur Point-in-Time-Features | Eliminiert Data Leakage, die häufigste Ursache überhöhter Metriken in Pharma-Vorhersagen |
| CV-basierte Modellauswahl | Vermeidet Selection Bias durch Evaluation vieler Modelle auf einem einzelnen Testset |
| 1-Stunden-SQL-Cache (TTL) | Dashboard bleibt responsiv trotz Azure Auto-Pause Cold Starts |

---

## Literaturverweise & Benchmarks

Methodik, Feature Engineering und Evaluationsstrategie dieses Projekts basieren auf veröffentlichter Pharma-ML-Forschung:

### ML-Benchmarks

| Referenz | Relevanz für dieses Projekt |
|----------|----------------------------|
| **Lo, Siah & Wong (2019)** — *Machine Learning in Clinical Trial Outcome Prediction.* Harvard Data Science Review. | Primärer Benchmark. 140+ Features, Random Forest auf öffentlichen Daten. AUC 0,78 bei Phase III zur Zulassung. Unser CV-AUC von 0,779–0,947 ist konsistent mit ihren Ergebnissen bei einem ähnlichen Public-Data-Ansatz. |
| **Wong, Siah & Lo (2019)** — *Estimation of Clinical Trial Success Rates and Related Parameters.* Biostatistics, 20(2), 273-286. | Veröffentlichte Probability-of-Success (PoS)-Tabellen nach Phase und therapeutischem Gebiet — der Standard-Benchmark für die Validierung von Übergangsraten-Vorhersagen. Zur Kalibrierung unserer Modell-Outputs verwendet. |
| **Hay, Thomas, Craighead, Economides & Rosenthal (2014)** — *Clinical Development Success Rates for Investigational Drugs.* Nature Biotechnology, 32(1), 40-51. | Historische Basisraten: Phase 1 zu 2 (~63 %), Phase 2 zu 3 (~31 %), Phase 3 zur Zulassung (~58 %). Als Plausibilitätsprüfung für die Übergangsraten und Modellvorhersagen unseres Datensatzes verwendet. |
| **Thomas, Burns, Audette, Carroll, Dow-Hygelund & Hay (2016)** — *Clinical Development Success Rates 2006-2015.* BIO Industry Analysis. | Aktualisierte Erfolgsraten mit größerem Datensatz. Bestätigt Phase 2 als den risikoreichsten Übergang. Hat den Fokus unseres Feature Engineering auf Phase-2-Prädiktoren beeinflusst. |
| **DiMasi, Grabowski & Hansen (2016)** — *Innovation in the Pharmaceutical Industry: New Estimates of R&D Costs.* Journal of Health Economics, 47, 20-33. | F&E-Kostenschätzungen (2,6 Mrd. $ pro zugelassenem Medikament), die kontextualisieren, warum die Vorhersage von Studienerfolg enormen wirtschaftlichen Wert hat. |

### Methodik-Referenzen

| Referenz | Wie wir sie angewendet haben |
|----------|------------------------------|
| **Drug-level Temporal Splitting** (Lo et al. 2019) | Wir verwenden GroupKFold mit Wirkstoff-Gruppen und temporaler Ordnung — denselben Ansatz, den Lo et al. als essenziell zur Vermeidung von Data Leakage in Pharma-Vorhersagen validiert haben. |
| **Point-in-Time Feature Safety** (Lo et al. 2019) | Alle 60 Features verwenden strikte temporale Cutoffs. Features wie „Sponsor-Abschlussrate" nutzen expandierende Fenster bis zum (aber nicht einschließlich) Vorhersagedatum. |
| **Probability of Success Kalibrierung** (Wong et al. 2019) | Unsere vorhergesagten Wahrscheinlichkeiten werden mit den veröffentlichten PoS-Tabellen von Wong et al. verglichen, um die Kalibrierung über therapeutische Gebiete hinweg zu verifizieren. |
| **Basisraten-Validierung** (Hay et al. 2014) | Übergangsraten des Datensatzes werden gegen die veröffentlichten Erfolgsraten von Hay et al. geprüft, um sicherzustellen, dass unser ClinicalTrials.gov-Datensatz repräsentativ ist. |

---

## Lizenz

Dieses Projekt dient Portfolio- und Bildungszwecken.

---

Gebaut mit Azure SQL, XGBoost, Streamlit und 8 öffentlichen Daten-APIs.
