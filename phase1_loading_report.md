# Phase 1: Data Loading Report
**Projekt:** Pharma Pipeline Intelligence (Diabetes & Obesity)
**Datum:** 2026-02-09
**Status:** ✅ ABGESCHLOSSEN

---

## Executive Summary

| Metrik | Wert |
|--------|------|
| **Trials geladen** | 32.811 |
| **Drugs im Master** | 31 |
| **Drug-Trial-Verknüpfungen** | 6.184 |
| **FDA Approval Records** | 1.219 |
| **Indikationen (Seed)** | 13 |
| **Tabellen erstellt** | 13 |
| **Indexes erstellt** | 15 |

---

## 1. Azure SQL Database

- **Server:** pharma-pipeline-sql.database.windows.net
- **Database:** pharma_pipeline_db
- **Tier:** General Purpose, Serverless Gen5, 1 vCore
- **Auto-Pause:** 60 Minuten Inaktivität
- **Region:** West Europe
- **13 Tabellen**, 15 Indexes, 3 Junction Tables

---

## 2. ClinicalTrials.gov Ingest

### Trials pro Condition (vor Deduplizierung)
| Condition | Trials |
|-----------|--------|
| Diabetes Mellitus, Type 2 | 11.373 |
| Obesity | 14.043 |
| Overweight | 13.235 |
| Metabolic Syndrome | 5.711 |
| Diabetes Mellitus, Type 1 | 4.436 |
| NASH (Steatohepatitis) | 1.736 |
| NAFLD | 1.657 |
| **Gesamt (dedupliziert)** | **32.811** |

### Breakdown nach Status
| Status | Count |
|--------|-------|
| completed | 20.584 |
| unknown | 4.138 |
| recruiting | 3.150 |
| terminated | 1.508 |
| active_not_recruiting | 1.145 |
| not_yet_recruiting | 1.136 |
| withdrawn | 774 |
| enrolling_by_invitation | 284 |
| suspended | 92 |

### Breakdown nach Phase
| Phase | Count |
|-------|-------|
| n/a (observational etc.) | 21.907 |
| Phase 2 | 2.570 |
| Phase 4 | 2.445 |
| Phase 3 | 2.434 |
| Phase 1 | 2.144 |
| Phase 1/2 | 608 |
| Phase 2/3 | 398 |
| Early Phase 1 | 305 |

### Breakdown nach Study Type
| Type | Count |
|------|-------|
| Interventional | 26.074 |
| Observational | 6.702 |
| Expanded Access | 35 |

### Top 10 Sponsors
| # | Sponsor | Trials |
|---|---------|--------|
| 1 | Novo Nordisk A/S | 920 |
| 2 | AstraZeneca | 411 |
| 3 | Eli Lilly and Company | 384 |
| 4 | Sanofi | 297 |
| 5 | NIDDK (NIH) | 234 |
| 6 | Merck Sharp & Dohme | 223 |
| 7 | University of Colorado | 216 |
| 8 | Mayo Clinic | 189 |
| 9 | Massachusetts General Hospital | 186 |
| 10 | Cairo University | 180 |

### Zeitraum
- **Ältester Trial:** 1957-01-01
- **Neuester Trial:** 2030-04-15 (geplant)
- **Letztes Update:** 2026-02-06

### Qualitätsindikatoren
- **Stale Trials** (>12 Monate kein Update + Completion Date überschritten): **1.125**
- **Trials mit Ergebnissen:** **4.555** (13,9%)

---

## 3. ChEMBL Drug Entity Master

### 31 Drugs geladen

| MoA-Klasse | Drugs | ChEMBL Match |
|------------|-------|--------------|
| GLP-1 Receptor Agonist | semaglutide, liraglutide, dulaglutide, exenatide | 2/4 |
| GIP/GLP-1 Dual Agonist | tirzepatide | 1/1 |
| SGLT2 Inhibitor | empagliflozin, dapagliflozin, canagliflozin, ertugliflozin | 4/4 |
| SGLT1/SGLT2 Inhibitor | sotagliflozin | 1/1 |
| DPP-4 Inhibitor | sitagliptin, linagliptin, saxagliptin, alogliptin, vildagliptin | 3/5 |
| Biguanide | metformin | 1/1 |
| TZD (PPAR-gamma) | pioglitazone, rosiglitazone | 2/2 |
| Insulin (div.) | glargine, lispro, aspart, degludec, icodec | 5/5 |
| Oral GLP-1 (Pipeline) | orforglipron, danuglipron | 2/2 |
| Triple/Dual Agonist (Pipeline) | retatrutide, survodutide, cagrilintide, pemvidutide | 4/4 |
| NASH | resmetirom, obeticholic acid | 1/2 |

**Nicht in ChEMBL gefunden (manuell eingepflegt):** semaglutide, exenatide, sitagliptin, linagliptin, obeticholic acid
→ API-Suche hat kein exaktes Match geliefert. Drugs wurden manuell mit bekannten Daten eingefügt.

### Drug-Trial-Verknüpfungen: 6.184

| # | Drug | Verknüpfte Trials |
|---|------|-------------------|
| 1 | metformin | 1.292 |
| 2 | semaglutide | 479 |
| 3 | insulin glargine | 477 |
| 4 | sitagliptin | 417 |
| 5 | insulin aspart | 408 |
| 6 | liraglutide | 396 |
| 7 | pioglitazone | 371 |
| 8 | dapagliflozin | 321 |
| 9 | empagliflozin | 257 |
| 10 | exenatide | 233 |

**Alle 31 Drugs** haben mindestens eine Trial-Verknüpfung.

---

## 4. openFDA Approval Records

**1.219 FDA Approval Records** für 21 zugelassene Drugs.

### Drugs mit FDA-Approvals (21)
alogliptin, canagliflozin, dapagliflozin, dulaglutide, empagliflozin, ertugliflozin, exenatide, insulin aspart, insulin degludec, insulin glargine, insulin lispro, linagliptin, liraglutide, metformin, pioglitazone, resmetirom, saxagliptin, semaglutide, sitagliptin, sotagliflozin, tirzepatide

### Drugs ohne FDA-Daten (10, erwartet)
| Drug | Grund |
|------|-------|
| cagrilintide | Pipeline (Phase 3) |
| danuglipron | Pipeline (Phase 3) |
| insulin icodec | Noch nicht FDA-approved |
| obeticholic acid | Zurückgezogen/abgelehnt |
| orforglipron | Pipeline (Phase 3) |
| pemvidutide | Pipeline (Phase 2) |
| retatrutide | Pipeline (Phase 3) |
| rosiglitazone | Ältere API-Daten fehlen |
| survodutide | Pipeline (Phase 3) |
| vildagliptin | Nicht in USA zugelassen |

---

## 5. Datenbank-Übersicht

| Tabelle | Rows | Status |
|---------|------|--------|
| trials | 32.811 | ✅ Geladen |
| drugs | 31 | ✅ Geladen |
| drug_trials | 6.184 | ✅ Geladen |
| indications | 13 | ✅ Seeded |
| approvals | 1.219 | ✅ Geladen |
| companies | 0 | ⏳ Phase 2 |
| drug_indications | 0 | ⏳ Phase 2 |
| trial_indications | 0 | ⏳ Phase 2 |
| patents | 0 | ⏳ Phase 2 (Orange Book) |
| market_data | 0 | ⏳ Phase 5 |
| predictions | 0 | ⏳ Phase 3 |
| nlp_signals | 0 | ⏳ Phase 4 |
| alerts | 0 | ⏳ Phase 6 |

---

## 6. Nächste Schritte (Phase 2 Vorbereitung)

1. **Company Resolution:** Sponsor-Names aus Trials → Companies-Tabelle normalisieren
2. **Trial-Indications Linking:** `raw_conditions` aus Trials → `trial_indications` Junction
3. **Drug-Indications Linking:** Basierend auf Approval-Daten + Trial-Daten
4. **Orange Book Parsing:** Patent/Exclusivity-Daten für zugelassene Drugs
5. **UMLS Crosswalks:** Sobald API-Key da ist → RxNorm CUI, MeSH, MedDRA ergänzen
6. **Stale Trial Dashboard:** 1.125 stale Trials identifiziert → Phase 2 Visualisierung

---

## Dateien

| Datei | Beschreibung |
|-------|-------------|
| `deploy_schema.py` | Schema-Deployment-Skript |
| `ingest_clinicaltrials.py` | ClinicalTrials.gov Full Ingest |
| `ingest_chembl_drugs.py` | ChEMBL Drug Master + Trial Linking |
| `ingest_openfda_approvals.py` | openFDA Approval Data |
| `schema_stats.sql` | SQL-Queries für DB-Statistiken |
| `phase1_loading_report.md` | Dieser Report |

---

*Generiert: 2026-02-09 | Phase 1: Data Model & Entity Resolution*
