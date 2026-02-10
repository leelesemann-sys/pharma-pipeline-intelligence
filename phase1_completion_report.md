# Phase 1 Completion Report: Entity Linking & Company Resolution
**Projekt:** Pharma Pipeline Intelligence (Diabetes & Obesity)
**Datum:** 2026-02-09
**Status:** PHASE 1 VOLLSTAENDIG ABGESCHLOSSEN

---

## 1. Executive Summary

| Linking-Typ | Coverage | Links |
|-------------|----------|-------|
| Trial -> Indication | **76,2%** (25.006 / 32.811) | 29.762 |
| Drug -> Indication | **100%** (43 / 43) | 238 |
| Trial -> Company | **100%** (32.811 / 32.811) | 32.811 |
| Fully Linked Drugs (Drug->Trial->Ind->Company) | **100%** (43 / 43) | - |

### Datenbank-Uebersicht

| Tabelle | Rows | Status |
|---------|------|--------|
| trials | 32.811 | Geladen |
| drugs | 43 | Geladen |
| drug_trials | 6.766 | Geladen |
| indications | 13 | Geladen |
| trial_indications | 29.762 | NEU |
| drug_indications | 238 | NEU |
| approvals | 1.678 | Geladen |
| companies | 6.023 | NEU |
| patents | 0 | Phase 2 |
| market_data | 0 | Phase 5 |
| predictions | 0 | Phase 3 |
| nlp_signals | 0 | Phase 4 |
| alerts | 0 | Phase 6 |

---

## 2. Trial-Indication Breakdown

| Indication | Trials | % of Total |
|-----------|--------|------------|
| Obesity | 10.945 | 33,4% |
| Type 2 Diabetes Mellitus | 8.713 | 26,6% |
| Type 1 Diabetes Mellitus | 3.261 | 9,9% |
| Overweight | 2.196 | 6,7% |
| Metabolic Syndrome | 1.701 | 5,2% |
| Non-alcoholic Fatty Liver Disease | 1.116 | 3,4% |
| Prediabetes | 685 | 2,1% |
| NASH / MASH | 491 | 1,5% |
| Diabetic Kidney Disease | 227 | 0,7% |
| Gestational Diabetes | 188 | 0,6% |
| Diabetic Neuropathy | 154 | 0,5% |
| Diabetic Retinopathy | 85 | 0,3% |

**23,8% nicht zugeordnet:** Trials mit Co-Conditions wie "Cardiovascular Disease", "Hypertension", "Heart Failure" etc. die ausserhalb unseres Scope liegen. Diese Trials sind trotzdem in der DB (z.B. CVOTs fuer Diabetes-Drugs).

### Top Unmatched Conditions (fuer spaetere Scope-Erweiterung)
- Diabetes/Diabetes Mellitus (ohne Type): 2.316x - Edge Case, oft T2DM
- Hypertension: 797x
- Insulin Resistance: 693x
- Cardiovascular Diseases: 746x
- Polycystic Ovary Syndrome: 259x

---

## 3. Drug-Indication Matrix

### Approved Drugs (mit mindestens 1 approved Indication)

| Drug | T2DM | T1DM | Obesity | NASH | Weitere |
|------|------|------|---------|------|---------|
| semaglutide | P4 | P4 | P4 | P4 | DKD, MetSyn, Prediab |
| tirzepatide | V P4 | P4 | V P4 | P4 | NAFLD, Overweight |
| liraglutide | V P4 | P4 | V P4 | P4 | GDM, DKD, MetSyn |
| empagliflozin | P4 | P4 | P4 | P4 | DKD, NAFLD, MetSyn |
| dapagliflozin | P4 | P4 | P4 | P3 | DKD, NAFLD, MetSyn |
| metformin | P4 | P4 | P4 | P4 | Alle 12 Indikationen |
| insulin glargine | V P4 | V P4 | P4 | P2 | GDM, NAFLD |
| sitagliptin | P4 | P4 | P4 | P4 | DKD, MetSyn |

*V = approved, P1-P4 = hoechste investigational Phase*

### Pipeline Drugs (nur investigational)

| Drug | MoA | Key Indications | Hoechste Phase |
|------|-----|-----------------|----------------|
| retatrutide | GLP-1/GIP/Glucagon Triple | T2DM, Obesity | Phase 3 |
| survodutide | GLP-1/Glucagon Dual | T2DM, Obesity, NASH | Phase 3 |
| orforglipron | Oral GLP-1 (Small Mol.) | T2DM, Obesity | Phase 3 |
| cagrilintide | Amylin Analogue | T2DM, Obesity | Phase 3 |
| pemvidutide | GLP-1/Glucagon Dual | Obesity, NASH, NAFLD | Phase 2 |
| danuglipron | Oral GLP-1 (Small Mol.) | T2DM, Obesity | Phase 2 |

---

## 4. Company Breakdown

### Companies by Type
| Type | Count | Trials |
|------|-------|--------|
| Academic | 2.280 | 18.627 |
| Biotech/Industry | 3.590 | 9.468 |
| Big Pharma | 17 | 3.565 |
| Government | 136 | 1.151 |

### Top 20 Companies by Trial Count

| # | Company | Type | Trials |
|---|---------|------|--------|
| 1 | Novo Nordisk | Big Pharma | 920 |
| 2 | AstraZeneca | Big Pharma | 412 |
| 3 | Eli Lilly | Big Pharma | 384 |
| 4 | Sanofi | Big Pharma | 315 |
| 5 | NIDDK (NIH) | Government | 234 |
| 6 | Merck (MSD) | Big Pharma | 226 |
| 7 | Univ. of Colorado | Academic | 216 |
| 8 | Mayo Clinic | Academic | 189 |
| 9 | GSK | Big Pharma | 188 |
| 10 | Novartis | Big Pharma | 186 |
| 11 | Mass. General Hospital | Academic | 186 |
| 12 | Cairo University | Academic | 180 |
| 13 | Pfizer | Big Pharma | 178 |
| 14 | Boehringer Ingelheim | Big Pharma | 171 |
| 15 | Yale University | Academic | 164 |
| 16 | AP-HP Paris | Academic | 163 |
| 17 | Univ. of Aarhus | Academic | 153 |
| 18 | Washington Univ. | Academic | 153 |
| 19 | Maastricht UMC | Academic | 151 |
| 20 | Assiut University | Academic | 149 |

---

## 5. Competitive Landscape Preview (Top 20)

| MoA Class | Indication | Phase | Drugs | Trials |
|-----------|-----------|-------|-------|--------|
| Biguanide | T2DM | Phase 3 | 1 | 247 |
| DPP-4 Inhibitor | T2DM | Phase 3 | 5 | 235 |
| Biguanide | T2DM | Phase 4 | 1 | 171 |
| GLP-1 RA | T2DM | Phase 3 | 4 | 169 |
| Insulin (Basal) | T2DM | Phase 3 | 1 | 129 |
| DPP-4 Inhibitor | T2DM | Phase 4 | 5 | 122 |
| Insulin (Basal) | T2DM | Phase 4 | 1 | 111 |
| SGLT2 Inhibitor | T2DM | Phase 4 | 4 | 106 |
| SGLT2 Inhibitor | T2DM | Phase 3 | 5 | 102 |
| GLP-1 RA | T2DM | Phase 4 | 4 | 96 |
| GLP-1 RA | T2DM | Phase 1 | 4 | 96 |
| Biguanide | T2DM | Phase 1 | 1 | 86 |
| Insulin (Rapid) | T1DM | Phase 1 | 2 | 84 |
| Insulin (Rapid) | T2DM | Phase 4 | 2 | 78 |
| Insulin (Rapid) | T2DM | Phase 3 | 2 | 76 |
| Sulfonylharnstoff | T2DM | Phase 3 | 4 | 74 |
| TZD (PPAR-gamma) | T2DM | Phase 3 | 2 | 74 |
| Sulfonylharnstoff | T2DM | Phase 4 | 4 | 72 |
| Biguanide | T2DM | Phase 2 | 1 | 69 |
| TZD (PPAR-gamma) | T2DM | Phase 4 | 2 | 66 |

---

## 6. Data Quality Issues & Empfehlungen

### Bekannte Luecken
1. **Trial-Indication Gap (23,8%):** Trials mit nur generischen Conditions ("Diabetes", "Diabetes Mellitus"). Loesung: Heuristische Zuordnung basierend auf Co-Conditions und Interventions.
2. **ChEMBL-Luecken:** 5 Drugs ohne ChEMBL-ID (semaglutide, exenatide, sitagliptin, linagliptin, obeticholic acid). API-Suche lieferte kein exaktes Match. Manuell nachpflegen moeglich.
3. **Patent-Daten:** Noch leer. Orange Book Download + Parsing in Phase 2.
4. **UMLS Crosswalks:** Noch ausstehend (API-Key beantragt). RxNorm, MedDRA, SNOMED-Felder sind NULL.

### Datenqualitaet
- **Trial-Daten:** Hohe Qualitaet. Alle Felder aus ClinicalTrials.gov vorhanden.
- **Drug-Daten:** 26/43 Drugs mit ChEMBL ATC-Codes. Gute MoA/Target-Abdeckung.
- **Approval-Daten:** 1.678 FDA Records fuer 21 Drugs. EU/EMA fehlt noch.
- **Company-Daten:** 100% Trial-Coverage. Big Pharma korrekt normalisiert.

### Empfehlungen fuer Phase 2
1. Stale Trial Dashboard bauen (1.125 Trials identifiziert)
2. Orange Book Patent-Daten laden
3. Competitive Density Heatmap (MoA x Indication)
4. Pipeline Status Dashboard
5. UMLS-Integration sobald Key da ist

---

## 7. Dateien

| Datei | Beschreibung |
|-------|-------------|
| `phase1_trial_indication_linking.py` | Aufgabe 1: Trial->Indication |
| `phase1_drug_indication_linking.py` | Aufgabe 2: Drug->Indication |
| `phase1_company_resolution.py` | Aufgabe 3: Company Resolution |
| `phase1_completion_report.md` | Dieser Report |

---

*Phase 1: Data Model & Entity Resolution - ABGESCHLOSSEN*
*Naechster Schritt: Phase 2 - Pipeline Tracking & Competitive Intelligence*
