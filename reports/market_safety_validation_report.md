# Market & Safety Data Validation Report

**Datum:** 2026-02-09
**Scope:** 3 Datenquellen x 5 Test-Drugs (Semaglutide, Metformin, Empagliflozin, Sitagliptin, Tirzepatide)

---

## Zusammenfassung

| Quelle | Status | Verfügbar für X/5 Drugs | Datenqualität | Blocker? |
|--------|--------|------------------------|---------------|----------|
| NHS OpenPrescribing | ✅ | 5/5 | Gut | Keiner |
| FDA FAERS | ✅ | 5/5 | Gut (mit Caveats) | Keiner |
| CMS Medicare Part D | ✅ | 5/5 | Gut | Keiner |

**Gesamtbewertung:** Alle 3 Quellen sind API-basiert zugänglich, kostenlos, und liefern Daten für alle 5 Test-Drugs. Ready für Full Ingest.

---

## TEST 1: NHS OpenPrescribing

### 1a) BNF-Code Mapping

| Drug | BNF-Code | Ansatz | Formulations |
|------|----------|--------|-------------|
| Semaglutide | `0601023AW` | INN-Suche direkt ✅ | 10 (SC-Injektionen 0.25mg-2.4mg + Tablets 3mg/7mg/14mg) |
| Metformin | `0601022B0` | INN-Suche direkt ✅ | 28+ (Tabletten, Modified-Release, Oral Solution) |
| Empagliflozin | `0601023AN` | INN-Suche direkt ✅ | 2 (10mg + 25mg Tablets) |
| Sitagliptin | `0601023X0` | INN-Suche direkt ✅ | 5 (25mg/50mg/100mg Tablets + Oral Solutions) |
| Tirzepatide | `0601023AZ` | INN-Suche direkt ✅ | 8 (Injektionen 2.5mg-15mg) |

**Ergebnis:** INN-Suche funktioniert für alle 5 Drugs. Kein Brand-Name-Mapping nötig.
Alle unter BNF Section `0601` (Drugs used in diabetes).

**Automatisierbarkeit:** Hoch. API-Endpoint `GET /bnf_code/?q={drug_name}&format=json` liefert Chemical-Code (8-stellig) als Aggregations-Level. Für den Full Ingest reicht der Chemical-Code-Level (`0601023AW` = alle Semaglutide-Formulierungen).

### 1b) Verschreibungsdaten

| Metrik | Beschreibung |
|--------|-------------|
| `items` | Anzahl verschriebene Items (Rezepte) |
| `quantity` | Gesamtmenge (Einheiten) |
| `actual_cost` | Tatsächliche Kosten (£) |
| `date` | Monatliches Datum |
| `row_name` | Geografische Einheit (default: "england") |

**Zeitraum:** Dezember 2020 – November 2025 (61 Monate, ~5 Jahre)
**Granularität:** Monatlich, national (England). Auch pro Practice/CCG verfügbar.
**API:** `GET /spending/?code={BNF_CODE}&format=json` — keine Authentifizierung nötig.

### Verschreibungsdaten pro Drug (Dezember 2020 vs. November 2025)

| Drug | Items Dez 2020 | Items Nov 2025 | Wachstum | Kosten Nov 2025 |
|------|---------------|----------------|----------|----------------|
| Semaglutide | 37,858 | 122,276 | **+223%** | £10.8M/Monat |
| Metformin | 2,030,112 | 2,185,097 | +8% | £4.6M/Monat |
| Empagliflozin | 168,805 | 387,884 | **+130%** | £14.6M/Monat |
| Sitagliptin | 205,261 | 247,793 | +21% | £0.7M/Monat |
| Tirzepatide | n/a (ab Feb 2023) | 264,345 | **Exponentiell** | £60.1M/Monat |

### 1c) Semaglutide Rx-Trend (letzte 12 Monate)

| Monat | Items | Actual Cost (£) |
|-------|-------|----------------|
| Dez 2024 | ~125,000 | ~£11.3M |
| Jan 2025 | ~119,000 | ~£10.7M |
| Feb 2025 | ~117,000 | ~£10.5M |
| Mär 2025 | ~124,000 | ~£11.1M |
| ... | ... | ... |
| Sep 2025 | 129,022 | £11.5M |
| Okt 2025 | 132,520 | £11.8M |
| Nov 2025 | 122,276 | £10.8M |

**Trend:** Stabil-steigend um ~125K items/Monat mit saisonaler Variation. +223% Wachstum seit Dez 2020.

### Bemerkenswerte Findings

- **Tirzepatide:** Erst ab Feb 2023 Daten (1 Item!), dann explosives Wachstum auf 264K items im Nov 2025. Okt 2025: £63.2M — teuerster Drug in der Kohorte.
- **Sitagliptin:** Kosten drastisch gefallen (£6.9M → £0.7M) trotz steigender Items — Generika-Effekt.
- **Metformin:** 2.2M items/Monat — mit Abstand meistverordnet, aber niedrigste Kosten pro Item.

### Empfehlung für Full Ingest

✅ **Ready.** API ist stabil, gut dokumentiert, keine Auth nötig.
- Ingest-Strategie: Chemical-Code-Level pro Drug, monatlich, national
- Automatisierung: BNF-Code über `GET /bnf_code/?q={inn}`, dann `GET /spending/?code={bnf_code}`
- Mapping-Tabelle `drug_bnf_codes` in DB anlegen
- Geschätzter Aufwand: 2-3h

---

## TEST 2: FDA FAERS (openFDA)

### 2a) Adverse Events pro Drug

| Drug | Total Reports | Top 5 Adverse Events |
|------|--------------|---------------------|
| **Semaglutide** | **61,549** | Nausea (11,506), Vomiting (7,479), Off Label Use (6,855), Diarrhoea (6,516), Decreased Appetite (4,790) |
| **Metformin** | **215,593** | Nausea (29,316), Blood Glucose Increased (27,460), Diarrhoea (27,324), Drug Ineffective (22,203), Fatigue (20,905) |
| **Empagliflozin** | ~30,000* | Diabetic Ketoacidosis (4,089), Nausea (3,515), Blood Glucose Increased (3,409), Diarrhoea (3,005), Weight Decreased (2,999) |
| **Sitagliptin** | ~40,000* | Blood Glucose Increased (5,854), Nausea (4,779), Diarrhoea (4,713), Drug Ineffective (4,365), Fatigue (3,694) |
| **Tirzepatide** | ~80,000* | Incorrect Dose Administered (25,919), Injection Site Pain (12,325), Nausea (12,028), Extra Dose Administered (8,027), Off Label Use (7,726) |

*\* Total geschätzt aus Summe Top-10 (meta.results.total nicht immer direkt zurückgegeben)*

**API-Suche:** INN-Suche über `generic_name` funktioniert für alle 5 Drugs. Kein Brand-Name-Mapping nötig.

### 2b) Zeitlicher Trend (Semaglutide)

- **Erste Daten:** 19. Juni 2013 (1 Report)
- **Trend:** Stark steigend seit 2021, Spikes in Q1-Q2 2022 (931 bzw. 631 Reports/Tag)
- **Korrelation:** Passt zu steigenden Verschreibungszahlen (Ozempic-Boom)
- **Achtung:** FAERS `receivedate` liefert tägliche Counts — müssen für Quartalstrend aggregiert werden

### 2c) Namens-Normalisierung (Semaglutide)

| Suchfeld | Term | Reports |
|----------|------|---------|
| `generic_name` | semaglutide | 61,549 |
| `brand_name` | OZEMPIC | 30,160 |
| `brand_name` | WEGOVY | 7,214 |
| `brand_name` | RYBELSUS | 2,929 |
| **Summe Brand** | | **40,303** |

**Gap-Analyse:** `generic_name` liefert 61,549 Reports, Summe der 3 Brands nur 40,303.
→ ~21,000 Reports (34%) haben keinen Brand-Name oder verwenden andere Schreibweisen.
→ **Empfehlung:** `generic_name`-Suche verwenden, NICHT Brand-basiert (erfasst alle Reports).
→ Keine Deduplizierung nötig — die generic_name-Suche ist bereits die Obermenge.

### 2d) Serious vs. Non-Serious Reports (Semaglutide)

| Kategorie | Geschätzte Reports | Anteil |
|-----------|--------------------|--------|
| Serious (`serious:1`) | ~29,620* | ~48% |
| Non-Serious (`serious:2`) | ~31,929* | ~52% |

*\* Geschätzt aus Summe der Top-AE-Counts*

**Serious Top-Events:** Impaired Gastric Emptying, Dehydration, Abdominal Pain — klinisch relevanter als Non-Serious Top-Events (primär Nausea, Off Label Use).

### Bemerkenswerte Findings

- **Tirzepatide** hat auffällig viele "Incorrect Dose Administered" (25,919) und "Extra Dose Administered" (8,027) — deutet auf Usability-Probleme des Pen-Devices hin
- **Empagliflozin:** Diabetic Ketoacidosis als #1 AE — bekanntes SGLT2-Klassenrisiko
- **Metformin:** Lactic Acidosis (18,480 Reports) — bekanntes seltenes Risiko, durch Reporting-Volume überrepräsentiert
- **Reporting Bias:** Neuere Drugs (Semaglutide, Tirzepatide) haben proportional mehr Reports ("Stimulated Reporting" bei medialer Aufmerksamkeit)

### Empfehlung für Full Ingest

✅ **Ready.** API ist frei zugänglich, stabil.
- Ingest-Strategie: `generic_name`-Suche pro Drug, Top-20 AEs + Total Count + Serious/Non-Serious Split
- Zeitlicher Trend: `count=receivedate` für zeitliche Aggregation
- Rate Limit beachten: 240 req/min ohne Key, 120K/Tag mit Key
- Mapping-Tabelle `drug_faers_mapping` mit INN-zu-generic_name Mapping
- Geschätzter Aufwand: 3-4h (inkl. Dedup-Logik-Entscheidung)

---

## TEST 3: CMS Medicare Part D

### 3a) Aktuellster Datensatz

| Eigenschaft | Wert |
|-------------|------|
| **Neuestes Jahr** | **2023** |
| **Update-Frequenz** | Jährlich |
| **Dataset-Name** | Medicare Part D Spending by Drug |
| **API-Endpoint** | `https://data.cms.gov/data-api/v1/dataset/7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b/data` |
| **Format** | JSON (API) + CSV (Download) |
| **Auth** | Keine (Open Data) |
| **Zeitraum im Dataset** | 2019–2023 (5 Jahre, wide-format) |

### 3b) Felder im Dataset

| Feld | Beschreibung |
|------|-------------|
| `Brnd_Name` | Markenname (z.B. "Ozempic", "Jardiance") |
| `Gnrc_Name` | Generischer Name (z.B. "Semaglutide", "Empagliflozin") |
| `Tot_Mftr` | Hersteller-Aggregation ("Overall" oder spezifisch) |
| `Mftr_Name` | Hersteller (z.B. "Novo Nordisk", "Eli Lilly & Co") |
| `Tot_Spndng_{YYYY}` | Gesamtausgaben ($) pro Jahr |
| `Tot_Dsg_Unts_{YYYY}` | Dosierungseinheiten pro Jahr |
| `Tot_Clms_{YYYY}` | Gesamtansprüche (Claims) pro Jahr |
| `Tot_Benes_{YYYY}` | Begünstigte (Patienten) pro Jahr |
| `Avg_Spnd_Per_Dsg_Unt_Wghtd_{YYYY}` | Durchschnittliche Kosten pro Einheit |
| `Avg_Spnd_Per_Clm_{YYYY}` | Durchschnittliche Kosten pro Claim |
| `Avg_Spnd_Per_Bene_{YYYY}` | Durchschnittliche Kosten pro Patient |
| `Outlier_Flag_{YYYY}` | Outlier-Markierung |
| `Chg_Avg_Spnd_Per_Dsg_Unt_22_23` | Änderung Durchschnittskosten 2022→2023 |
| `CAGR_Avg_Spnd_Per_Dsg_Unt_19_23` | CAGR Durchschnittskosten 2019→2023 |

### 3c) Daten für 5 Test-Drugs (2023)

| Drug | Brand | Gnrc_Name in CMS | Tot_Spndng_2023 | Tot_Clms_2023 | Tot_Benes_2023 |
|------|-------|-------------------|-----------------|---------------|----------------|
| **Semaglutide (SC)** | Ozempic | SEMAGLUTIDE | **$9.19B** | 6,930,000 | 1,460,000 |
| **Semaglutide (oral)** | Rybelsus | SEMAGLUTIDE | **$1.67B** | 1,080,000 | 285,700 |
| **Semaglutide (SC)** | Wegovy | SEMAGLUTIDE | $199.8K* | 142 | 47 |
| **Metformin** | Metformin HCl | METFORMIN HCL | **$247.5M** | 23,871,472 | 5,918,026 |
| **Empagliflozin** | Jardiance | EMPAGLIFLOZIN | **$8.84B** | 8,153,238 | 1,882,768 |
| **Sitagliptin** | Januvia | SITAGLIPTIN PHOSPHATE | **$4.09B** | 4,029,094 | 843,391 |
| **Tirzepatide** | Mounjaro | TIRZEPATIDE | **$2.36B** | 1,821,486 | 370,203 |

*\* Wegovy minimal in Medicare Part D (primär Obesity-Indikation, weniger Medicare-Coverage)*

### Drug-Name-Matching Strategie

| Drug (INN) | CMS Gnrc_Name | Matching |
|------------|---------------|---------|
| semaglutide | SEMAGLUTIDE | Exakt (UPPER) ✅ |
| metformin | METFORMIN HCL | Suffix "HCL" → LIKE-Match nötig ✅ |
| empagliflozin | EMPAGLIFLOZIN | Exakt (UPPER) ✅ |
| sitagliptin | SITAGLIPTIN PHOSPHATE | Suffix "PHOSPHATE" → LIKE-Match nötig ✅ |
| tirzepatide | TIRZEPATIDE | Exakt (UPPER) ✅ |

**Normalisierung:** 3/5 Drugs matchen exakt (case-insensitive). 2/5 brauchen Salt-Suffix-Handling (`UPPER(inn) LIKE gnrc_name || '%'` oder Reverse).

### Bemerkenswerte Findings

- **Ozempic** ($9.19B): Zweitteuerster Drug in Medicare Part D überhaupt
- **Jardiance** ($8.84B): Extrem hohes Volumen — 8.15M Claims
- **Metformin** ($247.5M): 23.9M Claims zeigen enorme Verschreibungsbreite bei niedrigen Kosten
- **Wegovy** minimal: Medicare Part D deckt Obesity-Medikamente kaum → Daten nicht repräsentativ für Gesamtmarkt
- **Tirzepatide/Mounjaro**: Bereits $2.36B trotz kurzem Verfügbarkeitszeitraum (Zulassung Mai 2022)
- **Wide-Format:** Daten sind als 2019-2023 Spalten organisiert (nicht long-format) → ETL muss pivotieren

### Empfehlung für Full Ingest

✅ **Ready.** API ist stabil, JSON-Response direkt query-bar.
- Ingest-Strategie: `GET /data?filter[Gnrc_Name]={UPPER(inn)}` pro Drug, dann Jahr-Spalten zu Long-Format pivotieren
- Salt-Suffix-Handling für Metformin HCl, Sitagliptin Phosphate etc.
- Mehrere Brands pro Drug möglich (Ozempic + Rybelsus + Wegovy)
- Aggregation auf Drug-Level (Summe aller Brands/Hersteller mit `Tot_Mftr=Overall`)
- Geschätzter Aufwand: 3-4h

---

## Gesamtempfehlung

### Alle 3 Quellen sind ready für Full Ingest ✅

| Quelle | Datentyp | Aufwand | Priorität |
|--------|----------|---------|-----------|
| **FDA FAERS** | Safety / Adverse Events | 3-4h | 🔴 Hoch (Kern-Feature) |
| **NHS OpenPrescribing** | UK Prescribing Trends | 2-3h | 🟡 Mittel (EU-Perspektive) |
| **CMS Medicare Part D** | US Market Spending | 3-4h | 🔴 Hoch (Marktdaten) |

### Vorarbeit nötig

1. **DB-Schema erweitern:**
   - `drug_adverse_events` (drug_id, event_term, count, serious_flag, report_year)
   - `drug_prescriptions_uk` (drug_id, bnf_code, date, items, quantity, actual_cost)
   - `drug_spending_us` (drug_id, brand_name, year, total_spending, total_claims, total_beneficiaries, avg_cost_per_unit)
   - `drug_external_ids` (drug_id, source, external_id) — BNF-Codes, CMS-GenericNames

2. **Mapping-Tabellen:**
   - INN → BNF-Code (automatisierbar via API)
   - INN → CMS Gnrc_Name (mit Salt-Suffix-Handling)
   - INN → FAERS generic_name (case-insensitive, direkt)

3. **ETL-Pipelines:**
   - NHS: Monatlicher Batch, 43 Drugs × 1 API-Call = 43 Calls
   - FAERS: Top-20 AEs + Total + Serious Split pro Drug = ~130 Calls
   - CMS: 1 Call pro Drug (mit Pivot) = 43 Calls

### Geschätzter Gesamtaufwand Phase 3 Full Ingest: **8-12h**

### Reihenfolge

1. Schema-Migration + Mapping-Tabellen (1-2h)
2. FDA FAERS Ingest (3-4h) — höchster Dashboard-Mehrwert
3. CMS Medicare Part D Ingest (3-4h) — Marktdaten
4. NHS OpenPrescribing Ingest (2-3h) — UK-Ergänzung
5. Dashboard-Erweiterung: Safety Tab + Market Tab (4-6h)
