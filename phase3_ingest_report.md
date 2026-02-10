# Phase 3 Market & Safety Data Ingest Report

## Zusammenfassung

| Quelle | Drugs mit Daten | Records geladen | Zeitraum |
|--------|----------------|-----------------|----------|
| NHS OpenPrescribing | 30/43 | 1,632 Records | 2020-12-01 bis 2025-11-01 |
| FDA FAERS | 32/43 | 800 AE + 2,141 Trend Records | 2001-04-01 bis 2025-10-01 |
| CMS Medicare Part D | 31/43 | 391 Records | 2019–2023 |

---

## NHS OpenPrescribing

**Drugs mit Daten (30):** acarbose, alogliptin, bromocriptine, canagliflozin, colesevelam, dapagliflozin, dulaglutide, empagliflozin, ertugliflozin, exenatide, gliclazide, glimepiride, glipizide, insulin aspart, insulin degludec, insulin glargine, insulin lispro, linagliptin, liraglutide, metformin, nateglinide, obeticholic acid, pioglitazone, repaglinide, rosiglitazone, saxagliptin, semaglutide, sitagliptin, tirzepatide, vildagliptin

**Drugs ohne Daten (13):** bexagliflozin, cagrilintide, danuglipron, glyburide, insulin icodec, miglitol, orforglipron, pemvidutide, pramlintide, resmetirom, retatrutide, sotagliflozin, survodutide

**Top 10 by Items (aktuellster Monat):**

| Drug | Items | Cost (GBP) | Date |
|------|-------|-----------|------|
| dapagliflozin | 943,640 | 11,302,215.91 | 2025-11-01 |
| gliclazide | 462,776 | 770,576.54 | 2025-11-01 |
| empagliflozin | 387,884 | 14,581,704.77 | 2025-11-01 |
| tirzepatide | 264,345 | 60,108,836.56 | 2025-11-01 |
| insulin glargine | 165,344 | 7,318,521.46 | 2025-11-01 |
| semaglutide | 122,276 | 10,826,567.33 | 2025-11-01 |
| alogliptin | 86,403 | 2,269,113.65 | 2025-11-01 |
| canagliflozin | 66,637 | 2,634,518.47 | 2025-11-01 |
| insulin degludec | 57,594 | 3,412,732.16 | 2025-11-01 |
| insulin aspart | 48,817 | 2,332,971.61 | 2025-11-01 |

---

## FDA FAERS

**Drugs mit Daten (32):** acarbose, alogliptin, bexagliflozin, bromocriptine, canagliflozin, colesevelam, dapagliflozin, dulaglutide, empagliflozin, ertugliflozin, exenatide, glimepiride, glipizide, glyburide, insulin aspart, insulin degludec, insulin glargine, insulin lispro, linagliptin, liraglutide, metformin, miglitol, nateglinide, pioglitazone, pramlintide, repaglinide, resmetirom, saxagliptin, semaglutide, sitagliptin, sotagliflozin, tirzepatide

**Drugs ohne Daten (11):** cagrilintide, danuglipron, gliclazide, insulin icodec, obeticholic acid, orforglipron, pemvidutide, retatrutide, rosiglitazone, survodutide, vildagliptin

**Top 15 AEs gesamt:**

| Drug | Event Term | Total Count | Serious Count |
|------|-----------|------------|--------------|
| insulin glargine | BLOOD GLUCOSE INCREASED | 38,628 | 13,339 |
| insulin lispro | BLOOD GLUCOSE INCREASED | 38,590 | 10,448 |
| metformin | NAUSEA | 29,316 | 16,091 |
| metformin | BLOOD GLUCOSE INCREASED | 27,460 | 10,333 |
| metformin | DIARRHOEA | 27,324 | 16,610 |
| tirzepatide | INCORRECT DOSE ADMINISTERED | 25,919 | N/A |
| metformin | DRUG INEFFECTIVE | 22,203 | 10,711 |
| metformin | FATIGUE | 20,905 | 13,190 |
| metformin | VOMITING | 18,844 | 13,665 |
| metformin | LACTIC ACIDOSIS | 18,480 | 18,435 |
| metformin | ACUTE KIDNEY INJURY | 17,530 | 17,505 |
| metformin | WEIGHT DECREASED | 17,270 | 9,492 |
| metformin | DYSPNOEA | 16,261 | 13,324 |
| exenatide | BLOOD GLUCOSE INCREASED | 15,473 | 1,745 |
| metformin | DIZZINESS | 15,449 | 9,783 |

**Klassen-Signale (Top 15 MoA-Level AEs):**

| MoA Class | Event Term | Class Total |
|-----------|-----------|------------|
| Insulin (Rapid-acting) | BLOOD GLUCOSE INCREASED | 49,021 |
| GLP-1 Receptor Agonist | NAUSEA | 40,650 |
| Insulin (Basal) | BLOOD GLUCOSE INCREASED | 38,628 |
| GLP-1 Receptor Agonist | BLOOD GLUCOSE INCREASED | 32,354 |
| Biguanide | NAUSEA | 29,316 |
| Biguanide | BLOOD GLUCOSE INCREASED | 27,460 |
| Biguanide | DIARRHOEA | 27,324 |
| GIP/GLP-1 Dual Agonist | INCORRECT DOSE ADMINISTERED | 25,919 |
| Biguanide | DRUG INEFFECTIVE | 22,203 |
| GLP-1 Receptor Agonist | WEIGHT DECREASED | 21,832 |
| Biguanide | FATIGUE | 20,905 |
| GLP-1 Receptor Agonist | VOMITING | 20,684 |
| Biguanide | VOMITING | 18,844 |
| Biguanide | LACTIC ACIDOSIS | 18,480 |
| GLP-1 Receptor Agonist | DIARRHOEA | 18,341 |

---

## CMS Medicare Part D

**Drugs mit Daten (31):** acarbose, alogliptin, bromocriptine, canagliflozin, colesevelam, dapagliflozin, dulaglutide, empagliflozin, ertugliflozin, exenatide, glimepiride, glipizide, glyburide, insulin aspart, insulin degludec, insulin glargine, insulin lispro, linagliptin, liraglutide, metformin, miglitol, nateglinide, obeticholic acid, pioglitazone, pramlintide, repaglinide, saxagliptin, semaglutide, sitagliptin, sotagliflozin, tirzepatide

**Drugs ohne Daten (12):** bexagliflozin, cagrilintide, danuglipron, gliclazide, insulin icodec, orforglipron, pemvidutide, resmetirom, retatrutide, rosiglitazone, survodutide, vildagliptin

**Top 10 by 2023 Spending:**

| Drug | Brand | Spending (USD) | Claims | Beneficiaries |
|------|-------|---------------|--------|---------------|
| semaglutide | Ozempic | $9,194,048,435.10 | 6,927,972 | 1,464,468 |
| empagliflozin | Jardiance | $8,839,935,063.30 | 8,153,238 | 1,882,768 |
| dulaglutide | Trulicity | $7,363,856,224.30 | 5,316,020 | 938,731 |
| dapagliflozin | Farxiga | $4,342,182,307.30 | 4,298,827 | 993,909 |
| sitagliptin | Januvia | $4,090,836,820.70 | 4,029,094 | 843,391 |
| insulin glargine | Lantus Solostar | $3,157,233,281.80 | 4,936,965 | 1,198,294 |
| tirzepatide | Mounjaro | $2,361,384,157.10 | 1,821,486 | 370,203 |
| insulin aspart | Novolog Flexpen | $1,871,873,174.20 | 2,089,784 | 579,795 |
| semaglutide | Rybelsus | $1,665,906,943.30 | 1,075,026 | 285,693 |
| linagliptin | Tradjenta | $1,293,567,778.30 | 1,496,546 | 286,808 |

**GLP-1 Spending Trend 2019–2023:**

| Drug | Year | Spending (USD) |
|------|------|---------------|
| dulaglutide | 2019 | $2,272,876,283.70 |
| dulaglutide | 2020 | $3,284,873,061.90 |
| dulaglutide | 2021 | $4,702,174,723.60 |
| dulaglutide | 2022 | $6,225,291,667.60 |
| dulaglutide | 2023 | $7,363,856,224.30 |
| exenatide | 2019 | $61,401,577.13 |
| exenatide | 2020 | $47,676,631.52 |
| exenatide | 2021 | $37,988,662.49 |
| exenatide | 2022 | $27,500,961.16 |
| exenatide | 2023 | $20,645,626.17 |
| liraglutide | 2019 | $1,894,032,418.20 |
| liraglutide | 2020 | $1,895,291,573.55 |
| liraglutide | 2021 | $1,757,219,274.98 |
| liraglutide | 2022 | $1,557,800,381.69 |
| liraglutide | 2023 | $1,321,882,656.63 |
| semaglutide | 2019 | $552,769,003.15 |
| semaglutide | 2020 | $1,529,256,675.71 |
| semaglutide | 2021 | $3,076,258,171.59 |
| semaglutide | 2022 | $5,603,055,113.53 |
| semaglutide | 2023 | $10,860,155,151.99 |

---

## Cross-Source Coverage

| Drug | MoA Class | NHS | FAERS | CMS | NHS Latest | FAERS Total Reports |
|------|-----------|-----|-------|-----|-----------|-------------------|
| acarbose | Alpha-Glucosidase-Inhibitor | Yes | Yes | Yes | 2025-11-01 | 3,072 |
| miglitol | Alpha-Glucosidase-Inhibitor | No | Yes | Yes | - | 687 |
| cagrilintide | Amylin Analogue | No | No | No | - | - |
| pramlintide | Amylin Analogue | No | Yes | Yes | - | 162 |
| metformin | Biguanide | Yes | Yes | Yes | 2025-11-01 | 386,126 |
| colesevelam | Bile Acid Sequestrant | Yes | Yes | Yes | 2025-11-01 | 10,669 |
| bromocriptine | Dopamin-Agonist (D2) | Yes | Yes | Yes | 2025-11-01 | 564 |
| alogliptin | DPP-4 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 2,281 |
| linagliptin | DPP-4 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 21,212 |
| saxagliptin | DPP-4 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 7,594 |
| sitagliptin | DPP-4 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 71,891 |
| vildagliptin | DPP-4 Inhibitor | Yes | No | No | 2025-11-01 | - |
| obeticholic acid | FXR Agonist | Yes | No | Yes | 2025-06-01 | - |
| tirzepatide | GIP/GLP-1 Dual Agonist | Yes | Yes | Yes | 2025-11-01 | 132,168 |
| dulaglutide | GLP-1 Receptor Agonist | Yes | Yes | Yes | 2025-11-01 | 93,693 |
| exenatide | GLP-1 Receptor Agonist | Yes | Yes | Yes | 2025-11-01 | 89,120 |
| liraglutide | GLP-1 Receptor Agonist | Yes | Yes | Yes | 2025-11-01 | 47,582 |
| semaglutide | GLP-1 Receptor Agonist | Yes | Yes | Yes | 2025-11-01 | 91,723 |
| retatrutide | GLP-1/GIP/Glucagon Triple Agonist | No | No | No | - | - |
| pemvidutide | GLP-1/Glucagon Dual Agonist | No | No | No | - | - |
| survodutide | GLP-1/Glucagon Dual Agonist | No | No | No | - | - |
| insulin glargine | Insulin (Basal) | Yes | Yes | Yes | 2025-11-01 | 210,014 |
| insulin aspart | Insulin (Rapid-acting) | Yes | Yes | Yes | 2025-11-01 | 59,532 |
| insulin lispro | Insulin (Rapid-acting) | Yes | Yes | Yes | 2025-11-01 | 135,447 |
| insulin degludec | Insulin (Ultra-long-acting) | Yes | Yes | Yes | 2025-11-01 | 21,577 |
| insulin icodec | Insulin (Weekly) | No | No | No | - | - |
| nateglinide | Meglitinid | Yes | Yes | Yes | 2025-11-01 | 1,059 |
| repaglinide | Meglitinid | Yes | Yes | Yes | 2025-11-01 | 6,463 |
| danuglipron | Oral GLP-1 RA (Small Molecule) | No | No | No | - | - |
| orforglipron | Oral GLP-1 RA (Small Molecule) | No | No | No | - | - |
| sotagliflozin | SGLT1/SGLT2 Inhibitor | No | Yes | Yes | - | 469 |
| bexagliflozin | SGLT2 Inhibitor | No | Yes | No | - | 66 |
| canagliflozin | SGLT2 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 29,417 |
| dapagliflozin | SGLT2 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 46,361 |
| empagliflozin | SGLT2 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 54,940 |
| ertugliflozin | SGLT2 Inhibitor | Yes | Yes | Yes | 2025-11-01 | 1,081 |
| gliclazide | Sulfonylharnstoff | Yes | No | No | 2025-11-01 | - |
| glimepiride | Sulfonylharnstoff | Yes | Yes | Yes | 2025-11-01 | 33,332 |
| glipizide | Sulfonylharnstoff | Yes | Yes | Yes | 2025-11-01 | 58,240 |
| glyburide | Sulfonylharnstoff | No | Yes | Yes | - | 26,738 |
| pioglitazone | Thiazolidinedione (PPAR-gamma) | Yes | Yes | Yes | 2025-11-01 | 43,727 |
| rosiglitazone | Thiazolidinedione (PPAR-gamma) | Yes | No | No | 2023-04-01 | - |
| resmetirom | THR-beta Agonist | No | Yes | No | - | 1,821 |

**Coverage Summary:** 26 drugs with all 3 sources, 36 with at least 1 source, 7 with no data

---

## Probleme & Empfehlungen

### Bekannte Einschränkungen
- **Pipeline-Drugs** (cagrilintide, danuglipron, insulin icodec, orforglipron, pemvidutide, retatrutide, survodutide) haben erwartungsgemäß keine Markt-/Sicherheitsdaten — sie sind noch nicht zugelassen
- **Gliclazide** ist in den USA nicht zugelassen, daher keine FAERS/CMS-Daten
- **Vildagliptin** ist in den USA nicht zugelassen (nur EU), daher keine FAERS/CMS-Daten
- **Rosiglitazone** wurde vom Markt genommen, daher keine aktuellen CMS-Daten
- **Bexagliflozin** ist neu zugelassen (2023), daher eingeschränkte CMS-Daten
- **Resmetirom** ist neu zugelassen (2024), daher noch keine CMS-Daten

### Empfehlungen für Dashboard-Integration
1. **Neuer Tab: Market Data** — UK/US Spending-Trends, Cost-per-Unit Vergleich, Beneficiary-Trends
2. **Neuer Tab: Safety Profile** — AE Heatmap, Serious/Non-Serious Ratio, Quarterly Trends
3. **Drug Deep Dive erweitern** — Market + Safety Daten pro Drug hinzufügen
4. **ML Feature Engineering** — Missing Data als Signal nutzen (kein Marktdata = Pipeline-Drug)
