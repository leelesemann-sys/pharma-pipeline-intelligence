# Trial-Indication Linking: Insights zu nicht gemappten Trials
**Datum:** 2026-02-09
**Kontext:** 7.805 von 32.811 Trials (23,8%) haben keine Indication-Zuordnung

---

## 1. Zusammenfassung

| Metrik | Wert |
|--------|------|
| Trials ohne Indication-Link | 7.805 (23,8%) |
| Unique unmatched Condition-Strings | 6.840 |
| Davon "generisches Diabetes" (ohne Type) | 1.078 Trials |

---

## 2. Warum sind diese Trials nicht gemappt?

### Kategorie A: "Generisches Diabetes" (1.819 Condition-Nennungen)
**Das groesste Mapping-Problem.** Viele Trials listen nur:
- "Diabetes Mellitus" (503x)
- "Diabetes" (379x)
- "Type2 Diabetes Mellitus" (53x) - Schreibvariante ohne Leerzeichen
- "Diabetes Mellitus, Non-Insulin-Dependent" (52x)
- "T2DM (Type 2 Diabetes Mellitus)" (36x) - Klammer-Format
- "Type2diabetes" (35x)
- "Type1 Diabetes Mellitus" (43x)

**Empfehlung:** Diese koennten mit einer zweiten Matching-Runde aufgeloest werden:
- "Diabetes Mellitus" / "Diabetes" ohne Type -> Heuristik: Wenn Interventions GLP-1, SGLT2, DPP-4 oder Metformin enthalten -> T2DM
- Varianten wie "Type2diabetes", "T2DM (...)" -> Synonyme erweitern
- Geschaetzter Coverage-Gewinn: +5-8 Prozentpunkte

### Kategorie B: Co-Morbidity/Co-Condition Trials (irrelevant fuer Scope)
Trials die in unserer DB sind weil sie z.B. "Obesity" als sekundaere Condition hatten, aber deren primaere Indication ausserhalb unseres Scope liegt:

| Cluster | Vorkommen | Beispiele |
|---------|-----------|-----------|
| **Cancer/Onkologie** | 2.094 | Breast Cancer, Colorectal Cancer, Leukemia, Multiple Myeloma |
| **Kardiovaskulaer** | 873 | Hypertension, Heart Failure, Coronary Artery Disease, Stroke |
| **Infektionskrankheiten** | 650 | HIV/AIDS, Hepatitis, COVID-19 |
| **Lifestyle/Behavioral** | 511 | Exercise, Physical Activity, Diet, Nutrition |
| **Weight/Body Comp** | 476 | Weight Loss, BMI, Adiposity, Sarcopenia |
| **Leber (nicht-NAFLD)** | 425 | Cirrhosis, Hepatitis C, Liver Transplant |
| **Endokrin** | 389 | PCOS (214x!), Thyroid, Growth Hormone |
| **Niere** | 375 | CKD, Dialysis, Renal Failure |
| **Mental Health** | 246 | Depression, Schizophrenia, Bipolar |
| **Dyslipidaemie** | 233 | Hyperlipidemia, Hypercholesterolemia |
| **Autoimmun** | 232 | Rheumatoid Arthritis, Lupus |
| **Chirurgisch** | 230 | Bariatric Surgery, Transplant |

**Empfehlung:** Diese Trials sind KORREKT nicht zugeordnet. Sie gehoeren nicht zu unseren 12 Kern-Indikationen. Fuer Phase 2 koennte ein "Co-Condition Tag" eingefuehrt werden.

### Kategorie C: Seltene Erkrankungen mit Metabolischer Komponente
Ueberraschend viele Trials zu seltenen Erkrankungen:
- Prader-Willi Syndrome (101x) - hat Adipositas-Komponente
- Celiac Disease (231x) - Autoimmun, T1DM-Assoziation
- Lynch Syndrome (95x) - Krebs-Praedisposition
- Congenital Adrenal Hyperplasia (62x)
- Lipodystrophy (40x) - metabolisch relevant
- Fanconi Anemia (70x)

**Empfehlung:** Prader-Willi und Lipodystrophy koennten als Sub-Indikationen zu "Obesity" bzw. "Metabolic Syndrome" gemappt werden.

### Kategorie D: Healthy Volunteers / Quality of Life
- "Healthy" / "Healthy Volunteers" / "Healthy Participants" (180x)
- "Quality of Life" (27x)
- Phase-1-Studien mit gesunden Probanden

**Empfehlung:** Kein Mapping noetig. Diese Trials testen Pharmakokinetik bei Gesunden.

---

## 3. Phase-Verteilung der ungemappten Trials

| Phase | Count | Anteil |
|-------|-------|--------|
| N/A (observational etc.) | 5.055 | 64,8% |
| Phase 2 | 717 | 9,2% |
| Phase 4 | 604 | 7,7% |
| Phase 3 | 500 | 6,4% |
| Phase 1 | 486 | 6,2% |
| Phase 1/2 | 243 | 3,1% |
| Phase 2/3 | 118 | 1,5% |
| Early Phase 1 | 82 | 1,1% |

**Insight:** 64,8% sind observational/N/A - also Beobachtungsstudien, nicht interventionell. Fuer die Predictive Models (Phase 3 des Projekts) sind hauptsaechlich interventionelle Phase-2/3-Trials relevant, und davon fehlen nur ~1.200.

---

## 4. Status-Verteilung der ungemappten Trials

| Status | Count |
|--------|-------|
| completed | 4.623 (59,2%) |
| unknown | 1.088 (13,9%) |
| recruiting | 788 (10,1%) |
| terminated | 429 (5,5%) |
| not_yet_recruiting | 326 (4,2%) |
| active_not_recruiting | 232 (3,0%) |
| withdrawn | 211 (2,7%) |

**Insight:** Ueber 59% sind bereits abgeschlossen. Der "Verlust" an aktuell relevanten, laufenden Trials ist gering.

---

## 5. Quick-Win Mapping-Erweiterungen

Falls gewuenscht, koennten diese Synonyme noch hinzugefuegt werden:

```python
ADDITIONAL_SYNONYMS = {
    "Type 2 Diabetes Mellitus": [
        "Type2 Diabetes Mellitus",      # 53x
        "Diabetes Mellitus, Non-Insulin-Dependent",  # 52x
        "T2DM (Type 2 Diabetes Mellitus)",  # 36x
        "Type2diabetes",                 # 35x
        "Type 2 Diabetes Treated With Insulin",  # 33x
        "Non-alcoholic Fatty Liver Disease (NAFLD)",  # wird separat gemappt
        "Diabetes Mellitus (DM)",        # diverse
    ],
    "Type 1 Diabetes Mellitus": [
        "Type1 Diabetes Mellitus",       # 43x
    ],
    "NASH / MASH": [
        "Nonalcoholic Steatohepatitis (NASH)",  # 34x
        "Non-alcoholic Steatohepatitis (NASH)",  # 23x
    ],
    "Non-alcoholic Fatty Liver Disease": [
        "Non-alcoholic Fatty Liver Disease (NAFLD)",  # 27x
    ],
}
```

**Geschaetzter Gewinn:** +250-350 Trials (ca. +1% Coverage)

---

## 6. Fazit

Die **76,2% Coverage** sind fuer unseren Zweck **gut**. Die fehlenden 23,8% bestehen hauptsaechlich aus:
1. **Co-Morbidity-Trials** (Kardio, Onko, Infektio) die ausserhalb unseres Scope liegen (~60% der Luecke)
2. **Generisches "Diabetes"** ohne Type-Spezifikation (~15% der Luecke)
3. **Seltene Erkrankungen** mit metabolischer Assoziation (~10%)
4. **Healthy Volunteer / PK-Studien** (~5%)
5. **Schreibvarianten** die unsere Synonyme knapp verfehlen (~10%)

Fuer die Predictive Models (Phase 3) ist das kein Blocker - die relevanten interventionellen Phase-2/3-Trials sind zu >85% gemappt.

---

## 7. Begleitdatei

`unmatched_trials_2000_examples.csv` enthaelt die 2.000 groessten ungemappten Trials (sortiert nach Enrollment), mit allen relevanten Feldern fuer manuelle Inspektion.

Spalten: nct_id, title, raw_conditions, phase, overall_status, lead_sponsor, enrollment, study_type, start_date, has_results
