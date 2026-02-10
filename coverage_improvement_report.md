# Coverage Improvement Report
**Datum:** 2026-02-09
**Kontext:** Quick-Win Verbesserungen fuer Trial-Indication Linking

---

## Zusammenfassung

| Metrik | Vorher | Nachher | Diff |
|--------|--------|---------|------|
| **Trials mit Indication** | 25.006 | **25.852** | **+846** |
| **Trial-Indication Links** | 29.762 | **30.633** | **+871** |
| **Coverage** | 76,2% | **78,8%** | **+2,6 PP** |

---

## Quick-Win 1: Zusaetzliche Synonyme

**Ergebnis: +630 Trials, +655 Links**

| Indication | Neue Links | Beispiel-Synonyme |
|-----------|-----------|-------------------|
| Type 2 Diabetes Mellitus | +231 | "Type2 Diabetes Mellitus", "DM Type 2", "Non-Insulin Dependent DM" |
| Prediabetes | +180 | "Insulin Resistance" (139x), "Glucose Metabolism Disorders" |
| Type 1 Diabetes Mellitus | +80 | "Type1 Diabetes Mellitus", "DM Type 1", "Insulin Dependent DM" |
| NASH / MASH | +80 | "Nonalcoholic Steatohepatitis (NASH)", "MASH (...)" |
| Non-alcoholic Fatty Liver Disease | +69 | "NAFLD (...)", "MASLD (...)", "MAFLD (...)" |
| Obesity | +8 | "BMI", "Obesity Hypoventilation Syndrome" |
| Overweight | +7 | "Overweight and Obesity", "Obese or Overweight" |

**Groesster Gewinn:** "Insulin Resistance" als Prediabetes-Synonym (+139 Trials) und Klammer-Varianten von NASH/NAFLD.

---

## Quick-Win 2: Heuristik fuer Generisches "Diabetes"

**870 Trials** hatten nur "Diabetes Mellitus" / "Diabetes" als Condition ohne Type-Spezifikation.

| Zuordnungsmethode | Trials | Davon T2DM | Davon T1DM |
|-------------------|--------|-----------|-----------|
| Via Co-Conditions (Obesity, MetSyn...) | 38 | 35 | 3 |
| Via Interventions (Metformin, SGLT2...) | 173 | 130 | 43 |
| Via Fallback (Phase4 + Industry Sponsor) | 5 | 5 | 0 |
| **Total zugeordnet** | **216** | **170** | **46** |
| Ambiguous (nicht zugeordnet) | 654 | - | - |

**Ergebnis: +216 Trials (170x T2DM, 46x T1DM)**

Die 654 nicht zugeordneten Trials bleiben konservativ ungemappt - sie haben weder eindeutige Co-Conditions noch Diabetes-spezifische Interventions.

---

## Aktualisierte Indication-Verteilung

| Indication | Trials | Aenderung |
|-----------|--------|-----------|
| Obesity | 10.953 | +8 |
| Type 2 Diabetes Mellitus | 9.114 | +401 |
| Type 1 Diabetes Mellitus | 3.387 | +126 |
| Overweight | 2.203 | +7 |
| Metabolic Syndrome | 1.701 | -- |
| Non-alcoholic Fatty Liver Disease | 1.185 | +69 |
| Prediabetes | 865 | +180 |
| NASH / MASH | 571 | +80 |
| Diabetic Kidney Disease | 227 | -- |
| Gestational Diabetes | 188 | -- |
| Diabetic Neuropathy | 154 | -- |
| Diabetic Retinopathy | 85 | -- |

---

## Verbleibende Luecke: 6.959 Trials (21,2%)

| Kategorie | Geschaetzte Anzahl | Aktion |
|-----------|-------------------|--------|
| Co-Morbidity (Kardio, Onko, Infektio) | ~4.000 | Korrekt unmatched, ausserhalb Scope |
| Seltene Erkrankungen (Prader-Willi, Celiac...) | ~800 | Korrekt unmatched |
| Ambiguous Diabetes (kein Typ erkennbar) | ~654 | Konservativ ungemappt |
| Healthy Volunteers / PK-Studien | ~400 | Korrekt unmatched |
| Lifestyle/Behavioral (Exercise, Diet) | ~500 | Korrekt unmatched |
| Sonstige | ~600 | Diverse |

**Fazit:** Die verbleibenden 21,2% sind ganz ueberwiegend Trials deren primaere Indication ausserhalb unseres Diabetes/Obesity-Scope liegt. Die erreichbare Coverage mit den gegebenen 12 Indikationen liegt bei ca. 80% - wir sind jetzt bei 78,8% und damit nahe am Maximum.

---

## Dateien

| Datei | Beschreibung |
|-------|-------------|
| `coverage_quickwins.py` | Quick-Win Matching-Skript |
| `coverage_improvement_report.md` | Dieser Report |

---

*Trial-Indication Coverage: 76,2% -> 78,8% (+846 Trials)*
