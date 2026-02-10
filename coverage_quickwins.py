"""
Trial-Indication Linking: Quick-Win Coverage Improvements
1. Additional synonyms for unmatched trials
2. Heuristic for generic "Diabetes" without Type specification
"""
import pyodbc
import json
from datetime import datetime
from collections import Counter

from db_config import CONN_STR

# ============================================================
# QUICK-WIN 1: Additional Synonyms
# ============================================================

ADDITIONAL_SYNONYMS = {
    "Type 2 Diabetes Mellitus": [
        "Type2 Diabetes Mellitus", "Type2 Diabetes", "Type2diabetes",
        "Type 2 Diabets Mellitus", "Type 2 Diabetes Melitus",
        "T2DM (Type 2 Diabetes Mellitus)", "Type 2 Diabetes Mellitus (T2DM)",
        "T2DM - Type 2 Diabetes Mellitus", "Diabetes Mellitus Type II",
        "Diabetes Type II", "Diabetes Type 2",
        "Diabetes Mellitus, Non-Insulin-Dependent",
        "Diabetes Mellitus, Non Insulin Dependent",
        "Non-Insulin Dependent Diabetes Mellitus",
        "Non-Insulin Dependent Diabetes", "NIDDM",
        "Maturity Onset Diabetes Mellitus", "Maturity-Onset Diabetes Mellitus",
        "Adult Onset Diabetes Mellitus", "Adult-Onset Diabetes",
        "Type 2 Diabetes Treated With Insulin",
        "Insulin Treated Type 2 Diabetes", "Type 2 Diabetes on Insulin",
        "DM2", "DM Type 2", "DM, Type 2",
    ],
    "Type 1 Diabetes Mellitus": [
        "Type1 Diabetes Mellitus", "Type1 Diabetes", "Type1diabetes",
        "Type 1 Diabets Mellitus",
        "T1DM (Type 1 Diabetes Mellitus)", "Type 1 Diabetes Mellitus (T1DM)",
        "T1DM - Type 1 Diabetes Mellitus", "Diabetes Mellitus Type I",
        "Diabetes Type I", "Diabetes Type 1",
        "Diabetes Mellitus, Insulin-Dependent",
        "Diabetes Mellitus, Insulin Dependent",
        "Insulin Dependent Diabetes Mellitus",
        "DM1", "DM Type 1", "DM, Type 1",
    ],
    "NASH / MASH": [
        "Nonalcoholic Steatohepatitis (NASH)",
        "Non-alcoholic Steatohepatitis (NASH)",
        "Non Alcoholic Steatohepatitis (NASH)",
        "NASH (Nonalcoholic Steatohepatitis)",
        "NASH (Non-alcoholic Steatohepatitis)",
        "MASH (Metabolic Associated Steatohepatitis)",
        "Metabolic Associated Steatohepatitis (MASH)",
        "Metabolic Dysfunction-Associated Steatohepatitis",
        "Metabolic Dysfunction Associated Steatohepatitis (MASH)",
    ],
    "Non-alcoholic Fatty Liver Disease": [
        "Non-alcoholic Fatty Liver Disease (NAFLD)",
        "Nonalcoholic Fatty Liver Disease (NAFLD)",
        "NAFLD (Non-alcoholic Fatty Liver Disease)",
        "NAFLD (Nonalcoholic Fatty Liver Disease)",
        "MASLD (Metabolic Dysfunction-Associated Steatotic Liver Disease)",
        "Metabolic Dysfunction-Associated Steatotic Liver Disease",
        "Metabolic Dysfunction Associated Steatotic Liver Disease (MASLD)",
        "MAFLD (Metabolic Associated Fatty Liver Disease)",
    ],
    "Prediabetes": [
        "Insulin Resistance", "Insulin Sensitivity",
        "Glucose Metabolism Disorders", "Hyperinsulinemia",
    ],
    "Overweight": [
        "Overweight and Obesity", "Obesity and Overweight",
        "Overweight or Obese", "Obese or Overweight",
    ],
    "Obesity": [
        "Body Mass Index", "BMI", "Obesity Hypoventilation Syndrome",
    ],
}

# ============================================================
# QUICK-WIN 2: Heuristics for generic Diabetes
# ============================================================

GENERIC_DIABETES_CONDITIONS = [
    "diabetes mellitus", "diabetes", "diabetes mellitus (dm)", "dm",
    "diabete", "diabetic",
]

T2DM_INDICATOR_CONDITIONS = [
    "obesity", "overweight", "metabolic syndrome",
    "insulin resistance", "nafld", "nash", "mash", "masld",
    "dyslipidemia", "hyperlipidemia", "hypercholesterolemia",
    "fatty liver", "steatohepatitis", "polycystic ovary",
]

T1DM_INDICATOR_CONDITIONS = [
    "autoimmune", "celiac", "thyroiditis",
    "diabetic ketoacidosis", "dka", "type 1",
]

T2DM_INDICATOR_DRUGS = [
    "metformin", "glimepiride", "glipizide", "glyburide", "glibenclamide",
    "sitagliptin", "linagliptin", "saxagliptin", "alogliptin", "vildagliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    "liraglutide", "semaglutide", "dulaglutide", "exenatide", "tirzepatide",
    "pioglitazone", "rosiglitazone", "acarbose", "repaglinide", "nateglinide",
    "januvia", "jardiance", "farxiga", "invokana", "ozempic", "victoza",
    "trulicity", "mounjaro", "byetta", "bydureon", "tradjenta", "trajenta",
    "onglyza", "glucophage", "amaryl", "glucotrol",
]

T1DM_INDICATOR_DRUGS = [
    "insulin pump", "continuous glucose monitor", "cgm",
    "artificial pancreas", "closed loop", "closed-loop",
    "islet transplant", "beta cell", "teplizumab",
]

BIG_PHARMA_SPONSORS = [
    "novo nordisk", "eli lilly", "astrazeneca", "sanofi",
    "merck sharp", "pfizer", "boehringer", "janssen", "takeda",
    "novartis", "bristol-myers", "amgen", "roche", "abbvie",
    "gilead", "bayer", "glaxosmithkline", "gsk",
]


def main():
    print("=" * 70)
    print("COVERAGE QUICK-WINS: Synonym + Heuristic Matching")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Get baseline
    cursor.execute("SELECT COUNT(DISTINCT trial_id) FROM trial_indications")
    baseline_trials = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trial_indications")
    baseline_links = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trials")
    total_trials = cursor.fetchone()[0]
    print(f"\nBaseline: {baseline_trials:,} / {total_trials:,} trials linked ({baseline_trials/total_trials*100:.1f}%)")
    print(f"Baseline links: {baseline_links:,}")

    # Load indication map
    cursor.execute("SELECT indication_id, name FROM indications WHERE name != 'Diabetes & Metabolism'")
    indication_map = {row[1]: row[0] for row in cursor.fetchall()}

    # ============================================================
    # QUICK-WIN 1: Additional Synonyms
    # ============================================================
    print(f"\n{'='*70}")
    print("QUICK-WIN 1: Additional Synonyms")
    print(f"{'='*70}")

    # Build synonym lookup: lowercase synonym -> indication name
    synonym_lookup = {}
    for ind_name, synonyms in ADDITIONAL_SYNONYMS.items():
        for syn in synonyms:
            synonym_lookup[syn.strip().lower()] = ind_name

    # Get unlinked trials
    cursor.execute("""
        SELECT t.trial_id, t.raw_conditions
        FROM trials t
        WHERE NOT EXISTS (SELECT 1 FROM trial_indications ti WHERE ti.trial_id = t.trial_id)
        AND t.raw_conditions IS NOT NULL
    """)
    unlinked = cursor.fetchall()
    print(f"Unlinked trials to process: {len(unlinked):,}")

    qw1_new_links = 0
    qw1_trials_matched = 0
    qw1_by_indication = Counter()

    for trial_id, raw_cond in unlinked:
        try:
            conditions = json.loads(raw_cond)
        except:
            continue

        trial_matched = False
        for cond_text in conditions:
            cond_lower = cond_text.strip().lower()
            ind_name = synonym_lookup.get(cond_lower)
            if ind_name and ind_name in indication_map:
                ind_id = indication_map[ind_name]
                try:
                    cursor.execute("""
                        INSERT INTO trial_indications (trial_id, indication_id)
                        VALUES (?, ?)
                    """, (trial_id, ind_id))
                    qw1_new_links += 1
                    qw1_by_indication[ind_name] += 1
                    trial_matched = True
                except pyodbc.IntegrityError:
                    pass

        if trial_matched:
            qw1_trials_matched += 1

    print(f"\nQW1 Results:")
    print(f"  New links created: {qw1_new_links:,}")
    print(f"  Trials newly matched: {qw1_trials_matched:,}")
    print(f"\n  Breakdown by indication:")
    for ind, count in qw1_by_indication.most_common():
        print(f"    {ind:40s} +{count:>4,}")

    # ============================================================
    # QUICK-WIN 2: Heuristic for generic Diabetes
    # ============================================================
    print(f"\n{'='*70}")
    print("QUICK-WIN 2: Heuristic for Generic Diabetes")
    print(f"{'='*70}")

    # Re-fetch unlinked trials (after QW1)
    cursor.execute("""
        SELECT t.trial_id, t.raw_conditions, t.raw_interventions,
               t.phase, t.study_type, t.lead_sponsor_name
        FROM trials t
        WHERE NOT EXISTS (SELECT 1 FROM trial_indications ti WHERE ti.trial_id = t.trial_id)
        AND t.raw_conditions IS NOT NULL
    """)
    still_unlinked = cursor.fetchall()

    # Filter to only trials with generic "Diabetes" conditions
    generic_diabetes_trials = []
    for trial_id, raw_cond, raw_interv, phase, study_type, sponsor in still_unlinked:
        try:
            conditions = json.loads(raw_cond)
        except:
            continue

        has_generic_diabetes = False
        for cond in conditions:
            cond_lower = cond.strip().lower()
            if cond_lower in GENERIC_DIABETES_CONDITIONS:
                has_generic_diabetes = True
                break

        if has_generic_diabetes:
            generic_diabetes_trials.append(
                (trial_id, conditions, raw_interv, phase, study_type, sponsor)
            )

    print(f"Trials with generic 'Diabetes' (no type): {len(generic_diabetes_trials):,}")

    qw2_via_coconditions = 0
    qw2_via_interventions = 0
    qw2_via_fallback = 0
    qw2_ambiguous = 0
    qw2_t2dm = 0
    qw2_t1dm = 0

    t2dm_id = indication_map.get("Type 2 Diabetes Mellitus")
    t1dm_id = indication_map.get("Type 1 Diabetes Mellitus")

    for trial_id, conditions, raw_interv, phase, study_type, sponsor in generic_diabetes_trials:
        all_conds_lower = " ".join(c.lower() for c in conditions)
        assigned = None
        method = None

        # Step 1: Check co-conditions
        for indicator in T2DM_INDICATOR_CONDITIONS:
            if indicator in all_conds_lower:
                assigned = "T2DM"
                method = "co-condition"
                break

        if not assigned:
            for indicator in T1DM_INDICATOR_CONDITIONS:
                if indicator in all_conds_lower:
                    assigned = "T1DM"
                    method = "co-condition"
                    break

        # Step 2: Check interventions
        if not assigned and raw_interv:
            interv_lower = raw_interv.lower()
            for drug in T2DM_INDICATOR_DRUGS:
                if drug in interv_lower:
                    assigned = "T2DM"
                    method = "intervention"
                    break

            if not assigned:
                for drug in T1DM_INDICATOR_DRUGS:
                    if drug in interv_lower:
                        assigned = "T1DM"
                        method = "intervention"
                        break

        # Step 3: Phase/Sponsor fallback
        if not assigned:
            if phase == "phase4" and study_type == "interventional":
                if sponsor:
                    sponsor_lower = sponsor.lower()
                    is_industry = any(bp in sponsor_lower for bp in BIG_PHARMA_SPONSORS)
                    if is_industry:
                        assigned = "T2DM"
                        method = "fallback"

        # Step 4: Insert
        if assigned:
            ind_id = t2dm_id if assigned == "T2DM" else t1dm_id
            try:
                cursor.execute("""
                    INSERT INTO trial_indications (trial_id, indication_id)
                    VALUES (?, ?)
                """, (trial_id, ind_id))

                if method == "co-condition":
                    qw2_via_coconditions += 1
                elif method == "intervention":
                    qw2_via_interventions += 1
                elif method == "fallback":
                    qw2_via_fallback += 1

                if assigned == "T2DM":
                    qw2_t2dm += 1
                else:
                    qw2_t1dm += 1
            except pyodbc.IntegrityError:
                pass
        else:
            qw2_ambiguous += 1

    qw2_total = qw2_via_coconditions + qw2_via_interventions + qw2_via_fallback
    print(f"\nQW2 Results:")
    print(f"  Assigned via co-conditions: {qw2_via_coconditions:,}")
    print(f"  Assigned via interventions: {qw2_via_interventions:,}")
    print(f"  Assigned via fallback (Phase4+Industry): {qw2_via_fallback:,}")
    print(f"  Total assigned: {qw2_total:,}")
    print(f"  Ambiguous (not assigned): {qw2_ambiguous:,}")
    print(f"\n  Breakdown: T2DM +{qw2_t2dm:,}, T1DM +{qw2_t1dm:,}")

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL COVERAGE REPORT")
    print(f"{'='*70}")

    cursor.execute("SELECT COUNT(DISTINCT trial_id) FROM trial_indications")
    new_trials = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trial_indications")
    new_links = cursor.fetchone()[0]

    improvement_trials = new_trials - baseline_trials
    improvement_pct = new_trials / total_trials * 100
    baseline_pct = baseline_trials / total_trials * 100

    print(f"\n  {'Metric':<40s} {'Before':>10s} {'After':>10s} {'Diff':>10s}")
    print(f"  {'-'*75}")
    print(f"  {'Trials with indication':40s} {baseline_trials:>10,} {new_trials:>10,} {improvement_trials:>+10,}")
    print(f"  {'Total links':40s} {baseline_links:>10,} {new_links:>10,} {new_links-baseline_links:>+10,}")
    print(f"  {'Coverage %':40s} {baseline_pct:>9.1f}% {improvement_pct:>9.1f}% {improvement_pct-baseline_pct:>+9.1f}%")

    # Updated indication breakdown
    print(f"\n  Updated Trials per Indication:")
    cursor.execute("""
        SELECT i.name, COUNT(DISTINCT ti.trial_id) AS cnt
        FROM trial_indications ti
        JOIN indications i ON ti.indication_id = i.indication_id
        GROUP BY i.name ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"    {row[0]:40s} {row[1]:>6,}")

    # Remaining gap analysis
    cursor.execute("""
        SELECT COUNT(*) FROM trials t
        WHERE NOT EXISTS (SELECT 1 FROM trial_indications ti WHERE ti.trial_id = t.trial_id)
    """)
    remaining_gap = cursor.fetchone()[0]
    print(f"\n  Remaining unlinked trials: {remaining_gap:,}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return values for report
    return {
        "baseline_trials": baseline_trials,
        "baseline_links": baseline_links,
        "baseline_pct": baseline_pct,
        "new_trials": new_trials,
        "new_links": new_links,
        "improvement_pct": improvement_pct,
        "qw1_trials": qw1_trials_matched,
        "qw1_links": qw1_new_links,
        "qw1_by_ind": dict(qw1_by_indication),
        "qw2_total": qw2_total,
        "qw2_cocond": qw2_via_coconditions,
        "qw2_interv": qw2_via_interventions,
        "qw2_fallback": qw2_via_fallback,
        "qw2_ambiguous": qw2_ambiguous,
        "qw2_t2dm": qw2_t2dm,
        "qw2_t1dm": qw2_t1dm,
        "remaining_gap": remaining_gap,
        "total_trials": total_trials,
    }


if __name__ == "__main__":
    main()
