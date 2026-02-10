"""
Phase 1 Completion - Aufgabe 1: Trial → Indication Linking
Synonym-basiertes Matching von raw_conditions zu indications.
"""
import pyodbc
import json
from datetime import datetime
from collections import Counter

from db_config import CONN_STR

INDICATION_SYNONYMS = {
    "Type 2 Diabetes Mellitus": {
        "exact": [
            "Diabetes Mellitus, Type 2", "Type 2 Diabetes Mellitus", "Type 2 Diabetes",
            "Type2 Diabetes", "Diabetes Mellitus Type 2", "Diabetes Mellitus, Type II",
            "Type II Diabetes Mellitus", "Type II Diabetes", "T2DM", "Diabetes, Type 2",
            "Adult-Onset Diabetes Mellitus", "Non-Insulin-Dependent Diabetes Mellitus", "NIDDM",
            "Maturity-Onset Diabetes",
        ],
        "contains": [],
        "excludes": ["Type 1", "Gestational", "Insipidus"],
    },
    "Type 1 Diabetes Mellitus": {
        "exact": [
            "Diabetes Mellitus, Type 1", "Type 1 Diabetes Mellitus", "Type 1 Diabetes",
            "Type1 Diabetes", "Diabetes Mellitus Type 1", "Diabetes Mellitus, Type I",
            "Type I Diabetes Mellitus", "Type I Diabetes", "T1DM", "T1D", "Diabetes, Type 1",
            "Insulin-Dependent Diabetes Mellitus", "IDDM", "Juvenile Diabetes",
            "Juvenile-Onset Diabetes", "Autoimmune Diabetes",
        ],
        "contains": [],
        "excludes": [],
    },
    "Obesity": {
        "exact": [
            "Obesity", "Obesity, Morbid", "Morbid Obesity", "Obesity, Abdominal",
            "Abdominal Obesity", "Obesity, Severe", "Severe Obesity", "Class III Obesity",
            "Class II Obesity", "Pediatric Obesity", "Childhood Obesity", "Adolescent Obesity",
            "Central Obesity", "Visceral Obesity", "Truncal Obesity",
        ],
        "contains": ["Obesity"],
        "excludes": [],
    },
    "Overweight": {
        "exact": [
            "Overweight", "Overweight and Obesity", "Obesity and Overweight", "Body Weight",
        ],
        "contains": [],
        "excludes": [],
    },
    "NASH / MASH": {
        "exact": [
            "Non-alcoholic Steatohepatitis", "Nonalcoholic Steatohepatitis",
            "Non Alcoholic Steatohepatitis", "NASH",
            "Metabolic Associated Steatohepatitis", "Metabolic Dysfunction Associated Steatohepatitis",
            "MASH", "Steatohepatitis", "Fatty Liver, Non-Alcoholic",
        ],
        "contains": ["Steatohepatitis", "NASH", "MASH"],
        "excludes": ["Alcoholic Steatohepatitis"],
    },
    "Non-alcoholic Fatty Liver Disease": {
        "exact": [
            "Non-alcoholic Fatty Liver Disease", "Nonalcoholic Fatty Liver Disease",
            "Non Alcoholic Fatty Liver Disease", "NAFLD", "Fatty Liver", "Hepatic Steatosis",
            "Metabolic Associated Fatty Liver Disease", "MAFLD",
            "Metabolic Dysfunction Associated Steatotic Liver Disease", "MASLD",
        ],
        "contains": ["Fatty Liver", "NAFLD", "MAFLD", "MASLD"],
        "excludes": ["Alcoholic Fatty Liver"],
    },
    "Diabetic Kidney Disease": {
        "exact": [
            "Diabetic Nephropathies", "Diabetic Nephropathy", "Diabetic Kidney Disease",
            "Diabetic Kidney Diseases", "DKD",
        ],
        "contains": ["Diabetic Nephro", "Diabetic Kidney"],
        "excludes": [],
    },
    "Diabetic Retinopathy": {
        "exact": [
            "Diabetic Retinopathy", "Diabetic Retinopathies", "Diabetic Macular Edema",
            "Diabetic Eye Disease", "Diabetic Eye Diseases",
        ],
        "contains": ["Diabetic Retino", "Diabetic Macular"],
        "excludes": [],
    },
    "Diabetic Neuropathy": {
        "exact": [
            "Diabetic Neuropathies", "Diabetic Neuropathy", "Diabetic Peripheral Neuropathy",
            "Diabetic Polyneuropathy", "Diabetic Foot", "Diabetic Feet",
        ],
        "contains": ["Diabetic Neuro", "Diabetic Peripheral Neuro", "Diabetic Polyneuropath"],
        "excludes": [],
    },
    "Metabolic Syndrome": {
        "exact": [
            "Metabolic Syndrome", "Metabolic Syndrome X", "Syndrome X, Metabolic",
            "Insulin Resistance Syndrome", "Dysmetabolic Syndrome", "Reaven Syndrome",
        ],
        "contains": ["Metabolic Syndrome"],
        "excludes": [],
    },
    "Prediabetes": {
        "exact": [
            "Prediabetic State", "Prediabetes", "Pre-diabetes", "Pre-Diabetic State",
            "Impaired Glucose Tolerance", "Impaired Fasting Glucose", "Glucose Intolerance",
            "IGT", "IFG", "Borderline Diabetes",
        ],
        "contains": ["Prediabet", "Pre-diabet", "Impaired Glucose", "Impaired Fasting", "Glucose Intolerance"],
        "excludes": [],
    },
    "Gestational Diabetes": {
        "exact": [
            "Diabetes, Gestational", "Gestational Diabetes", "Gestational Diabetes Mellitus",
            "GDM", "Pregnancy Diabetes", "Diabetes in Pregnancy",
        ],
        "contains": ["Gestational Diabetes", "Gestational DM"],
        "excludes": [],
    },
}


def match_condition_to_indications(condition_text):
    """Match a single condition string to our indications. Returns list of indication names."""
    matches = []
    condition_lower = condition_text.strip().lower()

    for indication_name, rules in INDICATION_SYNONYMS.items():
        matched = False

        # 1. Exact match (case-insensitive)
        for synonym in rules["exact"]:
            if condition_lower == synonym.lower():
                matched = True
                break

        # 2. Contains match (only if no exact match)
        if not matched:
            for pattern in rules.get("contains", []):
                if pattern.lower() in condition_lower:
                    excluded = False
                    for exclude in rules.get("excludes", []):
                        if exclude.lower() in condition_lower:
                            excluded = True
                            break
                    if not excluded:
                        matched = True
                        break

        if matched:
            matches.append(indication_name)

    return matches


def main():
    print("=" * 70)
    print("AUFGABE 1: Trial -> Indication Linking")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Load indication_id mapping
    cursor.execute("SELECT indication_id, name FROM indications WHERE name != 'Diabetes & Metabolism'")
    indication_map = {}
    for row in cursor.fetchall():
        indication_map[row[1]] = row[0]
    print(f"Indications loaded: {len(indication_map)}")

    # Get all trials with raw_conditions
    cursor.execute("SELECT trial_id, nct_id, raw_conditions FROM trials WHERE raw_conditions IS NOT NULL")
    trials = cursor.fetchall()
    print(f"Trials to process: {len(trials)}")

    # Process
    links_created = 0
    trials_linked = 0
    trials_unlinked = 0
    indication_counts = Counter()
    unmatched_conditions = Counter()
    all_condition_counts = Counter()

    for trial_id, nct_id, raw_conditions in trials:
        try:
            conditions = json.loads(raw_conditions)
        except:
            continue

        trial_indications = set()

        for cond_text in conditions:
            all_condition_counts[cond_text] += 1
            matched_names = match_condition_to_indications(cond_text)

            if matched_names:
                for name in matched_names:
                    if name in indication_map:
                        trial_indications.add(indication_map[name])
                        indication_counts[name] += 1
            else:
                unmatched_conditions[cond_text] += 1

        # Insert trial-indication links
        if trial_indications:
            trials_linked += 1
            for ind_id in trial_indications:
                try:
                    cursor.execute("""
                        INSERT INTO trial_indications (trial_id, indication_id)
                        VALUES (?, ?)
                    """, (trial_id, ind_id))
                    links_created += 1
                except pyodbc.IntegrityError:
                    pass  # duplicate
        else:
            trials_unlinked += 1

    # Report
    print(f"\n{'='*70}")
    print("TRIAL-INDICATION LINKING REPORT")
    print(f"{'='*70}")
    print(f"\nTotal trial-indication links created: {links_created:,}")
    print(f"Trials with at least one indication: {trials_linked:,}")
    print(f"Trials with NO indication match: {trials_unlinked:,}")
    print(f"Coverage: {trials_linked/len(trials)*100:.1f}%")

    print(f"\nTrials per Indication:")
    for name, count in indication_counts.most_common():
        print(f"  {name:40s} {count:>6,}")

    print(f"\nUnique condition strings: {len(all_condition_counts):,}")
    print(f"Unmatched unique conditions: {len(unmatched_conditions):,}")

    print(f"\nTop 50 Unmatched Conditions:")
    for cond, count in unmatched_conditions.most_common(50):
        print(f"  {count:>5,}x  {cond[:80]}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
