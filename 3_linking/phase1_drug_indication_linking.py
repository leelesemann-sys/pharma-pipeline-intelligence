"""
Phase 1 Completion - Aufgabe 2: Drug -> Indication Linking
Source A: ChEMBL Drug Indication API
Source B: Transitive from drug_trials + trial_indications
"""
import requests
import pyodbc
import json
import time
from datetime import datetime
from collections import defaultdict

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Same synonym logic for matching ChEMBL mesh_heading to our indications
INDICATION_KEYWORDS = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes", "diabetes mellitus, type 2", "t2dm", "niddm", "non-insulin-dependent"],
    "Type 1 Diabetes Mellitus": ["type 1 diabetes", "diabetes mellitus, type 1", "t1dm", "iddm", "insulin-dependent", "juvenile diabetes"],
    "Obesity": ["obesity", "obese"],
    "Overweight": ["overweight"],
    "NASH / MASH": ["steatohepatitis", "nash", "mash"],
    "Non-alcoholic Fatty Liver Disease": ["fatty liver", "nafld", "mafld", "masld", "hepatic steatosis"],
    "Diabetic Kidney Disease": ["diabetic nephro", "diabetic kidney", "dkd"],
    "Diabetic Retinopathy": ["diabetic retino", "diabetic macular", "diabetic eye"],
    "Diabetic Neuropathy": ["diabetic neuro", "diabetic peripheral neuro", "diabetic foot"],
    "Metabolic Syndrome": ["metabolic syndrome"],
    "Prediabetes": ["prediabet", "pre-diabet", "impaired glucose", "glucose intolerance"],
    "Gestational Diabetes": ["gestational diabetes"],
}

PHASE_MAP_CHEMBL = {4: "phase4", 3: "phase3", 2: "phase2", 1: "phase1", 0.5: "early_phase1"}
PHASE_PRIORITY = {"early_phase1": 1, "phase1": 2, "phase1_phase2": 3, "phase2": 4, "phase2_phase3": 5, "phase3": 6, "phase4": 7, "na": 0}


def match_mesh_to_indication(mesh_heading, indication_map):
    """Match a ChEMBL mesh_heading to our indication names."""
    mesh_lower = mesh_heading.lower()
    for ind_name, keywords in INDICATION_KEYWORDS.items():
        for kw in keywords:
            if kw in mesh_lower:
                if ind_name in indication_map:
                    return ind_name
    return None


def main():
    print("=" * 70)
    print("AUFGABE 2: Drug -> Indication Linking")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Load indication map
    cursor.execute("SELECT indication_id, name FROM indications WHERE name != 'Diabetes & Metabolism'")
    indication_map = {row[1]: row[0] for row in cursor.fetchall()}
    print(f"Indications: {len(indication_map)}")

    # Load drugs
    cursor.execute("SELECT drug_id, inn, chembl_id FROM drugs")
    drugs = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    print(f"Drugs: {len(drugs)}")

    # =================================================
    # SOURCE A: ChEMBL Drug Indication API
    # =================================================
    print(f"\n{'='*70}")
    print("SOURCE A: ChEMBL Drug Indication API")
    print(f"{'='*70}")

    chembl_links = {}  # (drug_id, ind_name) -> {status, phase}
    chembl_count = 0

    for drug_id, inn, chembl_id in drugs:
        if not chembl_id:
            print(f"  {inn}: no ChEMBL ID, skipping API")
            continue

        url = f"{CHEMBL_BASE}/drug_indication?molecule_chembl_id={chembl_id}&format=json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"  {inn} ({chembl_id}): no indications in ChEMBL")
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  {inn} ({chembl_id}): ERROR - {e}")
            continue

        time.sleep(0.5)
        drug_indications = data.get("drug_indications", [])

        matched = 0
        for di in drug_indications:
            mesh = di.get("mesh_heading", "")
            max_phase = di.get("max_phase_for_ind", 0)
            if not mesh:
                continue

            ind_name = match_mesh_to_indication(mesh, indication_map)
            if ind_name:
                try:
                    max_phase_num = float(max_phase) if max_phase else 0
                except (ValueError, TypeError):
                    max_phase_num = 0
                status = "approved" if max_phase_num >= 4 else "investigational"
                phase_str = PHASE_MAP_CHEMBL.get(max_phase_num, f"phase{int(max_phase_num)}" if max_phase_num else "na")
                key = (drug_id, ind_name)
                # Keep highest phase
                if key not in chembl_links or PHASE_PRIORITY.get(phase_str, 0) > PHASE_PRIORITY.get(chembl_links[key]["phase"], 0):
                    chembl_links[key] = {"status": status, "phase": phase_str}
                matched += 1

        if matched > 0:
            chembl_count += matched
            print(f"  {inn}: {matched} indication matches from ChEMBL ({len(drug_indications)} total in ChEMBL)")
        else:
            print(f"  {inn}: {len(drug_indications)} ChEMBL indications, 0 matched our scope")

    print(f"\nChEMBL matches: {len(chembl_links)} unique drug-indication pairs")

    # Insert ChEMBL links
    chembl_inserted = 0
    for (drug_id, ind_name), info in chembl_links.items():
        ind_id = indication_map[ind_name]
        try:
            cursor.execute("""
                INSERT INTO drug_indications (drug_id, indication_id, status, phase)
                VALUES (?, ?, ?, ?)
            """, (drug_id, ind_id, info["status"], info["phase"]))
            chembl_inserted += 1
        except pyodbc.IntegrityError:
            pass
    print(f"ChEMBL links inserted: {chembl_inserted}")

    # =================================================
    # SOURCE B: Transitive from drug_trials + trial_indications
    # =================================================
    print(f"\n{'='*70}")
    print("SOURCE B: Transitive Derivation")
    print(f"{'='*70}")

    cursor.execute("""
        SELECT DISTINCT
            dt.drug_id,
            ti.indication_id,
            MAX(
                CASE t.phase
                    WHEN 'phase4' THEN 7
                    WHEN 'phase3' THEN 6
                    WHEN 'phase2_phase3' THEN 5
                    WHEN 'phase2' THEN 4
                    WHEN 'phase1_phase2' THEN 3
                    WHEN 'phase1' THEN 2
                    WHEN 'early_phase1' THEN 1
                    ELSE 0
                END
            ) AS max_phase_num
        FROM drug_trials dt
        JOIN trials t ON dt.trial_id = t.trial_id
        JOIN trial_indications ti ON t.trial_id = ti.trial_id
        WHERE t.overall_status NOT IN ('withdrawn')
        GROUP BY dt.drug_id, ti.indication_id
    """)
    transitive_rows = cursor.fetchall()
    print(f"Transitive drug-indication pairs found: {len(transitive_rows)}")

    phase_num_map = {7: "phase4", 6: "phase3", 5: "phase2_phase3", 4: "phase2", 3: "phase1_phase2", 2: "phase1", 1: "early_phase1", 0: "na"}
    transitive_inserted = 0
    transitive_updated = 0

    for drug_id, ind_id, max_phase_num in transitive_rows:
        phase_str = phase_num_map.get(max_phase_num, "na")

        # Check if already exists from ChEMBL
        cursor.execute("SELECT status, phase FROM drug_indications WHERE drug_id = ? AND indication_id = ?",
                        (drug_id, ind_id))
        existing = cursor.fetchone()

        if existing:
            # Update if transitive shows higher phase
            existing_priority = PHASE_PRIORITY.get(existing[1], 0)
            new_priority = PHASE_PRIORITY.get(phase_str, 0)
            if new_priority > existing_priority:
                cursor.execute("""
                    UPDATE drug_indications SET phase = ? WHERE drug_id = ? AND indication_id = ?
                """, (phase_str, drug_id, ind_id))
                transitive_updated += 1
        else:
            # Check if drug has an approval for this indication
            cursor.execute("""
                SELECT COUNT(*) FROM approvals WHERE drug_id = ? AND indication_id = ?
            """, (drug_id, ind_id))
            has_approval = cursor.fetchone()[0] > 0
            status = "approved" if has_approval else "investigational"

            try:
                cursor.execute("""
                    INSERT INTO drug_indications (drug_id, indication_id, status, phase)
                    VALUES (?, ?, ?, ?)
                """, (drug_id, ind_id, status, phase_str))
                transitive_inserted += 1
            except pyodbc.IntegrityError:
                pass

    print(f"Transitive links inserted: {transitive_inserted}")
    print(f"Existing links updated (higher phase): {transitive_updated}")

    # =================================================
    # REPORT
    # =================================================
    print(f"\n{'='*70}")
    print("DRUG-INDICATION REPORT")
    print(f"{'='*70}")

    cursor.execute("SELECT COUNT(*) FROM drug_indications")
    total_links = cursor.fetchone()[0]
    print(f"\nTotal drug-indication links: {total_links}")
    print(f"  From ChEMBL: {chembl_inserted}")
    print(f"  From transitive: {transitive_inserted}")
    print(f"  Updated: {transitive_updated}")

    # Drug x Indication matrix
    print(f"\nDrug-Indication Matrix:")
    cursor.execute("""
        SELECT d.inn, i.name, di.status, di.phase
        FROM drug_indications di
        JOIN drugs d ON di.drug_id = d.drug_id
        JOIN indications i ON di.indication_id = i.indication_id
        ORDER BY d.inn, i.name
    """)
    current_drug = None
    for inn, ind_name, status, phase in cursor.fetchall():
        if inn != current_drug:
            print(f"\n  {inn}:")
            current_drug = inn
        status_icon = "V" if status == "approved" else "?"
        print(f"    [{status_icon}] {ind_name:40s} {phase:15s} ({status})")

    # Drugs without indication links
    cursor.execute("""
        SELECT d.inn FROM drugs d
        LEFT JOIN drug_indications di ON d.drug_id = di.drug_id
        WHERE di.drug_id IS NULL
    """)
    unlinked = [row[0] for row in cursor.fetchall()]
    print(f"\nDrugs without indication links: {len(unlinked)}")
    for d in unlinked:
        print(f"  - {d}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
