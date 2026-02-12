"""
Phase 1 Nachtrag: Fehlende Wirkstoffklassen nachladen
ChEMBL Drug Master + Trial Linking + openFDA Approvals
"""
import requests
import pyodbc
import json
import time
from datetime import datetime

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_BASE = "https://api.fda.gov/drug/drugsfda.json"

# New drugs to load
NEW_DRUGS = [
    # Sulfonylharnstoffe
    {"inn": "glimepiride",   "moa_class": "Sulfonylharnstoff",           "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["amaryl"]},
    {"inn": "glipizide",     "moa_class": "Sulfonylharnstoff",           "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["glucotrol"]},
    {"inn": "glyburide",     "moa_class": "Sulfonylharnstoff",           "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["diabeta","micronase","glynase"],
     "aliases": ["glibenclamide"]},
    {"inn": "gliclazide",    "moa_class": "Sulfonylharnstoff",           "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["diamicron"]},
    # Alpha-Glucosidase-Inhibitoren
    {"inn": "acarbose",      "moa_class": "Alpha-Glucosidase-Inhibitor", "modality": "small_molecule", "targets": ["GAA"],  "brands": ["precose","glucobay"]},
    {"inn": "miglitol",      "moa_class": "Alpha-Glucosidase-Inhibitor", "modality": "small_molecule", "targets": ["GAA"],  "brands": ["glyset"]},
    # Meglitinide
    {"inn": "repaglinide",   "moa_class": "Meglitinid",                  "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["prandin"]},
    {"inn": "nateglinide",   "moa_class": "Meglitinid",                  "modality": "small_molecule", "targets": ["ABCC8","KCNJ11"], "brands": ["starlix"]},
    # Amylin-Analog
    {"inn": "pramlintide",   "moa_class": "Amylin Analogue",             "modality": "peptide",        "targets": ["CALCR","RAMP1","RAMP2","RAMP3"], "brands": ["symlin"]},
    # Dopamin-Agonist
    {"inn": "bromocriptine", "moa_class": "Dopamin-Agonist (D2)",        "modality": "small_molecule", "targets": ["DRD2"], "brands": ["cycloset","parlodel"]},
    # Bile Acid Sequestrant
    {"inn": "colesevelam",   "moa_class": "Bile Acid Sequestrant",       "modality": "small_molecule", "targets": ["Bile acids"], "brands": ["welchol"]},
    # SGLT2 fehlend
    {"inn": "bexagliflozin", "moa_class": "SGLT2 Inhibitor",             "modality": "small_molecule", "targets": ["SLC5A2"], "brands": ["brenzavvy"]},
]

MODALITY_MAP = {
    "Small molecule": "small_molecule",
    "Protein": "peptide",
    "Antibody": "biologic",
    "Unknown": "other",
}


def fetch_chembl_molecule(drug_name):
    """Fetch molecule data from ChEMBL."""
    url = f"{CHEMBL_BASE}/molecule/search?q={drug_name}&format=json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        molecules = data.get("molecules", [])
        if not molecules:
            return None
        for mol in molecules:
            pref = mol.get("pref_name")
            if pref and pref.lower() == drug_name.lower():
                return mol
        return molecules[0]
    except Exception as e:
        print(f"  ERROR fetching {drug_name}: {e}")
        return None


def parse_fda_date(date_str):
    if not date_str:
        return None
    try:
        date_str = date_str.replace("-", "")
        return datetime.strptime(date_str[:8], "%Y%m%d").date()
    except:
        return None


def fetch_fda_approvals(drug_inn):
    search_name = drug_inn.replace(" ", "+")
    url = f'{OPENFDA_BASE}?search=openfda.generic_name:"{search_name}"&limit=10'
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json().get("results", [])
    except:
        return []


def main():
    print("=" * 70)
    print("PHASE 1 NACHTRAG: Fehlende Wirkstoffklassen")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected to Azure SQL!")

    # Pre-check
    cursor.execute("SELECT COUNT(*) FROM drugs")
    drugs_before = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM drug_trials")
    links_before = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM approvals")
    approvals_before = cursor.fetchone()[0]
    print(f"\nBefore: {drugs_before} drugs, {links_before} drug-trial links, {approvals_before} approvals")

    # =========================================================
    # STEP 1: Load drugs from ChEMBL
    # =========================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: Loading {len(NEW_DRUGS)} new drugs from ChEMBL")
    print(f"{'='*70}")

    drugs_added = 0
    drugs_skipped = 0
    drug_ids = {}  # inn -> drug_id

    for drug_info in NEW_DRUGS:
        inn = drug_info["inn"]
        print(f"\n  {inn}...", end=" ")

        # Check if exists
        cursor.execute("SELECT drug_id FROM drugs WHERE inn = ?", (inn,))
        existing = cursor.fetchone()
        if existing:
            print("EXISTS (skipping)")
            drug_ids[inn] = existing[0]
            drugs_skipped += 1
            continue

        # Fetch from ChEMBL
        mol = fetch_chembl_molecule(inn)
        time.sleep(0.5)

        chembl_id = None
        atc_code = None
        if mol:
            chembl_id = mol.get("molecule_chembl_id", "")
            atc_list = mol.get("atc_classifications", [])
            atc_code = atc_list[0] if atc_list else None

        brands_json = json.dumps(drug_info["brands"]) if drug_info.get("brands") else None
        targets_json = json.dumps(drug_info["targets"]) if drug_info.get("targets") else None

        cursor.execute("""
            INSERT INTO drugs (inn, chembl_id, atc_code, modality, brand_names, moa_class, targets)
            OUTPUT INSERTED.drug_id
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            inn, chembl_id, atc_code, drug_info["modality"],
            brands_json, drug_info["moa_class"], targets_json
        ))
        new_id = cursor.fetchone()[0]
        drug_ids[inn] = new_id
        drugs_added += 1

        chembl_str = chembl_id if chembl_id else "no ChEMBL"
        atc_str = atc_code if atc_code else "no ATC"
        print(f"OK ({chembl_str}, {atc_str})")

    print(f"\nDrugs added: {drugs_added}, skipped: {drugs_skipped}")

    # =========================================================
    # STEP 2: Drug-Trial Linking
    # =========================================================
    print(f"\n{'='*70}")
    print("STEP 2: Drug-Trial Linking")
    print(f"{'='*70}")

    # Build search terms for new drugs
    search_drugs = []
    for drug_info in NEW_DRUGS:
        inn = drug_info["inn"]
        if inn not in drug_ids:
            continue
        drug_id = drug_ids[inn]
        terms = [inn.lower()]
        if drug_info.get("brands"):
            terms.extend([b.lower() for b in drug_info["brands"]])
        if drug_info.get("aliases"):
            terms.extend([a.lower() for a in drug_info["aliases"]])
        search_drugs.append((drug_id, inn, terms))

    print(f"Searching {len(search_drugs)} drugs against trial interventions...")

    # Get all trials
    cursor.execute("SELECT trial_id, raw_interventions FROM trials WHERE raw_interventions IS NOT NULL")
    trials = cursor.fetchall()
    print(f"Trials to scan: {len(trials)}")

    new_links = 0
    links_by_drug = {}

    for trial_id, raw_interv in trials:
        if not raw_interv:
            continue
        interv_lower = raw_interv.lower()

        for drug_id, inn, terms in search_drugs:
            matched = False
            for term in terms:
                if term and len(term) >= 3 and term in interv_lower:
                    matched = True
                    break
            if matched:
                try:
                    cursor.execute("""
                        INSERT INTO drug_trials (drug_id, trial_id, role)
                        VALUES (?, ?, 'comparator')
                    """, (drug_id, trial_id))
                    new_links += 1
                    links_by_drug[inn] = links_by_drug.get(inn, 0) + 1
                except pyodbc.IntegrityError:
                    pass

    print(f"\nNew drug-trial links: {new_links:,}")
    print("\nLinks per drug:")
    for drug, count in sorted(links_by_drug.items(), key=lambda x: x[1], reverse=True):
        print(f"  {drug:20s} {count:>5,} trials")

    # =========================================================
    # STEP 3: openFDA Approvals
    # =========================================================
    print(f"\n{'='*70}")
    print("STEP 3: openFDA Approvals")
    print(f"{'='*70}")

    new_approvals = 0
    drugs_with_fda = 0

    for drug_info in NEW_DRUGS:
        inn = drug_info["inn"]
        if inn not in drug_ids:
            continue
        drug_id = drug_ids[inn]

        print(f"  {inn}...", end=" ")
        results = fetch_fda_approvals(inn)
        time.sleep(0.3)

        # Also try aliases
        if not results and drug_info.get("aliases"):
            for alias in drug_info["aliases"]:
                results = fetch_fda_approvals(alias)
                time.sleep(0.3)
                if results:
                    break

        if not results:
            print("No FDA data")
            continue

        drug_approval_count = 0
        for result in results:
            app_number = result.get("application_number", "")
            submissions = result.get("submissions", [])
            for sub in submissions:
                if sub.get("submission_status", "").upper() == "AP":
                    sub_date = parse_fda_date(sub.get("submission_status_date", ""))
                    review_type = "standard"
                    if "PRIORITY" in sub.get("review_priority", "").upper():
                        review_type = "priority"
                    try:
                        cursor.execute("""
                            INSERT INTO approvals (drug_id, country, agency, application_number, approval_date, review_type)
                            VALUES (?, 'US', 'FDA', ?, ?, ?)
                        """, (drug_id, app_number, sub_date, review_type))
                        drug_approval_count += 1
                        new_approvals += 1
                    except pyodbc.IntegrityError:
                        pass

        if drug_approval_count > 0:
            drugs_with_fda += 1
            cursor.execute("""
                UPDATE drugs SET first_approval_date = (
                    SELECT MIN(approval_date) FROM approvals WHERE drug_id = ? AND approval_date IS NOT NULL
                ), highest_phase = 'approved', updated_at = GETUTCDATE()
                WHERE drug_id = ?
            """, (drug_id, drug_id))
            print(f"{drug_approval_count} approvals")
        else:
            print("No approval records")

    # =========================================================
    # FINAL REPORT
    # =========================================================
    print(f"\n{'='*70}")
    print("NACHTRAG REPORT")
    print(f"{'='*70}")

    cursor.execute("SELECT COUNT(*) FROM drugs")
    drugs_after = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM drug_trials")
    links_after = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM approvals")
    approvals_after = cursor.fetchone()[0]

    print(f"\n{'Metrik':<35s} {'Vorher':>8s} {'Nachher':>8s} {'Diff':>8s}")
    print("-" * 65)
    print(f"{'Drugs':<35s} {drugs_before:>8,} {drugs_after:>8,} {drugs_after-drugs_before:>+8,}")
    print(f"{'Drug-Trial Links':<35s} {links_before:>8,} {links_after:>8,} {links_after-links_before:>+8,}")
    print(f"{'FDA Approval Records':<35s} {approvals_before:>8,} {approvals_after:>8,} {approvals_after-approvals_before:>+8,}")

    print(f"\nDrugs mit FDA-Approvals (neu): {drugs_with_fda}")
    print(f"Neue Approval Records: {new_approvals}")

    # Updated top 15 drugs by trial count
    print(f"\nUpdated Top 15 Drugs by Trial Count:")
    cursor.execute("""
        SELECT TOP 15 d.inn, d.moa_class, COUNT(dt.trial_id) AS cnt
        FROM drugs d
        JOIN drug_trials dt ON d.drug_id = dt.drug_id
        GROUP BY d.inn, d.moa_class
        ORDER BY cnt DESC
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"  {i:2d}. {row[0]:25s} {row[1]:35s} {row[2]:>5,}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
