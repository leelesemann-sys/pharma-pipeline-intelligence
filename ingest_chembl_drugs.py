"""
Phase 1, Step 4: ChEMBL Drug Entity Master
Pharma Pipeline Intelligence - Diabetes & Obesity

Loads drug entities from ChEMBL API, then links them to trials
via string matching on raw_interventions.
"""
import requests
import pyodbc
import json
import time
import sys
from datetime import datetime

from db_config import CONN_STR

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Drug list from the prompt
DRUG_NAMES = [
    # GLP-1 RA
    "semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "exenatide",
    # SGLT2 inhibitors
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin", "sotagliflozin",
    # DPP-4 inhibitors
    "sitagliptin", "linagliptin", "saxagliptin", "alogliptin", "vildagliptin",
    # Older classes
    "metformin", "pioglitazone", "rosiglitazone",
    # Insulins
    "insulin glargine", "insulin lispro", "insulin aspart", "insulin degludec", "insulin icodec",
    # Pipeline (next-gen)
    "orforglipron", "danuglipron", "survodutide", "retatrutide", "cagrilintide", "pemvidutide",
    # NASH
    "resmetirom", "obeticholic acid",
]

# Known brand names for matching
BRAND_NAMES = {
    "semaglutide": ["ozempic", "wegovy", "rybelsus"],
    "tirzepatide": ["mounjaro", "zepbound"],
    "liraglutide": ["victoza", "saxenda"],
    "dulaglutide": ["trulicity"],
    "exenatide": ["byetta", "bydureon"],
    "empagliflozin": ["jardiance"],
    "dapagliflozin": ["farxiga", "forxiga"],
    "canagliflozin": ["invokana"],
    "ertugliflozin": ["steglatro"],
    "sotagliflozin": ["inpefa"],
    "sitagliptin": ["januvia"],
    "linagliptin": ["tradjenta", "trajenta"],
    "saxagliptin": ["onglyza"],
    "alogliptin": ["nesina"],
    "vildagliptin": ["galvus"],
    "metformin": ["glucophage"],
    "pioglitazone": ["actos"],
    "rosiglitazone": ["avandia"],
    "insulin glargine": ["lantus", "toujeo", "basaglar"],
    "insulin lispro": ["humalog", "lyumjev"],
    "insulin aspart": ["novolog", "novorapid", "fiasp"],
    "insulin degludec": ["tresiba"],
    "resmetirom": ["rezdiffra"],
}

# MoA class mapping
MOA_CLASSES = {
    "semaglutide": "GLP-1 Receptor Agonist",
    "tirzepatide": "GIP/GLP-1 Dual Agonist",
    "liraglutide": "GLP-1 Receptor Agonist",
    "dulaglutide": "GLP-1 Receptor Agonist",
    "exenatide": "GLP-1 Receptor Agonist",
    "empagliflozin": "SGLT2 Inhibitor",
    "dapagliflozin": "SGLT2 Inhibitor",
    "canagliflozin": "SGLT2 Inhibitor",
    "ertugliflozin": "SGLT2 Inhibitor",
    "sotagliflozin": "SGLT1/SGLT2 Inhibitor",
    "sitagliptin": "DPP-4 Inhibitor",
    "linagliptin": "DPP-4 Inhibitor",
    "saxagliptin": "DPP-4 Inhibitor",
    "alogliptin": "DPP-4 Inhibitor",
    "vildagliptin": "DPP-4 Inhibitor",
    "metformin": "Biguanide",
    "pioglitazone": "Thiazolidinedione (PPAR-gamma)",
    "rosiglitazone": "Thiazolidinedione (PPAR-gamma)",
    "insulin glargine": "Insulin (Basal)",
    "insulin lispro": "Insulin (Rapid-acting)",
    "insulin aspart": "Insulin (Rapid-acting)",
    "insulin degludec": "Insulin (Ultra-long-acting)",
    "insulin icodec": "Insulin (Weekly)",
    "orforglipron": "Oral GLP-1 RA (Small Molecule)",
    "danuglipron": "Oral GLP-1 RA (Small Molecule)",
    "survodutide": "GLP-1/Glucagon Dual Agonist",
    "retatrutide": "GLP-1/GIP/Glucagon Triple Agonist",
    "cagrilintide": "Amylin Analogue",
    "pemvidutide": "GLP-1/Glucagon Dual Agonist",
    "resmetirom": "THR-beta Agonist",
    "obeticholic acid": "FXR Agonist",
}

# Modality mapping from ChEMBL molecule_type
MODALITY_MAP = {
    "Small molecule": "small_molecule",
    "Protein": "peptide",
    "Antibody": "biologic",
    "Oligonucleotide": "oligonucleotide",
    "Cell": "cell_therapy",
    "Unknown": "other",
    "Enzyme": "biologic",
    "Gene": "gene_therapy",
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

        # Try to find exact match on pref_name
        for mol in molecules:
            if mol.get("pref_name", "").lower() == drug_name.lower():
                return mol

        # Fallback: first result
        return molecules[0]
    except Exception as e:
        print(f"  ERROR fetching {drug_name}: {e}")
        return None


def main():
    print("=" * 70)
    print("ChEMBL DRUG ENTITY MASTER - Diabetes & Obesity")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Connect
    print("\nConnecting to Azure SQL...")
    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Load drugs from ChEMBL
    print(f"\n--- Loading {len(DRUG_NAMES)} drugs from ChEMBL ---")
    drugs_loaded = 0
    drugs_not_found = []

    for drug_name in DRUG_NAMES:
        print(f"\n  Fetching: {drug_name}...", end=" ")
        mol = fetch_chembl_molecule(drug_name)
        time.sleep(0.5)  # Be nice to ChEMBL

        if mol is None:
            print("NOT FOUND in ChEMBL")
            drugs_not_found.append(drug_name)
            # Still insert with basic info from our known data
            brands = BRAND_NAMES.get(drug_name, [])
            cursor.execute("""
                INSERT INTO drugs (inn, brand_names, moa_class, modality, targets)
                VALUES (?, ?, ?, ?, ?)
            """, (
                drug_name,
                json.dumps(brands) if brands else None,
                MOA_CLASSES.get(drug_name, ""),
                "peptide" if "insulin" in drug_name.lower() else "small_molecule",
                None,
            ))
            drugs_loaded += 1
            print(f"-> Inserted (manual, no ChEMBL match)")
            continue

        chembl_id = mol.get("molecule_chembl_id", "")
        pref_name = mol.get("pref_name", drug_name)
        mol_type = mol.get("molecule_type", "Unknown")
        modality = MODALITY_MAP.get(mol_type, "other")

        # ATC
        atc_list = mol.get("atc_classifications", [])
        atc_code = atc_list[0] if atc_list else None

        # Cross references
        cross_refs = mol.get("cross_references", [])

        # Brand names from our list + ChEMBL
        brands = BRAND_NAMES.get(drug_name, [])

        # Targets (from cross_refs or our known data)
        targets_str = None
        if drug_name in MOA_CLASSES:
            # Derive targets from MoA class
            target_map = {
                "GLP-1 Receptor Agonist": ["GLP1R"],
                "GIP/GLP-1 Dual Agonist": ["GLP1R", "GIPR"],
                "SGLT2 Inhibitor": ["SLC5A2"],
                "SGLT1/SGLT2 Inhibitor": ["SLC5A1", "SLC5A2"],
                "DPP-4 Inhibitor": ["DPP4"],
                "Biguanide": ["AMPK"],
                "Thiazolidinedione (PPAR-gamma)": ["PPARG"],
                "Insulin (Basal)": ["INSR"],
                "Insulin (Rapid-acting)": ["INSR"],
                "Insulin (Ultra-long-acting)": ["INSR"],
                "Insulin (Weekly)": ["INSR"],
                "Oral GLP-1 RA (Small Molecule)": ["GLP1R"],
                "GLP-1/Glucagon Dual Agonist": ["GLP1R", "GCGR"],
                "GLP-1/GIP/Glucagon Triple Agonist": ["GLP1R", "GIPR", "GCGR"],
                "Amylin Analogue": ["CALCR", "RAMP1", "RAMP2", "RAMP3"],
                "THR-beta Agonist": ["THRB"],
                "FXR Agonist": ["NR1H4"],
            }
            moa = MOA_CLASSES[drug_name]
            targets = target_map.get(moa, [])
            targets_str = json.dumps(targets) if targets else None

        # Check if drug already exists
        cursor.execute("SELECT drug_id FROM drugs WHERE inn = ?", (drug_name,))
        existing = cursor.fetchone()
        if existing:
            print(f"EXISTS, updating ChEMBL data")
            cursor.execute("""
                UPDATE drugs SET chembl_id=?, atc_code=?, modality=?, brand_names=?, moa_class=?, targets=?, updated_at=GETUTCDATE()
                WHERE inn = ?
            """, (chembl_id, atc_code, modality, json.dumps(brands) if brands else None,
                  MOA_CLASSES.get(drug_name, ""), targets_str, drug_name))
        else:
            cursor.execute("""
                INSERT INTO drugs (inn, chembl_id, atc_code, modality, brand_names, moa_class, targets)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                drug_name, chembl_id, atc_code, modality,
                json.dumps(brands) if brands else None,
                MOA_CLASSES.get(drug_name, ""),
                targets_str,
            ))
            drugs_loaded += 1
            print(f"OK ({chembl_id}, ATC: {atc_code}, {modality})")

    print(f"\n\n--- Drug loading complete ---")
    print(f"Loaded: {drugs_loaded}")
    print(f"Not found in ChEMBL: {len(drugs_not_found)}")
    if drugs_not_found:
        print(f"  Missing: {', '.join(drugs_not_found)}")

    # Step 2: Link drugs to trials via string matching
    print(f"\n{'='*70}")
    print("DRUG-TRIAL LINKING via String Matching")
    print(f"{'='*70}")

    # Get all drugs with their names and brands
    cursor.execute("SELECT drug_id, inn, brand_names FROM drugs")
    drugs = []
    for row in cursor.fetchall():
        drug_id, inn, brand_names_json = row
        search_terms = [inn.lower()] if inn else []
        if brand_names_json:
            try:
                brands = json.loads(brand_names_json)
                search_terms.extend([b.lower() for b in brands])
            except:
                pass
        drugs.append((drug_id, inn, search_terms))

    print(f"Drugs to match: {len(drugs)}")

    # Get all trials with interventions
    cursor.execute("SELECT trial_id, nct_id, raw_interventions FROM trials WHERE raw_interventions IS NOT NULL")
    trials = cursor.fetchall()
    print(f"Trials with interventions: {len(trials)}")

    # Match
    links_created = 0
    links_by_drug = {}

    for trial_id, nct_id, raw_interv in trials:
        if not raw_interv:
            continue
        interv_lower = raw_interv.lower()

        for drug_id, inn, search_terms in drugs:
            matched = False
            for term in search_terms:
                if term and len(term) >= 3 and term in interv_lower:
                    matched = True
                    break

            if matched:
                try:
                    cursor.execute("""
                        INSERT INTO drug_trials (drug_id, trial_id, role)
                        VALUES (?, ?, 'experimental')
                    """, (drug_id, trial_id))
                    links_created += 1
                    links_by_drug[inn] = links_by_drug.get(inn, 0) + 1
                except pyodbc.IntegrityError:
                    pass  # Already linked

    print(f"\nDrug-Trial links created: {links_created:,}")

    # Report
    print(f"\n{'='*70}")
    print("DRUG-TRIAL LINKING REPORT")
    print(f"{'='*70}")

    # Top drugs by trial count
    print("\nTop 10 Drugs by Trial Count:")
    sorted_drugs = sorted(links_by_drug.items(), key=lambda x: x[1], reverse=True)
    for i, (drug, count) in enumerate(sorted_drugs[:10], 1):
        print(f"  {i:2d}. {drug:30s} {count:>5,} trials")

    # Drugs without trials
    print("\nDrugs without trial links:")
    cursor.execute("""
        SELECT d.inn FROM drugs d
        LEFT JOIN drug_trials dt ON d.drug_id = dt.drug_id
        WHERE dt.drug_id IS NULL
    """)
    unlinked = [row[0] for row in cursor.fetchall()]
    if unlinked:
        for d in unlinked:
            print(f"  - {d}")
    else:
        print("  (none - all drugs linked!)")

    # Total stats
    print(f"\n--- Final Stats ---")
    cursor.execute("SELECT COUNT(*) FROM drugs")
    print(f"Total drugs: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM drug_trials")
    print(f"Total drug-trial links: {cursor.fetchone()[0]}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
