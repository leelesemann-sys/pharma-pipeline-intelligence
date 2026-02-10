"""
Phase 2b - Step 3: Orange Book Data Ingest + Drug Matching
Loads products.txt, patent.txt, exclusivity.txt into Azure SQL.
Matches products against our 43 drugs via ingredient name.
"""

import pyodbc
import pandas as pd
from datetime import datetime

from db_config import CONN_STR

DATA_DIR = "orange_book_data"

SALT_SUFFIXES = [
    " HYDROCHLORIDE", " HCL", " PHOSPHATE", " MALEATE", " MESYLATE",
    " SODIUM", " POTASSIUM", " CALCIUM", " SUCCINATE", " FUMARATE",
    " TARTRATE", " SULFATE", " ACETATE", " BESYLATE", " BROMIDE",
    " PROPANEDIOL", " BENZOATE",
]


def parse_date(date_str):
    """Parse 'Jun 21, 2022' or similar FDA date strings to date."""
    if pd.isna(date_str) or not date_str or str(date_str).strip() == "":
        return None
    s = str(date_str).strip()
    # Handle "Approved Prior to Jan 1, 1982" type strings
    if "prior to" in s.lower():
        return datetime(1982, 1, 1).date()
    for fmt in ["%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def strip_salts(ingredient):
    """Remove salt suffixes for matching."""
    name = ingredient.upper().strip()
    for suffix in SALT_SUFFIXES:
        name = name.replace(suffix, "")
    return name.strip()


def match_drugs(ingredient, our_drugs):
    """Match ingredient against our drug list. Returns list of drug_ids."""
    if pd.isna(ingredient):
        return []
    # Split combo products on semicolon
    parts = [p.strip() for p in str(ingredient).split(";")]
    matched = []
    for part in parts:
        stripped = strip_salts(part)
        for drug_id, inn in our_drugs:
            if inn.upper() in stripped:
                matched.append(drug_id)
    return list(set(matched))


def main():
    print("=" * 60)
    print("Phase 2b Step 3: Orange Book Data Ingest")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    # Load our drugs
    cursor.execute("SELECT drug_id, inn FROM drugs ORDER BY inn")
    our_drugs = [(str(r[0]), r[1]) for r in cursor.fetchall()]
    print(f"\nLoaded {len(our_drugs)} drugs from database")

    # -- 3a) Load Products --
    print(f"\n{'-'*40}")
    print("Loading products.txt...")
    products = pd.read_csv(f"{DATA_DIR}/products.txt", sep="~", encoding="latin-1", dtype=str)
    products.columns = products.columns.str.strip()
    print(f"  Parsed {len(products)} product rows")

    # Build drug matching lookup
    product_drug_map = {}  # (appl_no, product_no) -> drug_id
    matched_count = 0
    unmatched_count = 0

    for _, row in products.iterrows():
        ingredient = row.get("Ingredient", "")
        appl_no = str(row.get("Appl_No", "")).strip()
        product_no = str(row.get("Product_No", "")).strip()

        drug_ids = match_drugs(ingredient, our_drugs)
        drug_id = drug_ids[0] if drug_ids else None

        if drug_id:
            product_drug_map[(appl_no, product_no)] = drug_id
            matched_count += 1
        else:
            unmatched_count += 1

        approval_date = parse_date(row.get("Approval_Date"))

        try:
            cursor.execute(
                """
                MERGE ob_products AS target
                USING (SELECT ? AS appl_no, ? AS product_no) AS src
                ON target.appl_no = src.appl_no AND target.product_no = src.product_no
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, ingredient, dosage_form_route, trade_name, applicant,
                            applicant_full_name, strength, appl_type, appl_no, product_no,
                            te_code, approval_date, rld, rs, product_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                appl_no, product_no,
                drug_id, str(ingredient)[:500] if ingredient else None,
                str(row.get("DF;Route", ""))[:200],
                str(row.get("Trade_Name", ""))[:200],
                str(row.get("Applicant", ""))[:100],
                str(row.get("Applicant_Full_Name", ""))[:500],
                str(row.get("Strength", ""))[:200],
                str(row.get("Appl_Type", ""))[:1],
                appl_no, product_no,
                str(row.get("TE_Code", ""))[:10] if row.get("TE_Code") and not pd.isna(row.get("TE_Code")) else None,
                approval_date,
                str(row.get("RLD", ""))[:5] if row.get("RLD") and not pd.isna(row.get("RLD")) else None,
                str(row.get("RS", ""))[:5] if row.get("RS") and not pd.isna(row.get("RS")) else None,
                str(row.get("Type", ""))[:5] if row.get("Type") and not pd.isna(row.get("Type")) else None,
            )
        except Exception as e:
            if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"  Product insert error: {e}")

    conn.commit()
    print(f"  Matched to our drugs: {matched_count}")
    print(f"  No match (other drugs): {unmatched_count}")

    # Show which of our 43 drugs matched
    matched_inns = set()
    for (appl_no, prod_no), did in product_drug_map.items():
        for drug_id, inn in our_drugs:
            if drug_id == did:
                matched_inns.add(inn)
    print(f"  Our drugs found in Orange Book: {len(matched_inns)}/43")
    for inn in sorted(matched_inns):
        print(f"    - {inn}")

    unmatched_inns = set(inn for _, inn in our_drugs) - matched_inns
    print(f"  Our drugs NOT in Orange Book: {len(unmatched_inns)}")
    for inn in sorted(unmatched_inns):
        print(f"    - {inn}")

    # -- 3c) Load Patents --
    print(f"\n{'-'*40}")
    print("Loading patent.txt...")
    patents = pd.read_csv(f"{DATA_DIR}/patent.txt", sep="~", encoding="latin-1", dtype=str)
    patents.columns = patents.columns.str.strip()
    print(f"  Parsed {len(patents)} patent rows")

    patent_inserted = 0
    patent_matched = 0
    for _, row in patents.iterrows():
        appl_no = str(row.get("Appl_No", "")).strip()
        product_no = str(row.get("Product_No", "")).strip()
        patent_no = str(row.get("Patent_No", "")).strip()

        drug_id = product_drug_map.get((appl_no, product_no))
        if drug_id:
            patent_matched += 1

        patent_expire = parse_date(row.get("Patent_Expire_Date_Text"))
        submission_date = parse_date(row.get("Submission_Date"))

        try:
            cursor.execute(
                """
                MERGE ob_patents AS target
                USING (SELECT ? AS appl_no, ? AS product_no, ? AS patent_no) AS src
                ON target.appl_no = src.appl_no AND target.product_no = src.product_no AND target.patent_no = src.patent_no
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, appl_type, appl_no, product_no, patent_no, patent_expire_date,
                            drug_substance_flag, drug_product_flag, patent_use_code, delist_flag, submission_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                appl_no, product_no, patent_no,
                drug_id,
                str(row.get("Appl_Type", ""))[:1],
                appl_no, product_no, patent_no,
                patent_expire,
                str(row.get("Drug_Substance_Flag", ""))[:1] if row.get("Drug_Substance_Flag") and not pd.isna(row.get("Drug_Substance_Flag")) else None,
                str(row.get("Drug_Product_Flag", ""))[:1] if row.get("Drug_Product_Flag") and not pd.isna(row.get("Drug_Product_Flag")) else None,
                str(row.get("Patent_Use_Code", ""))[:20] if row.get("Patent_Use_Code") and not pd.isna(row.get("Patent_Use_Code")) else None,
                str(row.get("Delist_Flag", ""))[:1] if row.get("Delist_Flag") and not pd.isna(row.get("Delist_Flag")) else None,
                submission_date,
            )
            patent_inserted += 1
        except Exception as e:
            if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"  Patent insert error: {e}")

    conn.commit()
    print(f"  Patents inserted: {patent_inserted}")
    print(f"  Patents matching our drugs: {patent_matched}")

    # -- 3d) Load Exclusivity --
    print(f"\n{'-'*40}")
    print("Loading exclusivity.txt...")
    exclusivity = pd.read_csv(f"{DATA_DIR}/exclusivity.txt", sep="~", encoding="latin-1", dtype=str)
    exclusivity.columns = exclusivity.columns.str.strip()
    print(f"  Parsed {len(exclusivity)} exclusivity rows")

    excl_inserted = 0
    excl_matched = 0
    for _, row in exclusivity.iterrows():
        appl_no = str(row.get("Appl_No", "")).strip()
        product_no = str(row.get("Product_No", "")).strip()
        excl_code = str(row.get("Exclusivity_Code", "")).strip()

        drug_id = product_drug_map.get((appl_no, product_no))
        if drug_id:
            excl_matched += 1

        excl_date = parse_date(row.get("Exclusivity_Date"))

        try:
            cursor.execute(
                """
                MERGE ob_exclusivity AS target
                USING (SELECT ? AS appl_no, ? AS product_no, ? AS excl_code) AS src
                ON target.appl_no = src.appl_no AND target.product_no = src.product_no AND target.exclusivity_code = src.excl_code
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, appl_type, appl_no, product_no, exclusivity_code, exclusivity_date)
                    VALUES (?, ?, ?, ?, ?, ?);
                """,
                appl_no, product_no, excl_code,
                drug_id,
                str(row.get("Appl_Type", ""))[:1],
                appl_no, product_no,
                excl_code,
                excl_date,
            )
            excl_inserted += 1
        except Exception as e:
            if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"  Exclusivity insert error: {e}")

    conn.commit()
    print(f"  Exclusivity records inserted: {excl_inserted}")
    print(f"  Exclusivity matching our drugs: {excl_matched}")

    # -- Validation --
    print(f"\n{'-'*40}")
    print("Validation...")
    cursor.execute("SELECT COUNT(*) FROM ob_products")
    print(f"  ob_products: {cursor.fetchone()[0]} rows")
    cursor.execute("SELECT COUNT(*) FROM ob_products WHERE drug_id IS NOT NULL")
    print(f"  ob_products matched: {cursor.fetchone()[0]} rows")
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM ob_products WHERE drug_id IS NOT NULL")
    print(f"  Distinct drugs matched: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM ob_patents")
    print(f"  ob_patents: {cursor.fetchone()[0]} rows")
    cursor.execute("SELECT COUNT(*) FROM ob_exclusivity")
    print(f"  ob_exclusivity: {cursor.fetchone()[0]} rows")

    # Show matched drugs with trade names
    cursor.execute("""
        SELECT d.inn, p.trade_name, p.appl_type, p.appl_no, p.rld
        FROM ob_products p
        JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.rld = 'Yes' AND p.appl_type = 'N'
        ORDER BY d.inn
    """)
    print(f"\n  Our drugs as RLD (Reference Listed Drug) in Orange Book:")
    for row in cursor.fetchall():
        print(f"    {row[0]:25s} {row[1]:25s} NDA-{row[3]}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
