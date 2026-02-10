"""
Phase 2b - Step 4: Purple Book Biologics Ingest
Uses FDA Drugs@FDA API to load BLA products for biologics not in Orange Book.
For biologics: 12-year Reference Product Exclusivity (BPCIA) as LOE proxy.
"""

import pyodbc
import requests
import time
import json
from datetime import datetime, timedelta

from db_config import CONN_STR

# Biologics that need Purple Book data (confirmed BLAs)
BIOLOGICS = [
    {"inn": "dulaglutide", "brand": "trulicity", "bla": "BLA125469", "sponsor": "Eli Lilly and Company"},
    {"inn": "insulin aspart", "brand": "novolog", "bla": "BLA021172", "sponsor": "Novo Nordisk Inc"},
    {"inn": "insulin degludec", "brand": "tresiba", "bla": "BLA203314", "sponsor": "Novo Nordisk Inc"},
    {"inn": "insulin glargine", "brand": "lantus", "bla": "BLA021081", "sponsor": "Sanofi-Aventis US LLC"},
    {"inn": "insulin lispro", "brand": "humalog", "bla": "BLA205747", "sponsor": "Eli Lilly and Company"},
]

# BPCIA: 12-year Reference Product Exclusivity for biologics
# 4-year data exclusivity + 12-year market exclusivity from date of first licensure
BPCIA_YEARS = 12


def parse_fda_date(date_str):
    """Parse FDA date string YYYYMMDD to date."""
    if not date_str or len(str(date_str)) < 8:
        return None
    try:
        return datetime.strptime(str(date_str)[:8], "%Y%m%d").date()
    except ValueError:
        return None


def get_fda_data(brand):
    """Get product data from Drugs@FDA API."""
    url = f"https://api.fda.gov/drug/drugsfda.json?search=products.brand_name:{brand}&limit=5"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json().get("results", [])
    except Exception as e:
        print(f"    API error: {e}")
    return []


def main():
    print("=" * 60)
    print("Phase 2b Step 4: Purple Book Biologics Ingest")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    # Load our drugs
    cursor.execute("SELECT drug_id, inn FROM drugs ORDER BY inn")
    drug_lookup = {r[1].lower(): str(r[0]) for r in cursor.fetchall()}

    # Check what's already in ob_products for these drugs
    stats = {"products": 0, "exclusivity": 0}

    for bio in BIOLOGICS:
        inn = bio["inn"]
        drug_id = drug_lookup.get(inn.lower())
        if not drug_id:
            print(f"\n[SKIP] {inn} - not in drugs table")
            continue

        print(f"\n[{inn}] ({bio['bla']})")

        # Get data from FDA API
        time.sleep(1)
        results = get_fda_data(bio["brand"])

        if not results:
            print(f"  No FDA data found for {bio['brand']}")
            continue

        # Find the matching BLA result
        target = None
        for r in results:
            if r.get("application_number", "").replace("BLA", "") == bio["bla"].replace("BLA", ""):
                target = r
                break
        if not target and results:
            target = results[0]

        app_no = target.get("application_number", bio["bla"]).replace("BLA", "")
        sponsor = target.get("sponsor_name", bio["sponsor"])

        # Find original approval date from submissions
        orig_approval = None
        for sub in target.get("submissions", []):
            if sub.get("submission_type") == "ORIG":
                orig_approval = parse_fda_date(sub.get("submission_status_date"))
                break

        print(f"  Application: BLA{app_no}, Sponsor: {sponsor}")
        print(f"  Original Approval: {orig_approval}")

        # Insert products
        for prod in target.get("products", []):
            brand_name = prod.get("brand_name", bio["brand"].upper())
            dosage_form = prod.get("dosage_form", "")
            route = prod.get("route", "")
            df_route = f"{dosage_form};{route}" if dosage_form else ""
            marketing = prod.get("marketing_status", "")

            # Skip discontinued products
            if marketing and "Discontinued" in marketing:
                continue

            ingredients = prod.get("active_ingredients", [])
            ingredient_str = "; ".join(
                f"{i.get('name', '')} {i.get('strength', '')}".strip()
                for i in ingredients
            ) if ingredients else inn.upper()
            strength = ingredients[0].get("strength", "") if ingredients else ""

            product_no = str(stats["products"] + 1).zfill(3)

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
                    app_no, product_no,
                    drug_id, ingredient_str[:500], df_route[:200], brand_name[:200],
                    sponsor[:100] if sponsor else None, sponsor[:500] if sponsor else None,
                    strength[:200], "B",  # B for BLA
                    app_no, product_no,
                    None,  # TE code
                    orig_approval,
                    "Yes",  # RLD = Yes for reference biologics
                    "Yes",  # RS
                    "RX",
                )
                stats["products"] += 1
                print(f"  + Product: {brand_name} ({strength})")
            except Exception as e:
                if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                    print(f"  Insert error: {e}")

        # Calculate BPCIA exclusivity (12 years from original approval)
        if orig_approval:
            bpcia_expiry = orig_approval.replace(year=orig_approval.year + BPCIA_YEARS)
            product_no_excl = "001"

            try:
                cursor.execute(
                    """
                    MERGE ob_exclusivity AS target
                    USING (SELECT ? AS appl_no, ? AS product_no, ? AS excl_code) AS src
                    ON target.appl_no = src.appl_no AND target.product_no = src.product_no
                        AND target.exclusivity_code = src.excl_code
                    WHEN NOT MATCHED THEN
                        INSERT (drug_id, appl_type, appl_no, product_no, exclusivity_code, exclusivity_date)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    app_no, product_no_excl, "BPCIA-12",
                    drug_id, "B", app_no, product_no_excl, "BPCIA-12", bpcia_expiry,
                )
                stats["exclusivity"] += 1
                print(f"  + BPCIA-12 Exclusivity: {bpcia_expiry}")
            except Exception as e:
                if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                    print(f"  Exclusivity insert error: {e}")

            # Also add 4-year data exclusivity
            data_excl = orig_approval.replace(year=orig_approval.year + 4)
            try:
                cursor.execute(
                    """
                    MERGE ob_exclusivity AS target
                    USING (SELECT ? AS appl_no, ? AS product_no, ? AS excl_code) AS src
                    ON target.appl_no = src.appl_no AND target.product_no = src.product_no
                        AND target.exclusivity_code = src.excl_code
                    WHEN NOT MATCHED THEN
                        INSERT (drug_id, appl_type, appl_no, product_no, exclusivity_code, exclusivity_date)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    app_no, product_no_excl, "BPCIA-4",
                    drug_id, "B", app_no, product_no_excl, "BPCIA-4", data_excl,
                )
                stats["exclusivity"] += 1
                print(f"  + BPCIA-4 Data Exclusivity: {data_excl}")
            except Exception as e:
                if "violation" not in str(e).lower() and "duplicate" not in str(e).lower():
                    print(f"  Exclusivity insert error: {e}")

        conn.commit()

    # Summary
    print(f"\n{'='*60}")
    print("Purple Book Ingest Complete")
    print(f"{'='*60}")
    print(f"Products inserted:     {stats['products']}")
    print(f"Exclusivity records:   {stats['exclusivity']}")

    # Verify
    cursor.execute("SELECT COUNT(*) FROM ob_products WHERE appl_type = 'B'")
    print(f"BLA products in DB:    {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM ob_products WHERE appl_type = 'B'")
    print(f"Distinct BLA drugs:    {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT d.inn, p.trade_name, p.appl_no, p.approval_date
        FROM ob_products p JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.appl_type = 'B' AND p.rld = 'Yes'
        ORDER BY d.inn
    """)
    print("\nBiologics loaded:")
    for row in cursor.fetchall():
        print(f"  {row[0]:25s} {row[1]:25s} BLA-{row[2]}  approved={row[3]}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
