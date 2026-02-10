"""
Phase 3 - Step 4: CMS Medicare Part D Spending Ingest
Loads US drug spending data (2019-2023) for all 43 drugs.
"""

import pyodbc
import requests
import time
import re

from db_config import CONN_STR

CMS_ENDPOINT = "https://data.cms.gov/data-api/v1/dataset/7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b/data"
PAUSE = 1.0
MAX_RETRIES = 3

# Known salt suffixes for CMS name matching
SALT_SUFFIXES = {
    "metformin": ["METFORMIN HCL", "METFORMIN HYDROCHLORIDE"],
    "sitagliptin": ["SITAGLIPTIN PHOSPHATE"],
    "glimepiride": ["GLIMEPIRIDE"],
    "glipizide": ["GLIPIZIDE"],
    "glyburide": ["GLYBURIDE"],
    "pioglitazone": ["PIOGLITAZONE HCL", "PIOGLITAZONE HYDROCHLORIDE"],
    "rosiglitazone": ["ROSIGLITAZONE MALEATE"],
    "acarbose": ["ACARBOSE"],
    "nateglinide": ["NATEGLINIDE"],
    "repaglinide": ["REPAGLINIDE"],
    "colesevelam": ["COLESEVELAM HCL"],
    "bromocriptine": ["BROMOCRIPTINE MESYLATE"],
    "pramlintide": ["PRAMLINTIDE ACETATE"],
}

YEARS = [2019, 2020, 2021, 2022, 2023]


def api_get(url, retries=MAX_RETRIES):
    """GET with retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                # CMS returns list directly or dict with data key
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "data" in data:
                    return data["data"]
                return data
            elif r.status_code == 429:
                time.sleep(30 * (attempt + 1))
            else:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_number(val):
    """Parse CMS number strings (may have commas, dollar signs, or be empty)."""
    if val is None or val == "" or val == "null" or str(val).lower() == "suppressed":
        return None
    s = str(val).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_int(val):
    """Parse to int or None."""
    n = parse_number(val)
    return int(n) if n is not None else None


def search_cms(inn):
    """Search CMS for a drug, trying exact INN then salt variants."""
    # Try 1: Exact INN uppercase
    inn_upper = inn.upper()
    url = f"{CMS_ENDPOINT}?filter[Gnrc_Name]={inn_upper}"
    data = api_get(url)
    if data and len(data) > 0:
        return data, inn_upper

    # Try 2: Known salt suffixes
    if inn.lower() in SALT_SUFFIXES:
        for variant in SALT_SUFFIXES[inn.lower()]:
            time.sleep(0.5)
            url = f"{CMS_ENDPOINT}?filter[Gnrc_Name]={variant}"
            data = api_get(url)
            if data and len(data) > 0:
                return data, variant

    # Try 3: Partial match via keyword parameter if available
    # CMS API may support keyword search
    time.sleep(0.5)
    url = f"{CMS_ENDPOINT}?keyword={inn_upper}"
    data = api_get(url)
    if data and isinstance(data, list):
        # Filter results to match our drug
        matches = [r for r in data if inn_upper in str(r.get("Gnrc_Name", "")).upper()]
        if matches:
            return matches, matches[0].get("Gnrc_Name", inn_upper)

    return None, None


def pivot_and_insert(cursor, drug_id, records, cms_name):
    """Pivot wide-format CMS data to long format and insert."""
    total_inserted = 0

    for record in records:
        brand = record.get("Brnd_Name", "Unknown")
        gnrc = record.get("Gnrc_Name", cms_name)
        mftr = record.get("Mftr_Name") or record.get("Tot_Mftr") or "Unknown"

        for year in YEARS:
            spending = parse_number(record.get(f"Tot_Spndng_{year}"))
            units = parse_int(record.get(f"Tot_Dsg_Unts_{year}"))
            claims = parse_int(record.get(f"Tot_Clms_{year}"))
            benes = parse_int(record.get(f"Tot_Benes_{year}"))
            cpu = parse_number(record.get(f"Avg_Spnd_Per_Dsg_Unt_Wghtd_{year}"))
            cpc = parse_number(record.get(f"Avg_Spnd_Per_Clm_{year}"))
            cpb = parse_number(record.get(f"Avg_Spnd_Per_Bene_{year}"))

            # Skip rows where all values are None
            if all(v is None for v in [spending, units, claims, benes]):
                continue

            try:
                cursor.execute(
                    """
                    MERGE spending_us AS target
                    USING (SELECT ? AS drug_id, ? AS brand_name, ? AS year) AS src
                    ON target.drug_id = src.drug_id AND target.brand_name = src.brand_name AND target.year = src.year
                    WHEN MATCHED THEN
                        UPDATE SET total_spending = ?, total_dosage_units = ?, total_claims = ?,
                                   total_beneficiaries = ?, avg_cost_per_unit = ?,
                                   avg_cost_per_claim = ?, avg_cost_per_beneficiary = ?,
                                   generic_name_cms = ?, manufacturer = ?
                    WHEN NOT MATCHED THEN
                        INSERT (drug_id, brand_name, generic_name_cms, manufacturer, year,
                                total_spending, total_dosage_units, total_claims, total_beneficiaries,
                                avg_cost_per_unit, avg_cost_per_claim, avg_cost_per_beneficiary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    drug_id, brand, year,
                    spending, units, claims, benes, cpu, cpc, cpb, gnrc, mftr,
                    drug_id, brand, gnrc, mftr, year,
                    spending, units, claims, benes, cpu, cpc, cpb,
                )
                total_inserted += 1
            except Exception as e:
                if "duplicate" not in str(e).lower() and "violation" not in str(e).lower():
                    print(f"    Insert error: {e}")

    return total_inserted


def main():
    print("=" * 60)
    print("Phase 3 Step 4: CMS Medicare Part D Spending Ingest")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    cursor.execute("SELECT drug_id, inn, moa_class FROM drugs ORDER BY inn")
    drugs = cursor.fetchall()
    print(f"\nLoaded {len(drugs)} drugs\n")

    stats = {"found": 0, "skipped": 0, "total_records": 0}
    skipped = []
    found = []

    for i, (drug_id, inn, moa_class) in enumerate(drugs):
        print(f"[{i+1:2d}/43] {inn} ({moa_class})")

        time.sleep(PAUSE)
        records, cms_name = search_cms(inn)

        if not records:
            print(f"    -> No CMS data found")
            stats["skipped"] += 1
            skipped.append((inn, "Not in Medicare Part D"))
            continue

        print(f"    CMS name: {cms_name}, {len(records)} brand records")

        # Save CMS external ID
        try:
            cursor.execute(
                """
                MERGE drug_external_ids AS target
                USING (SELECT ? AS drug_id, ? AS source, ? AS external_id) AS src
                ON target.drug_id = src.drug_id AND target.source = src.source AND target.external_id = src.external_id
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, source, external_id) VALUES (?, ?, ?);
                """,
                drug_id, "cms_gnrc_name", cms_name,
                drug_id, "cms_gnrc_name", cms_name,
            )
        except:
            pass

        inserted = pivot_and_insert(cursor, drug_id, records, cms_name)
        conn.commit()

        print(f"    -> {inserted} year-brand records inserted")
        stats["found"] += 1
        stats["total_records"] += inserted
        found.append((inn, cms_name, len(records), inserted))

    # Summary
    print("\n" + "=" * 60)
    print("CMS Medicare Part D Ingest Complete")
    print("=" * 60)
    print(f"Drugs with data:    {stats['found']}/43")
    print(f"Drugs skipped:      {stats['skipped']}/43")
    print(f"Total records:      {stats['total_records']}")

    print(f"\nDrugs WITH data ({stats['found']}):")
    for inn, cms_name, brands, recs in found:
        print(f"  {inn:30s} CMS={cms_name:30s} brands={brands} records={recs}")

    print(f"\nDrugs WITHOUT data ({stats['skipped']}):")
    for inn, reason in skipped:
        print(f"  {inn:30s} Reason: {reason}")

    # Validation
    print("\n--- Validation ---")
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM spending_us")
    print(f"Distinct drugs in spending_us: {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT TOP 10 d.inn, s.brand_name, s.total_spending, s.total_claims, s.total_beneficiaries
        FROM spending_us s JOIN drugs d ON s.drug_id = d.drug_id
        WHERE s.year = 2023 AND (s.manufacturer = 'Overall' OR s.brand_name LIKE '%Overall%')
        ORDER BY s.total_spending DESC
    """)
    print(f"\nTop 10 by 2023 spending (Overall):")
    for row in cursor.fetchall():
        spending = f"${row[2]:>14,.2f}" if row[2] else "N/A"
        claims = f"{row[3]:>10,}" if row[3] else "N/A"
        print(f"  {row[0]:25s} {row[1]:20s} spending={spending}  claims={claims}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
