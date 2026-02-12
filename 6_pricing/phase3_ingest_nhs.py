"""
Phase 3 - Step 2: NHS OpenPrescribing Ingest
Loads BNF codes and prescription data for all 43 drugs.
"""

import pyodbc
import requests
import time
import sys

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

BASE_URL = "https://openprescribing.net/api/1.0"
PAUSE = 1.0  # seconds between requests
MAX_RETRIES = 3


def api_get(url, retries=MAX_RETRIES):
    """GET with retry + backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def find_bnf_code(inn):
    """Find BNF chemical-level code for a drug INN."""
    url = f"{BASE_URL}/bnf_code/?q={inn}&format=json"
    data = api_get(url)
    if not data:
        return None

    # Look for chemical-level codes (typically 9 chars)
    # Prefer diabetes section (0601) but accept others
    candidates = []
    for item in data:
        code = item.get("id", "")
        name = item.get("name", "").lower()
        if len(code) >= 7 and inn.lower() in name:
            candidates.append(code)
        elif len(code) >= 7:
            candidates.append(code)

    if not candidates:
        # Sometimes the API returns broader matches — try exact chemical codes
        for item in data:
            code = item.get("id", "")
            if 7 <= len(code) <= 10:
                candidates.append(code)

    return candidates[0] if candidates else None


def load_prescriptions(cursor, drug_id, inn, bnf_code):
    """Load all prescription data for a BNF code."""
    url = f"{BASE_URL}/spending/?code={bnf_code}&format=json"
    data = api_get(url)
    if not data:
        return 0

    rows_inserted = 0
    for record in data:
        date = record.get("date")
        items = record.get("items")
        quantity = record.get("quantity")
        actual_cost = record.get("actual_cost")

        if date is None:
            continue

        try:
            cursor.execute(
                """
                MERGE prescriptions_uk AS target
                USING (SELECT ? AS drug_id, ? AS date) AS source
                ON target.drug_id = source.drug_id AND target.date = source.date
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, bnf_code, date, items, quantity, actual_cost)
                    VALUES (?, ?, ?, ?, ?, ?);
                """,
                drug_id, date,
                drug_id, bnf_code, date, items, quantity, actual_cost,
            )
            rows_inserted += 1
        except Exception as e:
            if "duplicate" not in str(e).lower() and "violation" not in str(e).lower():
                print(f"    Insert error: {e}")

    return rows_inserted


def main():
    print("=" * 60)
    print("Phase 3 Step 2: NHS OpenPrescribing Ingest")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    # Load all drugs
    cursor.execute("SELECT drug_id, inn, moa_class FROM drugs ORDER BY inn")
    drugs = cursor.fetchall()
    print(f"\nLoaded {len(drugs)} drugs from database\n")

    stats = {"found": 0, "skipped": 0, "total_records": 0}
    skipped_drugs = []
    found_drugs = []

    for i, (drug_id, inn, moa_class) in enumerate(drugs):
        print(f"[{i+1:2d}/43] {inn} ({moa_class})")

        # Step 1: Find BNF code
        time.sleep(PAUSE)
        bnf_code = find_bnf_code(inn)

        if not bnf_code:
            print(f"    -> No BNF code found, skipping")
            stats["skipped"] += 1
            skipped_drugs.append((inn, "No BNF code"))
            continue

        print(f"    BNF code: {bnf_code}")

        # Save BNF code to drug_external_ids
        try:
            cursor.execute(
                """
                MERGE drug_external_ids AS target
                USING (SELECT ? AS drug_id, ? AS source, ? AS external_id) AS src
                ON target.drug_id = src.drug_id AND target.source = src.source AND target.external_id = src.external_id
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, source, external_id) VALUES (?, ?, ?);
                """,
                drug_id, "nhs_bnf", bnf_code,
                drug_id, "nhs_bnf", bnf_code,
            )
            conn.commit()
        except Exception as e:
            if "duplicate" not in str(e).lower() and "violation" not in str(e).lower():
                print(f"    External ID save error: {e}")

        # Step 2: Load prescription data
        time.sleep(PAUSE)
        records = load_prescriptions(cursor, drug_id, inn, bnf_code)
        conn.commit()

        if records > 0:
            print(f"    -> {records} monthly records loaded")
            stats["found"] += 1
            stats["total_records"] += records
            found_drugs.append((inn, bnf_code, records))
        else:
            print(f"    -> No prescription data")
            stats["skipped"] += 1
            skipped_drugs.append((inn, f"BNF {bnf_code} but no data"))

    # Summary
    print("\n" + "=" * 60)
    print("NHS OpenPrescribing Ingest Complete")
    print("=" * 60)
    print(f"Drugs with data:    {stats['found']}/43")
    print(f"Drugs skipped:      {stats['skipped']}/43")
    print(f"Total records:      {stats['total_records']}")

    print(f"\nDrugs WITH data ({stats['found']}):")
    for inn, bnf, count in found_drugs:
        print(f"  {inn:30s} BNF={bnf}  records={count}")

    print(f"\nDrugs WITHOUT data ({stats['skipped']}):")
    for inn, reason in skipped_drugs:
        print(f"  {inn:30s} Reason: {reason}")

    # Validation queries
    print("\n--- Validation ---")
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM prescriptions_uk")
    print(f"Distinct drugs in prescriptions_uk: {cursor.fetchone()[0]}")

    cursor.execute("SELECT MIN(date), MAX(date) FROM prescriptions_uk")
    row = cursor.fetchone()
    print(f"Date range: {row[0]} to {row[1]}")

    cursor.execute("""
        SELECT TOP 5 d.inn, p.items, p.actual_cost, p.date
        FROM prescriptions_uk p JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
        ORDER BY p.items DESC
    """)
    print(f"\nTop 5 by items (latest month):")
    for row in cursor.fetchall():
        print(f"  {row[0]:25s} items={row[1]:>10,}  cost=GBP {row[2]:>12,.2f}  date={row[3]}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
