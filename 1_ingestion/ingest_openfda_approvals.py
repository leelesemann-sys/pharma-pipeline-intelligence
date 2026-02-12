"""
Phase 1, Step 5: openFDA Approval Data
Pharma Pipeline Intelligence - Diabetes & Obesity

For each drug in our DB, queries openFDA for FDA approval data.
"""
import requests
import pyodbc
import json
import time
from datetime import datetime

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

OPENFDA_BASE = "https://api.fda.gov/drug/drugsfda.json"


def parse_fda_date(date_str):
    """Parse FDA date format (YYYYMMDD or YYYY-MM-DD)."""
    if not date_str:
        return None
    try:
        date_str = date_str.replace("-", "")
        return datetime.strptime(date_str[:8], "%Y%m%d").date()
    except:
        return None


def fetch_fda_approvals(drug_inn):
    """Fetch FDA approval data for a drug by INN."""
    # Clean the name for search
    search_name = drug_inn.replace(" ", "+")
    url = f'{OPENFDA_BASE}?search=openfda.generic_name:"{search_name}"&limit=10'

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            return []
        print(f"    HTTP Error for {drug_inn}: {e}")
        return []
    except Exception as e:
        print(f"    Error fetching {drug_inn}: {e}")
        return []


def main():
    print("=" * 70)
    print("openFDA APPROVAL DATA INGEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Connect
    print("\nConnecting to Azure SQL...")
    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Get all drugs
    cursor.execute("SELECT drug_id, inn FROM drugs WHERE inn IS NOT NULL")
    drugs = cursor.fetchall()
    print(f"Drugs to check: {len(drugs)}")

    approvals_loaded = 0
    drugs_with_approvals = 0
    drugs_without = []

    for drug_id, inn in drugs:
        print(f"\n  Checking: {inn}...", end=" ")
        results = fetch_fda_approvals(inn)
        time.sleep(0.3)  # Rate limit

        if not results:
            print("No FDA data")
            drugs_without.append(inn)
            continue

        drug_approvals = 0
        for result in results:
            app_number = result.get("application_number", "")
            sponsor = result.get("sponsor_name", "")
            openfda = result.get("openfda", {})
            brand_names = openfda.get("brand_name", [])

            # Look through submissions for approval dates
            submissions = result.get("submissions", [])
            for sub in submissions:
                sub_type = sub.get("submission_type", "")
                sub_status = sub.get("submission_status", "")

                if sub_status.upper() == "AP":  # Approved
                    sub_date = parse_fda_date(sub.get("submission_status_date", ""))

                    # Determine review type from submission_type
                    review_type = "standard"
                    sub_type_upper = sub_type.upper() if sub_type else ""
                    if "PRIORITY" in sub.get("review_priority", "").upper():
                        review_type = "priority"

                    try:
                        cursor.execute("""
                            INSERT INTO approvals (drug_id, country, agency, application_number, approval_date, review_type)
                            VALUES (?, 'US', 'FDA', ?, ?, ?)
                        """, (drug_id, app_number, sub_date, review_type))
                        drug_approvals += 1
                        approvals_loaded += 1
                    except pyodbc.IntegrityError:
                        pass

            # Update drug's first_approval_date
            if drug_approvals > 0:
                cursor.execute("""
                    UPDATE drugs SET first_approval_date = (
                        SELECT MIN(approval_date) FROM approvals WHERE drug_id = ? AND approval_date IS NOT NULL
                    ), highest_phase = 'approved', updated_at = GETUTCDATE()
                    WHERE drug_id = ?
                """, (drug_id, drug_id))

        if drug_approvals > 0:
            drugs_with_approvals += 1
            print(f"{drug_approvals} approval records")
        else:
            drugs_without.append(inn)
            print("No approval records found")

    # Report
    print(f"\n{'='*70}")
    print("openFDA APPROVALS REPORT")
    print(f"{'='*70}")
    print(f"\nTotal approval records loaded: {approvals_loaded}")
    print(f"Drugs with FDA approvals: {drugs_with_approvals}")
    print(f"Drugs without FDA data: {len(drugs_without)}")
    if drugs_without:
        print("  Missing FDA data for:")
        for d in drugs_without:
            print(f"    - {d}")

    # Final DB stats
    print(f"\n--- Database Stats ---")
    cursor.execute("SELECT COUNT(*) FROM approvals")
    print(f"Total approvals in DB: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM approvals WHERE agency = 'FDA'")
    print(f"FDA approvals: {cursor.fetchone()[0]}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
