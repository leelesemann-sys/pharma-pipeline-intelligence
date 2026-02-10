"""
Phase 3 - Step 3: FDA FAERS Adverse Events Ingest
Loads top AEs, serious AEs, and temporal trends for all 43 drugs.
"""

import pyodbc
import requests
import time
from datetime import datetime

from db_config import CONN_STR

BASE_URL = "https://api.fda.gov/drug/event.json"
PAUSE = 1.0
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
            elif r.status_code == 404:
                return None  # No data for this drug
            else:
                print(f"    HTTP {r.status_code}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def load_top_aes(cursor, drug_id, inn):
    """3a) Load top 25 adverse events for a drug."""
    url = (
        f"{BASE_URL}?search=patient.drug.openfda.generic_name:"
        f'"{inn}"&count=patient.reaction.reactionmeddrapt.exact&limit=25'
    )
    data = api_get(url)
    if not data or "results" not in data:
        return 0

    count = 0
    for item in data["results"]:
        term = item.get("term", "")
        total = item.get("count", 0)
        if not term:
            continue
        try:
            cursor.execute(
                """
                MERGE adverse_events AS target
                USING (SELECT ? AS drug_id, ? AS event_term) AS src
                ON target.drug_id = src.drug_id AND target.event_term = src.event_term
                WHEN MATCHED THEN
                    UPDATE SET total_count = ?, last_updated = GETDATE()
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, event_term, total_count, last_updated)
                    VALUES (?, ?, ?, GETDATE());
                """,
                drug_id, term,
                total,
                drug_id, term, total,
            )
            count += 1
        except Exception as e:
            if "duplicate" not in str(e).lower():
                print(f"    AE insert error: {e}")
    return count


def load_serious_aes(cursor, drug_id, inn):
    """3b) Load serious AE counts and update existing records."""
    url = (
        f"{BASE_URL}?search=patient.drug.openfda.generic_name:"
        f'"{inn}"+AND+serious:1&count=patient.reaction.reactionmeddrapt.exact&limit=25'
    )
    data = api_get(url)
    if not data or "results" not in data:
        return 0

    updated = 0
    for item in data["results"]:
        term = item.get("term", "")
        serious = item.get("count", 0)
        if not term:
            continue
        try:
            cursor.execute(
                """
                UPDATE adverse_events
                SET serious_count = ?,
                    non_serious_count = CASE WHEN total_count IS NOT NULL THEN total_count - ? ELSE NULL END
                WHERE drug_id = ? AND event_term = ?
                """,
                serious, serious, drug_id, term,
            )
            if cursor.rowcount > 0:
                updated += 1
        except Exception as e:
            print(f"    Serious update error: {e}")
    return updated


def load_ae_trends(cursor, drug_id, inn, serious_only=False):
    """3c/3d) Load temporal AE trends, aggregate to quarters."""
    search = f'patient.drug.openfda.generic_name:"{inn}"'
    if serious_only:
        search += "+AND+serious:1"
    url = f"{BASE_URL}?search={search}&count=receivedate"
    data = api_get(url)
    if not data or "results" not in data:
        return 0

    # Aggregate daily data to quarters
    quarters = {}
    for item in data["results"]:
        time_str = item.get("time", "")
        count = item.get("count", 0)
        if len(time_str) < 6:
            continue
        try:
            year = int(time_str[:4])
            month = int(time_str[4:6])
        except ValueError:
            continue

        # Determine quarter
        if month <= 3:
            q_date = f"{year}-01-01"
        elif month <= 6:
            q_date = f"{year}-04-01"
        elif month <= 9:
            q_date = f"{year}-07-01"
        else:
            q_date = f"{year}-10-01"

        quarters[q_date] = quarters.get(q_date, 0) + count

    inserted = 0
    for q_date, total in quarters.items():
        try:
            if serious_only:
                # Update serious_reports on existing rows
                cursor.execute(
                    """
                    UPDATE adverse_event_trends
                    SET serious_reports = ?
                    WHERE drug_id = ? AND quarter_date = ?
                    """,
                    total, drug_id, q_date,
                )
                if cursor.rowcount > 0:
                    inserted += 1
            else:
                # Insert total reports
                cursor.execute(
                    """
                    MERGE adverse_event_trends AS target
                    USING (SELECT ? AS drug_id, ? AS quarter_date) AS src
                    ON target.drug_id = src.drug_id AND target.quarter_date = src.quarter_date
                    WHEN MATCHED THEN
                        UPDATE SET total_reports = ?
                    WHEN NOT MATCHED THEN
                        INSERT (drug_id, quarter_date, total_reports)
                        VALUES (?, ?, ?);
                    """,
                    drug_id, q_date,
                    total,
                    drug_id, q_date, total,
                )
                inserted += 1
        except Exception as e:
            if "duplicate" not in str(e).lower():
                print(f"    Trend insert error: {e}")
    return inserted


def main():
    print("=" * 60)
    print("Phase 3 Step 3: FDA FAERS Adverse Events Ingest")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    cursor.execute("SELECT drug_id, inn, moa_class FROM drugs ORDER BY inn")
    drugs = cursor.fetchall()
    print(f"\nLoaded {len(drugs)} drugs\n")

    stats = {"found": 0, "skipped": 0, "ae_records": 0, "trend_records": 0}
    skipped = []
    found = []

    for i, (drug_id, inn, moa_class) in enumerate(drugs):
        print(f"[{i+1:2d}/43] {inn} ({moa_class})")

        # 3a) Top 25 AEs
        time.sleep(PAUSE)
        ae_count = load_top_aes(cursor, drug_id, inn)

        if ae_count == 0:
            print(f"    -> No FAERS data, skipping")
            stats["skipped"] += 1
            skipped.append((inn, "No FAERS data"))
            conn.commit()
            continue

        print(f"    -> {ae_count} AE terms loaded")

        # Save external ID
        try:
            cursor.execute(
                """
                MERGE drug_external_ids AS target
                USING (SELECT ? AS drug_id, ? AS source, ? AS external_id) AS src
                ON target.drug_id = src.drug_id AND target.source = src.source AND target.external_id = src.external_id
                WHEN NOT MATCHED THEN
                    INSERT (drug_id, source, external_id) VALUES (?, ?, ?);
                """,
                drug_id, "faers_generic_name", inn,
                drug_id, "faers_generic_name", inn,
            )
        except:
            pass

        # 3b) Serious AEs
        time.sleep(PAUSE)
        serious_count = load_serious_aes(cursor, drug_id, inn)
        print(f"    -> {serious_count} serious counts updated")

        # 3c) Temporal trends (all)
        time.sleep(PAUSE)
        trend_count = load_ae_trends(cursor, drug_id, inn, serious_only=False)
        print(f"    -> {trend_count} quarterly trend records")

        # 3d) Temporal trends (serious only)
        time.sleep(PAUSE)
        serious_trend = load_ae_trends(cursor, drug_id, inn, serious_only=True)
        print(f"    -> {serious_trend} serious trend records updated")

        conn.commit()
        stats["found"] += 1
        stats["ae_records"] += ae_count
        stats["trend_records"] += trend_count
        found.append((inn, ae_count, trend_count))

    # Summary
    print("\n" + "=" * 60)
    print("FDA FAERS Ingest Complete")
    print("=" * 60)
    print(f"Drugs with data:    {stats['found']}/43")
    print(f"Drugs skipped:      {stats['skipped']}/43")
    print(f"AE records:         {stats['ae_records']}")
    print(f"Trend records:      {stats['trend_records']}")

    print(f"\nDrugs WITH data ({stats['found']}):")
    for inn, aes, trends in found:
        print(f"  {inn:30s} AEs={aes:3d}  trends={trends:3d}")

    print(f"\nDrugs WITHOUT data ({stats['skipped']}):")
    for inn, reason in skipped:
        print(f"  {inn:30s} Reason: {reason}")

    # Validation
    print("\n--- Validation ---")
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM adverse_events")
    print(f"Distinct drugs in adverse_events: {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT TOP 10 d.inn, ae.event_term, ae.total_count, ae.serious_count
        FROM adverse_events ae JOIN drugs d ON ae.drug_id = d.drug_id
        ORDER BY ae.total_count DESC
    """)
    print(f"\nTop 10 AEs by total count:")
    for row in cursor.fetchall():
        serious = row[3] if row[3] else 0
        print(f"  {row[0]:25s} {row[1]:30s} total={row[2]:>8,}  serious={serious:>8,}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
