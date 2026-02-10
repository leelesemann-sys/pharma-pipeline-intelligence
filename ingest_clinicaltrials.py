"""
Phase 1, Step 3: ClinicalTrials.gov API v2 Full Ingest
Pharma Pipeline Intelligence - Diabetes & Obesity

Loads ALL trials for Diabetes/Obesity conditions into Azure SQL.
Deduplicates by NCT ID across condition queries.
"""
import requests
import pyodbc
import json
import time
import sys
from datetime import datetime
from collections import Counter

from db_config import CONN_STR

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
SLEEP_BETWEEN_REQUESTS = 1.5  # seconds

CONDITIONS = [
    "diabetes mellitus, type 2",
    "diabetes mellitus, type 1",
    "obesity",
    "overweight",
    "nonalcoholic steatohepatitis",
    "non-alcoholic fatty liver disease",
    "metabolic syndrome",
]

# Phase mapping from API values to our schema
PHASE_MAP = {
    "EARLY_PHASE1": "early_phase1",
    "PHASE1": "phase1",
    "PHASE2": "phase2",
    "PHASE3": "phase3",
    "PHASE4": "phase4",
    "NA": "na",
}

STATUS_MAP = {
    "RECRUITING": "recruiting",
    "NOT_YET_RECRUITING": "not_yet_recruiting",
    "ENROLLING_BY_INVITATION": "enrolling_by_invitation",
    "ACTIVE_NOT_RECRUITING": "active_not_recruiting",
    "COMPLETED": "completed",
    "TERMINATED": "terminated",
    "WITHDRAWN": "withdrawn",
    "SUSPENDED": "suspended",
    "UNKNOWN_STATUS": "unknown",
    "UNKNOWN": "unknown",
    "APPROVED_FOR_MARKETING": "completed",
    "AVAILABLE": "completed",
    "NO_LONGER_AVAILABLE": "completed",
    "TEMPORARILY_NOT_AVAILABLE": "suspended",
    "WITHHELD": "unknown",
}

STUDY_TYPE_MAP = {
    "INTERVENTIONAL": "interventional",
    "OBSERVATIONAL": "observational",
    "EXPANDED_ACCESS": "expanded_access",
}


def parse_date(date_struct):
    """Parse date from ClinicalTrials.gov date struct."""
    if not date_struct:
        return None
    date_str = date_struct.get("date", "")
    if not date_str:
        return None
    try:
        # Try YYYY-MM-DD
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            # Try YYYY-MM
            return datetime.strptime(date_str, "%Y-%m").date()
        except ValueError:
            try:
                # Try YYYY
                return datetime.strptime(date_str, "%Y").date()
            except ValueError:
                return None


def map_phase(phases_list):
    """Map API phases to our schema. Take highest phase."""
    if not phases_list:
        return "na"
    # Phase priority (higher = later)
    priority = {"na": 0, "early_phase1": 1, "phase1": 2, "phase2": 3, "phase3": 4, "phase4": 5}
    mapped = []
    for p in phases_list:
        mapped_phase = PHASE_MAP.get(p.upper().replace(" ", ""), "na")
        mapped.append(mapped_phase)

    # Handle combined phases like ["PHASE1", "PHASE2"] -> phase1_phase2
    if len(mapped) == 2:
        sorted_phases = sorted(mapped, key=lambda x: priority.get(x, 0))
        combo = f"{sorted_phases[0]}_{sorted_phases[1]}"
        valid_combos = {"phase1_phase2", "phase2_phase3"}
        if combo in valid_combos:
            return combo

    # Return highest single phase
    return max(mapped, key=lambda x: priority.get(x, 0))


def extract_trial(study):
    """Extract relevant fields from a ClinicalTrials.gov study."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    enrollment_info = design.get("enrollmentInfo", {})
    conditions_mod = proto.get("conditionsModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})

    # Get interventions
    interventions = arms_mod.get("interventions", [])
    intervention_list = []
    for interv in interventions:
        intervention_list.append({
            "type": interv.get("type", ""),
            "name": interv.get("name", ""),
            "description": interv.get("description", "")[:500] if interv.get("description") else "",
        })

    # Get conditions
    conditions = conditions_mod.get("conditions", [])

    # Get lead sponsor
    lead_sponsor = sponsor_mod.get("leadSponsor", {})

    # Get phases
    phases_raw = design.get("phases", [])

    # Status
    raw_status = status_mod.get("overallStatus", "UNKNOWN")
    mapped_status = STATUS_MAP.get(raw_status.upper(), "unknown")

    # Study type
    raw_study_type = design.get("studyType", "INTERVENTIONAL") if design else "INTERVENTIONAL"
    # Sometimes studyType is at proto level
    if not raw_study_type or raw_study_type not in STUDY_TYPE_MAP:
        raw_study_type = proto.get("designModule", {}).get("studyType", "INTERVENTIONAL")
    mapped_study_type = STUDY_TYPE_MAP.get(raw_study_type, "interventional")

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", "")[:4000],  # Truncate for NVARCHAR(MAX) safety
        "phase": map_phase(phases_raw),
        "overall_status": mapped_status,
        "start_date": parse_date(status_mod.get("startDateStruct")),
        "primary_completion_date": parse_date(status_mod.get("primaryCompletionDateStruct")),
        "completion_date": parse_date(status_mod.get("completionDateStruct")),
        "last_update_date": parse_date(status_mod.get("lastUpdatePostDateStruct")),
        "enrollment": enrollment_info.get("count"),
        "enrollment_type": enrollment_info.get("type", ""),
        "study_type": mapped_study_type,
        "has_results": 1 if study.get("hasResults", False) else 0,
        "why_stopped": status_mod.get("whyStopped", ""),
        "lead_sponsor_name": lead_sponsor.get("name", ""),
        "raw_conditions": json.dumps(conditions),
        "raw_interventions": json.dumps(intervention_list),
    }


def fetch_all_trials_for_condition(condition):
    """Fetch all trials for a given condition using pagination."""
    all_studies = []
    page_token = None
    page_num = 0

    # First request to get count
    params = {
        "query.cond": condition,
        "pageSize": PAGE_SIZE,
        "countTotal": "true",
    }

    while True:
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    ERROR fetching page {page_num}: {e}")
            # Retry once after 5s
            time.sleep(5)
            try:
                resp = requests.get(API_BASE, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                print(f"    RETRY FAILED: {e2}. Skipping remaining pages.")
                break

        studies = data.get("studies", [])
        total = data.get("totalCount", "?")
        all_studies.extend(studies)
        page_num += 1

        loaded = len(all_studies)
        print(f"    Loaded {loaded}/{total} trials for '{condition}' (page {page_num})")

        # Check for next page
        page_token = data.get("nextPageToken")
        if not page_token or not studies:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_studies


def insert_trials_batch(cursor, trials):
    """Insert trials in batch using executemany."""
    sql = """
        INSERT INTO trials (
            nct_id, title, phase, overall_status, start_date,
            primary_completion_date, completion_date, last_update_date,
            enrollment, enrollment_type, study_type, has_results,
            why_stopped, lead_sponsor_name, raw_conditions, raw_interventions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch = []
    for t in trials:
        batch.append((
            t["nct_id"], t["title"], t["phase"], t["overall_status"],
            t["start_date"], t["primary_completion_date"], t["completion_date"],
            t["last_update_date"], t["enrollment"], t["enrollment_type"],
            t["study_type"], t["has_results"], t["why_stopped"],
            t["lead_sponsor_name"], t["raw_conditions"], t["raw_interventions"],
        ))

    # Insert in chunks of 500
    chunk_size = 500
    inserted = 0
    errors = 0
    for i in range(0, len(batch), chunk_size):
        chunk = batch[i:i + chunk_size]
        for row in chunk:
            try:
                cursor.execute(sql, row)
                inserted += 1
            except pyodbc.IntegrityError:
                # Duplicate NCT ID - skip
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Insert error for {row[0]}: {e}")

    return inserted, errors


def main():
    print("=" * 70)
    print("CLINICALTRIALS.GOV FULL INGEST - Diabetes & Obesity")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Connect to DB
    print("\nConnecting to Azure SQL...")
    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Check existing count
    cursor.execute("SELECT COUNT(*) FROM trials")
    existing = cursor.fetchone()[0]
    print(f"Existing trials in DB: {existing}")

    # Fetch all trials
    all_trials = {}  # NCT ID -> trial data (dedup)
    condition_counts = {}

    for condition in CONDITIONS:
        print(f"\n--- Fetching: {condition} ---")
        studies = fetch_all_trials_for_condition(condition)
        condition_counts[condition] = len(studies)

        for study in studies:
            trial = extract_trial(study)
            nct_id = trial["nct_id"]
            if nct_id and nct_id not in all_trials:
                all_trials[nct_id] = trial

        print(f"    Raw: {len(studies)} | Unique so far: {len(all_trials)}")

    # Summary before insert
    print(f"\n{'='*70}")
    print(f"FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"\nTrials per condition:")
    for cond, count in condition_counts.items():
        print(f"  {cond}: {count:,}")
    print(f"\nTotal unique trials (deduplicated): {len(all_trials):,}")

    # Insert into DB
    print(f"\nInserting {len(all_trials):,} trials into Azure SQL...")
    trials_list = list(all_trials.values())
    inserted, errors = insert_trials_batch(cursor, trials_list)
    print(f"Inserted: {inserted:,} | Skipped (duplicates/errors): {errors:,}")

    # Generate summary report
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    # Total count
    cursor.execute("SELECT COUNT(*) FROM trials")
    total = cursor.fetchone()[0]
    print(f"\nTotal trials in DB: {total:,}")

    # By status
    print("\nBreakdown by Status:")
    cursor.execute("SELECT overall_status, COUNT(*) as cnt FROM trials GROUP BY overall_status ORDER BY cnt DESC")
    for row in cursor.fetchall():
        print(f"  {row[0]:30s} {row[1]:>6,}")

    # By phase
    print("\nBreakdown by Phase:")
    cursor.execute("SELECT phase, COUNT(*) as cnt FROM trials GROUP BY phase ORDER BY cnt DESC")
    for row in cursor.fetchall():
        print(f"  {row[0]:20s} {row[1]:>6,}")

    # By study type
    print("\nBreakdown by Study Type:")
    cursor.execute("SELECT study_type, COUNT(*) as cnt FROM trials GROUP BY study_type ORDER BY cnt DESC")
    for row in cursor.fetchall():
        print(f"  {row[0]:20s} {row[1]:>6,}")

    # Top 20 sponsors
    print("\nTop 20 Sponsors:")
    cursor.execute("""
        SELECT TOP 20 lead_sponsor_name, COUNT(*) as cnt
        FROM trials
        WHERE lead_sponsor_name IS NOT NULL AND lead_sponsor_name != ''
        GROUP BY lead_sponsor_name
        ORDER BY cnt DESC
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"  {i:2d}. {row[0][:50]:50s} {row[1]:>5,}")

    # Oldest and newest
    print("\nDate Range:")
    cursor.execute("SELECT MIN(start_date), MAX(start_date) FROM trials WHERE start_date IS NOT NULL")
    row = cursor.fetchone()
    print(f"  Oldest trial start: {row[0]}")
    print(f"  Newest trial start: {row[1]}")

    cursor.execute("SELECT MIN(last_update_date), MAX(last_update_date) FROM trials WHERE last_update_date IS NOT NULL")
    row = cursor.fetchone()
    print(f"  Oldest update: {row[0]}")
    print(f"  Latest update: {row[1]}")

    # Stale trials
    cursor.execute("SELECT COUNT(*) FROM trials WHERE is_stale = 1")
    stale = cursor.fetchone()[0]
    print(f"\nStale trials (no update >12mo + past completion): {stale:,}")

    # Has results
    cursor.execute("SELECT COUNT(*) FROM trials WHERE has_results = 1")
    with_results = cursor.fetchone()[0]
    print(f"Trials with results: {with_results:,}")

    cursor.close()
    conn.close()

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
