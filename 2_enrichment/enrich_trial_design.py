"""
Phase 1 Nachtrag 2: Trial Design Fields Enrichment
Loads missing trial design columns from ClinicalTrials.gov v2 API
and updates the trials table in Azure SQL.

Expected runtime: ~7-10 minutes for 32,811 trials (328 API batches)
"""

import pyodbc
import requests
import json
import time
import sys
import os
from datetime import datetime

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
API_FIELDS = (
    "protocolSection.identificationModule.nctId,"
    "protocolSection.designModule,"
    "protocolSection.eligibilityModule,"
    "protocolSection.armsInterventionsModule,"
    "protocolSection.outcomesModule,"
    "protocolSection.oversightModule"
)
BATCH_SIZE = 100  # NCT IDs per API call (API limit)
API_DELAY = 1.2   # seconds between requests (50 req/min limit)
DB_BATCH_SIZE = 500  # rows per executemany


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {"INFO": "INFO ", "WARN": "WARN ", "ERROR": "ERROR"}
    print(f"{ts} | {prefix.get(level, 'INFO ')} | {msg}", flush=True)


def connect_db():
    """Connect to Azure SQL with retry for serverless auto-pause."""
    for attempt in range(5):
        try:
            conn = pyodbc.connect(CONN_STR)
            conn.autocommit = True
            conn.cursor().execute("SELECT 1").fetchone()
            return conn
        except Exception as e:
            if attempt < 4:
                delay = 5 * (2 ** attempt)
                log(f"DB connection attempt {attempt+1} failed, retrying in {delay}s: {e}", "WARN")
                time.sleep(delay)
            else:
                raise


def step1_check_columns(cursor):
    """Check existing columns in trials table."""
    log("=" * 60)
    log("STEP 1: Checking existing trials columns")
    log("=" * 60)

    cursor.execute("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'trials' ORDER BY ORDINAL_POSITION
    """)
    existing = {row[0] for row in cursor.fetchall()}
    log(f"  Existing columns: {len(existing)}")
    for col in sorted(existing):
        log(f"    - {col}")
    return existing


def step2_add_columns(cursor, existing_columns):
    """Add missing columns via ALTER TABLE."""
    log("=" * 60)
    log("STEP 2: Adding new columns to trials table")
    log("=" * 60)

    new_columns = [
        ("allocation",              "NVARCHAR(50)"),
        ("intervention_model",      "NVARCHAR(50)"),
        ("primary_purpose",         "NVARCHAR(50)"),
        ("masking",                 "NVARCHAR(50)"),
        ("who_masked",              "NVARCHAR(200)"),
        ("number_of_arms",          "INT"),
        ("has_placebo",             "BIT"),
        ("intervention_types",      "NVARCHAR(200)"),
        ("has_dmc",                 "BIT"),
        ("minimum_age",             "NVARCHAR(20)"),
        ("maximum_age",             "NVARCHAR(20)"),
        ("sex",                     "NVARCHAR(10)"),
        ("healthy_volunteers",      "BIT"),
        ("eligibility_criteria",    "NVARCHAR(MAX)"),
        ("n_primary_outcomes",      "INT"),
        ("n_secondary_outcomes",    "INT"),
        ("primary_outcome_measures", "NVARCHAR(MAX)"),
    ]

    added = 0
    skipped = 0
    for col_name, col_type in new_columns:
        if col_name in existing_columns:
            log(f"  EXISTS: {col_name} (skipping)")
            skipped += 1
        else:
            try:
                cursor.execute(f"ALTER TABLE trials ADD [{col_name}] {col_type} NULL")
                log(f"  ADDED:  {col_name} ({col_type})")
                added += 1
            except Exception as e:
                if "already" in str(e).lower():
                    log(f"  EXISTS: {col_name} (skipping)")
                    skipped += 1
                else:
                    log(f"  ERROR:  {col_name}: {e}", "ERROR")

    log(f"  Summary: {added} added, {skipped} already existed")
    return added


def extract_design_fields(study):
    """Extract all needed fields from an API study object."""
    protocol = study.get("protocolSection", {})
    design = protocol.get("designModule", {})
    design_info = design.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})
    eligibility = protocol.get("eligibilityModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    oversight = protocol.get("oversightModule", {})
    enrollment_info = design.get("enrollmentInfo", {})

    nct_id = protocol.get("identificationModule", {}).get("nctId")
    if not nct_id:
        return None

    # Arms analysis
    arm_groups = arms_module.get("armGroups", [])
    interventions = arms_module.get("interventions", [])
    n_arms = len(arm_groups) if arm_groups else None
    has_placebo = any(
        arm.get("type") == "PLACEBO_COMPARATOR" for arm in arm_groups
    ) if arm_groups else None
    intervention_types = sorted(set(
        intv.get("type", "") for intv in interventions if intv.get("type")
    )) if interventions else []

    # Outcomes count
    primary_outcomes = outcomes.get("primaryOutcomes", [])
    secondary_outcomes = outcomes.get("secondaryOutcomes", [])
    primary_measures = [po.get("measure", "") for po in primary_outcomes if po.get("measure")]

    return {
        "nct_id": nct_id,
        "allocation": design_info.get("allocation"),
        "intervention_model": design_info.get("interventionModel"),
        "primary_purpose": design_info.get("primaryPurpose"),
        "masking": masking_info.get("masking"),
        "who_masked": json.dumps(masking_info.get("whoMasked", [])) if masking_info.get("whoMasked") else None,
        "number_of_arms": n_arms,
        "has_placebo": 1 if has_placebo else (0 if has_placebo is not None else None),
        "intervention_types": json.dumps(intervention_types) if intervention_types else None,
        "has_dmc": 1 if oversight.get("oversightHasDmc") else (0 if "oversightHasDmc" in oversight else None),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "sex": eligibility.get("sex"),
        "healthy_volunteers": 1 if eligibility.get("healthyVolunteers") else (0 if "healthyVolunteers" in eligibility else None),
        "eligibility_criteria": eligibility.get("eligibilityCriteria"),
        "n_primary_outcomes": len(primary_outcomes) if primary_outcomes else 0,
        "n_secondary_outcomes": len(secondary_outcomes) if secondary_outcomes else 0,
        "primary_outcome_measures": json.dumps(primary_measures) if primary_measures else None,
        "enrollment_type": enrollment_info.get("type"),
    }


def fetch_batch(nct_ids, session):
    """Fetch a batch of studies from ClinicalTrials.gov API v2."""
    ids_str = ",".join(nct_ids)
    params = {
        "filter.ids": ids_str,
        "fields": API_FIELDS,
        "pageSize": 1000,
    }

    for retry in range(3):
        try:
            resp = session.get(API_BASE, params=params, timeout=60)
            if resp.status_code == 429:
                log("  Rate limited (429), waiting 60s...", "WARN")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("studies", [])
            return studies
        except requests.exceptions.RequestException as e:
            if retry < 2:
                log(f"  API error (attempt {retry+1}): {e}, retrying in 10s...", "WARN")
                time.sleep(10)
            else:
                log(f"  API failed after 3 attempts: {e}", "ERROR")
                return []
    return []


def step3_fetch_and_update(conn):
    """Fetch trial design data from API and update DB."""
    log("=" * 60)
    log("STEP 3: Fetching trial design data from ClinicalTrials.gov API v2")
    log("=" * 60)

    cursor = conn.cursor()

    # Load all NCT IDs
    cursor.execute("SELECT nct_id FROM trials WHERE nct_id IS NOT NULL ORDER BY nct_id")
    all_nct_ids = [row[0] for row in cursor.fetchall()]
    total = len(all_nct_ids)
    log(f"  Total NCT IDs to process: {total}")

    # Prepare UPDATE statement
    update_sql = """
        UPDATE trials SET
            allocation = ?,
            intervention_model = ?,
            primary_purpose = ?,
            masking = ?,
            who_masked = ?,
            number_of_arms = ?,
            has_placebo = ?,
            intervention_types = ?,
            has_dmc = ?,
            minimum_age = ?,
            maximum_age = ?,
            sex = ?,
            healthy_volunteers = ?,
            eligibility_criteria = ?,
            n_primary_outcomes = ?,
            n_secondary_outcomes = ?,
            primary_outcome_measures = ?,
            enrollment_type = ?
        WHERE nct_id = ?
    """

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    updated_total = 0
    not_found = []
    api_errors = 0
    start_time = time.time()

    for batch_idx in range(0, total, BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE + 1
        batch_ids = all_nct_ids[batch_idx:batch_idx + BATCH_SIZE]

        # Fetch from API
        studies = fetch_batch(batch_ids, session)

        if not studies:
            api_errors += 1
            log(f"  Batch {batch_num}/{total_batches}: API returned 0 studies for {len(batch_ids)} IDs", "WARN")
            time.sleep(API_DELAY)
            continue

        # Parse results
        results = {}
        for study in studies:
            fields = extract_design_fields(study)
            if fields:
                results[fields["nct_id"]] = fields

        # Track not found
        for nid in batch_ids:
            if nid not in results:
                not_found.append(nid)

        # Batch update DB
        update_rows = []
        for nid in batch_ids:
            if nid in results:
                f = results[nid]
                update_rows.append((
                    f["allocation"], f["intervention_model"], f["primary_purpose"],
                    f["masking"], f["who_masked"], f["number_of_arms"],
                    f["has_placebo"], f["intervention_types"], f["has_dmc"],
                    f["minimum_age"], f["maximum_age"], f["sex"],
                    f["healthy_volunteers"], f["eligibility_criteria"],
                    f["n_primary_outcomes"], f["n_secondary_outcomes"],
                    f["primary_outcome_measures"], f["enrollment_type"],
                    nid,  # WHERE nct_id = ?
                ))

        if update_rows:
            # Execute in sub-batches for DB performance
            for i in range(0, len(update_rows), DB_BATCH_SIZE):
                sub_batch = update_rows[i:i + DB_BATCH_SIZE]
                cursor.executemany(update_sql, sub_batch)
            updated_total += len(update_rows)

        elapsed = time.time() - start_time
        rate = updated_total / elapsed if elapsed > 0 else 0
        eta_s = (total - batch_idx - BATCH_SIZE) / rate if rate > 0 else 0
        eta_m = eta_s / 60

        if batch_num % 10 == 0 or batch_num == total_batches:
            log(f"  Batch {batch_num}/{total_batches}: "
                f"{updated_total}/{total} trials updated "
                f"({len(results)}/{len(batch_ids)} found in batch, "
                f"ETA: {eta_m:.1f}min)")

        time.sleep(API_DELAY)

    elapsed_total = time.time() - start_time

    log(f"\n  === FETCH SUMMARY ===")
    log(f"  Total updated:    {updated_total}/{total}")
    log(f"  Not found in API: {len(not_found)}")
    log(f"  API errors:       {api_errors}")
    log(f"  Duration:         {elapsed_total/60:.1f} minutes")

    if not_found and len(not_found) <= 50:
        log(f"  Not found IDs: {', '.join(not_found[:50])}")
    elif not_found:
        log(f"  Not found IDs (first 50): {', '.join(not_found[:50])}...")

    return updated_total, not_found


def step4_validate(cursor):
    """Validate completeness of enriched fields."""
    log("=" * 60)
    log("STEP 4: Validation Report")
    log("=" * 60)

    cursor.execute("""
        SELECT
            COUNT(*) as total_trials,
            COUNT(allocation) as has_allocation,
            COUNT(intervention_model) as has_intervention_model,
            COUNT(primary_purpose) as has_primary_purpose,
            COUNT(masking) as has_masking,
            COUNT(who_masked) as has_who_masked,
            COUNT(number_of_arms) as has_number_of_arms,
            COUNT(has_placebo) as has_has_placebo,
            COUNT(intervention_types) as has_intervention_types,
            COUNT(has_dmc) as has_has_dmc,
            COUNT(minimum_age) as has_minimum_age,
            COUNT(maximum_age) as has_maximum_age,
            COUNT(sex) as has_sex,
            COUNT(healthy_volunteers) as has_healthy_volunteers,
            COUNT(eligibility_criteria) as has_eligibility_criteria,
            COUNT(n_primary_outcomes) as has_n_primary_outcomes,
            COUNT(n_secondary_outcomes) as has_n_secondary_outcomes,
            COUNT(primary_outcome_measures) as has_primary_outcome_measures
        FROM trials
    """)
    row = cursor.fetchone()
    total = row[0]

    fields = [
        "allocation", "intervention_model", "primary_purpose", "masking",
        "who_masked", "number_of_arms", "has_placebo", "intervention_types",
        "has_dmc", "minimum_age", "maximum_age", "sex", "healthy_volunteers",
        "eligibility_criteria", "n_primary_outcomes", "n_secondary_outcomes",
        "primary_outcome_measures",
    ]

    log(f"\n  FIELD COMPLETENESS ({total} total trials):")
    log(f"  {'Field':<30s} {'Non-NULL':>10s} {'%':>8s}")
    log(f"  {'-'*50}")
    for i, field in enumerate(fields):
        count = row[i + 1]
        pct = 100.0 * count / total if total > 0 else 0
        marker = " !!!" if pct < 50 else ""
        log(f"  {field:<30s} {count:>10,d} {pct:>7.1f}%{marker}")

    # Value distributions for key fields
    log(f"\n  VALUE DISTRIBUTIONS:")

    for field in ["allocation", "intervention_model", "primary_purpose", "masking", "sex"]:
        cursor.execute(f"SELECT [{field}], COUNT(*) as n FROM trials GROUP BY [{field}] ORDER BY n DESC")
        log(f"\n  {field}:")
        for r in cursor.fetchall():
            val = r[0] if r[0] else "NULL"
            log(f"    {val:<35s} {r[1]:>8,d}")

    # Arms distribution
    cursor.execute("""
        SELECT number_of_arms, COUNT(*) as n
        FROM trials
        WHERE number_of_arms IS NOT NULL
        GROUP BY number_of_arms
        ORDER BY number_of_arms
    """)
    log(f"\n  number_of_arms:")
    for r in cursor.fetchall():
        log(f"    {r[0]:<35d} {r[1]:>8,d}")


def main():
    log("=" * 60)
    log("PHASE 1 NACHTRAG 2: Trial Design Fields Enrichment")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    start_time = time.time()

    # Connect
    log("\nConnecting to Azure SQL...")
    conn = connect_db()
    cursor = conn.cursor()
    log("Connected successfully!")

    # Step 1: Check columns
    existing = step1_check_columns(cursor)

    # Step 2: Add missing columns
    step2_add_columns(cursor, existing)

    # Step 3: Fetch from API and update DB
    updated, not_found = step3_fetch_and_update(conn)

    # Step 4: Validate
    step4_validate(cursor)

    # Final summary
    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log(f"ENRICHMENT COMPLETE")
    log(f"  Trials updated:  {updated}")
    log(f"  Not found:       {len(not_found)}")
    log(f"  Total duration:  {elapsed/60:.1f} minutes")
    log("=" * 60)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
