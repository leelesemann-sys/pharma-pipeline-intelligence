"""
Phase 4a: ML Feature Engineering – Clinical Trial Success Prediction
Computes ~105 features per trial and writes them to ml_features_trial + ml_features_drug_indication.
Tracks everything via MLflow.

Usage: python compute_features.py
"""

import os
import sys
import time
import logging
import json
import re
from datetime import datetime

import pyodbc
import pandas as pd
import numpy as np
from scipy import stats

import mlflow

from config import (
    DB_CONN_STR, FEATURE_VERSION, MLFLOW_TRACKING_DIR, MLFLOW_EXPERIMENT_FEATURES,
    LOG_DIR, ARTIFACT_DIR, BATCH_INSERT_SIZE, CORRELATION_THRESHOLD,
    PHASE_NUMERIC, PHASE_ORDER, NEXT_PHASE, MOA_GENERATION,
)

# ═══════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# MLFLOW SETUP
# ═══════════════════════════════════════════════════════════
mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLFLOW_TRACKING_DIR).replace(os.sep, '/')}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURES)


# ═══════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════
def connect_db():
    for attempt in range(5):
        try:
            conn = pyodbc.connect(DB_CONN_STR)
            conn.autocommit = True
            conn.cursor().execute("SELECT 1").fetchone()
            return conn
        except Exception as e:
            if attempt < 4:
                delay = 5 * (2 ** attempt)
                logger.warning(f"DB connect attempt {attempt+1} failed, retry in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise


def read_sql(query, conn, params=None):
    return pd.read_sql(query, conn, params=params)


def batch_insert(conn, table, df, columns):
    """Insert DataFrame into table in batches."""
    if df.empty:
        return 0
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(f"[{c}]" for c in columns)
    sql = f"INSERT INTO [{table}] ({col_str}) VALUES ({placeholders})"

    rows = df[columns].values.tolist()
    # Replace numpy types with Python types
    clean_rows = []
    for row in rows:
        clean = []
        for val in row:
            if isinstance(val, (np.integer,)):
                clean.append(int(val))
            elif isinstance(val, (np.floating,)):
                clean.append(None if np.isnan(val) else float(val))
            elif isinstance(val, (np.bool_,)):
                clean.append(bool(val))
            elif pd.isna(val):
                clean.append(None)
            else:
                clean.append(val)
        clean_rows.append(clean)

    total = 0
    for i in range(0, len(clean_rows), BATCH_INSERT_SIZE):
        batch = clean_rows[i:i + BATCH_INSERT_SIZE]
        cursor.executemany(sql, batch)
        total += len(batch)
    return total


def parse_age_years(age_str):
    """Parse age string like '18 Years' to float."""
    if not age_str or pd.isna(age_str):
        return None
    m = re.match(r"(\d+)\s*(Year|Month|Week|Day|Hour|Minute)", str(age_str), re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("year"):
        return val
    elif unit.startswith("month"):
        return val / 12.0
    elif unit.startswith("week"):
        return val / 52.0
    elif unit.startswith("day"):
        return val / 365.25
    return val


# ═══════════════════════════════════════════════════════════
# CREATE TABLES
# ═══════════════════════════════════════════════════════════
def create_tables(conn):
    logger.info("=" * 60)
    logger.info("STEP 1/13: Creating ML feature tables")
    logger.info("=" * 60)
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN ('ml_features_trial','ml_features_drug_indication')")
    existing = {r[0] for r in cursor.fetchall()}

    if "ml_features_trial" not in existing:
        cursor.execute("""
        CREATE TABLE ml_features_trial (
            trial_id UNIQUEIDENTIFIER NOT NULL,
            drug_id UNIQUEIDENTIFIER NOT NULL,
            phase_transition_success BIT NULL,
            current_phase VARCHAR(20),
            next_phase_reached VARCHAR(20),
            feat_phase_numeric FLOAT, feat_enrollment INT, feat_enrollment_log FLOAT, feat_enrollment_relative FLOAT,
            feat_number_of_arms INT, feat_has_multiple_arms BIT, feat_is_randomized BIT, feat_is_blinded BIT,
            feat_masking_level INT, feat_is_placebo_controlled BIT, feat_is_active_comparator BIT,
            feat_intervention_model_parallel BIT, feat_intervention_model_crossover BIT,
            feat_primary_purpose_treatment BIT, feat_primary_purpose_prevention BIT,
            feat_is_multicenter BIT, feat_number_of_sites INT, feat_number_of_sites_log FLOAT,
            feat_number_of_countries INT, feat_includes_usa BIT,
            feat_study_duration_days INT, feat_study_duration_months FLOAT,
            feat_has_dmc BIT, feat_has_results BIT, feat_is_fda_regulated BIT,
            feat_gender_all BIT, feat_min_age_years FLOAT, feat_max_age_years FLOAT, feat_age_range_years FLOAT,
            feat_sponsor_type_industry BIT, feat_sponsor_type_academic BIT, feat_sponsor_type_nih BIT,
            feat_sponsor_total_trials INT, feat_sponsor_total_trials_log FLOAT,
            feat_sponsor_success_rate FLOAT, feat_sponsor_drugs_approved INT, feat_sponsor_has_approved_drug BIT,
            feat_drug_prior_approval BIT, feat_drug_num_prior_approvals INT,
            feat_drug_num_indications INT, feat_drug_num_total_trials INT,
            feat_drug_trial_rank INT, feat_drug_years_since_first_trial FLOAT, feat_is_originator_sponsor BIT,
            feat_moa_class VARCHAR(50),
            feat_moa_glp1 BIT, feat_moa_sglt2 BIT, feat_moa_dpp4 BIT, feat_moa_insulin BIT,
            feat_moa_dual_agonist BIT, feat_moa_triple_agonist BIT, feat_moa_other BIT,
            feat_modality_peptide BIT, feat_modality_small_molecule BIT, feat_modality_biologic BIT,
            feat_is_oral BIT, feat_is_injectable BIT,
            feat_indication_t2dm BIT, feat_indication_t1dm BIT, feat_indication_obesity BIT,
            feat_indication_nash BIT, feat_indication_cardiovascular BIT, feat_indication_ckd BIT,
            feat_moa_generation INT, feat_moa_novelty_score FLOAT,
            feat_competing_trials_same_phase INT, feat_competing_drugs_same_moa INT,
            feat_time_since_last_moa_approval_years FLOAT, feat_moa_has_any_approval BIT,
            feat_drug_class_approved_count INT,
            feat_ae_total_count INT, feat_ae_serious_ratio FLOAT, feat_ae_top_event_count INT,
            feat_ae_unique_events INT, feat_ae_trend_slope FLOAT, feat_ae_trend_increasing BIT,
            feat_has_faers_data BIT,
            feat_rx_uk_total_items INT, feat_rx_uk_trend_slope FLOAT, feat_rx_uk_has_data BIT,
            feat_spending_us_total FLOAT, feat_spending_us_per_claim FLOAT,
            feat_spending_us_beneficiaries INT, feat_spending_us_has_data BIT,
            feat_market_data_sources INT,
            feat_has_patent_protection BIT, feat_patent_count INT, feat_years_until_loe FLOAT,
            feat_has_substance_patent BIT, feat_has_use_patent BIT,
            feat_is_stale BIT, feat_days_since_last_update INT, feat_enrollment_vs_target FLOAT,
            feat_was_previously_suspended BIT, feat_has_why_stopped VARCHAR(200),
            feature_version VARCHAR(10) DEFAULT 'v1.0',
            computed_at DATETIME2 DEFAULT GETDATE(),
            PRIMARY KEY (trial_id, drug_id)
        )""")
        logger.info("  CREATED: ml_features_trial")
    else:
        cursor.execute("TRUNCATE TABLE ml_features_trial")
        logger.info("  TRUNCATED: ml_features_trial (idempotent)")

    if "ml_features_drug_indication" not in existing:
        cursor.execute("""
        CREATE TABLE ml_features_drug_indication (
            drug_id UNIQUEIDENTIFIER NOT NULL,
            indication_id UNIQUEIDENTIFIER NOT NULL,
            agg_num_trials INT, agg_num_completed_trials INT, agg_num_terminated_trials INT,
            agg_completion_rate FLOAT, agg_avg_enrollment FLOAT, agg_max_phase_numeric FLOAT,
            agg_avg_sponsor_success_rate FLOAT, agg_total_ae_count INT, agg_avg_ae_serious_ratio FLOAT,
            drug_prior_approval_any BIT, drug_moa_class VARCHAR(50), drug_modality VARCHAR(50),
            drug_originator_type VARCHAR(50),
            indication_name VARCHAR(200), indication_overall_success_rate FLOAT,
            indication_total_drugs INT, indication_approved_drugs INT,
            feature_version VARCHAR(10) DEFAULT 'v1.0',
            computed_at DATETIME2 DEFAULT GETDATE(),
            PRIMARY KEY (drug_id, indication_id)
        )""")
        logger.info("  CREATED: ml_features_drug_indication")
    else:
        cursor.execute("TRUNCATE TABLE ml_features_drug_indication")
        logger.info("  TRUNCATED: ml_features_drug_indication (idempotent)")


# ═══════════════════════════════════════════════════════════
# LOAD BASE DATA
# ═══════════════════════════════════════════════════════════
def load_base_data(conn):
    logger.info("=" * 60)
    logger.info("STEP 2/13: Loading base data from DB")
    logger.info("=" * 60)
    start = time.time()

    # Core trial-drug pairs
    trial_drug = read_sql("""
        SELECT dt.drug_id, dt.trial_id, dt.role,
               t.nct_id, t.phase, t.overall_status, t.start_date, t.completion_date,
               t.primary_completion_date, t.last_update_date, t.enrollment, t.enrollment_type,
               t.study_type, t.has_results, t.why_stopped, t.lead_sponsor_name,
               t.allocation, t.intervention_model, t.primary_purpose, t.masking, t.who_masked,
               t.number_of_arms, t.has_placebo, t.intervention_types, t.has_dmc,
               t.minimum_age, t.maximum_age, t.sex, t.healthy_volunteers,
               t.n_primary_outcomes, t.n_secondary_outcomes, t.is_stale,
               d.inn, d.moa_class, d.modality, d.highest_phase, d.originator_company_id,
               d.first_approval_date
        FROM drug_trials dt
        JOIN trials t ON dt.trial_id = t.trial_id
        JOIN drugs d ON dt.drug_id = d.drug_id
        WHERE t.phase IN ('early_phase1','phase1','phase1_phase2','phase2','phase2_phase3','phase3','phase4')
    """, conn)

    drugs = read_sql("SELECT drug_id, inn, moa_class, modality, highest_phase, originator_company_id, first_approval_date FROM drugs", conn)
    indications = read_sql("SELECT indication_id, name FROM indications", conn)
    drug_ind = read_sql("SELECT drug_id, indication_id, status, phase FROM drug_indications", conn)
    trial_ind = read_sql("SELECT trial_id, indication_id FROM trial_indications", conn)
    approvals = read_sql("SELECT drug_id, approval_date, application_number FROM approvals", conn)

    # Safety data
    ae_data = read_sql("SELECT drug_id, SUM(total_count) as ae_total, SUM(serious_count) as ae_serious, COUNT(DISTINCT event_term) as ae_unique, MAX(total_count) as ae_top FROM adverse_events GROUP BY drug_id", conn)
    ae_trends = read_sql("SELECT drug_id, quarter_date, total_reports, serious_reports FROM adverse_event_trends ORDER BY drug_id, quarter_date", conn)

    # Market data
    uk_rx = read_sql("SELECT drug_id, SUM(items) as total_items, MIN(date) as min_date, MAX(date) as max_date FROM prescriptions_uk GROUP BY drug_id", conn)
    uk_rx_trend = read_sql("SELECT drug_id, date, items FROM prescriptions_uk ORDER BY drug_id, date", conn)
    us_spend = read_sql("SELECT drug_id, SUM(total_spending) as total_spending, SUM(total_claims) as total_claims, SUM(total_beneficiaries) as total_beneficiaries FROM spending_us GROUP BY drug_id", conn)

    # Patent/LOE
    loe = read_sql("SELECT drug_id, effective_loe_date, patent_count, has_substance_patent, has_use_patent, has_product_patent, years_until_loe FROM loe_summary", conn)

    elapsed = time.time() - start
    logger.info(f"  Loaded {len(trial_drug)} trial-drug pairs")
    logger.info(f"  {len(drugs)} drugs, {len(indications)} indications")
    logger.info(f"  {len(approvals)} approvals, {len(ae_data)} drugs with AE data")
    logger.info(f"  {len(uk_rx)} drugs with UK Rx, {len(us_spend)} drugs with US spending")
    logger.info(f"  {len(loe)} drugs with LOE data")
    logger.info(f"  Duration: {elapsed:.1f}s")

    return {
        "trial_drug": trial_drug, "drugs": drugs, "indications": indications,
        "drug_ind": drug_ind, "trial_ind": trial_ind, "approvals": approvals,
        "ae_data": ae_data, "ae_trends": ae_trends,
        "uk_rx": uk_rx, "uk_rx_trend": uk_rx_trend, "us_spend": us_spend, "loe": loe,
    }


# ═══════════════════════════════════════════════════════════
# FEATURE CATEGORIES
# ═══════════════════════════════════════════════════════════

def compute_trial_design(df):
    """STEP 3: Trial Design Features (~30)."""
    logger.info("=" * 60)
    logger.info(f"STEP 3/13: Trial Design Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    df["feat_phase_numeric"] = df["phase"].map(PHASE_NUMERIC)
    df["feat_enrollment"] = df["enrollment"]
    df["feat_enrollment_log"] = np.log1p(df["enrollment"].fillna(0).clip(lower=0))
    df["feat_number_of_arms"] = df["number_of_arms"]
    df["feat_has_multiple_arms"] = (df["number_of_arms"].fillna(0) > 1).astype(int)
    df["feat_is_randomized"] = df["allocation"].fillna("").str.contains("RANDOMIZED", case=False, na=False).astype(int)
    # Exclude NON_RANDOMIZED
    df.loc[df["allocation"] == "NON_RANDOMIZED", "feat_is_randomized"] = 0
    df["feat_is_blinded"] = (~df["masking"].fillna("NONE").isin(["NONE", ""])).astype(int)
    masking_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
    df["feat_masking_level"] = df["masking"].map(masking_map)
    df["feat_is_placebo_controlled"] = df["has_placebo"].fillna(0).astype(int)
    # Active comparator: number_of_arms > 1 and not placebo-only
    df["feat_is_active_comparator"] = ((df["number_of_arms"].fillna(0) > 1) & (df["has_placebo"].fillna(0) == 0)).astype(int)
    df["feat_intervention_model_parallel"] = (df["intervention_model"] == "PARALLEL").astype(int)
    df["feat_intervention_model_crossover"] = (df["intervention_model"] == "CROSSOVER").astype(int)
    df["feat_primary_purpose_treatment"] = (df["primary_purpose"] == "TREATMENT").astype(int)
    df["feat_primary_purpose_prevention"] = (df["primary_purpose"] == "PREVENTION").astype(int)

    # Sites/countries — not in our DB, set to NULL
    df["feat_is_multicenter"] = None
    df["feat_number_of_sites"] = None
    df["feat_number_of_sites_log"] = None
    df["feat_number_of_countries"] = None
    df["feat_includes_usa"] = None

    # Study duration
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    dur = (df["completion_date"] - df["start_date"]).dt.days
    df["feat_study_duration_days"] = dur.where(dur > 0)
    df["feat_study_duration_months"] = df["feat_study_duration_days"] / 30.44

    df["feat_has_dmc"] = df["has_dmc"].fillna(0).astype(int)
    df["feat_has_results"] = df["has_results"].fillna(0).astype(int)
    df["feat_is_fda_regulated"] = None  # not in DB

    # Eligibility
    df["feat_gender_all"] = (df["sex"] == "ALL").astype(int)
    df["feat_min_age_years"] = df["minimum_age"].apply(parse_age_years)
    df["feat_max_age_years"] = df["maximum_age"].apply(parse_age_years)
    df["feat_age_range_years"] = df["feat_max_age_years"] - df["feat_min_age_years"]
    df.loc[df["feat_age_range_years"] < 0, "feat_age_range_years"] = None

    # Enrollment relative (vs avg in same phase x indication — computed later after joining indications)
    df["feat_enrollment_relative"] = None  # placeholder

    elapsed = time.time() - start
    design_cols = [c for c in df.columns if c.startswith("feat_") and any(
        c.startswith(p) for p in ["feat_phase", "feat_enroll", "feat_number_of_arms", "feat_has_m",
                                   "feat_is_r", "feat_is_b", "feat_mask", "feat_is_p", "feat_is_a",
                                   "feat_inter", "feat_prim", "feat_is_multi", "feat_number_of_s",
                                   "feat_number_of_c", "feat_incl", "feat_study", "feat_has_d",
                                   "feat_has_r", "feat_is_f", "feat_gender", "feat_min_a",
                                   "feat_max_a", "feat_age"]
    )]
    non_null_pct = df[design_cols].notna().mean().mean() * 100
    logger.info(f"  Computed: {len(design_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df, non_null_pct


def compute_sponsor_features(df, data):
    """STEP 4: Program/Sponsor Features (~15)."""
    logger.info("=" * 60)
    logger.info(f"STEP 4/13: Program/Sponsor Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    # Sponsor type heuristics from lead_sponsor_name
    name_lower = df["lead_sponsor_name"].fillna("").str.lower()
    df["feat_sponsor_type_industry"] = (
        name_lower.str.contains("pharma|lilly|novo nordisk|sanofi|astrazeneca|merck|pfizer|bayer|boehringer|novartis|roche|johnson|abbott|amgen|gsk|glaxo|bristol|gilead|biogen|takeda|allergan", regex=True)
    ).astype(int)
    df["feat_sponsor_type_academic"] = (
        name_lower.str.contains("university|universit|college|hospital|medical center|institute|akadem|clinic|school of medicine", regex=True)
    ).astype(int)
    df["feat_sponsor_type_nih"] = (
        name_lower.str.contains("national institute|nih|niddk|nhlbi|nci|nimh|veterans|va |department of", regex=True)
    ).astype(int)

    # Sponsor trial counts and success rates (temporal: only trials BEFORE current trial start)
    # Sort by start_date, compute cumulative stats per sponsor, map back via loc
    df_sorted = df.sort_values("start_date", na_position="last")

    # Cumulative count of prior trials per sponsor (shift to exclude current)
    df_sorted["_sponsor_cumcount"] = df_sorted.groupby("lead_sponsor_name").cumcount()
    df_sorted["_sponsor_cum_completed"] = df_sorted.groupby("lead_sponsor_name")["overall_status"].transform(
        lambda x: (x == "completed").cumsum().shift(1, fill_value=0)
    )
    df_sorted["_sponsor_cum_terminated"] = df_sorted.groupby("lead_sponsor_name")["overall_status"].transform(
        lambda x: x.isin(["terminated", "withdrawn"]).cumsum().shift(1, fill_value=0)
    )
    denominator = df_sorted["_sponsor_cum_completed"] + df_sorted["_sponsor_cum_terminated"]
    df_sorted["_sponsor_success_rate"] = df_sorted["_sponsor_cum_completed"] / denominator.replace(0, np.nan)

    # Map back via loc (using original index preserved in df_sorted)
    df["feat_sponsor_total_trials"] = df_sorted.loc[df.index, "_sponsor_cumcount"]
    df["feat_sponsor_total_trials_log"] = np.log1p(df["feat_sponsor_total_trials"].fillna(0))
    df["feat_sponsor_success_rate"] = df_sorted.loc[df.index, "_sponsor_success_rate"]
    df.loc[df["lead_sponsor_name"].isna() | df["start_date"].isna(), ["feat_sponsor_total_trials", "feat_sponsor_success_rate"]] = None

    # Sponsor approved drugs (approximate: total unique drugs approved before trial start)
    approvals = data["approvals"]
    if not approvals.empty and "approval_date" in approvals.columns:
        approvals_dt = approvals.copy()
        approvals_dt["approval_date"] = pd.to_datetime(approvals_dt["approval_date"], errors="coerce")
        approvals_dt = approvals_dt.dropna(subset=["approval_date"]).sort_values("approval_date")
        app_dates = approvals_dt["approval_date"].values  # numpy datetime64
        trial_starts_np = df["start_date"].values  # already datetime64 from step 3
        sponsor_approved = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            ts = trial_starts_np[i]
            if pd.notna(ts):
                sponsor_approved[i] = int(np.searchsorted(app_dates, ts))
        df["feat_sponsor_drugs_approved"] = sponsor_approved
    else:
        df["feat_sponsor_drugs_approved"] = 0
    df["feat_sponsor_has_approved_drug"] = (df["feat_sponsor_drugs_approved"] > 0).astype(int)

    # Ensure approval_date is datetime
    approvals = approvals.copy()
    approvals["approval_date"] = pd.to_datetime(approvals["approval_date"], errors="coerce")

    # Drug prior approval (drug has approval BEFORE this trial start)
    drug_first_approval = approvals.groupby("drug_id")["approval_date"].min().reset_index()
    drug_first_approval.columns = ["drug_id", "first_app_date"]
    df = df.merge(drug_first_approval, on="drug_id", how="left")
    df["feat_drug_prior_approval"] = (
        (df["first_app_date"].notna()) & (df["first_app_date"] < df["start_date"])
    ).astype(int)
    # Vectorized: merge approvals, then count prior per drug
    if not approvals.empty:
        app_by_drug = approvals.groupby("drug_id")["approval_date"].apply(list).to_dict()
        def count_prior_approvals(row):
            if pd.isna(row["start_date"]):
                return 0
            dates = app_by_drug.get(row["drug_id"], [])
            return sum(1 for d in dates if pd.notna(d) and d < row["start_date"])
        df["feat_drug_num_prior_approvals"] = df.apply(count_prior_approvals, axis=1)
    else:
        df["feat_drug_num_prior_approvals"] = 0
    df.drop(columns=["first_app_date"], inplace=True, errors="ignore")

    # Drug program features
    drug_trial_counts = df.groupby("drug_id").size().reset_index(name="drug_total_trials")
    df = df.merge(drug_trial_counts, on="drug_id", how="left")
    df["feat_drug_num_total_trials"] = df["drug_total_trials"]
    df.drop(columns=["drug_total_trials"], inplace=True)

    # Drug indication count
    di = data["drug_ind"]
    drug_ind_counts = di.groupby("drug_id")["indication_id"].nunique().reset_index(name="n_ind")
    df = df.merge(drug_ind_counts, on="drug_id", how="left")
    df["feat_drug_num_indications"] = df["n_ind"].fillna(0).astype(int)
    df.drop(columns=["n_ind"], inplace=True)

    # Drug trial rank (chronological position)
    df["feat_drug_trial_rank"] = df.groupby("drug_id")["start_date"].rank(method="min", na_option="bottom").astype("Int64")

    # Years since first trial for this drug
    drug_first_trial = df.groupby("drug_id")["start_date"].min().reset_index(name="first_trial_date")
    df = df.merge(drug_first_trial, on="drug_id", how="left")
    df["feat_drug_years_since_first_trial"] = (df["start_date"] - df["first_trial_date"]).dt.days / 365.25
    df.drop(columns=["first_trial_date"], inplace=True)

    # Originator sponsor
    df["feat_is_originator_sponsor"] = 0  # complex matching, approximate

    elapsed = time.time() - start
    sponsor_cols = [c for c in df.columns if c.startswith("feat_sponsor") or c.startswith("feat_drug_") or c.startswith("feat_is_orig")]
    non_null_pct = df[sponsor_cols].notna().mean().mean() * 100
    logger.info(f"  Computed: {len(sponsor_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  Duration: {elapsed:.1f}s")
    if elapsed > 60:
        logger.warning(f"  Sponsor features took {elapsed:.0f}s — consider optimizing")
    return df, non_null_pct


def compute_diabetes_features(df, data):
    """STEP 5: Diabetes-specific Features (~25)."""
    logger.info("=" * 60)
    logger.info(f"STEP 5/13: Diabetes-specific Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    moa = df["moa_class"].fillna("")
    df["feat_moa_class"] = df["moa_class"]
    df["feat_moa_glp1"] = moa.str.contains("GLP-1 Receptor Agonist", na=False).astype(int)
    df["feat_moa_sglt2"] = moa.str.contains("SGLT2 Inhibitor|SGLT1/SGLT2", na=False).astype(int)
    df["feat_moa_dpp4"] = moa.str.contains("DPP-4 Inhibitor", na=False).astype(int)
    df["feat_moa_insulin"] = moa.str.contains("Insulin", na=False).astype(int)
    df["feat_moa_dual_agonist"] = moa.str.contains("Dual Agonist", na=False).astype(int)
    df["feat_moa_triple_agonist"] = moa.str.contains("Triple Agonist", na=False).astype(int)
    df["feat_moa_other"] = (
        (df["feat_moa_glp1"] == 0) & (df["feat_moa_sglt2"] == 0) & (df["feat_moa_dpp4"] == 0) &
        (df["feat_moa_insulin"] == 0) & (df["feat_moa_dual_agonist"] == 0) & (df["feat_moa_triple_agonist"] == 0)
    ).astype(int)

    mod = df["modality"].fillna("")
    df["feat_modality_peptide"] = (mod == "peptide").astype(int)
    df["feat_modality_small_molecule"] = (mod == "small_molecule").astype(int)
    df["feat_modality_biologic"] = (mod == "biologic").astype(int)

    # Oral vs injectable (heuristic from modality + moa)
    df["feat_is_oral"] = ((mod == "small_molecule") | moa.str.contains("Oral GLP-1", na=False)).astype(int)
    df["feat_is_injectable"] = ((mod.isin(["peptide", "biologic"])) | moa.str.contains("Insulin", na=False)).astype(int)

    # Indication flags via trial_indications
    ti = data["trial_ind"]
    ind = data["indications"]
    ti_names = ti.merge(ind, on="indication_id", how="left")

    trial_ind_map = {}
    for _, row in ti_names.iterrows():
        tid = row["trial_id"]
        name = str(row["name"]).lower() if pd.notna(row["name"]) else ""
        if tid not in trial_ind_map:
            trial_ind_map[tid] = set()
        trial_ind_map[tid].add(name)

    def has_indication(trial_id, keywords):
        names = trial_ind_map.get(trial_id, set())
        return int(any(kw in n for n in names for kw in keywords))

    df["feat_indication_t2dm"] = df["trial_id"].apply(lambda x: has_indication(x, ["type 2 diabetes"]))
    df["feat_indication_t1dm"] = df["trial_id"].apply(lambda x: has_indication(x, ["type 1 diabetes"]))
    df["feat_indication_obesity"] = df["trial_id"].apply(lambda x: has_indication(x, ["obesity", "overweight"]))
    df["feat_indication_nash"] = df["trial_id"].apply(lambda x: has_indication(x, ["nash", "mash", "fatty liver", "nafld"]))
    df["feat_indication_cardiovascular"] = df["trial_id"].apply(lambda x: has_indication(x, ["cardiovascular", "heart", "mace"]))
    df["feat_indication_ckd"] = df["trial_id"].apply(lambda x: has_indication(x, ["kidney", "renal", "nephro", "dkd", "ckd"]))

    # MoA generation
    df["feat_moa_generation"] = df["moa_class"].map(MOA_GENERATION).fillna(2).astype(int)

    # MoA novelty score
    drug_first_trial = df.groupby(["moa_class", "drug_id"])["start_date"].min().reset_index()
    drug_first_trial = drug_first_trial.sort_values("start_date")
    drug_first_trial["moa_rank"] = drug_first_trial.groupby("moa_class").cumcount() + 1
    novelty_map = drug_first_trial.set_index("drug_id")["moa_rank"]
    df["_moa_rank"] = df["drug_id"].map(novelty_map)
    df["feat_moa_novelty_score"] = df["_moa_rank"].apply(
        lambda x: 1.0 if x == 1 else (0.7 if x <= 3 else 0.3) if pd.notna(x) else 0.5
    )
    df.drop(columns=["_moa_rank"], inplace=True, errors="ignore")

    # Competing trials (active trials in same indication x moa x phase)
    df["feat_competing_trials_same_phase"] = 0  # simplified: count all in same moa+phase
    moa_phase_counts = df.groupby(["moa_class", "phase"]).size().reset_index(name="n_competing")
    df = df.merge(moa_phase_counts, on=["moa_class", "phase"], how="left")
    df["feat_competing_trials_same_phase"] = (df["n_competing"].fillna(0) - 1).clip(lower=0).astype(int)
    df.drop(columns=["n_competing"], inplace=True)

    # Competing drugs in same MoA
    moa_drug_counts = data["drugs"].groupby("moa_class")["drug_id"].nunique().reset_index(name="n_drugs_moa")
    df = df.merge(moa_drug_counts, on="moa_class", how="left")
    df["feat_competing_drugs_same_moa"] = df["n_drugs_moa"].fillna(0).astype(int)
    df.drop(columns=["n_drugs_moa"], inplace=True)

    # Time since last MoA approval
    approvals = data["approvals"]
    app_drugs = approvals.merge(data["drugs"][["drug_id", "moa_class"]], on="drug_id", how="left")
    moa_last_app = app_drugs.groupby("moa_class")["approval_date"].max().reset_index(name="last_moa_approval")
    df = df.merge(moa_last_app, on="moa_class", how="left")
    df["last_moa_approval"] = pd.to_datetime(df["last_moa_approval"], errors="coerce")
    df["feat_time_since_last_moa_approval_years"] = (df["start_date"] - df["last_moa_approval"]).dt.days / 365.25
    df["feat_moa_has_any_approval"] = df["last_moa_approval"].notna().astype(int)
    df.drop(columns=["last_moa_approval"], inplace=True, errors="ignore")

    # Drug class approved count
    moa_app_count = app_drugs.groupby("moa_class")["drug_id"].nunique().reset_index(name="moa_approved_count")
    df = df.merge(moa_app_count, on="moa_class", how="left")
    df["feat_drug_class_approved_count"] = df["moa_approved_count"].fillna(0).astype(int)
    df.drop(columns=["moa_approved_count"], inplace=True)

    elapsed = time.time() - start
    diab_cols = [c for c in df.columns if c.startswith("feat_moa") or c.startswith("feat_modality") or
                 c.startswith("feat_is_oral") or c.startswith("feat_is_inject") or
                 c.startswith("feat_indication") or c.startswith("feat_competing") or
                 c.startswith("feat_time_since") or c.startswith("feat_drug_class")]
    non_null_pct = df[diab_cols].notna().mean().mean() * 100
    logger.info(f"  Computed: {len(diab_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  MoA distribution: {df['feat_moa_class'].value_counts().head(5).to_dict()}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df, non_null_pct


def compute_safety_market(df, data):
    """STEP 6: Safety/Market Signal Features (~15)."""
    logger.info("=" * 60)
    logger.info(f"STEP 6/13: Safety/Market Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    ae = data["ae_data"]
    if not ae.empty:
        ae_map = ae.set_index("drug_id")
        ae_total_dict = ae_map["ae_total"].to_dict()
        ae_serious_dict = ae_map["ae_serious"].to_dict()
        ae_top_dict = ae_map["ae_top"].to_dict()
        ae_unique_dict = ae_map["ae_unique"].to_dict()
        df["feat_ae_total_count"] = df["drug_id"].map(ae_total_dict)
        df["feat_ae_serious_ratio"] = df["drug_id"].map(
            lambda did: ae_serious_dict.get(did, 0) / ae_total_dict[did]
            if did in ae_total_dict and ae_total_dict[did] > 0 else None
        )
        df["feat_ae_top_event_count"] = df["drug_id"].map(ae_top_dict)
        df["feat_ae_unique_events"] = df["drug_id"].map(ae_unique_dict)
    else:
        df["feat_ae_total_count"] = None
        df["feat_ae_serious_ratio"] = None
        df["feat_ae_top_event_count"] = None
        df["feat_ae_unique_events"] = None

    # AE trend slope
    ae_trends = data["ae_trends"]
    trend_slopes = {}
    if not ae_trends.empty:
        for did, grp in ae_trends.groupby("drug_id"):
            if len(grp) >= 3:
                grp = grp.sort_values("quarter_date")
                x = np.arange(len(grp))
                y = grp["total_reports"].values.astype(float)
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    trend_slopes[did] = slope
                except Exception:
                    pass
    df["feat_ae_trend_slope"] = df["drug_id"].map(trend_slopes)
    df["feat_ae_trend_increasing"] = (df["feat_ae_trend_slope"].fillna(0) > 0).astype(int)
    df["feat_has_faers_data"] = df["feat_ae_total_count"].notna().astype(int)

    # UK prescriptions
    uk = data["uk_rx"]
    if not uk.empty:
        uk_items_dict = uk.set_index("drug_id")["total_items"].to_dict()
        df["feat_rx_uk_total_items"] = df["drug_id"].map(uk_items_dict)
    else:
        df["feat_rx_uk_total_items"] = None

    # UK trend slope
    uk_slopes = {}
    uk_trend = data["uk_rx_trend"]
    if not uk_trend.empty:
        for did, grp in uk_trend.groupby("drug_id"):
            if len(grp) >= 3:
                grp = grp.sort_values("date")
                x = np.arange(len(grp))
                y = grp["items"].values.astype(float)
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    uk_slopes[did] = slope
                except Exception:
                    pass
    df["feat_rx_uk_trend_slope"] = df["drug_id"].map(uk_slopes)
    df["feat_rx_uk_has_data"] = df["feat_rx_uk_total_items"].notna().astype(int)

    # US spending
    us = data["us_spend"]
    if not us.empty:
        us_map = us.set_index("drug_id")
        us_spending_dict = us_map["total_spending"].to_dict()
        us_benef_dict = us_map["total_beneficiaries"].to_dict()
        us_claims_dict = us_map["total_claims"].to_dict()
        df["feat_spending_us_total"] = df["drug_id"].map(us_spending_dict)
        df["feat_spending_us_beneficiaries"] = df["drug_id"].map(us_benef_dict)
        spending = df["drug_id"].map(us_spending_dict)
        claims = df["drug_id"].map(us_claims_dict)
        df["feat_spending_us_per_claim"] = spending / claims.replace(0, np.nan)
    else:
        df["feat_spending_us_total"] = None
        df["feat_spending_us_per_claim"] = None
        df["feat_spending_us_beneficiaries"] = None

    df["feat_spending_us_has_data"] = df["feat_spending_us_total"].notna().astype(int)
    df["feat_market_data_sources"] = (
        df["feat_has_faers_data"] + df["feat_rx_uk_has_data"] + df["feat_spending_us_has_data"]
    )

    elapsed = time.time() - start
    safety_cols = [c for c in df.columns if c.startswith("feat_ae") or c.startswith("feat_has_faers") or
                   c.startswith("feat_rx_uk") or c.startswith("feat_spending") or c.startswith("feat_market")]
    non_null_pct = df[safety_cols].notna().mean().mean() * 100
    n_faers = df["feat_has_faers_data"].sum()
    n_uk = df["feat_rx_uk_has_data"].sum()
    n_us = df["feat_spending_us_has_data"].sum()
    logger.info(f"  Computed: {len(safety_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  Coverage: FAERS={n_faers}, UK Rx={n_uk}, US Spending={n_us} trial-drug pairs")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df, non_null_pct


def compute_patent_features(df, data):
    """STEP 7: Patent/LOE Features (~5)."""
    logger.info("=" * 60)
    logger.info(f"STEP 7/13: Patent/LOE Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    loe = data["loe"]
    if not loe.empty:
        loe_map = loe.set_index("drug_id")
        patent_count_dict = loe_map["patent_count"].to_dict()
        years_loe_dict = loe_map["years_until_loe"].to_dict()
        sub_patent_dict = loe_map["has_substance_patent"].to_dict()
        use_patent_dict = loe_map["has_use_patent"].to_dict()
        df["feat_has_patent_protection"] = df["drug_id"].map(
            lambda d: int(patent_count_dict.get(d, 0) > 0) if d in patent_count_dict else 0
        )
        df["feat_patent_count"] = df["drug_id"].map(patent_count_dict)
        df["feat_years_until_loe"] = df["drug_id"].map(years_loe_dict)
        df["feat_has_substance_patent"] = df["drug_id"].map(sub_patent_dict)
        df["feat_has_use_patent"] = df["drug_id"].map(use_patent_dict)
    else:
        for c in ["feat_has_patent_protection", "feat_patent_count", "feat_years_until_loe", "feat_has_substance_patent", "feat_has_use_patent"]:
            df[c] = None

    elapsed = time.time() - start
    patent_cols = [c for c in df.columns if c.startswith("feat_has_patent") or c.startswith("feat_patent") or c.startswith("feat_years_until") or c.startswith("feat_has_sub") or c.startswith("feat_has_use")]
    non_null_pct = df[patent_cols].notna().mean().mean() * 100
    logger.info(f"  Computed: {len(patent_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df, non_null_pct


def compute_operational_features(df):
    """STEP 8: Operational Risk Features (~5)."""
    logger.info("=" * 60)
    logger.info(f"STEP 8/13: Operational Risk Features ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    df["feat_is_stale"] = df["is_stale"].fillna(0).astype(int)
    df["last_update_date"] = pd.to_datetime(df["last_update_date"], errors="coerce")
    df["feat_days_since_last_update"] = (pd.Timestamp.now() - df["last_update_date"]).dt.days
    df["feat_enrollment_vs_target"] = None  # actual vs target not separately tracked
    df["feat_was_previously_suspended"] = (
        (df["overall_status"] == "suspended") |
        df["why_stopped"].fillna("").str.lower().str.contains("suspend")
    ).astype(int)
    df["feat_has_why_stopped"] = df["why_stopped"].fillna("").str[:200]
    df.loc[df["feat_has_why_stopped"] == "", "feat_has_why_stopped"] = None

    elapsed = time.time() - start
    op_cols = ["feat_is_stale", "feat_days_since_last_update", "feat_enrollment_vs_target",
               "feat_was_previously_suspended", "feat_has_why_stopped"]
    non_null_pct = df[op_cols].notna().mean().mean() * 100
    stale = df["feat_is_stale"].sum()
    logger.info(f"  Computed: {len(op_cols)} features")
    logger.info(f"  Non-null rate: {non_null_pct:.1f}%")
    logger.info(f"  Stale trials: {stale}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df, non_null_pct


def compute_target_variable(df, data):
    """STEP 9: Target Variable — Phase Transition Success."""
    logger.info("=" * 60)
    logger.info(f"STEP 9/13: Target Variable ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    df["current_phase"] = df["phase"]
    df["next_phase_reached"] = None
    df["phase_transition_success"] = None

    # Vectorized: For each drug, compute the max phase numeric reached
    drug_max_phase = df.groupby("drug_id")["feat_phase_numeric"].max().to_dict()

    # Pre-compute: for each drug, the phases available (set of phase_numeric values)
    drug_phases = df.groupby("drug_id").apply(
        lambda g: dict(zip(g["feat_phase_numeric"], g["phase"]))
    ).to_dict()

    # Drug approvals
    approved_drugs = set(data["approvals"]["drug_id"].unique())
    known_outcomes = {"completed", "terminated", "withdrawn", "suspended"}

    # Vectorized conditions
    is_known = df["overall_status"].isin(known_outcomes)
    is_not_phase4 = ~df["phase"].isin(["phase4", "na"])
    has_next = df["phase"].isin(NEXT_PHASE.keys())
    eligible = is_known & is_not_phase4 & has_next

    # For eligible rows, compute success vectorized
    for idx in df.index[eligible]:
        phase = df.at[idx, "phase"]
        current_num = PHASE_NUMERIC.get(phase, 0)
        drug_id = df.at[idx, "drug_id"]
        max_phase = drug_max_phase.get(drug_id, 0)

        if max_phase > current_num:
            df.at[idx, "phase_transition_success"] = 1
            # Find which phase was reached
            phases_dict = drug_phases.get(drug_id, {})
            higher = {k: v for k, v in phases_dict.items() if k > current_num}
            if higher:
                df.at[idx, "next_phase_reached"] = higher[min(higher.keys())]
        elif drug_id in approved_drugs and phase == "phase3":
            df.at[idx, "phase_transition_success"] = 1
            df.at[idx, "next_phase_reached"] = "approved"
        else:
            df.at[idx, "phase_transition_success"] = 0

    elapsed = time.time() - start

    # Class balance
    known = df[df["phase_transition_success"].notna()]
    unknown = df[df["phase_transition_success"].isna()]
    logger.info(f"  Trials with known outcome: {len(known)}")
    logger.info(f"  Trials with unknown outcome: {len(unknown)}")

    for phase_name, next_list in NEXT_PHASE.items():
        phase_df = known[known["current_phase"] == phase_name]
        if len(phase_df) > 0:
            success = int(phase_df["phase_transition_success"].sum())
            fail = len(phase_df) - success
            ratio = success / len(phase_df) if len(phase_df) > 0 else 0
            logger.info(f"  {phase_name}: {success} success / {fail} fail ({ratio:.1%})")

    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


def compute_drug_indication_agg(df, data, conn):
    """STEP 10: Drug-Indication Level Aggregation."""
    logger.info("=" * 60)
    logger.info(f"STEP 10/13: Drug-Indication Aggregation ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    ti = data["trial_ind"]
    ind = data["indications"]
    drugs = data["drugs"]
    drug_ind = data["drug_ind"]
    approvals = data["approvals"]

    # Join trials to indications via trial_indications
    trial_features = df.merge(ti, on="trial_id", how="inner")

    if trial_features.empty:
        logger.warning("  No trial-indication pairs found!")
        return

    # Aggregate per drug x indication
    agg = trial_features.groupby(["drug_id", "indication_id"]).agg(
        agg_num_trials=("trial_id", "count"),
        agg_num_completed_trials=("overall_status", lambda x: (x == "completed").sum()),
        agg_num_terminated_trials=("overall_status", lambda x: x.isin(["terminated", "withdrawn"]).sum()),
        agg_avg_enrollment=("feat_enrollment", "mean"),
        agg_max_phase_numeric=("feat_phase_numeric", "max"),
        agg_avg_sponsor_success_rate=("feat_sponsor_success_rate", "mean"),
        agg_total_ae_count=("feat_ae_total_count", "first"),
        agg_avg_ae_serious_ratio=("feat_ae_serious_ratio", "first"),
    ).reset_index()

    agg["agg_completion_rate"] = agg["agg_num_completed_trials"] / (
        agg["agg_num_completed_trials"] + agg["agg_num_terminated_trials"]
    ).replace(0, np.nan)

    # Drug-level features
    drug_map = drugs.set_index("drug_id")
    drug_moa_dict = drug_map["moa_class"].to_dict() if "moa_class" in drug_map.columns else {}
    drug_mod_dict = drug_map["modality"].to_dict() if "modality" in drug_map.columns else {}
    agg["drug_moa_class"] = agg["drug_id"].map(drug_moa_dict)
    agg["drug_modality"] = agg["drug_id"].map(drug_mod_dict)
    approved_drugs = set(approvals["drug_id"].unique())
    agg["drug_prior_approval_any"] = agg["drug_id"].isin(approved_drugs).astype(int)
    agg["drug_originator_type"] = None

    # Indication-level features
    ind_name_dict = ind.set_index("indication_id")["name"].to_dict() if "name" in ind.columns else {}
    agg["indication_name"] = agg["indication_id"].map(ind_name_dict)

    # Indication success rate (from drug_indications)
    di_approved = drug_ind[drug_ind["status"] == "approved"].groupby("indication_id")["drug_id"].nunique().reset_index(name="approved_count")
    di_total = drug_ind.groupby("indication_id")["drug_id"].nunique().reset_index(name="total_count")
    ind_stats = di_total.merge(di_approved, on="indication_id", how="left")
    ind_stats["approved_count"] = ind_stats["approved_count"].fillna(0)
    ind_stats["success_rate"] = ind_stats["approved_count"] / ind_stats["total_count"]
    ind_stats_map = ind_stats.set_index("indication_id")

    ind_sr_dict = ind_stats_map["success_rate"].to_dict() if "success_rate" in ind_stats_map.columns else {}
    ind_tc_dict = ind_stats_map["total_count"].to_dict() if "total_count" in ind_stats_map.columns else {}
    ind_ac_dict = ind_stats_map["approved_count"].to_dict() if "approved_count" in ind_stats_map.columns else {}
    agg["indication_overall_success_rate"] = agg["indication_id"].map(ind_sr_dict)
    agg["indication_total_drugs"] = agg["indication_id"].map(ind_tc_dict)
    agg["indication_approved_drugs"] = agg["indication_id"].map(ind_ac_dict)

    # Insert into DB
    insert_cols = [
        "drug_id", "indication_id",
        "agg_num_trials", "agg_num_completed_trials", "agg_num_terminated_trials",
        "agg_completion_rate", "agg_avg_enrollment", "agg_max_phase_numeric",
        "agg_avg_sponsor_success_rate", "agg_total_ae_count", "agg_avg_ae_serious_ratio",
        "drug_prior_approval_any", "drug_moa_class", "drug_modality", "drug_originator_type",
        "indication_name", "indication_overall_success_rate", "indication_total_drugs", "indication_approved_drugs",
    ]
    n = batch_insert(conn, "ml_features_drug_indication", agg, insert_cols)

    elapsed = time.time() - start
    logger.info(f"  Drug-Indication pairs: {len(agg)}")
    logger.info(f"  Inserted: {n}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return len(agg)


# ═══════════════════════════════════════════════════════════
# REPORTS & VALIDATION
# ═══════════════════════════════════════════════════════════

def generate_quality_report(df, completeness, n_drug_ind):
    """STEP 11: Data Quality Report."""
    logger.info("=" * 60)
    logger.info(f"STEP 11/13: Data Quality Report ({FEATURE_VERSION})")
    logger.info("=" * 60)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    known = df[df["phase_transition_success"].notna()]
    unknown = df[df["phase_transition_success"].isna()]

    report = []
    report.append("=== ML FEATURE ENGINEERING REPORT ===\n")
    report.append(f"Feature Version: {FEATURE_VERSION}")
    report.append(f"Computed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("TRIAL-LEVEL FEATURES:")
    report.append(f"- Total trial-drug pairs with features: {len(df)}")
    report.append(f"- Trials with known outcome (training): {len(known)}")
    report.append(f"- Trials with unknown outcome (prediction): {len(unknown)}")
    report.append(f"- Total feature columns: {len(feat_cols)}\n")

    report.append("FEATURE COMPLETENESS PER CATEGORY:")
    for cat, pct in completeness.items():
        report.append(f"  - {cat}: {pct:.1f}% non-null")

    report.append("\nTARGET VARIABLE DISTRIBUTION:")
    for phase in PHASE_ORDER[:-1]:  # exclude phase4
        phase_df = df[df["current_phase"] == phase]
        if len(phase_df) > 0:
            s = phase_df["phase_transition_success"]
            success = int(s.sum()) if s.notna().any() else 0
            fail = int((s == 0).sum())
            unk = int(s.isna().sum())
            report.append(f"  {phase}: {success} success / {fail} fail / {unk} unknown")

    report.append(f"\nDRUG-INDICATION LEVEL:")
    report.append(f"- Total Drug-Indication pairs: {n_drug_ind}")

    # Missing data
    report.append("\nMISSING DATA:")
    missing_pct = df[feat_cols].isna().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 0.5]
    report.append(f"- Features with >50% missing: {len(high_missing)}")
    for feat, pct in high_missing.items():
        report.append(f"    {feat}: {pct:.1%} missing")

    report_str = "\n".join(report)
    logger.info("\n" + report_str)

    with open(f"{ARTIFACT_DIR}/data_quality_report.txt", "w") as f:
        f.write(report_str)

    # Feature list
    feat_list = pd.DataFrame({
        "feature": feat_cols,
        "non_null_pct": [(1 - df[c].isna().mean()) * 100 for c in feat_cols],
        "dtype": [str(df[c].dtype) for c in feat_cols],
    })
    feat_list.to_csv(f"{ARTIFACT_DIR}/feature_list_v1.csv", index=False)

    # Target distribution
    target_data = []
    for phase in PHASE_ORDER[:-1]:
        phase_df = df[df["current_phase"] == phase]
        s = phase_df["phase_transition_success"]
        target_data.append({
            "phase": phase,
            "success": int(s.sum()) if s.notna().any() else 0,
            "fail": int((s == 0).sum()),
            "unknown": int(s.isna().sum()),
        })
    target_dist = pd.DataFrame(target_data)
    target_dist.to_csv(f"{ARTIFACT_DIR}/target_distribution.csv", index=False)

    return report_str, feat_list, target_dist, len(high_missing)


def run_validation(df):
    """STEP 12: Validation checks."""
    logger.info("=" * 60)
    logger.info(f"STEP 12/13: Validation Checks ({FEATURE_VERSION})")
    logger.info("=" * 60)
    start = time.time()

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    # Numeric features only
    numeric_feats = [c for c in feat_cols if df[c].dtype in ["float64", "int64", "Int64", "float32", "int32"]]

    # Correlation matrix
    if len(numeric_feats) > 1:
        corr = df[numeric_feats].corr()
        corr.to_csv(f"{ARTIFACT_DIR}/feature_correlation_matrix.csv")

        # Find high-correlation pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = abs(corr.iloc[i, j])
                if r > CORRELATION_THRESHOLD:
                    high_corr.append((corr.columns[i], corr.columns[j], r))
        logger.info(f"  High correlation pairs (|r|>{CORRELATION_THRESHOLD}): {len(high_corr)}")
        for f1, f2, r in high_corr[:10]:
            logger.info(f"    {f1} <-> {f2}: r={r:.3f}")
    else:
        high_corr = []
        logger.info("  Not enough numeric features for correlation analysis")

    # Feature completeness report
    completeness = pd.DataFrame({
        "feature": feat_cols,
        "non_null_pct": [(1 - df[c].isna().mean()) * 100 for c in feat_cols],
        "mean": [df[c].mean() if df[c].dtype in ["float64", "int64"] else None for c in feat_cols],
        "std": [df[c].std() if df[c].dtype in ["float64", "int64"] else None for c in feat_cols],
    })
    completeness.to_csv(f"{ARTIFACT_DIR}/feature_completeness_report.csv", index=False)

    elapsed = time.time() - start
    logger.info(f"  Duration: {elapsed:.1f}s")
    return len(high_corr)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info(f"PHASE 4a: ML Feature Engineering ({FEATURE_VERSION})")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    conn = connect_db()
    logger.info("Connected to Azure SQL")

    with mlflow.start_run(run_name=f"feature_eng_{FEATURE_VERSION}_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.set_tags({
            "feature_version": FEATURE_VERSION,
            "data_source": "azure_sql_pharma_pipeline_db",
            "pipeline_step": "feature_engineering",
            "trigger": "manual",
        })

        # Step 1: Create tables
        create_tables(conn)

        # Step 2: Load data
        data = load_base_data(conn)
        df = data["trial_drug"]

        # Steps 3-8: Compute features
        completeness = {}
        df, pct = compute_trial_design(df)
        completeness["Trial Design"] = pct

        df, pct = compute_sponsor_features(df, data)
        completeness["Program/Sponsor"] = pct

        df, pct = compute_diabetes_features(df, data)
        completeness["Diabetes-specific"] = pct

        df, pct = compute_safety_market(df, data)
        completeness["Safety/Market"] = pct

        df, pct = compute_patent_features(df, data)
        completeness["Patent/LOE"] = pct

        df, pct = compute_operational_features(df)
        completeness["Operational Risk"] = pct

        # Step 9: Target variable
        df = compute_target_variable(df, data)

        # Insert trial-level features into DB
        logger.info("=" * 60)
        logger.info("Inserting trial-level features into DB...")
        logger.info("=" * 60)

        insert_cols = ["trial_id", "drug_id", "phase_transition_success", "current_phase", "next_phase_reached"]
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        insert_cols += feat_cols
        n_inserted = batch_insert(conn, "ml_features_trial", df, insert_cols)
        logger.info(f"  Inserted: {n_inserted} rows into ml_features_trial")

        # Step 10: Drug-indication aggregation
        n_drug_ind = compute_drug_indication_agg(df, data, conn) or 0

        # Step 11: Quality report
        report_str, feat_list, target_dist, n_high_missing = generate_quality_report(df, completeness, n_drug_ind)

        # Step 12: Validation
        n_high_corr = run_validation(df)

        # Step 13: MLflow logging
        logger.info("=" * 60)
        logger.info(f"STEP 13/13: MLflow Logging ({FEATURE_VERSION})")
        logger.info("=" * 60)

        known = df[df["phase_transition_success"].notna()]
        mlflow.log_params({
            "total_feature_count": len(feat_cols),
            "feature_categories": "trial_design,sponsor,diabetes,safety_market,patent,operational",
            "target_variable": "phase_transition_success",
            "imputation_strategy": "median_numeric_mode_categorical",
            "temporal_leak_check": True,
        })

        mlflow.log_metric("total_trials_processed", len(df))
        mlflow.log_metric("trials_with_known_outcome", len(known))
        mlflow.log_metric("trials_unknown_outcome", len(df) - len(known))

        for cat, pct in completeness.items():
            mlflow.log_metric(f"feature_completeness_{cat.lower().replace('/', '_').replace(' ', '_')}", pct)

        # Class balance per transition
        for phase in ["phase1", "phase2", "phase3"]:
            phase_df = known[known["current_phase"] == phase]
            if len(phase_df) > 0:
                ratio = phase_df["phase_transition_success"].mean()
                mlflow.log_metric(f"class_balance_{phase}", ratio)

        mlflow.log_metric("high_correlation_pairs", n_high_corr)
        mlflow.log_metric("features_above_50pct_missing", n_high_missing)
        mlflow.log_metric("drug_indication_pairs_total", n_drug_ind)

        total_time = time.time() - total_start
        mlflow.log_metric("computation_time_seconds", total_time)

        # Log artifacts
        for f in os.listdir(ARTIFACT_DIR):
            mlflow.log_artifact(os.path.join(ARTIFACT_DIR, f))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"  MLflow Run ID: {run_id}")

    conn.close()

    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 4a COMPLETE")
    logger.info(f"  Total duration: {total_time/60:.1f} minutes")
    logger.info(f"  Trial features: {n_inserted}")
    logger.info(f"  Drug-indication pairs: {n_drug_ind}")
    logger.info(f"  MLflow Run: {run_id}")
    logger.info(f"\n  Start MLflow UI with: mlflow ui --backend-store-uri file:///{os.path.abspath(MLFLOW_TRACKING_DIR).replace(os.sep, '/')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
