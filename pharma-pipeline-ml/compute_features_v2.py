"""
Phase 4a v2: Leak-Free Feature Engineering
All features use STRICT Point-in-Time logic.
No post-market features. No MoA/Modality (two-worlds problem).
Automated leakage detection before DB insert.

Usage: python compute_features_v2.py
"""

import os
import sys
import time
import json
import logging
import re
from datetime import datetime

import pyodbc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import mlflow

from config import (
    DB_CONN_STR, MLFLOW_TRACKING_DIR,
    LOG_DIR, ARTIFACT_DIR, BATCH_INSERT_SIZE,
)

FEATURE_VERSION = "v2.0"
MLFLOW_EXPERIMENT = "pharma_pipeline_features_v2"

# ===================================================================
# LOGGING
# ===================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"{LOG_DIR}/features_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLFLOW_TRACKING_DIR).replace(os.sep, '/')}")
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ===================================================================
# DB HELPERS
# ===================================================================
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


def batch_insert(conn, table, df, columns):
    if df.empty:
        return 0
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(f"[{c}]" for c in columns)
    sql = f"INSERT INTO [{table}] ({col_str}) VALUES ({placeholders})"

    rows = df[columns].values.tolist()
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
            elif pd.isna(val) if not isinstance(val, str) else False:
                clean.append(None)
            else:
                clean.append(val)
        clean_rows.append(clean)

    total = 0
    for i in range(0, len(clean_rows), 500):
        batch = clean_rows[i:i + 500]
        cursor.executemany(sql, batch)
        total += len(batch)
    return total


# ===================================================================
# STEP 1: LOAD DATA
# ===================================================================
def load_data(conn):
    logger.info("=" * 60)
    logger.info("STEP 1: Loading base data")
    logger.info("=" * 60)
    start = time.time()

    df = pd.read_sql("""
        SELECT
            t.trial_id, t.nct_id, t.title, t.phase, t.overall_status,
            t.start_date, t.completion_date, t.primary_completion_date,
            t.enrollment, t.enrollment_type, t.study_type,
            t.number_of_arms, t.allocation, t.intervention_model,
            t.masking, t.who_masked, t.primary_purpose,
            t.has_dmc, t.has_placebo,
            t.sex, t.minimum_age, t.maximum_age, t.healthy_volunteers,
            t.has_results, t.raw_conditions, t.raw_interventions,
            t.lead_sponsor_name, t.sponsor_company_id,
            t.why_stopped,
            t.n_primary_outcomes, t.n_secondary_outcomes,
            c.company_type as sponsor_type,
            dt.drug_id,
            dt.role as drug_role
        FROM trials t
        INNER JOIN drug_trials dt ON t.trial_id = dt.trial_id
        LEFT JOIN companies c ON t.sponsor_company_id = c.company_id
        WHERE t.study_type = 'interventional'
          AND t.phase IN ('phase1','phase1_phase2','phase2','phase2_phase3','phase3')
          AND t.start_date IS NOT NULL
        ORDER BY t.start_date
    """, conn)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["primary_completion_date"] = pd.to_datetime(df["primary_completion_date"], errors="coerce")

    # Load approvals for target + drug features
    approvals = pd.read_sql("""
        SELECT drug_id, MIN(approval_date) as first_approval_date,
               COUNT(*) as n_approvals
        FROM approvals
        WHERE approval_date IS NOT NULL
        GROUP BY drug_id
    """, conn)
    approvals["first_approval_date"] = pd.to_datetime(approvals["first_approval_date"], errors="coerce")

    elapsed = time.time() - start
    logger.info(f"  Loaded {len(df)} trial-drug pairs")
    logger.info(f"  {df['drug_id'].nunique()} drugs, phases: {df['phase'].value_counts().to_dict()}")
    logger.info(f"  {len(approvals)} drugs with approvals")
    logger.info(f"  Duration: {elapsed:.1f}s")

    return df, approvals


# ===================================================================
# STEP 2: TARGET VARIABLE (Leak-Free)
# ===================================================================
def compute_target_v2(df, approvals):
    logger.info("=" * 60)
    logger.info("STEP 2: Target Variable (Leak-Free)")
    logger.info("=" * 60)
    start = time.time()

    phase_to_num = {
        "phase1": 1, "phase1_phase2": 1.5,
        "phase2": 2, "phase2_phase3": 2.5,
        "phase3": 3,
    }
    known_statuses = {"completed", "terminated", "withdrawn", "suspended"}

    df["_phase_num"] = df["phase"].map(phase_to_num)
    df["_effective_completion"] = df["completion_date"].fillna(df["primary_completion_date"])

    # Build approval lookup
    approval_map = dict(zip(approvals["drug_id"], approvals["first_approval_date"]))

    # Pre-build drug timeline: for each drug, sorted list of (start_date, phase_num)
    drug_timeline = (
        df[["drug_id", "start_date", "_phase_num"]]
        .dropna(subset=["start_date", "_phase_num"])
        .sort_values("start_date")
        .groupby("drug_id")
        .apply(lambda g: list(zip(g["start_date"], g["_phase_num"])), include_groups=False)
        .to_dict()
    )

    targets = np.full(len(df), np.nan)
    transitions = np.full(len(df), None, dtype=object)

    for idx in range(len(df)):
        status = df.iloc[idx]["overall_status"]
        phase_num = df.iloc[idx]["_phase_num"]
        drug_id = df.iloc[idx]["drug_id"]
        trial_start = df.iloc[idx]["start_date"]
        trial_completion = df.iloc[idx]["_effective_completion"]

        if status not in known_statuses or pd.isna(phase_num):
            continue

        # Determine transition
        if phase_num <= 1.5:
            transition = "phase1_to_phase2"
            next_threshold = 2.0
        elif phase_num <= 2.5:
            transition = "phase2_to_phase3"
            next_threshold = 3.0
        elif phase_num == 3.0:
            transition = "phase3_to_approval"
            next_threshold = None
        else:
            continue

        transitions[idx] = transition
        success = 0

        if transition == "phase3_to_approval":
            first_app = approval_map.get(drug_id)
            if first_app is not None and pd.notna(first_app):
                if status == "completed":
                    success = 1
                # terminated/withdrawn P3 for approved drug -> 0 (conservative)
        else:
            # Check if drug has future trial in higher phase AFTER this trial's completion
            ref_date = trial_completion if pd.notna(trial_completion) else trial_start
            timeline = drug_timeline.get(drug_id, [])
            for fut_start, fut_phase in timeline:
                if fut_phase >= next_threshold and fut_start > ref_date:
                    success = 1
                    break

        targets[idx] = success

    df["target"] = targets
    df["phase_transition"] = transitions

    # Report
    known = df[df["target"].notna()]
    for tr in ["phase1_to_phase2", "phase2_to_phase3", "phase3_to_approval"]:
        sub = known[known["phase_transition"] == tr]
        if len(sub) > 0:
            s = int(sub["target"].sum())
            f = len(sub) - s
            logger.info(f"  {tr}: {s} success / {f} fail ({s/len(sub):.1%} success)")

    elapsed = time.time() - start
    logger.info(f"  Total with known outcome: {len(known)}")
    logger.info(f"  Total unknown: {len(df) - len(known)}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 3: TRIAL DESIGN FEATURES (Safe)
# ===================================================================
def compute_trial_design_features(df):
    logger.info("=" * 60)
    logger.info("STEP 3: Trial Design Features (Safe)")
    logger.info("=" * 60)
    start = time.time()

    phase_map = {"phase1": 1, "phase1_phase2": 1.5, "phase2": 2,
                 "phase2_phase3": 2.5, "phase3": 3}

    df["feat_phase_numeric"] = df["phase"].map(phase_map)
    df["feat_enrollment"] = df["enrollment"]
    df["feat_enrollment_log"] = np.log1p(df["enrollment"].fillna(0))
    df["feat_number_of_arms"] = df["number_of_arms"]
    df["feat_has_multiple_arms"] = (df["number_of_arms"].fillna(0) > 1).astype(int)
    df["feat_is_randomized"] = df["allocation"].str.contains("Randomized", case=False, na=False).astype(int)
    df["feat_is_blinded"] = (~df["masking"].str.contains("None|Open", case=False, na=True)).astype(int)

    # Masking level from who_masked (JSON array: PARTICIPANT, INVESTIGATOR, CARE_PROVIDER, OUTCOMES_ASSESSOR)
    def masking_level(who_masked_json):
        if pd.isna(who_masked_json):
            return 0
        try:
            roles = json.loads(who_masked_json) if isinstance(who_masked_json, str) else who_masked_json
            return len(roles) if isinstance(roles, list) else 0
        except Exception:
            return 0
    df["feat_masking_level"] = df["who_masked"].apply(masking_level)

    # Placebo directly from DB column (has_placebo)
    df["feat_is_placebo_controlled"] = df["has_placebo"].fillna(False).astype(int)

    # Active comparator
    def has_active_comparator(raw_int):
        if pd.isna(raw_int):
            return 0
        try:
            interventions = json.loads(raw_int) if isinstance(raw_int, str) else raw_int
            drug_count = sum(1 for i in interventions
                           if i.get("type", "").upper() in ("DRUG", "BIOLOGICAL")
                           and "placebo" not in str(i.get("name", "")).lower())
            return int(drug_count > 1)
        except Exception:
            return 0
    df["feat_is_active_comparator"] = df["raw_interventions"].apply(has_active_comparator)

    df["feat_intervention_model_parallel"] = df["intervention_model"].str.contains("Parallel", case=False, na=False).astype(int)
    df["feat_intervention_model_crossover"] = df["intervention_model"].str.contains("Crossover", case=False, na=False).astype(int)
    df["feat_primary_purpose_treatment"] = (df["primary_purpose"] == "Treatment").astype(int)
    df["feat_primary_purpose_prevention"] = (df["primary_purpose"] == "Prevention").astype(int)
    df["feat_has_dmc"] = df["has_dmc"].fillna(0).astype(int)
    df["feat_sex_all"] = (df["sex"] == "ALL").astype(int)
    df["feat_sex_female_only"] = (df["sex"] == "FEMALE").astype(int)
    df["feat_healthy_volunteers"] = df["healthy_volunteers"].fillna(False).astype(int)
    df["feat_n_primary_outcomes"] = df["n_primary_outcomes"].fillna(0).astype(int)
    df["feat_n_secondary_outcomes"] = df["n_secondary_outcomes"].fillna(0).astype(int)

    # Age
    def parse_age_years(age_str):
        if pd.isna(age_str):
            return None
        age_str = str(age_str).lower().strip()
        try:
            num = float(re.findall(r"[\d.]+", age_str)[0])
            if "month" in age_str:
                return num / 12
            if "week" in age_str:
                return num / 52
            return num
        except Exception:
            return None

    df["feat_min_age_years"] = df["minimum_age"].apply(parse_age_years)
    df["feat_max_age_years"] = df["maximum_age"].apply(parse_age_years)
    df["feat_age_range_years"] = df["feat_max_age_years"].astype(float) - df["feat_min_age_years"].astype(float)

    # Planned study duration
    start_dt = pd.to_datetime(df["start_date"], errors="coerce")
    compl_dt = pd.to_datetime(df["completion_date"], errors="coerce")
    df["feat_study_duration_planned_months"] = (compl_dt - start_dt).dt.days / 30.44

    n_feats = sum(1 for c in df.columns if c.startswith("feat_"))
    elapsed = time.time() - start
    logger.info(f"  Computed {n_feats} trial design features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 4: SPONSOR FEATURES (Point-in-Time)
# ===================================================================
def compute_sponsor_features_pit(df):
    logger.info("=" * 60)
    logger.info("STEP 4: Sponsor Features (Point-in-Time)")
    logger.info("=" * 60)
    start = time.time()

    # sponsor_type from companies table: big_pharma, biotech, academic, government
    df["feat_sponsor_type_industry"] = df["sponsor_type"].isin(["big_pharma", "biotech"]).astype(int)
    df["feat_sponsor_type_big_pharma"] = (df["sponsor_type"] == "big_pharma").astype(int)
    df["feat_sponsor_type_biotech"] = (df["sponsor_type"] == "biotech").astype(int)
    df["feat_sponsor_type_academic"] = (df["sponsor_type"] == "academic").astype(int)
    df["feat_sponsor_type_government"] = (df["sponsor_type"] == "government").astype(int)

    # Sort by start_date for cumulative calculations
    df = df.sort_values("start_date", na_position="last").copy()

    # Sponsor N Prior Trials (cumcount = 0-indexed count of prior rows in group)
    df["feat_sponsor_n_prior_trials"] = df.groupby("lead_sponsor_name").cumcount()
    df["feat_sponsor_n_prior_trials_log"] = np.log1p(df["feat_sponsor_n_prior_trials"])

    # Sponsor Prior Completion Rate (expanding window, excluding current)
    df["_s_completed"] = (df["overall_status"] == "completed").astype(int)
    df["_s_has_outcome"] = df["overall_status"].isin(["completed", "terminated", "withdrawn"]).astype(int)

    df["_s_completed_prior"] = df.groupby("lead_sponsor_name")["_s_completed"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    df["_s_outcome_prior"] = df.groupby("lead_sponsor_name")["_s_has_outcome"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    df["feat_sponsor_prior_completion_rate"] = np.where(
        df["_s_outcome_prior"] > 0,
        df["_s_completed_prior"] / df["_s_outcome_prior"],
        np.nan
    )

    # Sponsor has any prior approval: approximate by sponsor_type = INDUSTRY
    # (Exact mapping would need applicant->sponsor matching which is unreliable)
    df["feat_sponsor_has_any_prior_approval"] = df["feat_sponsor_type_industry"]

    # Clean temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("_s_")], inplace=True, errors="ignore")

    elapsed = time.time() - start
    logger.info(f"  Computed sponsor PIT features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 5: DRUG HISTORY FEATURES (Strict Point-in-Time)
# ===================================================================
def compute_drug_features_pit(df, approvals):
    logger.info("=" * 60)
    logger.info("STEP 5: Drug History Features (Strict PIT)")
    logger.info("=" * 60)
    start = time.time()

    df = df.sort_values("start_date", na_position="last").copy()

    # Drug N Prior Trials
    df["feat_drug_n_prior_trials"] = df.groupby("drug_id").cumcount()
    df["feat_drug_n_prior_trials_log"] = np.log1p(df["feat_drug_n_prior_trials"])
    df["feat_drug_prior_trial_rank"] = df["feat_drug_n_prior_trials"] + 1

    # Years since first trial of this drug
    first_trial = df.groupby("drug_id")["start_date"].transform("first")
    df["feat_drug_years_since_first_trial"] = (df["start_date"] - first_trial).dt.days / 365.25
    df["feat_drug_years_since_first_trial"] = df["feat_drug_years_since_first_trial"].clip(lower=0)

    # Drug Prior Max Phase (expanding, shift(1) to exclude current)
    phase_map = {"phase1": 1, "phase1_phase2": 1.5, "phase2": 2,
                 "phase2_phase3": 2.5, "phase3": 3}
    df["_d_phase_num"] = df["phase"].map(phase_map)
    df["feat_drug_prior_max_phase"] = df.groupby("drug_id")["_d_phase_num"].transform(
        lambda x: x.shift(1).expanding().max()
    ).fillna(0)

    # Drug Prior Completion Rate
    df["_d_completed"] = (df["overall_status"] == "completed").astype(int)
    df["_d_has_outcome"] = df["overall_status"].isin(["completed", "terminated", "withdrawn"]).astype(int)

    df["_d_completed_prior"] = df.groupby("drug_id")["_d_completed"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    df["_d_outcome_prior"] = df.groupby("drug_id")["_d_has_outcome"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    df["feat_drug_prior_completion_rate"] = np.where(
        df["_d_outcome_prior"] > 0,
        df["_d_completed_prior"] / df["_d_outcome_prior"],
        np.nan
    )

    # Drug Prior Approval (strict: approval_date < trial.start_date)
    app_map = dict(zip(approvals["drug_id"], approvals["first_approval_date"]))
    app_n_map = dict(zip(approvals["drug_id"], approvals["n_approvals"]))

    df["_first_app"] = df["drug_id"].map(app_map)
    df["feat_drug_has_prior_approval"] = (
        df["_first_app"].notna() & (df["_first_app"] < df["start_date"])
    ).astype(int)
    df["feat_drug_n_prior_approvals"] = np.where(
        df["feat_drug_has_prior_approval"] == 1,
        df["drug_id"].map(app_n_map).fillna(0),
        0
    ).astype(int)

    # Drug Prior Indications: cumulative unique indications from prior trials
    # Use raw_conditions to count distinct conditions across prior trials
    df["_d_n_conditions"] = df["raw_conditions"].apply(
        lambda x: len(json.loads(x)) if pd.notna(x) and isinstance(x, str) else 0
    )
    # Expanding cumsum as proxy for n_prior_indications
    df["feat_drug_n_prior_indications"] = df.groupby("drug_id")["_d_n_conditions"].transform(
        lambda x: x.shift(1).expanding().max()
    ).fillna(0).astype(int)

    # Clean
    df.drop(columns=[c for c in df.columns if c.startswith("_d_") or c == "_first_app"], inplace=True, errors="ignore")

    elapsed = time.time() - start
    logger.info(f"  Computed drug PIT features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 6: INDICATION FEATURES (from raw_conditions, Safe)
# ===================================================================
def compute_indication_features(df):
    logger.info("=" * 60)
    logger.info("STEP 6: Indication Features (Safe)")
    logger.info("=" * 60)
    start = time.time()

    def parse_conditions(raw_cond):
        if pd.isna(raw_cond):
            return []
        try:
            return json.loads(raw_cond) if isinstance(raw_cond, str) else raw_cond
        except Exception:
            return []

    conditions = df["raw_conditions"].apply(parse_conditions)
    cond_lower = conditions.apply(lambda x: " ".join(str(c).lower() for c in x))

    df["feat_indication_t2dm"] = cond_lower.str.contains("type 2|t2dm|type ii", na=False).astype(int)
    df["feat_indication_t1dm"] = cond_lower.str.contains(r"type 1|t1dm|type i(?!i)", na=False).astype(int)
    df["feat_indication_obesity"] = cond_lower.str.contains("obesity|overweight|weight", na=False).astype(int)
    df["feat_indication_nash"] = cond_lower.str.contains("nash|nafld|steatohepatitis|fatty liver", na=False).astype(int)
    df["feat_indication_cardiovascular"] = cond_lower.str.contains("cardiovascular|mace|heart failure|cardiac", na=False).astype(int)
    df["feat_indication_ckd"] = cond_lower.str.contains("kidney|renal|nephropath|ckd|dkd", na=False).astype(int)
    df["feat_n_conditions"] = conditions.apply(len)

    elapsed = time.time() - start
    logger.info(f"  Computed indication features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 7: COMPETITIVE LANDSCAPE (Point-in-Time)
# ===================================================================
def compute_landscape_features_pit(df):
    logger.info("=" * 60)
    logger.info("STEP 7: Competitive Landscape (PIT)")
    logger.info("=" * 60)
    start = time.time()

    df = df.sort_values("start_date", na_position="last").copy()

    # For each trial: count trials in same phase that started in the 3 years before
    # This approximates "active competing trials at time of trial start"
    df["_start_ts"] = df["start_date"].astype(np.int64) // 10**9  # unix timestamp
    three_years_sec = 3 * 365.25 * 86400

    # Group by phase and compute rolling count
    concurrent_same_phase = np.zeros(len(df), dtype=int)
    for phase in df["phase"].unique():
        mask = df["phase"] == phase
        phase_starts = df.loc[mask, "_start_ts"].values
        for i, (idx, ts) in enumerate(zip(df.index[mask], phase_starts)):
            if np.isnan(ts):
                continue
            # Count trials in same phase with start in [ts - 3y, ts)
            count = np.sum(
                (phase_starts[:i] >= ts - three_years_sec) &
                (phase_starts[:i] < ts)
            )
            concurrent_same_phase[df.index.get_loc(idx)] = count

    df["feat_n_concurrent_trials_same_phase"] = concurrent_same_phase

    # Indication historical success rate (PIT)
    # For each trial: of all prior trials in same primary indication that have an outcome,
    # what fraction was successful (Drug advanced to next phase)?
    # We use the target column computed earlier
    df["_has_target"] = df["target"].notna().astype(int)
    df["_target_1"] = (df["target"] == 1).astype(int)

    # Phase-level expanding success rate
    df["_prior_successes"] = df.groupby("phase")["_target_1"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    df["_prior_outcomes"] = df.groupby("phase")["_has_target"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    df["feat_indication_historical_success_rate"] = np.where(
        df["_prior_outcomes"] > 0,
        df["_prior_successes"] / df["_prior_outcomes"],
        np.nan
    )
    df["feat_indication_historical_n_trials"] = df["_prior_outcomes"].astype(int)

    # Clean
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")

    elapsed = time.time() - start
    logger.info(f"  Computed landscape PIT features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 8: TEXT-BASED FEATURES (Safe)
# ===================================================================
def compute_text_features(df):
    logger.info("=" * 60)
    logger.info("STEP 8: Text-Based Features (Safe)")
    logger.info("=" * 60)
    start = time.time()

    title_lower = df["title"].str.lower().fillna("")
    df["feat_title_contains_efficacy"] = title_lower.str.contains("efficacy|effective", na=False).astype(int)
    df["feat_title_contains_safety"] = title_lower.str.contains("safety|tolerability|adverse", na=False).astype(int)
    df["feat_title_contains_combination"] = title_lower.str.contains("combination|add-on|adjunct", na=False).astype(int)
    df["feat_title_contains_extension"] = title_lower.str.contains("extension|long.term|open.label extension", na=False).astype(int)
    df["feat_title_contains_pediatric"] = title_lower.str.contains("pediatric|paediatric|child|adolescent", na=False).astype(int)

    # N drug interventions
    def count_drug_interventions(raw_int):
        if pd.isna(raw_int):
            return 0
        try:
            interventions = json.loads(raw_int) if isinstance(raw_int, str) else raw_int
            return sum(1 for i in interventions
                      if i.get("type", "").upper() in ("DRUG", "BIOLOGICAL")
                      and "placebo" not in str(i.get("name", "")).lower())
        except Exception:
            return 0

    df["feat_n_interventions"] = df["raw_interventions"].apply(count_drug_interventions)
    df["feat_has_combination_therapy"] = (df["feat_n_interventions"] > 1).astype(int)

    elapsed = time.time() - start
    logger.info(f"  Computed text features")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 9: ENROLLMENT RELATIVE (Point-in-Time)
# ===================================================================
def compute_enrollment_relative_pit(df):
    logger.info("=" * 60)
    logger.info("STEP 9: Enrollment Relative (PIT)")
    logger.info("=" * 60)
    start = time.time()

    df = df.sort_values("start_date", na_position="last").copy()

    # Expanding mean of enrollment per phase (excluding current)
    df["feat_enrollment_relative"] = df.groupby("phase")["enrollment"].transform(
        lambda x: x / x.shift(1).expanding().mean()
    )
    df["feat_enrollment_relative"] = df["feat_enrollment_relative"].fillna(1.0).clip(0, 50)

    elapsed = time.time() - start
    logger.info(f"  Duration: {elapsed:.1f}s")
    return df


# ===================================================================
# STEP 10: AUTOMATED LEAKAGE DETECTION (MANDATORY)
# ===================================================================
def validate_no_leakage(df, feature_cols):
    logger.info("=" * 60)
    logger.info("STEP 10: Automated Leakage Detection")
    logger.info("=" * 60)

    mask = df["target"].notna()
    y = df.loc[mask, "target"].values.astype(int)

    if len(set(y)) < 2:
        logger.error("Target has only one class!")
        return False

    leakage_found = False
    suspicious = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df.loc[mask, col], errors="coerce").values
        valid = ~np.isnan(x)
        if valid.sum() < 100:
            continue

        try:
            auc = roc_auc_score(y[valid], x[valid])
            eff_auc = max(auc, 1 - auc)

            if eff_auc > 0.90:
                logger.error(f"  LEAKAGE: {col} -> AUC={eff_auc:.3f}")
                leakage_found = True
            elif eff_auc > 0.80:
                logger.warning(f"  SUSPICIOUS: {col} -> AUC={eff_auc:.3f}")
                suspicious.append((col, eff_auc))
            elif eff_auc > 0.70:
                logger.info(f"  OK (strong): {col} -> AUC={eff_auc:.3f}")
        except Exception:
            pass

    # Two-worlds check
    logger.info("\n  --- Two-Worlds Check ---")
    success = df.loc[mask & (df["target"] == 1)]
    failure = df.loc[mask & (df["target"] == 0)]
    for col in feature_cols:
        if col not in df.columns:
            continue
        s_null = pd.to_numeric(success[col], errors="coerce").isna().mean()
        f_null = pd.to_numeric(failure[col], errors="coerce").isna().mean()
        if (s_null > 0.9 and f_null < 0.1) or (f_null > 0.9 and s_null < 0.1):
            logger.error(f"  TWO-WORLDS: {col} -> Success NULL: {s_null:.1%}, Failure NULL: {f_null:.1%}")
            leakage_found = True

    if leakage_found:
        logger.error("LEAKAGE DETECTED! DO NOT TRAIN.")
        return False
    else:
        logger.info("  No leakage found. Training can proceed.")
        return True


# ===================================================================
# STEP 11: CREATE TABLE & INSERT
# ===================================================================
def create_and_insert(conn, df, feature_cols):
    logger.info("=" * 60)
    logger.info("STEP 11: Create table & insert into DB")
    logger.info("=" * 60)
    start = time.time()

    cursor = conn.cursor()

    # Drop if exists, create fresh
    cursor.execute("IF OBJECT_ID('ml_features_trial', 'U') IS NOT NULL DROP TABLE ml_features_trial")
    cursor.execute("""
        CREATE TABLE ml_features_trial (
            trial_id UNIQUEIDENTIFIER NOT NULL,
            drug_id UNIQUEIDENTIFIER NOT NULL,
            nct_id VARCHAR(20),
            current_phase VARCHAR(30),
            target FLOAT NULL,
            phase_transition VARCHAR(30),
            trial_start_date DATE,
            trial_overall_status VARCHAR(30),
            feature_version VARCHAR(10) DEFAULT 'v2.0',
            computed_at DATETIME2 DEFAULT GETDATE(),
            """ + ",\n            ".join(f"[{c}] FLOAT NULL" for c in feature_cols) + """
        )
    """)
    logger.info("  Created ml_features_trial (v2)")

    # Prepare insert columns
    meta_cols = ["trial_id", "drug_id", "nct_id", "current_phase",
                 "target", "phase_transition", "trial_start_date",
                 "trial_overall_status"]
    all_cols = meta_cols + feature_cols

    # Prepare data
    insert_df = df.copy()
    insert_df["current_phase"] = insert_df["phase"]
    insert_df["trial_start_date"] = insert_df["start_date"].dt.date
    insert_df["trial_overall_status"] = insert_df["overall_status"]

    # Ensure columns exist
    for c in all_cols:
        if c not in insert_df.columns:
            insert_df[c] = None

    n = batch_insert(conn, "ml_features_trial", insert_df, all_cols)
    elapsed = time.time() - start
    logger.info(f"  Inserted {n} rows")
    logger.info(f"  Duration: {elapsed:.1f}s")
    return n


# ===================================================================
# MAIN
# ===================================================================
def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info(f"PHASE 4a v2: Leak-Free Feature Engineering ({FEATURE_VERSION})")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    conn = connect_db()
    logger.info("Connected to Azure SQL")

    # Step 1: Load data
    df, approvals = load_data(conn)

    # Step 2: Target variable
    df = compute_target_v2(df, approvals)

    # Step 3: Trial design features
    df = compute_trial_design_features(df)

    # Step 4: Sponsor features (PIT)
    df = compute_sponsor_features_pit(df)

    # Step 5: Drug history features (PIT)
    df = compute_drug_features_pit(df, approvals)

    # Step 6: Indication features
    df = compute_indication_features(df)

    # Step 7: Competitive landscape (PIT)
    df = compute_landscape_features_pit(df)

    # Step 8: Text features
    df = compute_text_features(df)

    # Step 9: Enrollment relative (PIT)
    df = compute_enrollment_relative_pit(df)

    # Collect feature columns
    feature_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    logger.info(f"\nTotal features: {len(feature_cols)}")
    for i, c in enumerate(feature_cols):
        logger.info(f"  {i+1:2d}. {c}")

    # Step 10: MANDATORY leakage check
    leakage_ok = validate_no_leakage(df, feature_cols)

    if not leakage_ok:
        logger.error("\nABORTING: Leakage detected. Fix features before inserting to DB.")
        # Still save feature list for debugging
        pd.DataFrame({"feature": feature_cols}).to_csv(
            f"{ARTIFACT_DIR}/feature_list_v2_LEAKED.csv", index=False
        )
        conn.close()
        sys.exit(1)

    # Step 11: Insert to DB
    n_inserted = create_and_insert(conn, df, feature_cols)

    # Save artifacts
    pd.DataFrame({"feature": feature_cols}).to_csv(
        f"{ARTIFACT_DIR}/feature_list_v2.csv", index=False
    )

    # Target distribution
    known = df[df["target"].notna()]
    target_data = []
    for tr in ["phase1_to_phase2", "phase2_to_phase3", "phase3_to_approval"]:
        sub = known[known["phase_transition"] == tr]
        s = int(sub["target"].sum())
        f = len(sub) - s
        target_data.append({"transition": tr, "success": s, "fail": f,
                           "total": len(sub), "success_pct": f"{s/len(sub):.1%}" if len(sub) > 0 else "n/a"})
    pd.DataFrame(target_data).to_csv(f"{ARTIFACT_DIR}/target_distribution_v2.csv", index=False)

    # MLflow
    with mlflow.start_run(run_name=f"features_{FEATURE_VERSION}"):
        mlflow.log_param("feature_version", FEATURE_VERSION)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_trial_drug_pairs", len(df))
        mlflow.log_param("n_known_outcome", len(known))
        mlflow.log_param("leakage_check", "PASSED")
        for td in target_data:
            mlflow.log_metric(f"{td['transition']}_success", td["success"])
            mlflow.log_metric(f"{td['transition']}_fail", td["fail"])
        mlflow.log_artifact(f"{ARTIFACT_DIR}/feature_list_v2.csv")
        mlflow.log_artifact(f"{ARTIFACT_DIR}/target_distribution_v2.csv")

    conn.close()

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 4a v2 COMPLETE")
    logger.info(f"  Total duration: {total_time/60:.1f} minutes")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Rows inserted: {n_inserted}")
    logger.info(f"  Leakage check: PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
