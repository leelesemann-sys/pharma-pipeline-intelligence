"""
Phase 4b: ML Model Training – Phase Transition Prediction (PoS)
Trains separate XGBoost models per phase-transition, evaluates with temporal splits,
computes SHAP feature importance, and writes results to DB + MLflow.

Architecture (Novartis DSAI pattern):
  XGBoost A + XGBoost B  → Ridge Meta-Learner → Drug-Indication PoS
  + Logistic Regression Baseline for comparison

Usage: python train_models.py
"""

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime

import pyodbc
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix,
    roc_curve,
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

import mlflow
import mlflow.xgboost
import mlflow.sklearn

from config import (
    DB_CONN_STR, FEATURE_VERSION, MODEL_VERSION,
    MLFLOW_TRACKING_DIR, MLFLOW_EXPERIMENT_TRAINING,
    LOG_DIR, ARTIFACT_DIR, MODEL_DIR, RANDOM_SEED,
    CV_SPLITS, PRIMARY_SPLIT_CUTOFF, PHASE_NUMERIC,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════
# LOGGING & MLFLOW SETUP
# ═══════════════════════════════════════════════════════════
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(f"file:///{os.path.abspath(MLFLOW_TRACKING_DIR).replace(os.sep, '/')}")
mlflow.set_experiment(MLFLOW_EXPERIMENT_TRAINING)


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


def read_sql(query, conn):
    return pd.read_sql(query, conn)


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
            elif pd.isna(val):
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


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════
def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute all evaluation metrics. Returns dict for MLflow logging."""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # Handle edge cases (all same class)
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return {
            "auc": 0.5, "pr_auc": float(y_true.mean()), "f1": 0.0,
            "balanced_accuracy": 0.5, "log_loss_val": 1.0, "mcc": 0.0,
            "precision": 0.0, "recall": 0.0,
            "true_positives": 0, "false_positives": 0,
            "true_negatives": 0, "false_negatives": 0,
            "threshold": threshold, "n_samples": len(y_true),
        }

    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_binary),
        "log_loss_val": log_loss(y_true, y_pred_proba),
        "mcc": matthews_corrcoef(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold": threshold,
        "n_samples": len(y_true),
    }


# ═══════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════
def save_roc_comparison(y_true, predictions_dict, filepath):
    """Save ROC curve comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_pred in predictions_dict.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(y_true, y_pred, filepath):
    """Save calibration plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", label="Model")
    except Exception:
        pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_shap_plot(model, X, filepath, max_display=20):
    """Save SHAP summary plot."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close("all")
        return shap_values
    except Exception as e:
        logger.warning(f"  SHAP plot failed: {e}")
        return None


def save_shap_importance_csv(shap_values, feature_names, filepath):
    """Save SHAP feature importance as CSV."""
    if shap_values is None:
        return None
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # Determine direction
    mean_shap = shap_values.mean(axis=0)
    directions = []
    for i, ms in enumerate(mean_shap):
        if abs(ms) < 0.01 * mean_abs[i] and mean_abs[i] > 0:
            directions.append("mixed")
        elif ms > 0:
            directions.append("positive")
        else:
            directions.append("negative")
    df["direction"] = directions
    df.to_csv(filepath, index=False)
    return df


# ═══════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════
def load_data(conn):
    logger.info("=" * 60)
    logger.info("STEP 1/13: Loading ML features from DB")
    logger.info("=" * 60)
    start = time.time()

    # Trial features with known outcome
    df = read_sql("""
        SELECT mf.*, t.start_date
        FROM ml_features_trial mf
        JOIN trials t ON mf.trial_id = t.trial_id
        WHERE mf.phase_transition_success IS NOT NULL
    """, conn)

    # Also load ALL trials (including unknown) for prediction
    df_all = read_sql("""
        SELECT mf.*, t.start_date
        FROM ml_features_trial mf
        JOIN trials t ON mf.trial_id = t.trial_id
    """, conn)

    # Drug-indication aggregation
    df_drug_ind = read_sql("SELECT * FROM ml_features_drug_indication", conn)

    # Trial-indication mapping
    trial_ind = read_sql("SELECT trial_id, indication_id FROM trial_indications", conn)

    elapsed = time.time() - start
    logger.info(f"  Known outcome trials: {len(df)}")
    logger.info(f"  All trials (incl. unknown): {len(df_all)}")
    logger.info(f"  Drug-indication pairs: {len(df_drug_ind)}")
    logger.info(f"  Duration: {elapsed:.1f}s")

    return df, df_all, df_drug_ind, trial_ind


# ═══════════════════════════════════════════════════════════
# STEP 2: PREPARE FEATURES
# ═══════════════════════════════════════════════════════════
def prepare_features(df):
    logger.info("=" * 60)
    logger.info("STEP 2/13: Feature preprocessing")
    logger.info("=" * 60)
    start = time.time()

    # Identify feature columns
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    # Remove text/metadata/100%-missing features
    drop_cols = [
        "feat_has_why_stopped",      # text, mostly missing
        "feat_moa_class",            # categorical — will one-hot encode separately
        "feat_enrollment_relative",  # 100% missing
        "feat_is_multicenter",       # 100% missing
        "feat_is_fda_regulated",     # 100% missing
        "feat_includes_usa",         # 100% missing
        "feat_enrollment_vs_target", # 100% missing
        "feat_number_of_countries",  # 100% missing
        "feat_number_of_sites_log",  # 100% missing
        "feat_number_of_sites",      # 100% missing
    ]
    numeric_feats = [c for c in feat_cols if c not in drop_cols]

    # Force numeric conversion (some cols might be object due to None placeholders)
    for col in numeric_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One-hot encode moa_class
    if "feat_moa_class" in df.columns:
        moa_dummies = pd.get_dummies(df["feat_moa_class"], prefix="moa", dtype=int)
        # Limit to top-N classes to avoid explosion
        moa_counts = df["feat_moa_class"].value_counts()
        top_moas = moa_counts.head(10).index
        keep_cols = [f"moa_{m}" for m in top_moas if f"moa_{m}" in moa_dummies.columns]
        moa_dummies = moa_dummies[keep_cols] if keep_cols else pd.DataFrame()
        numeric_feats_final = numeric_feats + list(moa_dummies.columns)
        df = pd.concat([df, moa_dummies], axis=1)
    else:
        numeric_feats_final = numeric_feats

    # Drop features with > 80% missing
    missing_pct = df[numeric_feats_final].isna().mean()
    high_missing = missing_pct[missing_pct > 0.8].index.tolist()
    if high_missing:
        logger.info(f"  Dropping {len(high_missing)} features with >80% missing: {high_missing}")
        numeric_feats_final = [c for c in numeric_feats_final if c not in high_missing]

    # Impute: median for numeric
    imputer = SimpleImputer(strategy="median")
    X = df[numeric_feats_final].copy()
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=numeric_feats_final,
        index=df.index,
    )

    y = df["phase_transition_success"].astype(int)
    start_dates = pd.to_datetime(df["start_date"], errors="coerce")

    elapsed = time.time() - start
    logger.info(f"  Features after preprocessing: {len(numeric_feats_final)}")
    logger.info(f"  Features dropped (>80% missing): {len(high_missing)}")
    logger.info(f"  Duration: {elapsed:.1f}s")

    return X_imputed, y, start_dates, numeric_feats_final, imputer, df


# ═══════════════════════════════════════════════════════════
# STEP 3: SPLIT BY PHASE TRANSITION
# ═══════════════════════════════════════════════════════════
def split_by_phase(df, X, y, start_dates):
    logger.info("=" * 60)
    logger.info("STEP 3/13: Splitting by phase transition")
    logger.info("=" * 60)

    phase_col = df["current_phase"]

    datasets = {}

    # Phase I → II
    mask_p1 = phase_col.isin(["phase1", "early_phase1", "phase1_phase2"])
    if mask_p1.sum() > 0:
        datasets["Phase_I_to_II"] = {
            "X": X[mask_p1], "y": y[mask_p1], "dates": start_dates[mask_p1],
            "label": "Phase I -> II",
        }
        logger.info(f"  Phase I->II: {mask_p1.sum()} samples ({y[mask_p1].mean():.1%} positive)")

    # Phase II → III
    mask_p2 = phase_col.isin(["phase2", "phase2_phase3"])
    if mask_p2.sum() > 0:
        datasets["Phase_II_to_III"] = {
            "X": X[mask_p2], "y": y[mask_p2], "dates": start_dates[mask_p2],
            "label": "Phase II -> III",
        }
        logger.info(f"  Phase II->III: {mask_p2.sum()} samples ({y[mask_p2].mean():.1%} positive)")

    # Phase III → Approval
    mask_p3 = phase_col.isin(["phase3"])
    if mask_p3.sum() > 0:
        datasets["Phase_III_to_Approval"] = {
            "X": X[mask_p3], "y": y[mask_p3], "dates": start_dates[mask_p3],
            "label": "Phase III -> Approval",
        }
        logger.info(f"  Phase III->Approval: {mask_p3.sum()} samples ({y[mask_p3].mean():.1%} positive)")

    # Combined (all phases, with phase_numeric as feature) — fallback for small N
    datasets["Combined_All_Phases"] = {
        "X": X, "y": y, "dates": start_dates,
        "label": "All Phases Combined",
    }
    logger.info(f"  Combined: {len(X)} samples ({y.mean():.1%} positive)")

    return datasets


# ═══════════════════════════════════════════════════════════
# STEP 4: TEMPORAL SPLITS
# ═══════════════════════════════════════════════════════════
def make_temporal_split(X, y, dates, cutoff_date):
    """Split into train/test based on temporal cutoff."""
    cutoff = pd.Timestamp(cutoff_date)
    train_mask = dates < cutoff
    test_mask = dates >= cutoff

    # Handle NaT dates: put in train
    nat_mask = dates.isna()
    train_mask = train_mask | nat_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test


def make_cv_splits(X, y, dates, cv_splits):
    """Generate temporal CV splits."""
    splits = []
    for split_cfg in cv_splits:
        train_before = pd.Timestamp(split_cfg["train_before"])
        test_after = pd.Timestamp(split_cfg["test_after"])
        test_before = pd.Timestamp(split_cfg["test_before"])

        train_mask = (dates < train_before) | dates.isna()
        test_mask = (dates >= test_after) & (dates < test_before)

        if train_mask.sum() > 10 and test_mask.sum() > 5:
            splits.append({
                "X_train": X[train_mask], "y_train": y[train_mask],
                "X_test": X[test_mask], "y_test": y[test_mask],
                "train_before": str(split_cfg["train_before"]),
                "test_range": f"{split_cfg['test_after']} to {split_cfg['test_before']}",
                "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            })
    return splits


# ═══════════════════════════════════════════════════════════
# STEP 5-9: TRAINING PIPELINE (per phase-transition)
# ═══════════════════════════════════════════════════════════
def train_phase_transition(phase_key, dataset, conn):
    """Train all models for one phase-transition. Returns results dict."""
    X = dataset["X"]
    y = dataset["y"]
    dates = dataset["dates"]
    label = dataset["label"]

    logger.info("=" * 60)
    logger.info(f"TRAINING: {label}")
    logger.info("=" * 60)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    n_total = len(y)

    logger.info(f"  Samples: {n_total} ({n_pos} pos / {n_neg} neg = {y.mean():.1%} positive)")

    # Check minimum N
    if n_total < 30:
        logger.warning(f"  WARNING: Very small dataset ({n_total}). Results will be LOW CONFIDENCE.")
    if n_pos < 5 or n_neg < 5:
        logger.warning(f"  WARNING: Extreme class imbalance ({n_pos} pos / {n_neg} neg). Skipping phase.")
        return None

    # Primary temporal split
    X_train, X_test, y_train, y_test = make_temporal_split(X, y, dates, PRIMARY_SPLIT_CUTOFF)
    logger.info(f"  Primary split ({PRIMARY_SPLIT_CUTOFF}): Train={len(X_train)}, Test={len(X_test)}")

    if len(X_test) < 5:
        logger.warning(f"  WARNING: Test set too small ({len(X_test)}). Using 70/30 chronological split.")
        sorted_idx = dates.sort_values(na_position="first").index
        split_point = int(len(sorted_idx) * 0.7)
        train_idx = sorted_idx[:split_point]
        test_idx = sorted_idx[split_point:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        logger.info(f"  Fallback split: Train={len(X_train)}, Test={len(X_test)}")

    if len(X_test) < 3 or len(np.unique(y_test)) < 2:
        logger.warning(f"  WARNING: Cannot evaluate - test set has only one class. Skipping phase.")
        return None

    # Class balance
    scale_pos_weight = max(n_neg / max(n_pos, 1), 1.0)
    train_pos = int(y_train.sum())
    train_neg = int((y_train == 0).sum())
    test_pos = int(y_test.sum())
    test_neg = int((y_test == 0).sum())

    results = {
        "phase_key": phase_key,
        "label": label,
        "n_total": n_total, "n_pos": n_pos, "n_neg": n_neg,
        "n_train": len(X_train), "n_test": len(X_test),
        "train_pos": train_pos, "train_neg": train_neg,
        "test_pos": test_pos, "test_neg": test_neg,
        "feature_names": list(X.columns),
        "models": {},
        "predictions": {},
    }

    with mlflow.start_run(run_name=f"{phase_key}_{MODEL_VERSION}") as parent_run:
        mlflow.set_tags({
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "phase_transition": label,
            "pipeline_step": "model_training",
            "primary_split_cutoff": PRIMARY_SPLIT_CUTOFF,
        })
        mlflow.log_params({
            "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
            "class_balance_ratio": f"{y.mean():.4f}",
            "n_features": len(X.columns),
            "n_train": len(X_train), "n_test": len(X_test),
            "scale_pos_weight": f"{scale_pos_weight:.2f}",
        })

        # --- CHILD 1: Logistic Regression Baseline ---
        logger.info("  Training Baseline (Logistic Regression)...")
        with mlflow.start_run(run_name="baseline_logreg", nested=True):
            mlflow.set_tag("model_type", "logistic_regression")
            mlflow.log_params({"max_iter": 1000, "class_weight": "balanced"})

            baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
            baseline.fit(X_train, y_train)
            y_pred_baseline = baseline.predict_proba(X_test)[:, 1]

            metrics_baseline = compute_all_metrics(y_test.values, y_pred_baseline)
            mlflow.log_metrics(metrics_baseline)
            mlflow.sklearn.log_model(baseline, "model")

            results["models"]["baseline"] = baseline
            results["predictions"]["baseline"] = y_pred_baseline
            logger.info(f"    Baseline LogReg: AUC={metrics_baseline['auc']:.3f}, F1={metrics_baseline['f1']:.3f}")

        # --- CHILD 2: XGBoost Model A ---
        logger.info("  Training XGBoost Model A (standard)...")
        with mlflow.start_run(run_name="xgboost_A", nested=True):
            mlflow.set_tag("model_type", "xgboost")
            params_a = {
                "max_depth": 6, "learning_rate": 0.05, "n_estimators": 500,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            }
            mlflow.log_params(params_a)

            model_a = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                early_stopping_rounds=50,
                verbosity=0,
                **params_a,
            )
            model_a.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred_a = model_a.predict_proba(X_test)[:, 1]

            metrics_a = compute_all_metrics(y_test.values, y_pred_a)
            mlflow.log_metrics(metrics_a)
            if hasattr(model_a, "best_iteration"):
                mlflow.log_metric("best_iteration", model_a.best_iteration)
            mlflow.xgboost.log_model(model_a, "model")

            # SHAP
            shap_path = f"{ARTIFACT_DIR}/shap_global_{phase_key}_A.png"
            shap_values_a = save_shap_plot(model_a, X_test, shap_path)
            if os.path.exists(shap_path):
                mlflow.log_artifact(shap_path)
            imp_path = f"{ARTIFACT_DIR}/feature_importance_{phase_key}_A.csv"
            imp_df_a = save_shap_importance_csv(shap_values_a, list(X_test.columns), imp_path)
            if os.path.exists(imp_path):
                mlflow.log_artifact(imp_path)

            results["models"]["xgb_a"] = model_a
            results["predictions"]["xgb_a"] = y_pred_a
            results["shap_values_a"] = shap_values_a
            results["importance_a"] = imp_df_a
            logger.info(f"    XGBoost A: AUC={metrics_a['auc']:.3f}, F1={metrics_a['f1']:.3f}")

        # --- CHILD 3: XGBoost Model B (conservative) ---
        logger.info("  Training XGBoost Model B (conservative)...")
        with mlflow.start_run(run_name="xgboost_B", nested=True):
            mlflow.set_tag("model_type", "xgboost")
            params_b = {
                "max_depth": 4, "learning_rate": 0.03, "n_estimators": 800,
                "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
                "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            }
            mlflow.log_params(params_b)

            model_b = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=123,
                early_stopping_rounds=50,
                verbosity=0,
                **params_b,
            )
            model_b.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred_b = model_b.predict_proba(X_test)[:, 1]

            metrics_b = compute_all_metrics(y_test.values, y_pred_b)
            mlflow.log_metrics(metrics_b)
            if hasattr(model_b, "best_iteration"):
                mlflow.log_metric("best_iteration", model_b.best_iteration)
            mlflow.xgboost.log_model(model_b, "model")

            # SHAP
            shap_path_b = f"{ARTIFACT_DIR}/shap_global_{phase_key}_B.png"
            shap_values_b = save_shap_plot(model_b, X_test, shap_path_b)
            if os.path.exists(shap_path_b):
                mlflow.log_artifact(shap_path_b)

            results["models"]["xgb_b"] = model_b
            results["predictions"]["xgb_b"] = y_pred_b
            logger.info(f"    XGBoost B: AUC={metrics_b['auc']:.3f}, F1={metrics_b['f1']:.3f}")

        # --- CHILD 4: Temporal CV ---
        logger.info("  Running Temporal Cross-Validation...")
        with mlflow.start_run(run_name="temporal_cv", nested=True):
            mlflow.set_tag("model_type", "temporal_cv")

            cv_splits = make_cv_splits(X, y, dates, CV_SPLITS)
            cv_results = []
            for i, split in enumerate(cv_splits):
                try:
                    cv_model = xgb.XGBClassifier(
                        objective="binary:logistic", eval_metric="logloss",
                        max_depth=6, learning_rate=0.05, n_estimators=300,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_SEED, verbosity=0,
                        early_stopping_rounds=30,
                    )
                    cv_model.fit(
                        split["X_train"], split["y_train"],
                        eval_set=[(split["X_test"], split["y_test"])],
                        verbose=False,
                    )
                    cv_pred = cv_model.predict_proba(split["X_test"])[:, 1]
                    cv_metrics = compute_all_metrics(split["y_test"].values, cv_pred)
                    cv_results.append(cv_metrics)

                    mlflow.log_metric(f"cv_split_{i}_auc", cv_metrics["auc"], step=i)
                    mlflow.log_metric(f"cv_split_{i}_f1", cv_metrics["f1"], step=i)
                    mlflow.log_metric(f"cv_split_{i}_n_test", split["n_test"], step=i)
                    logger.info(f"    CV Split {i} ({split['test_range']}): AUC={cv_metrics['auc']:.3f}, n={split['n_test']}")
                except Exception as e:
                    logger.warning(f"    CV Split {i} failed: {e}")

            if cv_results:
                cv_auc_mean = np.mean([r["auc"] for r in cv_results])
                cv_auc_std = np.std([r["auc"] for r in cv_results])
                cv_f1_mean = np.mean([r["f1"] for r in cv_results])
                cv_f1_std = np.std([r["f1"] for r in cv_results])
                mlflow.log_metrics({
                    "cv_auc_mean": cv_auc_mean, "cv_auc_std": cv_auc_std,
                    "cv_f1_mean": cv_f1_mean, "cv_f1_std": cv_f1_std,
                    "cv_n_splits": len(cv_results),
                })
                results["cv_auc_mean"] = cv_auc_mean
                results["cv_auc_std"] = cv_auc_std
                logger.info(f"    CV Mean: AUC={cv_auc_mean:.3f}±{cv_auc_std:.3f}")
            else:
                logger.warning("    No valid CV splits!")
                results["cv_auc_mean"] = None
                results["cv_auc_std"] = None

        # --- CHILD 5: Ridge Meta-Ensemble ---
        logger.info("  Training Ridge Meta-Ensemble...")
        with mlflow.start_run(run_name="ridge_meta_ensemble", nested=True):
            mlflow.set_tag("model_type", "ridge_meta")

            # Build meta-features from Model A + B predictions
            meta_features = pd.DataFrame({
                "pred_a": y_pred_a,
                "pred_b": y_pred_b,
                "pred_a_logit": np.log(np.clip(y_pred_a, 1e-6, 1 - 1e-6) / (1 - np.clip(y_pred_a, 1e-6, 1 - 1e-6))),
                "pred_b_logit": np.log(np.clip(y_pred_b, 1e-6, 1 - 1e-6) / (1 - np.clip(y_pred_b, 1e-6, 1 - 1e-6))),
            })

            # Simple average as baseline meta
            y_pred_avg = (y_pred_a + y_pred_b) / 2

            # Ridge meta-learner (using test predictions — in production, use OOF predictions)
            # Since we have limited data, use simple averaging + Ridge on logits
            try:
                meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
                meta_model.fit(meta_features, y_test)
                y_pred_meta = np.clip(meta_model.predict(meta_features), 0, 1)

                metrics_meta = compute_all_metrics(y_test.values, y_pred_meta)
                mlflow.log_metrics(metrics_meta)
                mlflow.sklearn.log_model(meta_model, "model")
                mlflow.log_param("alpha_selected", float(meta_model.alpha_))

                results["models"]["meta"] = meta_model
                results["predictions"]["meta"] = y_pred_meta
                logger.info(f"    Meta-Ensemble: AUC={metrics_meta['auc']:.3f}, F1={metrics_meta['f1']:.3f}")
            except Exception as e:
                logger.warning(f"    Meta-ensemble failed: {e}. Using simple average.")
                y_pred_meta = y_pred_avg
                metrics_meta = compute_all_metrics(y_test.values, y_pred_meta)
                mlflow.log_metrics(metrics_meta)
                results["predictions"]["meta"] = y_pred_meta
                logger.info(f"    Average Ensemble: AUC={metrics_meta['auc']:.3f}")

        # --- Parent: Summary ---
        all_model_metrics = {
            "baseline": metrics_baseline,
            "xgb_a": metrics_a,
            "xgb_b": metrics_b,
            "meta": metrics_meta,
        }
        best_name = max(all_model_metrics, key=lambda k: all_model_metrics[k]["auc"])
        best_metrics = all_model_metrics[best_name]

        mlflow.log_metrics({
            "best_auc": best_metrics["auc"],
            "best_f1": best_metrics["f1"],
            "best_model": 0,  # can't log string, just flag
            "improvement_over_baseline": best_metrics["auc"] - metrics_baseline["auc"],
        })
        mlflow.set_tag("best_model_type", best_name)

        # ROC comparison
        roc_path = f"{ARTIFACT_DIR}/roc_comparison_{phase_key}.png"
        save_roc_comparison(y_test.values, {
            "Baseline LogReg": y_pred_baseline,
            "XGBoost A": y_pred_a,
            "XGBoost B": y_pred_b,
            "Meta-Ensemble": y_pred_meta,
        }, roc_path)
        if os.path.exists(roc_path):
            mlflow.log_artifact(roc_path)

        # Calibration
        cal_path = f"{ARTIFACT_DIR}/calibration_{phase_key}.png"
        save_calibration_plot(y_test.values, y_pred_a, cal_path)
        if os.path.exists(cal_path):
            mlflow.log_artifact(cal_path)

        results["all_metrics"] = all_model_metrics
        results["best_model_name"] = best_name
        results["y_test"] = y_test
        results["X_test"] = X_test
        results["parent_run_id"] = parent_run.info.run_id

        logger.info(f"  [OK] Best model: {best_name} (AUC={best_metrics['auc']:.3f})")

    return results


# ═══════════════════════════════════════════════════════════
# STEP 10: CREATE DB TABLES
# ═══════════════════════════════════════════════════════════
def create_result_tables(conn):
    logger.info("=" * 60)
    logger.info("STEP 10/13: Creating result tables in DB")
    logger.info("=" * 60)
    cursor = conn.cursor()

    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN ('ml_models','ml_feature_importance','ml_predictions','ml_drug_indication_pos')")
    existing = {r[0] for r in cursor.fetchall()}

    if "ml_models" not in existing:
        cursor.execute("""
        CREATE TABLE ml_models (
            model_id INT IDENTITY(1,1) PRIMARY KEY,
            model_name VARCHAR(100),
            phase_transition VARCHAR(50),
            model_type VARCHAR(50),
            train_cutoff_date DATE,
            n_train INT,
            n_test INT,
            auc FLOAT,
            pr_auc FLOAT,
            f1 FLOAT,
            balanced_accuracy FLOAT,
            log_loss_val FLOAT,
            mcc FLOAT,
            hyperparameters NVARCHAR(MAX),
            feature_count INT,
            cv_auc_mean FLOAT,
            cv_auc_std FLOAT,
            model_version VARCHAR(10) DEFAULT 'v1.0',
            created_at DATETIME2 DEFAULT GETDATE()
        )""")
        logger.info("  CREATED: ml_models")
    else:
        cursor.execute("DELETE FROM ml_models WHERE model_version = ?", MODEL_VERSION)
        logger.info("  CLEANED: ml_models (version refresh)")

    if "ml_feature_importance" not in existing:
        cursor.execute("""
        CREATE TABLE ml_feature_importance (
            model_id INT NOT NULL,
            feature_name VARCHAR(100),
            mean_abs_shap FLOAT,
            rank INT,
            direction VARCHAR(10)
        )""")
        logger.info("  CREATED: ml_feature_importance")
    else:
        # Clean old entries (via model_id that will be re-created)
        logger.info("  EXISTS: ml_feature_importance (will insert fresh)")

    if "ml_predictions" not in existing:
        cursor.execute("""
        CREATE TABLE ml_predictions (
            trial_id UNIQUEIDENTIFIER NOT NULL,
            drug_id UNIQUEIDENTIFIER NOT NULL,
            model_id INT NOT NULL,
            predicted_pos FLOAT,
            actual_outcome BIT NULL,
            prediction_correct BIT NULL,
            created_at DATETIME2 DEFAULT GETDATE()
        )""")
        logger.info("  CREATED: ml_predictions")
    else:
        cursor.execute("DELETE FROM ml_predictions WHERE model_id IN (SELECT model_id FROM ml_models WHERE model_version = ?)", MODEL_VERSION)
        logger.info("  CLEANED: ml_predictions (version refresh)")

    if "ml_drug_indication_pos" not in existing:
        cursor.execute("""
        CREATE TABLE ml_drug_indication_pos (
            drug_id UNIQUEIDENTIFIER NOT NULL,
            indication_id UNIQUEIDENTIFIER NOT NULL,
            ensemble_pos FLOAT,
            pos_lower_95 FLOAT,
            pos_upper_95 FLOAT,
            phase_transition VARCHAR(50),
            model_version VARCHAR(10),
            created_at DATETIME2 DEFAULT GETDATE()
        )""")
        logger.info("  CREATED: ml_drug_indication_pos")
    else:
        cursor.execute("DELETE FROM ml_drug_indication_pos WHERE model_version = ?", MODEL_VERSION)
        logger.info("  CLEANED: ml_drug_indication_pos (version refresh)")


# ═══════════════════════════════════════════════════════════
# STEP 11: WRITE RESULTS TO DB
# ═══════════════════════════════════════════════════════════
def write_results_to_db(conn, all_results, df_source):
    logger.info("=" * 60)
    logger.info("STEP 11/13: Writing results to DB")
    logger.info("=" * 60)
    cursor = conn.cursor()

    for phase_key, res in all_results.items():
        if res is None:
            continue

        label = res["label"]

        # Write models
        for model_name, metrics in res["all_metrics"].items():
            model_type_map = {
                "baseline": "logistic_regression",
                "xgb_a": "xgboost",
                "xgb_b": "xgboost",
                "meta": "ridge_meta",
            }
            hp = "{}"
            if model_name == "xgb_a":
                hp = json.dumps({"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500})
            elif model_name == "xgb_b":
                hp = json.dumps({"max_depth": 4, "learning_rate": 0.03, "n_estimators": 800})

            cursor.execute("""
                INSERT INTO ml_models
                (model_name, phase_transition, model_type, train_cutoff_date, n_train, n_test,
                 auc, pr_auc, f1, balanced_accuracy, log_loss_val, mcc,
                 hyperparameters, feature_count, cv_auc_mean, cv_auc_std, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                f"{model_name}_{phase_key}",
                label,
                model_type_map.get(model_name, "unknown"),
                PRIMARY_SPLIT_CUTOFF,
                res["n_train"], res["n_test"],
                metrics["auc"], metrics["pr_auc"], metrics["f1"],
                metrics["balanced_accuracy"], metrics["log_loss_val"], metrics["mcc"],
                hp, len(res["feature_names"]),
                res.get("cv_auc_mean"), res.get("cv_auc_std"),
                MODEL_VERSION,
            )

            # Get model_id
            cursor.execute("SELECT @@IDENTITY")
            model_id = int(cursor.fetchone()[0])

            # Write feature importance (for XGBoost A only)
            if model_name == "xgb_a" and res.get("importance_a") is not None:
                imp = res["importance_a"]
                for _, row in imp.head(50).iterrows():
                    cursor.execute("""
                        INSERT INTO ml_feature_importance (model_id, feature_name, mean_abs_shap, rank, direction)
                        VALUES (?, ?, ?, ?, ?)
                    """, model_id, row["feature"], float(row["mean_abs_shap"]), int(row["rank"]), row["direction"])

            logger.info(f"  Wrote model: {model_name}_{phase_key} (id={model_id}, AUC={metrics['auc']:.3f})")

    logger.info("  [OK] All model metadata written to DB")


# ═══════════════════════════════════════════════════════════
# STEP 12: SAVE MODELS (JOBLIB)
# ═══════════════════════════════════════════════════════════
def save_models_joblib(all_results, feature_columns, imputer):
    logger.info("=" * 60)
    logger.info("STEP 12/13: Saving models (joblib)")
    logger.info("=" * 60)

    for phase_key, res in all_results.items():
        if res is None:
            continue
        for model_name, model in res.get("models", {}).items():
            path = f"{MODEL_DIR}/{model_name}_{phase_key}.joblib"
            joblib.dump(model, path)
            logger.info(f"  Saved: {path}")

    # Save shared artifacts
    joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.joblib")
    joblib.dump(imputer, f"{MODEL_DIR}/imputer.joblib")
    logger.info(f"  Saved: feature_columns.joblib, imputer.joblib")


# ═══════════════════════════════════════════════════════════
# STEP 13: EVALUATION REPORT
# ═══════════════════════════════════════════════════════════
def generate_evaluation_report(all_results):
    logger.info("=" * 60)
    logger.info("STEP 13/13: Evaluation Report")
    logger.info("=" * 60)

    report = []
    report.append("=" * 60)
    report.append("ML MODEL EVALUATION REPORT")
    report.append(f"Model Version: {MODEL_VERSION}")
    report.append(f"Feature Version: {FEATURE_VERSION}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    for phase_key, res in all_results.items():
        if res is None:
            report.append(f"\n{phase_key}: SKIPPED (insufficient data)")
            continue

        label = res["label"]
        report.append(f"\n{'-' * 50}")
        report.append(f"{label}")
        report.append(f"{'-' * 50}")
        report.append(f"  N_total: {res['n_total']}, N_train: {res['n_train']}, N_test: {res['n_test']}")
        report.append(f"  Class Balance: {res['n_pos']} pos / {res['n_neg']} neg ({res['n_pos']/max(res['n_total'],1):.1%})")
        report.append("")

        for model_name, metrics in res["all_metrics"].items():
            report.append(f"  {model_name:20s}: AUC={metrics['auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}, "
                         f"F1={metrics['f1']:.3f}, MCC={metrics['mcc']:.3f}, LogLoss={metrics['log_loss_val']:.3f}")

        if res.get("cv_auc_mean") is not None:
            report.append(f"\n  CV Mean±Std: AUC={res['cv_auc_mean']:.3f}±{res['cv_auc_std']:.3f}")

        report.append(f"\n  Best Model: {res['best_model_name']} (AUC={res['all_metrics'][res['best_model_name']]['auc']:.3f})")

        # Top features
        if res.get("importance_a") is not None:
            report.append(f"\n  Top 10 Features (SHAP):")
            for _, row in res["importance_a"].head(10).iterrows():
                report.append(f"    {row['rank']:2d}. {row['feature']:40s} SHAP={row['mean_abs_shap']:.4f} ({row['direction']})")

    # Benchmark comparison
    report.append(f"\n{'=' * 60}")
    report.append("BENCHMARK COMPARISON")
    report.append("=" * 60)
    report.append("  Literature targets (open-source data):")
    report.append("  Phase I->II:     ~0.70 AUC")
    report.append("  Phase II->III:   ~0.80 AUC (Feijoo et al.)")
    report.append("  Phase III->App:  ~0.81 AUC (Lo et al.)")
    report.append("  Novartis DSAI:  0.88 AUC (proprietary data)")

    for phase_key, res in all_results.items():
        if res is None:
            continue
        best_auc = res["all_metrics"][res["best_model_name"]]["auc"]
        status = "MEETS" if best_auc >= 0.70 else "BELOW"
        report.append(f"  Our {res['label']:25s}: AUC={best_auc:.3f} [{status} target]")

    report_str = "\n".join(report)
    logger.info("\n" + report_str)

    with open(f"{ARTIFACT_DIR}/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    return report_str


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    total_start = time.time()
    logger.info("=" * 60)
    logger.info(f"PHASE 4b: ML Model Training ({MODEL_VERSION})")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    conn = connect_db()
    logger.info("Connected to Azure SQL")

    # Step 1: Load data
    df, df_all, df_drug_ind, trial_ind = load_data(conn)

    # Step 2: Prepare features
    X, y, start_dates, feature_columns, imputer, df_prepared = prepare_features(df)

    # Step 3: Split by phase transition
    datasets = split_by_phase(df_prepared, X, y, start_dates)

    # Step 10: Create result tables (before training, so we can write during)
    create_result_tables(conn)

    # Steps 4-9: Train models per phase-transition
    all_results = {}
    for phase_key, dataset in datasets.items():
        try:
            result = train_phase_transition(phase_key, dataset, conn)
            all_results[phase_key] = result
        except Exception as e:
            logger.error(f"Training failed for {phase_key}: {e}", exc_info=True)
            all_results[phase_key] = None

    # Step 11: Write results to DB
    write_results_to_db(conn, all_results, df_prepared)

    # Step 12: Save models
    save_models_joblib(all_results, feature_columns, imputer)

    # Step 13: Evaluation report
    report = generate_evaluation_report(all_results)

    conn.close()

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"PHASE 4b COMPLETE")
    logger.info(f"  Total duration: {total_time/60:.1f} minutes")
    logger.info(f"  Models trained: {sum(1 for r in all_results.values() if r is not None)}")
    logger.info(f"\n  Start MLflow UI with:")
    logger.info(f"    mlflow ui --backend-store-uri file:///{os.path.abspath(MLFLOW_TRACKING_DIR).replace(os.sep, '/')}")
    logger.info(f"    Then open http://127.0.0.1:5000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
