"""
Phase 4 Diagnostic: Model Performance Plausibility Check
Runs all 6 diagnoses and generates phase4_diagnostic_report.md
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import pyodbc
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from config import DB_CONN_STR, PHASE_NUMERIC, MODEL_DIR, ARTIFACT_DIR

warnings.filterwarnings("ignore")

# ===================================================================
# DB CONNECTION
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
                print(f"  DB connect attempt {attempt+1} failed, retry in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise

def read_sql(query, conn):
    return pd.read_sql(query, conn)

report_sections = {}

conn = connect_db()
print("Connected to Azure SQL\n")

# ===================================================================
# DIAGNOSE 1: TARGET-VERTEILUNG (Class Balance)
# ===================================================================
print("=" * 70)
print("DIAGNOSE 1: TARGET-VERTEILUNG (Class Balance)")
print("=" * 70)

# 1a: Gesamte Target-Verteilung pro Phase
d1a = read_sql("""
    SELECT
        current_phase,
        phase_transition_success as target,
        COUNT(*) as cnt
    FROM ml_features_trial
    WHERE phase_transition_success IS NOT NULL
    GROUP BY current_phase, phase_transition_success
    ORDER BY current_phase, phase_transition_success
""", conn)
print("\n1a) Target-Verteilung pro Phase:")
print(d1a.to_string(index=False))

# Compute success rates per phase group
phase_groups = {
    "Phase_I_to_II": ["phase1", "early_phase1", "phase1_phase2"],
    "Phase_II_to_III": ["phase2", "phase2_phase3"],
    "Phase_III_to_Approval": ["phase3"],
}

d1a_summary = []
for group_name, phases in phase_groups.items():
    grp = d1a[d1a["current_phase"].isin(phases)]
    total = grp["cnt"].sum()
    success = grp[grp["target"] == 1]["cnt"].sum()
    fail = grp[grp["target"] == 0]["cnt"].sum()
    rate = success / total * 100 if total > 0 else 0
    d1a_summary.append({"Transition": group_name, "Success": int(success), "Failure": int(fail), "Total": int(total), "Success%": f"{rate:.1f}%"})
    print(f"\n  {group_name}: {success} success / {fail} fail ({rate:.1f}% success)")

# Expected: Phase I->II: 60-65%, Phase II->III: 25-35%, Phase III->Approval: 50-60%
print("\n  Expected (Hay 2014): P1->2: 60-65%, P2->3: 25-35%, P3->App: 50-60%")

# 1b: Drug sources
d1b = read_sql("""
    SELECT
        CASE WHEN d.moa_class IS NULL THEN 'Nachtrag 3 (neu, ohne MoA)'
             ELSE 'Original (mit MoA)'
        END as drug_source,
        COUNT(DISTINCT d.drug_id) as drug_count,
        COUNT(DISTINCT dt.trial_id) as trial_count
    FROM drugs d
    LEFT JOIN drug_trials dt ON d.drug_id = dt.drug_id
    GROUP BY CASE WHEN d.moa_class IS NULL THEN 'Nachtrag 3 (neu, ohne MoA)'
                  ELSE 'Original (mit MoA)' END
""", conn)
print("\n1b) Drug-Quellen:")
print(d1b.to_string(index=False))

# 1c: Target per source
try:
    d1c = read_sql("""
        SELECT
            CASE WHEN d.moa_class IS NULL THEN 'Nachtrag 3' ELSE 'Original' END as source,
            mf.current_phase,
            mf.phase_transition_success as target,
            COUNT(*) as cnt
        FROM ml_features_trial mf
        JOIN drugs d ON mf.drug_id = d.drug_id
        WHERE mf.phase_transition_success IS NOT NULL
        GROUP BY
            CASE WHEN d.moa_class IS NULL THEN 'Nachtrag 3' ELSE 'Original' END,
            mf.current_phase, mf.phase_transition_success
        ORDER BY source, mf.current_phase, mf.phase_transition_success
    """, conn)
    print("\n1c) Target pro Drug-Quelle:")
    print(d1c.to_string(index=False))
except Exception as e:
    print(f"\n1c) Fehler: {e}")
    d1c = pd.DataFrame()

report_sections["diagnose_1"] = {
    "target_distribution": d1a_summary,
    "drug_sources": d1b.to_dict("records"),
}


# ===================================================================
# DIAGNOSE 2: DATA LEAKAGE -- Feature-Analyse
# ===================================================================
print("\n" + "=" * 70)
print("DIAGNOSE 2: DATA LEAKAGE -- Feature-Analyse")
print("=" * 70)

# 2a: Feature Importance aus DB
print("\n2a) Feature Importance aus DB (ml_feature_importance):")
try:
    d2a = read_sql("""
        SELECT
            fi.model_id,
            m.model_name,
            m.phase_transition,
            fi.feature_name,
            fi.mean_abs_shap,
            fi.rank as importance_rank,
            fi.direction
        FROM ml_feature_importance fi
        JOIN ml_models m ON fi.model_id = m.model_id
        WHERE fi.rank <= 20
        ORDER BY m.phase_transition, fi.rank
    """, conn)
    if len(d2a) > 0:
        for pt in d2a["phase_transition"].unique():
            sub = d2a[d2a["phase_transition"] == pt].head(20)
            print(f"\n  === {pt} - Top 20 Features ===")
            for _, row in sub.iterrows():
                print(f"    {int(row['importance_rank']):2d}. {row['feature_name']:45s} SHAP={row['mean_abs_shap']:.4f} ({row['direction']})")
    else:
        print("  (Keine Feature Importance in DB -- SHAP fehlgeschlagen)")
except Exception as e:
    print(f"  DB-Abfrage fehlgeschlagen: {e}")
    d2a = pd.DataFrame()

# 2a-alt: Lade Joblib-Modelle fuer Feature Importance
print("\n2a-alt) Feature Importance aus Joblib-Modellen:")
feature_cols = None
try:
    feature_cols = joblib.load(f'{MODEL_DIR}/feature_columns.joblib')
    print(f"  Feature columns loaded: {len(feature_cols)} features")
except Exception as e:
    print(f"  Fehler beim Laden: {e}")

model_importance = {}
for transition in ['Phase_I_to_II', 'Phase_II_to_III', 'Phase_III_to_Approval', 'Combined_All_Phases']:
    try:
        model = joblib.load(f'{MODEL_DIR}/xgb_a_{transition}.joblib')
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        print(f"\n  === {transition} - Top 20 Features (XGBoost A gain) ===")
        top_feats = []
        for i, idx in enumerate(top_idx):
            feat_name = feature_cols[idx] if feature_cols is not None and idx < len(feature_cols) else f"feat_{idx}"
            imp_val = importances[idx]
            print(f"    {i+1:2d}. {feat_name:45s}: {imp_val:.4f}")
            top_feats.append({"rank": i+1, "feature": feat_name, "importance": float(imp_val)})
        model_importance[transition] = top_feats
    except Exception as e:
        print(f"  Error loading {transition}: {e}")


# 2b: Leakage-Suspects pruefen
print("\n2b) Leakage-Suspect Check:")
LEAKAGE_SUSPECTS = [
    'has_results', 'overall_status', 'why_stopped',
    'completion_date', 'primary_completion_date', 'last_update_date',
    'first_approval_date', 'highest_phase', 'approval_count',
    'has_fda_approval', 'is_approved',
    'n_trials_completed', 'n_prior_approvals',
    'is_stale', 'was_previously_suspended',
]

leakage_found = []
if feature_cols:
    for suspect in LEAKAGE_SUSPECTS:
        matches = [f for f in feature_cols if suspect.lower() in f.lower()]
        if matches:
            print(f"  !! FOUND: {suspect} -> {matches}")
            leakage_found.extend(matches)
        else:
            print(f"  OK: '{suspect}' not in features")
else:
    print("  (Feature-Liste nicht verfuegbar)")


# 2c: Feature-Werte fuer Success vs. Failure vergleichen
print("\n2c) Feature-Werte Success vs. Failure:")

# Lade alle Features mit Target
df_all = read_sql("""
    SELECT * FROM ml_features_trial
    WHERE phase_transition_success IS NOT NULL
""", conn)

# Map phases to transitions
df_all["transition"] = "unknown"
df_all.loc[df_all["current_phase"].isin(["phase1", "early_phase1", "phase1_phase2"]), "transition"] = "Phase_I_to_II"
df_all.loc[df_all["current_phase"].isin(["phase2", "phase2_phase3"]), "transition"] = "Phase_II_to_III"
df_all.loc[df_all["current_phase"] == "phase3", "transition"] = "Phase_III_to_Approval"

feat_columns = [c for c in df_all.columns if c.startswith("feat_")]
perfect_separators = []
strong_separators = []

for transition in ["Phase_I_to_II", "Phase_II_to_III", "Phase_III_to_Approval"]:
    sub = df_all[df_all["transition"] == transition]
    success = sub[sub["phase_transition_success"] == 1]
    failure = sub[sub["phase_transition_success"] == 0]

    print(f"\n  === {transition}: Success={len(success)}, Failure={len(failure)} ===")

    for col in feat_columns:
        try:
            s_mean = pd.to_numeric(success[col], errors="coerce").mean()
            f_mean = pd.to_numeric(failure[col], errors="coerce").mean()
            if pd.notna(s_mean) and pd.notna(f_mean):
                denom = max(abs(s_mean), abs(f_mean)) + 1e-10
                ratio = abs(s_mean - f_mean) / denom
                if ratio > 0.9:
                    print(f"    !! PERFECT SEPARATOR: {col} (Success mean={s_mean:.3f}, Failure mean={f_mean:.3f}, ratio={ratio:.3f})")
                    perfect_separators.append({"transition": transition, "feature": col, "s_mean": s_mean, "f_mean": f_mean, "ratio": ratio})
                elif ratio > 0.7:
                    print(f"    !  STRONG SEPARATOR: {col} (Success mean={s_mean:.3f}, Failure mean={f_mean:.3f}, ratio={ratio:.3f})")
                    strong_separators.append({"transition": transition, "feature": col, "s_mean": s_mean, "f_mean": f_mean, "ratio": ratio})
        except:
            pass

report_sections["diagnose_2"] = {
    "model_importance": model_importance,
    "leakage_found": leakage_found,
    "perfect_separators": perfect_separators,
    "strong_separators": strong_separators,
}


# ===================================================================
# DIAGNOSE 3: TEMPORAL LEAKAGE -- CV-Split-Analyse
# ===================================================================
print("\n" + "=" * 70)
print("DIAGNOSE 3: TEMPORAL LEAKAGE -- CV-Split-Analyse")
print("=" * 70)

# Lade Features mit Start-Datum
df_temporal = read_sql("""
    SELECT mf.*, t.start_date, t.completion_date
    FROM ml_features_trial mf
    JOIN trials t ON mf.trial_id = t.trial_id
    WHERE mf.phase_transition_success IS NOT NULL
    ORDER BY t.start_date
""", conn)

df_temporal["start_date"] = pd.to_datetime(df_temporal["start_date"], errors="coerce")
df_temporal["completion_date"] = pd.to_datetime(df_temporal["completion_date"], errors="coerce")

print(f"\nZeitraum: {df_temporal['start_date'].min()} bis {df_temporal['start_date'].max()}")
print(f"Median Start: {df_temporal['start_date'].median()}")
print(f"NaT start_date: {df_temporal['start_date'].isna().sum()}")

# Pruefe: Drug-Level Features - ist drug_num_total_trials temporal oder global?
print("\n3a) Temporal-Check fuer Drug-Level Features:")
temporal_checks = []
for col in ['feat_drug_num_total_trials', 'feat_drug_num_prior_approvals',
            'feat_drug_prior_approval', 'feat_drug_trial_rank',
            'feat_drug_years_since_first_trial', 'feat_sponsor_total_trials',
            'feat_sponsor_success_rate', 'feat_sponsor_drugs_approved',
            'feat_drug_class_approved_count', 'feat_moa_has_any_approval',
            'feat_competing_trials_same_phase', 'feat_competing_drugs_same_moa']:
    if col in df_temporal.columns:
        col_numeric = pd.to_numeric(df_temporal[col], errors="coerce")
        valid = df_temporal["start_date"].notna() & col_numeric.notna()
        if valid.sum() > 100:
            early_mask = df_temporal["start_date"] < df_temporal["start_date"].median()
            late_mask = df_temporal["start_date"] >= df_temporal["start_date"].median()
            early_mean = col_numeric[valid & early_mask].mean()
            late_mean = col_numeric[valid & late_mask].mean()

            # Check if same drug appears in both early and late
            print(f"  {col}:")
            print(f"    Early trials mean: {early_mean:.3f}, Late trials mean: {late_mean:.3f}")

            temporal_checks.append({
                "feature": col,
                "early_mean": float(early_mean),
                "late_mean": float(late_mean),
                "ratio": float(late_mean / early_mean) if early_mean > 0 else float('inf'),
            })

# 3b: Drug overlap between train/test
print("\n3b) Drug Overlap zwischen Train/Test (cutoff=2020-01-01):")
cutoff = pd.Timestamp("2020-01-01")
train_drugs = set(df_temporal[df_temporal["start_date"] < cutoff]["drug_id"].unique())
test_drugs = set(df_temporal[df_temporal["start_date"] >= cutoff]["drug_id"].dropna().unique())
overlap_drugs = train_drugs & test_drugs
print(f"  Train drugs: {len(train_drugs)}")
print(f"  Test drugs: {len(test_drugs)}")
print(f"  Overlap: {len(overlap_drugs)} drugs ({len(overlap_drugs)/max(len(test_drugs),1)*100:.1f}% of test drugs)")
print(f"  Test-only drugs: {len(test_drugs - train_drugs)}")

# 3c: Fuer ueberlappende Drugs - wie viel Info leckt?
if len(overlap_drugs) > 0:
    print(f"\n3c) Info-Leak fuer ueberlappende Drugs ({len(overlap_drugs)}):")
    overlap_list = list(overlap_drugs)[:10]
    for drug_id in overlap_list:
        drug_data = df_temporal[df_temporal["drug_id"] == drug_id].sort_values("start_date")
        phases = drug_data[["current_phase", "start_date", "phase_transition_success"]].to_dict("records")
        if len(phases) > 1:
            in_train = sum(1 for p in phases if pd.notna(p.get("start_date")) and pd.Timestamp(p["start_date"]) < cutoff)
            in_test = sum(1 for p in phases if pd.notna(p.get("start_date")) and pd.Timestamp(p["start_date"]) >= cutoff)
            if in_train > 0 and in_test > 0:
                print(f"  Drug {str(drug_id)[:8]}...: {in_train} trials in train, {in_test} in test")

report_sections["diagnose_3"] = {
    "temporal_checks": temporal_checks,
    "train_drugs": len(train_drugs),
    "test_drugs": len(test_drugs),
    "overlap_drugs": len(overlap_drugs),
    "overlap_pct": float(len(overlap_drugs)/max(len(test_drugs),1)*100),
}


# ===================================================================
# DIAGNOSE 4: TARGET-DEFINITION
# ===================================================================
print("\n" + "=" * 70)
print("DIAGNOSE 4: TARGET-DEFINITION")
print("=" * 70)

# 4a: Was bedeutet target=1 und target=0 genau?
d4a = read_sql("""
    SELECT
        mf.current_phase,
        mf.phase_transition_success as target,
        t.overall_status,
        COUNT(*) as cnt
    FROM ml_features_trial mf
    JOIN trials t ON mf.trial_id = t.trial_id
    WHERE mf.phase_transition_success IS NOT NULL
    GROUP BY mf.current_phase, mf.phase_transition_success, t.overall_status
    ORDER BY mf.current_phase, mf.phase_transition_success, cnt DESC
""", conn)
print("\n4a) Target Definition (was bedeutet Success/Failure):")
print(d4a.to_string(index=False))

# 4b: Wie wird success bestimmt? Basiert es auf drug_max_phase?
print("\n4b) Target-Logik Analyse:")
print("  Target-Berechnung in compute_features.py:")
print("  - drug_max_phase = df.groupby('drug_id')['feat_phase_numeric'].max()")
print("  - success = 1 WENN drug_max_phase > current_phase_numeric")
print("  - success = 1 WENN drug_id in approved_drugs UND phase == 'phase3'")
print("  - success = 0 SONST (fuer known-outcome trials)")
print("")
print("  PROBLEM: drug_max_phase nutzt ALLE Trials des Drugs (inkl. zukuenftiger Phasen)")
print("  -> Wenn Drug X Phase-1-Trials UND Phase-3-Trials hat:")
print("     Phase-1-Trial bekommt success=1 weil max_phase > 1")
print("     Gleichzeitig hat es feat_drug_num_total_trials = hohe Zahl")
print("     -> ZIRKULAER: Feature spiegelt das Label wider")

# 4c: Pruefe: Sind completed Trials immer success=1?
d4c = read_sql("""
    SELECT
        t.overall_status,
        SUM(CASE WHEN mf.phase_transition_success = 1 THEN 1 ELSE 0 END) as n_success,
        SUM(CASE WHEN mf.phase_transition_success = 0 THEN 1 ELSE 0 END) as n_failure,
        COUNT(*) as total
    FROM ml_features_trial mf
    JOIN trials t ON mf.trial_id = t.trial_id
    WHERE mf.phase_transition_success IS NOT NULL
    GROUP BY t.overall_status
    ORDER BY total DESC
""", conn)
print("\n4c) Overall Status vs. Target:")
print(d4c.to_string(index=False))

report_sections["diagnose_4"] = {
    "status_vs_target": d4c.to_dict("records"),
}


# ===================================================================
# DIAGNOSE 5: SANITY CHECK -- Triviale Baseline
# ===================================================================
print("\n" + "=" * 70)
print("DIAGNOSE 5: SANITY CHECK -- Triviale Baseline (Single-Feature AUC)")
print("=" * 70)

single_feature_aucs = []
for transition in ["Phase_I_to_II", "Phase_II_to_III", "Phase_III_to_Approval"]:
    sub = df_all[df_all["transition"] == transition].copy()
    y = sub["phase_transition_success"].astype(int).values
    n_success = y.sum()
    n_fail = len(y) - n_success

    print(f"\n  === {transition}: {n_success} Success, {n_fail} Failure ===")
    print(f"  Majority class accuracy: {max(n_success, n_fail)/len(y):.3f}")

    # Single feature AUCs
    for col in feat_columns:
        try:
            x = pd.to_numeric(sub[col], errors="coerce").values
            valid = ~np.isnan(x) & ~np.isnan(y.astype(float))
            if valid.sum() > 100 and len(np.unique(y[valid])) == 2:
                auc = roc_auc_score(y[valid], x[valid])
                # Flip if < 0.5
                auc_eff = max(auc, 1 - auc)
                if auc_eff > 0.85:
                    print(f"    !! LEAKAGE: '{col}': AUC = {auc:.3f} (effective {auc_eff:.3f})")
                    single_feature_aucs.append({"transition": transition, "feature": col, "auc": float(auc), "auc_effective": float(auc_eff)})
                elif auc_eff > 0.70:
                    print(f"    !  STRONG: '{col}': AUC = {auc:.3f} (effective {auc_eff:.3f})")
                    single_feature_aucs.append({"transition": transition, "feature": col, "auc": float(auc), "auc_effective": float(auc_eff)})
        except:
            pass

report_sections["diagnose_5"] = {
    "single_feature_aucs": single_feature_aucs,
}


# ===================================================================
# DIAGNOSE 6: ANTI-LEAK VERIFICATION
# ===================================================================
print("\n" + "=" * 70)
print("DIAGNOSE 6: ANTI-LEAK VERIFICATION")
print("=" * 70)

anti_leak_results = []

if feature_cols:
    # Check 1: Verbotene Features
    print("\n6a) Verbotene Features:")
    forbidden = ['has_results', 'overall_status', 'why_stopped', 'is_stale']
    for f in forbidden:
        matches = [col for col in feature_cols if f in col.lower()]
        if matches:
            print(f"  !! VIOLATION: '{f}' found: {matches}")
            anti_leak_results.append({"check": f"forbidden_{f}", "status": "FAIL", "matches": matches})
        else:
            print(f"  OK: '{f}' not in features")
            anti_leak_results.append({"check": f"forbidden_{f}", "status": "PASS"})

    # Check 2: Approval-related features
    print("\n6b) Approval-Related Features:")
    approval_suspects = ['approval', 'approved', 'first_approval', 'fda_approval']
    for f in approval_suspects:
        matches = [col for col in feature_cols if f in col.lower()]
        if matches:
            print(f"  !! APPROVAL FEATURES: {matches}")
            anti_leak_results.append({"check": f"approval_{f}", "status": "WARNING", "matches": matches})
        else:
            print(f"  OK: '{f}' not directly in features")

    # Check 3: highest_phase features
    print("\n6c) Highest-Phase Features:")
    hp_features = [col for col in feature_cols if 'highest_phase' in col.lower() or 'max_phase' in col.lower()]
    if hp_features:
        print(f"  !! HIGHEST_PHASE FEATURES: {hp_features}")
        print(f"     If this reflects CURRENT state (not at trial start), it leaks!")
        anti_leak_results.append({"check": "highest_phase", "status": "WARNING", "matches": hp_features})

    # Check 4: feat_drug_num_total_trials -- should be feat_drug_num_PRIOR_trials
    print("\n6d) Temporal-Awareness Check:")
    temporal_suspects = {
        'feat_drug_num_total_trials': 'Should be feat_drug_num_PRIOR_trials (before trial start)',
        'feat_sponsor_total_trials': 'Should count only trials BEFORE current trial start',
        'feat_competing_trials_same_phase': 'Counts ALL trials, not just prior ones',
        'feat_competing_drugs_same_moa': 'Counts ALL drugs, not just prior ones',
        'feat_drug_class_approved_count': 'Should count only approvals BEFORE trial start',
        'feat_moa_has_any_approval': 'Should check only approvals BEFORE trial start',
    }
    for feat, issue in temporal_suspects.items():
        if feat in feature_cols:
            print(f"  !! TEMPORAL: '{feat}' present")
            print(f"     Issue: {issue}")
            anti_leak_results.append({"check": f"temporal_{feat}", "status": "FAIL", "issue": issue})

    # Complete feature list
    print(f"\n6e) Vollstaendige Feature-Liste ({len(feature_cols)} Features):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:3d}. {col}")

report_sections["diagnose_6"] = {
    "anti_leak_results": anti_leak_results,
    "total_features": len(feature_cols) if feature_cols else 0,
}


# ===================================================================
# GENERATE REPORT
# ===================================================================
print("\n" + "=" * 70)
print("GENERATING DIAGNOSTIC REPORT")
print("=" * 70)

# Determine severity
critical_issues = []
high_issues = []
medium_issues = []

# From Diagnose 1
for item in d1a_summary:
    pct = float(item["Success%"].replace("%",""))
    if pct > 85:
        critical_issues.append(f"Class Balance: {item['Transition']} has {item['Success%']} success rate (>85%)")

# From Diagnose 2
for sep in perfect_separators:
    critical_issues.append(f"Perfect Separator: {sep['feature']} in {sep['transition']} (ratio={sep['ratio']:.3f})")
for sep in strong_separators:
    high_issues.append(f"Strong Separator: {sep['feature']} in {sep['transition']} (ratio={sep['ratio']:.3f})")
if leakage_found:
    for lf in leakage_found:
        high_issues.append(f"Leakage Suspect in features: {lf}")

# From Diagnose 3
if report_sections["diagnose_3"]["overlap_pct"] > 50:
    high_issues.append(f"Drug overlap train/test: {report_sections['diagnose_3']['overlap_pct']:.1f}% of test drugs also in train")

# From Diagnose 5
for sfa in single_feature_aucs:
    if sfa["auc_effective"] > 0.95:
        critical_issues.append(f"Single-Feature Leakage: {sfa['feature']} AUC={sfa['auc_effective']:.3f} in {sfa['transition']}")
    elif sfa["auc_effective"] > 0.85:
        high_issues.append(f"Single-Feature High AUC: {sfa['feature']} AUC={sfa['auc_effective']:.3f} in {sfa['transition']}")

# From Diagnose 6
for alr in anti_leak_results:
    if alr["status"] == "FAIL":
        if "forbidden" in alr["check"]:
            critical_issues.append(f"Forbidden feature in model: {alr.get('matches', alr['check'])}")
        elif "temporal" in alr["check"]:
            high_issues.append(f"Non-temporal feature: {alr.get('issue', alr['check'])}")

# Build report
report_lines = []
report_lines.append("# Phase 4 Diagnostic Report: Model Performance Plausibility Check")
report_lines.append(f"\n**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Feature Version:** v1.0")
report_lines.append(f"**Model Version:** v1.0")
report_lines.append("")

# Executive Summary
report_lines.append("## 1. Executive Summary")
report_lines.append("")
if critical_issues:
    report_lines.append(f"**VERDICT: MODELL-PERFORMANCE IST NICHT VALIDE.** {len(critical_issues)} kritische und {len(high_issues)} hohe Probleme gefunden.")
    report_lines.append("Die AUC-Werte von 0.96-0.999 sind Artefakte von Data Leakage und methodischen Problemen.")
else:
    report_lines.append("Keine kritischen Probleme gefunden.")
report_lines.append("")

# Target Distribution
report_lines.append("## 2. Target-Verteilung (Diagnose 1)")
report_lines.append("")
report_lines.append("| Transition | Success | Failure | Total | Success% | Hay 2014 Benchmark |")
report_lines.append("|---|---|---|---|---|---|")
benchmarks = {"Phase_I_to_II": "60-65%", "Phase_II_to_III": "25-35%", "Phase_III_to_Approval": "50-60%"}
for item in d1a_summary:
    bm = benchmarks.get(item["Transition"], "n/a")
    report_lines.append(f"| {item['Transition']} | {item['Success']} | {item['Failure']} | {item['Total']} | {item['Success%']} | {bm} |")
report_lines.append("")

# Drug sources
report_lines.append("### Drug-Quellen:")
for _, row in d1b.iterrows():
    report_lines.append(f"- **{row['drug_source']}**: {row['drug_count']} Drugs, {row['trial_count']} Trials")
report_lines.append("")

# Leakage Findings
report_lines.append("## 3. Leakage-Findings (Diagnose 2)")
report_lines.append("")

if perfect_separators:
    report_lines.append("### PERFECT SEPARATORS (ratio > 0.9) -- KRITISCH:")
    report_lines.append("")
    report_lines.append("| Transition | Feature | Success Mean | Failure Mean | Ratio |")
    report_lines.append("|---|---|---|---|---|")
    for sep in sorted(perfect_separators, key=lambda x: -x["ratio"]):
        report_lines.append(f"| {sep['transition']} | `{sep['feature']}` | {sep['s_mean']:.3f} | {sep['f_mean']:.3f} | {sep['ratio']:.3f} |")
    report_lines.append("")

if strong_separators:
    report_lines.append("### STRONG SEPARATORS (ratio > 0.7) -- HOCH:")
    report_lines.append("")
    report_lines.append("| Transition | Feature | Success Mean | Failure Mean | Ratio |")
    report_lines.append("|---|---|---|---|---|")
    for sep in sorted(strong_separators, key=lambda x: -x["ratio"])[:30]:
        report_lines.append(f"| {sep['transition']} | `{sep['feature']}` | {sep['s_mean']:.3f} | {sep['f_mean']:.3f} | {sep['ratio']:.3f} |")
    report_lines.append("")

if leakage_found:
    report_lines.append(f"### Leakage-Verdacht in Feature-Liste: {leakage_found}")
    report_lines.append("")

# Feature Importance
report_lines.append("## 4. Feature Importance (Diagnose 2a)")
report_lines.append("")
for transition, feats in model_importance.items():
    report_lines.append(f"### {transition} (XGBoost A - Top 10)")
    report_lines.append("")
    report_lines.append("| Rank | Feature | Importance |")
    report_lines.append("|---|---|---|")
    for f in feats[:10]:
        report_lines.append(f"| {f['rank']} | `{f['feature']}` | {f['importance']:.4f} |")
    report_lines.append("")

# Temporal Analysis
report_lines.append("## 5. Temporal Analysis (Diagnose 3)")
report_lines.append("")
report_lines.append(f"- **Train Drugs:** {report_sections['diagnose_3']['train_drugs']}")
report_lines.append(f"- **Test Drugs:** {report_sections['diagnose_3']['test_drugs']}")
report_lines.append(f"- **Overlap:** {report_sections['diagnose_3']['overlap_drugs']} Drugs ({report_sections['diagnose_3']['overlap_pct']:.1f}% of test)")
report_lines.append("")
if temporal_checks:
    report_lines.append("### Feature-Werte Early vs. Late Trials:")
    report_lines.append("")
    report_lines.append("| Feature | Early Mean | Late Mean | Ratio Late/Early |")
    report_lines.append("|---|---|---|---|")
    for tc in temporal_checks:
        report_lines.append(f"| `{tc['feature']}` | {tc['early_mean']:.3f} | {tc['late_mean']:.3f} | {tc['ratio']:.2f}x |")
    report_lines.append("")

# Baseline Comparison
report_lines.append("## 6. Single-Feature Baselines (Diagnose 5)")
report_lines.append("")
if single_feature_aucs:
    report_lines.append("| Transition | Feature | AUC | Effective AUC | Severity |")
    report_lines.append("|---|---|---|---|---|")
    for sfa in sorted(single_feature_aucs, key=lambda x: -x["auc_effective"]):
        sev = "CRITICAL" if sfa["auc_effective"] > 0.95 else ("HIGH" if sfa["auc_effective"] > 0.85 else "MEDIUM")
        report_lines.append(f"| {sfa['transition']} | `{sfa['feature']}` | {sfa['auc']:.3f} | {sfa['auc_effective']:.3f} | {sev} |")
    report_lines.append("")
else:
    report_lines.append("Keine Single-Feature mit AUC > 0.70 gefunden.\n")

# Anti-Leak Verification
report_lines.append("## 7. Anti-Leak Verification (Diagnose 6)")
report_lines.append("")
for alr in anti_leak_results:
    status_icon = "FAIL" if alr["status"] == "FAIL" else ("WARN" if alr["status"] == "WARNING" else "PASS")
    detail = alr.get("matches", alr.get("issue", ""))
    report_lines.append(f"- [{status_icon}] {alr['check']}: {detail}")
report_lines.append("")

# Target Definition
report_lines.append("## 8. Target-Definition (Diagnose 4)")
report_lines.append("")
report_lines.append("### Overall Status vs. Target:")
report_lines.append("")
report_lines.append("| Status | Success | Failure | Total |")
report_lines.append("|---|---|---|---|")
for _, row in d4c.iterrows():
    report_lines.append(f"| {row['overall_status']} | {row['n_success']} | {row['n_failure']} | {row['total']} |")
report_lines.append("")
report_lines.append("### Target-Logik:")
report_lines.append("- `success = 1` wenn Drug IRGENDWANN eine hoehere Phase erreicht (`drug_max_phase > current_phase`)")
report_lines.append("- `success = 1` wenn Drug in `approved_drugs` und Phase = phase3")
report_lines.append("- `success = 0` sonst (fuer Trials mit known outcome status)")
report_lines.append("- **PROBLEM**: Drug-Level Target, nicht Trial-Level -> zirkulaer mit Drug-Level Features")
report_lines.append("")

# Bewertungsmatrix
report_lines.append("## 9. Bewertungsmatrix")
report_lines.append("")
report_lines.append("| Diagnose | Ergebnis | Severity | Fix |")
report_lines.append("|---|---|---|---|")

# Class Balance
max_success = max(float(item["Success%"].replace("%","")) for item in d1a_summary)
if max_success > 85:
    report_lines.append(f"| Class Balance | {max_success:.0f}% Success -> FAIL | CRITICAL | Nachtrag erweitern |")
else:
    report_lines.append(f"| Class Balance | Max {max_success:.0f}% Success -> OK | PASS | - |")

# Direct Leakage
if leakage_found:
    report_lines.append(f"| Direct Leakage | {len(leakage_found)} forbidden features -> FAIL | CRITICAL | Features entfernen |")
else:
    report_lines.append("| Direct Leakage | Keine verbotenen Features | PASS | - |")

# Indirect Leakage
n_critical_sf = sum(1 for s in single_feature_aucs if s["auc_effective"] > 0.95)
n_high_sf = sum(1 for s in single_feature_aucs if s["auc_effective"] > 0.85)
if n_critical_sf > 0:
    report_lines.append(f"| Indirect Leakage | {n_critical_sf} features AUC>0.95 -> FAIL | CRITICAL | Features entfernen/temporalisieren |")
elif n_high_sf > 0:
    report_lines.append(f"| Indirect Leakage | {n_high_sf} features AUC>0.85 -> WARN | HIGH | Untersuchen |")
else:
    report_lines.append("| Indirect Leakage | Keine Single-Feature AUC>0.85 | PASS | - |")

# Temporal Leak
temporal_fails = sum(1 for a in anti_leak_results if a["status"] == "FAIL" and "temporal" in a["check"])
if temporal_fails > 0:
    report_lines.append(f"| Temporal Leak | {temporal_fails} non-temporal features -> FAIL | HIGH | Point-in-time Features |")
else:
    report_lines.append("| Temporal Leak | Alle Features temporal | PASS | - |")

# Drug Overlap
if report_sections["diagnose_3"]["overlap_pct"] > 50:
    report_lines.append(f"| Drug Overlap | {report_sections['diagnose_3']['overlap_pct']:.0f}% overlap -> FAIL | HIGH | Drug-Level Splits |")
else:
    report_lines.append(f"| Drug Overlap | {report_sections['diagnose_3']['overlap_pct']:.0f}% overlap | OK | - |")

# Target Definition
report_lines.append("| Target Definition | Drug-Level (zirkulaer) -> FAIL | HIGH | Trial-Level Target |")
report_lines.append("")

# Empfehlung
report_lines.append("## 10. Empfehlung: Konkreter Fix-Plan")
report_lines.append("")
report_lines.append("### Phase 1: Leakage entfernen (compute_features.py)")
report_lines.append("1. `feat_drug_num_total_trials` -> `feat_drug_num_prior_trials` (nur Trials VOR start_date)")
report_lines.append("2. `feat_was_previously_suspended` pruefen/entfernen (leakt Trial-Outcome)")
report_lines.append("3. `feat_drug_prior_approval` temporal sicherstellen (nur Approvals VOR start_date)")
report_lines.append("4. `feat_drug_class_approved_count` temporal filtern")
report_lines.append("5. `feat_moa_has_any_approval` temporal filtern")
report_lines.append("6. `feat_competing_*` Features temporal filtern")
report_lines.append("7. Post-Market Features (FAERS, UK Rx, US Spending) komplett entfernen oder als separates Set")
report_lines.append("")
report_lines.append("### Phase 2: Target-Definition ueberarbeiten (compute_features.py)")
report_lines.append("1. Statt drug_max_phase: Trial-Level Outcome basierend auf overall_status")
report_lines.append("   - success = 1 wenn Trial completed UND Drug spaeter hoehere Phase hat")
report_lines.append("   - success = 0 wenn Trial terminated/withdrawn")
report_lines.append("   - Temporal: Nur zukuenftige Phasen zaehlen die NACH Trial-Completion begonnen haben")
report_lines.append("")
report_lines.append("### Phase 3: Drug-Level Splits (train_models.py)")
report_lines.append("1. GroupKFold mit drug_id als Group-Variable")
report_lines.append("2. Kein Drug gleichzeitig in Train und Test")
report_lines.append("3. Alternativ: Drug-stratified temporal split")
report_lines.append("")
report_lines.append("### Erwartete realistische AUC nach Fixes:")
report_lines.append("- Phase I->II: 0.60-0.70")
report_lines.append("- Phase II->III: 0.65-0.75")
report_lines.append("- Phase III->Approval: 0.70-0.80")
report_lines.append("- (Im Einklang mit Literatur fuer oeffentliche Daten)")

report_text = "\n".join(report_lines)

# Write report
report_path = f"{ARTIFACT_DIR}/phase4_diagnostic_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport gespeichert: {report_path}")

conn.close()
print("\nDone.")
