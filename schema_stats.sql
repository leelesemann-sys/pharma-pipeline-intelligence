-- ============================================
-- PHARMA PIPELINE INTELLIGENCE - Schema Stats
-- Run against pharma_pipeline_db
-- ============================================

-- Overall counts
SELECT 'trials' AS table_name, COUNT(*) AS row_count FROM trials
UNION ALL
SELECT 'drugs', COUNT(*) FROM drugs
UNION ALL
SELECT 'drug_trials', COUNT(*) FROM drug_trials
UNION ALL
SELECT 'indications', COUNT(*) FROM indications
UNION ALL
SELECT 'approvals', COUNT(*) FROM approvals
UNION ALL
SELECT 'companies', COUNT(*) FROM companies
UNION ALL
SELECT 'patents', COUNT(*) FROM patents
UNION ALL
SELECT 'market_data', COUNT(*) FROM market_data
UNION ALL
SELECT 'predictions', COUNT(*) FROM predictions
UNION ALL
SELECT 'nlp_signals', COUNT(*) FROM nlp_signals
UNION ALL
SELECT 'alerts', COUNT(*) FROM alerts
ORDER BY row_count DESC;

-- Trials by status
SELECT overall_status, COUNT(*) AS cnt
FROM trials
GROUP BY overall_status
ORDER BY cnt DESC;

-- Trials by phase
SELECT phase, COUNT(*) AS cnt
FROM trials
GROUP BY phase
ORDER BY cnt DESC;

-- Trials by study type
SELECT study_type, COUNT(*) AS cnt
FROM trials
GROUP BY study_type
ORDER BY cnt DESC;

-- Top 20 sponsors
SELECT TOP 20 lead_sponsor_name, COUNT(*) AS trial_count
FROM trials
WHERE lead_sponsor_name IS NOT NULL AND lead_sponsor_name != ''
GROUP BY lead_sponsor_name
ORDER BY trial_count DESC;

-- Drugs by modality
SELECT modality, COUNT(*) AS cnt
FROM drugs
GROUP BY modality
ORDER BY cnt DESC;

-- Drugs by MoA class
SELECT moa_class, COUNT(*) AS cnt
FROM drugs
WHERE moa_class IS NOT NULL
GROUP BY moa_class
ORDER BY cnt DESC;

-- Top 10 drugs by trial linkage
SELECT TOP 10 d.inn, COUNT(dt.trial_id) AS linked_trials
FROM drugs d
JOIN drug_trials dt ON d.drug_id = dt.drug_id
GROUP BY d.inn
ORDER BY linked_trials DESC;

-- Approvals by drug
SELECT d.inn, COUNT(a.approval_id) AS approval_count
FROM drugs d
LEFT JOIN approvals a ON d.drug_id = a.drug_id
GROUP BY d.inn
ORDER BY approval_count DESC;

-- Stale trials
SELECT COUNT(*) AS stale_trials
FROM trials
WHERE is_stale = 1;

-- Trials with results
SELECT COUNT(*) AS trials_with_results
FROM trials
WHERE has_results = 1;

-- Date range
SELECT
    MIN(start_date) AS oldest_trial,
    MAX(start_date) AS newest_trial,
    MIN(last_update_date) AS oldest_update,
    MAX(last_update_date) AS latest_update
FROM trials
WHERE start_date IS NOT NULL;
