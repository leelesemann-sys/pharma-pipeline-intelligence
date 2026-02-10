"""
SQL query functions for Pharma Pipeline Dashboard.
All queries are centralized here for maintainability.
Each function returns a pandas DataFrame via the cached run_query utility.
"""

from utils.db import run_query


# ─────────────────────────────────────────────
# KPI Queries
# ─────────────────────────────────────────────

def get_kpis():
    """Top-level KPI metrics for the dashboard header."""
    return run_query("""
        SELECT
            (SELECT COUNT(*) FROM drugs) AS total_drugs,
            (SELECT COUNT(*) FROM trials) AS total_trials,
            (SELECT COUNT(DISTINCT moa_class) FROM drugs WHERE moa_class IS NOT NULL) AS moa_classes,
            (SELECT COUNT(*) FROM indications) AS total_indications,
            (SELECT COUNT(*) FROM companies) AS total_companies,
            (SELECT COUNT(*) FROM trials WHERE overall_status IN ('recruiting', 'active_not_recruiting', 'enrolling_by_invitation')) AS active_trials,
            (SELECT COUNT(*) FROM trials WHERE overall_status = 'terminated') AS terminated_trials,
            (SELECT COUNT(*) FROM trials WHERE is_stale = 1) AS stale_trials,
            (SELECT COUNT(*) FROM trials WHERE has_results = 1) AS trials_with_results
    """)


# ─────────────────────────────────────────────
# Tab 1: Pipeline Overview
# ─────────────────────────────────────────────

def get_pipeline_overview():
    """All drugs with aggregated info for the pipeline table."""
    return run_query("""
        SELECT
            d.drug_id,
            d.inn AS drug_name,
            d.moa_class,
            d.modality,
            -- Derive highest phase from trials (human-readable format)
            (SELECT TOP 1
                CASE tp.phase
                    WHEN 'phase4' THEN 'Phase 4'
                    WHEN 'phase3' THEN 'Phase 3'
                    WHEN 'phase2_phase3' THEN 'Phase 2/3'
                    WHEN 'phase2' THEN 'Phase 2'
                    WHEN 'phase1_phase2' THEN 'Phase 1/2'
                    WHEN 'phase1' THEN 'Phase 1'
                    WHEN 'early_phase1' THEN 'Early Phase 1'
                    ELSE tp.phase
                END
             FROM drug_trials dtp
             JOIN trials tp ON dtp.trial_id = tp.trial_id
             WHERE dtp.drug_id = d.drug_id AND tp.phase IS NOT NULL AND tp.phase != 'na'
             ORDER BY CASE tp.phase
                 WHEN 'phase4' THEN 7 WHEN 'phase3' THEN 6
                 WHEN 'phase2_phase3' THEN 5 WHEN 'phase2' THEN 4
                 WHEN 'phase1_phase2' THEN 3 WHEN 'phase1' THEN 2
                 WHEN 'early_phase1' THEN 1 ELSE 0
             END DESC
            ) AS highest_phase,
            -- Top sponsor from trials (most frequent lead_sponsor)
            (SELECT TOP 1 t2.lead_sponsor_name
             FROM drug_trials dt2
             JOIN trials t2 ON dt2.trial_id = t2.trial_id
             WHERE dt2.drug_id = d.drug_id
             GROUP BY t2.lead_sponsor_name
             ORDER BY COUNT(*) DESC
            ) AS top_sponsor,
            -- Dominant sponsor type from trials
            (SELECT TOP 1 c2.company_type
             FROM drug_trials dt3
             JOIN trials t3 ON dt3.trial_id = t3.trial_id
             LEFT JOIN companies c2 ON t3.lead_sponsor_name = c2.name
             WHERE dt3.drug_id = d.drug_id AND c2.company_type IS NOT NULL
             GROUP BY c2.company_type
             ORDER BY COUNT(*) DESC
            ) AS company_type,
            -- Indications via subquery to avoid STRING_AGG DISTINCT issues
            (SELECT STRING_AGG(i2.name, ', ')
             FROM (SELECT DISTINCT i1.name
                   FROM drug_indications di1
                   JOIN indications i1 ON di1.indication_id = i1.indication_id
                   WHERE di1.drug_id = d.drug_id) i2
            ) AS indications,
            (SELECT MAX(di3.phase) FROM drug_indications di3 WHERE di3.drug_id = d.drug_id) AS max_investigation_phase,
            (SELECT STRING_AGG(i4.name, ', ')
             FROM (SELECT DISTINCT i3.name
                   FROM drug_indications di4
                   JOIN indications i3 ON di4.indication_id = i3.indication_id
                   WHERE di4.drug_id = d.drug_id AND di4.status = 'approved') i4
            ) AS approved_indications,
            COUNT(DISTINCT dt.trial_id) AS total_trials,
            COUNT(DISTINCT CASE WHEN t.overall_status IN ('recruiting', 'active_not_recruiting', 'enrolling_by_invitation') THEN dt.trial_id END) AS active_trials,
            COUNT(DISTINCT CASE WHEN t.phase IN ('phase3', 'phase2_phase3') THEN dt.trial_id END) AS phase3_trials,
            CASE WHEN EXISTS (SELECT 1 FROM approvals a WHERE a.drug_id = d.drug_id) THEN 1 ELSE 0 END AS has_fda_approval
        FROM drugs d
        LEFT JOIN drug_trials dt ON d.drug_id = dt.drug_id
        LEFT JOIN trials t ON dt.trial_id = t.trial_id
        GROUP BY d.drug_id, d.inn, d.moa_class, d.modality
        ORDER BY
            MAX(CASE t.phase
                WHEN 'phase4' THEN 7 WHEN 'phase3' THEN 6
                WHEN 'phase2_phase3' THEN 5 WHEN 'phase2' THEN 4
                WHEN 'phase1_phase2' THEN 3 WHEN 'phase1' THEN 2
                WHEN 'early_phase1' THEN 1 ELSE 0
            END) DESC,
            COUNT(DISTINCT CASE WHEN t.overall_status IN ('recruiting', 'active_not_recruiting', 'enrolling_by_invitation') THEN dt.trial_id END) DESC
    """)


def get_drugs_by_phase():
    """Drug count by highest phase for the pipeline bar chart."""
    return run_query("""
        SELECT phase, COUNT(*) AS drug_count
        FROM (
            SELECT d.drug_id,
                COALESCE(
                    (SELECT TOP 1
                        CASE tp.phase
                            WHEN 'phase4' THEN 'Phase 4'
                            WHEN 'phase3' THEN 'Phase 3'
                            WHEN 'phase2_phase3' THEN 'Phase 2/3'
                            WHEN 'phase2' THEN 'Phase 2'
                            WHEN 'phase1_phase2' THEN 'Phase 1/2'
                            WHEN 'phase1' THEN 'Phase 1'
                            WHEN 'early_phase1' THEN 'Early Phase 1'
                            ELSE tp.phase
                        END
                     FROM drug_trials dtp
                     JOIN trials tp ON dtp.trial_id = tp.trial_id
                     WHERE dtp.drug_id = d.drug_id AND tp.phase IS NOT NULL AND tp.phase != 'na'
                     ORDER BY CASE tp.phase
                         WHEN 'phase4' THEN 7 WHEN 'phase3' THEN 6
                         WHEN 'phase2_phase3' THEN 5 WHEN 'phase2' THEN 4
                         WHEN 'phase1_phase2' THEN 3 WHEN 'phase1' THEN 2
                         WHEN 'early_phase1' THEN 1 ELSE 0
                     END DESC
                    ), 'Unknown'
                ) AS phase
            FROM drugs d
        ) sub
        GROUP BY phase
        ORDER BY
            CASE phase
                WHEN 'Early Phase 1' THEN 1
                WHEN 'Phase 1' THEN 2
                WHEN 'Phase 1/2' THEN 3
                WHEN 'Phase 2' THEN 4
                WHEN 'Phase 2/3' THEN 5
                WHEN 'Phase 3' THEN 6
                WHEN 'Phase 4' THEN 7
                ELSE 0
            END
    """)


def get_filter_options():
    """Get unique values for all filter dropdowns."""
    moa = run_query("SELECT DISTINCT moa_class FROM drugs WHERE moa_class IS NOT NULL ORDER BY moa_class")
    indications = run_query("SELECT DISTINCT name FROM indications ORDER BY name")
    phases = run_query("""
        SELECT phase_label AS phase FROM (
            SELECT DISTINCT
                CASE phase
                    WHEN 'early_phase1' THEN 'Early Phase 1'
                    WHEN 'phase1' THEN 'Phase 1'
                    WHEN 'phase1_phase2' THEN 'Phase 1/2'
                    WHEN 'phase2' THEN 'Phase 2'
                    WHEN 'phase2_phase3' THEN 'Phase 2/3'
                    WHEN 'phase3' THEN 'Phase 3'
                    WHEN 'phase4' THEN 'Phase 4'
                END AS phase_label,
                CASE phase
                    WHEN 'early_phase1' THEN 0
                    WHEN 'phase1' THEN 1
                    WHEN 'phase1_phase2' THEN 2
                    WHEN 'phase2' THEN 3
                    WHEN 'phase2_phase3' THEN 4
                    WHEN 'phase3' THEN 5
                    WHEN 'phase4' THEN 6
                END AS phase_order
            FROM trials
            WHERE phase IS NOT NULL AND phase != 'na'
        ) sub
        WHERE phase_label IS NOT NULL
        ORDER BY phase_order
    """)
    sponsor_types = run_query("SELECT DISTINCT company_type FROM companies WHERE company_type IS NOT NULL ORDER BY company_type")
    return {
        "moa_classes": moa["moa_class"].tolist() if not moa.empty else [],
        "indications": indications["name"].tolist() if not indications.empty else [],
        "phases": phases["phase"].tolist() if not phases.empty else [],
        "sponsor_types": sponsor_types["company_type"].tolist() if not sponsor_types.empty else [],
    }


# ─────────────────────────────────────────────
# Tab 2: Competitive Landscape
# ─────────────────────────────────────────────

def get_heatmap_data(phase_filter=None):
    """MoA x Indication heatmap data with trial counts."""
    phase_clause = ""
    if phase_filter:
        placeholders = ", ".join(["?" for _ in phase_filter])
        phase_clause = f"AND t.phase IN ({placeholders})"

    query = f"""
        SELECT
            d.moa_class,
            i.name AS indication,
            COUNT(DISTINCT t.trial_id) AS trial_count,
            COUNT(DISTINCT d.drug_id) AS drug_count,
            COUNT(DISTINCT CASE WHEN t.overall_status IN ('recruiting', 'active_not_recruiting') THEN t.trial_id END) AS active_trial_count,
            COUNT(DISTINCT CASE WHEN t.start_date >= DATEADD(year, -2, GETDATE()) THEN t.trial_id END) AS recent_trial_starts
        FROM drugs d
        JOIN drug_trials dt ON d.drug_id = dt.drug_id
        JOIN trials t ON dt.trial_id = t.trial_id
        JOIN trial_indications ti ON t.trial_id = ti.trial_id
        JOIN indications i ON ti.indication_id = i.indication_id
        WHERE t.study_type = 'INTERVENTIONAL'
        {phase_clause}
        GROUP BY d.moa_class, i.name
    """
    if phase_filter:
        return run_query(query, params=tuple(phase_filter))
    return run_query(query)


def get_trial_starts_trend():
    """Trial starts per year by MoA class for the trend chart."""
    return run_query("""
        SELECT
            d.moa_class,
            YEAR(t.start_date) AS start_year,
            COUNT(DISTINCT t.trial_id) AS trial_starts
        FROM drugs d
        JOIN drug_trials dt ON d.drug_id = dt.drug_id
        JOIN trials t ON dt.trial_id = t.trial_id
        WHERE t.start_date IS NOT NULL
        AND YEAR(t.start_date) >= 2010
        AND t.study_type = 'INTERVENTIONAL'
        GROUP BY d.moa_class, YEAR(t.start_date)
        ORDER BY start_year
    """)


def get_hot_cold_zones():
    """Hot zones (growing) and cold zones (declining) MoA x Indication combos."""
    return run_query("""
        SELECT
            d.moa_class,
            i.name AS indication,
            COUNT(DISTINCT CASE WHEN t.start_date >= DATEADD(year, -2, GETDATE()) THEN t.trial_id END) AS recent_2y,
            COUNT(DISTINCT CASE WHEN t.start_date BETWEEN DATEADD(year, -4, GETDATE()) AND DATEADD(year, -2, GETDATE()) THEN t.trial_id END) AS prior_2y,
            CAST(COUNT(DISTINCT CASE WHEN t.start_date >= DATEADD(year, -2, GETDATE()) THEN t.trial_id END) AS FLOAT) /
            NULLIF(COUNT(DISTINCT CASE WHEN t.start_date BETWEEN DATEADD(year, -4, GETDATE()) AND DATEADD(year, -2, GETDATE()) THEN t.trial_id END), 0) AS growth_ratio
        FROM drugs d
        JOIN drug_trials dt ON d.drug_id = dt.drug_id
        JOIN trials t ON dt.trial_id = t.trial_id
        JOIN trial_indications ti ON t.trial_id = ti.trial_id
        JOIN indications i ON ti.indication_id = i.indication_id
        WHERE t.study_type = 'INTERVENTIONAL'
        GROUP BY d.moa_class, i.name
        HAVING COUNT(DISTINCT CASE WHEN t.start_date >= DATEADD(year, -2, GETDATE()) THEN t.trial_id END) >= 3
        ORDER BY growth_ratio DESC
    """)


# ─────────────────────────────────────────────
# Tab 3: Trial Analytics
# ─────────────────────────────────────────────

def get_trial_status_breakdown(indication_filter=None, phase_filter=None, sponsor_type_filter=None):
    """Trial count by status for the donut chart."""
    conditions = ["1=1"]
    if indication_filter:
        ind_list = ", ".join([f"'{i}'" for i in indication_filter])
        conditions.append(f"i.name IN ({ind_list})")
    if phase_filter:
        ph_list = ", ".join([f"'{p}'" for p in phase_filter])
        conditions.append(f"t.phase IN ({ph_list})")
    if sponsor_type_filter:
        sp_list = ", ".join([f"'{s}'" for s in sponsor_type_filter])
        conditions.append(f"c.company_type IN ({sp_list})")

    where = " AND ".join(conditions)

    has_joins = indication_filter or sponsor_type_filter
    join_clause = ""
    if indication_filter:
        join_clause += " JOIN trial_indications ti ON t.trial_id = ti.trial_id JOIN indications i ON ti.indication_id = i.indication_id"
    if sponsor_type_filter:
        join_clause += " LEFT JOIN companies c ON t.lead_sponsor_name = c.name"

    return run_query(f"""
        SELECT
            t.overall_status,
            COUNT(DISTINCT t.trial_id) AS trial_count
        FROM trials t
        {join_clause}
        WHERE {where}
        GROUP BY t.overall_status
        ORDER BY trial_count DESC
    """)


def get_trial_phase_breakdown():
    """Trial count by phase for the donut chart."""
    return run_query("""
        SELECT
            COALESCE(phase, 'N/A') AS phase,
            COUNT(*) AS trial_count
        FROM trials
        GROUP BY phase
        ORDER BY trial_count DESC
    """)


def get_trial_starts_over_time():
    """Trial starts over time stacked by phase."""
    return run_query("""
        SELECT
            YEAR(t.start_date) AS start_year,
            COALESCE(t.phase, 'N/A') AS phase,
            COUNT(*) AS trial_count
        FROM trials t
        WHERE t.start_date IS NOT NULL
        AND YEAR(t.start_date) BETWEEN 2000 AND 2026
        AND t.study_type = 'INTERVENTIONAL'
        GROUP BY YEAR(t.start_date), t.phase
        ORDER BY start_year, t.phase
    """)


def get_top_sponsors(limit=20):
    """Top sponsors by trial count."""
    return run_query(f"""
        SELECT TOP {limit}
            t.lead_sponsor_name,
            c.company_type,
            COUNT(*) AS trial_count,
            COUNT(CASE WHEN t.overall_status IN ('recruiting', 'active_not_recruiting') THEN 1 END) AS active_trials
        FROM trials t
        LEFT JOIN companies c ON t.lead_sponsor_name = c.name
        GROUP BY t.lead_sponsor_name, c.company_type
        ORDER BY trial_count DESC
    """)


def get_termination_rate_by_moa():
    """Termination rate by MoA class."""
    return run_query("""
        SELECT
            d.moa_class,
            COUNT(DISTINCT t.trial_id) AS total_trials,
            COUNT(DISTINCT CASE WHEN t.overall_status = 'terminated' THEN t.trial_id END) AS terminated,
            CAST(COUNT(DISTINCT CASE WHEN t.overall_status = 'terminated' THEN t.trial_id END) AS FLOAT) /
            NULLIF(COUNT(DISTINCT t.trial_id), 0) * 100 AS termination_rate_pct
        FROM drugs d
        JOIN drug_trials dt ON d.drug_id = dt.drug_id
        JOIN trials t ON dt.trial_id = t.trial_id
        WHERE t.study_type = 'INTERVENTIONAL'
        AND t.overall_status IN ('completed', 'terminated')
        GROUP BY d.moa_class
        ORDER BY termination_rate_pct DESC
    """)


def get_termination_rate_by_indication():
    """Termination rate by indication."""
    return run_query("""
        SELECT
            i.name AS indication,
            COUNT(DISTINCT t.trial_id) AS total_trials,
            COUNT(DISTINCT CASE WHEN t.overall_status = 'terminated' THEN t.trial_id END) AS terminated,
            CAST(COUNT(DISTINCT CASE WHEN t.overall_status = 'terminated' THEN t.trial_id END) AS FLOAT) /
            NULLIF(COUNT(DISTINCT t.trial_id), 0) * 100 AS termination_rate_pct
        FROM trials t
        JOIN trial_indications ti ON t.trial_id = ti.trial_id
        JOIN indications i ON ti.indication_id = i.indication_id
        WHERE t.study_type = 'INTERVENTIONAL'
        AND t.overall_status IN ('completed', 'terminated')
        GROUP BY i.name
        ORDER BY termination_rate_pct DESC
    """)


def get_termination_rate_by_phase():
    """Termination rate by trial phase."""
    return run_query("""
        SELECT
            COALESCE(t.phase, 'N/A') AS phase,
            COUNT(*) AS total_trials,
            COUNT(CASE WHEN t.overall_status = 'terminated' THEN 1 END) AS terminated,
            CAST(COUNT(CASE WHEN t.overall_status = 'terminated' THEN 1 END) AS FLOAT) /
            NULLIF(COUNT(*), 0) * 100 AS termination_rate_pct
        FROM trials t
        WHERE t.study_type = 'INTERVENTIONAL'
        AND t.overall_status IN ('completed', 'terminated')
        GROUP BY t.phase
        ORDER BY termination_rate_pct DESC
    """)


def get_stale_trials():
    """Stale trials that haven't been updated in >12 months and past completion date."""
    return run_query("""
        SELECT
            t.nct_id,
            t.title,
            t.phase,
            t.overall_status,
            t.lead_sponsor_name,
            t.enrollment,
            t.last_update_date,
            t.completion_date,
            DATEDIFF(day, t.last_update_date, GETDATE()) AS days_since_update,
            (SELECT STRING_AGG(ind.name, ', ')
             FROM (SELECT DISTINCT i.name
                   FROM trial_indications ti
                   JOIN indications i ON ti.indication_id = i.indication_id
                   WHERE ti.trial_id = t.trial_id) ind
            ) AS indications
        FROM trials t
        WHERE t.is_stale = 1
        ORDER BY t.enrollment DESC
    """)


# ─────────────────────────────────────────────
# Tab 4: Drug Deep Dive
# ─────────────────────────────────────────────

def get_drug_list():
    """List of all drugs for the selectbox."""
    return run_query("""
        SELECT drug_id, inn AS drug_name, moa_class
        FROM drugs
        ORDER BY inn
    """)


def get_drug_detail(drug_id):
    """Detailed info for a single drug."""
    return run_query("""
        SELECT d.*, c.name AS originator_company
        FROM drugs d
        LEFT JOIN companies c ON d.originator_company_id = c.company_id
        WHERE d.drug_id = ?
    """, params=(drug_id,))


def get_drug_indications(drug_id):
    """Indications linked to a specific drug."""
    return run_query("""
        SELECT i.name, di.status, di.phase
        FROM drug_indications di
        JOIN indications i ON di.indication_id = i.indication_id
        WHERE di.drug_id = ?
        ORDER BY di.phase DESC
    """, params=(drug_id,))


def get_drug_trials(drug_id):
    """All trials linked to a specific drug."""
    return run_query("""
        SELECT t.nct_id, t.title, t.phase, t.overall_status,
               t.lead_sponsor_name, t.enrollment, t.start_date,
               t.completion_date, t.has_results,
               (SELECT STRING_AGG(ind.name, ', ')
                FROM (SELECT DISTINCT i.name
                      FROM trial_indications ti
                      JOIN indications i ON ti.indication_id = i.indication_id
                      WHERE ti.trial_id = t.trial_id) ind
               ) AS indications
        FROM drug_trials dt
        JOIN trials t ON dt.trial_id = t.trial_id
        WHERE dt.drug_id = ?
        ORDER BY t.start_date DESC
    """, params=(drug_id,))


def get_drug_approvals(drug_id):
    """FDA approval records for a specific drug."""
    return run_query("""
        SELECT * FROM approvals
        WHERE drug_id = ?
        ORDER BY approval_date
    """, params=(drug_id,))


def get_competitive_position(drug_id):
    """Other drugs in the same MoA class for competitive comparison."""
    return run_query("""
        SELECT d2.inn, d2.highest_phase,
               COUNT(DISTINCT dt2.trial_id) AS trial_count,
               COUNT(DISTINCT CASE WHEN t2.overall_status IN ('recruiting', 'active_not_recruiting') THEN dt2.trial_id END) AS active_trials
        FROM drugs d
        JOIN drugs d2 ON d.moa_class = d2.moa_class AND d.drug_id != d2.drug_id
        LEFT JOIN drug_trials dt2 ON d2.drug_id = dt2.drug_id
        LEFT JOIN trials t2 ON dt2.trial_id = t2.trial_id
        WHERE d.drug_id = ?
        GROUP BY d2.drug_id, d2.inn, d2.highest_phase
        ORDER BY active_trials DESC
    """, params=(drug_id,))


# =============================================
# Tab 5: Market Intelligence
# =============================================

def get_uk_prescriptions_trend():
    """UK monthly prescription trend per drug."""
    return run_query("""
        SELECT d.inn, d.moa_class, p.date, p.items, p.quantity, p.actual_cost,
               p.actual_cost / NULLIF(p.quantity, 0) AS cost_per_unit_gbp
        FROM prescriptions_uk p
        JOIN drugs d ON p.drug_id = d.drug_id
        ORDER BY d.inn, p.date
    """)


def get_uk_latest_month():
    """UK prescriptions for the latest month."""
    return run_query("""
        SELECT d.inn, d.moa_class, p.items, p.actual_cost, p.quantity,
               p.actual_cost / NULLIF(p.quantity, 0) AS cost_per_unit_gbp, p.date
        FROM prescriptions_uk p
        JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
        ORDER BY p.items DESC
    """)


def get_uk_moa_aggregation():
    """UK MoA class aggregation for latest month."""
    return run_query("""
        SELECT d.moa_class, SUM(p.items) AS total_items, SUM(p.actual_cost) AS total_cost,
               COUNT(DISTINCT d.drug_id) AS drug_count
        FROM prescriptions_uk p
        JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
        GROUP BY d.moa_class
        ORDER BY total_items DESC
    """)


def get_us_spending_trend():
    """US Medicare spending trend per drug (yearly)."""
    return run_query("""
        SELECT d.inn, d.moa_class, s.brand_name, s.year,
               s.total_spending, s.total_claims, s.total_beneficiaries, s.avg_cost_per_unit
        FROM spending_us s
        JOIN drugs d ON s.drug_id = d.drug_id
        ORDER BY d.inn, s.year
    """)


def get_us_spending_2023_top():
    """US top drugs by 2023 spending."""
    return run_query("""
        SELECT d.inn, d.moa_class, s.brand_name,
               s.total_spending, s.total_claims, s.total_beneficiaries, s.avg_cost_per_unit
        FROM spending_us s
        JOIN drugs d ON s.drug_id = d.drug_id
        WHERE s.year = 2023
        ORDER BY s.total_spending DESC
    """)


def get_us_spending_growth():
    """US spending YoY growth 2019 to 2023."""
    return run_query("""
        SELECT d.inn, d.moa_class,
               MIN(CASE WHEN s.year = 2019 THEN s.total_spending END) AS spend_2019,
               MIN(CASE WHEN s.year = 2023 THEN s.total_spending END) AS spend_2023
        FROM spending_us s
        JOIN drugs d ON s.drug_id = d.drug_id
        GROUP BY d.inn, d.moa_class
        HAVING MIN(CASE WHEN s.year = 2023 THEN s.total_spending END) IS NOT NULL
    """)


def get_market_kpis():
    """Market intelligence KPIs."""
    return run_query("""
        SELECT
            (SELECT SUM(total_spending) FROM spending_us WHERE year = 2023) AS us_total_2023,
            (SELECT TOP 1 items FROM prescriptions_uk p
             JOIN drugs d ON p.drug_id = d.drug_id
             WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
             ORDER BY p.items DESC) AS uk_top_items,
            (SELECT TOP 1 actual_cost FROM prescriptions_uk p
             WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
             ORDER BY p.actual_cost DESC) AS uk_top_cost,
            (SELECT COUNT(DISTINCT drug_id) FROM prescriptions_uk) AS uk_drug_count,
            (SELECT COUNT(DISTINCT drug_id) FROM spending_us) AS us_drug_count
    """)


# =============================================
# Tab 6: Safety Profile
# =============================================

def get_ae_heatmap_data():
    """Drug x AE heatmap data with serious ratio."""
    return run_query("""
        SELECT d.inn, d.moa_class, ae.event_term, ae.total_count, ae.serious_count,
               ae.non_serious_count,
               CAST(ae.serious_count AS FLOAT) / NULLIF(ae.total_count, 0) AS serious_ratio,
               ROW_NUMBER() OVER (PARTITION BY d.drug_id ORDER BY ae.total_count DESC) AS rank_in_drug
        FROM adverse_events ae
        JOIN drugs d ON ae.drug_id = d.drug_id
        ORDER BY d.inn, ae.total_count DESC
    """)


def get_ae_class_signals():
    """MoA class-level AE signals."""
    return run_query("""
        SELECT d.moa_class, ae.event_term, SUM(ae.total_count) AS class_total,
               SUM(ae.serious_count) AS class_serious,
               COUNT(DISTINCT d.drug_id) AS drugs_reporting
        FROM adverse_events ae
        JOIN drugs d ON ae.drug_id = d.drug_id
        GROUP BY d.moa_class, ae.event_term
        HAVING SUM(ae.total_count) > 1000
        ORDER BY d.moa_class, class_total DESC
    """)


def get_ae_trends():
    """Quarterly AE trends per drug."""
    return run_query("""
        SELECT d.inn, d.moa_class, t.quarter_date, t.total_reports, t.serious_reports,
               CAST(t.serious_reports AS FLOAT) / NULLIF(t.total_reports, 0) AS serious_ratio
        FROM adverse_event_trends t
        JOIN drugs d ON t.drug_id = d.drug_id
        ORDER BY d.inn, t.quarter_date
    """)


def get_ae_trends_by_moa():
    """Quarterly AE trends aggregated by MoA class."""
    return run_query("""
        SELECT d.moa_class, t.quarter_date,
               SUM(t.total_reports) AS class_total_reports,
               SUM(t.serious_reports) AS class_serious_reports
        FROM adverse_event_trends t
        JOIN drugs d ON t.drug_id = d.drug_id
        GROUP BY d.moa_class, t.quarter_date
        ORDER BY d.moa_class, t.quarter_date
    """)


def get_safety_kpis():
    """Safety profile KPIs."""
    return run_query("""
        SELECT
            (SELECT COUNT(DISTINCT drug_id) FROM adverse_events) AS drugs_with_faers,
            (SELECT TOP 1 d.inn FROM adverse_events ae
             JOIN drugs d ON ae.drug_id = d.drug_id
             GROUP BY d.inn ORDER BY SUM(ae.total_count) DESC) AS top_drug_by_reports,
            (SELECT TOP 1 SUM(ae.total_count) FROM adverse_events ae
             GROUP BY ae.drug_id ORDER BY SUM(ae.total_count) DESC) AS top_drug_total
    """)


def get_drug_ae_data(drug_id):
    """Top AEs for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT event_term, total_count, serious_count, non_serious_count,
               CAST(serious_count AS FLOAT) / NULLIF(total_count, 0) AS serious_ratio
        FROM adverse_events WHERE drug_id = ?
        ORDER BY total_count DESC
    """, params=(drug_id,))


def get_drug_ae_trend(drug_id):
    """AE trend for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT quarter_date, total_reports, serious_reports
        FROM adverse_event_trends WHERE drug_id = ?
        ORDER BY quarter_date
    """, params=(drug_id,))


# =============================================
# Tab 7: Patent & LOE
# =============================================

def get_loe_calendar():
    """Full LOE calendar with status classification."""
    return run_query("""
        SELECT d.inn, d.moa_class, d.modality,
               l.trade_name, l.applicant,
               l.effective_loe_date, l.years_until_loe,
               l.latest_patent_expiry, l.latest_exclusivity_expiry,
               l.patent_count, l.has_substance_patent, l.has_use_patent, l.has_product_patent,
               l.exclusivity_codes,
               CASE
                   WHEN l.effective_loe_date < GETDATE() THEN 'Past LOE'
                   WHEN l.years_until_loe <= 2 THEN 'Imminent (<2y)'
                   WHEN l.years_until_loe <= 5 THEN 'Medium (2-5y)'
                   ELSE 'Protected (>5y)'
               END AS loe_status
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        ORDER BY l.effective_loe_date ASC
    """)


def get_patent_type_distribution():
    """Patent type distribution for donut chart."""
    return run_query("""
        SELECT
            CASE
                WHEN drug_substance_flag = 'Y' AND drug_product_flag = 'Y' THEN 'Substance + Product'
                WHEN drug_substance_flag = 'Y' THEN 'Substance Only'
                WHEN drug_product_flag = 'Y' THEN 'Product Only'
                WHEN patent_use_code IS NOT NULL AND patent_use_code != '' THEN 'Use Patent'
                ELSE 'Other'
            END AS patent_type,
            COUNT(*) AS patent_count,
            COUNT(DISTINCT drug_id) AS drug_count
        FROM ob_patents
        WHERE (delist_flag IS NULL OR delist_flag != 'Y')
        AND appl_type IN ('N', 'B')
        AND drug_id IS NOT NULL
        GROUP BY
            CASE
                WHEN drug_substance_flag = 'Y' AND drug_product_flag = 'Y' THEN 'Substance + Product'
                WHEN drug_substance_flag = 'Y' THEN 'Substance Only'
                WHEN drug_product_flag = 'Y' THEN 'Product Only'
                WHEN patent_use_code IS NOT NULL AND patent_use_code != '' THEN 'Use Patent'
                ELSE 'Other'
            END
    """)


def get_exclusivity_distribution():
    """Exclusivity code distribution."""
    return run_query("""
        SELECT e.exclusivity_code, COUNT(DISTINCT e.drug_id) AS drug_count,
               STRING_AGG(sub.inn, ', ') AS drugs
        FROM ob_exclusivity e
        JOIN (SELECT DISTINCT e2.exclusivity_code, e2.drug_id, d2.inn
              FROM ob_exclusivity e2
              JOIN drugs d2 ON e2.drug_id = d2.drug_id
              WHERE e2.appl_type IN ('N', 'B')) sub
            ON e.exclusivity_code = sub.exclusivity_code AND e.drug_id = sub.drug_id
        WHERE e.appl_type IN ('N', 'B')
        GROUP BY e.exclusivity_code
        ORDER BY drug_count DESC
    """)


def get_loe_kpis():
    """Patent & LOE KPIs."""
    return run_query("""
        SELECT
            (SELECT COUNT(DISTINCT drug_id) FROM loe_summary) AS drugs_with_loe,
            (SELECT COUNT(DISTINCT drug_id) FROM loe_summary WHERE effective_loe_date > GETDATE()) AS future_loe,
            (SELECT COUNT(DISTINCT drug_id) FROM loe_summary WHERE effective_loe_date < GETDATE()) AS past_loe,
            (SELECT TOP 1 d.inn FROM loe_summary l JOIN drugs d ON l.drug_id = d.drug_id
             WHERE l.effective_loe_date > GETDATE() ORDER BY l.effective_loe_date ASC) AS next_loe_drug,
            (SELECT TOP 1 l.years_until_loe FROM loe_summary l
             WHERE l.effective_loe_date > GETDATE() ORDER BY l.effective_loe_date ASC) AS next_loe_years
    """)


def get_drug_loe_summary(drug_id):
    """LOE summary for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT * FROM loe_summary WHERE drug_id = ?
    """, params=(drug_id,))


def get_drug_patents(drug_id):
    """Patent list for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT patent_no, patent_expire_date, drug_substance_flag, drug_product_flag,
               patent_use_code, delist_flag
        FROM ob_patents WHERE drug_id = ?
        AND (delist_flag IS NULL OR delist_flag != 'Y')
        ORDER BY patent_expire_date DESC
    """, params=(drug_id,))


def get_drug_uk_trend(drug_id):
    """UK prescription trend for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT date, items, quantity, actual_cost,
               actual_cost / NULLIF(quantity, 0) AS cost_per_unit_gbp
        FROM prescriptions_uk WHERE drug_id = ?
        ORDER BY date
    """, params=(drug_id,))


def get_drug_us_spending(drug_id):
    """US spending for a single drug (Drug Deep Dive)."""
    return run_query("""
        SELECT brand_name, year, total_spending, total_claims,
               total_beneficiaries, avg_cost_per_unit
        FROM spending_us WHERE drug_id = ?
        ORDER BY year
    """, params=(drug_id,))
