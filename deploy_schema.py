"""
Phase 1: Deploy database schema to Azure SQL
Pharma Pipeline Intelligence - Diabetes & Obesity
"""
import pyodbc
import sys
from db_config import CONN_STR

# Split into individual statements (GO-equivalent)
SCHEMA_STATEMENTS = [
    # COMPANIES
    """
    CREATE TABLE companies (
        company_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        name NVARCHAR(500) NOT NULL,
        aliases NVARCHAR(MAX),
        sec_cik NVARCHAR(20),
        lei NVARCHAR(20),
        hq_country NVARCHAR(5),
        company_type NVARCHAR(20) CHECK (company_type IN ('big_pharma','mid_pharma','biotech','academic','cro','government','other')),
        is_public BIT DEFAULT 0,
        parent_company_id UNIQUEIDENTIFIER REFERENCES companies(company_id),
        created_at DATETIME2 DEFAULT GETUTCDATE(),
        updated_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # INDICATIONS
    """
    CREATE TABLE indications (
        indication_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        name NVARCHAR(500) NOT NULL,
        mesh_term NVARCHAR(500),
        icd10_code NVARCHAR(20),
        icd11_code NVARCHAR(20),
        meddra_pt NVARCHAR(500),
        snomed_id NVARCHAR(20),
        parent_indication_id UNIQUEIDENTIFIER REFERENCES indications(indication_id),
        therapeutic_area NVARCHAR(200),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # DRUGS
    """
    CREATE TABLE drugs (
        drug_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        inn NVARCHAR(500),
        brand_names NVARCHAR(MAX),
        company_codes NVARCHAR(MAX),
        atc_code NVARCHAR(20),
        rxnorm_cui NVARCHAR(20),
        chembl_id NVARCHAR(30),
        unii NVARCHAR(20),
        drugbank_id NVARCHAR(20),
        modality NVARCHAR(30) CHECK (modality IN ('small_molecule','biologic','peptide','cell_therapy','gene_therapy','oligonucleotide','other')),
        moa_class NVARCHAR(200),
        targets NVARCHAR(MAX),
        highest_phase NVARCHAR(20) CHECK (highest_phase IN ('preclinical','phase1','phase2','phase3','approved','withdrawn')),
        first_approval_date DATE,
        originator_company_id UNIQUEIDENTIFIER REFERENCES companies(company_id),
        created_at DATETIME2 DEFAULT GETUTCDATE(),
        updated_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # TRIALS
    """
    CREATE TABLE trials (
        trial_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        nct_id NVARCHAR(20) UNIQUE NOT NULL,
        eu_ctr_id NVARCHAR(30),
        title NVARCHAR(MAX),
        phase NVARCHAR(20) CHECK (phase IN ('early_phase1','phase1','phase1_phase2','phase2','phase2_phase3','phase3','phase4','na')),
        overall_status NVARCHAR(30) CHECK (overall_status IN ('recruiting','not_yet_recruiting','enrolling_by_invitation','active_not_recruiting','completed','terminated','withdrawn','suspended','unknown')),
        start_date DATE,
        primary_completion_date DATE,
        completion_date DATE,
        last_update_date DATE,
        enrollment INT,
        enrollment_type NVARCHAR(20),
        study_type NVARCHAR(20) CHECK (study_type IN ('interventional','observational','expanded_access')),
        has_results BIT DEFAULT 0,
        why_stopped NVARCHAR(MAX),
        sponsor_company_id UNIQUEIDENTIFIER REFERENCES companies(company_id),
        lead_sponsor_name NVARCHAR(500),
        is_stale AS (CASE
            WHEN overall_status IN ('recruiting','active_not_recruiting','not_yet_recruiting','enrolling_by_invitation')
            AND last_update_date < DATEADD(MONTH, -12, GETUTCDATE())
            AND (primary_completion_date IS NOT NULL AND primary_completion_date < GETUTCDATE())
            THEN 1 ELSE 0 END),
        termination_risk_score FLOAT,
        raw_conditions NVARCHAR(MAX),
        raw_interventions NVARCHAR(MAX),
        created_at DATETIME2 DEFAULT GETUTCDATE(),
        updated_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # PATENTS
    """
    CREATE TABLE patents (
        patent_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        patent_number NVARCHAR(30),
        expiry_date DATE,
        patent_type NVARCHAR(30) CHECK (patent_type IN ('substance','formulation','method_of_use','process','other')),
        exclusivity_code NVARCHAR(10),
        exclusivity_expiry DATE,
        orange_book_listed BIT DEFAULT 0,
        paragraph_iv_filed BIT DEFAULT 0,
        source NVARCHAR(50) DEFAULT 'orange_book',
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # APPROVALS
    """
    CREATE TABLE approvals (
        approval_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        indication_id UNIQUEIDENTIFIER REFERENCES indications(indication_id),
        country NVARCHAR(5),
        agency NVARCHAR(20) CHECK (agency IN ('FDA','EMA','BfArM','MHRA','PMDA','other')),
        application_number NVARCHAR(30),
        approval_date DATE,
        submission_date DATE,
        review_type NVARCHAR(30) CHECK (review_type IN ('standard','priority','accelerated','breakthrough','fast_track','orphan','conditional','other')),
        amnog_benefit NVARCHAR(100),
        amnog_target_population INT,
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # MARKET DATA
    """
    CREATE TABLE market_data (
        market_data_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        country NVARCHAR(5),
        source NVARCHAR(30) CHECK (source IN ('cms_medicare','nhs_openprescribing','gkv_index')),
        period DATE,
        total_claims INT,
        total_patients INT,
        total_quantity FLOAT,
        total_cost FLOAT,
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # JUNCTION: drug_trials
    """
    CREATE TABLE drug_trials (
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        trial_id UNIQUEIDENTIFIER REFERENCES trials(trial_id),
        role NVARCHAR(20) CHECK (role IN ('experimental','comparator','combination','background')),
        PRIMARY KEY (drug_id, trial_id)
    )
    """,
    # JUNCTION: drug_indications
    """
    CREATE TABLE drug_indications (
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        indication_id UNIQUEIDENTIFIER REFERENCES indications(indication_id),
        status NVARCHAR(20) CHECK (status IN ('approved','investigational','discontinued')),
        phase NVARCHAR(20),
        PRIMARY KEY (drug_id, indication_id)
    )
    """,
    # JUNCTION: trial_indications
    """
    CREATE TABLE trial_indications (
        trial_id UNIQUEIDENTIFIER REFERENCES trials(trial_id),
        indication_id UNIQUEIDENTIFIER REFERENCES indications(indication_id),
        PRIMARY KEY (trial_id, indication_id)
    )
    """,
    # PREDICTIONS
    """
    CREATE TABLE predictions (
        prediction_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        indication_id UNIQUEIDENTIFIER REFERENCES indications(indication_id),
        model_name NVARCHAR(100),
        model_version NVARCHAR(50),
        prediction_date DATE,
        phase_transition_prob FLOAT,
        approval_prob FLOAT,
        estimated_approval_date DATE,
        confidence_interval NVARCHAR(MAX),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # NLP SIGNALS
    """
    CREATE TABLE nlp_signals (
        signal_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        source_type NVARCHAR(30) CHECK (source_type IN ('epar','adcom','sec_10k','pubmed','earnings_call','press_release')),
        document_date DATE,
        sentiment_score FLOAT,
        key_findings NVARCHAR(MAX),
        raw_text_ref NVARCHAR(500),
        created_at DATETIME2 DEFAULT GETUTCDATE()
    )
    """,
    # ALERTS
    """
    CREATE TABLE alerts (
        alert_id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        alert_type NVARCHAR(30) CHECK (alert_type IN ('new_approval','status_change','stale_trial','high_termination_risk','loe_approaching','new_trial','pipeline_removal')),
        entity_type NVARCHAR(10) CHECK (entity_type IN ('drug','trial','company')),
        entity_id UNIQUEIDENTIFIER,
        triggered_at DATETIME2 DEFAULT GETUTCDATE(),
        details NVARCHAR(MAX)
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IX_trials_nct_id ON trials(nct_id)",
    "CREATE INDEX IX_trials_status ON trials(overall_status)",
    "CREATE INDEX IX_trials_phase ON trials(phase)",
    "CREATE INDEX IX_trials_sponsor ON trials(sponsor_company_id)",
    "CREATE INDEX IX_trials_last_update ON trials(last_update_date)",
    "CREATE INDEX IX_drugs_inn ON drugs(inn)",
    "CREATE INDEX IX_drugs_atc ON drugs(atc_code)",
    "CREATE INDEX IX_drugs_chembl ON drugs(chembl_id)",
    "CREATE INDEX IX_drugs_highest_phase ON drugs(highest_phase)",
    "CREATE INDEX IX_patents_expiry ON patents(expiry_date)",
    "CREATE INDEX IX_approvals_drug ON approvals(drug_id)",
    "CREATE INDEX IX_approvals_date ON approvals(approval_date)",
    "CREATE INDEX IX_market_data_drug_period ON market_data(drug_id, period)",
    "CREATE INDEX IX_predictions_drug ON predictions(drug_id)",
    "CREATE INDEX IX_alerts_type_date ON alerts(alert_type, triggered_at)",
]

SEED_STATEMENTS = [
    # Top-level therapeutic area
    """
    INSERT INTO indications (indication_id, name, therapeutic_area)
    VALUES (NEWID(), 'Diabetes & Metabolism', 'Diabetes & Metabolism')
    """,
    # Core indications
    """
    DECLARE @ta_id UNIQUEIDENTIFIER = (SELECT indication_id FROM indications WHERE name = 'Diabetes & Metabolism');
    INSERT INTO indications (name, mesh_term, icd10_code, parent_indication_id, therapeutic_area) VALUES
    ('Type 2 Diabetes Mellitus', 'Diabetes Mellitus, Type 2', 'E11', @ta_id, 'Diabetes & Metabolism'),
    ('Type 1 Diabetes Mellitus', 'Diabetes Mellitus, Type 1', 'E10', @ta_id, 'Diabetes & Metabolism'),
    ('Obesity', 'Obesity', 'E66', @ta_id, 'Diabetes & Metabolism'),
    ('Overweight', 'Overweight', 'E66.0', @ta_id, 'Diabetes & Metabolism'),
    ('NASH / MASH', 'Non-alcoholic Steatohepatitis', 'K75.81', @ta_id, 'Diabetes & Metabolism'),
    ('Non-alcoholic Fatty Liver Disease', 'Non-alcoholic Fatty Liver Disease', 'K76.0', @ta_id, 'Diabetes & Metabolism'),
    ('Diabetic Kidney Disease', 'Diabetic Nephropathies', 'E11.2', @ta_id, 'Diabetes & Metabolism'),
    ('Diabetic Retinopathy', 'Diabetic Retinopathy', 'E11.3', @ta_id, 'Diabetes & Metabolism'),
    ('Diabetic Neuropathy', 'Diabetic Neuropathies', 'E11.4', @ta_id, 'Diabetes & Metabolism'),
    ('Metabolic Syndrome', 'Metabolic Syndrome', 'E88.81', @ta_id, 'Diabetes & Metabolism'),
    ('Prediabetes', 'Prediabetic State', 'R73.03', @ta_id, 'Diabetes & Metabolism'),
    ('Gestational Diabetes', 'Diabetes, Gestational', 'O24.4', @ta_id, 'Diabetes & Metabolism')
    """,
]


def main():
    print("=" * 60)
    print("PHARMA PIPELINE INTELLIGENCE - Schema Deployment")
    print("=" * 60)

    print(f"\nConnecting to Azure SQL...")
    try:
        conn = pyodbc.connect(CONN_STR, timeout=30)
        conn.autocommit = True
        cursor = conn.cursor()
        print("Connected successfully!")
    except Exception as e:
        print(f"ERROR connecting: {e}")
        sys.exit(1)

    # Deploy tables
    print(f"\n--- Creating {len(SCHEMA_STATEMENTS)} tables ---")
    for i, stmt in enumerate(SCHEMA_STATEMENTS):
        table_name = stmt.strip().split("CREATE TABLE ")[1].split(" ")[0].split("(")[0]
        try:
            cursor.execute(stmt)
            print(f"  [{i+1}/{len(SCHEMA_STATEMENTS)}] Created: {table_name}")
        except pyodbc.ProgrammingError as e:
            if "There is already an object named" in str(e):
                print(f"  [{i+1}/{len(SCHEMA_STATEMENTS)}] EXISTS:  {table_name} (skipping)")
            else:
                print(f"  [{i+1}/{len(SCHEMA_STATEMENTS)}] ERROR:   {table_name}: {e}")

    # Deploy indexes
    print(f"\n--- Creating {len(INDEX_STATEMENTS)} indexes ---")
    for i, stmt in enumerate(INDEX_STATEMENTS):
        idx_name = stmt.split("CREATE INDEX ")[1].split(" ON")[0]
        try:
            cursor.execute(stmt)
            print(f"  [{i+1}/{len(INDEX_STATEMENTS)}] Created: {idx_name}")
        except pyodbc.ProgrammingError as e:
            if "already exists" in str(e):
                print(f"  [{i+1}/{len(INDEX_STATEMENTS)}] EXISTS:  {idx_name} (skipping)")
            else:
                print(f"  [{i+1}/{len(INDEX_STATEMENTS)}] ERROR:   {idx_name}: {e}")

    # Seed data
    print(f"\n--- Seeding indication data ---")
    for i, stmt in enumerate(SEED_STATEMENTS):
        try:
            cursor.execute(stmt)
            print(f"  [{i+1}/{len(SEED_STATEMENTS)}] Seed data inserted")
        except Exception as e:
            if "Violation of UNIQUE KEY" in str(e) or "duplicate" in str(e).lower():
                print(f"  [{i+1}/{len(SEED_STATEMENTS)}] Already seeded (skipping)")
            else:
                print(f"  [{i+1}/{len(SEED_STATEMENTS)}] ERROR: {e}")

    # Verify
    print(f"\n--- Verification ---")
    cursor.execute("SELECT COUNT(*) FROM indications")
    ind_count = cursor.fetchone()[0]
    print(f"  Indications: {ind_count}")

    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  Tables created: {len(tables)}")
    for t in tables:
        print(f"    - {t}")

    cursor.close()
    conn.close()
    print(f"\n{'='*60}")
    print("Schema deployment complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
