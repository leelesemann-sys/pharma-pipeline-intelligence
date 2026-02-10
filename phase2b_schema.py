"""
Phase 2b - Step 2: Schema Migration for Orange Book / Purple Book tables
"""

import pyodbc

from db_config import CONN_STR

TABLES = [
    # Orange Book Products
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ob_products')
    CREATE TABLE ob_products (
        id INT IDENTITY(1,1) PRIMARY KEY,
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        ingredient VARCHAR(500) NOT NULL,
        dosage_form_route VARCHAR(200),
        trade_name VARCHAR(200),
        applicant VARCHAR(100),
        applicant_full_name VARCHAR(500),
        strength VARCHAR(200),
        appl_type CHAR(1),
        appl_no VARCHAR(10) NOT NULL,
        product_no VARCHAR(5) NOT NULL,
        te_code VARCHAR(10),
        approval_date DATE,
        rld VARCHAR(5),
        rs VARCHAR(5),
        product_type VARCHAR(5),
        created_at DATETIME2 DEFAULT GETDATE(),
        UNIQUE(appl_no, product_no)
    );
    """,
    # Patents
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ob_patents')
    CREATE TABLE ob_patents (
        id INT IDENTITY(1,1) PRIMARY KEY,
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        appl_type CHAR(1),
        appl_no VARCHAR(10) NOT NULL,
        product_no VARCHAR(5) NOT NULL,
        patent_no VARCHAR(20) NOT NULL,
        patent_expire_date DATE,
        drug_substance_flag CHAR(1),
        drug_product_flag CHAR(1),
        patent_use_code VARCHAR(20),
        delist_flag CHAR(1),
        submission_date DATE,
        created_at DATETIME2 DEFAULT GETDATE(),
        UNIQUE(appl_no, product_no, patent_no)
    );
    """,
    # Exclusivity
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ob_exclusivity')
    CREATE TABLE ob_exclusivity (
        id INT IDENTITY(1,1) PRIMARY KEY,
        drug_id UNIQUEIDENTIFIER REFERENCES drugs(drug_id),
        appl_type CHAR(1),
        appl_no VARCHAR(10) NOT NULL,
        product_no VARCHAR(5) NOT NULL,
        exclusivity_code VARCHAR(20) NOT NULL,
        exclusivity_date DATE,
        created_at DATETIME2 DEFAULT GETDATE(),
        UNIQUE(appl_no, product_no, exclusivity_code)
    );
    """,
    # LOE Summary
    """
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'loe_summary')
    CREATE TABLE loe_summary (
        id INT IDENTITY(1,1) PRIMARY KEY,
        drug_id UNIQUEIDENTIFIER NOT NULL REFERENCES drugs(drug_id),
        trade_name VARCHAR(200),
        applicant VARCHAR(500),
        latest_patent_expiry DATE,
        latest_exclusivity_expiry DATE,
        effective_loe_date DATE,
        patent_count INT,
        has_substance_patent BIT,
        has_use_patent BIT,
        has_product_patent BIT,
        exclusivity_codes VARCHAR(200),
        years_until_loe AS DATEDIFF(day, GETDATE(), effective_loe_date) / 365.25,
        created_at DATETIME2 DEFAULT GETDATE(),
        UNIQUE(drug_id)
    );
    """,
]

INDEXES = [
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_products_ingredient') CREATE INDEX idx_ob_products_ingredient ON ob_products(ingredient);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_products_drug_id') CREATE INDEX idx_ob_products_drug_id ON ob_products(drug_id);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_patents_appl') CREATE INDEX idx_ob_patents_appl ON ob_patents(appl_no, product_no);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_patents_expire') CREATE INDEX idx_ob_patents_expire ON ob_patents(patent_expire_date);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_patents_drug_id') CREATE INDEX idx_ob_patents_drug_id ON ob_patents(drug_id);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_ob_exclusivity_appl') CREATE INDEX idx_ob_exclusivity_appl ON ob_exclusivity(appl_no, product_no);",
    "IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_loe_summary_date') CREATE INDEX idx_loe_summary_date ON loe_summary(effective_loe_date);",
]


def main():
    print("=" * 60)
    print("Phase 2b Step 2: Schema Migration")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    for i, ddl in enumerate(TABLES):
        table_name = ddl.split("'")[1]
        print(f"  Creating table: {table_name}")
        cursor.execute(ddl)
        conn.commit()

    for idx_sql in INDEXES:
        idx_name = idx_sql.split("'")[1]
        print(f"  Creating index: {idx_name}")
        cursor.execute(idx_sql)
        conn.commit()

    # Verify
    cursor.execute("""
        SELECT t.name FROM sys.tables t
        WHERE t.name IN ('ob_products','ob_patents','ob_exclusivity','loe_summary')
        ORDER BY t.name
    """)
    tables = [r[0] for r in cursor.fetchall()]
    print(f"\nVerified tables: {', '.join(tables)}")

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
