"""
Phase 2b - Step 5: LOE Summary Calculation
Computes the effective Loss of Exclusivity date for each drug.
"""

import pyodbc

from db_config import CONN_STR


def main():
    print("=" * 60)
    print("Phase 2b Step 5: LOE Summary Calculation")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    # Clear existing LOE summary
    cursor.execute("DELETE FROM loe_summary")
    conn.commit()

    # For each drug that has ob_products as RLD (Reference Listed Drug) or BLA reference
    # We want one row per drug_id, picking the "main" trade name
    cursor.execute("""
        INSERT INTO loe_summary (drug_id, trade_name, applicant, latest_patent_expiry,
                                  latest_exclusivity_expiry, effective_loe_date,
                                  patent_count, has_substance_patent, has_use_patent,
                                  has_product_patent, exclusivity_codes)
        SELECT
            p.drug_id,

            -- Trade name: pick the most common RLD trade name for this drug
            (SELECT TOP 1 p2.trade_name
             FROM ob_products p2
             WHERE p2.drug_id = p.drug_id AND p2.appl_type IN ('N', 'B') AND p2.rld = 'Yes'
             GROUP BY p2.trade_name
             ORDER BY COUNT(*) DESC),

            -- Applicant
            (SELECT TOP 1 p2.applicant_full_name
             FROM ob_products p2
             WHERE p2.drug_id = p.drug_id AND p2.appl_type IN ('N', 'B') AND p2.rld = 'Yes'
             GROUP BY p2.applicant_full_name
             ORDER BY COUNT(*) DESC),

            -- Latest patent expiry (non-delisted, innovator only)
            (SELECT MAX(pat.patent_expire_date)
             FROM ob_patents pat
             WHERE pat.drug_id = p.drug_id
             AND (pat.delist_flag IS NULL OR pat.delist_flag != 'Y')
             AND pat.appl_type IN ('N', 'B')),

            -- Latest exclusivity expiry
            (SELECT MAX(ex.exclusivity_date)
             FROM ob_exclusivity ex
             WHERE ex.drug_id = p.drug_id
             AND ex.appl_type IN ('N', 'B')),

            -- Effective LOE = MAX(patent, exclusivity)
            (SELECT MAX(d) FROM (
                SELECT MAX(pat.patent_expire_date) as d
                FROM ob_patents pat
                WHERE pat.drug_id = p.drug_id
                AND (pat.delist_flag IS NULL OR pat.delist_flag != 'Y')
                AND pat.appl_type IN ('N', 'B')
                UNION ALL
                SELECT MAX(ex.exclusivity_date)
                FROM ob_exclusivity ex
                WHERE ex.drug_id = p.drug_id
                AND ex.appl_type IN ('N', 'B')
            ) dates),

            -- Patent count (distinct patents, non-delisted, innovator)
            (SELECT COUNT(DISTINCT pat.patent_no)
             FROM ob_patents pat
             WHERE pat.drug_id = p.drug_id
             AND (pat.delist_flag IS NULL OR pat.delist_flag != 'Y')
             AND pat.appl_type IN ('N', 'B')),

            -- Has substance patent
            CASE WHEN EXISTS (
                SELECT 1 FROM ob_patents pat
                WHERE pat.drug_id = p.drug_id AND pat.drug_substance_flag = 'Y'
                AND pat.appl_type IN ('N', 'B')
            ) THEN 1 ELSE 0 END,

            -- Has use patent
            CASE WHEN EXISTS (
                SELECT 1 FROM ob_patents pat
                WHERE pat.drug_id = p.drug_id AND pat.patent_use_code IS NOT NULL
                AND pat.patent_use_code != ''
                AND pat.appl_type IN ('N', 'B')
            ) THEN 1 ELSE 0 END,

            -- Has product patent
            CASE WHEN EXISTS (
                SELECT 1 FROM ob_patents pat
                WHERE pat.drug_id = p.drug_id AND pat.drug_product_flag = 'Y'
                AND pat.appl_type IN ('N', 'B')
            ) THEN 1 ELSE 0 END,

            -- Exclusivity codes (comma-separated distinct)
            (SELECT STRING_AGG(exc_code, ', ')
             FROM (SELECT DISTINCT ex.exclusivity_code as exc_code
                   FROM ob_exclusivity ex
                   WHERE ex.drug_id = p.drug_id
                   AND ex.appl_type IN ('N', 'B')) sub)

        FROM ob_products p
        WHERE p.drug_id IS NOT NULL
        AND p.appl_type IN ('N', 'B')
        AND p.rld = 'Yes'
        GROUP BY p.drug_id
    """)
    conn.commit()

    # Show results
    cursor.execute("SELECT COUNT(*) FROM loe_summary")
    total = cursor.fetchone()[0]
    print(f"\nLOE Summary computed for {total} drugs\n")

    # LOE Calendar - upcoming
    print("LOE Calendar (upcoming):")
    print("-" * 100)
    cursor.execute("""
        SELECT d.inn, l.trade_name, l.effective_loe_date, l.years_until_loe,
               l.patent_count, l.exclusivity_codes,
               l.has_substance_patent, l.has_use_patent, l.has_product_patent
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        WHERE l.effective_loe_date >= GETDATE()
        ORDER BY l.effective_loe_date ASC
    """)
    for row in cursor.fetchall():
        years = f"{row[3]:.1f}" if row[3] is not None else "N/A"
        patents = row[4] or 0
        excl = row[5] or "-"
        sub = "S" if row[6] else "-"
        use = "U" if row[7] else "-"
        prod = "P" if row[8] else "-"
        print(f"  {row[0]:25s} {(row[1] or ''):25s} LOE={row[2]}  {years:>5s}y  patents={patents:2d} [{sub}{use}{prod}]  excl={excl}")

    # Already post-LOE
    print(f"\nAlready post-LOE (generics/biosimilars possible):")
    print("-" * 100)
    cursor.execute("""
        SELECT d.inn, l.trade_name, l.effective_loe_date, l.years_until_loe,
               l.patent_count
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        WHERE l.effective_loe_date < GETDATE()
        ORDER BY l.effective_loe_date DESC
    """)
    for row in cursor.fetchall():
        years = f"{row[3]:.1f}" if row[3] is not None else "N/A"
        print(f"  {row[0]:25s} {(row[1] or ''):25s} LOE={row[2]}  {years:>6s}y ago  patents={row[4] or 0}")

    # Drugs without LOE data
    cursor.execute("""
        SELECT d.inn, d.moa_class FROM drugs d
        WHERE NOT EXISTS (SELECT 1 FROM loe_summary l WHERE l.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    no_loe = cursor.fetchall()
    print(f"\nDrugs without LOE data ({len(no_loe)}):")
    for row in no_loe:
        print(f"  {row[0]:25s} ({row[1]})")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
