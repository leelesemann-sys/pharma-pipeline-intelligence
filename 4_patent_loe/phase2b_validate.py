"""
Phase 2b - Step 6: Validation & Report Generation
"""

import pyodbc

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR


def main():
    print("=" * 60)
    print("Phase 2b Step 6: Validation & Report")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    report = []
    report.append("# Phase 2b: Orange Book & Purple Book Ingest Report\n")

    # -- Summary --
    cursor.execute("SELECT COUNT(*) FROM ob_products")
    total_products = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM ob_products WHERE appl_type = 'B'")
    bla_products = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM ob_products WHERE drug_id IS NOT NULL")
    matched_drugs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM ob_patents")
    total_patents = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM ob_exclusivity")
    total_excl = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM loe_summary")
    loe_count = cursor.fetchone()[0]

    report.append("## Zusammenfassung\n")
    report.append("| Metrik | Wert |")
    report.append("|--------|------|")
    report.append(f"| Orange Book Products geladen | {total_products - bla_products:,} |")
    report.append(f"| Purple Book Products geladen | {bla_products} |")
    report.append(f"| Drugs mit Match (von 43) | {matched_drugs} |")
    report.append(f"| Patente geladen | {total_patents:,} |")
    report.append(f"| Exclusivity Records | {total_excl:,} |")
    report.append(f"| LOE-Summary berechnet | {loe_count} Drugs |")
    report.append("")

    print(f"Products: {total_products:,} (OB: {total_products - bla_products:,}, PB: {bla_products})")
    print(f"Matched drugs: {matched_drugs}/43")
    print(f"Patents: {total_patents:,}, Exclusivity: {total_excl:,}")
    print(f"LOE Summary: {loe_count} drugs")

    # -- Drug Matching --
    report.append("## Drug Matching\n")
    report.append("| Drug | Quelle | Appl_No | Trade Name | Match |")
    report.append("|------|--------|---------|------------|-------|")

    cursor.execute("""
        SELECT d.inn,
               CASE p.appl_type WHEN 'N' THEN 'Orange Book' WHEN 'B' THEN 'Purple Book' WHEN 'A' THEN 'Orange Book (ANDA)' END,
               p.appl_type + p.appl_no,
               p.trade_name
        FROM ob_products p
        JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.rld = 'Yes' AND p.appl_type IN ('N', 'B')
        GROUP BY d.inn, p.appl_type, p.appl_no, p.trade_name
        ORDER BY d.inn, p.trade_name
    """)
    for row in cursor.fetchall():
        report.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | Y |")

    # Unmatched drugs
    cursor.execute("""
        SELECT d.inn, d.moa_class FROM drugs d
        WHERE NOT EXISTS (SELECT 1 FROM ob_products p WHERE p.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    for row in cursor.fetchall():
        report.append(f"| {row[0]} | - | - | - | N ({row[1]}) |")
    report.append("")

    # -- LOE Calendar (next 10 years) --
    report.append("## LOE-Kalender (bis 2036)\n")
    report.append("| Drug | Trade Name | LOE Date | Jahre bis LOE | Patente | Patent-Typ | Exclusivity |")
    report.append("|------|-----------|----------|---------------|---------|-----------|-------------|")

    cursor.execute("""
        SELECT d.inn, l.trade_name, l.effective_loe_date, l.years_until_loe,
               l.patent_count, l.has_substance_patent, l.has_use_patent, l.has_product_patent,
               l.exclusivity_codes
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        WHERE l.effective_loe_date >= GETDATE() AND l.effective_loe_date < '2036-01-01'
        ORDER BY l.effective_loe_date ASC
    """)
    for row in cursor.fetchall():
        years = f"{row[3]:.1f}" if row[3] is not None else "N/A"
        patents = row[4] or 0
        types = []
        if row[5]: types.append("Substance")
        if row[6]: types.append("Use")
        if row[7]: types.append("Product")
        type_str = ", ".join(types) if types else "-"
        excl = row[8] or "-"
        report.append(f"| {row[0]} | {row[1] or ''} | {row[2]} | {years} | {patents} | {type_str} | {excl} |")
    report.append("")

    # -- Long-term LOE (2036+) --
    report.append("## LOE nach 2036 (starker Patentschutz)\n")
    report.append("| Drug | Trade Name | LOE Date | Jahre bis LOE | Patente |")
    report.append("|------|-----------|----------|---------------|---------|")

    cursor.execute("""
        SELECT d.inn, l.trade_name, l.effective_loe_date, l.years_until_loe, l.patent_count
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        WHERE l.effective_loe_date >= '2036-01-01'
        ORDER BY l.effective_loe_date ASC
    """)
    for row in cursor.fetchall():
        years = f"{row[3]:.1f}" if row[3] is not None else "N/A"
        report.append(f"| {row[0]} | {row[1] or ''} | {row[2]} | {years} | {row[4] or 0} |")
    report.append("")

    # -- Already post-LOE --
    report.append("## Bereits post-LOE (Generika/Biosimilars moeglich)\n")
    report.append("| Drug | Trade Name | LOE Date | Jahre seit LOE |")
    report.append("|------|-----------|----------|----------------|")

    cursor.execute("""
        SELECT d.inn, l.trade_name, l.effective_loe_date, l.years_until_loe
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        WHERE l.effective_loe_date < GETDATE()
        ORDER BY l.effective_loe_date DESC
    """)
    for row in cursor.fetchall():
        years = f"{abs(row[3]):.1f}" if row[3] is not None else "N/A"
        report.append(f"| {row[0]} | {row[1] or ''} | {row[2]} | {years} |")
    report.append("")

    # -- Patent Analysis --
    report.append("## Patent-Analyse\n")

    cursor.execute("SELECT COUNT(*) FROM loe_summary WHERE has_substance_patent = 1")
    sub_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM loe_summary WHERE has_use_patent = 1")
    use_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM loe_summary WHERE has_product_patent = 1")
    prod_count = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(CAST(patent_count AS FLOAT)) FROM loe_summary WHERE patent_count > 0")
    avg_patents = cursor.fetchone()[0]

    report.append(f"- Drugs mit Substance-Patenten: {sub_count}")
    report.append(f"- Drugs mit Use-Patenten: {use_count}")
    report.append(f"- Drugs mit Product-Patenten: {prod_count}")
    report.append(f"- Durchschnittliche Patente pro Drug (nur Drugs mit Patenten): {avg_patents:.1f}")
    report.append("")

    # MoA class patent landscape
    report.append("### Patent-Landschaft pro MoA-Klasse\n")
    report.append("| MoA Class | Avg Patents | Earliest LOE | Latest LOE |")
    report.append("|-----------|------------|-------------|-----------|")

    cursor.execute("""
        SELECT d.moa_class,
               AVG(CAST(l.patent_count AS FLOAT)) as avg_patents,
               MIN(l.effective_loe_date) as earliest_loe,
               MAX(l.effective_loe_date) as latest_loe
        FROM loe_summary l
        JOIN drugs d ON l.drug_id = d.drug_id
        GROUP BY d.moa_class
        ORDER BY earliest_loe
    """)
    for row in cursor.fetchall():
        avg = f"{row[1]:.1f}" if row[1] is not None else "0"
        report.append(f"| {row[0]} | {avg} | {row[2]} | {row[3]} |")
    report.append("")

    # -- Problems & Recommendations --
    report.append("## Probleme & Empfehlungen\n")
    report.append("### Bekannte Einschraenkungen")
    report.append("- **9 Drugs ohne LOE-Daten:** 7 Pipeline-Drugs (nicht zugelassen), Gliclazide (nicht in USA), Vildagliptin (nur EU)")
    report.append("- **Biologics (Purple Book):** Keine Patent-Daten verfuegbar -- BPCIA 12-Jahres-Exklusivitaet als Proxy verwendet")
    report.append("- **Combo-Products:** Manche Drugs erscheinen in Kombinations-Produkten (z.B. Metformin in Janumet, Invokamet) -- LOE bezieht sich auf das letzte ablaufende Patent aller zugehoerigen Produkte")
    report.append("- **Patent Challenges:** Paragraph IV Challenges und IPR-Entscheidungen nicht beruecksichtigt -- tatsaechliche Generic-Entry kann frueher erfolgen")
    report.append("")
    report.append("### Empfehlungen")
    report.append("1. **Dashboard-Integration:** LOE-Kalender als Timeline-Chart, Patent-Portfolio als Bubble-Chart")
    report.append("2. **Commercial Alerts:** Drugs mit LOE < 2 Jahre markieren (Dulaglutide, Insulin Lispro, Insulin Degludec)")
    report.append("3. **BD&L Opportunities:** Post-LOE Drugs als Generika/Biosimilar-Opportunities flaggen")
    report.append("4. **Patent Cliff Analysis:** GLP-1 Klasse hat starken Schutz bis 2037-2041 -- Hochpreisphase")
    report.append("")

    # Write report
    report_text = "\n".join(report)
    with open("phase2b_orange_book_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport written to phase2b_orange_book_report.md")
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
