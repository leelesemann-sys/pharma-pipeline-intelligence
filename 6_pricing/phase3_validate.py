"""
Phase 3 - Step 5: Validation & Report Generation
Runs cross-source validation queries and generates phase3_ingest_report.md
"""

import pyodbc

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR


def main():
    print("=" * 60)
    print("Phase 3 Step 5: Validation & Report")
    print("=" * 60)

    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()

    report_lines = []
    report_lines.append("# Phase 3 Market & Safety Data Ingest Report\n")

    # ── Summary counts ──
    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM prescriptions_uk")
    nhs_drugs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM prescriptions_uk")
    nhs_records = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM prescriptions_uk")
    nhs_min, nhs_max = cursor.fetchone()

    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM adverse_events")
    faers_drugs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM adverse_events")
    faers_ae_records = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM adverse_event_trends")
    faers_trend_records = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(quarter_date), MAX(quarter_date) FROM adverse_event_trends")
    faers_min, faers_max = cursor.fetchone()

    cursor.execute("SELECT COUNT(DISTINCT drug_id) FROM spending_us")
    cms_drugs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM spending_us")
    cms_records = cursor.fetchone()[0]

    print(f"\nNHS: {nhs_drugs} drugs, {nhs_records} records, {nhs_min} to {nhs_max}")
    print(f"FAERS: {faers_drugs} drugs, {faers_ae_records} AE + {faers_trend_records} trend records, {faers_min} to {faers_max}")
    print(f"CMS: {cms_drugs} drugs, {cms_records} records, 2019-2023\n")

    report_lines.append("## Zusammenfassung\n")
    report_lines.append("| Quelle | Drugs mit Daten | Records geladen | Zeitraum |")
    report_lines.append("|--------|----------------|-----------------|----------|")
    report_lines.append(f"| NHS OpenPrescribing | {nhs_drugs}/43 | {nhs_records:,} Records | {nhs_min} bis {nhs_max} |")
    report_lines.append(f"| FDA FAERS | {faers_drugs}/43 | {faers_ae_records:,} AE + {faers_trend_records:,} Trend Records | {faers_min} bis {faers_max} |")
    report_lines.append(f"| CMS Medicare Part D | {cms_drugs}/43 | {cms_records:,} Records | 2019–2023 |")
    report_lines.append("")

    # ── NHS Details ──
    report_lines.append("---\n")
    report_lines.append("## NHS OpenPrescribing\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE EXISTS (SELECT 1 FROM prescriptions_uk p WHERE p.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    nhs_found = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs mit Daten ({len(nhs_found)}):** {', '.join(nhs_found)}\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE NOT EXISTS (SELECT 1 FROM prescriptions_uk p WHERE p.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    nhs_missing = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs ohne Daten ({len(nhs_missing)}):** {', '.join(nhs_missing)}\n")

    cursor.execute("""
        SELECT TOP 10 d.inn, p.items, p.actual_cost, p.date
        FROM prescriptions_uk p JOIN drugs d ON p.drug_id = d.drug_id
        WHERE p.date = (SELECT MAX(date) FROM prescriptions_uk)
        ORDER BY p.items DESC
    """)
    report_lines.append(f"**Top 10 by Items (aktuellster Monat):**\n")
    report_lines.append("| Drug | Items | Cost (GBP) | Date |")
    report_lines.append("|------|-------|-----------|------|")
    for row in cursor.fetchall():
        cost_str = f"{row[2]:,.2f}" if row[2] else "N/A"
        items_str = f"{row[1]:,}" if row[1] else "N/A"
        report_lines.append(f"| {row[0]} | {items_str} | {cost_str} | {row[3]} |")
        print(f"  NHS Top: {row[0]:25s} items={items_str:>10s}  cost=GBP {cost_str:>12s}")
    report_lines.append("")

    # ── FAERS Details ──
    report_lines.append("---\n")
    report_lines.append("## FDA FAERS\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE EXISTS (SELECT 1 FROM adverse_events ae WHERE ae.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    faers_found = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs mit Daten ({len(faers_found)}):** {', '.join(faers_found)}\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE NOT EXISTS (SELECT 1 FROM adverse_events ae WHERE ae.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    faers_missing = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs ohne Daten ({len(faers_missing)}):** {', '.join(faers_missing)}\n")

    cursor.execute("""
        SELECT TOP 15 d.inn, ae.event_term, ae.total_count, ae.serious_count
        FROM adverse_events ae JOIN drugs d ON ae.drug_id = d.drug_id
        ORDER BY ae.total_count DESC
    """)
    report_lines.append("**Top 15 AEs gesamt:**\n")
    report_lines.append("| Drug | Event Term | Total Count | Serious Count |")
    report_lines.append("|------|-----------|------------|--------------|")
    for row in cursor.fetchall():
        total = f"{row[2]:,}" if row[2] else "N/A"
        serious = f"{row[3]:,}" if row[3] else "N/A"
        report_lines.append(f"| {row[0]} | {row[1]} | {total} | {serious} |")
        print(f"  FAERS Top: {row[0]:20s} {row[1]:35s} total={total:>10s} serious={serious:>10s}")
    report_lines.append("")

    # MoA class signals
    cursor.execute("""
        SELECT TOP 15 d.moa_class, ae.event_term, SUM(ae.total_count) as class_total
        FROM adverse_events ae JOIN drugs d ON ae.drug_id = d.drug_id
        GROUP BY d.moa_class, ae.event_term
        ORDER BY class_total DESC
    """)
    report_lines.append("**Klassen-Signale (Top 15 MoA-Level AEs):**\n")
    report_lines.append("| MoA Class | Event Term | Class Total |")
    report_lines.append("|-----------|-----------|------------|")
    for row in cursor.fetchall():
        report_lines.append(f"| {row[0]} | {row[1]} | {row[2]:,} |")
        print(f"  MoA Signal: {row[0]:35s} {row[1]:35s} total={row[2]:>10,}")
    report_lines.append("")

    # ── CMS Details ──
    report_lines.append("---\n")
    report_lines.append("## CMS Medicare Part D\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE EXISTS (SELECT 1 FROM spending_us s WHERE s.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    cms_found = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs mit Daten ({len(cms_found)}):** {', '.join(cms_found)}\n")

    cursor.execute("""
        SELECT d.inn FROM drugs d
        WHERE NOT EXISTS (SELECT 1 FROM spending_us s WHERE s.drug_id = d.drug_id)
        ORDER BY d.inn
    """)
    cms_missing = [r[0] for r in cursor.fetchall()]
    report_lines.append(f"**Drugs ohne Daten ({len(cms_missing)}):** {', '.join(cms_missing)}\n")

    # Top 10 by 2023 spending
    cursor.execute("""
        SELECT TOP 10 d.inn, s.brand_name,
               SUM(s.total_spending) as total_spending,
               SUM(s.total_claims) as total_claims,
               SUM(s.total_beneficiaries) as total_benes
        FROM spending_us s JOIN drugs d ON s.drug_id = d.drug_id
        WHERE s.year = 2023
        GROUP BY d.inn, s.brand_name
        ORDER BY total_spending DESC
    """)
    report_lines.append("**Top 10 by 2023 Spending:**\n")
    report_lines.append("| Drug | Brand | Spending (USD) | Claims | Beneficiaries |")
    report_lines.append("|------|-------|---------------|--------|---------------|")
    for row in cursor.fetchall():
        spending = f"${row[2]:,.2f}" if row[2] else "N/A"
        claims = f"{row[3]:,}" if row[3] else "N/A"
        benes = f"{row[4]:,}" if row[4] else "N/A"
        report_lines.append(f"| {row[0]} | {row[1]} | {spending} | {claims} | {benes} |")
        print(f"  CMS Top: {row[0]:20s} {row[1]:20s} spending={spending:>16s}  claims={claims:>10s}")
    report_lines.append("")

    # GLP-1 spending trend
    cursor.execute("""
        SELECT d.inn, s.year, SUM(s.total_spending) as spending
        FROM spending_us s JOIN drugs d ON s.drug_id = d.drug_id
        WHERE d.moa_class = 'GLP-1 Receptor Agonist'
        GROUP BY d.inn, s.year
        ORDER BY d.inn, s.year
    """)
    report_lines.append("**GLP-1 Spending Trend 2019–2023:**\n")
    report_lines.append("| Drug | Year | Spending (USD) |")
    report_lines.append("|------|------|---------------|")
    for row in cursor.fetchall():
        spending = f"${row[2]:,.2f}" if row[2] else "N/A"
        report_lines.append(f"| {row[0]} | {row[1]} | {spending} |")
    report_lines.append("")

    # ── Cross-Source Coverage Matrix ──
    report_lines.append("---\n")
    report_lines.append("## Cross-Source Coverage\n")

    cursor.execute("""
        SELECT
            d.inn,
            d.moa_class,
            CASE WHEN EXISTS (SELECT 1 FROM prescriptions_uk p WHERE p.drug_id = d.drug_id) THEN 'Yes' ELSE 'No' END AS has_nhs,
            CASE WHEN EXISTS (SELECT 1 FROM adverse_events ae WHERE ae.drug_id = d.drug_id) THEN 'Yes' ELSE 'No' END AS has_faers,
            CASE WHEN EXISTS (SELECT 1 FROM spending_us s WHERE s.drug_id = d.drug_id) THEN 'Yes' ELSE 'No' END AS has_cms,
            (SELECT MAX(date) FROM prescriptions_uk p WHERE p.drug_id = d.drug_id) AS nhs_latest,
            (SELECT SUM(total_count) FROM adverse_events ae WHERE ae.drug_id = d.drug_id) AS faers_total
        FROM drugs d
        ORDER BY d.moa_class, d.inn
    """)

    report_lines.append("| Drug | MoA Class | NHS | FAERS | CMS | NHS Latest | FAERS Total Reports |")
    report_lines.append("|------|-----------|-----|-------|-----|-----------|-------------------|")

    all_3 = 0
    any_data = 0
    no_data = 0
    for row in cursor.fetchall():
        nhs = row[2]
        faers = row[3]
        cms = row[4]
        nhs_latest = str(row[5]) if row[5] else "-"
        faers_total = f"{row[6]:,}" if row[6] else "-"

        has_count = sum(1 for x in [nhs, faers, cms] if x == "Yes")
        if has_count == 3:
            all_3 += 1
        if has_count > 0:
            any_data += 1
        else:
            no_data += 1

        report_lines.append(f"| {row[0]} | {row[1]} | {nhs} | {faers} | {cms} | {nhs_latest} | {faers_total} |")
        print(f"  {row[0]:25s} {row[1]:35s} NHS={nhs:3s} FAERS={faers:3s} CMS={cms:3s}")

    report_lines.append("")
    report_lines.append(f"**Coverage Summary:** {all_3} drugs with all 3 sources, {any_data} with at least 1 source, {no_data} with no data\n")

    # ── Problems & Recommendations ──
    report_lines.append("---\n")
    report_lines.append("## Probleme & Empfehlungen\n")
    report_lines.append("### Bekannte Einschränkungen")
    report_lines.append("- **Pipeline-Drugs** (cagrilintide, danuglipron, insulin icodec, orforglipron, pemvidutide, retatrutide, survodutide) haben erwartungsgemäß keine Markt-/Sicherheitsdaten — sie sind noch nicht zugelassen")
    report_lines.append("- **Gliclazide** ist in den USA nicht zugelassen, daher keine FAERS/CMS-Daten")
    report_lines.append("- **Vildagliptin** ist in den USA nicht zugelassen (nur EU), daher keine FAERS/CMS-Daten")
    report_lines.append("- **Rosiglitazone** wurde vom Markt genommen, daher keine aktuellen CMS-Daten")
    report_lines.append("- **Bexagliflozin** ist neu zugelassen (2023), daher eingeschränkte CMS-Daten")
    report_lines.append("- **Resmetirom** ist neu zugelassen (2024), daher noch keine CMS-Daten")
    report_lines.append("")
    report_lines.append("### Empfehlungen für Dashboard-Integration")
    report_lines.append("1. **Neuer Tab: Market Data** — UK/US Spending-Trends, Cost-per-Unit Vergleich, Beneficiary-Trends")
    report_lines.append("2. **Neuer Tab: Safety Profile** — AE Heatmap, Serious/Non-Serious Ratio, Quarterly Trends")
    report_lines.append("3. **Drug Deep Dive erweitern** — Market + Safety Daten pro Drug hinzufügen")
    report_lines.append("4. **ML Feature Engineering** — Missing Data als Signal nutzen (kein Marktdata = Pipeline-Drug)")
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)
    with open("phase3_ingest_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n{'=' * 60}")
    print(f"Report written to phase3_ingest_report.md")
    print(f"{'=' * 60}")
    print(f"Coverage: {all_3} drugs all 3 sources, {any_data} at least 1, {no_data} no data")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
