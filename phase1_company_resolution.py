"""
Phase 1 Completion - Aufgabe 3: Company Resolution
Normalize sponsor names, create companies, link trials.
"""
import pyodbc
import json
import re
from datetime import datetime
from collections import Counter

from db_config import CONN_STR

COMPANY_NORMALIZATION = {
    "Novo Nordisk": [
        "Novo Nordisk A/S", "Novo Nordisk", "Novo Nordisk Pharmaceuticals",
        "Novo Nordisk Inc", "Novo Nordisk Inc.",
    ],
    "Eli Lilly": [
        "Eli Lilly and Company", "Eli Lilly", "Lilly",
        "Eli Lilly and Company, Indianapolis, IN",
    ],
    "AstraZeneca": [
        "AstraZeneca", "AstraZeneca AB", "AstraZeneca Pharmaceuticals",
    ],
    "Sanofi": [
        "Sanofi", "Sanofi-Aventis", "Sanofi Aventis", "Sanofi S.A.",
    ],
    "Merck (MSD)": [
        "Merck Sharp & Dohme LLC", "Merck Sharp & Dohme Corp.",
        "Merck & Co., Inc.", "Merck Sharp and Dohme", "MSD",
    ],
    "Pfizer": ["Pfizer", "Pfizer Inc", "Pfizer Inc."],
    "Boehringer Ingelheim": [
        "Boehringer Ingelheim", "Boehringer Ingelheim Pharmaceuticals",
        "Boehringer Ingelheim Pharmaceuticals, Inc.",
    ],
    "Johnson & Johnson": [
        "Janssen", "Janssen Pharmaceuticals", "Janssen Pharmaceutical K.K.",
        "Janssen Research & Development, LLC", "Janssen-Cilag",
        "Janssen-Cilag Ltd.", "Johnson & Johnson",
    ],
    "Takeda": [
        "Takeda", "Takeda Pharmaceutical Company",
        "Takeda Development Center Americas, Inc.", "Takeda Pharmaceuticals",
    ],
    "Novartis": [
        "Novartis", "Novartis Pharmaceuticals", "Novartis Pharmaceutical",
        "Novartis Pharmaceuticals Corporation",
    ],
    "Bristol-Myers Squibb": ["Bristol-Myers Squibb", "Bristol Myers Squibb"],
    "Amgen": ["Amgen", "Amgen Inc."],
    "Roche": [
        "Hoffmann-La Roche", "F. Hoffmann-La Roche Ltd", "Roche",
        "Genentech, Inc.",
    ],
    "AbbVie": ["AbbVie", "AbbVie Inc.", "Abbott"],
    "Gilead Sciences": ["Gilead Sciences"],
    "Bayer": ["Bayer", "Bayer AG", "Bayer Healthcare"],
    "GSK": ["GlaxoSmithKline", "GSK", "GlaxoSmithKline Consumer Healthcare"],
    "Zealand Pharma": ["Zealand Pharma", "Zealand Pharma A/S"],
    "Altimmune": ["Altimmune, Inc.", "Altimmune"],
}

BIG_PHARMA = [
    "Novo Nordisk", "Eli Lilly", "AstraZeneca", "Sanofi", "Merck (MSD)",
    "Pfizer", "Boehringer Ingelheim", "Johnson & Johnson", "Takeda",
    "Novartis", "Bristol-Myers Squibb", "Amgen", "Roche", "AbbVie",
    "Gilead Sciences", "Bayer", "GSK",
]

ACADEMIC_PATTERNS = [
    "university", "hospital", "medical center", "medical school",
    "institute", "school of medicine", "college of medicine",
    "clinic", "centre hospitalier", "hopitaux", "hopital",
    "fakultat", "charite", "klinik",
]

GOVERNMENT_PATTERNS = [
    "nih", "niddk", "nhlbi", "nci", "cdc", "department of",
    "ministry", "national institute", "veterans", " va ",
    "national health", "national cancer",
]

SUFFIX_PATTERNS = [
    r",?\s*(Inc\.?|Ltd\.?|LLC|Corp\.?|Corporation|A/S|GmbH|S\.A\.|AG|Co\.)$",
    r",?\s*(Pharmaceuticals?|Pharmaceutical)$",
    r",?\s*(Research|Development|Sciences?)$",
]


def normalize_name(name):
    """Remove common suffixes for fuzzy matching."""
    normalized = name.strip()
    for pattern in SUFFIX_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()
    return normalized


def classify_company(name, canonical_name=None):
    """Determine company_type."""
    check_name = canonical_name or name
    if check_name in BIG_PHARMA:
        return "big_pharma"

    name_lower = name.lower()
    for pattern in GOVERNMENT_PATTERNS:
        if pattern in name_lower:
            return "government"
    for pattern in ACADEMIC_PATTERNS:
        if pattern in name_lower:
            return "academic"
    return "biotech"


def main():
    print("=" * 70)
    print("AUFGABE 3: Company Resolution")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = pyodbc.connect(CONN_STR, timeout=30)
    conn.autocommit = True
    cursor = conn.cursor()
    print("Connected!")

    # Get all unique sponsors
    cursor.execute("""
        SELECT lead_sponsor_name, COUNT(*) AS trial_count
        FROM trials
        WHERE lead_sponsor_name IS NOT NULL AND lead_sponsor_name != ''
        GROUP BY lead_sponsor_name
        ORDER BY COUNT(*) DESC
    """)
    sponsors = cursor.fetchall()
    print(f"Unique sponsor names: {len(sponsors)}")

    # Build reverse lookup: variant -> canonical name
    variant_to_canonical = {}
    for canonical, variants in COMPANY_NORMALIZATION.items():
        for v in variants:
            variant_to_canonical[v.lower()] = canonical

    # Process sponsors
    company_cache = {}  # canonical_name -> company_id
    sponsor_to_company = {}  # raw_sponsor_name -> company_id
    type_counts = Counter()
    unmatched_sponsors = []

    for sponsor_name, trial_count in sponsors:
        sponsor_lower = sponsor_name.lower().strip()

        # 1. Check normalization map
        canonical = variant_to_canonical.get(sponsor_lower)

        # 2. If no exact match, try normalized name
        if not canonical:
            normalized = normalize_name(sponsor_name).lower()
            canonical = variant_to_canonical.get(normalized)

        # 3. If still no match, try partial matching on big pharma
        if not canonical:
            for can_name, variants in COMPANY_NORMALIZATION.items():
                for v in variants:
                    if v.lower() in sponsor_lower or sponsor_lower in v.lower():
                        canonical = can_name
                        break
                if canonical:
                    break

        # 4. Determine final name and type
        if canonical:
            display_name = canonical
        else:
            display_name = sponsor_name.strip()

        company_type = classify_company(sponsor_name, canonical)

        # Create or get company
        if display_name not in company_cache:
            # Check if already in DB
            cursor.execute("SELECT company_id FROM companies WHERE name = ?", (display_name,))
            existing = cursor.fetchone()
            if existing:
                company_cache[display_name] = existing[0]
            else:
                aliases = []
                if canonical and canonical in COMPANY_NORMALIZATION:
                    aliases = COMPANY_NORMALIZATION[canonical]

                is_public = 1 if company_type == "big_pharma" else 0
                cursor.execute("""
                    INSERT INTO companies (name, aliases, company_type, is_public)
                    OUTPUT INSERTED.company_id
                    VALUES (?, ?, ?, ?)
                """, (
                    display_name,
                    json.dumps(aliases) if aliases else None,
                    company_type,
                    is_public,
                ))
                company_cache[display_name] = cursor.fetchone()[0]
                type_counts[company_type] += 1

        sponsor_to_company[sponsor_name] = company_cache[display_name]

    print(f"\nCompanies created: {sum(type_counts.values())}")
    for ctype, count in type_counts.most_common():
        print(f"  {ctype}: {count}")

    # Link trials to companies
    print(f"\nLinking trials to companies...")
    linked = 0
    for sponsor_name, company_id in sponsor_to_company.items():
        cursor.execute("""
            UPDATE trials SET sponsor_company_id = ?
            WHERE lead_sponsor_name = ? AND sponsor_company_id IS NULL
        """, (company_id, sponsor_name))
        linked += cursor.rowcount

    print(f"Trials linked: {linked:,}")

    # Report
    print(f"\n{'='*70}")
    print("COMPANY RESOLUTION REPORT")
    print(f"{'='*70}")

    cursor.execute("SELECT COUNT(*) FROM companies")
    total_companies = cursor.fetchone()[0]
    print(f"\nTotal companies: {total_companies}")

    cursor.execute("""
        SELECT company_type, COUNT(*) FROM companies
        GROUP BY company_type ORDER BY COUNT(*) DESC
    """)
    print("\nCompanies by type:")
    for row in cursor.fetchall():
        print(f"  {row[0]:15s} {row[1]:>6,}")

    # Top 20 by trials
    print("\nTop 20 Companies by Trial Count:")
    cursor.execute("""
        SELECT TOP 20 c.name, c.company_type, COUNT(t.trial_id) AS cnt
        FROM companies c
        JOIN trials t ON c.company_id = t.sponsor_company_id
        GROUP BY c.name, c.company_type
        ORDER BY cnt DESC
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"  {i:2d}. {row[0]:45s} [{row[1]:12s}] {row[2]:>5,}")

    # Coverage
    cursor.execute("SELECT COUNT(*) FROM trials WHERE sponsor_company_id IS NOT NULL")
    with_company = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trials")
    total_trials = cursor.fetchone()[0]
    print(f"\nTrial-Company coverage: {with_company:,} / {total_trials:,} ({with_company/total_trials*100:.1f}%)")

    # Industry vs Academic vs Government
    print("\nTrials by Sponsor Type:")
    cursor.execute("""
        SELECT c.company_type, COUNT(t.trial_id) AS cnt
        FROM companies c
        JOIN trials t ON c.company_id = t.sponsor_company_id
        GROUP BY c.company_type
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15s} {row[1]:>6,}")

    cursor.close()
    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
