"""
UMLS / RxNorm Enrichment Script
Pharma Pipeline Intelligence - Diabetes & Obesity

Looks up RxNorm CUI (RXCUI) for all 43 drugs via:
  1. RxNorm REST API  (rxnav.nlm.nih.gov — free, no auth)
  2. UMLS REST API     (uts-ws.nlm.nih.gov — for cross-references)

Then UPDATEs drugs.rxnorm_cui in Azure SQL.

Usage:
    python enrich_rxnorm.py              # dry-run (prints results, no DB write)
    python enrich_rxnorm.py --write      # writes to DB
    python enrich_rxnorm.py --write --umls  # also fetches UMLS cross-refs
"""
import os
import requests
import pyodbc
import sys
import time
import json
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db_config import CONN_STR

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "SET_UMLS_API_KEY_ENV_VAR")

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Rate limit: 20 req/s for both APIs — we use 0.15s delay to be safe
API_DELAY = 0.15

# Azure SQL Serverless retry settings
MAX_DB_RETRIES = 5
DB_BASE_DELAY = 5  # seconds, doubles each retry

# Manual overrides for drugs that RxNorm API won't find with INN alone
# Format: inn_lower -> (rxcui, preferred_name)
MANUAL_RXCUI = {
    # Insulin products: RxNorm has ingredient-level concepts
    "insulin glargine": ("274783", "insulin glargine"),
    "insulin lispro": ("86009", "insulin lispro"),
    "insulin aspart": ("86005", "insulin aspart"),
    "insulin degludec": ("1372738", "insulin degludec"),
    "faster-acting insulin aspart": ("86005", "insulin aspart (fast-acting)"),
    # Pipeline drugs not yet in RxNorm
    "insulin icodec": (None, "Not in RxNorm (pipeline)"),
    "orforglipron": (None, "Not in RxNorm (pipeline)"),
    "danuglipron": (None, "Not in RxNorm (pipeline)"),
    "survodutide": (None, "Not in RxNorm (pipeline)"),
    "retatrutide": (None, "Not in RxNorm (pipeline)"),
    "cagrilintide": (None, "Not in RxNorm (pipeline)"),
    "pemvidutide": (None, "Not in RxNorm (pipeline)"),
    "taspoglutide": (None, "Not in RxNorm (discontinued)"),
    # Setmelanotide (MC4R agonist, obesity but different MoA)
    "setmelanotide": ("2479677", "setmelanotide"),
    # Comparator entries (skip)
    "comparator: placebo": (None, "Not a drug (comparator)"),
}


# ─────────────────────────────────────────────────
# Azure SQL connection with retry (Serverless auto-resume)
# ─────────────────────────────────────────────────
def connect_with_retry():
    """Connect to Azure SQL with retry for Serverless auto-resume."""
    retryable = ["40613", "18456", "08S01", "08001", "40197", "40501",
                 "not currently available", "communication link", "login failed",
                 "timeout", "login timeout"]
    last_err = None
    for attempt in range(1, MAX_DB_RETRIES + 1):
        try:
            print(f"  Connecting (attempt {attempt}/{MAX_DB_RETRIES})...")
            conn = pyodbc.connect(CONN_STR)
            conn.cursor().execute("SELECT 1").fetchone()
            print(f"  Connected!")
            return conn
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(code in err_str for code in retryable) and attempt < MAX_DB_RETRIES:
                delay = DB_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  DB resuming... retry in {delay}s")
                time.sleep(delay)
            else:
                break
    raise ConnectionError(f"DB connection failed after {MAX_DB_RETRIES} attempts: {last_err}")


# ─────────────────────────────────────────────────
# RxNorm API (free, no auth)
# ─────────────────────────────────────────────────
def rxnorm_find_rxcui(drug_name: str) -> dict | None:
    """
    Look up RxCUI via RxNorm REST API.
    Uses findRxcuiByString with search=2 (exact, then normalized).
    Returns {"rxcui": str, "name": str} or None.
    """
    url = f"{RXNORM_BASE}/rxcui.json"
    params = {"name": drug_name, "search": 2}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])
        if rxnorm_ids:
            rxcui = rxnorm_ids[0]
            # Get the preferred name for this RXCUI
            name = rxnorm_get_name(rxcui)
            return {"rxcui": rxcui, "name": name or drug_name}
    except Exception as e:
        print(f"    [WARN] RxNorm API error for '{drug_name}': {e}")
    return None


def rxnorm_get_name(rxcui: str) -> str | None:
    """Get preferred name for an RXCUI."""
    url = f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        props = data.get("properties", {})
        return props.get("name")
    except Exception:
        return None


def rxnorm_get_related(rxcui: str) -> dict:
    """Get related concepts (brand names, dose forms, etc.)."""
    url = f"{RXNORM_BASE}/rxcui/{rxcui}/related.json"
    params = {"tty": "BN+IN+PIN"}  # Brand Name, Ingredient, Precise Ingredient
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        related = {}
        for group in data.get("relatedGroup", {}).get("conceptGroup", []):
            tty = group.get("tty", "")
            concepts = group.get("conceptProperties", [])
            if concepts:
                related[tty] = [{"rxcui": c["rxcui"], "name": c["name"]} for c in concepts]
        return related
    except Exception:
        return {}


# ─────────────────────────────────────────────────
# UMLS API (requires API key)
# ─────────────────────────────────────────────────
def umls_search(drug_name: str, source: str = "RXNORM") -> dict | None:
    """
    Search UMLS for a drug name, filtered by source vocabulary.
    Returns {"cui": str, "name": str} or None.
    """
    url = f"{UMLS_BASE}/search/current"
    params = {
        "apiKey": UMLS_API_KEY,
        "string": drug_name,
        "sabs": source,
        "returnIdType": "concept",
        "searchType": "exact",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", {}).get("results", [])
        if results and results[0].get("ui") != "NONE":
            return {"cui": results[0]["ui"], "name": results[0]["name"]}
    except Exception as e:
        print(f"    [WARN] UMLS API error for '{drug_name}': {e}")
    return None


def umls_get_crosswalk(cui: str, target_source: str) -> list:
    """
    Get cross-references from a UMLS CUI to a target vocabulary.
    E.g., target_source = "MSH" for MeSH, "MDR" for MedDRA.
    Returns list of {"code": str, "name": str}.
    """
    url = f"{UMLS_BASE}/content/current/CUI/{cui}/atoms"
    params = {
        "apiKey": UMLS_API_KEY,
        "sabs": target_source,
        "ttys": "PT,MH,NM",  # Preferred Term, MeSH Heading, Supplementary Name
        "pageSize": 25,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", [])
        codes = []
        seen = set()
        for atom in results:
            code = atom.get("code", "").split("/")[-1] if "code" in atom else ""
            name = atom.get("name", "")
            if code and code not in seen:
                seen.add(code)
                codes.append({"code": code, "name": name, "tty": atom.get("termType", "")})
        return codes
    except Exception as e:
        print(f"    [WARN] UMLS crosswalk error for CUI {cui} -> {target_source}: {e}")
        return []


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────
def main():
    write_mode = "--write" in sys.argv
    umls_mode = "--umls" in sys.argv

    print("=" * 70)
    print("PHARMA PIPELINE — RxNorm / UMLS Enrichment")
    print("=" * 70)
    print(f"  Mode:      {'WRITE to DB' if write_mode else 'DRY-RUN (no DB writes)'}")
    print(f"  UMLS:      {'Enabled (cross-references)' if umls_mode else 'Disabled (RxNorm only)'}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Step 1: Load drugs from DB ──
    print("Step 1: Loading core drugs from Azure SQL...")
    try:
        conn = connect_with_retry()
        cursor = conn.cursor()
        # Only our 43 core drugs (identified by having moa_class set)
        cursor.execute("""
            SELECT drug_id, inn, rxnorm_cui, chembl_id, brand_names, moa_class
            FROM drugs
            WHERE moa_class IS NOT NULL
            ORDER BY inn
        """)
        drugs = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        print(f"  Found {len(drugs)} core drugs (moa_class IS NOT NULL)\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # ── Step 2: RxNorm Lookup ──
    print("Step 2: Looking up RxCUI via RxNorm API...")
    print("-" * 70)

    results = []
    found = 0
    skipped = 0
    not_found = 0

    for row in drugs:
        drug_id = str(row[0])
        inn = row[1] or ""
        existing_rxcui = row[2]
        chembl_id = row[3]
        brand_names = row[4] or ""
        moa_class = row[5] or ""

        inn_lower = inn.strip().lower()

        # Skip if already has RxCUI
        if existing_rxcui:
            print(f"  [{inn:30s}] EXISTING rxnorm_cui={existing_rxcui}")
            results.append({
                "drug_id": drug_id, "inn": inn, "rxcui": existing_rxcui,
                "source": "existing", "name": inn, "status": "existing"
            })
            skipped += 1
            continue

        # Check manual overrides first
        if inn_lower in MANUAL_RXCUI:
            rxcui, note = MANUAL_RXCUI[inn_lower]
            if rxcui:
                print(f"  [{inn:30s}] MANUAL   rxcui={rxcui} ({note})")
                results.append({
                    "drug_id": drug_id, "inn": inn, "rxcui": rxcui,
                    "source": "manual", "name": note, "status": "found"
                })
                found += 1
            else:
                print(f"  [{inn:30s}] SKIP     {note}")
                results.append({
                    "drug_id": drug_id, "inn": inn, "rxcui": None,
                    "source": "manual", "name": note, "status": "not_found"
                })
                not_found += 1
            continue

        # Try RxNorm API with INN
        time.sleep(API_DELAY)
        result = rxnorm_find_rxcui(inn)
        if result:
            print(f"  [{inn:30s}] FOUND    rxcui={result['rxcui']} ({result['name']})")
            results.append({
                "drug_id": drug_id, "inn": inn, "rxcui": result["rxcui"],
                "source": "rxnorm_api", "name": result["name"], "status": "found"
            })
            found += 1
            continue

        # Try with first brand name as fallback
        brands = [b.strip() for b in brand_names.split(",") if b.strip()]
        brand_found = False
        for brand in brands[:3]:  # Try up to 3 brand names
            time.sleep(API_DELAY)
            result = rxnorm_find_rxcui(brand)
            if result:
                print(f"  [{inn:30s}] BRAND    rxcui={result['rxcui']} via '{brand}' ({result['name']})")
                results.append({
                    "drug_id": drug_id, "inn": inn, "rxcui": result["rxcui"],
                    "source": f"brand:{brand}", "name": result["name"], "status": "found"
                })
                found += 1
                brand_found = True
                break
        if brand_found:
            continue

        # Not found
        print(f"  [{inn:30s}] NOT FOUND")
        results.append({
            "drug_id": drug_id, "inn": inn, "rxcui": None,
            "source": "none", "name": "", "status": "not_found"
        })
        not_found += 1

    print("-" * 70)
    print(f"\n  Summary: {found} found, {skipped} existing, {not_found} not found "
          f"(total: {len(drugs)})")

    # ── Step 3: UMLS Cross-References (optional) ──
    umls_data = {}
    if umls_mode:
        print(f"\nStep 3: UMLS cross-references...")
        print("-" * 70)

        # Test API key first
        test = umls_search("metformin")
        if test:
            print(f"  UMLS API key verified (metformin -> CUI {test['cui']})")
        else:
            print("  WARNING: UMLS API key may be invalid. Skipping cross-references.")
            umls_mode = False

        if umls_mode:
            for r in results:
                if r["status"] == "not_found" or not r["rxcui"]:
                    continue
                inn = r["inn"]
                time.sleep(API_DELAY)

                # Find UMLS CUI for this drug
                umls_result = umls_search(inn, source="RXNORM")
                if not umls_result:
                    # Try without source filter
                    time.sleep(API_DELAY)
                    umls_result = umls_search(inn)

                if umls_result:
                    cui = umls_result["cui"]
                    print(f"  [{inn:30s}] UMLS CUI={cui}")

                    # Get MeSH cross-reference
                    time.sleep(API_DELAY)
                    mesh_codes = umls_get_crosswalk(cui, "MSH")

                    # Get MedDRA cross-reference
                    time.sleep(API_DELAY)
                    meddra_codes = umls_get_crosswalk(cui, "MDR")

                    umls_data[r["drug_id"]] = {
                        "umls_cui": cui,
                        "mesh_codes": mesh_codes,
                        "meddra_codes": meddra_codes,
                    }

                    if mesh_codes:
                        print(f"    MeSH:   {mesh_codes[0]['code']} ({mesh_codes[0]['name']})")
                    if meddra_codes:
                        print(f"    MedDRA: {meddra_codes[0]['code']} ({meddra_codes[0]['name']})")
                else:
                    print(f"  [{inn:30s}] UMLS not found")

        print("-" * 70)
        print(f"  UMLS enriched: {len(umls_data)} drugs with cross-references")

    # ── Step 4: Write to DB ──
    if write_mode:
        print(f"\nStep 4: Writing to Azure SQL...")
        print("-" * 70)

        updated = 0
        for r in results:
            if r["status"] == "found" and r["rxcui"]:
                try:
                    cursor.execute(
                        "UPDATE drugs SET rxnorm_cui = ?, updated_at = GETUTCDATE() WHERE drug_id = ?",
                        (r["rxcui"], r["drug_id"])
                    )
                    print(f"  [{r['inn']:30s}] SET rxnorm_cui = {r['rxcui']}")
                    updated += 1
                except Exception as e:
                    print(f"  [{r['inn']:30s}] ERROR: {e}")

        conn.commit()
        print("-" * 70)
        print(f"  Updated {updated} drugs in DB")

        # Verify
        cursor.execute("SELECT COUNT(*) FROM drugs WHERE rxnorm_cui IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"  Verification: {count} drugs now have rxnorm_cui")
    else:
        print(f"\nStep 4: SKIPPED (dry-run mode — use --write to update DB)")

    # ── Step 5: Summary Report ──
    print(f"\n{'=' * 70}")
    print("ENRICHMENT REPORT")
    print(f"{'=' * 70}")
    print(f"{'Drug INN':30s} {'RXCUI':>10s}  {'Source':15s}  {'RxNorm Name'}")
    print(f"{'-'*30} {'-'*10}  {'-'*15}  {'-'*30}")
    for r in sorted(results, key=lambda x: x["inn"]):
        rxcui = r["rxcui"] or "—"
        print(f"{r['inn']:30s} {rxcui:>10s}  {r['source']:15s}  {r['name']}")

    if umls_data:
        print(f"\n{'=' * 70}")
        print("UMLS CROSS-REFERENCES")
        print(f"{'=' * 70}")
        for drug_id, xref in umls_data.items():
            inn = next((r["inn"] for r in results if r["drug_id"] == drug_id), "?")
            print(f"\n  {inn} (UMLS CUI: {xref['umls_cui']})")
            if xref["mesh_codes"]:
                for m in xref["mesh_codes"][:3]:
                    print(f"    MeSH:   {m['code']} — {m['name']}")
            if xref["meddra_codes"]:
                for m in xref["meddra_codes"][:3]:
                    print(f"    MedDRA: {m['code']} — {m['name']}")

    print(f"\n{'=' * 70}")
    print("Done!")
    if not write_mode:
        print("NOTE: This was a dry-run. Use --write to update the database.")
    print(f"{'=' * 70}")

    # Cleanup
    try:
        cursor.close()
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
