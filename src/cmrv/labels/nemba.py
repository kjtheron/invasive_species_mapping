"""Extract and resolve the NEMBA AIS plant list from Gazette 43726 (2020).

Sources
-------
Primary:   ``data/labels/nemba/nemba_species_list_2020.pdf``  (Gazette 43726, 18 Sep 2020)
Secondary: ``data/labels/nemba/nemba_list_2020.pdf``  (Gazette 43735, 25 Sep 2020 — Regulations)

The species PDF is typeset in mirrored/RTL layout: each character within every
line is reversed, and lines within a table cell are in reverse vertical order.
``_clean_species`` and ``_clean_common`` restore the correct reading order.

List 1 (terrestrial + fresh-water plants) spans pages 7-29; List 2 (marine
plants) is page 30; List 3+ are non-plant taxa. We extract Lists 1 and 2
so the store is complete; callers can filter on ``list_number``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from cmrv.io import read_parquet_df, write_parquet_df

NEMBA_PRIMARY_PDF = Path("data/labels/nemba/nemba_species_list_2020.pdf")
NEMBA_OUT_LOCAL = Path("data/labels/nemba_plants.parquet")
NEMBA_CATEGORIES = frozenset({"1a", "1b", "2", "3"})

# Gazette page numbers (1-based) where each list begins.
_LIST_MARKER_RE = re.compile(r":(\d+)\ntsiL")  # matches ":N\ntsiL" → "List N:" reversed
_TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


# ---------------------------------------------------------------------------
# Cell-level text restoration
# ---------------------------------------------------------------------------


def _clean_species(raw: str | None) -> str:
    """Restore a SPECIES cell: reverse line order, reverse chars per line, merge split genus."""
    if not raw:
        return ""
    lines = [ln[::-1] for ln in reversed(raw.split("\n")) if ln.strip()]
    s = " ".join(lines)
    # Merge PDF line-split genus: 'Ac acia' → 'Acacia', etc.
    # Limit to ≤12 combined chars to avoid false positives like 'Acacia varieties'.
    s = re.sub(
        r"\b([A-Z][a-z]{1,3})\s+([a-z]{3,8})\b",
        lambda m: (
            m.group(1) + m.group(2) if len(m.group(1)) + len(m.group(2)) <= 12 else m.group(0)
        ),
        s,
    )
    return re.sub(r"\s+", " ", s).strip()


def _clean_common(raw: str | None) -> str:
    """Restore a COMMON NAME cell: reverse chars per line, keep line order."""
    if not raw:
        return ""
    lines = [ln[::-1] for ln in raw.split("\n") if ln.strip()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _clean_hdr(raw: str | None) -> str:
    """Decode a header cell (chars reversed, single or multi-line)."""
    if not raw:
        return ""
    return " ".join(ln[::-1] for ln in raw.split("\n") if ln.strip())


def _clean_cat(raw: str | None) -> str:
    """Decode a category cell (single token, chars reversed)."""
    if not raw:
        return ""
    return (raw.strip())[::-1]


# ---------------------------------------------------------------------------
# Per-page list-number detection
# ---------------------------------------------------------------------------


def _page_list_number(raw_text: str) -> int | None:
    """Return the List N that starts on this page, or None if no new list begins.

    List markers are stored as reversed text: ':01\ntsiL' = 'List 10:'.
    Reversing the digit group recovers the correct integer.
    """
    m = _LIST_MARKER_RE.search(raw_text)
    if m:
        return int(m.group(1)[::-1])  # '01' → '10' → 10
    return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_nemba_plants(
    pdf_path: str | Path = NEMBA_PRIMARY_PDF,
    include_lists: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    """Parse NEMBA AIS species lists from the gazette PDF.

    Returns a DataFrame with columns:
        scientific_name (str), common_name (str), nemba_category (str),
        list_number (int), page (int), gazette_ref (str), list_year (int).

    ``include_lists=(1,)`` restricts to terrestrial/fresh-water plants.
    ``include_lists=(1, 2)`` also includes the marine plant list.
    """
    try:
        import pdfplumber  # optional dep; listed in pyproject.toml
    except ImportError as exc:
        raise ImportError("pdfplumber required: run `uv add pdfplumber`") from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"NEMBA PDF not found: {pdf_path}")

    records: list[dict] = []
    current_list: int | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for pg_idx, page in enumerate(pdf.pages):
            raw_txt = page.extract_text() or ""

            new_list = _page_list_number(raw_txt)
            if new_list is not None:
                current_list = new_list
                logger.debug("p{} → List {}", pg_idx + 1, current_list)

            if current_list not in include_lists:
                continue

            tbl = page.extract_table(_TABLE_SETTINGS)
            if not tbl:
                continue

            species_row = common_row = cat_row = None
            for row in tbl:
                if not row:
                    continue
                hdr = _clean_hdr(row[0])
                if "SPECIES" in hdr.upper():
                    species_row = row
                elif "COMMON" in hdr.upper():
                    common_row = row
                elif "CATEGORY" in hdr.upper():
                    cat_row = row

            if species_row is None or cat_row is None:
                continue

            for col_i in range(1, len(species_row)):
                sp = _clean_species(species_row[col_i])
                cat = _clean_cat(cat_row[col_i])
                common = _clean_common(common_row[col_i] if common_row else None)
                if not sp:
                    continue
                cat_m = re.search(r"(1a|1b|2|3)", cat)
                if not cat_m:
                    continue
                records.append(
                    {
                        "scientific_name": sp,
                        "common_name": common,
                        "nemba_category": cat_m.group(1),
                        "list_number": current_list,
                        "page": pg_idx + 1,
                        "gazette_ref": "GG 43726 (2020-09-18)",
                        "list_year": 2020,
                    }
                )

    df = pd.DataFrame(records).drop_duplicates(
        subset=["scientific_name", "nemba_category"], keep="first"
    )
    logger.info(
        "extracted {} unique NEMBA plant entries from {} (Lists {})",
        len(df),
        pdf_path.name,
        sorted(include_lists),
    )
    return df


def write_nemba_plants(
    pdf_path: str | Path = NEMBA_PRIMARY_PDF,
    out_path: str | Path = NEMBA_OUT_LOCAL,
    include_lists: tuple[int, ...] = (1, 2),
) -> Path:
    """Extract + write NEMBA plant list to a local parquet file."""
    df = extract_nemba_plants(pdf_path=pdf_path, include_lists=include_lists)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_df(df, str(out))
    logger.success("wrote {} rows → {}", len(df), out)
    return out


# ---------------------------------------------------------------------------
# GBIF backbone resolution
# ---------------------------------------------------------------------------

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
NEMBA_RESOLVED_GCS = "gs://ism-data/labels/nemba_taxa_resolved.parquet"


def _extract_query_name(scientific_name: str) -> str:
    """Extract a clean query name: strip authority, merge known PDF split patterns."""
    tokens = scientific_name.split()
    genus_done = False
    epithet_done = False
    out = []
    for tok in tokens:
        if not out:
            out.append(tok)  # genus always first
            genus_done = True
            continue
        if genus_done and not epithet_done:
            if tok[0].islower() and not tok.endswith(".") and "(" not in tok:
                out.append(tok)
                epithet_done = True
                continue
            if tok.lower() in ("subsp.", "var.", "f.", "cv."):
                continue
            break
        break
    return " ".join(out)


def _gbif_match(name: str, retries: int = 3) -> dict | None:
    """Call /species/match with kingdom=Plantae; return JSON or None on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(
                GBIF_MATCH_URL,
                params={"name": name, "kingdom": "Plantae", "strict": "false"},
                timeout=30,
            )
            r.raise_for_status()
            j = r.json()
            if j.get("usageKey") and j.get("matchType") != "NONE":
                return j
            return None
        except Exception as exc:
            logger.debug("GBIF match attempt {}/{} for {!r}: {}", attempt + 1, retries, name, exc)
            if attempt < retries - 1:
                time.sleep(2**attempt)
    return None


def resolve_nemba_taxa(
    plants_parquet: str | Path = NEMBA_OUT_LOCAL,
    out_uri: str = NEMBA_RESOLVED_GCS,
    pause_s: float = 0.3,
) -> pd.DataFrame:
    """Resolve each NEMBA plant scientific name to a GBIF backbone usage key.

    Reads ``plants_parquet`` (output of ``extract_nemba_plants``), calls GBIF
    /species/match per entry, writes resolved rows to ``out_uri``.

    Output columns:
        scientific_name (str), common_name (str), nemba_category (str),
        list_number (int), gbif_usage_key (i64, nullable),
        canonical_name (str, nullable), match_type (str, nullable),
        confidence (i32, nullable), gazette_ref (str), list_year (int).
    """
    df = read_parquet_df(str(plants_parquet))
    rows = []
    n_resolved = 0
    for rec in df.to_dict("records"):
        query = _extract_query_name(rec["scientific_name"])
        hit = _gbif_match(query)
        if hit:
            n_resolved += 1
            rows.append(
                {
                    **rec,
                    "query_name": query,
                    "gbif_usage_key": int(hit["usageKey"]),
                    "canonical_name": hit.get("scientificName") or hit.get("canonicalName"),
                    "match_type": hit.get("matchType", "UNKNOWN"),
                    "confidence": int(hit.get("confidence") or 0),
                }
            )
            logger.info(
                "resolved {!r:<40} → key={} ({})",
                query,
                hit["usageKey"],
                hit.get("matchType"),
            )
        else:
            logger.warning("unresolved: {!r} (query={!r})", rec["scientific_name"], query)
            rows.append(
                {
                    **rec,
                    "query_name": query,
                    "gbif_usage_key": None,
                    "canonical_name": None,
                    "match_type": None,
                    "confidence": None,
                }
            )
        time.sleep(pause_s)

    out_df = pd.DataFrame(rows)
    rate = n_resolved / max(len(rows), 1) * 100
    logger.info(
        "GBIF resolution: {}/{} ({:.1f}%) taxa matched",
        n_resolved,
        len(rows),
        rate,
    )
    if rate < 95.0:
        logger.warning(
            "resolution rate {:.1f}% < 95% target — review unresolved entries in output",
            rate,
        )

    write_parquet_df(out_df, out_uri)
    logger.success("wrote {} rows → {}", len(out_df), out_uri)
    return out_df
