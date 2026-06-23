"""Chip-manifest exploration: species × spatial × temporal distribution.

Single source of truth for "what's in my chips": reads ``manifest.parquet``
from a chip-extraction run and prints summary tables to stdout.  No schema,
no class_map, no AOI re-derivation — the manifest already reflects what
got chipped, and class assignment lives downstream at ``make-split`` time.
"""

from __future__ import annotations

from loguru import logger

from cmrv.io import read_parquet_df


def _fmt_int(n: int) -> str:
    return f"{n:>8,}"


def _print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def chip_stats(
    manifest_uri: str = "data/chips/train/manifest.parquet",
    top_species: int = 30,
    top_blocks: int = 10,
) -> None:
    """Print species × spatial × temporal stats from a chip manifest.

    Parameters
    ----------
    manifest_uri : str
        Path to ``manifest.parquet``.
    top_species : int
        Number of top species (by obs_id count) to list.
    top_blocks : int
        Number of densest spatial blocks to list per top-species summary.
    """
    logger.info("reading {}", manifest_uri)
    m = read_parquet_df(manifest_uri)

    n_chips = len(m)
    n_obs = m["obs_id"].nunique()
    n_species = m["species"].nunique()
    n_blocks = m["block_id"].nunique() if "block_id" in m.columns else 0
    has_fold = "fold" in m.columns

    _print_section("Manifest overview")
    print(f"  total chips:        {_fmt_int(n_chips)}")
    print(f"  unique obs_ids:     {_fmt_int(n_obs)}")
    print(f"  unique species:     {_fmt_int(n_species)}")
    print(f"  unique blocks:      {_fmt_int(n_blocks)}")
    if "x_utm" in m.columns and "y_utm" in m.columns:
        x0, x1 = m["x_utm"].min(), m["x_utm"].max()
        y0, y1 = m["y_utm"].min(), m["y_utm"].max()
        print(f"  spatial extent:     {(x1 - x0) / 1000:.0f} km × {(y1 - y0) / 1000:.0f} km (UTM)")

    # --- species ---
    species_obs = (
        m.groupby("species")
        .agg(
            n_obs_ids=("obs_id", "nunique"),
            n_chips=("obs_id", "size"),
            n_blocks=("block_id", "nunique") if "block_id" in m.columns else ("obs_id", "size"),
        )
        .sort_values("n_obs_ids", ascending=False)
    )
    species_obs["chips_per_obs"] = (species_obs["n_chips"] / species_obs["n_obs_ids"]).round(2)

    _print_section(f"Top {top_species} species (by unique obs_ids)")
    print(species_obs.head(top_species).to_string())

    cum10 = species_obs.head(10)["n_obs_ids"].sum()
    cum30 = species_obs.head(30)["n_obs_ids"].sum()
    print()
    print(f"  top-10 species cover  {cum10:>6,} / {n_obs:,} obs_ids ({100 * cum10 / n_obs:.1f}%)")
    print(f"  top-30 species cover  {cum30:>6,} / {n_obs:,} obs_ids ({100 * cum30 / n_obs:.1f}%)")

    long_tail = (species_obs["n_obs_ids"] < 50).sum()
    print(f"  species with <50 obs: {long_tail:>6,} (will be data-poor for training)")

    # --- temporal: month coverage per obs_id ---
    if "month_label" in m.columns:
        months_per_obs = m.groupby("obs_id")["month_label"].nunique()
        month_dist = months_per_obs.value_counts().sort_index()
        _print_section("Month-completeness (chips per obs_id)")
        for n_months, count in month_dist.items():
            print(f"  {n_months} month(s):  {_fmt_int(int(count))} obs_ids")
        complete = int((months_per_obs == months_per_obs.max()).sum())
        print(
            f"  fully covered ({months_per_obs.max()} months): "
            f"{complete:,} obs_ids ({100 * complete / n_obs:.1f}%)"
        )

    # --- spatial: per-block top species (concentration check) ---
    if "block_id" in m.columns:
        per_block = m.groupby("block_id")["obs_id"].nunique().sort_values(ascending=False)
        _print_section(f"Top {top_blocks} densest blocks (by obs_id count)")
        print(per_block.head(top_blocks).to_string())

        # Single-block dominance: which species have >50% of obs_ids in one block?
        sp_block = m.groupby(["species", "block_id"])["obs_id"].nunique().reset_index(name="n")
        sp_total = sp_block.groupby("species")["n"].sum()
        sp_max = sp_block.groupby("species")["n"].max()
        dom = (sp_max / sp_total).rename("max_frac_one_block")
        dominated = dom[(dom >= 0.5) & (sp_total >= 50)].sort_values(ascending=False)
        if len(dominated):
            _print_section(
                "Spatially-dominated species (≥50 obs_ids, >50% in one block) — spatial split risk"
            )
            print(dominated.head(20).to_string())

    # --- folds (only if make-split has run) ---
    if has_fold:
        fold_counts = m.groupby(["fold", "species"])["obs_id"].nunique().unstack(fill_value=0)
        _print_section("Fold × species (top 15 species)")
        top15 = species_obs.head(15).index
        present = [c for c in top15 if c in fold_counts.columns]
        print(fold_counts[present].T.to_string())
    else:
        print()
        print("  (no fold column — run `cmrv make-split` to add splits)")

    # --- year distribution ---
    if "year" in m.columns:
        year_dist = m.groupby("year")["obs_id"].nunique().sort_index()
        _print_section("Obs_ids per chip year")
        for yr, count in year_dist.items():
            print(f"  {yr}:  {_fmt_int(int(count))} obs_ids")
