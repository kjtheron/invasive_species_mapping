"""Audit chip coverage: manifest vs GCS bucket.

Run with:
    uv run python scripts/audit_chips.py [--no-list]

--no-list  : skip the slow full-bucket listing (saves ~100s); shows only
             manifest + shard stats.
"""
import sys
import time

import gcsfs
import pandas as pd

NO_LIST = "--no-list" in sys.argv

PREFIX = "ism-data/chips/train"
MANIFEST_URI = f"gs://{PREFIX}/manifest.parquet"
SHARDS_GLOB  = f"{PREFIX}/_manifest_shards/*.parquet"

fs = gcsfs.GCSFileSystem()


# ---------------------------------------------------------------------------
# 1. Load manifest.parquet (end-of-last-run truth)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
try:
    with fs.open(MANIFEST_URI, "rb") as f:
        manifest = pd.read_parquet(f)
except FileNotFoundError:
    manifest = pd.DataFrame()
manifest_s = time.perf_counter() - t0
print(f"manifest.parquet : {len(manifest):>8,} rows  ({manifest_s:.2f}s)")


# ---------------------------------------------------------------------------
# 2. Load in-flight shards (written by currently-running or crashed run)
# ---------------------------------------------------------------------------
t1 = time.perf_counter()
shard_files = fs.glob(SHARDS_GLOB)
shard_frames = []
for s in shard_files:
    try:
        with fs.open(s, "rb") as f:
            shard_frames.append(pd.read_parquet(f))
    except Exception:
        pass
shards = pd.concat(shard_frames, ignore_index=True) if shard_frames else pd.DataFrame()
shard_s = time.perf_counter() - t1
print(f"_manifest_shards : {len(shards):>8,} rows  ({shard_s:.2f}s)  [{len(shard_files)} shard files]")


# ---------------------------------------------------------------------------
# 3. Combined truth (manifest + shards, deduplicated)
# ---------------------------------------------------------------------------
if not shards.empty:
    combined = pd.concat([manifest, shards], ignore_index=True)
    combined = combined.drop_duplicates(subset=["obs_id", "month_label"], keep="last")
else:
    combined = manifest.copy()

print(f"combined         : {len(combined):>8,} rows  ({combined['obs_id'].nunique():,} unique obs_ids)")


# ---------------------------------------------------------------------------
# 4. Optional full-bucket listing + cross-check
# ---------------------------------------------------------------------------
if NO_LIST:
    print("\n(skipped bucket listing — pass without --no-list to enable)")
else:
    print(f"\nListing gs://{PREFIX}/**/*.tif  (this takes ~100s) ...")
    t2 = time.perf_counter()
    all_files = set(fs.glob(f"{PREFIX}/**/*.tif"))
    listing_s = time.perf_counter() - t2
    print(f"  {len(all_files):,} chip files in {listing_s:.1f}s")

    combined_uris = set(combined["chip_uri"].str.replace("gs://", "", regex=False))
    in_manifest_not_bucket = combined_uris - all_files
    in_bucket_not_manifest = all_files - combined_uris

    print(f"  In combined manifest but MISSING from bucket : {len(in_manifest_not_bucket):,}")
    if in_manifest_not_bucket:
        for u in sorted(in_manifest_not_bucket)[:5]:
            print(f"    gs://{u}")

    print(f"  In bucket but NOT in combined manifest      : {len(in_bucket_not_manifest):,}")
    if in_bucket_not_manifest:
        print("  (these are truly orphaned — chips with no manifest record)")
        for u in sorted(in_bucket_not_manifest)[:5]:
            print(f"    gs://{u}")


# ---------------------------------------------------------------------------
# 5. Month coverage across combined manifest
# ---------------------------------------------------------------------------
print("\nMonth coverage (combined):")
months_per_obs = combined.groupby("obs_id")["month_label"].nunique()
print(months_per_obs.value_counts().sort_index().rename("obs_ids").to_string())
incomplete = months_per_obs[months_per_obs < 4]
print(f"  obs_ids with <4 months: {len(incomplete):,}  "
      f"({100*len(incomplete)/len(months_per_obs):.1f}% of total)")


# ---------------------------------------------------------------------------
# 6. Source breakdown
# ---------------------------------------------------------------------------
print("\nSource breakdown (combined, unique obs_ids):")
sources = combined.drop_duplicates("obs_id").copy()
sources["source"] = sources["obs_id"].str.split(":").str[0]
print(sources.groupby("source").size().sort_values(ascending=False).to_string())


# ---------------------------------------------------------------------------
# 7. Top IAP species
# ---------------------------------------------------------------------------
print("\nTop IAP species (unique obs_ids, combined manifest):")
iap = combined[~combined["obs_id"].str.startswith("nlc")]
sp_counts = iap.groupby("species")["obs_id"].nunique().sort_values(ascending=False)
print(sp_counts.head(20).to_string())
