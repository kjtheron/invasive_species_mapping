set shell := ["bash", "-cu"]
set dotenv-load := true

default: lint test

install:
    uv sync

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .

test:
    uv run pytest -x

type:
    uv run pyright src/cmrv

# --- Stage 1: AOI ---
aoi-fetch:
    uv run cmrv aoi-fetch

# Test AOI via HydroBASINS (fallback while DWS REST is down / WR2012 pending)
aoi-hydrobasins:
    uv run cmrv aoi-hydrobasins

tiles:
    uv run cmrv aoi-tiles

# --- Stage 4: Labels ---
gbif-resolve:
    uv run cmrv labels-gbif-resolve

vegmap-ingest:
    uv run cmrv labels-vegmap-ingest

labels-ingest:
    uv run cmrv labels-ingest

labels-ingest-gbif:
    uv run cmrv labels-ingest --source gbif

labels-merge:
    uv run cmrv labels-merge

# VIZ ONLY — produces per-tile sparse label COGs for QGIS / Streamlit overlay.
# Training does NOT consume these rasters; the chip / point regime reads
# the observation store directly via ingest-chips + make-split.
labels-fuse:
    uv run cmrv labels-fuse

# Print species × spatial × temporal stats for the chip manifest.
# Source of truth for "what's in my chips" — no schema, no class_map.
chips-stats:
    uv run cmrv chips-stats
