#!/usr/bin/env bash
# Measure chips/second at several "workers x read_pool" settings.
#
#   ./scripts/concurrency_probe.sh 16x8 32x8 32x16 64x8
#
# Each probe restarts the chipper, waits WARMUP seconds for the pools to fill,
# then counts chips written over WINDOW seconds. No work is wasted — chips made
# during a probe are real chips, and ingest-chips is incremental.
#
# ponytail: counts files by mtime rather than parsing the log. The log format
# can change; a .tif on disk cannot.
set -euo pipefail

REPO=${REPO:-$HOME/invasive_species_mapping}
CHIPS=$REPO/data/chips/train
WARMUP=${WARMUP:-180}
WINDOW=${WINDOW:-600}
PAT="bin/cmrv ingest-chips"

count() { find "$CHIPS" -name '*.tif' | wc -l; }

stop() {
  pkill -f "$PAT" 2>/dev/null || true
  # ingest-chips drains open reads on SIGTERM; give it up to 2 min
  for _ in $(seq 60); do pgrep -f "$PAT" >/dev/null || return 0; sleep 2; done
  echo "WARN: chipper still running after 120 s" >&2
}

probe() {
  local w=$1 r=$2
  local log=/tmp/probe_${w}x${r}.log
  stop
  ( cd "$REPO" && uv run cmrv ingest-chips --max-workers "$w" --read-pool "$r" >"$log" 2>&1 ) &
  sleep "$WARMUP"
  local before after n retries aborts
  before=$(count); sleep "$WINDOW"; after=$(count)
  n=$((after - before))
  retries=$(grep -c "STAC call failed" "$log" || true)
  aborts=$(grep -c "Aborting load" "$log" || true)
  awk -v w="$w" -v r="$r" -v n="$n" -v s="$WINDOW" \
      -v ld="$(cut -d' ' -f1 /proc/loadavg)" -v rt="$retries" -v ab="$aborts" \
      'BEGIN{printf "%3sx%-3s  %5d chips / %ds = %.3f obs/s   load %-5s  stac_retries %-4s  read_aborts %s\n",
             w, r, n, s, n/s, ld, rt, ab}'
}

trap stop EXIT
printf "warmup %ss, window %ss, %s cores\n\n" "$WARMUP" "$WINDOW" "$(nproc)"
for pair in "$@"; do probe "${pair%%x*}" "${pair##*x}"; done
