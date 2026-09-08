#!/usr/bin/env bash
#
# Run `cmrv ingest-chips` inside a nightly window, then stop cleanly.
#
# Cron starts this once a night. It works out how long until the stop time and
# hands that to `timeout`, so it needs no second cron job to stop the run — and
# it still stops at the right moment if cron fires late or you start it by hand.
#
# Stopping is a plain SIGTERM, which `cmrv ingest-chips` turns into the same
# graceful save Ctrl+C does: queued work is cancelled, chips already in flight
# are given ~30 s to finish and record themselves, and anything still on disk
# without a manifest row is adopted on the next start rather than downloaded
# again. Nothing is lost by being interrupted.
#
# A stop takes up to about two minutes: the drain window, then the reads that
# were already open unwinding. Expect the log to run a little past the stop time.
#
# Install (once):
#     crontab -e
#     30 22 * * * /home/kjtheron/Projects/invasive_species_mapping/scripts/chip_window.sh
#
# Run it by hand to test, with a short window:
#     ./scripts/chip_window.sh "+2 minutes"
#
# Override the concurrency for one run without editing the file:
#     WORKERS=12 READ_POOL=4 ./scripts/chip_window.sh "+20 minutes"
#
set -uo pipefail

PROJECT="${PROJECT:-/home/kjtheron/Projects/invasive_species_mapping}"
STOP_AT="${1:-${STOP_AT:-05:00}}"    # when to stop; anything `date -d` understands
# 20 x 4 = 80 concurrent streams. Both are passed explicitly rather than left to
# pipeline.yaml, so editing the config cannot silently change what cron runs.
#
# Measured on this link (2026-09-08), all four configurations tried:
#    32 streams (8x4)    0.079 obs/s   50% of the 40 Mb line
#    80 streams (20x4)   0.087 obs/s   74%          <- best
#   160 streams (20x8)   0.077 obs/s   reads stall
#   160 streams (40x4)   0.077 obs/s   reads stall
# The stall threshold follows the TOTAL stream count, not how it is split
# between workers and read pool. Below 80 the link starves; above it, streams
# stall and each stall costs a full re-screen plus re-download. Do not raise
# either number without re-measuring both the rate and the read_error count.
WORKERS="${WORKERS:-20}"
READ_POOL="${READ_POOL:-4}"
LOG="${LOG:-$PROJECT/logs/chip_window.log}"
LOCK="$PROJECT/.chip_window.lock"
DONE_MARK="$PROJECT/data/chips/train/.all_chipped"

# cron gives almost no PATH, so name uv's directory explicitly.
export PATH="/home/kjtheron/.local/bin:/usr/local/bin:/usr/bin:/bin"

cd "$PROJECT" || exit 1
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

say() { echo "[$(date '+%F %T')] $*"; }

# Only one window at a time. Without this, a run that overruns its window would
# be joined by tomorrow's, and both would fight for the same bandwidth.
exec 9>"$LOCK"
if ! flock -n 9; then
    say "another chip window is still running — skipping tonight"
    exit 0
fi

# Refuse to start alongside a chipper someone launched by hand. Two runs share
# one link and one manifest, so the second only halves the first. flock covers
# two *windows*; this covers a window meeting a manual run.
#
# Match on the process NAME as well as the command line. `pgrep -f` alone also
# matches any shell whose command line happens to contain the text — an editor,
# a `grep`, a stale wrapper — and a false positive here means cron skips the
# night and says nothing. The real process reports comm `cmrv` or `uv`.
chipper_running() {
    ps -eo comm=,args= |
        awk '$1 != "bash" && $1 != "sh" && /bin\/cmrv ingest-chips/ { found = 1 }
             END { exit !found }'
}

if chipper_running; then
    say "a cmrv ingest-chips run is already going — leaving it alone"
    exit 0
fi

if [ -f "$DONE_MARK" ]; then
    say "all chips are done ($DONE_MARK exists) — nothing to do."
    say "Delete that file to force another pass, e.g. after adding labels."
    exit 0
fi

# Seconds until the next stop time. Computed now rather than hard-coded, so a
# late cron start still stops at 05:00 instead of running into the morning.
now=$(date +%s)
stop=$(date -d "$STOP_AT" +%s 2>/dev/null) || { say "bad STOP_AT: $STOP_AT"; exit 1; }
[ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
secs=$(( stop - now ))

before=$(find data/chips/train -name '*.tif' 2>/dev/null | wc -l)
say "starting: $WORKERS workers x $READ_POOL reads = $((WORKERS * READ_POOL)) streams, stopping in $((secs / 3600))h$(( (secs % 3600) / 60 ))m at $(date -d "@$stop" '+%F %T'). Chips now: $before"

# --signal=TERM is the graceful path. --kill-after is a backstop for a process
# that ignores it; it is safe because write_parquet_df renames into place, so a
# hard kill leaves the previous manifest rather than a truncated one.
# Run it in the background and wait, so a TERM aimed at THIS script is passed
# on. Without the trap, killing the wrapper left the chipper running as an
# orphan that would then fight the next window.
timeout --signal=TERM --kill-after=5m "$secs" \
    uv run cmrv ingest-chips --max-workers "$WORKERS" --read-pool "$READ_POOL" &
child=$!
trap 'say "signal received — passing it to the chip run"; kill -TERM "$child" 2>/dev/null' TERM INT HUP
wait "$child"
rc=$?
trap - TERM INT HUP

after=$(find data/chips/train -name '*.tif' 2>/dev/null | wc -l)
say "stopped (exit $rc). Chips now: $after (+$((after - before)) this window)"

case $rc in
    0)
        # The run finished on its own, so there was nothing left to chip.
        touch "$DONE_MARK"
        say "ingest-chips completed — all chips downloaded. Marker written."
        say "Remove the crontab entry, or delete $DONE_MARK to run again."
        ;;
    124|137)
        say "window ended; work was saved. Next window continues where this left off."
        ;;
    *)
        say "exited $rc — check the cmrv log in logs/ for the reason."
        ;;
esac
