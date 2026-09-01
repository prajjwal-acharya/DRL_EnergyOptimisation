#!/bin/bash
# Sequencer: run the Week-4 forecasting mission to completion via the detached
# conductor daemon, verify it, and only then start the Week-5 RQ1 mission.
# Proper sequence is enforced: Week 5 never starts unless Week 4 is completed
# AND its phase gate (/Volumes/code/rp/automation/check_week4.sh) exits 0.
#
# Usage (detached):
#   cd "/Volumes/code/Research Project"
#   nohup ./conductor/chain_week4_then_week5.sh >> ./conductor/chain.log 2>&1 &
#
# Stop everything:
#   conductor daemon stop <week4-id> ; conductor daemon stop <week5-id>
#   pkill -f chain_week4_then_week5.sh

set -u

MISSIONS_DIR="/Volumes/code/Research Project/.agent-conductor/missions"
W4="2026-08-26-week-4-probabilistic-demand-solar-fo-4050fc"
W5="2026-08-26-week-5-uncertainty-aware-ppo-point-v-e45c06"
CHECK_W4="/Volumes/code/rp/automation/check_week4.sh"
POLL_SECONDS=60

log() { echo "[chain $(date '+%Y-%m-%dT%H:%M:%S')] $*"; }

mission_status() {
  /usr/bin/python3 - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get("status", "unknown"))
except FileNotFoundError:
    print("missing")
PY
}

cd "/Volumes/code/Research Project" || { log "FATAL: cannot cd to repo root"; exit 1; }

# --- Week 4 ---------------------------------------------------------------
W4_STATUS=$(mission_status "$MISSIONS_DIR/$W4/state.json")
if [ "$W4_STATUS" = "completed" ]; then
  log "Week-4 mission already completed; skipping its daemon."
else
  log "Starting Week-4 daemon ($W4)."
  conductor daemon start "$W4" --retry-failures 10 --retry-delay 10 >> /dev/stderr || {
    log "FATAL: conductor daemon start failed for Week 4"; exit 1; }
  log "Week-4 daemon started; polling every ${POLL_SECONDS}s until completion."
  while true; do
    sleep "$POLL_SECONDS"
    W4_STATUS=$(mission_status "$MISSIONS_DIR/$W4/state.json")
    case "$W4_STATUS" in
      completed) break ;;
      failed|blocked)
        log "Week-4 mission status is '$W4_STATUS' - NOT starting Week 5."
        log "Inspect: conductor status $W4 ; conductor daemon logs $W4"
        exit 1 ;;
      *)
        # Every few minutes, note that the mission is alive and progressing.
        log "Week-4 status: $W4_STATUS (continuing to wait)" ;;
    esac
  done
  log "Week-4 mission completed."
fi

# --- Gate: Week-4 phase verifier must pass before Week 5 may start --------
if ! "$CHECK_W4" > /dev/null 2>&1; then
  log "FATAL: $CHECK_W4 did not exit 0 although Week 4 reports completed."
  log "Week 5 NOT started. Run it manually to see the failing check."
  exit 1
fi
log "Week-4 phase gate passed ($CHECK_W4 exit 0)."

# --- Week 5 ---------------------------------------------------------------
W5_STATUS=$(mission_status "$MISSIONS_DIR/$W5/state.json")
if [ "$W5_STATUS" = "completed" ]; then
  log "Week-5 mission already completed; nothing to do."
  exit 0
fi
log "Starting Week-5 daemon ($W5)."
conductor daemon start "$W5" --retry-failures 10 --retry-delay 10 >> /dev/stderr || {
  log "FATAL: conductor daemon start failed for Week 5"; exit 1; }
log "Week-5 daemon started; sequencer done (monitor with: conductor watch $W5)."
