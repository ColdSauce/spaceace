#!/usr/bin/env bash
# Ralph Wiggum loop: restart Claude on every iteration so context is fresh.
# Beads holds the durable state across iterations.
#
# All output goes to stdout — `docker logs` captures it; `cs logs -f` tails it.

set -u

ITER=0
MAX_ITER="${MAX_ITER:-0}"           # 0 = forever
PROMPT_FILE="${PROMPT_FILE:-/workspace/PROMPT.md}"
IDLE_SLEEP="${IDLE_SLEEP:-60}"      # seconds to wait when no ready work
CRASH_SLEEP="${CRASH_SLEEP:-5}"     # seconds to wait after non-zero exit

log() {
    echo "[ralph $(date -u +%FT%TZ)] $*"
}

if [[ ! -f "$PROMPT_FILE" ]]; then
    log "FATAL: $PROMPT_FILE not found. Create one in your project root."
    exit 1
fi

cd /workspace || { log "FATAL: cannot cd to /workspace"; exit 1; }

if [[ ! -d .beads ]]; then
    log "No .beads/ directory yet — run 'cs init' on the host first."
    exit 1
fi

log "Starting Ralph loop. MAX_ITER=$MAX_ITER, PROMPT_FILE=$PROMPT_FILE"

while :; do
    ITER=$((ITER + 1))
    log "=== iteration $ITER ==="

    # Skip the iteration if there's no ready work — saves Claude tokens.
    READY_COUNT="$(bd ready --json 2>/dev/null | jq 'length' 2>/dev/null || echo 0)"
    if [[ "$READY_COUNT" == "0" ]]; then
        log "No ready beads. Sleeping ${IDLE_SLEEP}s..."
        sleep "$IDLE_SLEEP"
        continue
    fi
    log "Ready beads: $READY_COUNT"

    # Run Claude. stream-json gives structured logs for postmortems.
    if claude -p "$(cat "$PROMPT_FILE")" \
         --dangerously-skip-permissions \
         --output-format stream-json \
         --verbose; then
        log "Claude exited 0"
    else
        rc=$?
        log "Claude exited $rc, sleeping ${CRASH_SLEEP}s before retry"
        sleep "$CRASH_SLEEP"
    fi

    if [[ "$MAX_ITER" -gt 0 && "$ITER" -ge "$MAX_ITER" ]]; then
        log "Hit MAX_ITER=$MAX_ITER, stopping."
        break
    fi
done
