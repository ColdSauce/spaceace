#!/usr/bin/env bash
# Smoke test: engine, solver, and agents. Exercises level 7 (the real
# workload), not just level 0.

set -euo pipefail
cd "$(dirname "$0")/.."

green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }

run() {
    local label="$1"; shift
    printf '  %-44s ' "$label"
    if out=$("$@" 2>&1); then
        green "OK"
    else
        red "FAIL"
        echo "$out" | tail -20
        exit 1
    fi
}

echo "=== SpaceAce smoke tests ==="

run "random agent, level 0" \
    uv run python run.py --agent random --level 0 --headless --episodes 2 --max-steps 200 --no-save-ghost

run "ace tape replays exactly on level 7" \
    uv run python scripts/smoke_replay.py

run "solver beam finds a completing tape (level 7)" \
    uv run python scripts/smoke_solve.py

run "tas agent replays sidecar, level 7" \
    uv run python run.py --agent tas --level 7 --headless --episodes 1 \
        --max-steps 4000 --tas-label tas --tas-validate --no-save-ghost

run "ace agent, level 7" \
    uv run python run.py --agent ace --level 7 --headless --episodes 1 \
        --max-steps 4000 --no-save-ghost

run "unit tests" uv run pytest tests/ -q

green "=== all smoke tests passed ==="
