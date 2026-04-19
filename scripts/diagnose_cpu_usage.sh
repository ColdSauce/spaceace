#!/usr/bin/env bash
# Diagnose SpaceAce training CPU utilization.
#
# Two modes:
#   (A) If a curriculum_train process is already running, profile it.
#   (B) Otherwise, launch a long-running bench and profile that.
#
# Usage:
#     scripts/diagnose_cpu_usage.sh           # auto-detect
#     scripts/diagnose_cpu_usage.sh --bench   # force a fresh bench
#
# Requires: ps, top, pgrep (built-in). py-spy optional (install with
# `uv pip install py-spy` for richer tracebacks).

set -u

SAMPLE_SECS="${SAMPLE_SECS:-8}"
FORCE_BENCH=0
if [ "${1:-}" = "--bench" ]; then FORCE_BENCH=1; fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAIN_PID=""
LAUNCHED=0
LOG=""

if [ "$FORCE_BENCH" = "0" ]; then
    # Prefer the *real* python process. `uv run` spawns a wrapper at 0% CPU
    # whose child is the actual interpreter — descend if we grabbed the wrapper.
    for pid in $(pgrep -f "spaceace.agents.ppo.curriculum_train"); do
        cmd=$(ps -o command= -p "$pid" 2>/dev/null || true)
        case "$cmd" in
            *"/.venv/bin/python"*) MAIN_PID="$pid"; break;;
        esac
    done
    if [ -z "$MAIN_PID" ]; then
        MAIN_PID=$(pgrep -f "spaceace.agents.ppo.curriculum_train" | head -1 || true)
    fi
fi

if [ -z "$MAIN_PID" ]; then
    LOG=$(mktemp -t spaceace_diag.XXXXXX.log)
    echo "No live training found — launching benchmark (300k steps, ~12s)..."
    echo "Log: $LOG"
    uv run python scripts/bench_train_throughput.py env \
        --preset new --n-envs 6 --steps 300000 \
        > "$LOG" 2>&1 &
    WRAPPER_PID=$!
    LAUNCHED=1
    trap '[ -n "$WRAPPER_PID" ] && kill "$WRAPPER_PID" 2>/dev/null; pkill -P "$WRAPPER_PID" 2>/dev/null; exit 0' EXIT INT TERM
    echo "uv wrapper PID: $WRAPPER_PID"
    echo "Giving workers ${SAMPLE_SECS}s to come up..."
    sleep 3
    # `uv run` spawns the real python as its child. Descend to the real process.
    REAL=$(pgrep -P "$WRAPPER_PID" | head -1 || true)
    if [ -n "$REAL" ]; then
        MAIN_PID="$REAL"
        echo "Real python PID: $MAIN_PID"
    else
        MAIN_PID="$WRAPPER_PID"
    fi
else
    echo "Found live training: PID $MAIN_PID"
    sleep 3
fi

# Sample CPU usage three times over SAMPLE_SECS and average.
echo
echo "=== Sampling CPU for ${SAMPLE_SECS}s (3 samples) ==="
for i in 1 2 3; do
    SAMPLE_FILE=$(mktemp)
    ps -axo pid,ppid,%cpu,rss,command > "$SAMPLE_FILE"

    WORKERS=$(pgrep -P "$MAIN_PID" || true)
    MAIN_CPU=$(awk -v p="$MAIN_PID" '$1==p {print $3}' "$SAMPLE_FILE")
    WORKER_CPU_TOTAL=0
    WORKER_COUNT=0
    for w in $WORKERS; do
        C=$(awk -v p="$w" '$1==p {print $3}' "$SAMPLE_FILE")
        if [ -n "$C" ]; then
            WORKER_CPU_TOTAL=$(awk -v a="$WORKER_CPU_TOTAL" -v b="$C" 'BEGIN{printf "%.1f", a+b}')
            WORKER_COUNT=$((WORKER_COUNT+1))
        fi
    done
    printf "  sample %d: main=%5s%%  workers=%d  workers_total=%s%%\n" \
        "$i" "${MAIN_CPU:-0}" "$WORKER_COUNT" "$WORKER_CPU_TOTAL"
    rm -f "$SAMPLE_FILE"
    sleep $((SAMPLE_SECS / 3))
done

echo
echo "=== Process tree (main + workers) ==="
printf "%-8s %-8s %-6s %-10s %s\n" PID PPID %CPU RSS_MB COMMAND
printf "%-8s %-8s %-6s %-10s %s\n" "$MAIN_PID" \
    "$(ps -o ppid= -p "$MAIN_PID" | tr -d ' ')" \
    "$(ps -o %cpu= -p "$MAIN_PID" | tr -d ' ')" \
    "$(ps -o rss= -p "$MAIN_PID" | awk '{printf "%.0f", $1/1024}')" \
    "$(ps -o command= -p "$MAIN_PID" | cut -c1-80)"
for w in $(pgrep -P "$MAIN_PID"); do
    printf "%-8s %-8s %-6s %-10s %s\n" "$w" \
        "$MAIN_PID" \
        "$(ps -o %cpu= -p "$w" | tr -d ' ')" \
        "$(ps -o rss= -p "$w" | awk '{printf "%.0f", $1/1024}')" \
        "$(ps -o command= -p "$w" | cut -c1-80)"
done

echo
echo "=== System-wide CPU snapshot ==="
top -l 1 -n 0 | grep -E "CPU usage|Load Avg"

# Total cores available — macOS specific.
CORES=$(sysctl -n hw.ncpu)
PERF_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "$CORES")
echo "Physical perf cores: $PERF_CORES  (total logical: $CORES)"
echo "One core saturated = 100% in ps/top. Full machine = ${CORES}00%."

# py-spy deeper dive if available.
if command -v py-spy >/dev/null 2>&1; then
    echo
    echo "=== py-spy dump of main process (PID $MAIN_PID) ==="
    py-spy dump --pid "$MAIN_PID" 2>&1 | head -40 || echo "(py-spy dump failed — may need sudo on macOS)"
else
    echo
    echo "Tip: install py-spy for live profiling:  uv pip install py-spy"
    echo "     then:  sudo py-spy top --pid $MAIN_PID"
fi

echo
echo "=== Interpretation ==="
if [ "$LAUNCHED" = "1" ]; then
    echo "This was the bench (random actions, no PPO update)."
    echo "If workers are pinned near 100% and main is low, the env side is healthy."
    echo "The real training run may look different because the main process has"
    echo "to run the policy forward pass + gradient updates."
else
    echo "If main_cpu ≈ 100% and each worker is <30%, the bottleneck is the"
    echo "MAIN process (policy inference + PPO update), not env stepping."
    echo "Fixes to try:"
    echo "  1. Switch to DummyVecEnv for n_envs <= 8 (skip IPC overhead)."
    echo "     Edit sb3_trainer.py _make_curriculum_vec_env to use DummyVecEnv."
    echo "  2. Reduce action_repeat (currently 5) — more policy calls per env frame,"
    echo "     but if policy is cheap, net gain from avoiding long inference gaps."
    echo "  3. Profile with py-spy top --pid $MAIN_PID to confirm."
fi

if [ "$LAUNCHED" = "1" ]; then
    echo
    echo "Bench log tail:"
    tail -10 "$LOG" 2>/dev/null || true
fi
