#!/usr/bin/env bash
# Run bench_train_throughput.py in TRAIN mode across all presets and print
# a ranked comparison. Use from the project root:
#
#     scripts/bench_sweep.sh          # all presets, 25k steps each
#     STEPS=60000 scripts/bench_sweep.sh
#     PRESETS="new dummy small_net" scripts/bench_sweep.sh
#
# Each run is independent (fresh python process per preset) so CPU/heap state
# doesn't carry over.

set -u

STEPS="${STEPS:-25000}"
PRESETS="${PRESETS:-default pre_m1 dummy small_net more_envs}"
LEVELS="${LEVELS:-3000,3001,3002}"
MAX_EP_STEPS="${MAX_EP_STEPS:-500}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESULTS=$(mktemp -t spaceace_sweep.XXXXXX.tsv)
printf "preset\twall_s\tfps\n" > "$RESULTS"

for P in $PRESETS; do
    echo
    echo "================ PRESET: $P ================"
    LOG=$(mktemp -t spaceace_sweep_$P.XXXXXX.log)
    if ! uv run python scripts/bench_train_throughput.py train \
            --preset "$P" --steps "$STEPS" \
            --levels "$LEVELS" --max-episode-steps "$MAX_EP_STEPS" \
            2>&1 | tee "$LOG"; then
        echo "  ($P failed — see $LOG)"
        printf "%s\tFAIL\tFAIL\n" "$P" >> "$RESULTS"
        continue
    fi
    WALL=$(grep -E "^Wall time:" "$LOG" | tail -1 | awk '{print $3}' | tr -d 's')
    FPS=$(grep -E "^FPS:" "$LOG" | tail -1 | awk '{print $2}' | tr -d ',')
    printf "%s\t%s\t%s\n" "$P" "$WALL" "$FPS" >> "$RESULTS"
done

echo
echo "==============================================="
echo "SWEEP RESULTS (steps=$STEPS, levels=$LEVELS)"
echo "==============================================="
column -t -s $'\t' "$RESULTS"

# Rank by FPS desc (skip FAILs and header).
echo
echo "Ranked by FPS:"
tail -n +2 "$RESULTS" | grep -v FAIL | sort -t $'\t' -k3 -rn | awk -F'\t' \
    'BEGIN{best=0}
    { if (best==0) best=$3; ratio=($3/best); printf "  %-10s %8s FPS  (%.2fx)\n", $1, $3, ratio }'

echo
echo "Raw TSV: $RESULTS"
