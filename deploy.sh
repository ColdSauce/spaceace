#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# deploy.sh — rsync local repo to the Hetzner training box and rebuild.
#
# Usage:
#   ./deploy.sh                       # sync + rebuild Rust extension
#   ./deploy.sh --no-build            # sync only (skip `uv sync`)
#   ./deploy.sh --run                 # sync, rebuild, then run train.py
#   ./deploy.sh --run train.py --level 0
#   ./deploy.sh --run run.py --agent mcts --level 6 --headless
#   ./deploy.sh --watch               # continuous sync on file changes (no rebuild)
#   ./deploy.sh --shell               # exec into the training container
#   ./deploy.sh --tensorboard         # SSH tunnel for TensorBoard on :6006
# =============================================================================

REMOTE_HOST="root@65.108.96.172"
REMOTE_DIR="/data/training/project"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="rl-trainer"
WORKDIR="/workspace/project"

# Source and manifests the remote build/runtime need.
SYNC_PATHS=(
    Cargo.toml
    Cargo.lock
    pyproject.toml
    uv.lock
    src
    spaceace
    data
    scripts
    tests
    train.py
    run.py
    play.py
)

EXCLUDES=(
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '*.pyo'
    --exclude '.git'
    --exclude '.venv'
    --exclude 'venv'
    --exclude '.mypy_cache'
    --exclude '.pytest_cache'
    --exclude '.ruff_cache'
    --exclude '*.egg-info'
    --exclude '.DS_Store'
    --exclude 'target'          # Rust build artifacts — rebuilt remotely
    --exclude 'models'          # Checkpoints stay on the box
    --exclude 'tensorboard_logs'
    --exclude 'logs'
    --exclude 'screenshots'
    --exclude 'diagnostics'
    --exclude 'wandb'
    --exclude '*.swp'
    --exclude '*.swo'
)

sync_files() {
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
    local existing=()
    for p in "${SYNC_PATHS[@]}"; do
        [[ -e "$PROJECT_DIR/$p" ]] && existing+=("$p")
    done
    (
        cd "$PROJECT_DIR"
        rsync -avzR --delete \
            "${EXCLUDES[@]}" \
            "${existing[@]}" \
            "$REMOTE_HOST:$REMOTE_DIR/"
    )
    echo "Synced at $(date '+%H:%M:%S')"
}

remote_build() {
    echo "Rebuilding Rust extension in $CONTAINER ..."
    ssh "$REMOTE_HOST" "docker exec $CONTAINER bash -c 'cd $WORKDIR && uv sync --reinstall-package spaceace-rl'"
}

remote_run() {
    local script="${1:-train.py}"
    shift 2>/dev/null || true
    ssh -t "$REMOTE_HOST" "docker exec -it $CONTAINER bash -c 'cd $WORKDIR && uv run python $script $*'"
}

cmd_watch() {
    echo "Watching for changes (no rebuild on sync — run ./deploy.sh manually after Rust edits)"
    if command -v fswatch &>/dev/null; then
        fswatch -o "${SYNC_PATHS[@]/#/$PROJECT_DIR/}" | while read -r _; do
            sync_files
        done
    elif command -v inotifywait &>/dev/null; then
        while inotifywait -r -e modify,create,delete,move "${SYNC_PATHS[@]/#/$PROJECT_DIR/}"; do
            sync_files
        done
    else
        echo "ERROR: Install fswatch (macOS) or inotifywait (Linux: apt install inotify-tools)"
        exit 1
    fi
}

case "${1:-sync}" in
    --no-build)
        sync_files
        ;;
    --run)
        sync_files
        remote_build
        shift
        remote_run "$@"
        ;;
    --watch)
        sync_files
        cmd_watch
        ;;
    --shell)
        ssh -t "$REMOTE_HOST" "docker exec -it $CONTAINER bash -c 'cd $WORKDIR && exec bash'"
        ;;
    --tensorboard)
        echo "SSH tunnel: http://localhost:6006"
        ssh -N -L 6006:localhost:6006 "$REMOTE_HOST"
        ;;
    --help|-h)
        sed -n '4,16p' "$0"
        ;;
    *)
        sync_files
        remote_build
        ;;
esac
