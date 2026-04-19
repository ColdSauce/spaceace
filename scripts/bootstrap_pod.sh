#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod GPU pod.
#
# Assumes the RunPod network volume is mounted at $REPO_DIR and already
# contains the repo (you push it from your laptop via the S3 API).
#
# Usage on the pod:
#   bash bootstrap_pod.sh
#
# Override defaults if needed:
#   REPO_DIR=/runpod-volume/spaceace bash bootstrap_pod.sh
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/spaceace}"

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bootstrap ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$REPO_DIR" ]] || die "REPO_DIR=$REPO_DIR does not exist — did the network volume mount?"
[[ -f "$REPO_DIR/pyproject.toml" ]] || die "$REPO_DIR/pyproject.toml not found — is the repo synced to the volume?"

# ---------- 1. system packages ----------
log "installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  curl ca-certificates git tmux \
  build-essential pkg-config libssl-dev

# ---------- 2. Rust toolchain ----------
if ! command -v cargo >/dev/null 2>&1; then
  log "installing Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"

# ---------- 3. uv ----------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------- 4. uv: put venv + cache on LOCAL pod disk, not the network volume ----------
# The RunPod network volume is FUSE-mounted and doesn't support hardlinks /
# rapid rename ops, which breaks `uv sync` ("stale file handle"). Source stays
# on the volume; venv lives on local disk and gets rebuilt on each fresh pod.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/.venvs/spaceace}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_LINK_MODE=copy
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$UV_CACHE_DIR"

# Nuke any half-built venv that landed on the network volume from a prior run
if [[ -d "$REPO_DIR/.venv" ]]; then
  log "removing stale .venv on network volume"
  rm -rf "$REPO_DIR/.venv"
fi

# ---------- 5. persist env for future shells ----------
if ! grep -q 'spaceace-bootstrap' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF

# spaceace-bootstrap
. "\$HOME/.cargo/env"
export PATH="\$HOME/.local/bin:\$PATH"
export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_LINK_MODE=copy
EOF
fi

# ---------- 6. build Rust extension ----------
cd "$REPO_DIR"
log "building Rust extension (uv sync --reinstall-package spaceace-rl) -> $UV_PROJECT_ENVIRONMENT"
uv sync --reinstall-package spaceace-rl

# ---------- 6. GPU smoke test ----------
log "GPU smoke test"
uv run python - <<'PY'
import torch
print("torch version :", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name   :", torch.cuda.get_device_name(0))
    print("device count  :", torch.cuda.device_count())
PY

# ---------- 7. import smoke test ----------
log "import smoke test"
uv run python -c "import spaceace_rl; print('spaceace_rl OK')"

cat <<EOF

\033[1;32m[bootstrap] done\033[0m

Next steps (in tmux panes):

  tmux new -s train
  cd ${REPO_DIR}

  # PPO
  uv run python train.py --level 0

  # AlphaZero
  uv run python -m spaceace.agents.alphazero.train

  # tensorboard (if pod exposes a port)
  uv run tensorboard --logdir tensorboard_logs --host 0.0.0.0 --port 6006

Models and tensorboard_logs write directly to the network volume — no sync needed.
EOF
