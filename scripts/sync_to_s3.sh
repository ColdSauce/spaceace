#!/usr/bin/env bash
# Background loop that syncs training artifacts to the RunPod S3 network volume.
# Run alongside training (e.g. in a tmux pane) so checkpoints survive pod destruction.
set -e

cd "$(dirname "$0")/.."

BUCKET="s3://60n3qskdnh/spaceace"
REGION="eu-cz-1"
ENDPOINT="https://s3api-eu-cz-1.runpod.io"
INTERVAL="${SYNC_INTERVAL:-300}"

while true; do
  echo "[sync_to_s3] $(date -u +%FT%TZ) syncing..."
  aws s3 sync ./models "$BUCKET/models" \
    --region "$REGION" --endpoint-url "$ENDPOINT" --no-progress
  aws s3 sync ./tensorboard_logs "$BUCKET/tensorboard_logs" \
    --region "$REGION" --endpoint-url "$ENDPOINT" --no-progress
  sleep "$INTERVAL"
done
