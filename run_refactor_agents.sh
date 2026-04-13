#!/usr/bin/env bash
# Launch 4 Claude Code agents in parallel tmux panes, one per refactor plan.
# Each gets its own git worktree so they don't conflict.
#
# Usage: ./run_refactor_agents.sh

set -euo pipefail

SESSION="refactor"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with first pane (Plan 4 — smallest, no deps)
tmux new-session -d -s "$SESSION" -n agents \
  "claude -w pathfinder-backend 'Wire pathfinder backend arg. See docs/refactor-plans/04-wire-pathfinder-backend-arg.md. Run tests/smoke.sh when done.'; read"

# Split into 4 panes (2x2 grid)
tmux split-window -t "$SESSION" -h \
  "claude -w replace-bfs 'Replace BFS in generate_maps. See docs/refactor-plans/03-replace-generate-maps-bfs.md. Run tests/smoke.sh when done.'; read"

tmux split-window -t "$SESSION".0 -v \
  "claude -w fold-curriculum 'Fold curriculum_train into Sb3Trainer. See docs/refactor-plans/01-fold-curriculum-train.md. Run tests/smoke.sh when done.'; read"

tmux split-window -t "$SESSION".1 -v \
  "claude -w port-trainers 'Port AlphaZero+HRL to Trainer subclasses. See docs/refactor-plans/02-port-alphazero-hrl-trainers.md. Run tests/smoke.sh when done.'; read"

# Even out the layout
tmux select-layout -t "$SESSION" tiled

# Label panes
tmux select-pane -t "$SESSION".0 -T "Plan 4: pathfinder backend"
tmux select-pane -t "$SESSION".1 -T "Plan 3: generate_maps BFS"
tmux select-pane -t "$SESSION".2 -T "Plan 1: curriculum"
tmux select-pane -t "$SESSION".3 -T "Plan 2: AZ/HRL trainers"

tmux set -t "$SESSION" pane-border-status top

# Attach
tmux attach -t "$SESSION"
