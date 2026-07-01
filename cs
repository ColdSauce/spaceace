#!/usr/bin/env bash
# cs — claude-sandbox wrapper
#
# Drop this in your project root, `chmod +x cs`, and you should never need to
# `docker exec` or `docker run` directly again.

set -euo pipefail

IMAGE="claude-sandbox"
CONTAINER="claude-sandbox-runner"
VOLUME="claude-config"
WORKSPACE="${PWD}"

# Args used for any one-shot helper invocation (auth, bd, shell)
oneshot_args=(
    --rm
    -v "${WORKSPACE}:/workspace"
    -v "${VOLUME}:/home/node/.claude"
    -e CLAUDE_CONFIG_DIR=/home/node/.claude
)

is_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"
}

require_image() {
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo "Image '${IMAGE}' not found. Run: cs build" >&2
        exit 1
    fi
}

cmd_build() {
    docker build -t "${IMAGE}" .
    echo "Built ${IMAGE}."
}

cmd_auth() {
    require_image
    echo "One-time auth. A URL will appear; open it in your browser."
    echo "After signing in, type /exit and press Enter to finish."
    echo
    docker run -it "${oneshot_args[@]}" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- claude
    echo "Done."
}

cmd_init() {
    require_image
    docker run --rm -v "${WORKSPACE}:/workspace" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bd init --quiet
    echo "Initialized .beads/ in ${WORKSPACE}"
}

cmd_add() {
    require_image
    docker run --rm -v "${WORKSPACE}:/workspace" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bd create "$@"
}

cmd_ready() {
    require_image
    docker run --rm -v "${WORKSPACE}:/workspace" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bd ready
}

cmd_bd() {
    require_image
    docker run --rm -v "${WORKSPACE}:/workspace" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bd "$@"
}

cmd_status() {
    require_image
    if is_running; then
        echo "Ralph: RUNNING (${CONTAINER})"
    else
        echo "Ralph: stopped"
    fi
    echo "---"
    docker run --rm -v "${WORKSPACE}:/workspace" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bd stats 2>/dev/null || echo "(no beads stats — run 'cs init')"
}

cmd_start() {
    require_image
    if is_running; then
        echo "Already running. Stop first with: cs stop"
        exit 1
    fi
    if [[ ! -f "${WORKSPACE}/PROMPT.md" ]]; then
        echo "Missing PROMPT.md in ${WORKSPACE}." >&2
        echo "Create one describing what the agent should do each iteration." >&2
        exit 1
    fi
    if [[ ! -d "${WORKSPACE}/.beads" ]]; then
        echo "No .beads/ in ${WORKSPACE}. Run: cs init" >&2
        exit 1
    fi
    docker run -d \
        --name "${CONTAINER}" \
        -v "${WORKSPACE}:/workspace" \
        -v "${VOLUME}:/home/node/.claude" \
        -e CLAUDE_CONFIG_DIR=/home/node/.claude \
        ${MAX_ITER:+-e MAX_ITER=${MAX_ITER}} \
        --restart unless-stopped \
        "${IMAGE}" >/dev/null
    echo "Started ${CONTAINER}."
    echo "Tail logs:  cs logs -f"
    echo "Stop:       cs stop"
}

cmd_logs() {
    if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
        echo "No container named ${CONTAINER}. Start one with: cs start" >&2
        exit 1
    fi
    if [[ "${1:-}" == "-f" ]]; then
        docker logs -f --tail 100 "${CONTAINER}"
    else
        docker logs --tail 200 "${CONTAINER}"
    fi
}

cmd_stop() {
    docker stop "${CONTAINER}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER}" >/dev/null 2>&1 || true
    echo "Stopped."
}

cmd_shell() {
    require_image
    docker run -it "${oneshot_args[@]}" \
        --entrypoint /usr/bin/tini \
        "${IMAGE}" -- bash
}

usage() {
    cat <<EOF
Usage: cs <command>

Setup (run once per project):
  build                  Build the docker image
  auth                   One-time Claude Code OAuth (interactive)
  init                   Initialize beads in this project

Backlog management:
  add "title" [flags]    Create a bead (passes flags to 'bd create')
  ready                  Show unblocked work
  status                 Beads stats + ralph state
  bd <args...>           Run any 'bd' command in the container

Run the loop:
  start                  Start Ralph in the background
  logs [-f]              Show ralph logs (-f to follow)
  stop                   Stop the ralph container

Escape hatch:
  shell                  Drop into a bash session (you said you'd never use this)

Env vars:
  MAX_ITER=N cs start    Cap the loop at N iterations (default: unlimited)
EOF
}

case "${1:-}" in
    build)   shift; cmd_build "$@" ;;
    auth)    shift; cmd_auth "$@" ;;
    init)    shift; cmd_init "$@" ;;
    add)     shift; cmd_add "$@" ;;
    ready)   cmd_ready ;;
    bd)      shift; cmd_bd "$@" ;;
    status)  cmd_status ;;
    start)   cmd_start ;;
    logs)    shift; cmd_logs "$@" ;;
    stop)    cmd_stop ;;
    shell)   cmd_shell ;;
    -h|--help|help|"") usage ;;
    *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
