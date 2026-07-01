# Agent instructions

Use `bd` (beads) for all task tracking. Do not use Markdown plan files.

- `bd ready` — what's actionable
- `bd create "title" -t task -p 1` — file new work
- `bd show <id>` — read details
- `bd update <id> --status in_progress`
- `bd close <id> --reason "..."`
- `bd --json` — machine-readable output for parsing

Project overview, build commands, and solver architecture: see `CLAUDE.md`
and `README.md`. Quality gate: `bash tests/smoke.sh` (exercises level 7).
