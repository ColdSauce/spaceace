# Agent instructions

**Read first:** `CLAUDE.md` (build, hard rules, workflow) and
`docs/SOLVER.md` (AI design rationale, failure modes already hit, the
diagnostic loop, open problems). Do not re-learn documented lessons.

## Hard rules
- No wall clipping in saved ghosts (solve strict — the default).
- No human-ghost data as search guidance; human ghosts are benchmarks only.
- Never modify `src/real_*.rs` (engine): saved tapes depend on exact floats.
- Validate any tape on `PyGameInstance` before saving; quality gate is
  `bash tests/smoke.sh` (exercises level 7, not level 0).

## Task tracking
Use `bd` (beads) for all task tracking. Do not use Markdown plan files.

- `bd ready` — what's actionable
- `bd create "title" -t task -p 1` — file new work
- `bd show <id>` / `bd update <id> --status in_progress` / `bd close <id>`
- `bd --json` — machine-readable output
