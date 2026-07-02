# The Ace Solver — design, rationale, and lessons learned

This document explains *why* the AI is built the way it is, every failure
mode we hit on the way, and how to keep improving it. Read this before
touching `src/solver.rs` or `scripts/solve.py`.

## Premise

SpaceAce is fully deterministic: fixed 60Hz tick, no RNG, exact float
physics. That makes "play at superhuman level" a **search problem, not a
learning problem**. Years of prior attempts (PPO, AlphaZero, MCTS, HRL —
all deleted in commit 27545f5) under-delivered because they sampled noisy
value estimates over a game where the exact outcome of any action sequence
is computable at ~6M steps/s/core. The solver plans a complete per-tick
action tape offline; the `ace` agent replays it.

## Ground rules (user rulings — do not violate)

1. **No wall clipping.** The engine skips collision checks every other
   frame above ~316 px/s (a faithful port of the original JS performance
   hack). Tapes that thread walls on skipped frames are engine-legal but
   were ruled out. The solver's `strict` mode (default) checks collision
   every tick; strict tapes replay identically on the real engine because
   the engine's checks are a subset of strict's. `--allow-clip` /
   `PySolver(level, strict=False)` exist for experiments only — never save
   clipping ghosts.
2. **No human-derived guidance.** Do not seed, corridor-restrict, or
   otherwise bias the search with human ghost data. The user wants to
   *learn from the beam*, which requires its lines to be independent.
   Human ghosts are benchmarks for diagnostics only: find where the human
   is faster, explain it as a physics/ranking phenomenon, fix the general
   mechanism.
3. **The engine is sacred.** `real_physics.rs` / `real_game.rs` /
   `real_collision.rs` must not change: every saved tape and ghost depends
   on exact float behavior. The solver's inline stepper must remain
   float-op-identical to the engine (same constants, same operation
   order). Any tape is validated on the untouched `PyGameInstance` before
   being saved.

## Architecture (src/solver.rs, one file)

1. **Exact stepper** — `SimState` (x, y, vx, vy, rot, skip-parity, pickup
   mask; 24 bytes) + `AceSolver::step`. Mirrors the engine exactly,
   including the collision-skip counter (parity is real state and is part
   of the dedup key).
2. **Tick-synchronized beam search** (`beam_impl`) — the whole frontier
   advances one tick at a time; each layer expands 6 actions per node,
   dedups by quantized (pos, vel, rot, mask, skip-parity), and keeps the
   best `width` by rank. Parent links (u32+u8 per node per layer) allow
   tape reconstruction; memory ≈ width × ticks × 5 bytes.
3. **Route heuristic** — per-pickup Dijkstra distance fields on a 10px
   grid (walls inflated 30px, with tighter fallbacks for narrow-corridor
   pickups), plus `rem[mask][p]`: an exact DP over pickup subsets giving
   the optimal remaining-tour length from each pickup. `h(x, y, mask)` =
   min over remaining pickups of field distance + optimal tour. This is a
   route lower bound in px.
4. **Rank** (the heart — most failures lived here; see Lessons):
   `rank = (1-mix)·h_now + mix·h_stop + doom + turn_pen + jitter`, where
   `h_stop` is h at the velocity-projected position (LOS-clamped, with
   fly-through pickup credit), `doom` is a physical can-it-still-brake
   penalty, and `turn_pen` charges misaligned arrival velocity at pickups.
5. **Anytime refinement** — `refine` (corridor + warm start), `polish`
   (exact local search on the tape), `resolve_suffix` (re-plan tails),
   `resolve_prefix` (rendezvous splicing; see Lessons for why it mostly
   fails).
6. **Driver** (`scripts/solve.py`) — portfolio of global solves →
   improvement loop cycling global/tube refines, polish, suffix re-solves
   → validate on the engine → save sidecar + dashboard ghosts (only when
   strictly faster).

## Lessons learned (chronological — each cost real debugging time)

1. **Beam cliff / ballistic mass extinction.** With a pure route-progress
   rank, the entire beam dives at max speed and dies simultaneously ~40
   ticks later; no survivors, no signal. Fix: the *doom* penalty — a
   physical model of whether the ship can still avoid the wall ahead
   (rotation time to retrograde + v²/2a braking run vs a raycast along
   velocity). Hard penalty (+3000), so doomed states never displace viable
   ones. Without doom, beams die; with it too strong, they crawl.
2. **Doom must be direction-aware.** Gravity assists braking upward
   (decel ≈ 500 px/s²) and fights it downward (≈ 300). A fixed 330
   over-braked climbs, exactly where expert lines carry speed.
3. **Mask stratification.** One fast cluster otherwise extinguishes all
   alternative route orderings/pacings. Selection gives each pickup-set
   its own share of the width; groups whose best rank trails the global
   best by >900px are dropped (a strictly worse ordering never recovers).
4. **Pessimistic distance fields beat optimistic ones.** Tight wall
   inflation (16px) lured the beam into converging dead-ends (the L7
   tower notch) where whole cohorts die; it can also flip the route DP
   through ship-impassable cracks. 30px inflation is mildly pessimistic
   and stable. (On L7 this was investigated as the suspected cause of a
   wrong route order — it wasn't; the maze pocket genuinely has no bottom
   entrance. Verify geometry before blaming the heuristic.)
5. **Never project the velocity reward through a wall** (LOS clamp). With
   thin partitions, the far side can be much closer to the goal; an
   unclamped projection literally rewards flying into walls.
6. **Fly-through credit — the single biggest win.** h measures distance
   *to* a pickup, so a fast fly-through projects past it and looks like
   overshoot; the rank then rewards braking to a stop at every pickup. If
   the projected segment passes within the collection radius, evaluate the
   projection with that pickup collected. Took L7 from 21.80 → 21.05
   (pre-strict) in one round after weeks-equivalent of plateau.
7. **Final-dash exemption.** Crashing after the last pickup is free
   (explosion is checked before pickup collection each tick, so the
   collecting tick itself must be clean — but everything after is
   irrelevant). No doom for projections that complete the level; the
   solver ends levels with full-throttle suicide dashes, as it should.
8. **Turnaround penalty.** The beam favors whoever *reaches* a pickup
   first; a fast misaligned approach wins the beam long before its
   scramble-back cost is visible, and the well-aligned approaches are
   pruned dozens of layers earlier. Charge `v_wasted²/2a` at projected
   collection, where wasted = velocity not pointing down the next route
   leg (next target from the subset DP; direction from the field
   gradient).
9. **Warm-started refinement is the workhorse.** Inject the incumbent
   tape's state into every beam layer: the search provably cannot do
   worse, and any locally faster deviation wins. "Global" refine
   (radius=∞) is a full re-search with a safety net — this is what
   actually improves finished tapes. Corridor (tube) refines with finer
   quantization grind out the rest. Because the reference node cannot
   die, refinement can run `doom_scale` near 0 and let raw physics prune.
10. **Suffix re-solves fix ends, nothing fixes beginnings.** Improvements
    to early segments must re-earn the entire remaining route inside the
    beam before adoption, so finished tapes ossify from the front.
    Rendezvous splicing (`resolve_prefix`: reach the tape's state at tick
    r earlier within tolerance, splice, revalidate) was built for this —
    measured result: matches arrive only 1-2 ticks early and spliced
    suffixes diverge within ~100 ticks even at 10px/15px/s tolerance.
    The game is too chaotic for approximate-state splicing. Kept as a
    tool; don't expect miracles.
11. **Quantization is the exploration knob.** Coarse (6px/12px/s, 64 rot
    bins) for global solves — merges aggressively, reaches far. Fine
    (1.5-3px) inside corridors where width is concentrated. The dedup key
    must include collision-skip parity (real state even in strict mode).
12. **Beam telemetry or blindness.** `ACE_DEBUG=1` prints per-layer
    frontier stats (best h + position, doomed fraction, mean speed).
    Every major bug above was found by watching those lines, then
    `trace`/`h_at` probes and per-segment speed analysis of tapes
    (see the "diagnostic loop" below).

## The diagnostic loop (how to keep improving)

1. Get the current tape's story: per-segment ticks, mean/max/arrival
   speeds, distance traveled vs geodesic (a replay + numpy script; see
   git history for examples). Render the trajectory colored by speed.
2. Compare against physics limits (bang-bang time between waypoints) and
   against human ghosts *as benchmarks* — where is the beam slower, and
   is the constraint real (geometry) or self-inflicted (ranking)?
3. Explain the gap as a mechanism, fix it generically in the rank or the
   search, A/B on fresh solves (width 30k is a fast, meaningful probe),
   then let the driver's refinement loop consolidate.
4. Changes that alter `SimState` semantics or the stepper require
   `bash tests/smoke.sh` — the smoke suite checks tick-exact replay of
   the saved L7 sidecar against the engine.

## Known open problems

- **The elite line.** A top player reportedly ran L7 in ~18-19s on this
  build ("same route, just faster"). The solver's toolset converges at
  ~21s clean. The remaining gap concentrates in climb/turn segments that
  need needle-precise bang-bang control (full burn → one flip → full
  brake); tick-level beams merge/prune those needles. Tracked in beads:
  segment-level optimal-control search (parameterize burn/flip/brake
  switch times, exhaustively search the low-dimensional switch space,
  reconnect with a beam re-solve of the remainder).
- Beams ossify from the front (lesson 10) — a principled prefix improver
  is an open algorithmic problem.
- `polish` move set is basic (delete/overwrite/boundary-shift/insert);
  smarter moves (rotation-pair cancellation, thrust-pulse retiming) are
  unexplored.

## Practical notes

- Build: `export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/llvm/20.1.8/lib`
  first (Homebrew rustc links LLVM 20; `/opt/homebrew/opt/llvm` points at
  22). If cargo keeps reporting a stale dyld error, delete
  `target/.rustc_info.json`.
- A width-30k fresh solve of L7 takes ~20s; width 250k global refine
  ~3min; a full level pipeline ~30min. Memory ≈ width × ticks × 5 bytes
  for parent links — width 250k on a 4000-tick level is ~5GB; budget
  accordingly.
- Levels 0-7 have ≤17 pickups; the subset DP asserts n ≤ 20.
- Ghost bookkeeping: sidecars in `ghost_actions/L{n}_tas.json` (per-tick
  action indices; seconds = ticks/60); dashboard ghosts in
  `ghost_replays` under labels `tas` and `ai` (the web UI renders `ai` in
  magenta at `/play`). Both are only overwritten by strictly faster runs.
