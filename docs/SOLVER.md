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
4. **px rank** (the original heart — most early failures lived here; see
   Lessons): `rank = (1-mix)·h_now + mix·h_stop + doom + turn_pen +
   jitter`, where `h_stop` is h at the ballistic-parabola-projected
   position (wall-clamped per segment, with fly-through pickup credit at
   45px), `doom` is a graded escape-feasibility penalty
   (min(stop-model, arc-model), 6×deficit capped at 2500 — doom_scale 2.0
   restores cautious-but-reliable completion for fresh globals), and
   `turn_pen` charges misaligned pickup arrival in time-calibrated px
   (253·misalign + 0.875·waste).
5. **Time lattice** (`TimeLattice`, `lattice=true` — the 2026-07 upgrade
   that broke the 21s plateau): a velocity-aware time-to-go value
   function V_S(20px cell, 32 velocity headings, 11 speed bands,
   pro/retro posture) per remaining-pickup subset, built in ~1-3s
   by backward Dijkstra over motion-primitive edges (cruise / gravity-
   adjusted accel & brake / constant-speed turn arcs with swept
   clearance / 0.72s posture flips that drift v·0.72s). Touching a
   pickup seeds V_S(n) = V_{S\p}(n) at the same velocity node, so
   arrival momentum carries across legs and the free terminal dash
   emerges naturally. Rank = V·380px + jitter; off-lattice states fall
   back to 30000 + h_px. Corner speed limits, flip commitment distances,
   and pickup-order-by-time all price physically. The lattice's spawn
   ETA prints at build time — a useful per-level sanity probe.
6. **Selection** — mask groups get merit-weighted quotas
   (exp(-(h_best_group - h_best_global)/350), retain gate at ≥2 groups on
   doom-free h); within a group, survivors are drawn round-robin across
   (128px cell × 3 speed band) buckets (≤ cell_strat_m each, protection
   gated to group_best_rank + 900) before rank fills the rest. This keeps
   wide corner entries and slow-careful vs fast-needle pacings alive for
   the 20-40 ticks until their payoff — beams die from monocultures, not
   from missing width.
7. **Anytime refinement** — `refine` (corridor + warm start), `polish`
   (exact local search on the tape; move set includes net-rotation-
   preserving pair deletes, pair strips, near transpositions and thrust
   retiming), warm `resolve_suffix` (re-plan tails), `resolve_window_exact`
   (phase-advanced window beam plus exhaustive exact suffix rollout), and
   `resolve_prefix` (approximate rendezvous diagnostics).
8. **Driver** (`scripts/solve.py`) — portfolio of global solves (lattice
   first, then px, then px-safe doom_scale=2.0) → improvement loop
   cycling lattice/px global and tube refines, exact-continuation windows,
   polish, and warm suffix re-solves → validate on the engine → save sidecar
   + dashboard ghosts (only when strictly faster). `--order 2,1,0` forces a
   pickup order (diagnostics); `--no-lattice` disables the lattice rank.

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
    (see the "diagnostic loop" below; `scripts/analyze_tape.py` is that
    loop as a tool — per-leg splits, speed profile, slow sections, and a
    speed-colored trajectory render).
13. **The px rank could not represent the elite line at all** (2026-07
    diagnosis). Stop-model doom flagged 21.9% of the user's own surviving
    21.86s ghost as dead (it modeled stop-in-place; experts arc);
    straight-segment projection sagged 50px below the true parabola over
    a 1s horizon (more than the capture radius, so fly-through credit
    rewarded aim that missed); LOS clamping killed momentum credit
    exactly at corners; and distance-px as a currency cannot see that a
    2400px terminal dive is nearly free while a 370px bowl exit against
    gravity is not. Fix: rank in (approximate) SECONDS via the time
    lattice; keep the px rank as fallback and for n > 4 levels.
14. **Selection monoculture, not width, was the binding constraint.**
    Reachable quantized cells outnumber any practical width 5-30×, so
    top-k-by-rank keeps tens of thousands of ±6px micro-variants of one
    line. Whole-beam extinctions (everyone commits to the same doomed
    dive) are pace monocultures. Fix: bucketed round-robin protection
    (cell × speed band) — and two bugs to never re-introduce: capping
    buckets with truncate() capped the WHOLE beam at cells×m
    (near-extinction at 432 states); ungated round-robin handed protected
    slots to dead/off-lattice buckets ("walking dead") at the same
    priority as viable lines.
15. **Under-sampled swept paths poison the lattice.** Turn-arc templates
    sampled at a fixed 7 points tunneled through thin walls at high
    bands (65-100% of band-7+ turn edges on L7 were wall-crossing but
    marked valid), silently injecting ~5s of optimism into V (spawn ETA
    17.6s → 22.7s after the fix). Sample swept paths at ~12px spacing,
    always.
16. **`radius=1e9` "global" refines never ran before 2026-07.** The
    corridor mark loop dilated by radius/cell ≈ 4×10⁷ cells per tape
    position — an effectively infinite loop. Every driver run that hit a
    global-refine stage silently hung there (this is why historical runs
    produced no round lines and why timeout-killed sweeps never saved
    sidecars). Clamped now (whole-map dilation = grid diameter). The
    first actually-executed warm global lattice refine took L7
    26.58s → 20.03s in ONE round.
17. **Exact continuation beats approximate splicing.** A shortened local
    trajectory usually cannot match an incumbent state closely enough for a
    long chaotic suffix; tolerance-based rendezvous candidates diverge. In
    `resolve_window_exact`, the beam instead tracks the incumbent one tick
    ahead across a bounded window, then skips terminal rank/dedup and executes
    the untouched suffix from every final expanded state. Only a full strict
    completion is accepted. Repeated 30k-250k windows reduced L7 from 19.583s
    to 19.217s, including gains in the opening, pendulum, and final slalom.
    Warm suffix re-solves likewise inject their known-valid reference at every
    layer; the former blind suffix beam could prune its only proven route.

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

- **The remaining 1.217s to the reported 18s record.** The current saved L7
  tape is 1153 ticks / 19.217s, strict and validated on `PyGameInstance`.
  No 18s action tape is available locally, so its collision behavior and the
  true strict floor are not established; do not turn the engine's >316px/s
  alternate-tick collision quirk into search guidance or a saved ghost.
  Measured remaining slack includes the 292-tick P1→P2 pendulum (267px/s
  mean, 337px/s max) and a 0.75s sub-150px/s section in the final slalom.
- Exact-continuation windows mitigate front ossification but still need a
  viable suffix basin. Long early windows remain expensive and can stall; a
  more principled global prefix improver is still open.
- Lattice fidelity: 20px cells / 11.25° headings / band quantization make V
  pessimistic in tight slaloms. Speed changes and turns remain separate
  primitives, while whole-arc clearance can reject feasible compound flight.
  Any compound primitive must sample and price its snapped endpoint; an
  unchecked rounded endpoint can poison every upstream value.

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
