# SpaceAce Mobile — proof of concept

A touch-first HTML5 port of SpaceAce that runs on any phone. Completely
self-contained in this folder — **nothing in the original game is touched**;
level data is *copied* from `data/spaceace_levels.json` into `levels.js`.

## Try it

```bash
cd mobile_poc
python3 -m http.server 8080
# phone on same network → http://<your-ip>:8080
# or desktop: http://localhost:8080 (arrows/WASD + space, mouse works too)
```

Add to Home Screen on iOS/Android for a fullscreen app-like experience
(the page sets the PWA-style meta tags).

## What "SpaceAce on mobile" looks like

The desktop game is a tick-precision keyboard game about momentum. Mobile
can't be about frame-perfect inputs. After trying direct-control retrofits
(see the alternate modes below), the POC's answer is a **structural**
change: keep the physics, change *when decisions are made*.

### Burn mode (default) — from execution to planning

Touch the screen and time dilates to 12%. Drag to aim a **burn**:
direction = thrust direction, length = burn duration, with a live
predicted trajectory simulated by the real physics (walls included — a
red ✕ marks where a plan ends in a crash). Release, and time resumes
while the ship *physically* rotates to the heading at the real
4.363 rad/s and burns for the planned ticks. Between burns it coasts
ballistically; tap to cancel a plan. The prediction is exact, not an
approximation — the same `Sim` stepped forward, so what you see is what
happens.

Two properties make this the right structural move:

- **The clock counts sim ticks, not wall time.** Slow-mo aiming is free
  but doesn't improve your time, so every mobile run is still a
  legitimate 60 Hz action tape on the unmodified engine — mobile times,
  desktop times, and Ace-solver ghosts all stay comparable.
- **It's the solver's own game.** Planning burns against gravity is
  literally what the Ace planner does; the human gets the same verb with
  a thumb. Touch is bad at held precision and great at deliberate
  drag-and-release — this plays to that instead of fighting it.

The POC also bets on three things that hold in every mode:

1. **Direct-control modes for purists**, one tap away in the menu:
   "Dual" (left half = horizontal rotation stick, right half =
   hold-to-thrust — fully independent, mirrors the desktop keys),
   "Slide" (hold to thrust + vertical slide to rotate), "Swoop" (one
   thumb: ship aims at your finger; close to the ship = aim only,
   farther = aim + thrust), and "Classic" (rotate/rotate/thrust
   buttons). Every mode, Burn included, is rate-limited by the engine's
   real 4.363 rad/s rotation — the schemes change *how you express*
   inputs, not what the ship can do.
2. **Race your ghost.** Every personal best is recorded and replayed as a
   translucent magenta ship on your next attempt — the core loop of this
   whole project (human vs. Ace ghosts) is *natively* a mobile hook.
   The obvious next step is shipping the Ace solver's tapes as
   downloadable "developer ghosts" per level, and async friend ghosts.
3. **Short, restartable runs.** Death → tap → flying again in under a
   second. Runs are 15–90 seconds. That's a perfect mobile session shape.

## Faithfulness to the real engine

The physics is a direct port of `src/real_physics.rs` / `real_game.rs`:

| Aspect | Value |
|---|---|
| Tick | fixed 1/60 s (accumulator loop, render decoupled) |
| Gravity | 100 px/s² |
| Thrust | 400 px/s² along heading, 0 rad = up (`rotation − π/2`) |
| Rotation | ±4.363323 rad/s |
| Drag / speed cap | none — pure momentum |
| Hull | the exact 5 collision segments from `shipVerts` |
| Pickup | ship center within 46.5 px (36.5 + 10) |
| Spawn | start vertex, 100 px up, rotation 0 |

One deliberate deviation: wall collision is checked **every** frame. The
original engine skips every other collision check above ~316 px/s — the
wall-clip exploit that's banned for ghosts in this repo — so the mobile
port simply doesn't have it (strict mode is the only mode).

Levels are the real ones, parsed at runtime with the same algorithm as
`src/real_map_parser.rs`: **Tutorial** (level 5000, a closed box with 2
pickups — level 0 in the JSON is malformed and unusable), **Caverns**
(level 8), **The Loop** (level 6), and **Deep Dive** (level 7, the
project's benchmark level).

## Mobile-specific additions

- Camera follows the ship with velocity lookahead and speed-based zoom-out
  (the maps are 2000–3000 px tall; a phone shows ~750 px, so you need to
  see where you're going).
- Off-screen pickups get cyan edge arrows.
- Haptics (`navigator.vibrate`) on pickup / crash / clear (Android; iOS
  Safari doesn't expose it).
- Best times + ghost tapes in `localStorage`.
- Neon vector look matching the desktop renderer: black, `#00ff41` walls
  and ship, cyan pickups, magenta ghosts/records.

## Files

- `index.html` — shell, viewport/PWA meta, canvas
- `game.js` — engine port, controls, camera, renderer, menus (~600 lines)
- `levels.js` — level arrays copied verbatim from `data/spaceace_levels.json`
- `parity_test.mjs` — replays an action tape through the JS sim and the
  real Rust engine and diffs the trajectories (see file header for usage)

## Where this could go (not built)

- **Ace ghosts as content**: ship `ghost_actions/L*_tas.json` tapes as
  per-level "beat the AI" challenges — bronze/silver/gold vs. the solver's
  line. This is the killer feature and needs zero new game design.
- Async multiplayer: friend ghosts via a tiny share-code backend.
- Daily level: one generated level (the repo already has a generator) per
  day, global leaderboard.
- Progressive unlock of the remaining real levels (1–5, 9, 10).
- Capacitor/Tauri wrap for app-store distribution; the game is already
  offline-capable and input-complete.
