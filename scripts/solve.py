#!/usr/bin/env python
"""Ace solver driver: find a superhuman action tape for a level.

The entire AI pipeline in one command:

    uv run python scripts/solve.py --level 7 --budget-min 20

Stages:
  1. Load the existing sidecar (ghost_actions/L{level}_tas.json) as the
     incumbent, if it replays correctly.
  2. Global beam-search portfolio (several seeds / parameter mixes) to get
     initial complete tapes; keep the best.
  3. Anytime improvement loop until the budget runs out: warm-started
     corridor refinement (cycling radius/quantization/seed), local-search
     polish, and suffix re-solves.
  4. Validate the best tape on the real engine (PyGameInstance), then write
     the ghost sidecar and dashboard ghost rows (only if faster).

Everything heavy runs in Rust (spaceace_rl.PySolver); this file is just
orchestration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spaceace_rl  # noqa: E402

from spaceace.ghost_actions import load_sidecar_actions, write_sidecar_if_best  # noqa: E402
from spaceace.strategies.actions import ALL_ACTIONS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "dashboard" / "spaceace_dashboard.db"


def validate_on_engine(level: int, tape: list[int]) -> tuple[bool, int]:
    """Authoritative replay through PyGameInstance."""
    game = spaceace_rl.PyGameInstance(level, len(tape) + 10)
    game.reset()
    for i, a in enumerate(tape):
        _, _, terminated, _, info = game.step(tuple(int(v) for v in ALL_ACTIONS[a]))
        if terminated:
            return bool(info["level_completed"]), i + 1
    return False, len(tape)


def capture_frames(level: int, tape: list[int], stride: int = 6) -> list[dict]:
    """Ghost frames in the dashboard schema ({x, y, rotation, thrusting, time})."""
    game = spaceace_rl.PyGameInstance(level, len(tape) + 10)
    game.reset()
    frames = []
    for i, a in enumerate(tape):
        act = ALL_ACTIONS[a]
        obs, _, terminated, _, _ = game.step(tuple(int(v) for v in act))
        if i % stride == 0 or terminated or i == len(tape) - 1:
            frames.append({
                "x": float(obs[0]),
                "y": float(obs[1]),
                "rotation": float(obs[4]),
                "thrusting": bool(act[2]),
                "time": round((i + 1) / 60.0, 4),
            })
        if terminated:
            break
    return frames


def save_ghost_rows(level: int, tape: list[int], ticks: int, labels: list[str]) -> None:
    seconds = ticks / 60.0
    frames = capture_frames(level, tape)
    conn = sqlite3.connect(DB_PATH)
    try:
        for label in labels:
            row = conn.execute(
                "SELECT time_seconds FROM ghost_replays WHERE level=? AND ghost_type=?",
                (level, label),
            ).fetchone()
            if row is not None and row[0] <= seconds:
                print(f"  [ghost] existing {label} ghost is faster ({row[0]:.3f}s <= {seconds:.3f}s), keeping it")
                continue
            conn.execute(
                "INSERT INTO ghost_replays (level, ghost_type, steps, time_seconds, frames_json) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(level, ghost_type) DO UPDATE SET "
                "steps=excluded.steps, time_seconds=excluded.time_seconds, "
                "frames_json=excluded.frames_json, created_at=datetime('now')",
                (level, label, len(frames), seconds, json.dumps(frames)),
            )
            print(f"  [ghost] saved {label} ghost: {seconds:.3f}s ({len(frames)} frames)")
        conn.commit()
    finally:
        conn.close()


def human_time(level: int) -> float | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT time_seconds FROM ghost_replays WHERE level=? AND ghost_type='human'",
            (level,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--budget-min", type=float, default=15.0, help="improvement budget in minutes")
    ap.add_argument("--width", type=int, default=60_000, help="global beam width")
    ap.add_argument("--refine-width", type=int, default=100_000)
    ap.add_argument("--max-ticks", type=int, default=0, help="beam horizon (0 = auto)")
    ap.add_argument("--seeds", type=int, default=3, help="global portfolio seeds")
    ap.add_argument("--labels", type=str, default="tas,ai", help="ghost labels to save")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore the existing sidecar")
    ap.add_argument("--deep", action="store_true",
                    help="finer/wider refinement schedule for squeezing an already-good tape")
    ap.add_argument("--allow-clip", action="store_true",
                    help="model the engine's high-speed collision skip exactly, allowing "
                         "tapes that thread walls on skipped frames (engine-legal, but "
                         "looks like clipping; off by default)")
    args = ap.parse_args()

    level = args.level
    solver = spaceace_rl.PySolver(level, strict=not args.allow_clip)
    href = human_time(level)
    print(f"L{level}: {solver.n_pickups()} pickups; human ghost: {href and round(href, 3)}s")

    best: list[int] | None = None

    if not args.fresh:
        prior = load_sidecar_actions(level, "tas")
        if prior:
            ok, _, ticks = solver.replay(bytes(prior))
            if ok:
                best = list(prior[:ticks])
                print(f"incumbent sidecar: {ticks} ticks ({ticks/60:.3f}s)")

    # Horizon: generously above the incumbent/human, capped for memory.
    if args.max_ticks:
        horizon = args.max_ticks
    else:
        ref = min(x for x in [best and len(best) * 1.4, href and href * 60 * 1.8, 6000] if x)
        horizon = int(ref)

    # --- stage 1: global portfolio -------------------------------------------
    # With an incumbent, global solves are capped below it and rarely win;
    # spend the budget on warm-started refinement instead.
    t_start = time.time()
    portfolio = [] if best is not None else list(
        itertools.product(range(args.seeds), [(1.0, 300.0), (1.0, 400.0), (1.2, 300.0)]))
    for seed, (mix, proj_div) in portfolio:
        if best is not None and (seed, (mix, proj_div)) != portfolio[0] and time.time() - t_start > args.budget_min * 30:
            break  # got something and half the budget is gone: move to refinement
        cap = min(horizon, len(best) - 1) if best else horizon
        tape = solver.solve(width=args.width, max_ticks=cap, seed=seed * 1000 + 1,
                            mix=mix, proj_div=proj_div)
        if tape is not None and (best is None or len(tape) < len(best)):
            best = list(tape)
            print(f"  [solve] seed={seed} mix={mix} pd={proj_div}: {len(best)} ticks ({len(best)/60:.3f}s)")
        elif tape is None:
            print(f"  [solve] seed={seed} mix={mix} pd={proj_div}: no completion")
        if best is not None and time.time() - t_start > args.budget_min * 30:
            break

    if best is None:
        print("FAILED: no completing tape found; increase --width or --budget-min")
        return 1

    # --- stage 2: anytime improvement loop ------------------------------------
    deadline = t_start + args.budget_min * 60
    # "global" = warm-started whole-map re-search (radius covers everything):
    # a full beam solve that provably can't do worse than the incumbent. It
    # is the workhorse; tubes and polish grind out the rest.
    if args.deep:
        stages = [
            ("refine", dict(radius=1e9, quant_pos=3.0, quant_vel=6.0, doom_scale=0.3)),
            ("refine", dict(radius=250.0, quant_pos=2.0, quant_vel=4.0, doom_scale=0.1)),
            ("polish", {}),
            ("refine", dict(radius=1e9, quant_pos=4.0, quant_vel=8.0, doom_scale=1.0)),
            ("refine", dict(radius=150.0, quant_pos=1.5, quant_vel=3.0, rot_bins=128, doom_scale=0.1)),
            ("suffix", {}),
            ("polish", {}),
        ]
        stale_limit = 3 * len(stages)
    else:
        stages = [
            ("refine", dict(radius=1e9, quant_pos=3.0, quant_vel=6.0, doom_scale=0.3)),
            ("refine", dict(radius=200.0, quant_pos=2.5, quant_vel=5.0, doom_scale=0.1)),
            ("polish", {}),
            ("refine", dict(radius=1e9, quant_pos=4.0, quant_vel=8.0, doom_scale=1.0)),
            ("refine", dict(radius=250.0, quant_pos=2.0, quant_vel=4.0, doom_scale=0.1)),
            ("suffix", {}),
        ]
        stale_limit = 2 * len(stages)
    schedule = itertools.cycle(stages)
    round_idx = 0
    stale = 0
    while time.time() < deadline and stale < stale_limit:
        kind, kw = next(schedule)
        round_idx += 1
        before = len(best)
        if kind == "refine":
            r = solver.refine(bytes(best), width=args.refine_width, seed=round_idx * 31,
                              mix=1.0, proj_div=300.0, **kw)
            if r:
                best = list(r)
        elif kind == "polish":
            budget_steps = 2_000_000_000 if args.deep else 500_000_000
            iters = max(60_000, min(2_000_000, budget_steps // max(len(best), 1)))
            tape, ticks = solver.polish(bytes(best), iters=iters, chains=10, seed=round_idx * 97)
            if ticks < len(best):
                best = list(tape[:ticks])
        elif kind == "suffix":
            for frac in (0.75, 0.5, 0.25):
                r = solver.resolve_suffix(bytes(best), int(len(best) * frac),
                                          width=args.width, seed=round_idx * 13 + int(frac * 10),
                                          mix=1.0, proj_div=300.0)
                if r and len(r) < len(best):
                    best = list(r)
        gained = before - len(best)
        stale = 0 if gained > 0 else stale + 1
        print(f"  [{kind}] round {round_idx}: {before} -> {len(best)} ticks "
              f"({len(best)/60:.3f}s, {'-'+str(gained) if gained else 'no gain'}, "
              f"{(deadline - time.time())/60:.1f}min left)", flush=True)

    ticks = len(best)
    print(f"best: {ticks} ticks = {ticks/60:.3f}s"
          + (f" vs human {href:.3f}s ({'BEATS human by ' + format(href - ticks/60, '.3f') + 's' if ticks/60 < href else 'SLOWER'})" if href else ""))

    # --- stage 3: validate + persist -------------------------------------------
    ok, engine_ticks = validate_on_engine(level, best)
    if not ok or engine_ticks != ticks:
        print(f"VALIDATION FAILED: engine says completed={ok} ticks={engine_ticks} (solver said {ticks})")
        return 2
    print("validated on PyGameInstance: exact match")

    if not args.no_save:
        write_sidecar_if_best(level, "tas", best, ticks)
        save_ghost_rows(level, best, ticks, [s.strip() for s in args.labels.split(",") if s.strip()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
