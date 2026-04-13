"""MCTS-based calibration of max_episode_steps per level."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _calibrate_one_level(args: tuple) -> tuple:
    """Worker: run Rust MCTS on a single level. Designed for ProcessPoolExecutor."""
    level, ceiling = args
    import spaceace_rl

    engine = spaceace_rl.PyMCTSEngine(level, ceiling)
    results = engine.play_games(3, 5000, action_repeat=5, max_steps=ceiling)
    return (level, results)


def _load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def _save_cache(cache: dict[int, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, indent=2)


def calibrate_max_steps(
    level: int,
    cache: Path | None = None,
    multiplier: float = 3.0,
    floor: int = 300,
    ceiling: int = 5000,
    fallback: int = 3000,
) -> int:
    """Run Rust MCTS to estimate how many steps an optimal player needs.

    Rounds up to nearest 100, applies a safety multiplier. Cached to
    *cache* keyed by level.
    """
    disk_cache: dict[int, dict] = {}
    if cache is not None:
        disk_cache = _load_cache(cache)
        if level in disk_cache:
            return disk_cache[level]["max_steps"]

    _, results = _calibrate_one_level((level, ceiling))
    win_steps = [steps for completed, _, steps in results if completed]

    if win_steps:
        base = min(win_steps)
        ms = max(floor, min(ceiling, int(base * multiplier)))
    else:
        ms = fallback

    entry = {
        "max_steps": ms,
        "mcts_wins": len(win_steps),
        "mcts_games": len(results),
        "mcts_fastest": min(win_steps) if win_steps else None,
        "mcts_avg_steps": sum(s for _, _, s in results) / len(results) if results else 0,
    }

    if cache is not None:
        disk_cache[level] = entry
        _save_cache(disk_cache, cache)

    return ms


def calibrate_stages(
    stages: list,
    cache: Path | None = None,
    multiplier: float = 3.0,
    floor: int = 300,
    ceiling: int = 5000,
    fallback: int = 3000,
) -> None:
    """Resolve ``max_episode_steps`` for every stage that has ``None``.

    Uses the hardest level in the stage (highest calibrated value) to set the
    stage-wide cap. Mutates stages in place.
    """
    for stage in stages:
        if stage.max_episode_steps is not None:
            continue
        per_level = [
            calibrate_max_steps(lvl, cache, multiplier, floor, ceiling, fallback)
            for lvl in stage.levels
        ]
        stage.max_episode_steps = max(per_level)
