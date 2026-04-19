"""Shared curriculum builder: pickup-variant progression for PPO and AlphaZero.

For each base level (e.g. L3000), the agent trains on the base first, then on
pickup variants with +1, +2, +3 extra pickups before advancing to the next
base level. Variants are numbered at `VARIANT_START + (base - SOURCE_START) *
VARIANT_STRIDE + (extra - 1)` and produced by `spaceace.tools.add_pickups`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spaceace.training.trainer import LevelStage

SOURCE_START = 3000
VARIANT_START = 5000
VARIANT_STRIDE = 5  # must match add_pickups.py --max-extra default
PICKUPS_PER_LEVEL = 3  # +1 through +PICKUPS_PER_LEVEL per base level

LEVELS_PATH = Path("data/spaceace_levels.json")


def pickup_variant(base_level: int, extra: int) -> int:
    """Return the level number for `base_level` with `extra` additional pickups."""
    idx = base_level - SOURCE_START
    return VARIANT_START + idx * VARIANT_STRIDE + (extra - 1)


def build_curriculum(
    base_levels: list[int],
    advance_win_rate: float = 0.7,
    min_steps: int = 50_000,
    min_iters: int = 1,
    max_episode_steps: int | None = 3000,
    pickups_per_level: int = PICKUPS_PER_LEVEL,
) -> list[LevelStage]:
    """Build pickup-variant curriculum: base -> +1 -> ... -> +N per source level."""
    stages: list[LevelStage] = []
    for base in base_levels:
        stages.append(LevelStage(
            levels=[base],
            max_episode_steps=max_episode_steps,
            advance_win_rate=advance_win_rate,
            min_steps=min_steps,
            min_iters=min_iters,
        ))
        for extra in range(1, pickups_per_level + 1):
            stages.append(LevelStage(
                levels=[pickup_variant(base, extra)],
                max_episode_steps=max_episode_steps,
                advance_win_rate=advance_win_rate,
                min_steps=min_steps,
                min_iters=min_iters,
            ))
    return stages


def _existing_level_ids() -> set[int]:
    if not LEVELS_PATH.exists():
        return set()
    with open(LEVELS_PATH) as f:
        data = json.load(f)
    out: set[int] = set()
    for k in data.keys():
        if k.startswith("_"):
            continue
        try:
            out.add(int(k))
        except ValueError:
            pass
    return out


def ensure_pickup_variants(
    base_levels: list[int],
    pickups_per_level: int = PICKUPS_PER_LEVEL,
) -> None:
    """Run add_pickups.py if any required pickup variants are missing.

    `add_pickups.py` wipes all levels >= its --start-level before regenerating,
    so we issue one invocation over the full base range starting at the variant
    slot for the smallest base. Callers who need finer control should invoke
    `spaceace.tools.add_pickups` directly.
    """
    if not base_levels:
        return

    existing = _existing_level_ids()
    missing: list[int] = []
    for base in base_levels:
        if base not in existing:
            continue  # base itself is missing; caller's responsibility
        for extra in range(1, pickups_per_level + 1):
            if pickup_variant(base, extra) not in existing:
                missing.append(base)
                break

    if not missing:
        return

    bases = sorted(set(base_levels))
    first, last = bases[0], bases[-1]
    source_spec = f"{first}-{last}" if first != last else str(first)
    start_level_num = VARIANT_START + (first - SOURCE_START) * VARIANT_STRIDE

    print(
        f"Pickup variants missing for {len(missing)} base level(s); "
        f"regenerating variants for L{first}-L{last} via add_pickups "
        f"(will overwrite level IDs >= {start_level_num})."
    )
    subprocess.run(
        [
            "uv", "run", "python", "-m", "spaceace.tools.add_pickups",
            "--source", source_spec,
            "--start-level", str(start_level_num),
            "--max-extra", str(VARIANT_STRIDE),
        ],
        check=True,
    )
