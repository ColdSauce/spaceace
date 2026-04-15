"""Pathfinder strategies wrapping the Rust PyO3 implementation."""

from __future__ import annotations

import spaceace_rl


class RustPathfinder:
    """Thin adapter over spaceace_rl.PyPathfinder implementing the Pathfinder protocol.

    The Rust side owns the grid BFS and (optionally) the momentum-aware state-space
    search. Python never reimplements this.

    Includes a per-step cache: observation builder and reward shaper both call
    nearest_pickup_info with the same args on the same physics step.  Call
    ``clear_cache()`` between physics steps (StrategyWrapper does this).
    """

    def __init__(self, level: int, backend: str = "grid"):
        if backend not in ("grid", "momentum"):
            raise ValueError(f"unknown backend {backend!r}")
        self.level = level
        self.backend = backend
        self._impl = spaceace_rl.PyPathfinder(level, backend)
        self._cache_key: tuple | None = None
        self._cache_val: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def nearest_pickup_info(
        self, ship_x: float, ship_y: float, collected: list[bool]
    ) -> tuple[float, float, float]:
        key = (ship_x, ship_y, tuple(collected))
        if key == self._cache_key:
            return self._cache_val
        val = self._impl.get_nearest_pickup_info(ship_x, ship_y, collected)
        self._cache_key = key
        self._cache_val = val
        return val

    def clear_cache(self) -> None:
        self._cache_key = None
