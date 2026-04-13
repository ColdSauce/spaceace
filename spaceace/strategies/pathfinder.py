"""Pathfinder strategies wrapping the Rust PyO3 implementation."""

from __future__ import annotations

import spaceace_rl


class RustPathfinder:
    """Thin adapter over spaceace_rl.PyPathfinder implementing the Pathfinder protocol.

    The Rust side owns the grid BFS and (optionally) the momentum-aware state-space
    search. Python never reimplements this.
    """

    def __init__(self, level: int, backend: str = "grid"):
        # backend currently unused at the PyO3 boundary (PyPathfinder is grid-only).
        # Phase 6 unifies the surface and will wire this selector through.
        self.level = level
        self.backend = backend
        self._impl = spaceace_rl.PyPathfinder(level)

    def nearest_pickup_info(
        self, ship_x: float, ship_y: float, collected: list[bool]
    ) -> tuple[float, float, float]:
        return self._impl.get_nearest_pickup_info(ship_x, ship_y, collected)
