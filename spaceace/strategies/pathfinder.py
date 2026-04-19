"""Pathfinder strategies wrapping the Rust PyO3 implementation."""

from __future__ import annotations

import spaceace_rl


# Only switch targets when another pickup's total TSP cost is this fraction
# better than the current target's. Prevents oscillation between equidistant
# pickups while still allowing switches when the ship genuinely gets closer
# to a different pickup.
_STICKY_HYSTERESIS = 0.85


class RustPathfinder:
    """Thin adapter over spaceace_rl.PyPathfinder implementing the Pathfinder protocol.

    The Rust side owns the grid BFS and (optionally) the momentum-aware state-space
    search. Python never reimplements this.

    Includes a per-step cache: observation builder and reward shaper both call
    nearest_pickup_info with the same args on the same physics step.  Call
    ``clear_cache()`` between physics steps (StrategyWrapper does this).

    Target stickiness: once committed to a pickup, keeps targeting it until
    collected or another pickup becomes significantly closer (15% better path
    cost). Prevents oscillation when the ship is equidistant between pickups.
    """

    def __init__(self, level: int, backend: str = "grid"):
        if backend not in ("grid", "momentum"):
            raise ValueError(f"unknown backend {backend!r}")
        self.level = level
        self.backend = backend
        self._impl = spaceace_rl.PyPathfinder(level, backend)
        self._cache_key: tuple | None = None
        self._cache_val: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._sticky_target: int | None = None
        self._sticky_collected: tuple | None = None

    def nearest_pickup_info(
        self, ship_x: float, ship_y: float, collected: list[bool]
    ) -> tuple[float, float, float]:
        key = (ship_x, ship_y, tuple(collected))
        if key == self._cache_key:
            return self._cache_val

        collected_tuple = tuple(collected)

        # If the pickup set changed (a pickup was collected), reset stickiness.
        if collected_tuple != self._sticky_collected:
            self._sticky_target = None
            self._sticky_collected = collected_tuple

        # Ask the Rust pathfinder for the globally best target.
        best_val = self._impl.get_nearest_pickup_info(ship_x, ship_y, collected)
        best_dist = best_val[0]
        used_sticky = False

        # If we have a sticky target that's still uncollected, check if we
        # should keep it or switch.
        if (
            self._sticky_target is not None
            and self._sticky_target < len(collected)
            and not collected[self._sticky_target]
            and best_dist > 0
        ):
            sticky_val = self._impl.get_distance_to_specific_pickup(
                ship_x, ship_y, self._sticky_target
            )
            sticky_dist = sticky_val[0]
            # Keep the sticky target unless the new one is significantly better.
            if sticky_dist > 0 and best_dist >= sticky_dist * _STICKY_HYSTERESIS:
                val = sticky_val
                used_sticky = True

        if not used_sticky:
            val = best_val
            # Adopt the new best target as sticky.
            debug = self._impl.get_debug_target_info(ship_x, ship_y, collected)
            self._sticky_target = int(debug[0]) if debug[0] >= 0 else None

        self._cache_key = key
        self._cache_val = val
        return val

    def clear_cache(self) -> None:
        self._cache_key = None

    def reset_sticky(self) -> None:
        """Reset target stickiness. Call on episode reset."""
        self._sticky_target = None
        self._sticky_collected = None
