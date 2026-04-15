"""HRL agent for SpaceAce — TSP planner + reactive waypoint pilot.

High-level: Held-Karp exact TSP solver computes optimal pickup ordering.
Low-level: Reactive pilot follows pathfinder breadcrumbs with curvature-aware
speed control and wall-TTI safety braking.
"""

import math
from typing import Tuple, Dict, Any

import numpy as np

import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.agents.hrl.reactive_pilot import compute_action
from spaceace.core.env import SpaceAceDirectEnv


@register_agent("hrl")
class HRLAgent(BaseAgent):
    """Hierarchical RL agent: TSP planner + reactive waypoint pilot."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._level = level
        self._max_steps = max_steps

        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

        # Pathfinder for TSP planning and navigation
        self._pathfinder = spaceace_rl.PyPathfinder(level)
        self._pickup_coords = self._pathfinder.get_pickup_coords()

        self._tsp_order = []
        self._current_target_pos = 0
        self._step_count = 0

        # Breadcrumb navigation — tighter spacing for better corner tracking
        self._breadcrumb_chain = []
        self._breadcrumb_idx = 0
        self._breadcrumb_spacing = 80.0
        self._breadcrumb_reach_radius = 50.0
        self._breadcrumb_refresh_interval = 40
        self._steps_since_refresh = 0

        # Debug info for visualization
        self.debug_info = {}

    def _compute_breadcrumbs(self, path: list) -> list:
        """Discretize a dense grid path into waypoints ~spacing px apart."""
        if not path:
            return []
        breadcrumbs = [path[0]]
        accum = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            accum += math.sqrt(dx * dx + dy * dy)
            if accum >= self._breadcrumb_spacing:
                breadcrumbs.append(path[i])
                accum = 0.0
        if len(path) > 1 and breadcrumbs[-1] != path[-1]:
            breadcrumbs.append(path[-1])
        return breadcrumbs

    def _refresh_breadcrumbs(self, ship_x: float, ship_y: float):
        """Recompute breadcrumbs from current ship position to current target."""
        target_idx = self._get_current_target()
        if target_idx < 0:
            self._breadcrumb_chain = []
            self._breadcrumb_idx = 0
            return
        path = self._pathfinder.get_path_to_specific_pickup(ship_x, ship_y, target_idx)
        self._breadcrumb_chain = self._compute_breadcrumbs(path)
        self._breadcrumb_idx = 0
        self._steps_since_refresh = 0

    def reset(self) -> None:
        self._env.reset()
        self._step_count = 0

        raw_obs = self._env.get_observation()
        ship_x, ship_y = float(raw_obs[0]), float(raw_obs[1])
        collected = list(self._env.get_pickup_states())

        self._tsp_order = self._pathfinder.get_tsp_order(ship_x, ship_y, collected)
        self._current_target_pos = 0

        if self._tsp_order:
            self._refresh_breadcrumbs(ship_x, ship_y)

        self._update_debug_info(ship_x, ship_y, collected)

    def _get_current_target(self) -> int:
        """Return the current target pickup index, or -1 if none."""
        if self._current_target_pos < len(self._tsp_order):
            return self._tsp_order[self._current_target_pos]
        return -1

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_obs = self._env.get_observation()

        # Reactive pilot decides action from obs + breadcrumbs
        action = compute_action(
            raw_obs,
            self._breadcrumb_chain,
            self._breadcrumb_idx,
        )

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        self._steps_since_refresh += 1

        ship_x, ship_y = float(obs[0]), float(obs[1])

        # Advance breadcrumb if within reach
        while (self._breadcrumb_chain
               and self._breadcrumb_idx < len(self._breadcrumb_chain)):
            bx, by = self._breadcrumb_chain[self._breadcrumb_idx]
            dist = math.sqrt((ship_x - bx) ** 2 + (ship_y - by) ** 2)
            if dist < self._breadcrumb_reach_radius:
                self._breadcrumb_idx += 1
            else:
                break

        # Check pickup collection — advance target if needed
        collected = list(self._env.get_pickup_states())
        target_idx = self._get_current_target()

        if target_idx >= 0 and collected[target_idx]:
            self._current_target_pos += 1

            # Re-plan on remaining pickups
            remaining = [i for i in range(len(collected)) if not collected[i]]
            if remaining:
                new_order = self._pathfinder.get_tsp_order(ship_x, ship_y, collected)
                self._tsp_order = self._tsp_order[:self._current_target_pos] + new_order

            self._refresh_breadcrumbs(ship_x, ship_y)

        elif self._steps_since_refresh >= self._breadcrumb_refresh_interval:
            self._refresh_breadcrumbs(ship_x, ship_y)

        self._update_debug_info(ship_x, ship_y, collected)

        return action, reward, terminated, truncated, info

    def _update_debug_info(self, ship_x: float, ship_y: float, collected: list):
        """Update debug info for visualization overlay."""
        target_idx = self._get_current_target()

        target_info = {}
        if target_idx >= 0:
            tx, ty = self._pickup_coords[target_idx]
            path_dist, dir_x, dir_y = self._pathfinder.get_distance_to_specific_pickup(
                ship_x, ship_y, target_idx
            )
            euclidean = math.sqrt((ship_x - tx) ** 2 + (ship_y - ty) ** 2)
            target_info = {
                "idx": target_idx,
                "x": tx, "y": ty,
                "path_dist": path_dist,
                "euclidean_dist": euclidean,
                "dir_x": dir_x, "dir_y": dir_y,
            }

        remaining_order = self._tsp_order[self._current_target_pos:]

        self.debug_info = {
            "tsp_order": remaining_order,
            "current_target_pos": self._current_target_pos,
            "total_targets": len(self._tsp_order),
            "target": target_info,
            "heuristic": {
                "path_dist": target_info.get("path_dist", 0),
            },
            "breadcrumbs": self._breadcrumb_chain,
            "breadcrumb_idx": self._breadcrumb_idx,
            "breadcrumb_total": len(self._breadcrumb_chain),
        }

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
