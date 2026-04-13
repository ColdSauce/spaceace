"""HRL agent for SpaceAce — TSP planner + trained waypoint pilot.

High-level: Held-Karp exact TSP solver computes optimal pickup ordering.
Low-level: DQN pilot trained to fly from current position to target pickup.

At inference, the agent owns the raw DirectEnv and manually computes the
augmented+normalized observations to feed the pilot model. This avoids
VecEnv auto-reset issues when waypoints are reached mid-level.
"""

import math
import os
from typing import Tuple, Dict, Any

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv

# 8 raycast directions relative to ship heading (must match Rust BASE_DIRS)
_BASE_DIRS = [
    (0.0, -1.0),      # forward
    (0.707, -0.707),   # forward-right
    (1.0, 0.0),        # right
    (0.707, 0.707),    # back-right
    (0.0, 1.0),        # back
    (-0.707, 0.707),   # back-left
    (-1.0, 0.0),       # left
    (-0.707, -0.707),  # forward-left
]


@register_agent("hrl")
class HRLAgent(BaseAgent):
    """Hierarchical RL agent: TSP planner + PPO waypoint pilot."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        model_path = kwargs.get("model_path")
        if model_path is None:
            # Prefer final_model (trained longer) over best_model (early checkpoint)
            for candidate in ["models/hrl/pilot/final_model", "models/hrl/pilot/best_model"]:
                if os.path.exists(candidate + ".zip"):
                    model_path = candidate
                    break
            if model_path is None:
                model_path = "models/hrl/pilot/final_model"

        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Pilot model not found: {model_path}.zip\n"
                f"Train one first: uv run python -m spaceace.agents.hrl.train_pilot"
            )

        self._level = level
        self._max_steps = max_steps

        # Raw environment — we own stepping directly
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

        # Load the trained pilot model (try PPO first, fall back to DQN)
        try:
            self._model = PPO.load(model_path)
            self._uses_discrete_actions = False
        except Exception:
            self._model = DQN.load(model_path)
            self._uses_discrete_actions = True

        # Load normalization stats
        norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        self._obs_rms = None
        self._clip_obs = 10.0
        if os.path.exists(norm_path):
            # Load VecNormalize to extract running mean/std
            from stable_baselines3.common.vec_env import VecNormalize as VN
            import pickle
            with open(norm_path, "rb") as f:
                vec_norm_data = pickle.load(f)
            self._obs_rms = vec_norm_data.obs_rms
            self._clip_obs = vec_norm_data.clip_obs

        # Pathfinder for TSP planning and directed navigation
        self._pathfinder = spaceace_rl.PyPathfinder(level)
        self._pickup_coords = self._pathfinder.get_pickup_coords()

        self._tsp_order = []
        self._current_target_pos = 0
        self._step_count = 0
        self._prev_path_dist = None
        self._prev_rotation = 0.0

        # Breadcrumb navigation
        self._breadcrumb_chain = []
        self._breadcrumb_idx = 0
        self._breadcrumb_spacing = 175.0   # px between waypoints
        self._breadcrumb_reach_radius = 30.0
        self._breadcrumb_refresh_interval = 60  # steps between path refreshes
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
        # Always include the final point (actual pickup location)
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
        self._prev_path_dist = None

        # Compute TSP ordering
        raw_obs = self._env.get_observation()
        ship_x, ship_y = float(raw_obs[0]), float(raw_obs[1])
        collected = list(self._env.get_pickup_states())

        self._tsp_order = self._pathfinder.get_tsp_order(ship_x, ship_y, collected)
        self._current_target_pos = 0
        self._prev_rotation = float(raw_obs[4])

        if self._tsp_order:
            target_idx = self._tsp_order[0]
            path_dist, _, _ = self._pathfinder.get_distance_to_specific_pickup(
                ship_x, ship_y, target_idx
            )
            self._prev_path_dist = path_dist
            # Compute breadcrumb chain to first target
            self._refresh_breadcrumbs(ship_x, ship_y)

        self._update_debug_info(ship_x, ship_y, collected)

    def _get_current_target(self) -> int:
        """Return the current target pickup index, or -1 if none."""
        if self._current_target_pos < len(self._tsp_order):
            return self._tsp_order[self._current_target_pos]
        return -1

    def _build_augmented_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Build 24-dim augmented observation for the pilot, directed at current breadcrumb."""
        ship_x, ship_y = float(raw_obs[0]), float(raw_obs[1])
        ship_vx, ship_vy = raw_obs[2], raw_obs[3]
        ship_rot = float(raw_obs[4])

        # Determine target: breadcrumb if available, otherwise pathfinder to pickup
        if self._breadcrumb_chain and self._breadcrumb_idx < len(self._breadcrumb_chain):
            bx, by = self._breadcrumb_chain[self._breadcrumb_idx]
            dx_bc = bx - ship_x
            dy_bc = by - ship_y
            euclid_dist = math.sqrt(dx_bc * dx_bc + dy_bc * dy_bc)
            mag = euclid_dist if euclid_dist > 1e-6 else 1.0
            dir_x, dir_y = dx_bc / mag, dy_bc / mag
            path_dist = euclid_dist
        else:
            target_idx = self._get_current_target()
            if target_idx < 0:
                path_dist, dir_x, dir_y = 0.0, 0.0, 0.0
                euclid_dist = 0.0
            else:
                path_dist, dir_x, dir_y = self._pathfinder.get_distance_to_specific_pickup(
                    ship_x, ship_y, target_idx
                )
                tx, ty = self._pickup_coords[target_idx]
                euclid_dist = math.sqrt((tx - ship_x) ** 2 + (ty - ship_y) ** 2)

        min_tti = self._compute_min_tti(raw_obs)

        # Index 3: euclidean distance to current target waypoint (replaces nearest pickup dist)
        target_dist_norm = min(euclid_dist / 2000.0, 1.0)

        # Drop absolute positions, keep relative features
        filtered_obs = np.concatenate([
            raw_obs[2:5],          # [0-2]: vx, vy, rotation
            [target_dist_norm],    # [3]: euclidean dist to current target waypoint
            raw_obs[8:16],         # [4-11]: wall distances (8)
            raw_obs[16:19],        # [12-14]: pickups_remaining, norm_x, norm_y
        ])

        path_dist_norm = min(path_dist / 5000.0, 1.0)
        speed = math.sqrt(float(ship_vx) ** 2 + float(ship_vy) ** 2)

        if speed > 1e-6 and (abs(dir_x) > 1e-6 or abs(dir_y) > 1e-6):
            speed_toward = (float(ship_vx) * dir_x + float(ship_vy) * dir_y) / speed
        else:
            speed_toward = 0.0

        heading_x = math.sin(ship_rot)
        heading_y = -math.cos(ship_rot)
        heading_alignment = heading_x * dir_x + heading_y * dir_y

        min_tti_norm = min(min_tti, 2.0) / 2.0
        time_remaining = 1.0 - self._step_count / self._max_steps

        # Angular velocity
        delta_rot = ship_rot - self._prev_rotation
        # Handle wraparound
        if delta_rot > math.pi:
            delta_rot -= 2 * math.pi
        elif delta_rot < -math.pi:
            delta_rot += 2 * math.pi
        angular_vel_norm = delta_rot / (5.0 * (1.0 / 60.0))  # normalize by max rotation speed * dt
        angular_vel_norm = max(-1.0, min(1.0, angular_vel_norm))
        self._prev_rotation = ship_rot

        extra = np.array([
            path_dist_norm, dir_x, dir_y, speed, speed_toward,
            heading_alignment, min_tti_norm, time_remaining, angular_vel_norm,
        ], dtype=np.float32)

        return np.concatenate([filtered_obs, extra])

    def _compute_min_tti(self, obs: np.ndarray) -> float:
        """Compute minimum time-to-impact across 8 raycast directions."""
        ship_vx, ship_vy = float(obs[2]), float(obs[3])
        ship_rot = float(obs[4])
        wall_distances = obs[8:16]

        cos_r = math.cos(ship_rot)
        sin_r = math.sin(ship_rot)
        min_tti = float('inf')

        for i, (dx, dy) in enumerate(_BASE_DIRS):
            world_dx = dx * cos_r - dy * sin_r
            world_dy = dx * sin_r + dy * cos_r
            v_toward = ship_vx * world_dx + ship_vy * world_dy
            if v_toward > 1.0:
                tti = float(wall_distances[i]) / v_toward
                if tti < min_tti:
                    min_tti = tti
        return min_tti

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply running mean/std normalization (matching VecNormalize)."""
        if self._obs_rms is not None:
            obs = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
            obs = np.clip(obs, -self._clip_obs, self._clip_obs)
        return obs.astype(np.float32)

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Build observation directed at current target
        raw_obs = self._env.get_observation()
        aug_obs = self._build_augmented_obs(raw_obs)
        norm_obs = self._normalize_obs(aug_obs)

        # Get action from pilot model
        action, _ = self._model.predict(
            norm_obs.reshape(1, -1), deterministic=True
        )
        action = action[0]  # Remove batch dim

        # Convert Discrete(8) → MultiDiscrete([2,2,2]) if using DQN
        if self._uses_discrete_actions:
            from spaceace.agents.hrl.waypoint_env import ACTION_TABLE
            action = ACTION_TABLE[int(action)]

        # Step the raw environment
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        self._steps_since_refresh += 1

        ship_x, ship_y = float(obs[0]), float(obs[1])

        # Advance breadcrumb if within reach
        if self._breadcrumb_chain and self._breadcrumb_idx < len(self._breadcrumb_chain):
            bx, by = self._breadcrumb_chain[self._breadcrumb_idx]
            dist_to_bc = math.sqrt((ship_x - bx) ** 2 + (ship_y - by) ** 2)
            if dist_to_bc < self._breadcrumb_reach_radius:
                self._breadcrumb_idx += 1

        # Check pickup collection — advance target if needed
        collected = list(self._env.get_pickup_states())
        target_idx = self._get_current_target()

        if target_idx >= 0 and collected[target_idx]:
            self._current_target_pos += 1

            # Dynamic re-planning on remaining pickups
            remaining = [i for i in range(len(collected)) if not collected[i]]
            if remaining:
                new_order = self._pathfinder.get_tsp_order(ship_x, ship_y, collected)
                self._tsp_order = self._tsp_order[:self._current_target_pos] + new_order
                self._prev_path_dist = None

            # Recompute breadcrumbs for new target
            self._refresh_breadcrumbs(ship_x, ship_y)

        # Periodically refresh breadcrumbs to handle drift
        elif self._steps_since_refresh >= self._breadcrumb_refresh_interval:
            self._refresh_breadcrumbs(ship_x, ship_y)

        # Update debug info
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
            # Compute euclidean distance
            euclidean = math.sqrt((ship_x - tx) ** 2 + (ship_y - ty) ** 2)
            target_info = {
                "idx": target_idx,
                "x": tx,
                "y": ty,
                "path_dist": path_dist,
                "euclidean_dist": euclidean,
                "dir_x": dir_x,
                "dir_y": dir_y,
            }

        # Show TSP order info
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
