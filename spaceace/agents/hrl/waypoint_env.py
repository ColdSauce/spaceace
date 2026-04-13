"""Waypoint pilot training environment for HRL agent.

Trains a low-level pilot to fly from current position to a specific target
pickup as fast as possible. Reward philosophy: reward speed, only punish death.
No safety penalties — the agent discovers optimal racing lines by balancing
speed against crash risk.
"""

import math
import random

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import spaceace_rl
from spaceace.core.gym_wrapper import SpaceAceGymWrapper

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

# Flatten MultiDiscrete([2,2,2]) → Discrete(8) for DQN
ACTION_TABLE = np.array([
    [0, 0, 0],  # 0: coast
    [0, 0, 1],  # 1: thrust
    [1, 0, 0],  # 2: left
    [1, 0, 1],  # 3: left + thrust
    [0, 1, 0],  # 4: right
    [0, 1, 1],  # 5: right + thrust
    [1, 1, 0],  # 6: left + right (cancel)
    [0, 0, 1],  # 7: all (same as thrust since rotations cancel)
], dtype=np.int32)


class ActionFlattenWrapper(gym.ActionWrapper):
    """Converts Discrete(8) actions to MultiDiscrete([2,2,2]) for DQN compatibility."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act):
        return ACTION_TABLE[act]


class WaypointPilotEnv(gym.Wrapper):
    """
    Wraps SpaceAceGymWrapper to train a waypoint-to-waypoint pilot.

    Reward: speed toward target, penalize only death. No wall proximity
    penalties — the agent finds racing lines by balancing speed vs crash risk.
    """

    # PBRS reward constants
    GAMMA = 0.99
    DIST_SCALE = 500.0    # normalize potential so shaping rewards are small
    WAYPOINT_BONUS = 50.0
    CRASH_PENALTY = -20.0

    def __init__(self, env: gym.Env, level: int = 0, max_steps: int = 500,
                 action_repeat: int = 2):
        super().__init__(env)
        self._level = level
        self._max_steps = max_steps
        self._action_repeat = action_repeat
        self._pathfinder = spaceace_rl.PyPathfinder(level)
        self._pickup_coords = self._pathfinder.get_pickup_coords()
        self._num_pickups = len(self._pickup_coords)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        self._target_pickup_idx = 0
        self._prev_path_dist = None
        self._prev_target_collected = False
        self._prev_rotation = 0.0

        # Metrics tracking
        self.episode_thrust_steps = 0
        self.episode_steps = 0
        self.episode_pickups_collected = 0
        self.episode_crashed = False
        self.episode_completed = False
        self.episode_waypoint_reached = False
        # Last-episode snapshot
        self.last_episode_thrust_steps = 0
        self.last_episode_steps = 0
        self.last_episode_pickups_collected = 0
        self.last_episode_crashed = False
        self.last_episode_completed = False
        self.last_episode_waypoint_reached = False

    def set_target(self, pickup_idx: int):
        """Set the target pickup for the pilot to navigate to."""
        self._target_pickup_idx = pickup_idx

    def _get_target_info(self, obs: np.ndarray) -> tuple:
        """Query pathfinder for distance and direction to the target pickup."""
        ship_x, ship_y = float(obs[0]), float(obs[1])
        return self._pathfinder.get_distance_to_specific_pickup(
            ship_x, ship_y, self._target_pickup_idx
        )

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

    def _augment_obs(self, obs: np.ndarray, path_dist: float, dir_x: float,
                     dir_y: float, min_tti: float) -> np.ndarray:
        """Filter raw 19-dim obs to remove absolute positions, add 9 derived features → 24-dim."""
        ship_vx, ship_vy = obs[2], obs[3]
        ship_rot = float(obs[4])

        # Euclidean distance to current target waypoint (replaces nearest pickup dist)
        ship_x, ship_y = float(obs[0]), float(obs[1])
        tx, ty = self._pickup_coords[self._target_pickup_idx]
        euclid_dist = math.sqrt((tx - ship_x) ** 2 + (ty - ship_y) ** 2)
        target_dist_norm = min(euclid_dist / 2000.0, 1.0)

        filtered_obs = np.concatenate([
            obs[2:5],          # [0-2]: vx, vy, rotation
            [target_dist_norm],# [3]: euclidean dist to current target waypoint
            obs[8:16],         # [4-11]: wall distances (8)
            obs[16:19],        # [12-14]: pickups_remaining, norm_x, norm_y
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
        time_remaining = 1.0 - self.episode_steps / self._max_steps

        # Angular velocity
        delta_rot = ship_rot - self._prev_rotation
        if delta_rot > math.pi:
            delta_rot -= 2 * math.pi
        elif delta_rot < -math.pi:
            delta_rot += 2 * math.pi
        angular_vel_norm = delta_rot / (5.0 * (1.0 / 60.0))
        angular_vel_norm = max(-1.0, min(1.0, angular_vel_norm))
        self._prev_rotation = ship_rot

        extra = np.array([
            path_dist_norm, dir_x, dir_y, speed, speed_toward,
            heading_alignment, min_tti_norm, time_remaining, angular_vel_norm,
        ], dtype=np.float32)

        return np.concatenate([filtered_obs, extra])

    def _shaped_reward(self, obs: np.ndarray, action: np.ndarray, info: dict) -> float:
        """PBRS reward: R = gamma * Phi(s') - Phi(s) + terminal_bonus."""
        collected = list(self.env.get_pickup_states())
        target_collected = collected[self._target_pickup_idx] if self._target_pickup_idx < len(collected) else False

        pickups_now = info.get("pickups_remaining", 0)
        prev = getattr(self, '_prev_pickups_remaining', pickups_now)
        newly_collected = prev - pickups_now
        if newly_collected > 0:
            self.episode_pickups_collected += newly_collected
        self._prev_pickups_remaining = pickups_now

        if info.get("ship_exploded", False):
            self.episode_crashed = True
            return self.CRASH_PENALTY

        if target_collected and not self._prev_target_collected:
            self.episode_waypoint_reached = True
            return self.WAYPOINT_BONUS
        self._prev_target_collected = target_collected

        if info.get("level_completed", False):
            self.episode_completed = True
            return self.WAYPOINT_BONUS

        # PBRS: Phi(s) = -distance / DIST_SCALE (normalized to keep shaping small)
        path_dist, _, _ = self._get_target_info(obs)
        phi_new = -path_dist / self.DIST_SCALE
        phi_old = -self._prev_path_dist / self.DIST_SCALE if self._prev_path_dist is not None else phi_new
        reward = self.GAMMA * phi_new - phi_old
        self._prev_path_dist = path_dist

        # Track thrust for metrics
        if len(action) > 2 and int(action[2]) > 0:
            self.episode_thrust_steps += 1

        return reward

    def reset(self, **kwargs):
        if self.episode_steps > 0:
            self.last_episode_thrust_steps = self.episode_thrust_steps
            self.last_episode_steps = self.episode_steps
            self.last_episode_pickups_collected = self.episode_pickups_collected
            self.last_episode_crashed = self.episode_crashed
            self.last_episode_completed = self.episode_completed
            self.last_episode_waypoint_reached = self.episode_waypoint_reached

        obs, info = self.env.reset(**kwargs)

        # Random target (corridor levels have 1 pickup, so always idx 0)
        self._target_pickup_idx = random.randint(0, self._num_pickups - 1)

        self._prev_pickups_remaining = int(obs[16])
        self._prev_path_dist = None
        self._prev_target_collected = False
        self._prev_rotation = float(obs[4])
        self.episode_thrust_steps = 0
        self.episode_steps = 0
        self.episode_pickups_collected = 0
        self.episode_crashed = False
        self.episode_completed = False
        self.episode_waypoint_reached = False

        path_dist, dir_x, dir_y = self._get_target_info(obs)
        self._prev_path_dist = path_dist

        min_tti = self._compute_min_tti(obs)
        return self._augment_obs(obs, path_dist, dir_x, dir_y, min_tti), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self._action_repeat):
            obs, _base_reward, terminated, truncated, info = self.env.step(action)
            reward = self._shaped_reward(obs, action, info)
            total_reward += reward
            self.episode_steps += 1

            if self.episode_waypoint_reached:
                terminated = True
                break
            if terminated or truncated:
                break
            if self.episode_steps >= self._max_steps:
                truncated = True
                break

        if terminated or truncated:
            total = max(self.episode_steps, 1)
            info["episode_metrics"] = {
                "thrust_ratio": self.episode_thrust_steps / total,
                "pickups_collected": self.episode_pickups_collected,
                "crashed": self.episode_crashed,
                "completed": self.episode_completed,
                "waypoint_reached": self.episode_waypoint_reached,
                "length": self.episode_steps,
            }

        path_dist, dir_x, dir_y = self._get_target_info(obs)
        min_tti = self._compute_min_tti(obs)
        return self._augment_obs(obs, path_dist, dir_x, dir_y, min_tti), total_reward, terminated, truncated, info


class WaypointMetricsCallback(BaseCallback):
    """Logs per-episode waypoint pilot metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is not None:
                self.logger.record("episode/thrust_ratio", metrics["thrust_ratio"])
                self.logger.record("episode/pickups_collected", metrics["pickups_collected"])
                self.logger.record("episode/crashed", float(metrics["crashed"]))
                self.logger.record("episode/completed", float(metrics["completed"]))
                self.logger.record("episode/waypoint_reached", float(metrics["waypoint_reached"]))
                self.logger.record("episode/length", metrics["length"])
        return True


def make_waypoint_env(level: int, max_steps: int = 500, action_repeat: int = 2,
                      flatten_actions: bool = False):
    """Create a single WaypointPilotEnv wrapped in Monitor."""
    def _init():
        base = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        waypoint = WaypointPilotEnv(base, level=level, max_steps=max_steps,
                                     action_repeat=action_repeat)
        env = ActionFlattenWrapper(waypoint) if flatten_actions else waypoint
        monitored = Monitor(env)
        return monitored
    return _init
