"""Trajectory optimization via sliding-window mini beam search."""

from __future__ import annotations

import math
import time

import numpy as np

from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS
from spaceace.strategies.pathfinder import RustPathfinder

NUM_ACTIONS = len(ALL_ACTIONS)


class TrajectoryOptimizer:
    """Phase 2: locally improve a trajectory by re-searching short windows.

    For each window, runs a mini beam search to find a shorter action sequence.
    Validates by replaying the full spliced trajectory through the physics engine.
    """

    def __init__(
        self,
        env: SpaceAceDirectEnv,
        level: int,
        window_size: int = 50,
        stride: int = 25,
        mini_beam_width: int = 200,
        max_passes: int = 10,
    ) -> None:
        self._env = env
        self._level = level
        self._window_size = window_size
        self._stride = stride
        self._mini_beam_width = mini_beam_width
        self._max_passes = max_passes
        self._pathfinder = RustPathfinder(level, backend="grid")

    def optimize(self, actions: list[int]) -> list[int]:
        """Iteratively optimize the trajectory. Returns improved action list."""
        if not actions:
            return actions

        best = list(actions)
        print(f"\nPhase 2: Trajectory optimization (starting length: {len(best)})")

        for pass_num in range(1, self._max_passes + 1):
            t_start = time.time()
            improved = False
            windows_tried = 0
            windows_improved = 0

            start = 0
            while start + self._window_size <= len(best):
                end = start + self._window_size
                windows_tried += 1

                # Replay prefix to get the start state
                prefix_state = self._replay_prefix(best, start)
                if prefix_state is None:
                    start += self._stride
                    continue

                # Mini beam search for a shorter alternative
                alternative = self._mini_beam_search(
                    prefix_state, best, start, end,
                )

                if alternative is not None and len(alternative) < (end - start):
                    # Splice in the shorter alternative
                    new_trajectory = best[:start] + alternative + best[end:]

                    # Validate the full spliced trajectory
                    if self._validate_full(new_trajectory):
                        saved = len(best) - len(new_trajectory)
                        best = new_trajectory
                        improved = True
                        windows_improved += 1
                        # Don't advance — re-check with new context
                        continue

                start += self._stride

            elapsed = time.time() - t_start
            print(
                f"  Pass {pass_num}: tried={windows_tried} "
                f"improved={windows_improved} "
                f"length={len(best)} "
                f"elapsed={elapsed:.1f}s"
            )

            if not improved:
                print("  No improvements found, stopping optimization")
                break

        print(f"Final optimized length: {len(best)}")
        return best

    def _replay_prefix(self, actions: list[int], up_to: int) -> object | None:
        """Replay actions[0:up_to] and return the resulting state snapshot."""
        env = self._env
        env.reset()
        for i in range(up_to):
            obs, reward, terminated, truncated, info = env.step(ALL_ACTIONS[actions[i]])
            if terminated or truncated:
                return None
        return env.save_state()

    def _mini_beam_search(
        self,
        start_state: object,
        full_actions: list[int],
        window_start: int,
        window_end: int,
    ) -> list[int] | None:
        """Search for a shorter action sequence through the window.

        Uses a simple beam search scored by path distance progress.
        Returns the shortest alternative found that, when spliced in,
        leaves the ship in a similar enough state for the suffix to work.
        """
        env = self._env
        window_len = window_end - window_start

        # Get start state info
        env.load_state(start_state)
        start_obs = env.get_observation()
        start_pickups = list(env.get_pickup_states())

        # Replay original window to get target end state
        for i in range(window_start, window_end):
            env.step(ALL_ACTIONS[full_actions[i]])
        target_obs = env.get_observation()
        target_pickups = list(env.get_pickup_states())

        # How many pickups are collected in this window?
        pickups_in_window = sum(
            1 for s, t in zip(start_pickups, target_pickups) if not s and t
        )

        # Beam: (state, action_list, pickups_collected_count)
        beam: list[tuple[object, list[int], int]] = [
            (start_state, [], 0)
        ]
        best_alt: list[int] | None = None

        for t in range(1, window_len):  # must be strictly shorter
            candidates: list[tuple[float, object, list[int], int]] = []

            for state, acts, got in beam:
                for action_idx in range(NUM_ACTIONS):
                    env.load_state(state)
                    obs, reward, terminated, truncated, info = env.step(
                        ALL_ACTIONS[action_idx]
                    )

                    if info.get("ship_exploded", False) or terminated or truncated:
                        continue

                    cur_pickups = list(env.get_pickup_states())
                    new_got = sum(
                        1 for s, c in zip(start_pickups, cur_pickups) if not s and c
                    )
                    new_acts = acts + [action_idx]
                    new_state = env.save_state()

                    # Check if we've matched the pickup requirement
                    if new_got >= pickups_in_window:
                        # Check position/velocity similarity to target
                        pos_dist = math.sqrt(
                            (obs[0] - target_obs[0]) ** 2 +
                            (obs[1] - target_obs[1]) ** 2
                        )
                        vel_dist = math.sqrt(
                            (obs[2] - target_obs[2]) ** 2 +
                            (obs[3] - target_obs[3]) ** 2
                        )
                        if pos_dist < 80.0 and vel_dist < 50.0:
                            if best_alt is None or len(new_acts) < len(best_alt):
                                best_alt = new_acts
                            continue

                    # Score: negative path distance to next pickup
                    self._pathfinder.clear_cache()
                    path_dist, _, _ = self._pathfinder.nearest_pickup_info(
                        float(obs[0]), float(obs[1]), cur_pickups,
                    )
                    score = -path_dist + new_got * 1000.0
                    candidates.append((score, new_state, new_acts, new_got))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [
                (s, a, g) for _, s, a, g in candidates[: self._mini_beam_width]
            ]

        return best_alt

    def _validate_full(self, actions: list[int]) -> bool:
        """Replay the full trajectory and check it completes the level."""
        env = self._env
        env.reset()
        for action_idx in actions:
            obs, reward, terminated, truncated, info = env.step(ALL_ACTIONS[action_idx])
            if info.get("ship_exploded", False):
                return False
            if info.get("level_completed", False):
                return True
            if terminated or truncated:
                return False
        return False
