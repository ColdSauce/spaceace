"""A* planner agent — runs the solver in setup(), replays per-frame in step().

Mirrors the structure of :class:`spaceace.agents.beam.agent.BeamSearchAgent`:
offline solve up front so the pygame window stays responsive during playback.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from spaceace.agents.astar.solver import AStarSolver
from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS


@register_agent("astar")
class AStarAgent(BaseAgent):
    """Offline A* planner with live replay (outer TSP + inner kinodynamic A*)."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = level
        self._max_steps = max_steps
        self._replay_idx = 0

        solver = AStarSolver(
            env=self._env,
            level=level,
            action_repeat=int(kwargs.get("astar_action_repeat",
                                         kwargs.get("action_repeat", 4))),
            pos_bucket=float(kwargs.get("astar_pos_bucket", 8.0)),
            vel_bucket=float(kwargs.get("astar_vel_bucket", 8.0)),
            rot_bucket_deg=float(kwargs.get("astar_rot_bucket_deg", 10.0)),
            leg_max_expansions=int(kwargs.get("astar_leg_max_expansions", 200_000)),
            leg_time_limit_s=float(kwargs.get("astar_leg_time_limit_s", 60.0)),
            heuristic_weight=float(kwargs.get("astar_heuristic_weight", 1.0)),
            max_steps=max_steps,
            verbose=bool(kwargs.get("astar_verbose", True)),
        )
        self._solution: list[int] = solver.solve()

        if self._solution:
            print(f"A* solution ready: {len(self._solution)} frames")
        else:
            print("A*: no solution found.")

    def reset(self) -> None:
        self._env.reset()
        self._replay_idx = 0

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._replay_idx >= len(self._solution):
            action = ALL_ACTIONS[0]
        else:
            action = ALL_ACTIONS[self._solution[self._replay_idx]]
            self._replay_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
