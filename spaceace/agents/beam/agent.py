"""Beam search agent — finds short action sequences offline, then replays them."""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS

from spaceace.agents.beam.solver import BeamSearchSolver
from spaceace.agents.beam.optimizer import TrajectoryOptimizer


@register_agent("beam")
class BeamSearchAgent(BaseAgent):
    """Offline solver: runs beam search in setup(), replays via step().

    The solve happens in setup() (before the pygame window opens) so the
    window is responsive immediately when rendering starts.
    """

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = level
        self._max_steps = max_steps
        self._beam_width = kwargs.get("beam_width", 1000)
        self._step_penalty = kwargs.get("step_penalty", 0.01)
        self._action_repeat = kwargs.get("action_repeat", 3)
        self._optimize = kwargs.get("optimize", True)
        self._replay_idx = 0

        # Solve immediately so the window isn't blocked during reset()
        print(f"\nBeam search: level={self._level} width={self._beam_width} "
              f"action_repeat={self._action_repeat} "
              f"penalty={self._step_penalty} optimize={self._optimize}")

        self._env.reset()
        solver = BeamSearchSolver(
            env=self._env,
            level=self._level,
            beam_width=self._beam_width,
            max_steps=self._max_steps,
            step_penalty=self._step_penalty,
            action_repeat=self._action_repeat,
        )
        self._solution: list[int] = solver.solve()

        if self._optimize and self._solution:
            optimizer = TrajectoryOptimizer(env=self._env, level=self._level)
            self._solution = optimizer.optimize(self._solution)

        if self._solution:
            print(f"Solution ready: {len(self._solution)} steps")
        else:
            print("No solution found!")

    def reset(self) -> None:
        self._env.reset()
        self._replay_idx = 0

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._replay_idx >= len(self._solution):
            action = ALL_ACTIONS[0]
            obs, reward, terminated, truncated, info = self._env.step(action)
            return action, reward, terminated, truncated, info

        action_idx = self._solution[self._replay_idx]
        action = ALL_ACTIONS[action_idx]
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._replay_idx += 1
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
