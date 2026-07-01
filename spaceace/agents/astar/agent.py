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
    """Offline whole-level kinodynamic A* planner with live replay."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = level
        self._max_steps = max_steps
        self._replay_idx = 0
        self._fallback_agent: BaseAgent | None = None

        solver = AStarSolver(
            env=self._env,
            level=level,
            action_repeat=int(kwargs.get("astar_action_repeat",
                                         kwargs.get("action_repeat", 10))),
            pos_bucket=float(kwargs.get("astar_pos_bucket", 16.0)),
            vel_bucket=float(kwargs.get("astar_vel_bucket", 16.0)),
            rot_bucket_deg=float(kwargs.get("astar_rot_bucket_deg", 15.0)),
            max_expansions=int(
                kwargs.get(
                    "astar_max_expansions",
                    kwargs.get("astar_leg_max_expansions", 200_000),
                )
            ),
            time_limit_s=float(
                kwargs.get(
                    "astar_time_limit_s",
                    kwargs.get("astar_leg_time_limit_s", 60.0),
                )
            ),
            heuristic_weight=float(kwargs.get("astar_heuristic_weight", 2.0)),
            max_steps=max_steps,
            verbose=bool(kwargs.get("astar_verbose", True)),
        )
        self._solution: list[int] = solver.solve()

        if self._solution:
            print(f"A* solution ready: {len(self._solution)} frames")
        else:
            print("A*: no solution found.")
            self._setup_fallback(level, max_steps, kwargs)

    def _setup_fallback(self, level: int, max_steps: int, kwargs: dict) -> None:
        fallback = str(kwargs.get("astar_fallback", "mcts")).strip().lower()
        if fallback in {"", "none", "off", "false", "0"}:
            return
        if fallback != "mcts":
            raise ValueError(f"Unsupported A* fallback policy: {fallback!r}")

        from spaceace.agents.mcts.agent import MCTSAgent

        fallback_kwargs = dict(kwargs)
        fallback_kwargs["num_simulations"] = int(
            kwargs.get(
                "astar_fallback_simulations",
                max(500, int(kwargs.get("num_simulations", 0) or 0)),
            )
        )
        fallback_kwargs["action_repeat"] = int(
            kwargs.get("astar_fallback_action_repeat", 5)
        )

        print(
            "A*: falling back to MCTS "
            f"({fallback_kwargs['num_simulations']} sims, "
            f"action_repeat={fallback_kwargs['action_repeat']})."
        )
        self._fallback_agent = MCTSAgent()
        self._fallback_agent.setup(level=level, max_steps=max_steps, **fallback_kwargs)

    def reset(self) -> None:
        if self._fallback_agent is not None:
            self._fallback_agent.reset()
            return
        self._env.reset()
        self._replay_idx = 0

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._fallback_agent is not None:
            return self._fallback_agent.step()
        if self._replay_idx >= len(self._solution):
            action = ALL_ACTIONS[0]
        else:
            action = ALL_ACTIONS[self._solution[self._replay_idx]]
            self._replay_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        if self._fallback_agent is not None:
            return self._fallback_agent.get_raw_env()
        return self._env

    def close(self) -> None:
        if self._fallback_agent is not None:
            self._fallback_agent.close()
        self._env.close()
