"""BaseAgent wrapper for the kinodynamic planner.

Runs the three-layer pipeline (ATSP -> reference trajectory -> cascaded PD
tracker) once in ``setup()`` and replays the resulting action sequence
during ``step()`` so the pygame window stays responsive.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.agents.kinodyn.solver import KinodynSolver
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS


@register_agent("kinodyn")
class KinodynAgent(BaseAgent):
    """Gravity-aware ATSP + phase-space trajectory + cascaded PD tracker."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = level
        self._max_steps = max_steps
        self._replay_idx = 0

        self._solver = KinodynSolver(
            env=self._env,
            level=level,
            ds=float(kwargs.get("kinodyn_ds", 6.0)),
            smooth_sigma_samples=float(kwargs.get("kinodyn_smooth_sigma", 3.0)),
            a_lat_max=float(kwargs.get("kinodyn_a_lat", 220.0)),
            v_cap=float(kwargs.get("kinodyn_v_cap", 500.0)),
            v_final=float(kwargs.get("kinodyn_v_final", 180.0)),
            kp_pos=float(kwargs.get("kinodyn_kp_pos", 6.0)),
            kd_vel=float(kwargs.get("kinodyn_kd_vel", 3.2)),
            rot_tolerance_thrust_deg=float(kwargs.get("kinodyn_rot_tol_deg", 18.0)),
            thrust_deadband_accel=float(kwargs.get("kinodyn_thrust_deadband", 40.0)),
            lookahead_samples=int(kwargs.get("kinodyn_lookahead_samples", 4)),
            enumerate_orders=bool(kwargs.get("kinodyn_enumerate_orders", True)),
            enumerate_threshold=int(kwargs.get("kinodyn_enumerate_threshold", 5)),
            max_tick_budget=int(kwargs.get("kinodyn_tick_budget", 6000)),
            max_idle_frames=int(kwargs.get("kinodyn_max_idle", 600)),
            verbose=bool(kwargs.get("kinodyn_verbose", True)),
        )
        self._solution: list[int] = self._solver.solve()

        if self._solution:
            print(
                f"Kinodyn solution ready: {len(self._solution)} frames "
                f"= {len(self._solution) / 60.0:.2f}s"
            )
        else:
            print("Kinodyn: no solution found.")

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
