"""Ace agent: the solver-backed player.

On setup it loads the best known action tape for the level (the ``tas``
sidecar produced by ``scripts/solve.py``); if none exists it plans one from
scratch with the Rust beam-search solver. It then replays the tape through
the real engine tick by tick.

This is the "runtime" half of the AI — all the intelligence lives in
``src/solver.rs`` and the offline driver ``scripts/solve.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.ghost_actions import load_sidecar_actions
from spaceace.strategies.actions import ALL_ACTIONS


@register_agent("ace")
class AceAgent(BaseAgent):
    """Replay the level's solved tape; plan one with the beam solver if absent."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = int(level)
        self._replay_idx = 0
        self.debug_info: Dict[str, Any] = {}

        actions = load_sidecar_actions(level, "tas")
        if actions is None:
            width = int(kwargs.get("ace_width") or 40_000)
            print(f"[ace] no solved tape for level {level}; planning (width={width})...")
            solver = spaceace_rl.PySolver(level)
            tape = solver.solve(width=width, max_ticks=6000, mix=1.0, proj_div=300.0)
            if tape is None:
                raise RuntimeError(
                    f"ace solver found no completing tape for level {level}; "
                    "run scripts/solve.py with a bigger budget first"
                )
            tape, ticks = solver.polish(bytes(tape), iters=150_000, chains=8)
            actions = list(tape[:ticks])
            print(f"[ace] planned {ticks} ticks ({ticks / 60.0:.2f}s)")
        else:
            print(f"[ace] loaded solved tape: {len(actions)} ticks ({len(actions) / 60.0:.2f}s)")
        self._actions = actions

    def reset(self) -> None:
        self._env.reset()
        self._replay_idx = 0

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._replay_idx < len(self._actions):
            action_idx = self._actions[self._replay_idx]
            self._replay_idx += 1
        else:
            action_idx = 0
        action = ALL_ACTIONS[action_idx]
        obs, reward, terminated, truncated, info = self._env.step(action)
        return action, reward, terminated, truncated, info

    def get_raw_env(self):
        return self._env

    def close(self) -> None:
        pass
