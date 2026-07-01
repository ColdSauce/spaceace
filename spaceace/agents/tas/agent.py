"""Exact action-trace replay agent.

This is the runtime half of the TAS/ghost workflow: offline search or manual
polishing produces a per-physics-tick sidecar, and this agent replays it
deterministically through the real engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.ghost_actions import load_action_file, sidecar_path
from spaceace.strategies.actions import ALL_ACTIONS


@register_agent("tas")
class TASReplayAgent(BaseAgent):
    """Replay an exact per-tick action trace from ``ghost_actions/``."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._level = int(level)
        self._max_steps = int(max_steps)
        self._replay_idx = 0
        self.debug_info: Dict[str, Any] = {}

        label = str(kwargs.get("tas_label") or kwargs.get("ghost_label") or "ai")
        trace_path_arg = kwargs.get("tas_path")
        trace_path = Path(trace_path_arg).expanduser() if trace_path_arg else sidecar_path(level, label)
        self._trace_path = trace_path

        if not trace_path.exists():
            raise FileNotFoundError(
                f"No TAS action sidecar found at {trace_path}. "
                "Generate one with scripts/tas_polish.py --dump-json or by "
                "capturing a completed run with --save-ghost."
            )

        trace_level, actions = load_action_file(trace_path)
        if trace_level is not None and int(trace_level) != self._level:
            raise ValueError(
                f"{trace_path} declares level {trace_level}, expected level {self._level}"
            )
        if not actions:
            raise ValueError(f"{trace_path} contains no actions")

        self._actions = actions
        print(
            f"[tas] loaded {len(self._actions)} actions "
            f"({len(self._actions) / 60.0:.2f}s) from {self._trace_path}"
        )

        if bool(kwargs.get("tas_validate", False)):
            ticks, completed = self._validate_trace()
            if not completed:
                raise ValueError(
                    f"{trace_path} did not complete level {self._level} "
                    f"when replayed ({ticks} ticks)"
                )
            print(f"[tas] validated completion in {ticks} ticks ({ticks / 60.0:.2f}s)")
            self._env.reset()

    def _validate_trace(self) -> tuple[int, bool]:
        self._env.reset()
        for tick, action_idx in enumerate(self._actions, start=1):
            _obs, _reward, terminated, truncated, info = self._env.step(ALL_ACTIONS[action_idx])
            if info.get("level_completed"):
                return tick, True
            if terminated or truncated:
                return tick, False
        return len(self._actions), self._env.get_pickups_remaining() == 0

    def reset(self) -> None:
        self._env.reset()
        self._replay_idx = 0
        self.debug_info = {}

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._replay_idx < len(self._actions):
            action_idx = self._actions[self._replay_idx]
            self._replay_idx += 1
        else:
            action_idx = 0

        action = ALL_ACTIONS[action_idx]
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.debug_info = {
            "trace_path": str(self._trace_path),
            "trace_index": self._replay_idx,
            "trace_len": len(self._actions),
            "action_idx": action_idx,
        }
        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
