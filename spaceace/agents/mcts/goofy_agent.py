"""GoofyMCTS — MCTS variant that must always have thrust on.

Restricts tree expansion to thrust-on actions only (indices 1, 3, 5:
THRUST, LEFT+THRUST, RIGHT+THRUST). Everything else matches MCTSAgent.
"""

from typing import Tuple, Dict, Any

import numpy as np

import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.strategies.actions import ALL_ACTIONS, ACTION_NAMES


@register_agent("goofymcts")
class GoofyMCTSAgent(BaseAgent):
    """MCTS agent that can never coast — thrust is always on."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)
        self._num_simulations = kwargs.get("num_simulations", 5000)
        self._exploration = kwargs.get("exploration_constant", 1.41)
        self._gamma = kwargs.get("gamma", 0.99)
        self._action_repeat = kwargs.get("action_repeat", 5)

        use_momentum = kwargs.get("momentum_pathfinder", False)
        self._mcts = spaceace_rl.PyMCTSEngine(level, max_steps, use_momentum)
        print(f"[goofymcts] Pathfinder: {self._mcts.get_pathfinder_info()}")

        self._pending_action = None
        self._pending_repeats = 0
        self.debug_info = {}

    def reset(self) -> None:
        self._env.reset()
        self._pending_action = None
        self._pending_repeats = 0
        self.debug_info = {}

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._pending_repeats > 0:
            self._pending_repeats -= 1
            obs, reward, terminated, truncated, info = self._env.step(self._pending_action)
            return self._pending_action, reward, terminated, truncated, info

        current_state = self._env.save_state()

        obs = self._env.get_observation()
        speed = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)
        min_wall_dist = float(min(obs[8:16]))

        action_repeat = self._action_repeat + int(speed / 50.0)

        num_sims = self._num_simulations
        if min_wall_dist < 150.0:
            wall_factor = 1.0 + (150.0 - min_wall_dist) / 150.0
            num_sims = int(num_sims * wall_factor)
        speed_factor = 1.0 + speed / 300.0
        num_sims = int(num_sims * speed_factor)

        action_idx, action_stats, root_heuristic = self._mcts.search_with_stats(
            current_state,
            num_sims,
            action_repeat,
            self._exploration,
            self._gamma,
            0.5,
            True,  # goofy mode
        )
        action = ALL_ACTIONS[action_idx]

        heuristic_bd = self._mcts.get_heuristic_breakdown(current_state)
        target_info = self._mcts.get_debug_target_info(current_state)

        self.debug_info = {
            "action_stats": [
                {"name": ACTION_NAMES[a], "visits": v, "mean_value": mv}
                for a, v, mv in sorted(action_stats, key=lambda x: -x[1])
            ],
            "root_heuristic": root_heuristic,
            "num_simulations": num_sims,
            "action_repeat": action_repeat,
            "heuristic": dict(heuristic_bd),
            "target": {
                "idx": target_info[0],
                "x": target_info[1],
                "y": target_info[2],
                "path_dist": target_info[3],
                "euclidean_dist": target_info[4],
                "dir_x": target_info[5],
                "dir_y": target_info[6],
            },
            "goofy": True,
        }

        self._env.load_state(current_state)
        obs, reward, terminated, truncated, info = self._env.step(action)

        self._pending_action = action
        self._pending_repeats = action_repeat - 1

        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
