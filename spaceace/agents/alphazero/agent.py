"""AlphaZero agent for SpaceAce — uses Rust PUCT MCTS with neural net evaluation."""

import os
from typing import Tuple, Dict, Any

import numpy as np
import spaceace_rl

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv

# Action lookup (must match Rust ACTIONS order)
ALL_ACTIONS = [
    np.array([0, 0, 0], dtype=np.int32),  # coast
    np.array([0, 0, 1], dtype=np.int32),  # thrust
    np.array([1, 0, 0], dtype=np.int32),  # rotate left
    np.array([1, 0, 1], dtype=np.int32),  # rotate left + thrust
    np.array([0, 1, 0], dtype=np.int32),  # rotate right
    np.array([0, 1, 1], dtype=np.int32),  # rotate right + thrust
]

ACTION_NAMES = ["COAST", "THRUST", "LEFT", "LEFT+THR", "RIGHT", "RIGHT+THR"]


@register_agent("alphazero")
class AlphaZeroAgent(BaseAgent):
    """Uses Rust AlphaZero MCTS engine with neural net evaluation."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        self._env = SpaceAceDirectEnv(level=level, max_steps=max_steps)

        model_path = kwargs.get("model_path")
        if model_path is None:
            # Check curriculum model first, then per-level model
            candidates = [
                "models/alphazero/curriculum/best_model.onnx",
                f"models/alphazero/{level}/best_model.onnx",
            ]
            for path in candidates:
                if os.path.exists(path):
                    model_path = path
                    break

        self._num_simulations = kwargs.get("num_simulations", 800)
        self._c_puct = kwargs.get("c_puct", 1.5)
        self._action_repeat = kwargs.get("action_repeat", 5)

        self._engine = spaceace_rl.PyAlphaZeroEngine(level, max_steps, model_path)
        mode = "neural net" if model_path else "heuristic fallback"
        print(f"AlphaZero engine: {self._engine.get_pathfinder_info()} ({mode})")

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

        action_idx, policy, root_value = self._engine.search(
            current_state,
            self._num_simulations,
            self._c_puct,
            0.01,  # greedy for inference
            self._action_repeat,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,  # no exploration noise during inference
        )
        action = ALL_ACTIONS[action_idx]

        # Debug info
        self.debug_info = {
            "action_stats": [
                {"name": ACTION_NAMES[i], "visits": int(policy[i] * self._num_simulations), "mean_value": 0.0}
                for i in sorted(range(6), key=lambda x: -policy[x])
            ],
            "root_heuristic": float(root_value),
            "num_simulations": self._num_simulations,
            "action_repeat": self._action_repeat,
        }

        # Step real env
        self._env.load_state(current_state)
        obs, reward, terminated, truncated, info = self._env.step(action)

        self._pending_action = action
        self._pending_repeats = self._action_repeat - 1

        return action, reward, terminated, truncated, info

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._env

    def close(self) -> None:
        self._env.close()
