"""PPO agent for SpaceAce — wraps a trained stable-baselines3 PPO model."""

import os
from typing import Tuple, Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.strategies.actions import ALL_ACTIONS
from spaceace.training.envs import StrategyWrapper, _build_strategies


@register_agent("ppo")
class PPOAgent(BaseAgent):
    """Loads a trained PPO model and runs inference."""

    def setup(self, level: int, max_steps: int, **kwargs) -> None:
        model_path = kwargs.get("model_path")
        if model_path is None:
            # Check curriculum model first, then per-level model
            candidates = [
                "models/ppo/curriculum/best_model",
                f"models/{level}/best_model",
            ]
            for path in candidates:
                if os.path.exists(path + ".zip"):
                    model_path = path
                    break
            if model_path is None:
                model_path = f"models/{level}/best_model"

        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(
                f"Model not found: {model_path}.zip\n"
                f"Train one first: python train.py --level {level}"
            )

        self._base_env = SpaceAceGymWrapper(level=level, max_steps=max_steps)
        obs_strategy, reward_strategy, pf = _build_strategies(level, max_steps, "path_augmented", "dense_shaped")
        # action_repeat must match training (default 5) for correct policy behavior
        self._wrapped_env = StrategyWrapper(self._base_env, obs_strategy, reward_strategy, action_repeat=5, pathfinder=pf)
        self._vec_env = DummyVecEnv([lambda: self._wrapped_env])

        norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        loaded_norm = False
        if os.path.exists(norm_path):
            try:
                self._vec_env = VecNormalize.load(norm_path, self._vec_env)
                self._vec_env.training = False
                self._vec_env.norm_reward = False
                loaded_norm = True
            except AssertionError:
                print(f"[ppo] vec_normalize.pkl shape mismatch, skipping")
        if not loaded_norm:
            self._vec_env = VecNormalize(self._vec_env, norm_obs=False, norm_reward=False, training=False)

        self._model = PPO.load(model_path)
        self._obs = None

    def reset(self) -> None:
        self._obs = self._vec_env.reset()
        self._step_count = 0
        self._diag_logged = False

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action, _ = self._model.predict(self._obs, deterministic=True)

        # One-time diagnostic dump at the start of each episode.
        if not self._diag_logged:
            self._diag_logged = True
            self._log_inference_diag(action)

        self._obs, reward, dones, infos = self._vec_env.step(action)
        self._step_count += 1
        # Decode Discrete(6) index → MultiDiscrete triplet for the renderer
        # (dashboard JS expects [rot_left, rot_right, thrust]).
        decoded = ALL_ACTIONS[int(np.asarray(action[0]))]
        return decoded, float(reward[0]), bool(dones[0]), False, infos[0]

    def _log_inference_diag(self, action) -> None:
        """Dump diagnostic info at episode start for debugging stuck levels."""
        obs = self._obs[0]
        vec_env = self._vec_env

        # VecNormalize settings
        norm_obs = getattr(vec_env, "norm_obs", "N/A")
        norm_reward = getattr(vec_env, "norm_reward", "N/A")
        print(f"\n[ppo-diag] VecNormalize: norm_obs={norm_obs} norm_reward={norm_reward}")

        # Key observation features
        if len(obs) >= 24:
            print(f"[ppo-diag] obs: dir=({obs[17]:+.4f},{obs[18]:+.4f}) "
                  f"path_dist={obs[16]:.4f} pickups_rem={obs[13]:.1f} "
                  f"speed={obs[19]:.2f}")
            print(f"[ppo-diag] obs: wall_fwd={obs[5]:.3f} wall_r={obs[7]:.3f} "
                  f"wall_bk={obs[9]:.3f} wall_l={obs[11]:.3f}")

        print(f"[ppo-diag] first action: {action}")

    def get_raw_env(self):
        return self._base_env.env

    def close(self) -> None:
        self._vec_env.close()
