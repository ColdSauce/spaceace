"""PPO agent for SpaceAce — wraps a trained stable-baselines3 PPO model."""

import os
from typing import Tuple, Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from spaceace.agents.base import BaseAgent, register_agent
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.core.gym_wrapper import SpaceAceGymWrapper
from spaceace.agents.ppo.training_env import SpaceAceTrainingEnv


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
        # action_repeat=1 for inference — renderer handles frame-by-frame display
        self._training_env = SpaceAceTrainingEnv(self._base_env, level=level, max_steps=max_steps,
                                                 action_repeat=1)
        self._vec_env = DummyVecEnv([lambda: self._training_env])

        norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(norm_path):
            self._vec_env = VecNormalize.load(norm_path, self._vec_env)
            self._vec_env.training = False
            self._vec_env.norm_reward = False
        else:
            self._vec_env = VecNormalize(self._vec_env, norm_obs=True, norm_reward=False, training=False)

        self._model = PPO.load(model_path)
        self._obs = None

    def reset(self) -> None:
        self._obs = self._vec_env.reset()

    def step(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action, _ = self._model.predict(self._obs, deterministic=True)
        self._obs, reward, dones, infos = self._vec_env.step(action)
        return action[0], float(reward[0]), bool(dones[0]), False, infos[0]

    def get_raw_env(self) -> SpaceAceDirectEnv:
        return self._base_env.env

    def close(self) -> None:
        self._vec_env.close()
