"""Reusable SB3 callbacks. Agnostic to the specific RL algorithm."""

from __future__ import annotations

import tempfile
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class MetricsCallback(BaseCallback):
    """Pulls `episode_metrics` out of info dicts and records them to the SB3 logger.

    Any RewardShaper that populates `episode_metrics()` flows through here for
    free — no per-agent code needed.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            for key, value in metrics.items():
                self.logger.record(f"episode/{key}", float(value))
        return True


class CurriculumCallback(BaseCallback):
    """Advance through curriculum stages based on rolling completion rate.

    Reads ``info["episode_metrics"]["completed"]`` (already populated by
    DenseShapedReward). When the smoothed rate clears the stage threshold,
    swaps the VecEnv to the next stage's levels while preserving
    VecNormalize statistics.
    """

    def __init__(
        self,
        stages: list,
        make_env_fn,
        obs: str,
        reward: str,
        action_repeat: int,
        n_envs: int,
        window: int = 3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._stages = stages
        self._make_env_fn = make_env_fn
        self._obs = obs
        self._reward = reward
        self._action_repeat = action_repeat
        self._n_envs = n_envs
        self._window = window
        self._stage_idx = 0
        self._recent: deque[float] = deque(maxlen=window)
        self._episode_completions: list[float] = []
        self._stage_start_step = 0

    @property
    def current_stage(self):
        return self._stages[self._stage_idx]

    def _on_training_start(self) -> None:
        self._stage_start_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            self._episode_completions.append(float(metrics.get("completed", 0)))

        if len(self._episode_completions) >= self._n_envs:
            batch_rate = sum(self._episode_completions) / len(self._episode_completions)
            self._recent.append(batch_rate)
            self._episode_completions.clear()
            self.logger.record("curriculum/stage", self._stage_idx)
            self.logger.record("curriculum/win_rate", batch_rate)
            if self._recent:
                self.logger.record(
                    "curriculum/smoothed_win_rate",
                    sum(self._recent) / len(self._recent),
                )

        if self._should_advance():
            self._advance_stage()

        return True

    def _should_advance(self) -> bool:
        if self._stage_idx >= len(self._stages) - 1:
            return False
        stage = self.current_stage
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < stage.min_steps:
            return False
        if len(self._recent) < self._window:
            return False
        smoothed = sum(self._recent) / len(self._recent)
        return smoothed >= stage.advance_win_rate

    def _advance_stage(self) -> None:
        self._stage_idx += 1
        if self._stage_idx >= len(self._stages):
            return
        stage = self.current_stage
        print(
            f"\n>>> Advancing to stage {self._stage_idx + 1}/{len(self._stages)}: "
            f"levels {stage.levels}, max_steps={stage.max_episode_steps}"
        )

        old_env = self.model.get_env()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            stats_path = f.name
        old_env.save(stats_path)

        from spaceace.training.envs import make_random_level_env
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

        thunks = [
            make_random_level_env(
                stage.levels, stage.max_episode_steps, self._obs, self._reward, self._action_repeat
            )
            for _ in range(self._n_envs)
        ]
        if self._n_envs > 1:
            raw_env = SubprocVecEnv(thunks, start_method="fork")
        else:
            raw_env = DummyVecEnv(thunks)

        new_env = VecNormalize.load(stats_path, raw_env)
        new_env.training = True
        new_env.norm_reward = True

        self.model.set_env(new_env)
        old_env.close()

        import os
        os.unlink(stats_path)

        self._recent.clear()
        self._episode_completions.clear()
        self._stage_start_step = self.num_timesteps
