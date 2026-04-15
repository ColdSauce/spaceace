"""Reusable SB3 callbacks. Agnostic to the specific RL algorithm."""

from __future__ import annotations

from collections import deque

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


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
        pathfinder_backend: str = "grid",
        window: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._stages = stages
        self._make_env_fn = make_env_fn
        self._obs = obs
        self._reward = reward
        self._action_repeat = action_repeat
        self._n_envs = n_envs
        self._pathfinder_backend = pathfinder_backend
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
        self._pending_advance = False

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
            self._pending_advance = True

        return True

    def _on_rollout_start(self) -> None:
        """Swap env between rollouts so training never sees mixed data."""
        if self._pending_advance:
            self._pending_advance = False
            self._advance_stage()

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

        # Update the level pool on each RandomLevelEnv inside the VecEnv.
        # No env teardown/rebuild — just change what levels get sampled on reset.
        # Uses env_method() which works with both DummyVecEnv and SubprocVecEnv.
        vec_env = self.model.get_env()
        raw_vec = vec_env.venv if hasattr(vec_env, 'venv') else vec_env
        raw_vec.env_method("set_curriculum", stage.levels, stage.max_episode_steps)

        # Reset so the new levels take effect immediately
        self.model._last_obs = vec_env.reset()
        self.model._last_episode_starts = np.ones((self._n_envs,), dtype=bool)

        self._recent.clear()
        self._episode_completions.clear()
        self._stage_start_step = self.num_timesteps
