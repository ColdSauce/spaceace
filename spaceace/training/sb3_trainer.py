"""Generic stable-baselines3 trainer. PPO today; swap PolicyCls for other SB3 algos."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from spaceace.training.callbacks import CurriculumCallback, MetricsCallback
from spaceace.training.envs import make_random_level_env, make_vec_env
from spaceace.training.trainer import Trainer, TrainingConfig


DEFAULT_PPO_HPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [64, 64]},
    verbose=1,
)


def _configure_main_process_threads(n_envs: int) -> None:
    """Give the main-process PPO update whatever perf cores the workers aren't using."""
    cpu = multiprocessing.cpu_count()
    main_threads = max(1, cpu - n_envs)
    torch.set_num_threads(main_threads)


class _LatestModelCallback(BaseCallback):
    """Periodically overwrite latest_model.zip + vec_normalize.pkl together.

    Saving vec_normalize alongside the weights keeps the dashboard's Watch
    inference stack consistent with whatever shape the policy currently has
    (otherwise a stale normalizer from a previous training session breaks
    VecNormalize.load with a shape-mismatch error).
    """
    def __init__(self, save_freq: int, save_path: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            env = self.model.get_env()
            if hasattr(env, "save"):
                norm_path = os.path.join(os.path.dirname(self.save_path), "vec_normalize.pkl")
                env.save(norm_path)
        return True


class Sb3Trainer(Trainer):
    """Default PPO trainer. Consumes a TrainingConfig, returns the saved model path.

    Previously this logic was duplicated across ppo/train.py and ppo/curriculum_train.py.
    Everything PPO-specific lives in DEFAULT_PPO_HPARAMS; strategy selection is
    driven by config.obs / config.reward.

    If ``env_factory`` is provided, it is called instead of the default
    ``make_vec_env`` to build train/eval VecEnvs.  This lets callers like
    HrlTrainer inject custom environment wrappers (e.g. WaypointPilotEnv).
    """

    algo_cls = PPO

    def __init__(
        self,
        env_factory: Callable[[TrainingConfig, int], VecEnv] | None = None,
        extra_callbacks: list[BaseCallback] | None = None,
    ):
        self._env_factory = env_factory
        self._extra_callbacks = extra_callbacks or []

    def _make_env(self, config: TrainingConfig, n_envs: int, subprocess: bool) -> VecEnv:
        if self._env_factory is not None:
            return self._env_factory(config, n_envs)
        return make_vec_env(
            level=config.level,
            max_steps=config.max_episode_steps,
            n_envs=n_envs,
            obs=config.obs,
            reward=config.reward,
            action_repeat=config.action_repeat,
            subprocess=subprocess,
            pathfinder_backend=config.pathfinder_backend,
        )

    def fit(self, config: TrainingConfig) -> Path:
        if config.curriculum is not None:
            return self._fit_curriculum(config)
        save_dir = config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        config.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        n_envs = config.n_envs or multiprocessing.cpu_count()
        _configure_main_process_threads(n_envs)

        print("=== SpaceAce RL Training ===")
        print(f"Level: {config.level}")
        print(f"Timesteps: {config.total_steps:,}")
        print(f"Max steps/episode: {config.max_episode_steps}")
        print(f"Action repeat: {config.action_repeat}")
        print(f"Parallel envs: {n_envs}")
        print(f"Save dir: {save_dir}")
        print(f"Strategies: obs={config.obs} reward={config.reward}")
        print()

        inner_train = self._make_env(config, n_envs, subprocess=True)
        norm_path = (
            os.path.join(os.path.dirname(config.resume_from), "vec_normalize.pkl")
            if config.resume_from else None
        )
        if norm_path and os.path.exists(norm_path):
            train_env = VecNormalize.load(norm_path, inner_train)
            # Reward norm intentionally off: shaped rewards are bounded and stable,
            # and a running RMS across curriculum stages produces stale scaling.
            train_env.norm_reward = False
            print(f"Loaded normalization stats from {norm_path}")
        else:
            train_env = VecNormalize(
                inner_train, norm_obs=False, norm_reward=False, clip_obs=10.0,
            )

        eval_env = self._make_env(config, 1, subprocess=False)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.0)

        if config.resume_from:
            print(f"Resuming from: {config.resume_from}")
            model = self.algo_cls.load(config.resume_from, env=train_env)
        else:
            base_lr = config.learning_rate
            lr_schedule = lambda progress: base_lr * (0.1 + 0.9 * progress)
            model = self.algo_cls(
                "MlpPolicy",
                train_env,
                tensorboard_log=str(config.tensorboard_dir),
                seed=config.seed,
                **{**DEFAULT_PPO_HPARAMS, "learning_rate": lr_schedule},
            )

        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(save_dir),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
            ),
            MetricsCallback(),
            *self._extra_callbacks,
        ]

        steps_k = config.total_steps // 1000
        run_name = f"lvl{config.level}_ar{config.action_repeat}_{steps_k}k"
        print(f"Starting training (run: {run_name})...")
        start = time.time()
        model.learn(
            total_timesteps=config.total_steps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
        )
        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")

        final_path = save_dir / "final_model"
        model.save(str(final_path))
        train_env.save(str(save_dir / "vec_normalize.pkl"))
        print(f"Saved final model to {final_path}.zip")

        train_env.close()
        eval_env.close()
        return final_path.with_suffix(".zip")

    # ------------------------------------------------------------------
    # Curriculum path
    # ------------------------------------------------------------------

    def _fit_curriculum(self, config: TrainingConfig) -> Path:
        """Train across a sequence of LevelStages, advancing on win-rate."""
        from spaceace.training.calibration import calibrate_stages

        stages = config.curriculum
        assert stages is not None

        save_dir = config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        config.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        n_envs = config.n_envs or multiprocessing.cpu_count()
        _configure_main_process_threads(n_envs)

        # Resolve any stages that need MCTS calibration.
        calibrate_stages(stages, cache=config.calibration_cache_path)

        print("=== SpaceAce Curriculum Training ===")
        print(f"Stages: {len(stages)}")
        for i, s in enumerate(stages):
            print(f"  Stage {i + 1}: levels {s.levels}, "
                  f"max_steps={s.max_episode_steps}, advance@{s.advance_win_rate:.0%}")
        print(f"Total timesteps: {config.total_steps:,}")
        print(f"Parallel envs: {n_envs}")
        print(f"Strategies: obs={config.obs} reward={config.reward}")
        print()

        # Build initial env from the first stage.
        first = stages[0]
        inner_train = self._make_curriculum_vec_env(
            first.levels, first.max_episode_steps, n_envs, config, subprocess=True,
        )
        norm_path = (
            os.path.join(os.path.dirname(config.resume_from), "vec_normalize.pkl")
            if config.resume_from else None
        )
        if norm_path and os.path.exists(norm_path):
            train_env = VecNormalize.load(norm_path, inner_train)
            train_env.norm_reward = False
            print(f"Loaded normalization stats from {norm_path}")
        else:
            train_env = VecNormalize(
                inner_train, norm_obs=False, norm_reward=False, clip_obs=10.0,
            )

        eval_env = self._make_curriculum_vec_env(
            first.levels, first.max_episode_steps, 1, config, subprocess=False,
        )
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_obs=10.0)

        if config.resume_from:
            print(f"Resuming from: {config.resume_from}")
            model = self.algo_cls.load(config.resume_from, env=train_env)
        else:
            base_lr = config.learning_rate
            lr_schedule = lambda progress: base_lr * (0.1 + 0.9 * progress)
            model = self.algo_cls(
                "MlpPolicy",
                train_env,
                tensorboard_log=str(config.tensorboard_dir),
                seed=config.seed,
                **{**DEFAULT_PPO_HPARAMS, "learning_rate": lr_schedule},
            )

        curriculum_cb = CurriculumCallback(
            stages=stages,
            make_env_fn=make_random_level_env,
            obs=config.obs,
            reward=config.reward,
            action_repeat=config.action_repeat,
            n_envs=n_envs,
            pathfinder_backend=config.pathfinder_backend,
            eval_env=eval_env,
        )

        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(save_dir),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
            ),
            _LatestModelCallback(
                save_freq=config.eval_freq,
                save_path=str(save_dir / "latest_model"),
            ),
            MetricsCallback(),
            curriculum_cb,
        ]

        run_name = f"curriculum_{config.total_steps // 1000}k"
        print(f"Starting curriculum training (run: {run_name})...")
        start = time.time()
        model.learn(
            total_timesteps=config.total_steps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
        )
        elapsed = time.time() - start
        print(f"\nCurriculum training complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")
        print(f"Reached stage {curriculum_cb._stage_idx + 1}/{len(stages)}")

        final_path = save_dir / "final_model"
        model.save(str(final_path))
        train_env.save(str(save_dir / "vec_normalize.pkl"))
        print(f"Saved final model to {final_path}.zip")

        train_env.close()
        eval_env.close()
        return final_path.with_suffix(".zip")

    @staticmethod
    def _make_curriculum_vec_env(levels, max_steps, n_envs, config, subprocess):
        thunks = [
            make_random_level_env(levels, max_steps, config.obs, config.reward, config.action_repeat, config.pathfinder_backend)
            for _ in range(n_envs)
        ]
        # CurriculumCallback uses env_method("set_curriculum", ...) which works
        # with both SubprocVecEnv and DummyVecEnv via _CurriculumMonitor.
        # DummyVecEnv is the default: on M1 with a small policy, IPC cost of
        # SubprocVecEnv dominates and we measured ~20% throughput loss from it.
        # Set SPACEACE_USE_SUBPROC=1 to opt back into subprocess workers (e.g.
        # when running on a many-core server where real parallelism wins).
        if os.environ.get("SPACEACE_USE_SUBPROC") == "1" and subprocess and n_envs > 1:
            return SubprocVecEnv(thunks, start_method="spawn")
        return DummyVecEnv(thunks)
