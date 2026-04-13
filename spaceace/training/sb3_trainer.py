"""Generic stable-baselines3 trainer. PPO today; swap PolicyCls for other SB3 algos."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from spaceace.training.callbacks import MetricsCallback
from spaceace.training.envs import make_vec_env
from spaceace.training.trainer import Trainer, TrainingConfig


DEFAULT_PPO_HPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
)


class Sb3Trainer(Trainer):
    """Default PPO trainer. Consumes a TrainingConfig, returns the saved model path.

    Previously this logic was duplicated across ppo/train.py and ppo/curriculum_train.py.
    Everything PPO-specific lives in DEFAULT_PPO_HPARAMS; strategy selection is
    driven by config.obs / config.reward.
    """

    algo_cls = PPO

    def fit(self, config: TrainingConfig) -> Path:
        save_dir = config.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        config.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        n_envs = config.n_envs or multiprocessing.cpu_count()

        print("=== SpaceAce RL Training ===")
        print(f"Level: {config.level}")
        print(f"Timesteps: {config.total_steps:,}")
        print(f"Max steps/episode: {config.max_episode_steps}")
        print(f"Action repeat: {config.action_repeat}")
        print(f"Parallel envs: {n_envs}")
        print(f"Save dir: {save_dir}")
        print(f"Strategies: obs={config.obs} reward={config.reward}")
        print()

        train_env = make_vec_env(
            level=config.level,
            max_steps=config.max_episode_steps,
            n_envs=n_envs,
            obs=config.obs,
            reward=config.reward,
            action_repeat=config.action_repeat,
            subprocess=True,
        )
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
        )

        eval_env = make_vec_env(
            level=config.level,
            max_steps=config.max_episode_steps,
            n_envs=1,
            obs=config.obs,
            reward=config.reward,
            action_repeat=config.action_repeat,
            subprocess=False,
        )
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        if config.resume_from:
            print(f"Resuming from: {config.resume_from}")
            model = self.algo_cls.load(config.resume_from, env=train_env)
            norm_path = os.path.join(os.path.dirname(config.resume_from), "vec_normalize.pkl")
            if os.path.exists(norm_path):
                train_env = VecNormalize.load(norm_path, train_env.venv)
                print(f"Loaded normalization stats from {norm_path}")
        else:
            model = self.algo_cls(
                "MlpPolicy",
                train_env,
                tensorboard_log=str(config.tensorboard_dir),
                seed=config.seed,
                **{**DEFAULT_PPO_HPARAMS, "learning_rate": config.learning_rate},
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
