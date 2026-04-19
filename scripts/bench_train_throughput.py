"""Micro-benchmark for env + policy training throughput.

Two modes:
  env   — raw SubprocVecEnv step throughput (isolates env/pathfinder wins)
  train — end-to-end PPO training FPS (includes gradient updates)

Presets:
  --preset new  (default)   current M1-tuned settings
  --preset old              simulate the pre-optimization config
                            (n_envs=32, n_steps=2048, batch_size=64,
                             n_epochs=10, pathfinder cache disabled)

Usage:
    uv run python scripts/bench_train_throughput.py env --preset new --steps 20000
    uv run python scripts/bench_train_throughput.py env --preset old --steps 20000
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Make `spaceace` importable when invoked as `python scripts/bench_...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def bench_env(steps: int, n_envs: int, levels: list[int], max_episode_steps: int):
    """Raw SubprocVecEnv stepping — isolates env/pathfinder/reset cost."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from spaceace.training.envs import make_random_level_env

    thunks = [
        make_random_level_env(levels, max_episode_steps, "path_augmented", "dense_shaped", 5, "grid")
        for _ in range(n_envs)
    ]
    try:
        vec = SubprocVecEnv(thunks, start_method="spawn")
    except TypeError:
        # Older versions that didn't accept start_method kwarg
        vec = SubprocVecEnv(thunks)

    print(f"[env] warm-up reset ({n_envs} envs)...")
    vec.reset()
    rng = np.random.default_rng(0)
    # MultiDiscrete([2,2,2]) = 3 binary dims
    action = rng.integers(0, 2, size=(n_envs, 3), dtype=np.int64)

    print(f"[env] stepping for {steps:,} total steps...")
    t0 = time.time()
    taken = 0
    while taken < steps:
        vec.step(action)
        taken += n_envs
        # Re-roll actions periodically so envs see variety
        if taken % (n_envs * 32) == 0:
            action = rng.integers(0, 2, size=(n_envs, 3), dtype=np.int64)
    elapsed = time.time() - t0
    vec.close()

    fps = taken / elapsed
    print()
    print(f"=== ENV BENCHMARK ===")
    print(f"Total steps: {taken:,}")
    print(f"Wall time:   {elapsed:.2f}s")
    print(f"FPS:         {fps:,.0f}")


def bench_train(
    steps: int, n_envs: int, levels: list[int], max_episode_steps: int,
    n_steps: int, batch_size: int, n_epochs: int,
    net: list[int] | None = None,
):
    """Full training stack: SubprocVecEnv + VecNormalize + PPO updates."""
    from spaceace.training import sb3_trainer as _sb3
    from spaceace.training.sb3_trainer import Sb3Trainer
    from spaceace.training.trainer import LevelStage, TrainingConfig

    # Override PPO hparams without editing module source.
    hparams = {
        **_sb3.DEFAULT_PPO_HPARAMS,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
    }
    if net is not None:
        hparams["policy_kwargs"] = {"net_arch": list(net)}
    _sb3.DEFAULT_PPO_HPARAMS = hparams

    stage = LevelStage(
        levels=levels,
        max_episode_steps=max_episode_steps,
        advance_win_rate=1.1,  # never advance
        min_steps=10**9,
    )

    with tempfile.TemporaryDirectory() as tmp:
        config = TrainingConfig(
            total_steps=steps,
            n_envs=n_envs,
            action_repeat=5,
            seed=42,
            curriculum=[stage],
            model_dir=Path(tmp) / "models",
            tensorboard_dir=Path(tmp) / "tb",
        )
        print(f"[train] steps={steps:,} n_envs={n_envs} levels={levels}")
        t0 = time.time()
        Sb3Trainer().fit(config)
        elapsed = time.time() - t0

    fps = steps / elapsed
    print()
    print(f"=== TRAIN BENCHMARK ===")
    print(f"Total steps: {steps:,}")
    print(f"Wall time:   {elapsed:.2f}s")
    print(f"FPS:         {fps:,.0f}")


PRESETS = {
    # Current default (winning combo promoted on 2026-04-15).
    "default":    dict(n_envs=16, n_steps=512,  batch_size=256, n_epochs=4,  cache=True,  net=[64, 64],   dummy=True),
    # Historical baselines, for comparison.
    "pre_m1":     dict(n_envs=6,  n_steps=512,  batch_size=256, n_epochs=4,  cache=True,  net=[256, 256], dummy=False),
    "old_server": dict(n_envs=32, n_steps=2048, batch_size=64,  n_epochs=10, cache=False, net=[256, 256], dummy=False),
    # Ablations against the pre_m1 baseline (one knob at a time).
    "dummy":      dict(n_envs=6,  n_steps=512,  batch_size=256, n_epochs=4,  cache=True,  net=[256, 256], dummy=True),
    "small_net":  dict(n_envs=6,  n_steps=512,  batch_size=256, n_epochs=4,  cache=True,  net=[64, 64],   dummy=False),
    "more_envs":  dict(n_envs=16, n_steps=512,  batch_size=256, n_epochs=4,  cache=True,  net=[256, 256], dummy=False),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["env", "train"])
    p.add_argument("--preset", choices=list(PRESETS), default="new")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--n-envs", type=int, default=None,
                   help="Override preset n_envs")
    p.add_argument("--levels", type=str, default="3000,3001,3002")
    p.add_argument("--max-episode-steps", type=int, default=500)
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.n_envs is not None:
        cfg["n_envs"] = args.n_envs
    if not cfg["cache"]:
        os.environ["SPACEACE_DISABLE_WRAPPER_CACHE"] = "1"
    else:
        os.environ.pop("SPACEACE_DISABLE_WRAPPER_CACHE", None)
    if cfg.get("dummy"):
        os.environ["SPACEACE_FORCE_DUMMY"] = "1"
    else:
        os.environ.pop("SPACEACE_FORCE_DUMMY", None)

    levels = [int(x) for x in args.levels.split(",")]
    print(f"Preset: {args.preset}  config: {cfg}")

    if args.mode == "env":
        bench_env(args.steps, cfg["n_envs"], levels, args.max_episode_steps)
    else:
        bench_train(
            args.steps, cfg["n_envs"], levels, args.max_episode_steps,
            n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"],
            net=cfg.get("net"),
        )


if __name__ == "__main__":
    main()
