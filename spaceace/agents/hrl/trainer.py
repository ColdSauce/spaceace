"""HRL Trainer: waypoint pilot trained via PPO, delegating to Sb3Trainer."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from spaceace.agents.hrl.waypoint_env import WaypointMetricsCallback, make_waypoint_env
from spaceace.training.sb3_trainer import Sb3Trainer
from spaceace.training.trainer import Trainer, TrainingConfig


def _parse_level_spec(spec: str) -> list[int]:
    """Parse level spec like '4000-4099' or '4000' into list of ints."""
    if "-" in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


class HrlTrainer(Trainer):
    """Waypoint pilot trainer. Wraps Sb3Trainer with a custom env factory
    that produces WaypointPilotEnv environments instead of StrategyWrapper envs.
    """

    def __init__(self, levels: list[int] | None = None):
        self._levels = levels

    def _resolve_levels(self, config: TrainingConfig) -> list[int]:
        if self._levels is not None:
            return self._levels
        if config.curriculum is not None:
            levels = []
            for stage in config.curriculum:
                levels.extend(stage.levels)
            return sorted(set(levels))
        return [config.level]

    def _ensure_corridors(self, levels: list[int]) -> None:
        """Generate corridor levels if they don't exist on disk."""
        needs_generation = any(lvl >= 4000 for lvl in levels)
        if not needs_generation:
            return

        from spaceace.tools.generate_corridors import generate_all_corridors
        import json
        import os

        levels_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data", "spaceace_levels.json"
        )
        levels_path = os.path.normpath(levels_path)

        # Check if corridor levels already exist
        if os.path.exists(levels_path):
            with open(levels_path, "r") as f:
                all_levels = json.load(f)
            if all(str(lvl) in all_levels for lvl in levels if lvl >= 4000):
                return

        print("Generating corridor levels...")
        corridor_levels = generate_all_corridors()

        if os.path.exists(levels_path):
            with open(levels_path, "r") as f:
                all_levels = json.load(f)
        else:
            all_levels = {}

        all_levels = {k: v for k, v in all_levels.items() if k.startswith("_") or int(k) < 4000}
        all_levels.update(corridor_levels)

        with open(levels_path, "w") as f:
            json.dump(all_levels, f)
        print(f"Generated {len(corridor_levels)} corridor levels.")

    def _make_waypoint_vec_env(
        self, config: TrainingConfig, n_envs: int
    ) -> VecEnv:
        levels = self._resolve_levels(config)
        thunks = []
        for i in range(n_envs):
            level = levels[i % len(levels)]
            thunks.append(
                make_waypoint_env(
                    level,
                    config.max_episode_steps,
                    config.action_repeat,
                    flatten_actions=True,
                )
            )
        return DummyVecEnv(thunks)

    def fit(self, config: TrainingConfig) -> Path:
        levels = self._resolve_levels(config)
        self._ensure_corridors(levels)

        print(f"=== HRL Waypoint Pilot Training (PPO) ===")
        print(f"Levels: {levels[0]}-{levels[-1]} ({len(levels)} total)")
        print()

        sb3 = Sb3Trainer(
            env_factory=self._make_waypoint_vec_env,
            extra_callbacks=[WaypointMetricsCallback()],
        )
        return sb3.fit(config)
