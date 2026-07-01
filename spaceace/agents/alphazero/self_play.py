"""Self-play data generation using AlphaZero MCTS."""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import spaceace_rl
from spaceace.core.env import SpaceAceDirectEnv


@dataclass
class GameExample:
    observation: np.ndarray   # 27-dim
    mcts_policy: np.ndarray   # 6-dim (visit count distribution)
    value_target: float = 0.0 # heuristic-based value in [-1, 1]


@dataclass
class GameStats:
    total_reward: float = 0.0
    pickups_collected: int = 0
    completed: bool = False
    crashed: bool = False
    steps: int = 0


def play_one_game(
    engine: spaceace_rl.PyAlphaZeroEngine,
    env: SpaceAceDirectEnv,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    action_repeat: int = 5,
    temp_threshold: int = 30,
    max_steps: int = 3000,
) -> tuple[list[GameExample], GameStats]:
    """Play a single self-play game, returning training examples and stats."""
    env.reset()
    examples: list[GameExample] = []
    stats = GameStats()
    step = 0
    prev_pickups = None

    while True:
        state = env.save_state()
        temperature = 1.0 if step < temp_threshold else 0.1

        action_idx, policy, value = engine.search(
            state, num_simulations, c_puct, temperature, action_repeat,
        )

        # Store observation and policy
        obs = engine.get_observation(state)
        examples.append(GameExample(
            observation=np.array(obs, dtype=np.float32),
            mcts_policy=np.array(policy, dtype=np.float32),
        ))

        # Execute action with action_repeat
        action = _action_from_index(action_idx)
        terminated = False
        truncated = False
        for _ in range(action_repeat):
            _, reward, terminated, truncated, info = env.step(action)
            stats.total_reward += reward
            step += 1

            # Track pickups
            pickups_now = info.get("pickups_remaining", prev_pickups)
            if prev_pickups is not None and pickups_now < prev_pickups:
                stats.pickups_collected += prev_pickups - pickups_now
            prev_pickups = pickups_now

            if terminated or truncated or step >= max_steps:
                break

        # Compute per-step value target from heuristic evaluation
        post_state = env.save_state()
        if info.get("level_completed", False):
            heuristic_value = 1.0
        elif info.get("ship_exploded", False):
            heuristic_value = -1.0
        else:
            heuristic_value = float(engine.evaluate_heuristic(post_state))
        examples[-1].value_target = heuristic_value

        if terminated or truncated or step >= max_steps:
            if info.get("level_completed", False):
                stats.completed = True
            elif info.get("ship_exploded", False):
                stats.crashed = True

            stats.steps = step
            break

    return examples, stats


# Action lookup (must match Rust ACTIONS order)
_ALL_ACTIONS = [
    np.array([0, 0, 0], dtype=np.int32),  # coast
    np.array([0, 0, 1], dtype=np.int32),  # thrust
    np.array([1, 0, 0], dtype=np.int32),  # rotate left
    np.array([1, 0, 1], dtype=np.int32),  # rotate left + thrust
    np.array([0, 1, 0], dtype=np.int32),  # rotate right
    np.array([0, 1, 1], dtype=np.int32),  # rotate right + thrust
]


def _action_from_index(idx: int) -> np.ndarray:
    return _ALL_ACTIONS[idx]


OBS_DIM = 27
NUM_ACTIONS = 6


def _unpack_rust_results(
    obs_flat: list[float],
    pol_flat: list[float],
    values: list[float],
    stats_raw: list[tuple],
) -> tuple[list[GameExample], list[GameStats]]:
    """Convert flat Rust output arrays into GameExample/GameStats lists."""
    n = len(values)
    obs_arr = np.array(obs_flat, dtype=np.float32).reshape(n, OBS_DIM)
    pol_arr = np.array(pol_flat, dtype=np.float32).reshape(n, NUM_ACTIONS)
    val_arr = np.array(values, dtype=np.float32)

    examples = [
        GameExample(observation=obs_arr[i], mcts_policy=pol_arr[i], value_target=float(val_arr[i]))
        for i in range(n)
    ]
    stats = [
        GameStats(total_reward=r, pickups_collected=p, completed=c, crashed=cr, steps=s)
        for r, p, c, cr, s in stats_raw
    ]
    return examples, stats


def _worker_play_games(args: tuple) -> tuple[list[float], list[float], list[float], list[tuple]]:
    """Worker function for parallel self-play. Runs entirely in Rust."""
    (level, num_games, num_simulations, c_puct, action_repeat, max_steps,
     model_path, temp_threshold, dirichlet_alpha, dirichlet_epsilon,
     temperature_after_threshold, rng_seed) = args
    engine = spaceace_rl.PyAlphaZeroEngine(level, max_steps, model_path)
    engine.set_rng_seed(rng_seed)
    obs_flat, pol_flat, values, stats = engine.play_games(
        num_games, num_simulations, c_puct, action_repeat,
        temp_threshold=temp_threshold, max_steps=max_steps,
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        temperature_after_threshold=temperature_after_threshold,
    )
    return obs_flat, pol_flat, values, stats


def run_self_play(
    level: int,
    num_games: int,
    num_simulations: int = 400,
    c_puct: float = 1.5,
    action_repeat: int = 5,
    max_steps: int = 3000,
    model_path: Optional[str] = None,
    num_workers: int = 0,
    temp_threshold: int = 30,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    temperature_after_threshold: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[list[GameExample], list[GameStats]]:
    """Run multiple self-play games. Returns (examples, per_game_stats).

    Games run entirely in Rust, parallelized across processes.
    """
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    base_seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")

    def worker_seed(worker_idx: int) -> int:
        return (base_seed + 0x9E3779B9 * (worker_idx + 1)) & 0xFFFFFFFF or 1

    if num_workers == 1 or num_games <= 2:
        # Sequential: single Rust call
        engine = spaceace_rl.PyAlphaZeroEngine(level, max_steps, model_path)
        engine.set_rng_seed(worker_seed(0))
        obs_flat, pol_flat, values, stats_raw = engine.play_games(
            num_games, num_simulations, c_puct, action_repeat,
            temp_threshold=temp_threshold, max_steps=max_steps,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
            temperature_after_threshold=temperature_after_threshold,
        )
        examples, stats = _unpack_rust_results(obs_flat, pol_flat, values, stats_raw)
    else:
        # Distribute games across workers
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1
        games_per_worker = [g for g in games_per_worker if g > 0]

        worker_args = [
            (level, g, num_simulations, c_puct, action_repeat, max_steps,
             model_path, temp_threshold, dirichlet_alpha, dirichlet_epsilon,
             temperature_after_threshold, worker_seed(i))
            for i, g in enumerate(games_per_worker)
        ]

        print(f"  Parallelizing across {len(worker_args)} workers...")

        all_obs: list[float] = []
        all_pol: list[float] = []
        all_val: list[float] = []
        all_stats_raw: list[tuple] = []

        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            for obs_flat, pol_flat, values, stats_raw in executor.map(_worker_play_games, worker_args):
                all_obs.extend(obs_flat)
                all_pol.extend(pol_flat)
                all_val.extend(values)
                all_stats_raw.extend(stats_raw)

        examples, stats = _unpack_rust_results(all_obs, all_pol, all_val, all_stats_raw)

    wins = sum(1 for s in stats if s.completed)
    crashes = sum(1 for s in stats if s.crashed)
    mean_reward = sum(s.total_reward for s in stats) / len(stats)
    mean_pickups = sum(s.pickups_collected for s in stats) / len(stats)
    print(f"  Games: {num_games}/{num_games}, "
          f"examples: {len(examples)}, "
          f"wins: {wins}, crashes: {crashes}, "
          f"mean_reward: {mean_reward:.1f}, mean_pickups: {mean_pickups:.1f}")

    return examples, stats


def save_examples(examples: list[GameExample], path: str):
    """Save training examples to .npz file."""
    observations = np.array([e.observation for e in examples])
    policies = np.array([e.mcts_policy for e in examples])
    values = np.array([e.value_target for e in examples])
    np.savez_compressed(path, observations=observations, policies=policies, values=values)


def load_examples(path: str) -> list[GameExample]:
    """Load training examples from .npz file."""
    data = np.load(path)
    # Support both old format (outcomes) and new format (values)
    value_key = "values" if "values" in data else "outcomes"
    examples = []
    for obs, pol, val in zip(data["observations"], data["policies"], data[value_key]):
        examples.append(GameExample(observation=obs, mcts_policy=pol, value_target=float(val)))
    return examples
