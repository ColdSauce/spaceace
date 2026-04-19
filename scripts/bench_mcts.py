"""Seeded, multi-episode benchmark for the MCTS agent.

Compares configs on the same set of (level, episode_seed) tuples — every config
sees identical starting conditions and identical MCTS RNG, so differences in
outcomes reflect the config itself rather than luck.

Reports mean ± stderr of steps (over completed episodes) and completion rate
per (level, config). Also a level-marginal summary.

Example:
    uv run python scripts/bench_mcts.py \\
        --levels 4 6 7 --episodes 10 --num-simulations 3000

Custom configs can be appended by extending CONFIGS below or via --configs
(comma-separated list of names from the CONFIGS dict).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import spaceace_rl  # noqa: E402
import spaceace.agents  # noqa: E402,F401 — populates AGENT_REGISTRY
from spaceace.agents.base import AGENT_REGISTRY  # noqa: E402


@dataclass
class Config:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


# Note: baseline now *includes* early-exit (matches the shipping defaults).
# `no_ee` reproduces the pre-shipping full-budget search for comparison.
CONFIGS: dict[str, Config] = {
    "baseline": Config("baseline", {}),
    "no_ee": Config("no_ee", {"early_exit_check_every": 0}),
    "ee_500_60_5": Config("ee_500_60_5", {
        "early_exit_check_every": 500,
        "early_exit_visit_frac": 0.6,
        "early_exit_q_gap": 5.0,
    }),
    "ee_500_70_10": Config("ee_500_70_10", {
        "early_exit_check_every": 500,
        "early_exit_visit_frac": 0.7,
        "early_exit_q_gap": 10.0,
    }),
}


@dataclass
class EpisodeResult:
    level: int
    config: str
    seed: int
    steps: int
    outcome: str  # "completed" | "crashed" | "truncated"
    elapsed_s: float


def run_one(level: int, config: Config, seed: int, num_simulations: int, max_steps: int) -> EpisodeResult:
    """Run a single episode with a fixed RNG seed. Returns step count + outcome."""
    # Seed BEFORE agent construction — PyMCTSEngine uses the thread-local RNG.
    spaceace_rl.set_rng_seed(seed)

    agent = AGENT_REGISTRY["mcts"]()
    setup_kwargs = {
        "num_simulations": num_simulations,
        "exploration_constant": 1.41,
        "action_repeat": 5,
    }
    setup_kwargs.update(config.kwargs)
    agent.setup(level=level, max_steps=max_steps, **setup_kwargs)
    agent.reset()

    # Re-seed again after setup — agent.setup() can call into Rust and would
    # otherwise consume seed entropy before the first search. This makes the
    # first MCTS decision deterministic per (level, seed).
    spaceace_rl.set_rng_seed(seed)

    t0 = time.time()
    step_count = 0
    info: dict = {}
    while True:
        _action, _reward, terminated, truncated, info = agent.step()
        step_count += 1
        if terminated or truncated:
            break
    elapsed = time.time() - t0

    if info.get("level_completed"):
        outcome = "completed"
    elif info.get("ship_exploded"):
        outcome = "crashed"
    else:
        outcome = "truncated"

    agent.close()
    return EpisodeResult(
        level=level, config=config.name, seed=seed,
        steps=step_count, outcome=outcome, elapsed_s=elapsed,
    )


def summarize(results: list[EpisodeResult]) -> None:
    """Print per-(level, config) and per-config summary tables.

    Uses only *completed* episodes for step means — a crash at step 200 and
    a completion in 1800 steps aren't commensurable. Completion rate is
    reported separately so regressions that trade speed for reliability
    (or vice versa) are visible.
    """
    by_key: dict[tuple[int, str], list[EpisodeResult]] = {}
    for r in results:
        by_key.setdefault((r.level, r.config), []).append(r)

    print("\n=== Per-level, per-config ===")
    print(f"{'level':>5}  {'config':<18}  {'n':>3}  {'compl':>6}  "
          f"{'steps (completed)':>22}  {'time/ep':>9}")
    for (level, config), rs in sorted(by_key.items()):
        n = len(rs)
        completed = [r for r in rs if r.outcome == "completed"]
        crate = len(completed) / n
        if completed:
            steps = [r.steps for r in completed]
            mean_s = sum(steps) / len(steps)
            if len(steps) >= 2:
                var = sum((s - mean_s) ** 2 for s in steps) / (len(steps) - 1)
                stderr = math.sqrt(var / len(steps))
                step_str = f"{mean_s:7.1f} ± {stderr:5.1f} (n={len(steps)})"
            else:
                step_str = f"{mean_s:7.1f}           (n=1)"
        else:
            step_str = "—                     "
        mean_t = sum(r.elapsed_s for r in rs) / n
        print(f"{level:>5}  {config:<18}  {n:>3}  {crate*100:>5.0f}%  "
              f"{step_str:>22}  {mean_t:>7.1f}s")

    # Marginalize across levels
    print("\n=== Per-config (marginal over levels) ===")
    by_config: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)
    print(f"{'config':<18}  {'n':>3}  {'compl':>6}  {'steps (completed)':>22}  {'time/ep':>9}")
    for config, rs in sorted(by_config.items()):
        n = len(rs)
        completed = [r for r in rs if r.outcome == "completed"]
        crate = len(completed) / n
        if completed:
            steps = [r.steps for r in completed]
            mean_s = sum(steps) / len(steps)
            if len(steps) >= 2:
                var = sum((s - mean_s) ** 2 for s in steps) / (len(steps) - 1)
                stderr = math.sqrt(var / len(steps))
                step_str = f"{mean_s:7.1f} ± {stderr:5.1f} (n={len(steps)})"
            else:
                step_str = f"{mean_s:7.1f}           (n=1)"
        else:
            step_str = "—                     "
        mean_t = sum(r.elapsed_s for r in rs) / n
        print(f"{config:<18}  {n:>3}  {crate*100:>5.0f}%  {step_str:>22}  {mean_t:>7.1f}s")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, nargs="+", default=[4, 6, 7])
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes per (level, config). Same seeds reused across configs.")
    p.add_argument("--num-simulations", type=int, default=3000)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--configs", type=str, default=None,
                   help="Comma-separated subset of CONFIG names. Default: all.")
    p.add_argument("--seed-base", type=int, default=20260416,
                   help="Base RNG seed. Episode i uses seed = seed_base + level*997 + i.")
    args = p.parse_args()

    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        configs = [CONFIGS[n] for n in names]
    else:
        configs = list(CONFIGS.values())

    print(f"Benchmarking {len(configs)} configs × {len(args.levels)} levels × {args.episodes} episodes")
    print(f"  sims={args.num_simulations}  max_steps={args.max_steps}")
    print(f"  configs: {', '.join(c.name for c in configs)}")
    print(f"  levels: {args.levels}")

    results: list[EpisodeResult] = []
    total_runs = len(configs) * len(args.levels) * args.episodes
    done = 0
    t_start = time.time()

    for level in args.levels:
        for ep in range(args.episodes):
            # Every config sees the same seed for this (level, episode) slot.
            seed = args.seed_base + level * 997 + ep
            for cfg in configs:
                r = run_one(level, cfg, seed,
                            num_simulations=args.num_simulations,
                            max_steps=args.max_steps)
                results.append(r)
                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total_runs - done)
                print(f"  [{done}/{total_runs}] L{level} ep{ep} {cfg.name:<18} "
                      f"{r.outcome:<10} steps={r.steps} ({r.elapsed_s:.1f}s)  "
                      f"eta={eta/60:.1f}m", flush=True)

    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
