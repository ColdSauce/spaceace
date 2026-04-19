"""Run the current PPO checkpoint on a specific level and log exactly what
it does. No theories — just raw trajectory data + outcome distributions.

Usage:
    uv run python scripts/probe_agent_on_level.py --level 5115 --episodes 50
    uv run python scripts/probe_agent_on_level.py --level 5115 --model models/curriculum/latest_model --episodes 100 --deterministic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spaceace_rl


def _load_model_and_env(level: int, max_steps: int, model_path: str):
    """Build the same env stack TRAINING uses (no obs normalization) and
    load the checkpoint. Training uses VecNormalize(norm_obs=False,
    norm_reward=True) — reward normalization doesn't affect inference, so
    we skip VecNormalize entirely here to avoid accidentally adding an
    obs-normalization layer the policy was never trained against.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from spaceace.core.gym_wrapper import SpaceAceGymWrapper
    from spaceace.training.envs import StrategyWrapper, _build_strategies

    base_env = SpaceAceGymWrapper(level=level, max_steps=max_steps)
    obs_strategy, reward_strategy, pf = _build_strategies(
        level, max_steps, "path_augmented", "dense_shaped",
    )
    wrapped = StrategyWrapper(base_env, obs_strategy, reward_strategy,
                              action_repeat=5, pathfinder=pf)
    vec = DummyVecEnv([lambda: wrapped])

    model = PPO.load(model_path)
    raw_env = base_env.env
    return model, vec, raw_env


def run_episodes(level: int, n_episodes: int, action_repeat: int,
                 model_path: str, deterministic: bool, max_steps: int):
    model, vec, raw_env = _load_model_and_env(level, max_steps, model_path)
    geo = raw_env.get_map_geometry()
    pickup_positions = [(float(p[0]), float(p[1])) for p in geo["pickup_positions"]]

    episodes = []
    for ep in range(n_episodes):
        obs = vec.reset()
        positions = []
        actions = []
        pickup_events = []   # (macro_step, pickup_index)
        prev_collected = [False] * len(pickup_positions)
        step = 0  # macro steps (one per model.predict call, = 5 physics frames)

        while True:
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = vec.step(a)   # StrategyWrapper runs 5 physics frames
            step += 1

            raw_obs = raw_env.get_observation()
            positions.append((float(raw_obs[0]), float(raw_obs[1])))
            from spaceace.strategies.actions import ALL_ACTIONS
            actions.append(tuple(int(x) for x in ALL_ACTIONS[int(np.asarray(a[0]))]))

            pickup_states = list(raw_env.get_pickup_states())
            for idx, (was, now) in enumerate(zip(prev_collected, pickup_states)):
                if now and not was:
                    pickup_events.append((step, idx))
            prev_collected = pickup_states

            if bool(dones[0]):
                break

        info = infos[0]
        if info.get("level_completed"):
            outcome = "completed"
        elif info.get("ship_exploded"):
            outcome = "crashed"
        else:
            outcome = "truncated"

        episodes.append({
            "outcome": outcome,
            "steps": step,
            "positions": positions,
            "actions": actions,
            "pickup_events": pickup_events,
            "pickups_collected": int(sum(prev_collected)),
            "final_position": positions[-1] if positions else None,
        })

    vec.close()
    return {
        "geo": geo,
        "pickup_positions": pickup_positions,
        "episodes": episodes,
    }


def analyze_and_plot(data, level: int, out_dir: Path):
    eps = data["episodes"]
    geo = data["geo"]
    pickups = data["pickup_positions"]
    bounds = geo["bounds"]
    walls = geo["map_lines"]

    n = len(eps)
    outcomes = Counter(e["outcome"] for e in eps)
    pickups_hist = Counter(e["pickups_collected"] for e in eps)
    avg_steps = np.mean([e["steps"] for e in eps])
    avg_pickups = np.mean([e["pickups_collected"] for e in eps])

    # Which pickups get collected, and in what order?
    pickup_collection_rate = Counter()
    collection_order_first = Counter()
    for e in eps:
        collected_set = set(idx for _, idx in e["pickup_events"])
        for idx in collected_set:
            pickup_collection_rate[idx] += 1
        if e["pickup_events"]:
            collection_order_first[e["pickup_events"][0][1]] += 1

    # Action usage
    action_counter = Counter()
    for e in eps:
        for a in e["actions"]:
            action_counter[a] += 1
    total_frames = sum(action_counter.values())

    # ---- Print summary ----
    print()
    print(f"===== L{level} probe: {n} episodes =====")
    print(f"Outcomes: {dict(outcomes)}  (win rate: {outcomes['completed'] / n:.1%})")
    print(f"Avg steps/episode: {avg_steps:.0f}, Avg pickups: {avg_pickups:.2f} / {len(pickups)}")
    print()
    print("Pickups-collected distribution:")
    for k in sorted(pickups_hist):
        print(f"  {k} pickups: {pickups_hist[k]:3d} episodes ({pickups_hist[k] / n:.1%})")
    print()
    print("Per-pickup collection rate (how often each one gets grabbed):")
    for idx, (x, y) in enumerate(pickups):
        rate = pickup_collection_rate[idx] / n
        first = collection_order_first[idx] / n
        print(f"  P{idx} @ ({x:.0f}, {y:.0f}): collected in {rate:.1%} of eps, "
              f"was FIRST pickup in {first:.1%}")
    print()
    print("Action usage (top 6):")
    for act, c in action_counter.most_common(6):
        print(f"  action={act}: {c / total_frames:.1%} of frames")

    # ---- Heatmap of positions visited ----
    all_x = []
    all_y = []
    for e in eps:
        for x, y in e["positions"]:
            all_x.append(x); all_y.append(y)

    final_x = [e["final_position"][0] for e in eps if e["final_position"]]
    final_y = [e["final_position"][1] for e in eps if e["final_position"]]

    crash_final_x = [e["final_position"][0] for e in eps if e["outcome"] == "crashed" and e["final_position"]]
    crash_final_y = [e["final_position"][1] for e in eps if e["outcome"] == "crashed" and e["final_position"]]
    trunc_final_x = [e["final_position"][0] for e in eps if e["outcome"] == "truncated" and e["final_position"]]
    trunc_final_y = [e["final_position"][1] for e in eps if e["outcome"] == "truncated" and e["final_position"]]

    def _map_axes(ax, title):
        for w in walls:
            ax.plot([w[0], w[2]], [w[1], w[3]], color="#8e8e93", lw=1.5)
        for idx, (x, y) in enumerate(pickups):
            ax.scatter(x, y, s=160, c="#00ffff", edgecolors="black", zorder=4)
            ax.annotate(f"P{idx}", (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=9, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
        ax.set_xlim(bounds["min_x"] - 20, bounds["max_x"] + 20)
        ax.set_ylim(bounds["max_y"] + 20, bounds["min_y"] - 20)
        ax.set_aspect("equal")
        ax.set_facecolor("#1c1c1e")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#000")

    _map_axes(axes[0], f"Position density — {n} episodes ({len(all_x)} frames)")
    if all_x:
        axes[0].hexbin(all_x, all_y, gridsize=50, cmap="magma",
                       extent=[bounds["min_x"], bounds["max_x"], bounds["min_y"], bounds["max_y"]],
                       mincnt=1, alpha=0.85)

    _map_axes(axes[1], f"Crash final positions ({len(crash_final_x)} episodes)")
    if crash_final_x:
        axes[1].scatter(crash_final_x, crash_final_y, s=40, c="#ff3b30",
                        edgecolors="white", linewidths=0.6, alpha=0.8)

    _map_axes(axes[2], f"Truncation final positions ({len(trunc_final_x)} episodes)")
    if trunc_final_x:
        axes[2].scatter(trunc_final_x, trunc_final_y, s=40, c="#ffcc00",
                        edgecolors="white", linewidths=0.6, alpha=0.8)

    fig.suptitle(
        f"L{level}: {outcomes['completed']}W / {outcomes['crashed']}X / "
        f"{outcomes['truncated']}T out of {n}",
        color="white", fontsize=13,
    )
    out = out_dir / f"{level}_agent_behavior.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"\n  wrote {out}")

    # Per-pickup rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#000")
    idxs = list(range(len(pickups)))
    rates = [pickup_collection_rate[i] / n for i in idxs]
    firsts = [collection_order_first[i] / n for i in idxs]
    xloc = np.arange(len(idxs))
    ax.bar(xloc - 0.2, rates, width=0.4, label="collected ever", color="#34c759")
    ax.bar(xloc + 0.2, firsts, width=0.4, label="collected FIRST", color="#ff9500")
    ax.set_xticks(xloc)
    ax.set_xticklabels([f"P{i}\n({pickups[i][0]:.0f},{pickups[i][1]:.0f})" for i in idxs],
                       color="white", fontsize=9)
    ax.set_ylabel("fraction of episodes", color="white")
    ax.set_ylim(0, 1.05)
    ax.set_facecolor("#1c1c1e")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("white")
    ax.set_title(f"L{level} — per-pickup collection rate", color="white")
    ax.legend(facecolor="#2c2c2e", edgecolor="none", labelcolor="white")
    out = out_dir / f"{level}_pickup_rates.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")

    # Sample overlaid trajectories — easier to see the agent's actual behavior
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#000")
    _map_axes(ax, f"L{level} — sample trajectories (up to 20)")
    for e in eps[:20]:
        xs = [p[0] for p in e["positions"]]
        ys = [p[1] for p in e["positions"]]
        color = {"completed": "#34c759", "crashed": "#ff3b30",
                 "truncated": "#ffcc00"}[e["outcome"]]
        ax.plot(xs, ys, color=color, lw=0.8, alpha=0.5)
        ax.scatter([xs[-1]], [ys[-1]], s=25, c=color, edgecolors="white",
                   linewidths=0.5, zorder=5)
    out = out_dir / f"{level}_trajectories.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#000")
    plt.close(fig)
    print(f"  wrote {out}")

    # Save raw JSON for further analysis
    summary = {
        "level": level,
        "n_episodes": n,
        "outcomes": dict(outcomes),
        "avg_steps": float(avg_steps),
        "avg_pickups": float(avg_pickups),
        "pickups_histogram": dict(pickups_hist),
        "pickup_collection_rate": {str(k): v / n for k, v in pickup_collection_rate.items()},
        "pickup_first_rate": {str(k): v / n for k, v in collection_order_first.items()},
        "pickup_positions": [list(p) for p in pickups],
    }
    out = out_dir / f"{level}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--model", default="models/curriculum/latest_model")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic policy (default stochastic — matches training)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(__file__).resolve().parent.parent / "viz" / f"agent_probe_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probing L{args.level} with model {args.model} "
          f"({'deterministic' if args.deterministic else 'stochastic'}) "
          f"for {args.episodes} episodes...")
    t0 = time.time()
    data = run_episodes(args.level, args.episodes, args.action_repeat,
                        args.model, args.deterministic, args.max_steps)
    elapsed = time.time() - t0
    print(f"Collected {len(data['episodes'])} episodes in {elapsed:.1f}s "
          f"({elapsed/len(data['episodes']):.2f}s/ep)")

    analyze_and_plot(data, args.level, out_dir)
    print(f"\nDone. Open: {out_dir}")


if __name__ == "__main__":
    main()
