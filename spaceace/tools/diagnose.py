#!/usr/bin/env python3
"""
Diagnostic runner for MCTS agent — logs detailed per-decision state to a JSON file
so stuck behavior can be analyzed without screenshots.

Usage:
    uv run python diagnose.py --level 6 --num-simulations 5000 --max-steps 3000
    uv run python diagnose.py --level 6 --num-simulations 5000 --headless

Output: diagnostics/level_N_TIMESTAMP.jsonl  (one JSON object per MCTS decision)
"""

import argparse
import json
import math
import os
import time
from collections import deque
from datetime import datetime

import numpy as np

import spaceace_rl
from spaceace.core.env import SpaceAceDirectEnv
from spaceace.core.viz import VisualRenderer, extract_game_info

ACTION_NAMES = ["COAST", "THRUST", "LEFT", "LEFT+THRUST", "RIGHT", "RIGHT+THRUST"]


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose MCTS agent behavior")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--num-simulations", type=int, default=5000)
    p.add_argument("--exploration", type=float, default=1.41)
    p.add_argument("--action-repeat", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--stuck-threshold", type=float, default=50.0,
                   help="Distance threshold to consider agent stuck (over last N decisions)")
    p.add_argument("--stuck-window", type=int, default=10,
                   help="Number of decisions to look back for stuck detection")
    return p.parse_args()


def detect_stuck(position_history, threshold, window):
    """Check if the agent has been looping/stuck based on position history."""
    if len(position_history) < window:
        return False, 0.0
    recent = list(position_history)[-window:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    spread = math.sqrt(spread_x**2 + spread_y**2)
    return spread < threshold, spread


def main():
    args = parse_args()

    os.makedirs("diagnostics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"diagnostics/level_{args.level}_{timestamp}.jsonl"

    env = SpaceAceDirectEnv(level=args.level, max_steps=args.max_steps)
    mcts = spaceace_rl.PyMCTSEngine(args.level, args.max_steps)
    print(f"Pathfinder: {mcts.get_pathfinder_info()}")

    renderer = None if args.headless else VisualRenderer(width=1200, height=800)

    all_actions = [
        np.array([0, 0, 0], dtype=np.int32),
        np.array([0, 0, 1], dtype=np.int32),
        np.array([1, 0, 0], dtype=np.int32),
        np.array([1, 0, 1], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([0, 1, 1], dtype=np.int32),
    ]

    print(f"Logging diagnostics to: {log_path}")
    print(f"Settings: sims={args.num_simulations}, repeat={args.action_repeat}, "
          f"explore={args.exploration}")

    with open(log_path, "w") as log_file:
        for episode in range(args.episodes):
            env.reset()
            total_reward = 0.0
            decision_count = 0
            pending_repeats = 0
            pending_action = None
            position_history = deque(maxlen=args.stuck_window * 2)

            while True:
                if pending_repeats > 0:
                    pending_repeats -= 1
                    obs, reward, terminated, truncated, info = env.step(pending_action)
                    total_reward += reward

                    if renderer:
                        raw_obs = env.get_observation()
                        game_state, pickups, map_bounds = extract_game_info(raw_obs, info, env)
                        game_state["level"] = args.level
                        state = env.save_state()
                        debug_path = mcts.get_debug_path(state)
                        if not renderer.render_frame(game_state, info, total_reward,
                                                     pending_action, pickups, map_bounds, debug_path):
                            return
                        renderer.wait_for_fps(args.fps)

                    if terminated or truncated:
                        break
                    continue

                # --- MCTS decision point ---
                current_state = env.save_state()

                action_idx, action_stats, root_heuristic = mcts.search_with_stats(
                    current_state,
                    args.num_simulations,
                    args.action_repeat,
                    args.exploration,
                    0.99,  # gamma
                )

                # Get pathfinder info
                path_dist, dir_x, dir_y = mcts.get_pathfinder_stats(current_state)

                # Get current obs
                env.load_state(current_state)
                obs = env.get_observation()
                ship_x, ship_y = float(obs[0]), float(obs[1])
                ship_vx, ship_vy = float(obs[2]), float(obs[3])
                ship_rot = float(obs[4])
                pickups_remaining = int(obs[16])

                position_history.append((ship_x, ship_y))
                is_stuck, spread = detect_stuck(position_history,
                                                args.stuck_threshold, args.stuck_window)

                # Format action stats
                stats_sorted = sorted(action_stats, key=lambda x: x[1], reverse=True)
                action_breakdown = [
                    {"action": ACTION_NAMES[int(a)], "visits": int(v), "mean_value": round(mv, 2)}
                    for a, v, mv in stats_sorted
                ]

                speed = math.sqrt(ship_vx**2 + ship_vy**2)
                heading_x = math.sin(ship_rot)
                heading_y = -math.cos(ship_rot)
                speed_toward_target = ship_vx * dir_x + ship_vy * dir_y if path_dist > 0 else 0
                heading_alignment = heading_x * dir_x + heading_y * dir_y if path_dist > 0 else 0

                record = {
                    "episode": episode,
                    "decision": decision_count,
                    "step": int(info.get("step_count", 0)) if pending_action is not None else decision_count * args.action_repeat,
                    "position": {"x": round(ship_x, 1), "y": round(ship_y, 1)},
                    "velocity": {"vx": round(ship_vx, 1), "vy": round(ship_vy, 1), "speed": round(speed, 1)},
                    "rotation_deg": round(math.degrees(ship_rot), 1),
                    "pickups_remaining": pickups_remaining,
                    "total_reward": round(total_reward, 2),
                    "chosen_action": ACTION_NAMES[action_idx],
                    "root_heuristic": round(root_heuristic, 2),
                    "path_distance": round(path_dist, 1),
                    "path_direction": {"dx": round(dir_x, 3), "dy": round(dir_y, 3)},
                    "speed_toward_target": round(speed_toward_target, 1),
                    "heading_alignment": round(heading_alignment, 3),
                    "action_stats": action_breakdown,
                    "is_stuck": is_stuck,
                    "position_spread": round(spread, 1),
                }

                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

                if is_stuck:
                    print(f"  [STUCK] decision={decision_count} pos=({ship_x:.0f},{ship_y:.0f}) "
                          f"spread={spread:.1f} pathDist={path_dist:.0f} "
                          f"action={ACTION_NAMES[action_idx]} heuristic={root_heuristic:.1f}")

                decision_count += 1
                action = all_actions[action_idx]

                env.load_state(current_state)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                pending_action = action
                pending_repeats = args.action_repeat - 1

                if renderer:
                    raw_obs = env.get_observation()
                    game_state, pickups, map_bounds = extract_game_info(raw_obs, info, env)
                    game_state["level"] = args.level
                    debug_path = mcts.get_debug_path(current_state)
                    if not renderer.render_frame(game_state, info, total_reward,
                                                 action, pickups, map_bounds, debug_path):
                        return
                    renderer.wait_for_fps(args.fps)

                if terminated or truncated:
                    break

            status = "COMPLETED" if info.get("level_completed") else (
                "CRASHED" if info.get("ship_exploded") else "TRUNCATED"
            )
            print(f"Episode {episode+1}: {status} | decisions={decision_count} "
                  f"reward={total_reward:.1f}")

    if renderer:
        renderer.close()
    env.close()

    # Print summary
    print(f"\nDiagnostics saved to: {log_path}")
    print("Analyze with: uv run python diagnose.py --help")
    print(f"  Quick stuck analysis: grep '\"is_stuck\": true' {log_path} | head -20")
    print(f"  Or: uv run python -c \"")
    print(f"import json")
    print(f"records = [json.loads(l) for l in open('{log_path}')]")
    print(f"stuck = [r for r in records if r['is_stuck']]")
    print(f"print(f'Stuck {{len(stuck)}}/{{len(records)}} decisions')")
    print(f"for r in stuck[:5]: print(r['decision'], r['position'], r['chosen_action'], r['path_distance'])\"")


if __name__ == "__main__":
    main()
